import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("analyzer")

DEFAULT_DATA_ROOT = "shared_volume" if os.name == "nt" else "/data/shared"
DATA_ROOT = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", os.path.join(os.getcwd(), "models"))
VLM_MODEL = os.environ.get("VLM_MODEL", "0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu")
VLM_DEVICE = os.environ.get("VLM_DEVICE", "NPU")
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "64"))
SAMPLE_INTERVAL = float(os.environ.get("SAMPLE_INTERVAL", "3.0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "60"))
NO_CAP_DURATION = float(os.environ.get("NO_CAP_DURATION", "600"))
AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "off")
FRAME_QUESTION = os.environ.get(
    "FRAME_QUESTION",
    "Describe what is visible in this video frame in one sentence.",
)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(video_path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path])
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_frames(video_path, fps, outdir):
    run(["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", os.path.join(outdir, "%05d.jpg")])


def is_model_dir(path):
    return os.path.exists(os.path.join(path, "openvino_config.json")) or os.path.exists(
        os.path.join(path, "openvino_model.xml")
    )


def resolve_model_dir(spec, cache_dir=MODEL_DIR):
    if os.path.isdir(spec) and is_model_dir(spec):
        return spec
    local = os.path.abspath(os.path.join(cache_dir, os.path.basename(spec)))
    if os.path.isdir(local) and is_model_dir(local):
        return local
    from huggingface_hub import snapshot_download

    last_err = None
    for endpoint in (os.environ.get("HF_ENDPOINT", "https://huggingface.co"), "https://hf-mirror.com"):
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            return snapshot_download(repo_id=spec, local_dir=local)
        except Exception as e:
            last_err = e
            print(f"model download failed via {endpoint}: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError(f"could not download model {spec}: {last_err}")


class NPUVlmCaptioner:
    def __init__(self, model=VLM_MODEL, device=VLM_DEVICE):
        import numpy as np
        import openvino_genai as genai

        self.np = np
        self.model = model
        self.model_path = resolve_model_dir(model)
        self.device = device
        hint = os.environ.get("VLM_GENERATE_HINT", "FAST_COMPILE")
        log.info("compiling %s on %s (hint=%s)...", self.model, self.device, hint)
        self.pipeline_config = {
            "GENERATE_HINT": hint,
            "MAX_PROMPT_LEN": 4096,
            "MIN_RESPONSE_LEN": 256,
            "CACHE_DIR": os.path.join(self.model_path, ".npucache"),
        }
        self.pipe = genai.VLMPipeline(self.model_path, self.device, **self.pipeline_config)
        log.info("pipeline ready")
        self.gen_cfg = genai.GenerationConfig()
        self.gen_cfg.max_new_tokens = VLM_MAX_TOKENS
        # Greedy decode is essential on NPU: temperature/top_p/min_new_tokens
        # force sampling on CPU, which falls back to ~1.3 s/token (see timing logs).
        # Leave do_sample off (default) and do not set temperature/top_p/min_new_tokens.
        self._stream_frame = 0
        self._stream_t0 = None
        self._stream_n = 0

        def _streamer(subword):
            if self._stream_t0 is None:
                self._stream_t0 = time.perf_counter()
            self._stream_n += 1
            if self._stream_frame <= 2 and self._stream_n <= 40:
                elapsed = time.perf_counter() - self._stream_t0
                log.info(
                    "[stream] frame#%d token %2d at +%.1fs",
                    self._stream_frame,
                    self._stream_n,
                    elapsed,
                )
            return False

        self.gen_cfg.streamer = _streamer
        self.gen_cfg.is_streaming = True
        log.info("streamer profiler active (tokens logged for first 2 frames)")

    def health(self):
        return {"device": self.device, "model": self.model, "model_path": self.model_path}

    def caption(self, image_path, prompt=FRAME_QUESTION):
        import time

        import openvino as ov

        self._stream_frame += 1
        t0 = time.perf_counter()
        pil = Image.open(image_path).convert("RGB")
        t1 = time.perf_counter()
        img_tensor = ov.Tensor(self.np.array(pil))
        t2 = time.perf_counter()
        try:
            out = self.pipe.generate(prompt, images=[img_tensor], generation_config=self.gen_cfg)
        except TypeError:
            out = self.pipe.generate(prompt, [img_tensor], self.gen_cfg)
        t3 = time.perf_counter()
        text = str(out).strip()
        t4 = time.perf_counter()
        tok_s = (len(text) / 4) / max((t3 - t2), 1e-6)
        log.info(
            "\n  [timing] load=%.1fms tensor=%.1fms GENERATE=%.1fs (~%.1f tok/s) str=%.1fms  total=%.1fs",
            (t1 - t0) * 1e3,
            (t2 - t1) * 1e3,
            t3 - t2,
            tok_s,
            (t4 - t3) * 1e3,
            t4 - t0,
        )
        return text


class Transcriber:
    def __init__(self, size):
        self.size = size
        self.model = None

    def ensure(self):
        if self.model is None and self.size not in ("off", "", "none"):
            from faster_whisper import WhisperModel

            self.model = WhisperModel(self.size, device="cpu", compute_type="int8")

    def transcribe(self, video_path):
        self.ensure()
        if self.model is None:
            return None
        segments, _ = self.model.transcribe(video_path, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text or None


class Analyzer:
    def __init__(self):
        self.captioner = NPUVlmCaptioner()
        self.transcriber = Transcriber(AUDIO_MODEL)
        self.lock = threading.Lock()

    def analyze(self, clip_path, want_audio=False, max_frames=None, seconds_per_frame=None):
        log.info(
            "analyze request: clip=%s audio=%s secondsPerFrame=%s max_frames=%s",
            clip_path,
            want_audio,
            seconds_per_frame,
            max_frames,
        )
        data_root = os.path.abspath(DATA_ROOT)
        if not os.path.abspath(clip_path).startswith(data_root + os.sep):
            log.warning("path %s rejected (outside %s)", clip_path, data_root)
            raise ValueError(f"path must be inside {data_root}")
        if not os.path.isfile(clip_path):
            log.warning("clip not found: %s", clip_path)
            raise FileNotFoundError(clip_path)
        log.info("clip verified")

        interval = seconds_per_frame if seconds_per_frame else SAMPLE_INTERVAL
        if interval <= 0:
            raise ValueError("secondsPerFrame must be > 0")
        fps = 1.0 / interval

        base = os.path.splitext(os.path.basename(clip_path))[0]
        out_dir = os.path.join(data_root, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        analysis_file = os.path.join(out_dir, base + ".analysis.json")

        duration = get_duration(clip_path)
        log.info("duration=%.1f s", duration)

        workdir = tempfile.mkdtemp(prefix="frames_")
        log.info("extracting frames (interval=%.1fs, fps=%.3f) -> %s", interval, fps, workdir)
        extract_frames(clip_path, fps, workdir)

        frame_paths = sorted(
            os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(".jpg")
        )
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
            log.info("frame cap applied (max_frames=%d)", max_frames)
        elif duration > NO_CAP_DURATION:
            frame_paths = frame_paths[:MAX_FRAMES]
            log.info("clip > %ds; frame cap applied (MAX_FRAMES=%d)", NO_CAP_DURATION, MAX_FRAMES)
        log.info("total frames to caption: %d", len(frame_paths))

        result = {
            "clip": clip_path,
            "analysis_file": analysis_file,
            "duration": round(duration, 2),
            "model": f"openvino:{self.captioner.device}:{self.captioner.model}",
            "device": self.captioner.device,
            "sample_interval_s": interval,
            "sample_fps": round(fps, 4),
            "total_frames": len(frame_paths),
            "frames": [],
            "transcript": None,
            "status": "running",
        }
        import time as _time
        frame_gap_t0 = None
        try:
            for i, p in enumerate(frame_paths):
                if frame_gap_t0 is not None:
                    log.info(
                        "  [gap] last-iteration overhead: %+.1fs (outside caption)",
                        _time.perf_counter() - frame_gap_t0,
                    )
                frame_gap_t0 = _time.perf_counter()
                t = round(i * interval, 2)
                caption = self.captioner.caption(p)
                result["frames"].append({"t": t, "caption": caption})
                result["status"] = "running"
                log.info("frame %d/%d t=%.1fs done (%d chars)", i + 1, len(frame_paths), t, len(caption))
                self._save(result)
        finally:
            for fname in os.listdir(workdir):
                try:
                    os.remove(os.path.join(workdir, fname))
                except OSError:
                    pass
            try:
                os.rmdir(workdir)
            except OSError:
                pass
            log.info("cleaned workdir %s", workdir)

        if want_audio:
            log.info("transcribing audio (%s)", self.transcriber.size)
            result["transcript"] = self.transcriber.transcribe(clip_path)
            log.info("transcript: %s", (result["transcript"] or "")[:120])
        result["status"] = "done"
        self._save(result)
        log.info("analysis complete -> %s (%d frames, %.1fs clip)", analysis_file, len(frame_paths), duration)
        return result

    @staticmethod
    def _save(result):
        with open(result["analysis_file"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)