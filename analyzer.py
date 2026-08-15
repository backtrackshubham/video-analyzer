import base64
import json
import logging
import os
import subprocess
import tempfile
import threading
import time

import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("analyzer")

DATA_ROOT = os.environ.get("DATA_ROOT", "/data/shared")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
# Full-res frames get inflated to ~3500 vision tokens by Ollama (batches of
# 512, ~1.8s each on a throttled 6GB GPU) — the prefill bottleneck. Downscale
# to a longest-edge of VLM_IMAGE_SIZE before sending. 448 = parity with the
# other branches; 0 disables.
VLM_IMAGE_SIZE = int(os.environ.get("VLM_IMAGE_SIZE", "448"))
SAMPLE_INTERVAL = float(os.environ.get("SAMPLE_INTERVAL", "3.0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "60"))
NO_CAP_DURATION = float(os.environ.get("NO_CAP_DURATION", "600"))
AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "off")
FRAME_QUESTION = os.environ.get(
    "FRAME_QUESTION",
    "Analyze this video frame. Describe the scene concisely: main subject, "
    "objects, people, actions, setting, and any visible text. Keep it to 1-2 "
    "sentences, factual and specific.",
)
GENERATION_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_gpu": -1,
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(video_path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path])
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_frames(video_path, fps, outdir):
    run(["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", f"{outdir}/%05d.jpg"])


def img_b64(path, max_edge=VLM_IMAGE_SIZE):
    img = Image.open(path).convert("RGB")
    if max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        log.info("[img] downscaled to %dx%d (VLM_IMAGE_SIZE=%d)", img.size[0], img.size[1], max_edge)
    with __import__("io").BytesIO() as buf:
        img.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaCaptioner:
    def __init__(self, host=OLLAMA_HOST, model=OLLAMA_MODEL):
        self.host = host
        self.model = model

    def health(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            return {"reachable": True, "model": self.model, "loaded": self.model in tags}
        except Exception as e:
            return {"reachable": False, "error": str(e)}

    def caption(self, image_path, question=FRAME_QUESTION):
        payload = {
            "model": self.model,
            "prompt": question,
            "images": [img_b64(image_path)],
            "stream": False,
            "options": GENERATION_OPTIONS,
        }
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
                r.raise_for_status()
                text = r.json().get("response", "").strip()
                if text:
                    return text
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"ollama caption failed: {last_err}")


class Transcriber:
    def __init__(self, size):
        self.size = size
        self.model = None

    def ensure(self):
        if self.model is None and self.size not in ("off", "", "none"):
            from faster_whisper import WhisperModel

            self.model = WhisperModel(self.size, device="auto", compute_type="int8")

    def transcribe(self, video_path):
        self.ensure()
        if self.model is None:
            return None
        segments, _ = self.model.transcribe(video_path, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text or None


class Analyzer:
    def __init__(self):
        self.captioner = OllamaCaptioner()
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
            "model": f"ollama:{self.captioner.model}",
            "sample_interval": interval,
            "total_frames": len(frame_paths),
            "frames": [],
            "transcript": None,
            "status": "running",
        }
        frame_gap_t0 = None
        try:
            for i, p in enumerate(frame_paths):
                if frame_gap_t0 is not None:
                    log.info(
                        "  [gap] last-iteration overhead: %+.1fs (outside caption)",
                        time.perf_counter() - frame_gap_t0,
                    )
                frame_gap_t0 = time.perf_counter()
                t = round(i * interval, 2)
                caption = self.captioner.caption(p)
                result["frames"].append({"t": t, "caption": caption})
                result["status"] = "running"
                log.info(
                    "frame %d/%d t=%.1fs done (%d chars) [caption=%.1fs]",
                    i + 1,
                    len(frame_paths),
                    t,
                    len(caption),
                    time.perf_counter() - frame_gap_t0,
                )
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
