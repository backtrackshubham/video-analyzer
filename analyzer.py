import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image

DEFAULT_DATA_ROOT = "shared_volume" if os.name == "nt" else "/data/shared"
DATA_ROOT = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", os.path.join(os.getcwd(), "models"))
VLM_MODEL = os.environ.get("VLM_MODEL", "0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu")
VLM_DEVICE = os.environ.get("VLM_DEVICE", "NPU")
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "64"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "1.0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "60"))
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
        self.pipeline_config = {
            "GENERATE_HINT": "BEST_PERF",
            "MAX_PROMPT_LEN": 4096,
            "MIN_RESPONSE_LEN": 256,
            "CACHE_DIR": os.path.join(self.model_path, ".npucache"),
        }
        self.pipe = genai.VLMPipeline(self.model_path, self.device, self.pipeline_config)
        self.gen_cfg = genai.GenerationConfig()
        self.gen_cfg.max_new_tokens = VLM_MAX_TOKENS
        self.gen_cfg.min_new_tokens = 2
        self.gen_cfg.temperature = 0.3
        self.gen_cfg.top_p = 0.9

    def health(self):
        return {"device": self.device, "model": self.model, "model_path": self.model_path}

    def caption(self, image_path, prompt=FRAME_QUESTION):
        import openvino_genai as genai

        pil_img = Image.open(image_path).convert("RGB")
        ov_img = self.np.array(pil_img)
        try:
            ov_img = genai.Image(ov_img)
        except Exception:
            pass
        try:
            out = self.pipe.generate(prompt, images=[ov_img], generation_config=self.gen_cfg)
        except TypeError:
            out = self.pipe.generate(prompt, [ov_img], self.gen_cfg)
        return str(out).strip()


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

    def analyze(self, clip_path, want_audio=False, max_frames=None):
        data_root = os.path.abspath(DATA_ROOT)
        if not os.path.abspath(clip_path).startswith(data_root + os.sep):
            raise ValueError(f"path must be inside {data_root}")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(clip_path)

        base = os.path.splitext(os.path.basename(clip_path))[0]
        out_dir = os.path.join(data_root, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        analysis_file = os.path.join(out_dir, base + ".analysis.json")

        duration = get_duration(clip_path)

        workdir = tempfile.mkdtemp(prefix="frames_")
        extract_frames(clip_path, SAMPLE_FPS, workdir)

        frame_paths = sorted(
            os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(".jpg")
        )
        if max_frames:
            frame_paths = frame_paths[:max_frames]

        result = {
            "clip": clip_path,
            "analysis_file": analysis_file,
            "duration": round(duration, 2),
            "model": f"openvino:{self.captioner.device}:{self.captioner.model}",
            "device": self.captioner.device,
            "sample_fps": SAMPLE_FPS,
            "total_frames": len(frame_paths),
            "frames": [],
            "transcript": None,
            "status": "running",
        }
        try:
            for i, p in enumerate(frame_paths):
                t = round(i / SAMPLE_FPS, 2)
                caption = self.captioner.caption(p)
                result["frames"].append({"t": t, "caption": caption})
                result["status"] = "running"
                print(f"frame {i + 1}/{len(frame_paths)} t={t}s done", flush=True)
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

        if want_audio:
            result["transcript"] = self.transcriber.transcribe(clip_path)
        result["status"] = "done"
        self._save(result)
        return result

    @staticmethod
    def _save(result):
        with open(result["analysis_file"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)