import json
import os
import subprocess
import tempfile
import threading

from PIL import Image

DATA_ROOT = os.environ.get("DATA_ROOT", "/data/shared")
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream2")
MOONDREAM_REVISION = os.environ.get("MOONDREAM_REVISION", "2025-01-09")
VISION_MAX_CROPS = int(os.environ.get("VISION_MAX_CROPS", "1"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "1.0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "60"))
AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "tiny")
FRAME_QUESTION = "Describe what is visible in this video frame in one sentence."
CAPTION_MAX_TOKENS = int(os.environ.get("CAPTION_MAX_TOKENS", "48"))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(video_path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path])
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_frame(video_path, t, out_path):
    run(["ffmpeg", "-ss", str(max(0.0, t)), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path])
    return os.path.exists(out_path)


def sample_times(duration):
    step = 1.0 / SAMPLE_FPS
    times = []
    t = 0.0
    while t < duration and len(times) < MAX_FRAMES:
        times.append(round(t, 2))
        t += step
    return times


class VisionModel:
    def caption(self, image_path, question, max_new_tokens=CAPTION_MAX_TOKENS):
        raise NotImplementedError


class MoondreamModel(VisionModel):
    def __init__(self):
        import torch

        torch.compile = lambda fn, **kwargs: fn

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = "vikhyatk/moondream2"
        cfg = AutoConfig.from_pretrained(model_id, revision=MOONDREAM_REVISION, trust_remote_code=True)
        cfg.config = dict(getattr(cfg, "config", {}) or {})
        cfg.config["vision"] = {"max_crops": VISION_MAX_CROPS}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=MOONDREAM_REVISION, trust_remote_code=True, config=cfg
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MOONDREAM_REVISION)
        self.model.eval()

    def caption(self, image_path, question, max_new_tokens=CAPTION_MAX_TOKENS):
        img = Image.open(image_path).convert("RGB")
        enc = self.model.encode_image(img)
        return self.model.answer_question(enc, question, self.tokenizer, max_new_tokens=max_new_tokens).strip()


class QwenVlModel(VisionModel):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise RuntimeError("Qwen2.5-VL requires torchvision; install it and rebuild")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def caption(self, image_path, question, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
        gen = self.model.generate(**inputs, max_new_tokens=64)
        out = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return out.replace(text, "").strip()


def build_vision_model():
    if VISION_MODEL == "qwen2.5-vl-3b":
        return QwenVlModel()
    if VISION_MODEL == "moondream2":
        return MoondreamModel()
    raise ValueError(f"unsupported VISION_MODEL: {VISION_MODEL}")


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
        self.vision = build_vision_model()
        self.transcriber = Transcriber(AUDIO_MODEL)
        self.lock = threading.Lock()

    def analyze(self, clip_path, want_audio=False, max_frames=None):
        if not clip_path.startswith(DATA_ROOT + "/"):
            raise ValueError(f"path must be inside {DATA_ROOT}")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(clip_path)

        base = os.path.splitext(os.path.basename(clip_path))[0]
        out_dir = os.path.join(DATA_ROOT, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        analysis_file = os.path.join(out_dir, base + ".analysis.json")

        duration = get_duration(clip_path)
        times = sample_times(duration)
        if max_frames:
            times = times[:max_frames]

        workdir = tempfile.mkdtemp(prefix="frames_")
        result = {
            "clip": clip_path,
            "analysis_file": analysis_file,
            "duration": round(duration, 2),
            "model": VISION_MODEL,
            "sample_fps": SAMPLE_FPS,
            "total_frames": len(times),
            "frames": [],
            "transcript": None,
            "status": "running",
        }
        try:
            for i, t in enumerate(times):
                p = os.path.join(workdir, f"f{i:04d}.jpg")
                if not extract_frame(clip_path, min(t, max(0.0, duration - 0.1)), p):
                    continue
                caption = self.vision.caption(p, FRAME_QUESTION)
                result["frames"].append({"t": t, "caption": caption})
                result["status"] = "running"
                print(f"frame {i + 1}/{len(times)} t={t}s done", flush=True)
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
