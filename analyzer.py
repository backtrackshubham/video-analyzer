import base64
import json
import os
import subprocess
import tempfile
import threading
import time

import requests

DATA_ROOT = os.environ.get("DATA_ROOT", "/data/shared")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "1.0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "60"))
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


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


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
            "model": f"ollama:{self.captioner.model}",
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
