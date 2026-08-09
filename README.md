# Video Analyzer — Mac/Dev laptop (CPU / Moondream)

FastAPI service that samples a short clip (30-60s), captions every frame with a
local Moondream vision model (CPU, no GPU), and writes an ordered
`{t, caption}` array into `shared_volume/analysis/{clip}.analysis.json` so an
orchestrator (n8n) can hand the text to a text-LLM (Gemini) for the final
"what's happening" summary.

## Device
- i5-5350U (4 cores, 8GB RAM), no GPU — dev/unit-test box only
- Vision backend: **Moondream2** via transformers, revision `2025-01-09`
  (this machine cannot run gigabytes of RAM easily; keep `SAMPLE_FPS` low)
- Whisper (faster-whisper) transcribes audio (defaults to `tiny`)

## ⚠️ Performance reality
Measured on this CPU: ~4.6 min/frame (encode-bound). A 60-frame clip is
~4.6h. Keep `MAX_FRAMES` small (e.g. 6-12) for anything interactive, or use
the Alienware/NPU branches for real work.

## Files
| File | Purpose |
| --- | --- |
| `analyzer.py` | Moondream captioner, frame extraction, analysis pipeline |
| `main.py` | FastAPI app (`GET /health`, `POST /analyze`) |
| `requirements.txt` | Python deps (torch CPU + transformers 4.49.0) |
| `Dockerfile` | Analyzer container (CPU torch wheel) |
| `docker-compose.yml` | `analyzer` (:31027) + model cache volume |
| `n8n-video-analysis.workflow.json` | Orchestration workflow import |

## Run (Linux + Docker)

```bash
docker compose up -d --build
curl localhost:31027/health
```

## Analyze a clip

```bash
cp /path/to/clip.mp4 shared_volume/clips/
curl -X POST localhost:31027/analyze \
  -H 'Content-Type: application/json' \
  -d '{"path": "/data/shared/clips/clip.mp4", "audio": true, "max_frames": 6}'
# -> shared_volume/analysis/clip.analysis.json
```

Tunables: `VISION_MODEL` (`moondream2` | `qwen2.5-vl-3b`), `MOONDREAM_REVISION`,
`VISION_MAX_CROPS`, `SAMPLE_FPS`, `MAX_FRAMES`, `CAPTION_MAX_TOKENS`, `AUDIO_MODEL`.