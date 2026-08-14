# Video Analyzer — Alienware (GPU / Ollama)

FastAPI service that samples a short clip (30-60s), captions every frame with a
local vision model, and writes an ordered `{t, caption}` array into
`shared_volume/analysis/{clip}.analysis.json` so an orchestrator (n8n) can hand
the text to a text-LLM (Gemini) for the final "what's happening" summary.

## Device
- Alienware desktop, Linux, NVIDIA GPU (RTX 3060 6GB)
- Vision backend: **Ollama + `qwen2.5vl:3b`** (Q4 ~2.5GB, GPU offloaded)
- Whisper (faster-whisper) transcribes audio (defaults to `off`)

## Files
| File | Purpose |
| --- | --- |
| `analyzer.py` | Ollama captioner, frame extraction, analysis pipeline |
| `main.py` | FastAPI app (`GET /health`, `POST /analyze`, `POST /analyze/upload`) |
| `requirements.txt` | Python deps for the analyzer container |
| `Dockerfile` | Analyzer container (no torch; talks to Ollama over HTTP) |
| `docker-compose.yml` | `ollama` (GPU-reserved) + `analyzer` (:31027) |
| `pull_model.sh` | Waits for Ollama, pulls `qwen2.5vl:3b` |
| `n8n-video-analysis.workflow.json` | Orchestration workflow import |

## Run (Linux + Docker + NVIDIA container toolkit)

```bash
docker compose up -d --build
./pull_model.sh                                  # once, pulls the VLM
docker compose logs -f analyzer                  # watch startup
curl localhost:31027/health
```

## Analyze a clip

```bash
cp /path/to/clip.mp4 shared_volume/clips/
curl -X POST localhost:31027/analyze \
  -H 'Content-Type: application/json' \
  -d '{"path": "/data/shared/clips/clip.mp4", "audio": false, "max_frames": null, "secondsPerFrame": 3.0}'
# or upload directly (no need to pre-place the clip):
curl -F "video=@/path/to/clip.mp4" -F "secondsPerFrame=3.0" localhost:31027/analyze/upload
# -> shared_volume/analysis/clip.analysis.json
```

Tunables: `OLLAMA_MODEL`, `OLLAMA_HOST`, `SAMPLE_INTERVAL` (s between frames,
default 3.0), `MAX_FRAMES` (applies only to clips > 10 min), `NO_CAP_DURATION`,
`OLLAMA_TIMEOUT`, `FRAME_QUESTION`, `ANALYZER_PORT`.