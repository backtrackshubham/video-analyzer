# Video Analyzer — HP Omnibook, Core Ultra 7 + NPU (Windows)

Windows-native. FastAPI service samples a short clip (30-60s), captions every
frame on the **Intel NPU** via OpenVINO GenAI, and writes an ordered
`{t, caption}` array into `shared_volume\analysis\{clip}.analysis.json` so an
orchestrator (n8n) can hand the text to a text-LLM (Gemini) for the final
"what's happening" summary.

## Device
- HP Omnibook, Intel Core Ultra 7 (Meteor Lake or Lunar Lake), 16GB RAM
- Vision backend: **OpenVINO GenAI `VLMPipeline`** on device `"NPU"`
  - `0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu` (NF4, GenAI-ready) — default.
    **Requires Core Ultra Series 2 (Lunar Lake) NPU or newer.**
  - Meteor Lake NPU → export your own INT4 checkpoint: `.\export_model.ps1`
- Whisper (faster-whisper, int8 CPU) for audio; defaults to `off`

## Prerequisites (Windows)
1. Python 3.11+ on PATH (verified with 3.13.12) — `winget install Python.Python.3.13`
2. FFmpeg on PATH (`winget install Gyan.FFmpeg`) — needed for `ffprobe`/`ffmpeg`
3. Latest **Intel NPU driver** (Windows Update usually handles it)
4. Git (`winget install Git.Git`)

## Setup & run (PowerShell)

```powershell
.\setup.ps1        # venv + pip install requirements.txt
.\pull_model.ps1   # download the OpenVINO vision model -> models\
.\run.ps1          # http://localhost:31027
```

## Analyze a clip

```powershell
# option A: path-based (clip must be under DATA_ROOT, default shared_volume\clips\)
$body = @{ path = "D:\...\video-analyzer\shared_volume\clips\clip.mp4"; audio = $false; secondsPerFrame = 3.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:31027/analyze -ContentType "application/json" -Body $body

# option B: upload the video directly (saved temporarily, removed after)
curl.exe -X POST http://localhost:31027/analyze/upload `
  -F "file=@C:\path\to\clip.mp4" `
  -F "secondsPerFrame=3.0" -F "max_frames=120"
# -> shared_volume\analysis\clip.analysis.json
```

## Sampling & caps
- `secondsPerFrame` (JSON key) / `secondsPerFrame` (upload form field) sets the
  frame interval, default **3 s** (`SAMPLE_INTERVAL`). One caption per N
  seconds: a 6-min clip → ~120 frames.
- **No default cap** for clips ≤ 10 min (`NO_CAP_DURATION`, default 600s),
  unless `max_frames` is passed explicitly (which truncates).
- Clips > 10 min are capped at `MAX_FRAMES` (default 60) unless you pass
  `max_frames`. Raise `MAX_FRAMES` to sample the whole thing.

## Expected speed
NPU frame captioning is roughly **1-3 s/frame**; a 6-min clip at
`secondsPerFrame=3` (~120 frames) finishes in **~2-6 min**, vs ~9 h on the
CPU-only dev box.

## Environment overrides
| Env var | Default | Purpose |
| --- | --- | --- |
| `VLM_DEVICE` | `NPU` | OpenVINO device (`NPU`, `GPU`, `CPU`) |
| `VLM_MODEL` | `0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu` | HF repo id or local dir |
| `VLM_MODEL_DIR` | `models\` | Download cache |
| `VLM_MAX_TOKENS` | `64` | Max caption length |
| `VLM_GENERATE_HINT` | `FAST_COMPILE` | `FAST_COMPILE` (first-load speed) or `BEST_PERF` (run perf) |
| `SAMPLE_INTERVAL` | `3.0` | Seconds between sampled frames |
| `NO_CAP_DURATION` | `600` | Below this (s) no frame cap applies |
| `MAX_FRAMES` | `60` | Frame cap for clips above `NO_CAP_DURATION` or explicit `max_frames` |
| `AUDIO_MODEL` | `off` | faster-whisper size (`tiny`, `off`) |
| `DATA_ROOT` | `shared_volume` | Where clips + results live |

## Notes / troubleshooting
- First NPU run compiles the model (~1-2 min); blobs are cached in the model
  dir's `.npucache`.
- If the analyzer runs OOM-flaky on a Core Ultra Series 2 (200V) machine, set
  `DISABLE_OPENVINO_GENAI_NPU_L0=1` before `run.ps1`.
- Alternatively on Lunar Lake you can pull the default NF4/Qwen2.5-VL-3B
  checkpoint explicitly: `.\pull_model.ps1 0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu`
  (`.pull_model.ps1` validates the download looks like a GenAI VLM before
  reporting success).
- To export your own checkpoint (`Qwen2-VL-2B-Instruct`, INT4): `.\export_model.ps1`.
- n8n runs on another host and calls the analyzer at `<this-host>:31027`;
  keep the shared volume reachable over the network (e.g. SMB share) so both
  see the same `clips\` and `analysis\` folders.