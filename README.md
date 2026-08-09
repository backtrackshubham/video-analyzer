# Video Analyzer — HP Omnibook, Core Ultra 7 + NPU (Windows)

Windows-native. FastAPI service samples a short clip (30-60s), captions every
frame on the **Intel NPU** via OpenVINO GenAI, and writes an ordered
`{t, caption}` array into `shared_volume\analysis\{clip}.analysis.json` so an
orchestrator (n8n) can hand the text to a text-LLM (Gemini) for the final
"what's happening" summary.

## Device
- HP Omnibook, Intel Core Ultra 7 (Meteor Lake or Lunar Lake), 16GB RAM
- Vision backend: **OpenVINO GenAI `VLMPipeline`** on device `"NPU"`
  - `llmware/qwen2-vl-2b-instruct-ov` (INT4, works on Meteor Lake *and* Lunar Lake) — default
  - `0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu` (NF4, **Lunar Lake only**) — optionally
    set `VLM_MODEL` to it for slightly better accuracy
- Whisper (faster-whisper, int8 CPU) for audio; defaults to `off`

## Prerequisites (Windows)
1. Python 3.11 on PATH (`winget install Python.Python.3.11`)
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
# put clip.mp4 under shared_volume\clips\
$body = @{ path = "C:\...\video-analyzer-setup\shared_volume\clips\clip.mp4"; audio = $false; max_frames = 60 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:31027/analyze -ContentType "application/json" -Body $body
# -> shared_volume\analysis\clip.analysis.json
```

## Expected speed
NPU frame captioning is roughly **1-3 s/frame** → a 30-60s clip at
`SAMPLE_FPS=1` (≤60 frames) finishes in **~1-4 min**, vs ~4.6 h on the CPU-only
dev box.

## Environment overrides
| Env var | Default | Purpose |
| --- | --- | --- |
| `VLM_DEVICE` | `NPU` | OpenVINO device (`NPU`, `GPU`, `CPU`) |
| `VLM_MODEL` | `llmware/qwen2-vl-2b-instruct-ov` | HF repo id or local dir |
| `VLM_MODEL_DIR` | `models\` | Download cache |
| `VLM_MAX_TOKENS` | `64` | Max caption length |
| `SAMPLE_FPS` / `MAX_FRAMES` | `1.0` / `60` | Sampling |
| `AUDIO_MODEL` | `off` | faster-whisper size (`tiny`, `off`) |
| `DATA_ROOT` | `shared_volume` | Where clips + results live |

## Notes / troubleshooting
- First NPU run compiles the model (~1-2 min); blobs are cached in the model
  dir's `.npucache`.
- If the analyzer runs OOM-flaky on a Core Ultra Series 2 (200V) machine, set
  `DISABLE_OPENVINO_GENAI_NPU_L0=1` before `run.ps1`.
- Alternatively on Lunar Lake, `0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu`:
  `pull_model.ps1 0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu` then
  `$env:VLM_MODEL="0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu"`.
- To export your own checkpoint (`Qwen2-VL-2B-Instruct`, INT4): `.\export_model.ps1`.
- n8n runs on another host and calls the analyzer at `<this-host>:31027`;
  keep the shared volume reachable over the network (e.g. SMB share) so both
  see the same `clips\` and `analysis\` folders.