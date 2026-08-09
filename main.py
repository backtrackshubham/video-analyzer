import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analyzer import Analyzer, DATA_ROOT

app = FastAPI(title="Video Analyzer (Alienware / Ollama)")
analyzer = Analyzer()
busy = threading.Lock()


class AnalyzeRequest(BaseModel):
    path: str
    audio: bool = False
    max_frames: int | None = None


@app.get("/health")
def health():
    info = analyzer.captioner.health()
    return {
        "status": "ok" if info["reachable"] else "degraded",
        "model": analyzer.captioner.model,
        "ollama": info,
        "data_root": DATA_ROOT,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not busy.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an analysis is already in progress")
    try:
        try:
            return analyzer.analyze(req.path, want_audio=req.audio, max_frames=req.max_frames)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    finally:
        busy.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
