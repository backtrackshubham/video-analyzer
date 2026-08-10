import logging
import os
import shutil
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from analyzer import Analyzer, DATA_ROOT

log = logging.getLogger("analyzer.http")

app = FastAPI(title="Video Analyzer (NPU / OpenVINO)")
analyzer = Analyzer()
busy = threading.Lock()


class AnalyzeRequest(BaseModel):
    path: str
    audio: bool = False
    max_frames: int | None = None
    secondsPerFrame: float | None = None


def _acquire():
    if not busy.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an analysis is already in progress")


@app.get("/health")
def health():
    return {"status": "ok", "inference": analyzer.captioner.health(), "data_root": DATA_ROOT}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    log.info("received POST /analyze: clip=%s audio=%s secondsPerFrame=%s max_frames=%s", req.path, req.audio, req.secondsPerFrame, req.max_frames)
    _acquire()
    log.info("request accepted (no other analysis running)")
    try:
        try:
            result = analyzer.analyze(
                req.path,
                want_audio=req.audio,
                max_frames=req.max_frames,
                seconds_per_frame=req.secondsPerFrame,
            )
            log.info("POST /analyze finished in %d frames -> status=%s", len(result.get("frames", [])), result.get("status"))
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    finally:
        busy.release()
        log.info("request released (busy=false)")


@app.post("/analyze/upload")
def analyze_upload(
    file: UploadFile = File(...),
    audio: bool = Form(False),
    max_frames: int | None = Form(None),
    secondsPerFrame: float | None = Form(None),
):
    log.info("received POST /analyze/upload: content_type=%s size=%s audio=%s secondsPerFrame=%s max_frames=%s", file.content_type, file.size, audio, secondsPerFrame, max_frames)
    _acquire()
    log.info("request accepted (no other analysis running)")
    tmp_path = None
    try:
        upload_dir = os.path.join(DATA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        safe = os.path.basename(file.filename or "clip")
        tmp_path = os.path.join(upload_dir, uuid.uuid4().hex[:12] + "_" + safe)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f, length=1024 * 1024)
        log.info("upload saved to %s (%d bytes)", tmp_path, os.path.getsize(tmp_path))
        try:
            result = analyzer.analyze(
                tmp_path,
                want_audio=audio,
                max_frames=max_frames,
                seconds_per_frame=secondsPerFrame,
            )
            log.info("POST /analyze/upload finished -> status=%s", result.get("status"))
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    finally:
        busy.release()
        log.info("request released (busy=false)")
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ANALYZER_PORT", "31027"))
    uvicorn.run(app, host="0.0.0.0", port=port)