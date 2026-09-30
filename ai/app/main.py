# AI 서버 진입점 - 상태 확인, 추론 및 확정 이벤트 스냅샷 API
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.models import AVAILABLE_MODELS, model_status, predict
from app.services.event_service import EventStorage


app = FastAPI(title="Fire and Smoke AI", version="1.0.0")


@app.get("/health")
def health():
    models = model_status()
    return {"status": "ok" if all(models.values()) else "degraded", "models": models}


@app.post("/predict")
async def predict_image(
    image: UploadFile = File(...),
    model: str = Form("yolo"),
    threshold: float | None = Form(None),
):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of {sorted(AVAILABLE_MODELS)}")
    if threshold is not None and not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        return {"filename": image.filename, **predict(payload, model, threshold)}
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="upload a JPG or PNG image") from error


@app.post("/events/confirm")
async def confirm_event_snapshot(
    image: UploadFile = File(...),
    detection: str = Form("{}"),
):
    payload = await image.read()
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="upload a JPG or PNG image")
    try:
        detection_data = json.loads(detection)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="detection must be valid JSON") from error
    if not isinstance(detection_data, dict):
        raise HTTPException(status_code=400, detail="detection must be a JSON object")
    return EventStorage().save_snapshot(frame, detection_data)
