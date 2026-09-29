# AI 서버 - 화재·연기 분류 FastAPI 엔드포인트
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .model_service import AVAILABLE_MODELS, model_status, predict


app = FastAPI(title="Fire and Smoke AI Server", version="1.0.0")


@app.get("/health")
def health():
    models = model_status()
    return {"status": "ok" if all(models.values()) else "degraded", "models": models}


@app.post("/predict")
async def predict_image(
    image: UploadFile = File(...),
    model: str = Form("random_forest"),
    threshold: float | None = Form(None),
):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"model은 {sorted(AVAILABLE_MODELS)} 중 하나여야 합니다.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold는 0~1 범위여야 합니다.")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="빈 이미지입니다.")
    try:
        result = predict(payload, model, threshold)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="JPG 또는 PNG 이미지를 전송하세요.")
    return {"filename": image.filename, **result}
