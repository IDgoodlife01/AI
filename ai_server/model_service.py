# AI 서버 - 학습된 6종 모델 로딩 및 이미지 예측
from __future__ import annotations

import io
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "ai_model" / "src"
MODEL_CANDIDATES = (
    Path(os.environ["AI_MODELS_DIR"]).resolve() if os.environ.get("AI_MODELS_DIR") else None,
    ROOT / "ai_model" / "results_optimized" / "models",
    ROOT / "ai_model" / "results_v2" / "models",
    ROOT / "ai_model" / "results" / "models",
)
if str(MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODEL_SOURCE))

from train_all import LABEL_NAMES, array_features, build_deep_model  # noqa: E402


DEEP_MODELS = {"cnn", "mobilenet_v2"}
YOLO_MODELS = {"yolo"}
CLASSICAL_MODELS = {"random_forest", "lightgbm", "xgboost"}
AVAILABLE_MODELS = DEEP_MODELS | YOLO_MODELS | CLASSICAL_MODELS


def select_models_dir() -> Path:
    expected = {f"{name}.pt" for name in DEEP_MODELS | YOLO_MODELS} | {
        f"{name}.joblib" for name in CLASSICAL_MODELS
    }
    for candidate in MODEL_CANDIDATES:
        if candidate and candidate.is_dir() and expected <= {item.name for item in candidate.iterdir()}:
            return candidate
    return next(candidate for candidate in MODEL_CANDIDATES if candidate is not None)


MODELS_DIR = select_models_dir()


@lru_cache(maxsize=6)
def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    if model_name in DEEP_MODELS:
        path = MODELS_DIR / f"{model_name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = build_deep_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return {"model": model, "image_size": int(checkpoint["image_size"]),
                "thresholds": checkpoint.get("thresholds", [0.5, 0.5])}
    if model_name in YOLO_MODELS:
        from ultralytics import YOLO
        return {"model": YOLO(str(MODELS_DIR / "yolo.pt")), "image_size": 640,
                "thresholds": [0.25, 0.25]}
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")


def predict(image_bytes: bytes, model_name: str, threshold: float | None):
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        boxes = []
        if model_name in YOLO_MODELS:
            bundle = load_model(model_name)
            probabilities = np.zeros(2, dtype=np.float32)
            result = bundle["model"].predict(image, imgsz=bundle["image_size"], verbose=False)[0]
            if result.boxes is not None:
                for xyxy, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    if class_id in (0, 1):
                        probabilities[class_id] = max(probabilities[class_id], float(confidence))
                        boxes.append({
                            "label": LABEL_NAMES[class_id],
                            "confidence": round(float(confidence), 6),
                            "xyxy": [round(float(value), 2) for value in xyxy],
                        })
        elif model_name in DEEP_MODELS:
            bundle = load_model(model_name)
            transform = transforms.Compose(
                [transforms.Resize((bundle["image_size"], bundle["image_size"])), transforms.ToTensor()]
            )
            with torch.inference_mode():
                probabilities = torch.sigmoid(bundle["model"](transform(image).unsqueeze(0)))[0].numpy()
        else:
            bundle = load_model(model_name)
            size = int(bundle.get("image_size", 64))
            rgb = np.asarray(image.resize((size, size)), dtype=np.uint8)
            features = array_features(rgb).reshape(1, -1)
            values = []
            for estimator, output in zip(bundle["model"].estimators_, bundle["model"].predict_proba(features)):
                classes = estimator.classes_.tolist()
                values.append(float(output[0, classes.index(1)]) if 1 in classes else float(classes[0] == 1))
            probabilities = np.asarray(values)
        saved_thresholds = np.asarray(bundle.get("thresholds", [0.5, 0.5]), dtype=np.float32)
    thresholds = (
        np.asarray([threshold, threshold], dtype=np.float32)
        if threshold is not None else saved_thresholds
    )
    return {
        "model": model_name,
        "thresholds": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, thresholds)},
        "probabilities": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, probabilities)},
        "detected": [name for name, value, cutoff in zip(LABEL_NAMES, probabilities, thresholds) if value >= cutoff],
        "boxes": boxes,
    }


def model_status():
    status = {}
    for name in sorted(AVAILABLE_MODELS):
        suffix = ".pt" if name in DEEP_MODELS | YOLO_MODELS else ".joblib"
        status[name] = (MODELS_DIR / f"{name}{suffix}").is_file()
    return status
