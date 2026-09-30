# AI model module - load six trained models and run image inference
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


LABEL_NAMES = ("smoke", "fire")
DEEP_MODELS = {"cnn", "mobilenet_v2"}
YOLO_MODELS = {"yolo"}
CLASSICAL_MODELS = {"random_forest", "lightgbm", "xgboost"}
AVAILABLE_MODELS = DEEP_MODELS | YOLO_MODELS | CLASSICAL_MODELS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("AI_MODELS_DIR", PROJECT_ROOT / "model")).resolve()


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, 2)

    def forward(self, value):
        return self.classifier(self.features(value).flatten(1))


def build_deep_model(name: str) -> nn.Module:
    if name == "cnn":
        return SmallCNN()
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        return model
    raise ValueError(name)


def array_features(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_uint8, dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        values = rgb[:, :, channel]
        histogram, _ = np.histogram(values, bins=16, range=(0, 1), density=True)
        features.extend(histogram.tolist())
        features.extend([float(values.mean()), float(values.std())])
    gray = rgb.mean(axis=2)
    features.extend([
        float(np.abs(np.diff(gray, axis=1)).mean()),
        float(np.abs(np.diff(gray, axis=0)).mean()),
    ])
    return np.asarray(features, dtype=np.float32)


@lru_cache(maxsize=6)
def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name in DEEP_MODELS:
        checkpoint = torch.load(MODELS_DIR / f"{model_name}.pt", map_location="cpu", weights_only=True)
        model = build_deep_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return {
            "model": model,
            "image_size": int(checkpoint["image_size"]),
            "thresholds": checkpoint.get("thresholds", [0.5, 0.5]),
        }
    if model_name in YOLO_MODELS:
        from ultralytics import YOLO

        return {"model": YOLO(str(MODELS_DIR / "yolo.pt")), "image_size": 640, "thresholds": [0.25, 0.25]}
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")


def predict(image_bytes: bytes, model_name: str = "yolo", threshold: float | None = None):
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
            transform = transforms.Compose([
                transforms.Resize((bundle["image_size"], bundle["image_size"])),
                transforms.ToTensor(),
            ])
            with torch.inference_mode():
                probabilities = torch.sigmoid(bundle["model"](transform(image).unsqueeze(0)))[0].numpy()
        else:
            bundle = load_model(model_name)
            size = int(bundle.get("image_size", 64))
            features = array_features(np.asarray(image.resize((size, size)), dtype=np.uint8)).reshape(1, -1)
            values = []
            for estimator, output in zip(bundle["model"].estimators_, bundle["model"].predict_proba(features)):
                classes = estimator.classes_.tolist()
                values.append(float(output[0, classes.index(1)]) if 1 in classes else float(classes[0] == 1))
            probabilities = np.asarray(values)
        saved_thresholds = np.asarray(bundle.get("thresholds", [0.5, 0.5]), dtype=np.float32)

    thresholds = np.asarray([threshold, threshold], dtype=np.float32) if threshold is not None else saved_thresholds
    return {
        "model": model_name,
        "thresholds": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, thresholds)},
        "probabilities": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, probabilities)},
        "detected": [name for name, value, cutoff in zip(LABEL_NAMES, probabilities, thresholds) if value >= cutoff],
        "boxes": boxes,
    }


def model_status():
    return {
        name: (MODELS_DIR / f"{name}{'.pt' if name in DEEP_MODELS | YOLO_MODELS else '.joblib'}").is_file()
        for name in sorted(AVAILABLE_MODELS)
    }
