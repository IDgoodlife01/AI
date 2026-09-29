# AI 모델 - 학습된 6종 모델의 단일 이미지 화재·연기 예측 실행
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_all import LABEL_NAMES, array_features, build_deep_model


DEEP_MODELS = {"cnn", "mobilenet_v2"}
YOLO_MODELS = {"yolo"}
CLASSICAL_MODELS = {"random_forest", "lightgbm", "xgboost"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_models_dir() -> Path:
    candidates = (
        PROJECT_ROOT / "ai_model" / "results_optimized" / "models",
        PROJECT_ROOT / "ai_model" / "results_v2" / "models",
        PROJECT_ROOT / "ai_model" / "results" / "models",
    )
    expected = {f"{name}.pt" for name in DEEP_MODELS | YOLO_MODELS} | {
        f"{name}.joblib" for name in CLASSICAL_MODELS
    }
    return next((path for path in candidates if path.is_dir() and expected <= {item.name for item in path.iterdir()}), candidates[0])


def predict_deep(model_name: str, artifact: Path, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
    model = build_deep_model(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    size = int(checkpoint["image_size"])
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].numpy()
    return probabilities, np.asarray(checkpoint.get("thresholds", [0.5, 0.5]), dtype=np.float32)


def predict_classical(artifact: Path, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bundle = joblib.load(artifact)
    size = int(bundle.get("image_size", 64))
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.uint8)
    features = array_features(rgb).reshape(1, -1)
    probabilities = []
    for estimator, output in zip(bundle["model"].estimators_, bundle["model"].predict_proba(features)):
        classes = estimator.classes_.tolist()
        probabilities.append(float(output[0, classes.index(1)]) if 1 in classes else float(classes[0] == 1))
    return np.asarray(probabilities), np.asarray(bundle.get("thresholds", [0.5, 0.5]), dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="화재·연기 이미지 분류")
    parser.add_argument("image", type=Path, help="판정할 JPG/PNG 이미지")
    parser.add_argument(
        "--model",
        choices=sorted(DEEP_MODELS | YOLO_MODELS | CLASSICAL_MODELS),
        default="random_forest",
        help="사용할 모델(기본: random_forest)",
    )
    parser.add_argument("--models-dir", type=Path, default=default_models_dir())
    parser.add_argument("--threshold", type=float, default=None, help="미지정 시 검증셋 최적값 사용")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {args.image}")
    suffix = ".pt" if args.model in DEEP_MODELS | YOLO_MODELS else ".joblib"
    artifact = args.models_dir / f"{args.model}{suffix}"
    if not artifact.is_file():
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {artifact}")
    if args.model in YOLO_MODELS:
        from yolo_model import predict_paths
        probabilities = predict_paths(artifact, [args.image])[0]
        saved_thresholds = np.asarray([0.25, 0.25], dtype=np.float32)
    else:
        probabilities, saved_thresholds = (
            predict_deep(args.model, artifact, args.image)
            if args.model in DEEP_MODELS else predict_classical(artifact, args.image)
        )
    thresholds = (
        np.asarray([args.threshold, args.threshold], dtype=np.float32)
        if args.threshold is not None else saved_thresholds
    )
    result = {
        "image": str(args.image.resolve()),
        "model": args.model,
        "thresholds": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, thresholds)},
        "probabilities": {name: round(float(value), 6) for name, value in zip(LABEL_NAMES, probabilities)},
        "detected": [
            name for name, value, threshold in zip(LABEL_NAMES, probabilities, thresholds)
            if value >= threshold
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
