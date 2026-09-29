# AI 모델 - 6종 성능 비교 및 Test 혼동행렬 PNG 생성
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from train_all import CachedFireDataset, array_features, build_deep_model, predict_deep


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
CACHE = RESULTS / "cache"
REPORTS = RESULTS / "reports"
LABEL_NAMES = ("Smoke", "Fire")
MODEL_NAMES = ("cnn", "mobilenet_v2", "yolo", "random_forest", "lightgbm", "xgboost")


def load_test_dataset(size: int = 64):
    images = np.load(CACHE / f"test_{size}_images.npy", mmap_mode="r")
    labels = np.load(CACHE / f"test_{size}_labels.npy", mmap_mode="r")
    return CachedFireDataset(images, labels)


def predict_model(model_name: str, dataset: CachedFireDataset):
    if model_name == "yolo":
        from yolo_model import predict_paths
        rows = list(csv.DictReader((RESULTS / "dataset_manifest.csv").open(encoding="utf-8-sig")))
        test_rows = [row for row in rows if row["split"] == "test"]
        y_true = np.asarray([[int(row["smoke"]), int(row["fire"])] for row in test_rows], dtype=np.int8)
        y_prob = predict_paths(MODELS / "yolo.pt", [Path(row["image"]) for row in test_rows])
        metric_rows = list(csv.DictReader((RESULTS / "model_comparison.csv").open(encoding="utf-8-sig")))
        metric = next(row for row in metric_rows if row["model"] == "yolo")
        return y_true, y_prob, np.asarray([float(metric["smoke_threshold"]), float(metric["fire_threshold"])])
    if model_name in {"cnn", "mobilenet_v2"}:
        checkpoint = torch.load(MODELS / f"{model_name}.pt", map_location="cpu", weights_only=True)
        model = build_deep_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        y_true, y_prob = predict_deep(model, loader, torch.device("cpu"))
        return y_true, y_prob, np.asarray(checkpoint.get("thresholds", [0.5, 0.5]))
    bundle = joblib.load(MODELS / f"{model_name}.joblib")
    x_test = np.stack([array_features(dataset.images[index]) for index in range(len(dataset))])
    probabilities = []
    for estimator, output in zip(bundle["model"].estimators_, bundle["model"].predict_proba(x_test)):
        classes = estimator.classes_.tolist()
        probabilities.append(output[:, classes.index(1)] if 1 in classes else np.full(len(x_test), float(classes[0] == 1)))
    return (np.asarray(dataset.labels, dtype=np.int8), np.column_stack(probabilities),
            np.asarray(bundle.get("thresholds", [0.5, 0.5])))


def plot_matrix(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for index, (axis, label_name) in enumerate(zip(axes, LABEL_NAMES)):
        matrix = confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(label_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], [f"non_{label_name.lower()}", label_name.lower()])
        axis.set_yticks([0, 1], [f"non_{label_name.lower()}", label_name.lower()])
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black", fontsize=18)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model_name} — Test Confusion Matrix", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015, f"Test n = {len(y_true):,} | thresholds: smoke={thresholds[0]:.2f}, fire={thresholds[1]:.2f}", ha="center", color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(REPORTS / f"{model_name}_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(test_count: int):
    with (RESULTS / "model_comparison.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    display = {name: name.replace("_", " ").title() for name in MODEL_NAMES}
    values = {row["model"]: row for row in rows}
    x = np.arange(len(MODEL_NAMES))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 6))
    series = [
        ("f1_micro", "Micro F1", "#4f79c7"),
        ("precision_micro", "Precision", "#f57c24"),
        ("recall_micro", "Recall", "#58a65c"),
    ]
    for offset, (key, label, color) in zip((-width, 0, width), series):
        scores = [float(values[name][key]) for name in MODEL_NAMES]
        bars = axis.bar(x + offset, scores, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=3, fontsize=8)
    axis.set_title("6 Model Comparison — Test", fontsize=17, fontweight="bold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x, [display[name] for name in MODEL_NAMES])
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="lower left")
    fig.text(0.5, 0.015, f"Test n = {test_count:,} | Validation-optimized thresholds", ha="center", color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REPORTS / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    global RESULTS, MODELS, CACHE, REPORTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    MODELS = RESULTS / "models"
    CACHE = RESULTS / "cache"
    REPORTS = RESULTS / "reports"
    REPORTS.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(MODELS / "cnn.pt", map_location="cpu", weights_only=True)
    dataset = load_test_dataset(int(checkpoint["image_size"]))
    plot_comparison(len(dataset))
    for model_name in MODEL_NAMES:
        print(f"[{model_name}] Test 예측 및 혼동행렬 생성", flush=True)
        y_true, y_prob, thresholds = predict_model(model_name, dataset)
        plot_matrix(model_name, y_true, y_prob, thresholds)
    print(f"완료: {REPORTS}", flush=True)


if __name__ == "__main__":
    main()
