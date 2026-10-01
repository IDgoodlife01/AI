# AI 모델 - CRNN을 대체하는 YOLO 화재·연기 객체 탐지 학습 및 추론
from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np


LABEL_NAMES = ("smoke", "fire")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _source_label(image: Path, data_root: Path, aihub_labels: dict[str, Path]) -> Path:
    if "dfire" in image.parts:
        return image.parent.parent / "labels" / f"{image.stem}.txt"
    label = aihub_labels.get(image.stem)
    if label is None:
        raise FileNotFoundError(f"YOLO 라벨 누락: {image}")
    return label


def prepare_dataset(splits, data_root: Path, destination: Path) -> Path:
    label_root = data_root / "aihub_clean" / "labels"
    aihub_labels = {path.stem: path for path in label_root.rglob("*.txt")}
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        for row in rows:
            key = hashlib.sha1(str(row.image).encode("utf-8")).hexdigest()[:12]
            name = f"{row.image.stem}_{key}"
            image_target = destination / "images" / split / f"{name}{row.image.suffix.lower()}"
            label_target = destination / "labels" / split / f"{name}.txt"
            _link(row.image, image_target)
            source_label = _source_label(row.image, data_root, aihub_labels)
            lines = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                if "aihub_clean" in row.image.parts:
                    class_id = 0 if "Smoke" in row.image.parts else 1
                lines.append(" ".join([str(class_id), *parts[1:5]]))
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(lines), encoding="utf-8")
    yaml_path = destination / "dataset.yaml"
    yaml_path.write_text(
        f"path: {destination.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: smoke\n  1: fire\n",
        encoding="utf-8",
    )
    return yaml_path


def predict_paths(artifact: Path, paths: list[Path], image_size: int = 640) -> np.ndarray:
    from ultralytics import YOLO

    model = YOLO(str(artifact))
    probabilities = np.zeros((len(paths), 2), dtype=np.float32)
    batch_size = 64
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        results = model.predict([str(path) for path in batch], imgsz=image_size,
                                verbose=False, stream=True)
        for offset, result in enumerate(results):
            if result.boxes is None:
                continue
            index = start + offset
            for class_id, confidence in zip(result.boxes.cls.cpu().numpy().astype(int),
                                            result.boxes.conf.cpu().numpy()):
                if class_id in (0, 1):
                    probabilities[index, class_id] = max(
                        probabilities[index, class_id], float(confidence)
                    )
    return probabilities


def train_yolo(splits, data_root: Path, output: Path, epochs: int, image_size: int,
               batch_size: int, seed: int, optimize_thresholds, metrics, smoke_test: bool = False):
    from ultralytics import YOLO

    dataset_dir = output / "yolo_dataset"
    yaml_path = prepare_dataset(splits, data_root, dataset_dir)
    started = time.perf_counter()
    base = "yolo11n.pt"
    model = YOLO(base)
    run = model.train(
        data=str(yaml_path),
        epochs=max(1, epochs),
        imgsz=image_size,
        batch=batch_size,
        seed=seed,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=10,
        project=str(output / "yolo_runs"),
        name="fire_smoke",
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    best = Path(run.save_dir) / "weights" / "best.pt"
    artifact = output / "models" / "yolo.pt"
    shutil.copy2(best, artifact)
    val_paths = [row.image for row in splits["val"]]
    val_true = np.asarray([row.label for row in splits["val"]], dtype=np.int8)
    thresholds = optimize_thresholds(val_true, predict_paths(artifact, val_paths, image_size))
    test_paths = [row.image for row in splits["test"]]
    test_true = np.asarray([row.label for row in splits["test"]], dtype=np.int8)
    result = metrics(test_true, predict_paths(artifact, test_paths, image_size), thresholds)
    result.update(model="yolo", seconds=round(time.perf_counter() - started, 3),
                  artifact=str(artifact))
    return result
