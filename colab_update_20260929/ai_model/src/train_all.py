# AI 모델 - 화재·연기 6종 모델 학습 및 동일 테스트셋 비교
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import models, transforms


LABEL_NAMES = ("smoke", "fire")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Sample:
    image: Path
    label: tuple[int, int]
    split: str


class FireDataset(Dataset):
    def __init__(self, samples: list[Sample], size: int):
        self.samples = samples
        self.transform = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(sample.label, dtype=torch.float32)


class CachedFireDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1).float().div_(255)
        return image, torch.from_numpy(np.array(self.labels[index], copy=True)).float()


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

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 16)),
        )
        self.sequence = nn.LSTM(96 * 8, 96, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(192, 2)

    def forward(self, x):
        x = self.features(x).permute(0, 3, 1, 2).flatten(2)
        x, _ = self.sequence(x)
        return self.classifier(x.mean(dim=1))


def label_from_yolo(path: Path) -> tuple[int, int]:
    smoke = fire = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            smoke |= class_id == 0
            fire |= class_id == 1
    return int(smoke), int(fire)


def discover_dfire(data_root: Path) -> list[Sample]:
    dfire = data_root / "dfire"
    samples: list[Sample] = []
    for source_split in ("train", "test"):
        image_dir = dfire / source_split / "images"
        label_dir = dfire / source_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"D-Fire 폴더 구조가 올바르지 않습니다: {image_dir}")
        for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label = label_from_yolo(label_dir / f"{image.stem}.txt")
            if source_split == "test":
                split = "test"
            else:
                bucket = int(hashlib.sha1(image.name.encode()).hexdigest()[:8], 16) % 100
                split = "val" if bucket < 18 else "train"
            samples.append(Sample(image.resolve(), label, split))
    if not samples:
        raise RuntimeError("학습 가능한 D-Fire 이미지가 없습니다.")
    return samples


def discover_aihub(data_root: Path) -> list[Sample]:
    image_root = data_root / "aihub_clean" / "images"
    label_root = data_root / "aihub_clean" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        return []
    samples: list[Sample] = []
    missing_labels: list[Path] = []
    label_stems = {path.stem for path in label_root.rglob("*.txt")}
    candidates = []
    for image in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image.relative_to(image_root)
        category = relative.parts[1].lower() if len(relative.parts) > 1 else ""
        if category == "smoke":
            label = (1, 0)
        elif category == "flame":
            label = (0, 1)
        else:
            continue
        if image.stem not in label_stems:
            missing_labels.append(relative)
            continue
        scene_key = "/".join(relative.parts[:-1])
        candidates.append((image, label, category, scene_key))
    scene_splits = {}
    for category in ("smoke", "flame"):
        scenes = sorted(
            {row[3] for row in candidates if row[2] == category},
            key=lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest(),
        )
        train_end = max(1, round(len(scenes) * 0.70))
        val_end = min(len(scenes) - 1, train_end + max(1, round(len(scenes) * 0.15)))
        for index, scene in enumerate(scenes):
            scene_splits[scene] = "train" if index < train_end else "val" if index < val_end else "test"
    for image, label, _category, scene_key in candidates:
        split = scene_splits[scene_key]
        samples.append(Sample(image.resolve(), label, split))
    if missing_labels:
        raise RuntimeError(f"AI Hub 라벨 누락: {len(missing_labels)}개")
    return samples


def write_manifest(samples: list[Sample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "smoke", "fire", "split"])
        for sample in samples:
            writer.writerow([sample.image, *sample.label, sample.split])


def build_image_cache(splits: dict[str, list[Sample]], cache_dir: Path, size: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split, rows in splits.items():
        image_path = cache_dir / f"{split}_{size}_images.npy"
        label_path = cache_dir / f"{split}_{size}_labels.npy"
        expected_shape = (len(rows), size, size, 3)
        reuse = image_path.exists() and label_path.exists()
        if reuse:
            images = np.load(image_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            reuse = images.shape == expected_shape and labels.shape == (len(rows), 2)
        if not reuse:
            images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=expected_shape)
            labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int8, shape=(len(rows), 2))
            for index, row in enumerate(rows):
                with Image.open(row.image) as image:
                    images[index] = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.uint8)
                labels[index] = row.label
                if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                    print(f"[cache] {split} {index + 1}/{len(rows)}", flush=True)
            images.flush()
            labels.flush()
            images = np.load(image_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
        datasets[split] = CachedFireDataset(images, labels)
    return datasets


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thresholds = []
    for class_index in range(y_true.shape[1]):
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in np.linspace(0.05, 0.95, 91):
            predicted = (y_prob[:, class_index] >= threshold).astype(np.int8)
            score = precision_recall_fscore_support(
                y_true[:, class_index], predicted, average="binary", zero_division=0
            )[2]
            if score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype=np.float32)


def metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None
) -> dict[str, float]:
    thresholds = np.asarray([0.5, 0.5] if thresholds is None else thresholds)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_micro": float(precision),
        "recall_micro": float(recall),
        "f1_micro": float(f1),
        "smoke_f1": float(per_class[2][0]),
        "fire_f1": float(per_class[2][1]),
        "smoke_threshold": float(thresholds[0]),
        "fire_threshold": float(thresholds[1]),
    }


def build_deep_model(name: str) -> nn.Module:
    if name == "cnn":
        return SmallCNN()
    if name == "crnn":
        return CRNN()
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        return model
    raise ValueError(name)


def train_deep(
    name: str,
    splits: dict[str, list[Sample]],
    datasets: dict[str, Dataset],
    output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    image_size: int,
    hard_negative_epochs: int,
    resume: bool = False,
) -> dict[str, float | str]:
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, rows in splits.items()
    }
    model = build_deep_model(name).to(device)
    targets = np.asarray([row.label for row in splits["train"]], dtype=np.float32)
    positives = targets.sum(axis=0)
    pos_weight = torch.tensor(
        (len(targets) - positives) / np.maximum(positives, 1), device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_val = -1.0
    stale_epochs = 0
    checkpoint = output / "models" / f"{name}.pt"
    if resume and checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=True)
        if previous["image_size"] != image_size:
            raise ValueError("Resume image size does not match checkpoint")
        model.load_state_dict(previous["state_dict"])
        best_val = float(previous["val_f1"])
        print(f"[{name}] restored best weights val_f1={best_val:.4f}; optimizer restarted", flush=True)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()
        val_true, val_prob = predict_deep(model, loaders["val"], device)
        thresholds = optimize_thresholds(val_true, val_prob)
        val_f1 = metrics(val_true, val_prob, thresholds)["f1_micro"]
        print(f"[{name}] epoch={epoch + 1}/{epochs} val_f1={val_f1:.4f}", flush=True)
        if val_f1 >= best_val:
            best_val = val_f1
            stale_epochs = 0
            torch.save(
                {"model": name, "state_dict": model.state_dict(), "labels": LABEL_NAMES,
                 "image_size": image_size, "val_f1": val_f1,
                 "thresholds": thresholds.tolist()},
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                print(f"[{name}] early_stop epoch={epoch + 1}", flush=True)
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    thresholds = np.asarray(saved.get("thresholds", [0.5, 0.5]), dtype=np.float32)
    if hard_negative_epochs > 0:
        train_eval_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=False, num_workers=0)
        train_true, train_prob = predict_deep(model, train_eval_loader, device)
        negative_indices = np.flatnonzero(train_true.sum(axis=1) == 0)
        hard_indices = negative_indices[
            np.any(train_prob[negative_indices] >= thresholds.reshape(1, -1), axis=1)
        ]
        if len(hard_indices):
            hard_dataset = Subset(datasets["train"], hard_indices.tolist())
            hard_loader = DataLoader(
                ConcatDataset([datasets["train"], hard_dataset, hard_dataset]),
                batch_size=batch_size, shuffle=True, num_workers=0,
            )
            print(f"[{name}] hard_negatives={len(hard_indices)}", flush=True)
            for hard_epoch in range(hard_negative_epochs):
                model.train()
                for images, labels in hard_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss_fn(model(images), labels).backward()
                    optimizer.step()
                val_true, val_prob = predict_deep(model, loaders["val"], device)
                candidate_thresholds = optimize_thresholds(val_true, val_prob)
                candidate_f1 = metrics(val_true, val_prob, candidate_thresholds)["f1_micro"]
                print(f"[{name}] hard_epoch={hard_epoch + 1}/{hard_negative_epochs} val_f1={candidate_f1:.4f}", flush=True)
                if candidate_f1 >= best_val:
                    best_val = candidate_f1
                    thresholds = candidate_thresholds
                    torch.save(
                        {"model": name, "state_dict": model.state_dict(), "labels": LABEL_NAMES,
                         "image_size": image_size, "val_f1": candidate_f1,
                         "thresholds": thresholds.tolist(), "hard_negatives": len(hard_indices)},
                        checkpoint,
                    )
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(saved["state_dict"])
    y_true, y_prob = predict_deep(model, loaders["test"], device)
    thresholds = np.asarray(saved.get("thresholds", [0.5, 0.5]), dtype=np.float32)
    result = metrics(y_true, y_prob, thresholds)
    result.update(model=name, seconds=round(time.perf_counter() - started, 3), artifact=str(checkpoint))
    return result


@torch.inference_mode()
def predict_deep(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    truths, probabilities = [], []
    for images, labels in loader:
        probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        truths.append(labels.numpy())
    return np.concatenate(truths).astype(np.int8), np.concatenate(probabilities)


def image_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        values = rgb[:, :, channel]
        histogram, _ = np.histogram(values, bins=16, range=(0, 1), density=True)
        features.extend(histogram.tolist())
        features.extend([float(values.mean()), float(values.std())])
    gray = rgb.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    features.extend([float(gx), float(gy)])
    return np.asarray(features, dtype=np.float32)


def array_features(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_uint8, dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        values = rgb[:, :, channel]
        histogram, _ = np.histogram(values, bins=16, range=(0, 1), density=True)
        features.extend(histogram.tolist())
        features.extend([float(values.mean()), float(values.std())])
    gray = rgb.mean(axis=2)
    features.extend([float(np.abs(np.diff(gray, axis=1)).mean()), float(np.abs(np.diff(gray, axis=0)).mean())])
    return np.asarray(features, dtype=np.float32)


def feature_matrix(rows: list[Sample]):
    x = np.stack([image_features(row.image) for row in rows])
    y = np.asarray([row.label for row in rows], dtype=np.int8)
    return x, y


def cached_feature_matrix(dataset: CachedFireDataset):
    x = np.stack([array_features(dataset.images[index]) for index in range(len(dataset))])
    return x, np.asarray(dataset.labels, dtype=np.int8)


def make_classical(name: str, seed: int):
    if name == "random_forest":
        base = RandomForestClassifier(n_estimators=160, class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(n_estimators=240, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=seed, verbose=-1)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        base = XGBClassifier(n_estimators=240, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=seed, eval_metric="logloss")
    else:
        raise ValueError(name)
    return MultiOutputClassifier(base, n_jobs=1)


def train_classical(name: str, matrices, output: Path, seed: int):
    x_train, y_train, x_val, y_val, x_test, y_test = matrices
    model = make_classical(name, seed)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    def positive_probabilities(features):
        probability_columns = []
        for estimator, item in zip(model.estimators_, model.predict_proba(features)):
            classes = estimator.classes_.tolist()
            if 1 in classes:
                probability_columns.append(item[:, classes.index(1)])
            else:
                probability_columns.append(np.full(len(features), float(classes[0] == 1)))
        return np.column_stack(probability_columns)
    val_probabilities = positive_probabilities(x_val)
    thresholds = optimize_thresholds(y_val, val_probabilities)
    train_probabilities = positive_probabilities(x_train)
    negative_indices = np.flatnonzero(y_train.sum(axis=1) == 0)
    hard_indices = negative_indices[
        np.any(train_probabilities[negative_indices] >= thresholds.reshape(1, -1), axis=1)
    ]
    if len(hard_indices):
        model.fit(
            np.concatenate([x_train, x_train[hard_indices], x_train[hard_indices]]),
            np.concatenate([y_train, y_train[hard_indices], y_train[hard_indices]]),
        )
        val_probabilities = positive_probabilities(x_val)
        thresholds = optimize_thresholds(y_val, val_probabilities)
    print(f"[{name}] hard_negatives={len(hard_indices)}", flush=True)
    probabilities = positive_probabilities(x_test)
    artifact = output / "models" / f"{name}.joblib"
    joblib.dump({"model": model, "labels": LABEL_NAMES, "feature": "rgb_histogram_edge_v1",
                 "thresholds": thresholds.tolist(), "hard_negatives": int(len(hard_indices))}, artifact)
    result = metrics(y_test, probabilities, thresholds)
    result.update(model=name, seconds=round(time.perf_counter() - started, 3), artifact=str(artifact))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "raw")
    parser.add_argument("--aihub-root", type=Path, default=PROJECT_ROOT / "ai_dataset_archive")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ai_model" / "results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hard-negative-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--completed-deep", nargs="*", choices=["cnn", "mobilenet_v2"], default=[])
    parser.add_argument("--yolo-image-size", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dfire_samples = discover_dfire(args.data_root)
    aihub_samples = discover_aihub(args.data_root)
    samples = dfire_samples + aihub_samples
    print(
        f"dataset=dfire:{len(dfire_samples)} aihub:{len(aihub_samples)} total:{len(samples)}",
        flush=True,
    )
    splits = {name: [row for row in samples if row.split == name] for name in ("train", "val", "test")}
    if args.smoke_test:
        limits = {"train": 128, "val": 64, "test": 64}
        rng = random.Random(args.seed)
        splits = {
            name: rng.sample(rows, min(limits[name], len(rows)))
            for name, rows in splits.items()
        }
        args.epochs = 1
        args.hard_negative_epochs = 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    write_manifest([row for rows in splits.values() for row in rows], args.output / "dataset_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = {name: len(rows) for name, rows in splits.items()}
    print(f"split={counts} device={device}", flush=True)
    datasets = build_image_cache(splits, args.output / "cache", args.image_size)
    results = []
    for name in ("cnn", "mobilenet_v2"):
        completed = name in args.completed_deep
        if completed and (not args.resume or not (args.output / "models" / f"{name}.pt").exists()):
            raise ValueError("Completed model requires --resume and an existing checkpoint")
        results.append(train_deep(name, splits, datasets, args.output, device,
                                  0 if completed else args.epochs, args.batch_size, args.image_size,
                                  0 if completed else args.hard_negative_epochs, args.resume))
    from yolo_model import train_yolo
    results.append(train_yolo(splits, args.data_root, args.output, args.epochs,
                              320 if args.smoke_test else args.yolo_image_size,
                              args.batch_size, args.seed, optimize_thresholds, metrics,
                              args.smoke_test))
    x_train, y_train = cached_feature_matrix(datasets["train"])
    x_val, y_val = cached_feature_matrix(datasets["val"])
    x_test, y_test = cached_feature_matrix(datasets["test"])
    matrices = (x_train, y_train, x_val, y_val, x_test, y_test)
    for name in ("random_forest", "lightgbm", "xgboost"):
        results.append(train_classical(name, matrices, args.output, args.seed))
    results.sort(key=lambda row: row["f1_micro"], reverse=True)
    fields = ["model", "subset_accuracy", "precision_micro", "recall_micro", "f1_micro", "smoke_f1", "fire_f1", "smoke_threshold", "fire_threshold", "seconds", "artifact"]
    with (args.output / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
