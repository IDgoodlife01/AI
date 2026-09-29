# AI 모델 - 정제·전이학습·Hard Negative·앙상블을 적용한 최종 모델 학습
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from train_all import (
    LABEL_NAMES,
    Sample,
    array_features,
    discover_aihub,
    discover_dfire,
    metrics,
    optimize_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FinalDataset(Dataset):
    def __init__(self, rows: list[Sample], size: int, augment: bool):
        self.rows = rows
        steps = []
        if augment:
            steps.extend([
                transforms.RandomResizedCrop(size, scale=(0.72, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(0.22, 0.22, 0.15, 0.04),
            ])
        else:
            steps.append(transforms.Resize((size, size)))
        steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row.image) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(row.label, dtype=torch.float32), index


def digest(path: Path) -> str:
    value = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean_samples(samples: list[Sample], audit_path: Path):
    by_split = {key: [] for key in ("train", "val", "test")}
    removed, seen = [], {}
    # Test is immutable. Its hashes are registered first to prevent leakage from train/val.
    ordered = [row for row in samples if row.split == "test"]
    ordered += [row for row in samples if row.split == "val"]
    ordered += [row for row in samples if row.split == "train"]
    for number, row in enumerate(ordered, 1):
        try:
            with Image.open(row.image) as image:
                image.verify()
            key = digest(row.image)
        except (OSError, UnidentifiedImageError) as error:
            if row.split == "test":
                raise RuntimeError(f"Test image is unreadable: {row.image}") from error
            removed.append({"image": str(row.image), "reason": "unreadable"})
            continue
        previous = seen.get(key)
        if previous and row.split != "test":
            reason = "conflicting_duplicate" if previous.label != row.label else "exact_duplicate"
            removed.append({"image": str(row.image), "reason": reason, "kept": str(previous.image)})
            continue
        if previous is None:
            seen[key] = row
        by_split[row.split].append(row)
        if number % 2000 == 0:
            print(f"[audit] {number}/{len(ordered)}", flush=True)
    audit = {
        "original": {key: sum(row.split == key for row in samples) for key in by_split},
        "cleaned": {key: len(value) for key, value in by_split.items()},
        "removed_count": len(removed),
        "removed": removed,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return by_split, audit


def build_model(pretrained: bool):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    truths, probabilities = [], []
    for images, labels, _ in loader:
        probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        truths.append(labels.numpy())
    return np.concatenate(truths).astype(np.int8), np.concatenate(probabilities)


def rf_probabilities(bundle, rows: list[Sample]):
    features = np.stack([array_features(np.asarray(Image.open(row.image).convert("RGB").resize((64, 64)), dtype=np.uint8)) for row in rows])
    columns = []
    for estimator, values in zip(bundle["model"].estimators_, bundle["model"].predict_proba(features)):
        classes = estimator.classes_.tolist()
        columns.append(values[:, classes.index(1)] if 1 in classes else np.zeros(len(rows)))
    return np.column_stack(columns)


def find_hard_negative_weights(rows: list[Sample], rf_bundle) -> tuple[np.ndarray, int]:
    weights = np.ones(len(rows), dtype=np.float64)
    negatives = [index for index, row in enumerate(rows) if sum(row.label) == 0]
    if not negatives:
        return weights, 0
    probabilities = rf_probabilities(rf_bundle, [rows[index] for index in negatives])
    thresholds = np.asarray(rf_bundle.get("thresholds", [0.5, 0.5]))
    hard = np.any(probabilities >= thresholds.reshape(1, -1), axis=1)
    hard_indices = np.asarray(negatives)[hard]
    weights[hard_indices] = 3.0
    return weights, int(len(hard_indices))


def set_trainable(model, fine_tune: bool):
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if fine_tune:
        for block in model.features[-5:].parameters():
            block.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def run_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch, (images, labels, _) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += float(loss)
        if batch % 100 == 0:
            print(f"[train] batch={batch}/{len(loader)} loss={total / batch:.4f}", flush=True)
    return total / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ai_model" / "results_final")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    samples = discover_dfire(PROJECT_ROOT / "raw") + discover_aihub(PROJECT_ROOT / "raw")
    splits, audit = clean_samples(samples, args.output / "data_cleaning_audit.json")
    print(f"[clean] {audit['cleaned']} removed={audit['removed_count']}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf_path = PROJECT_ROOT / "ai_model" / "results_optimized" / "models" / "random_forest.joblib"
    rf_bundle = joblib.load(rf_path)
    sample_weights, hard_count = find_hard_negative_weights(splits["train"], rf_bundle)
    print(f"[hard-negative] {hard_count}", flush=True)
    train_dataset = FinalDataset(splits["train"], args.image_size, True)
    val_dataset = FinalDataset(splits["val"], args.image_size, False)
    test_dataset = FinalDataset(splits["test"], args.image_size, False)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = args.output / "models" / "mobilenet_v2.pt"
    model = build_model(pretrained=True).to(device)
    labels = np.asarray([row.label for row in splits["train"]], dtype=np.float32)
    positives = labels.sum(axis=0)
    pos_weight = torch.tensor((len(labels) - positives) / np.maximum(positives, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_f1, best_thresholds = -1.0, np.asarray([0.5, 0.5])
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(previous["state_dict"])
        best_f1 = float(previous.get("val_f1", -1.0))
        best_thresholds = np.asarray(previous.get("thresholds", [0.5, 0.5]), dtype=np.float32)
        print(f"[resume] restored validation_f1={best_f1:.5f}", flush=True)
    stages = [(False, args.head_epochs, 1e-3), (True, args.fine_epochs, 1e-4)]
    started = time.time()
    for fine_tune, epochs, learning_rate in stages:
        set_trainable(model, fine_tune)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        for epoch in range(epochs):
            loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
            y_val, p_val = predict(model, val_loader, device)
            thresholds = optimize_thresholds(y_val, p_val)
            score = metrics(y_val, p_val, thresholds)["f1_micro"]
            print(f"[validation] stage={'fine' if fine_tune else 'head'} epoch={epoch + 1}/{epochs} loss={loss:.4f} f1={score:.5f}", flush=True)
            if score > best_f1:
                best_f1, best_thresholds = score, thresholds
                torch.save({"model": "mobilenet_v2", "state_dict": model.state_dict(), "labels": LABEL_NAMES,
                            "image_size": args.image_size, "val_f1": score, "thresholds": thresholds.tolist(),
                            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                            "pretrained": "IMAGENET1K_V2", "hard_negatives": hard_count}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Ensemble weight and thresholds are selected using validation only.
    y_val, mobile_val = predict(model, val_loader, device)
    rf_val = rf_probabilities(rf_bundle, splits["val"])
    best = (-1.0, 1.0, best_thresholds)
    for alpha in np.linspace(0.0, 1.0, 21):
        probability = alpha * mobile_val + (1 - alpha) * rf_val
        thresholds = optimize_thresholds(y_val, probability)
        score = metrics(y_val, probability, thresholds)["f1_micro"]
        if score > best[0]:
            best = (score, float(alpha), thresholds)
    # Independent test is read exactly once, after all choices are fixed.
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    y_test, mobile_test = predict(model, test_loader, device)
    rf_test = rf_probabilities(rf_bundle, splits["test"])
    final_probability = best[1] * mobile_test + (1 - best[1]) * rf_test
    result = metrics(y_test, final_probability, best[2])
    result.update({"model": "mobilenet_v2_rf_ensemble", "validation_f1": best[0], "mobilenet_weight": best[1],
                   "thresholds": best[2].tolist(), "test_count": len(splits["test"]),
                   "seconds": round(time.time() - started, 2)})
    (args.output / "ensemble.json").write_text(json.dumps({"mobilenet": str(checkpoint), "random_forest": str(rf_path),
                                                            "weight": best[1], "thresholds": best[2].tolist()}, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
