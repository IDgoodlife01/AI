# AI 모델 개선 - 6종 모델 추가 학습 및 검증 성능이 가장 좋은 모델 선택
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from train_all import (build_deep_model, cached_feature_matrix, make_classical,
                       metrics, optimize_thresholds, predict_deep)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Images(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images, self.labels = images, labels
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.tensor(np.array(self.images[index], copy=True)).permute(2, 0, 1).float() / 255
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(np.array(self.labels[index], copy=True), dtype=torch.float32)


def validation(y, p):
    thresholds = optimize_thresholds(y, p)
    return metrics(y, p, thresholds)["f1_micro"], thresholds


def probabilities(model, features):
    return np.column_stack([
        values[:, estimator.classes_.tolist().index(1)] if 1 in estimator.classes_
        else np.full(len(features), float(estimator.classes_[0] == 1))
        for estimator, values in zip(model.estimators_, model.predict_proba(features))
    ])


def refine_deep(name, source, output, datasets, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(source / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
    model = build_deep_model(name).to(device)
    model.load_state_dict(saved["state_dict"])
    loaders = {key: DataLoader(value, batch_size=args.batch_size) for key, value in datasets.items()}
    yt, yp = predict_deep(model, loaders["val"], device)
    best, thresholds = validation(yt, yp)
    initial_score = best
    saved.update(thresholds=thresholds.tolist(), val_f1=best)
    artifact = output / "models" / f"{name}.pt"
    torch.save(saved, artifact)
    train_y, train_p = predict_deep(model, loaders["train"], device)
    hard = np.any((train_y == 0) & (train_p >= thresholds), axis=1)
    weights = np.where(hard, 3.0, 1.0)
    augmented = Images(datasets["train"].images, datasets["train"].labels, augment=True)
    sampler = WeightedRandomSampler(torch.tensor(weights), len(weights), replacement=True)
    train_loader = DataLoader(augmented, batch_size=args.batch_size, sampler=sampler)
    positives = train_y.sum(0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(train_y) - positives) / np.maximum(positives, 1), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    stale, history = 0, []
    print(f"[{name}] baseline_val_f1={best:.5f} hard_examples={hard.sum()}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if batch % 100 == 0:
                print(f"[{name}] epoch={epoch+1} batch={batch+1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        yt, yp = predict_deep(model, loaders["val"], device)
        score, thresholds = validation(yt, yp)
        history.append({"epoch": epoch + 1, "val_f1": score, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step(score)
        print(f"[{name}] epoch={epoch+1}/{args.epochs} val_f1={score:.5f} best={max(best, score):.5f}", flush=True)
        stale += 1
        if score > best:
            best, stale = score, 0
            saved.update(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         thresholds=thresholds.tolist(), val_f1=score, optimization_epoch=epoch+1)
            torch.save(saved, artifact)
        if stale >= args.patience:
            break
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model, saved, {"baseline_val_f1": initial_score, "selected_val_f1": best,
                          "hard_examples": int(hard.sum()), "history": history}


def refine_tree(name, source, output, matrices, size):
    x, y = matrices["train"]
    xv, yv = matrices["val"]
    baseline = joblib.load(source / "models" / f"{name}.joblib")
    best_model = baseline["model"]
    best, thresholds = validation(yv, probabilities(best_model, xv))
    baseline_score = best
    hard = np.any((y == 0) & (probabilities(best_model, x) >= thresholds), axis=1)
    weights = np.where(hard, 3.0, 1.0)
    settings = {
        "random_forest": [dict(n_estimators=400, min_samples_leaf=1, max_features="sqrt"),
                          dict(n_estimators=400, min_samples_leaf=2, max_features=0.7)],
        "lightgbm": [dict(n_estimators=500, num_leaves=31, learning_rate=0.03, reg_lambda=1),
                     dict(n_estimators=700, num_leaves=63, learning_rate=0.03, reg_lambda=3)],
        "xgboost": [dict(n_estimators=500, max_depth=4, learning_rate=0.03, reg_lambda=2),
                    dict(n_estimators=700, max_depth=6, learning_rate=0.03, reg_lambda=5)],
    }[name]
    trials = []
    for params in settings:
        candidate = make_classical(name, 42)
        candidate.estimator.set_params(**params)
        candidate.fit(x, y, sample_weight=weights)
        score, cutoffs = validation(yv, probabilities(candidate, xv))
        trials.append({"params": params, "val_f1": score})
        print(f"[{name}] candidate_val_f1={score:.5f} baseline={baseline_score:.5f}", flush=True)
        if score > best:
            best, best_model, thresholds = score, candidate, cutoffs
    baseline.update(model=best_model, thresholds=thresholds.tolist(), image_size=size,
                    val_f1=best, hard_negatives=int(hard.sum()))
    joblib.dump(baseline, output / "models" / f"{name}.joblib")
    return best_model, baseline, {"baseline_val_f1": baseline_score, "selected_val_f1": best,
                                  "hard_examples": int(hard.sum()), "trials": trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "ai_model" / "results_v2")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ai_model" / "results_optimized")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Use a separate output directory to preserve baseline models")
    deep_names = ("cnn", "mobilenet_v2")
    tree_names = ("random_forest", "lightgbm", "xgboost")
    names = (*deep_names, "yolo", *tree_names)
    if not (args.source / "metrics.json").is_file():
        raise RuntimeError("Baseline training has not completed")
    for name in names:
        suffix = '.pt' if name in (*deep_names, "yolo") else '.joblib'
        if not (args.source / "models" / (name + suffix)).is_file():
            raise FileNotFoundError(name)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    (args.output / "models").mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source / "models/cnn.pt", map_location="cpu", weights_only=True)
    size = int(checkpoint["image_size"])
    datasets = {key: Images(np.load(args.source / f"cache/{key}_{size}_images.npy", mmap_mode="r"),
                            np.load(args.source / f"cache/{key}_{size}_labels.npy", mmap_mode="r"))
                for key in ("train", "val", "test")}
    audit, results = {}, []
    for name in deep_names:
        model, bundle, audit[name] = refine_deep(name, args.source, args.output, datasets, args)
        y, p = predict_deep(model, DataLoader(datasets["test"], batch_size=args.batch_size), next(model.parameters()).device)
        result = metrics(y, p, np.array(bundle["thresholds"]))
        result.update(model=name, artifact=str(args.output / "models" / f"{name}.pt"))
        results.append(result)
        (args.output / "optimization_progress.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    shutil.copy2(args.source / "models/yolo.pt", args.output / "models/yolo.pt")
    baseline_results = json.loads((args.source / "metrics.json").read_text(encoding="utf-8"))
    yolo_result = next(row.copy() for row in baseline_results if row["model"] == "yolo")
    yolo_result["artifact"] = str(args.output / "models/yolo.pt")
    results.append(yolo_result)
    audit["yolo"] = {"status": "preserved", "reason": "detector thresholds stay validation-only"}
    matrices = {key: cached_feature_matrix(value) for key, value in datasets.items()}
    for name in tree_names:
        model, bundle, audit[name] = refine_tree(name, args.source, args.output, matrices, size)
        x, y = matrices["test"]
        result = metrics(y, probabilities(model, x), np.array(bundle["thresholds"]))
        result.update(model=name, artifact=str(args.output / "models" / f"{name}.joblib"))
        results.append(result)
        (args.output / "optimization_progress.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    results.sort(key=lambda row: row["f1_micro"], reverse=True)
    with (args.output / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(args.source / "dataset_manifest.csv", args.output / "dataset_manifest.csv")
    # Only copy the test cache needed for reports, not the large training cache.
    (args.output / "cache").mkdir(exist_ok=True)
    for kind in ("images", "labels"):
        shutil.copy2(args.source / f"cache/test_{size}_{kind}.npy", args.output / f"cache/test_{size}_{kind}.npy")
    subprocess.run([sys.executable, str(Path(__file__).with_name("generate_reports.py")),
                    "--results", str(args.output)], check=True)
    print("OPTIMIZATION_AND_REPORTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
