# AI 모델 시험 - 작은 임시 데이터로 6종 추가 학습과 예측 기능 검사
import argparse
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from optimize_all import Images, refine_deep, refine_tree
from train_all import build_deep_model, cached_feature_matrix, make_classical
from predict import predict_classical
from PIL import Image


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    root = Path(tempfile.mkdtemp(prefix="fire_optimization_test_"))
    source, output = root / "source", root / "output"
    (source / "models").mkdir(parents=True)
    (output / "models").mkdir(parents=True)
    labels = np.tile(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8), (4, 1))
    datasets = {split: Images(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8), labels)
                for split in ("train", "val", "test")}
    args = argparse.Namespace(batch_size=8, epochs=1, patience=2)
    for name in ("cnn", "mobilenet_v2"):
        torch.save({"state_dict": build_deep_model(name).state_dict(), "model": name,
                    "image_size": 32, "labels": ("smoke", "fire")}, source / "models" / f"{name}.pt")
        model, bundle, audit = refine_deep(name, source, output, datasets, args)
        assert audit["selected_val_f1"] >= audit["baseline_val_f1"]
        assert len(bundle["thresholds"]) == 2
    matrices = {key: cached_feature_matrix(data) for key, data in datasets.items()}
    for name in ("random_forest", "lightgbm", "xgboost"):
        model = make_classical(name, 42)
        model.estimator.set_params(n_estimators=5, n_jobs=2)
        model.fit(*matrices["train"])
        joblib.dump({"model": model}, source / "models" / f"{name}.joblib")
        model, bundle, audit = refine_tree(name, source, output, matrices, 32)
        assert audit["selected_val_f1"] >= audit["baseline_val_f1"]
        assert bundle["image_size"] == 32
        image_path = root / "input.png"
        Image.fromarray(datasets["test"].images[0]).save(image_path)
        p, thresholds = predict_classical(output / "models" / f"{name}.joblib", image_path)
        assert p.shape == thresholds.shape == (2,)
        assert np.isfinite(p).all()
    print("PASS: all six refinement paths, baseline retention, serialization, inference size metadata")
    print(f"Isolated test artifacts: {root}")


if __name__ == "__main__":
    main()
