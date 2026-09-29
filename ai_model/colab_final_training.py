# AI 최종학습 - Drive 최신 패키지를 결합하고 6종 모델을 GPU 환경에서 학습
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DRIVE_ROOT = Path("/content/gdrive/MyDrive/fire_ai_training")
RAW_DRIVE = DRIVE_ROOT / "raw"
BASE_PACKAGE = RAW_DRIVE / "fire_ai_colab_package.zip"
UPDATE_PART_PATTERN = "fire_ai_colab_update_20260929.zip.part*"
UPDATE_SIZE = 421_644_275
WORKSPACE = Path("/content/fire_ai_latest")
UPDATE_ZIP = Path("/tmp/fire_ai_colab_update_20260929.zip")
RESULTS = DRIVE_ROOT / "results_final_6models"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("실행:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not DRIVE_ROOT.exists():
        raise RuntimeError("Google Drive를 먼저 /content/gdrive에 마운트하세요.")
    parts = sorted(RAW_DRIVE.glob(UPDATE_PART_PATTERN))
    if not BASE_PACKAGE.exists():
        raise FileNotFoundError(f"기본 패키지가 없습니다: {BASE_PACKAGE}")
    if len(parts) != 5:
        raise RuntimeError(f"업데이트 조각은 5개여야 합니다. 현재: {len(parts)}개")

    with UPDATE_ZIP.open("wb") as output:
        for part in parts:
            output.write(part.read_bytes())
    if UPDATE_ZIP.stat().st_size != UPDATE_SIZE:
        raise RuntimeError("업데이트 ZIP 크기가 일치하지 않습니다.")

    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    with zipfile.ZipFile(BASE_PACKAGE) as archive:
        archive.extractall(WORKSPACE)
    with zipfile.ZipFile(UPDATE_ZIP) as archive:
        archive.extractall(WORKSPACE)

    train_script = WORKSPACE / "ai_model/src/train_all.py"
    reports_script = WORKSPACE / "ai_model/src/generate_reports.py"
    visifire = WORKSPACE / "raw/video_validation/visifire"
    if not train_script.exists() or not reports_script.exists():
        raise RuntimeError("최신 ai_model 코드 결합에 실패했습니다.")
    if len(list(visifire.rglob("*.avi"))) != 39:
        raise RuntimeError("VisiFire AVI 39개가 확인되지 않습니다.")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(WORKSPACE / "ai_model/requirements.txt")])
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    run(
        [
            sys.executable,
            str(train_script),
            "--data-root",
            str(WORKSPACE / "raw"),
            "--output",
            str(RESULTS),
            "--epochs",
            "15",
            "--hard-negative-epochs",
            "2",
            "--batch-size",
            "32",
            "--image-size",
            "224",
            "--yolo-image-size",
            "640",
            "--resume",
        ],
        cwd=WORKSPACE,
    )
    run([sys.executable, str(reports_script), "--results", str(RESULTS)], cwd=WORKSPACE)

    models = sorted((RESULTS / "models").glob("*"))
    if len(models) != 6:
        raise RuntimeError(f"최종 모델은 6개여야 합니다. 현재: {len(models)}개")
    print("완료:", RESULTS)
    print("모델:", [model.name for model in models])


if __name__ == "__main__":
    main()
