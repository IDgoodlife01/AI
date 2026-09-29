# AI 학습 데이터 - AI Hub 3종과 D-Fire 데이터 소스 및 로컬 반영 상태 확인
from __future__ import annotations

import argparse
from pathlib import Path


SOURCES = {
    "71751": ("fire_video_location", "CCTV 영상 객체탐지 강화"),
    "176": ("fire_prediction_scenes", "유사·무관 장면 Hard Negative"),
    "71472": ("fire_smoke_boxes", "불꽃·연기 YOLO 바운딩박스"),
    "dfire": ("dfire", "기존 YOLO 이미지·라벨 21K+"),
    "dfire_videos": ("dfire_surveillance_videos", "실시간 CCTV 추론 시험용 MP4"),
    "mivia_fire_detection": ("video_validation/mivia", "외부 영상 최종 검증 전용(학습 제외)"),
}

DFIRE_VIDEO_URL = "https://1drv.ms/f/c/c0bd25b6b048b01d/EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ"
MIVIA_URL = "https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/"


def count_files(path: Path, suffixes: set[str]) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="화재 AI 데이터 소스와 로컬 반영 상태 확인")
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parents[2] / "raw")
    args = parser.parse_args()

    print("데이터 소스")
    for key, (name, purpose) in SOURCES.items():
        print(f"- {key}: {name} / {purpose}")

    checks = {
        "dfire": args.raw / "dfire",
        "aihub_71472_clean": args.raw / "aihub_clean",
        "aihub_71751": args.raw / "aihub_71751",
        "aihub_176": args.raw / "aihub_176",
    }
    print("\n로컬 상태")
    for name, path in checks.items():
        images = count_files(path, {".jpg", ".jpeg", ".png"}) if path.exists() else 0
        labels = count_files(path, {".txt", ".json"}) if path.exists() else 0
        print(f"- {name}: images={images:,}, labels={labels:,}, path={path}")

    video_path = args.raw / "videos"
    videos = count_files(video_path, {".mp4", ".avi", ".mov", ".mkv"}) if video_path.exists() else 0
    print(f"- videos: files={videos:,}, path={video_path}")
    if videos == 0:
        print(f"  공개 D-Fire 영상: {DFIRE_VIDEO_URL}")

    mivia_root = args.raw / "video_validation" / "mivia"
    (mivia_root / "fire").mkdir(parents=True, exist_ok=True)
    (mivia_root / "non_fire").mkdir(parents=True, exist_ok=True)
    fire_videos = count_files(mivia_root / "fire", {".mp4", ".avi", ".mov", ".mkv"})
    non_fire_videos = count_files(mivia_root / "non_fire", {".mp4", ".avi", ".mov", ".mkv"})
    print(f"- mivia_validation: fire={fire_videos:,}, non_fire={non_fire_videos:,}, path={mivia_root}")
    if fire_videos + non_fire_videos == 0:
        print(f"  등록 사용자 다운로드 필요: {MIVIA_URL}")


if __name__ == "__main__":
    main()
