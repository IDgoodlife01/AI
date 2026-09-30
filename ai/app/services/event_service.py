# AI 이벤트 서비스 - 확정 이벤트 JPG 스냅샷 및 MP4 영상 저장
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_STORAGE = PROJECT_ROOT / "storage" / "events"


class EventStorage:
    def __init__(self, root: Path = EVENT_STORAGE):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, frame, detection: dict) -> dict:
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_dir = self.root / event_id
        event_dir.mkdir(parents=True, exist_ok=False)
        snapshot = event_dir / "snapshot.jpg"
        if not cv2.imwrite(str(snapshot), frame):
            raise RuntimeError("Failed to save event snapshot")
        metadata = {
            "event_id": event_id,
            "snapshot": str(snapshot),
            "video": None,
            "detection": detection,
        }
        (event_dir / "event.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata

    def create(self, frame, frames, fps: float, detection: dict) -> dict:
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_dir = self.root / event_id
        event_dir.mkdir(parents=True, exist_ok=False)
        snapshot = event_dir / "snapshot.jpg"
        video = event_dir / "event.mp4"
        if not cv2.imwrite(str(snapshot), frame):
            raise RuntimeError("Failed to save event snapshot")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open event video")
        try:
            for buffered_frame in frames:
                writer.write(buffered_frame)
        finally:
            writer.release()
        metadata = {"event_id": event_id, "snapshot": str(snapshot), "video": str(video), "detection": detection}
        (event_dir / "event.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
