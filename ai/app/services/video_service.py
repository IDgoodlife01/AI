# AI video service - frame buffering, inference, and confirmed-event creation
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2

from app.models import predict
from app.services.event_service import EventStorage


def process_video(source: str | int, model: str = "yolo", threshold: float | None = None, buffer_seconds: int = 5):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 20.0
    frames = deque(maxlen=max(1, round(fps * buffer_seconds)))
    storage = EventStorage()
    events = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame.copy())
            encoded, jpeg = cv2.imencode(".jpg", frame)
            if not encoded:
                continue
            result = predict(jpeg.tobytes(), model, threshold)
            if result["detected"]:
                events.append(storage.create(frame, list(frames), fps, result))
                frames.clear()
    finally:
        capture.release()
    return events

