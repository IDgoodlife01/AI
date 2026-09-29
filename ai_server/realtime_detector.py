# AI 서버 - 실시간 영상 프레임 버퍼, 화재·연기 탐지 및 이벤트 영상 저장
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from .model_service import AVAILABLE_MODELS, predict
from .backend_client import BackendApiError, BackendClient


class EventRecorder:
    def __init__(self, output: Path, source: str, model: str, fps: float, post_seconds: float,
                 pre_seconds: float, camera_id: str | None, model_version: str,
                 media_path_prefix: str, backend: BackendClient | None):
        self.output = output
        self.source = source
        self.model = model
        self.fps = fps
        self.post_seconds = post_seconds
        self.pre_seconds = pre_seconds
        self.camera_id = camera_id
        self.model_version = model_version
        self.media_path_prefix = media_path_prefix.rstrip("/")
        self.backend = backend
        self.writer = None
        self.event_dir = None
        self.metadata = None
        self.started_at = None
        self.last_detection_at = None
        self.max_probabilities = {"smoke": 0.0, "fire": 0.0}
        self.detected_labels: set[str] = set()
        self.frames_written = 0
        self.backend_event_id = None
        self.backend_error = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, buffered_frames, frame, result, now: float):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.event_dir = self.output / stamp
        self.event_dir.mkdir(parents=True, exist_ok=False)
        height, width = frame.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.event_dir / "event.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError("이벤트 MP4 파일을 열 수 없습니다.")
        self.metadata = (self.event_dir / "detections.jsonl").open("w", encoding="utf-8")
        self.started_at = now
        self.last_detection_at = now
        self.max_probabilities = {"smoke": 0.0, "fire": 0.0}
        self.detected_labels = set()
        self.frames_written = 0
        self.backend_event_id = None
        self.backend_error = None
        for _captured_at, jpeg in buffered_frames:
            buffered = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            if buffered is not None:
                self.writer.write(buffered)
                self.frames_written += 1
        cv2.imwrite(str(self.event_dir / "first_detection.jpg"), frame)
        self.record_detection(result, now)
        self._create_backend_event(result)

    @staticmethod
    def event_type(labels) -> str:
        values = set(labels)
        if {"fire", "smoke"} <= values:
            return "FIRE_SMOKE"
        if "fire" in values:
            return "FIRE"
        if "smoke" in values:
            return "SMOKE"
        raise ValueError("탐지 클래스가 없어 이벤트 타입을 만들 수 없습니다.")

    def media_path(self, filename: str) -> str:
        return f"{self.media_path_prefix}/{self.event_dir.name}/{filename}"

    def _create_backend_event(self, result):
        if self.backend is None:
            return
        detected = result["detected"]
        payload = {
            "cameraId": self.camera_id,
            "eventType": self.event_type(detected),
            "confidence": max(float(result["probabilities"][name]) for name in detected),
            "detectedAt": datetime.now().isoformat(timespec="seconds"),
            "snapshotPath": self.media_path("first_detection.jpg"),
            "modelVersion": self.model_version,
        }
        try:
            self.backend_event_id = self.backend.create_event(payload)
            print(f"[AI-01 완료] eventId={self.backend_event_id}", flush=True)
        except BackendApiError as error:
            self.backend_error = str(error)
            print(f"[AI-01 실패] {error}", flush=True)

    def write_frame(self, frame):
        self.writer.write(frame)
        self.frames_written += 1

    def record_detection(self, result, now: float):
        if result["detected"]:
            self.last_detection_at = now
        self.detected_labels.update(result["detected"])
        for name, value in result["probabilities"].items():
            self.max_probabilities[name] = max(self.max_probabilities[name], float(value))
        row = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": round(now - self.started_at, 3),
            **result,
        }
        self.metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.metadata.flush()

    def should_close(self, now: float) -> bool:
        return self.active and now - self.last_detection_at >= self.post_seconds

    def close(self):
        if not self.active:
            return
        self.writer.release()
        self.metadata.close()
        summary = {
            "source": self.source,
            "model": self.model,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(self.frames_written / self.fps, 3),
            "frames_written": self.frames_written,
            "detected": sorted(self.detected_labels),
            "max_probabilities": {name: round(value, 6) for name, value in self.max_probabilities.items()},
            "camera_id": self.camera_id,
            "model_version": self.model_version,
            "backend_event_id": self.backend_event_id,
            "backend_error": self.backend_error,
        }
        if self.backend is not None and self.backend_event_id is not None:
            payload = {
                "videoPath": self.media_path("event.mp4"),
                "preSeconds": round(self.pre_seconds),
                "postSeconds": round(self.post_seconds),
            }
            try:
                self.backend.register_media(self.backend_event_id, payload)
                print(f"[AI-02 완료] eventId={self.backend_event_id}", flush=True)
            except BackendApiError as error:
                self.backend_error = str(error)
                summary["backend_error"] = self.backend_error
                print(f"[AI-02 실패] {error}", flush=True)
        (self.event_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[저장 완료] {self.event_dir}", flush=True)
        self.writer = None
        self.metadata = None


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def resize_for_recording(frame, width: int):
    if width <= 0 or frame.shape[1] <= width:
        return frame
    ratio = width / frame.shape[1]
    return cv2.resize(frame, (width, round(frame.shape[0] * ratio)), interpolation=cv2.INTER_AREA)


def draw_result(frame, result, recording: bool):
    probabilities = result.get("probabilities", {})
    text = " | ".join(f"{name}: {value:.3f}" for name, value in probabilities.items())
    detected = ", ".join(result.get("detected", [])) or "none"
    color = (0, 0, 255) if result.get("detected") else (0, 180, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"detected: {detected}", (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 24, 24), 9, (0, 0, 255), -1)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="실시간 화재·연기 탐지 및 이벤트 녹화")
    parser.add_argument("--source", default="0", help="웹캠 번호, 영상 파일 또는 RTSP/HTTP 주소")
    parser.add_argument("--model", choices=sorted(AVAILABLE_MODELS), default="random_forest")
    parser.add_argument("--threshold", type=float, default=None, help="미지정 시 모델의 검증 최적값 사용")
    parser.add_argument("--buffer-seconds", type=float, default=5.0, help="탐지 전 저장 시간")
    parser.add_argument("--post-seconds", type=float, default=5.0, help="마지막 탐지 후 저장 시간")
    parser.add_argument("--infer-every", type=int, default=3, help="AI 추론 프레임 간격")
    parser.add_argument("--record-width", type=int, default=1280, help="저장 영상 최대 너비, 0은 원본")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="프레임 버퍼 JPEG 품질")
    parser.add_argument("--output", type=Path, default=project_root / "events")
    parser.add_argument("--camera-id", default=os.environ.get("AI_CAMERA_ID"), help="Backend CAMERA의 cameraId")
    parser.add_argument("--backend-url", default=os.environ.get("AI_BACKEND_URL"), help="예: http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("AI_API_KEY"), help="미지정 시 AI_API_KEY 환경변수")
    parser.add_argument("--model-version", default=os.environ.get("AI_MODEL_VERSION"), help="Backend에 기록할 모델 버전")
    parser.add_argument("--media-path-prefix", default="/storage/events", help="Backend에 전달할 미디어 경로 접두사")
    parser.add_argument("--display", action="store_true", help="실시간 화면 표시, q로 종료")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.buffer_seconds < 0 or args.post_seconds < 0 or args.infer_every < 1:
        raise ValueError("buffer/post 시간은 0 이상, infer-every는 1 이상이어야 합니다.")
    if bool(args.backend_url) != bool(args.api_key):
        raise ValueError("Backend 연동 시 --backend-url과 --api-key를 함께 지정해야 합니다.")
    if args.backend_url and not args.camera_id:
        raise ValueError("Backend 연동 시 --camera-id가 필요합니다.")
    backend = BackendClient(args.backend_url, args.api_key) if args.backend_url else None
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"영상 입력을 열 수 없습니다: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = source_fps if 1 <= source_fps <= 120 else 20.0
    buffer = deque(maxlen=max(1, round(fps * args.buffer_seconds)))
    recorder = EventRecorder(
        args.output.resolve(), args.source, args.model, fps, args.post_seconds,
        args.buffer_seconds, args.camera_id, args.model_version or args.model,
        args.media_path_prefix, backend,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    result = {"model": args.model, "thresholds": {}, "probabilities": {}, "detected": []}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.time()
            record_frame = resize_for_recording(frame, args.record_width)
            encoded, jpeg = cv2.imencode(".jpg", record_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not encoded:
                raise RuntimeError("프레임 버퍼 JPEG 변환 실패")
            buffer.append((now, jpeg))
            inferred = frame_index % args.infer_every == 0
            if inferred:
                result = predict(jpeg.tobytes(), args.model, args.threshold)
            detected = bool(result["detected"]) if inferred else False
            just_started = False
            if detected and not recorder.active:
                recorder.start(buffer, record_frame, result, now)
                just_started = True
                print(f"[탐지 시작] {result['detected']} {result['probabilities']}", flush=True)
            elif inferred and recorder.active:
                recorder.record_detection(result, now)
            if recorder.active and not just_started:
                recorder.write_frame(record_frame)
            if recorder.should_close(now):
                recorder.close()
            if args.display:
                preview = record_frame.copy()
                draw_result(preview, result, recorder.active)
                cv2.imshow("Fire and Smoke Detection", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
