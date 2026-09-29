# AI 서버 - Backend AI-01 이벤트 생성 및 AI-02 미디어 등록 클라이언트
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BackendApiError(RuntimeError):
    pass


class BackendClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method=method,
            headers={"Content-Type": "application/json", "X-AI-API-KEY": self.api_key},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise BackendApiError(f"Backend HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise BackendApiError(f"Backend 연결 실패: {error.reason}") from error

    def create_event(self, payload: dict) -> int:
        response = self._request("POST", "/api/ai/events", payload)
        event_id = response.get("eventId")
        if not isinstance(event_id, int):
            raise BackendApiError("AI-01 응답에 정수 eventId가 없습니다.")
        return event_id

    def register_media(self, event_id: int, payload: dict) -> dict:
        return self._request("PATCH", f"/api/ai/events/{event_id}/media", payload)
