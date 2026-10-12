"""REST client for the CodeRunner server inside the sandbox, authenticated with the sandbox credential."""

from __future__ import annotations

from typing import Any

import httpx


class SandboxClient:
    def __init__(self, base_url: str, token: str, timeout: float = 330):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, trust_env=False,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def health(self) -> bool:
        try:
            return (await self._client.get("/health", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def execute(self, code: str, session_id: str | None = None) -> dict:
        payload: dict[str, Any] = {"command": code}
        if session_id:
            payload["session_id"] = session_id
        response = await self._client.post("/execute", json=payload)
        data = response.json()
        if response.status_code >= 400 and "stderr" not in data:
            data = {"stdout": "", "stderr": data.get("error", response.text)}
        return data

    async def start_session(self, session_id: str | None = None) -> dict:
        response = await self._client.post("/v1/sessions/session", json={"session_id": session_id} if session_id else {})
        if response.status_code == 409:
            return {"session_id": session_id, "status": "exists"}
        response.raise_for_status()
        return response.json()

    async def list_sessions(self) -> list[dict]:
        response = await self._client.get("/v1/sessions/session")
        response.raise_for_status()
        return response.json().get("sessions", [])

    async def stop_session(self, session_id: str) -> dict:
        response = await self._client.request("DELETE", "/v1/sessions/session", params={"session_id": session_id})
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
