"""REST client for the CodeRunner server inside the sandbox, authenticated with the sandbox credential."""

from __future__ import annotations

from typing import Any

import httpx


EXEC_TIMEOUT = 600          # the sandbox's own limit for a run (CODERUNNER_EXEC_TIMEOUT in the image)


class ShellUnsupported(Exception):
    """The sandbox image predates the shell API."""


class SandboxClient:
    # Waits a little longer than a run may take: at 330 s, a run the sandbox allowed 600 s for ended in a read
    # timeout, which the assistant reported as the sandbox not running.
    def __init__(self, base_url: str, token: str, timeout: float = EXEC_TIMEOUT + 30):
        self.base_url = base_url.rstrip("/")
        self.token = token
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

    async def shell(self, command: str, session_id: str = "main", timeout: float = 120,
                    background: bool = False) -> dict:
        """A command in the session's persistent bash (or started as a job)."""
        response = await self._client.post("/v1/shell", json={
            "command": command, "session_id": session_id, "timeout": timeout, "background": background},
            timeout=timeout + 30)
        if response.status_code == 404:
            raise ShellUnsupported()
        response.raise_for_status()
        return response.json()

    async def job(self, job: str, stop: bool = False, tail: int = 20000) -> dict:
        if stop:
            response = await self._client.request("DELETE", f"/v1/shell/jobs/{job}")
        else:
            response = await self._client.get(f"/v1/shell/jobs/{job}", params={"tail": tail})
        if response.status_code == 404:
            return {"error": response.json().get("error", f"No job {job}.")}
        response.raise_for_status()
        return response.json()

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

    async def desktop(self, start: bool = False, chrome: bool = False) -> dict:
        """The VM's screen: whether it has one, whether it runs, and its VNC port and password."""
        if start:
            response = await self._client.post("/v1/desktop", json={"chrome": chrome}, timeout=90)
        else:
            response = await self._client.get("/v1/desktop", timeout=10)
        if response.status_code == 404:
            return {"available": False, "running": False, "error": "This sandbox predates the desktop."}
        data = response.json()
        if response.status_code >= 400:
            data.setdefault("available", response.status_code != 409)
            data.setdefault("running", False)
        return data

    def terminal_url(self) -> str:
        """The VM's terminal websocket (the owner's PTY); opened with the same credential."""
        return self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/v1/terminal"

    async def aclose(self) -> None:
        await self._client.aclose()
