"""Sandbox VM drivers and REST client.

Drivers wrap the platform's isolation tool: Apple `container` on macOS and
Docker elsewhere (optionally with gVisor's runsc). The client talks to the
CodeRunner server inside the sandbox with the sandbox credential.
"""

from __future__ import annotations

import ipaddress
import json
import platform
import shutil
import subprocess
from typing import Any

import httpx

from .config import Config


def _run(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _find_ipv4(node: Any, keys=("gateway",)) -> str | None:
    """Search inspect JSON for a gateway address; formats differ across versions."""
    if isinstance(node, dict):
        for key, value in node.items():
            if any(k in key.lower() for k in keys) and isinstance(value, str):
                candidate = value.split("/")[0]
                try:
                    if ipaddress.ip_address(candidate).version == 4:
                        return candidate
                except ValueError:
                    pass
        for value in node.values():
            found = _find_ipv4(value, keys)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_ipv4(value, keys)
            if found:
                return found
    return None


class Driver:
    name = "none"

    def __init__(self, config: Config):
        self.config = config
        self.container = config.get("sandbox.name", "coderunner")

    def available(self) -> bool:
        return False

    def status(self) -> str:
        return "unavailable"

    def start(self) -> bool:
        return False

    def stop(self) -> bool:
        return False

    def gateway_address(self) -> str:
        """Host address the sandbox uses to reach the gateway."""
        return "127.0.0.1"

    def default_url(self) -> str:
        return "http://127.0.0.1:8222"


class AppleDriver(Driver):
    name = "apple"

    def available(self) -> bool:
        return platform.system() == "Darwin" and shutil.which("container") is not None

    def _inspect(self) -> dict | None:
        result = _run(["container", "inspect", self.container])
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return data[0] if isinstance(data, list) and data else None

    def status(self) -> str:
        info = self._inspect()
        if info is None:
            return "missing"
        return str(info.get("status", "unknown"))

    def start(self) -> bool:
        return _run(["container", "start", self.container], timeout=180).returncode == 0

    def stop(self) -> bool:
        return _run(["container", "stop", self.container], timeout=120).returncode == 0

    def network(self) -> str:
        info = self._inspect() or {}
        networks = (info.get("configuration") or {}).get("networks") or []
        return networks[0].get("network", "default") if networks else "default"

    def gateway_address(self) -> str:
        result = _run(["container", "network", "inspect", self.network()])
        if result.returncode == 0:
            try:
                found = _find_ipv4(json.loads(result.stdout))
            except json.JSONDecodeError:
                found = None
            if found:
                return found
        return "192.168.64.1"

    def default_url(self) -> str:
        if self.config.get("sandbox.network") in ("none", "proxy"):
            return "http://127.0.0.1:8222"
        return "http://coderunner.local:8222"


class DockerDriver(Driver):
    name = "docker"

    def available(self) -> bool:
        return shutil.which("docker") is not None and _run(["docker", "info"], timeout=20).returncode == 0

    def status(self) -> str:
        result = _run(["docker", "inspect", "-f", "{{.State.Status}}", self.container])
        return result.stdout.strip() if result.returncode == 0 else "missing"

    def start(self) -> bool:
        return _run(["docker", "start", self.container], timeout=180).returncode == 0

    def stop(self) -> bool:
        return _run(["docker", "stop", self.container], timeout=120).returncode == 0

    def gateway_address(self) -> str:
        # Docker Desktop (macOS/Windows) routes host.docker.internal to host
        # loopback, so the gateway listens there. On Linux the container reaches
        # the host through its bridge gateway.
        if platform.system() != "Linux":
            return "127.0.0.1"
        result = _run(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.Gateway}} {{end}}",
                       self.container])
        for value in result.stdout.split():
            try:
                if ipaddress.ip_address(value).version == 4:
                    return value
            except ValueError:
                continue
        return "172.17.0.1"


def driver_for(config: Config) -> Driver:
    choice = config.get("sandbox.driver", "auto")
    candidates = {"apple": AppleDriver, "docker": DockerDriver}
    if choice in candidates:
        return candidates[choice](config)
    for cls in (AppleDriver, DockerDriver):
        driver = cls(config)
        if driver.available():
            return driver
    return Driver(config)


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
