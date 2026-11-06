"""Drivers for the platform's isolation tool: Apple `container` on macOS, Docker elsewhere (optionally with
gVisor's runsc). Each starts, stops and locates the sandbox VM."""

from __future__ import annotations

import ipaddress
import json
import platform
import shutil
import subprocess
from typing import Any

from ..core.config import Config


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

    def delete(self) -> bool:
        """Remove the sandbox and its filesystem; the installer creates it again."""
        return False

    def exec_argv(self, command: list[str], tty: bool) -> list[str] | None:
        """The command line that runs `command` inside the sandbox, or None without one."""
        return None

    def logs_argv(self, follow: bool) -> list[str] | None:
        return None

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
            # With Apple's container service stopped (as after the Mac restarts) nothing can be inspected:
            # the sandbox then looked missing, and `coderunner up` told the owner to run the installer.
            return "missing" if self._service_running() else "service-stopped"
        status = info.get("status", "unknown")
        # container 1.5 reports {"state": "running", "networks": [...]}, earlier versions a plain string.
        return str(status.get("state", "unknown") if isinstance(status, dict) else status)

    def _service_running(self) -> bool:
        return _run(["container", "system", "status"], timeout=20).returncode == 0

    def _ensure_service(self) -> bool:
        return self._service_running() or _run(
            ["container", "system", "start", "--disable-kernel-install", "--timeout", "60"],
            timeout=90).returncode == 0

    def start(self) -> bool:
        if not self._ensure_service():
            return False
        return _run(["container", "start", self.container], timeout=180).returncode == 0

    def stop(self) -> bool:
        return _run(["container", "stop", self.container], timeout=120).returncode == 0

    def delete(self) -> bool:
        # With the service stopped (after a restart) nothing can be deleted, nor told apart from missing.
        if not self._ensure_service():
            return False
        if _run(["container", "inspect", self.container]).returncode != 0:
            return True
        return _run(["container", "delete", "--force", self.container], timeout=120).returncode == 0

    def exec_argv(self, command: list[str], tty: bool) -> list[str]:
        return ["container", "exec", "-it" if tty else "-i", self.container, *command]

    def logs_argv(self, follow: bool) -> list[str]:
        return ["container", "logs", *(["--follow"] if follow else []), self.container]

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

    def delete(self) -> bool:
        return _run(["docker", "rm", "--force", self.container], timeout=120).returncode == 0

    def exec_argv(self, command: list[str], tty: bool) -> list[str]:
        return ["docker", "exec", "-it" if tty else "-i", self.container, *command]

    def logs_argv(self, follow: bool) -> list[str]:
        return ["docker", "logs", *(["--follow"] if follow else []), self.container]

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
