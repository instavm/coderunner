"""Run the daemon as a per-user service: a LaunchAgent on macOS, a systemd user unit on Linux.

Every system command goes through an injectable runner so callers (and tests)
can see the exact argv. Nothing here needs root.

macOS privacy permissions (Automation for Calendar/Reminders/Notes/Mail,
Contacts, Full Disk Access for Messages) attach to the "responsible process".
For a LaunchAgent that is the Python binary in ProgramArguments, which has no
app bundle and may never show a consent prompt; requests then fail with
"Not authorized". Grant access first by running `coderunner connectors test
<name>` from Terminal so macOS can ask, and grant the same Python under
Privacy & Security if the agent is still refused. A small signed helper app
that owns these permissions is the robust long-term route.
"""

from __future__ import annotations

import os
import pathlib
import platform
import plistlib
import re
import subprocess
import sys
import time
from typing import Callable

from ..core import paths

LABEL = "io.instavm.coderunner"
UNIT = "coderunner.server.service"
SERVICE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class ServiceError(Exception):
    pass


def run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


def _python() -> str:
    return os.path.abspath(sys.executable)


def _package_root() -> str | None:
    """Source checkout to put on PYTHONPATH when coderunner is not installed into site-packages."""
    root = pathlib.Path(__file__).resolve().parents[2]
    return None if root.name in ("site-packages", "dist-packages") else str(root)


def _environment(path: str | None = None) -> dict[str, str]:
    env = {"PATH": path} if path else {}
    env["PYTHONUNBUFFERED"] = "1"
    if os.environ.get("CODERUNNER_HOME"):
        env["CODERUNNER_HOME"] = os.environ["CODERUNNER_HOME"]
    root = _package_root()
    if root:
        env["PYTHONPATH"] = root
    return env


def _detail(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or "").strip() or f"exit status {result.returncode}"


class LaunchAgent:
    def __init__(self, *, directory: pathlib.Path | str | None = None, runner: Runner = run,
                 python: str | None = None, uid: int | None = None, sleep: Callable[[float], None] = time.sleep):
        self.directory = pathlib.Path(directory or pathlib.Path.home() / "Library" / "LaunchAgents")
        self.runner = runner
        self.python = python or _python()
        self.uid = os.getuid() if uid is None else uid
        self.sleep = sleep

    @property
    def path(self) -> pathlib.Path:
        return self.directory / f"{LABEL}.plist"

    @property
    def domain(self) -> str:
        return f"gui/{self.uid}"

    @property
    def target(self) -> str:
        return f"{self.domain}/{LABEL}"

    def _path(self) -> str:
        # launchd starts agents with a bare PATH; add Homebrew and the venv's bin (model servers, tools).
        bin_dir = os.path.dirname(self.python)
        return SERVICE_PATH if bin_dir in SERVICE_PATH.split(":") else f"{bin_dir}:{SERVICE_PATH}"

    def definition(self) -> dict:
        logs = paths.assistant_dir() / "logs"
        return {
            "Label": LABEL,
            "ProgramArguments": [self.python, "-m", "coderunner", "daemon"],
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},   # restart after crashes, not after a clean stop
            "ProcessType": "Interactive",
            "LimitLoadToSessionType": "Aqua",         # needs the GUI session for Automation prompts
            "ThrottleInterval": 10,
            "Umask": 0o077,
            "WorkingDirectory": str(paths.assistant_dir()),
            "EnvironmentVariables": _environment(self._path()),
            "StandardOutPath": str(logs / "daemon.out.log"),
            "StandardErrorPath": str(logs / "daemon.err.log"),
        }

    def install(self) -> str:
        paths.ensure_dirs()
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(plistlib.dumps(self.definition()))
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.path)
        self.runner(["launchctl", "bootout", self.target])   # not loaded yet is fine
        result = self.runner(["launchctl", "bootstrap", self.domain, str(self.path)])
        if result.returncode != 0:
            # bootout finishes asynchronously; a quick bootstrap can race it.
            self.sleep(1)
            result = self.runner(["launchctl", "bootstrap", self.domain, str(self.path)])
        if result.returncode != 0:
            raise ServiceError(f"launchctl bootstrap failed: {_detail(result)}")
        return f"Installed LaunchAgent {LABEL} ({self.path}). Logs: {paths.assistant_dir() / 'logs'}"

    def uninstall(self) -> str:
        self.runner(["launchctl", "bootout", self.target])
        existed = self.path.exists()
        self.path.unlink(missing_ok=True)
        return f"Removed LaunchAgent {LABEL}." if existed else f"LaunchAgent {LABEL} was not installed."

    def status(self) -> dict:
        info = {"manager": "launchd", "label": LABEL, "path": str(self.path), "installed": self.path.exists(),
                "loaded": False, "running": False, "pid": None}
        result = self.runner(["launchctl", "print", self.target])
        if result.returncode != 0:
            return info
        info["loaded"] = True
        state = re.search(r"^\s*state = (\S+)", result.stdout, re.M)
        pid = re.search(r"^\s*pid = (\d+)", result.stdout, re.M)
        last_exit = re.search(r"^\s*last exit code = (.+)$", result.stdout, re.M)
        info["state"] = state.group(1) if state else "unknown"
        info["running"] = info["state"] == "running"
        info["pid"] = int(pid.group(1)) if pid else None
        if last_exit:
            info["last_exit"] = last_exit.group(1).strip()
        return info

    def restart(self) -> str:
        result = self.runner(["launchctl", "kickstart", "-k", self.target])
        if result.returncode != 0:
            raise ServiceError(f"launchctl kickstart failed (is the service installed?): {_detail(result)}")
        return f"Restarted {LABEL}."


def _quote(value: str) -> str:
    """Quote one word for systemd unit files; % starts a specifier and must be doubled."""
    value = value.replace("%", "%%")
    if re.fullmatch(r"[A-Za-z0-9_./:=@+-]+", value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class SystemdUnit:
    def __init__(self, *, directory: pathlib.Path | str | None = None, runner: Runner = run,
                 python: str | None = None):
        config_home = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config")
        self.directory = pathlib.Path(directory or config_home / "systemd" / "user")
        self.runner = runner
        self.python = python or _python()

    @property
    def path(self) -> pathlib.Path:
        return self.directory / UNIT

    def definition(self) -> str:
        environment = "".join(f"Environment={_quote(f'{key}={value}')}\n"
                              for key, value in _environment().items())
        return (
            "[Unit]\n"
            "Description=CodeRunner assistant daemon\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={' '.join(_quote(part) for part in (self.python, '-m', 'coderunner', 'daemon'))}\n"
            f"WorkingDirectory={_quote(str(paths.assistant_dir()))}\n"
            f"{environment}"
            "UMask=0077\n"
            "Restart=on-failure\n"
            "RestartSec=2\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )

    def _systemctl(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = self.runner(["systemctl", "--user", *args])
        if check and result.returncode != 0:
            raise ServiceError(f"systemctl --user {' '.join(args)} failed: {_detail(result)}")
        return result

    def install(self) -> str:
        paths.ensure_dirs()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.definition())
        self._systemctl("daemon-reload")
        self._systemctl("enable", "--now", UNIT)
        user = os.environ.get("USER") or "$USER"
        return (f"Installed {UNIT} ({self.path}).\nTo keep it running while you are logged out: "
                f"loginctl enable-linger {user}")

    def uninstall(self) -> str:
        self._systemctl("disable", "--now", UNIT, check=False)
        existed = self.path.exists()
        self.path.unlink(missing_ok=True)
        self._systemctl("daemon-reload", check=False)
        return f"Removed {UNIT}." if existed else f"{UNIT} was not installed."

    def status(self) -> dict:
        info = {"manager": "systemd", "unit": UNIT, "path": str(self.path), "installed": self.path.exists(),
                "loaded": False, "running": False, "pid": None}
        result = self._systemctl("show", UNIT, "--property=LoadState,ActiveState,SubState,MainPID", check=False)
        if result.returncode != 0:
            return info
        props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        info["loaded"] = props.get("LoadState") == "loaded"
        info["state"] = f"{props.get('ActiveState', 'unknown')}/{props.get('SubState', 'unknown')}"
        info["running"] = props.get("ActiveState") == "active"
        pid = props.get("MainPID", "0")
        info["pid"] = int(pid) if pid.isdigit() and int(pid) > 0 else None
        return info

    def restart(self) -> str:
        self._systemctl("restart", UNIT)
        return f"Restarted {UNIT}."


def manager(*, system: str | None = None, **kwargs) -> LaunchAgent | SystemdUnit:
    """The service manager for this platform. kwargs: directory, runner, python (and uid on macOS)."""
    system = system or platform.system()
    if system == "Darwin":
        return LaunchAgent(**kwargs)
    if system == "Linux":
        kwargs.pop("uid", None)
        kwargs.pop("sleep", None)
        return SystemdUnit(**kwargs)
    raise ServiceError(f"Running as a service is not supported on {system}; run `coderunner daemon` instead.")


def install(**kwargs) -> str:
    return manager(**kwargs).install()


def uninstall(**kwargs) -> str:
    return manager(**kwargs).uninstall()


def status(**kwargs) -> dict:
    return manager(**kwargs).status()


def restart(**kwargs) -> str:
    return manager(**kwargs).restart()
