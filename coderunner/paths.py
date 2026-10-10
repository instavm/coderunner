"""Filesystem layout for host-side state.

CODERUNNER_HOME (default ~/.coderunner) is shared with the installer. The
assistant keeps its own state under CODERUNNER_HOME/assistant so it never
collides with other tools that use the same directory.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import stat


def home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CODERUNNER_HOME", pathlib.Path.home() / ".coderunner")).expanduser()


def assistant_dir() -> pathlib.Path:
    return home() / "assistant"


def ensure_dirs() -> pathlib.Path:
    base = assistant_dir()
    for path in (base, base / "logs", base / "run", base / "models", base / "browser-profile"):
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    return base


def config_path() -> pathlib.Path:
    return assistant_dir() / "config.json"


def db_path() -> pathlib.Path:
    return assistant_dir() / "assistant.db"


def audit_path() -> pathlib.Path:
    return assistant_dir() / "audit.jsonl"


def api_token_path() -> pathlib.Path:
    """Sandbox credential, shared with install.sh and passed into the VM."""
    return home() / "api-token"


def admin_token_path() -> pathlib.Path:
    """Owner credential. Never passed into the sandbox."""
    return assistant_dir() / "admin-token"


def _read_or_create_secret(path: pathlib.Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            os.chmod(path, 0o600)
        value = path.read_text().strip()
        if len(value) >= 32 and not any(c.isspace() for c in value):
            return value
    value = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value + "\n")
    return value


def api_token() -> str:
    """CODERUNNER_API_TOKEN wins so operators can keep using their own secret."""
    env = os.environ.get("CODERUNNER_API_TOKEN", "")
    if env:
        return env
    return _read_or_create_secret(api_token_path())


def admin_token() -> str:
    return _read_or_create_secret(admin_token_path())
