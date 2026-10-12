"""Append-only audit trail of everything the assistant did or asked to do."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from . import paths

_SENSITIVE = ("token", "secret", "password", "authorization", "cookie", "api_key")


def _scrub(value: Any, key: str = "") -> Any:
    if any(word in key.lower() for word in _SENSITIVE):
        return "[redacted]"
    if isinstance(value, dict):
        return {k: _scrub(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…"
    return value


class AuditLog:
    def __init__(self, path=None):
        self.path = path or paths.audit_path()
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> dict:
        record = {"ts": time.time(), "event": event, **_scrub(fields)}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as handle:
                handle.write(line + "\n")
        return record

    def tail(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text().splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
