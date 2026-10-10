"""Owner secrets, kept on the host and injected at the network boundary.

Secrets never enter the sandbox or the model context. Code in the sandbox
writes a placeholder such as `{{secret:github}}` in a request header and the
gateway substitutes the value only when the destination host is one the owner
allowed for that secret (config `secrets.<name>.hosts`).
"""

from __future__ import annotations

import json
import os
import re
import threading

from . import paths

PLACEHOLDER = re.compile(r"\{\{secret:([A-Za-z0-9_.-]{1,64})\}\}")
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class SecretStore:
    def __init__(self, path=None):
        self.path = path or (paths.assistant_dir() / "secrets.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text() or "{}")

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, self.path)

    def set(self, name: str, value: str) -> None:
        if not NAME.match(name):
            raise ValueError("Secret names may contain letters, digits, '.', '_' and '-'.")
        with self._lock:
            data = self._load()
            data[name] = value
            self._save(data)

    def get(self, name: str) -> str | None:
        with self._lock:
            return self._load().get(name)

    def delete(self, name: str) -> bool:
        with self._lock:
            data = self._load()
            existed = data.pop(name, None) is not None
            self._save(data)
            return existed

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._load())
