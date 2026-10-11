"""JSON configuration with defaults.

The file is small and human-editable. Unknown keys are preserved so newer
versions can add settings without breaking older ones.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from . import paths

DEFAULTS: dict[str, Any] = {
    "owner": {"name": "", "timezone": "", "desktop_notifications": True},
    "sandbox": {
        "driver": "auto",            # auto | apple | docker
        "name": "coderunner",
        "url": "",                   # MCP/REST base, e.g. http://coderunner.local:8222
        "network": "default",        # default | none | proxy
        "memory_gb": 4,              # VM memory, subtracted from the model budget
    },
    "gateway": {
        "sandbox_bind": "auto",      # host address the sandbox can reach, or "auto"
        "sandbox_port": 8223,
        "admin_bind": "127.0.0.1",
        "admin_port": 8224,
        "proxy_port": 8225,
    },
    "models": {
        "runtime": "auto",           # auto | ollama | llama.cpp | mlx | openai
        "base_url": "",              # OpenAI-compatible endpoint when runtime is external
        "api_key": "",               # only for external endpoints that need one
        "chat": "",                  # model id served for the assistant
        "vision": False,             # set by `coderunner setup` after the image check passes
        "embedding": "",             # optional embedding model id for memory
        "guard": "",                 # decision model screening untrusted text; empty: models.decision
        "decision": "",              # decision model judging committing browser steps; empty: the chat model
        "decision_api": "",          # systemone (Ollama decision models) | chat; empty: chosen automatically
        "decision_threshold": 0.7,   # P(commits or unclear) at which a step needs approval
        "context": 32768,
        "allow_licenses": ["apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"],
        "accepted_licenses": [],     # licenses the owner explicitly accepted
    },
    "files": {
        "roots": [],                 # folders the owner shares with the assistant (coderunner files add)
        "embedding": "",             # multimodal embedding model; empty: models.embedding
        "media": True,               # also index images, audio and video when the model supports them
        "max_file_mb": 200,
        "scan_interval": 600,        # seconds between checks for new or changed files
    },
    "agent": {"max_steps": 20, "tool_output_chars": 12000, "approval_timeout": 300,
              "recall_min_similarity": 0.5},   # embedding similarity for memories to enter a task's prompt
    "policy": {
        "rules": [],                 # [{"kind": "browser.navigate", "target": "*.example.com", "decision": "allow"}]
    },
    "egress": {
        "allow": [
            "pypi.org", "files.pythonhosted.org", "registry.npmjs.org",
            "github.com", "codeload.github.com", "objects.githubusercontent.com",
            "raw.githubusercontent.com", "huggingface.co", "*.hf.co",
        ],
        "allow_private": [],
        "ports": [80, 443],
        "approval_wait": 60,
    },
    "secrets": {},                   # name -> {"hosts": ["api.github.com"], "header": "Authorization"}
    "browser": {
        "executable": "",            # empty = detect Chrome, Brave, Edge or Chromium
        "headless": False,
        "allow": [],                 # domains the agent may open without asking
        "block": ["*.local", "localhost"],
        "sensitive_hosts": [],       # extra sites where every click or keystroke needs one-time approval
    },
    "connectors": {"enabled": []},   # e.g. ["calendar", "reminders"]
    "channels": {},                  # per-channel settings, see coderunner/channels
    "remote": {"provider": "", "owners": []},
}


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    def __init__(self, data: dict | None = None, path=None):
        self.path = path or paths.config_path()
        self.data = _merge(DEFAULTS, data or {})

    @classmethod
    def load(cls, path=None) -> "Config":
        path = path or paths.config_path()
        data = {}
        if path.exists():
            data = json.loads(path.read_text() or "{}")
        return cls(data, path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, self.path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
