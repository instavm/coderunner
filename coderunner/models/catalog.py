"""Curated open-weight models with verified artifacts for each runtime."""

from __future__ import annotations

import functools
import json
import pathlib

PATH = pathlib.Path(__file__).with_name("catalog.json")
ARTIFACT = {"ollama": "ollama", "llama.cpp": "gguf", "mlx": "mlx", "prism": "prism"}  # runtime -> artifact key


@functools.lru_cache(maxsize=1)
def _load() -> tuple[dict, ...]:
    return tuple(json.loads(PATH.read_text())["models"])


def models() -> list[dict]:
    return list(_load())


def get(model_id: str) -> dict:
    for entry in _load():
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"Unknown model {model_id!r}; known: {', '.join(e['id'] for e in _load())}")


def by_role(role: str) -> list[dict]:
    return sorted((e for e in _load() if role in e["roles"]), key=lambda e: e["rank"])


def artifact(entry: dict, runtime: str | None) -> dict | None:
    """The artifact a runtime would fetch; GGUF sizes stand in when the runtime is unknown."""
    arts = entry.get("artifacts", {})
    if runtime in ARTIFACT:
        return arts.get(ARTIFACT[runtime])
    return arts.get("gguf") or arts.get("ollama") or arts.get("mlx") or arts.get("prism")


def has_vision(entry: dict, runtime: str | None = None) -> bool:
    art = artifact(entry, runtime) or {}
    return "vision" in entry["roles"] and art.get("vision", True)
