"""Curated open-weight models and unified registry over curated + Hugging Face discovered entries."""

from __future__ import annotations

import functools
import json
import pathlib

from ..core import paths

PATH = pathlib.Path(__file__).with_name("catalog.json")
ARTIFACT = {"ollama": "ollama", "llama.cpp": "gguf", "mlx": "mlx", "prism": "prism",
            "driver": "gguf", "judge": "gguf", "voice": "gguf"}  # runtime -> artifact key


@functools.lru_cache(maxsize=1)
def _load() -> tuple[dict, ...]:
    return tuple(json.loads(PATH.read_text())["models"])


def models() -> list[dict]:
    return list(_load())


def discovered_raw() -> list[dict]:
    """Cached raw discovery records from assistant_dir()/models/discovered.json (best-effort, never raises)."""
    path = paths.assistant_dir() / "models" / "discovered.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def lookup(model_id: str, extra: list[dict] | None = None, hw=None, config=None, runtime: str | None = None) -> dict:
    """Unified ModelRegistry lookup across curated catalog, caller-provided `extra`, and cached HF discovery."""
    for entry in extra or ():
        if entry.get("id") == model_id:
            return entry
    try:
        return get(model_id)
    except KeyError:
        pass
    for raw in discovered_raw():
        if raw.get("id") == model_id:
            if hw is not None and config is not None:
                from . import discover
                built = discover.best_build(raw, hw, config, runtime)
                if built:
                    return built
            # Construct minimal catalog-shaped entry from cached raw discovery record so downloads/bench never crash.
            quants = raw.get("quants") or {}
            arts: dict = {}
            if quants and raw.get("gguf_repo"):
                qname, qbuild = next(iter(quants.items()))
                art = {"repo": raw["gguf_repo"], "quant": qname, "file": qbuild.get("file"),
                       "size_gb": round(float(qbuild.get("size_gb") or 0), 2),
                       "mmproj_gb": round(float((raw.get("mmproj") or {}).get("size_gb") or 0), 2)}
                arts["gguf"] = art
                arts["ollama"] = {**art, "tag": f"hf.co/{raw['gguf_repo']}:{qname}"}
            if raw.get("mlx"):
                arts["mlx"] = {"repo": raw["mlx"]["repo"], "size_gb": raw["mlx"]["size_gb"]}
            return {**{k: v for k, v in raw.items() if k not in ("quants", "mmproj", "mlx")},
                    "artifacts": arts, "rank": raw.get("rank", 99)}
    raise KeyError(f"Unknown model {model_id!r}; known: {', '.join(e['id'] for e in _load())}")


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


def needs_mmproj(entry: dict, runtime: str | None = None) -> bool:
    """Whether the GGUF pull must fetch the multimodal projector (vision, or voice codec/encoder)."""
    art = artifact(entry, runtime) or {}
    return has_vision(entry, runtime) or bool(art.get("mmproj_required"))
