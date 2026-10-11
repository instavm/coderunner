"""Detect, choose, fetch, serve and verify a local model.

`plan` explains what would happen, `setup` does it and only writes the config
once the served model passes the tool-calling smoke test, and `endpoint` /
`ensure_running` are what the daemon uses afterwards.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
from typing import Callable

from ..llm import LLM
from . import catalog, fit, hardware, licenses, smoketest
from .hub import Hub
from .runtimes import OllamaRuntime, PrismRuntime, Runtime, choose_runtime, install_hint

MAX_FALLBACKS = 2
LOCAL = ("ollama", "llama.cpp", "prism")


def _prism(config, hw: hardware.Hardware, rt: Runtime | None) -> PrismRuntime | None:
    """Prism ML's llama.cpp build as a second runtime to rank models with, for ternary Bonsai models,
    when this platform has a build and the owner has not pinned another runtime."""
    if (rt is not None and rt.name == "prism") or (config.get("models.runtime") or "auto") != "auto":
        return None
    prism = PrismRuntime(config, hw=hw)
    return prism if prism.available() else None


def _ranked(hw, config, rt: Runtime | None, prism: Runtime | None) -> list[dict]:
    """Chat candidates for the main runtime and, if available, Prism's, best first."""
    found = fit.recommend(hw, config, "chat", _artifact_runtime(rt), include_rejected=True)
    if prism is not None:
        found += fit.recommend(hw, config, "chat", "prism")
    return sorted((c for c in found if c["ok"]), key=fit._order) + [c for c in found if not c["ok"]]


@dataclasses.dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str = ""
    vision: bool = False
    embedding_model: str = ""


def _hardware(config, hw: hardware.Hardware | None) -> hardware.Hardware:
    return hw or hardware.detect(float(config.get("sandbox.memory_gb") or 4))


def _runtime(config, hw, runtime: str | Runtime | None) -> Runtime | None:
    return runtime if isinstance(runtime, Runtime) else choose_runtime(config, hw, runtime)


def _artifact_runtime(rt: Runtime | None) -> str | None:
    return rt.name if rt and rt.name in catalog.ARTIFACT else None


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def target():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def plan(config, hw: hardware.Hardware | None = None, runtime: str | Runtime | None = None) -> dict:
    hw = _hardware(config, hw)
    rt = _runtime(config, hw, runtime)
    art = _artifact_runtime(rt)
    prism = _prism(config, hw, rt) if runtime is None else None     # only when the runtime is auto-chosen
    candidates = _ranked(hw, config, rt, prism)
    viable = [c for c in candidates if c["ok"]]
    chosen = viable[0] if viable else None
    notes = []
    if rt is None or not rt.available():
        notes.append(install_hint(hw, rt.name if rt else None))
    if chosen is None:
        notes.append(f"No chat model of at least {fit.MIN_CHAT_PARAMS_B:.0f}B parameters fits the "
                     f"{hw.budget_gb:.1f} GB budget. Free memory, lower sandbox.memory_gb, or point "
                     "models.base_url at a bigger machine with models.runtime set to openai.")
    elif chosen["license"] == "ask":
        notes.append(f"{chosen['id']}: {chosen['license_reason']}")
    chosen_rt = prism if chosen and chosen.get("runtime") == "prism" else rt
    return {
        "hardware": hw.to_dict(),
        "runtime": chosen_rt.describe() if chosen_rt else None,
        "model": chosen,
        "alternatives": viable[1:6],
        "rejected": [c for c in candidates if not c["ok"]],
        "license": ({"decision": chosen["license"], "reason": chosen["license_reason"],
                     "link": chosen["license_link"]} if chosen else None),
        # Embeddings come from Ollama, also when Prism's build serves the chat model.
        "embedding": next(iter(fit.recommend(hw, config, "embedding", "ollama")), None)
        if art == "ollama" or (chosen_rt is prism and OllamaRuntime(config).available()) else None,
        "notes": notes,
    }


def _accept(config, key: str) -> None:
    accepted = list(config.get("models.accepted_licenses") or [])
    if key and key not in accepted:
        config.set("models.accepted_licenses", accepted + [key])
        config.save()


def _license_gate(entry: dict, rt: Runtime, config, accept: bool, hub: Hub | None) -> tuple[str, str, str]:
    """Catalog license, plus the live license of the GGUF repo when llama.cpp downloads it."""
    sources: list = [entry]
    if rt.name == "llama.cpp":
        sources.append(entry["artifacts"]["gguf"]["repo"])
    for source in sources:
        decision, why, link = licenses.check(source, config, hub)
        if decision == "ask" and accept:
            info = entry.get("license", {}) if isinstance(source, dict) else (hub or Hub()).resolve_license(source)
            _accept(config, licenses.key(info.get("spdx"), info.get("name")))
            decision = "allow"
        if decision != "allow":
            return decision, why, link
    return "allow", "", ""


def _setup_embedding(config, rt: Runtime, hw, progress) -> str:
    """Ollama can serve an embedding model beside the chat model; other runtimes keep their setting."""
    if rt.name != "ollama":
        return config.get("models.embedding") or ""
    for cand in fit.recommend(hw, config, "embedding", "ollama"):
        if cand["license"] != "allow":
            continue
        entry = catalog.get(cand["id"])
        tag = entry["artifacts"]["ollama"]["tag"]
        try:
            if not rt.installed(entry):
                rt.pull(entry, progress)

            async def probe():
                llm = LLM(rt.base_url, tag)
                try:
                    return await llm.embed(["coderunner"])
                finally:
                    await llm.aclose()
            if _run_async(probe())[0]:
                return tag
        except Exception as exc:
            progress(f"embedding model {tag} unavailable: {exc}")
    return ""


def _external(config, rt: Runtime, model_id: str | None, progress, smoke) -> dict:
    model = model_id or config.get("models.chat") or ""
    if not model:
        return {"ok": False, "error": f"Set models.chat to the model served at {rt.base_url}"}
    if not rt.running():
        return {"ok": False, "error": f"{rt.base_url} is not answering"}
    progress(f"smoke testing {model} at {rt.base_url}")
    result = _run_async(smoke(rt.base_url, model, vision=False, api_key=rt.api_key))
    if not result["passed"]:
        return {"ok": False, "error": "The endpoint failed the tool-calling smoke test", "smoke": result}
    config.set("models.runtime", rt.name)
    config.set("models.base_url", rt.base_url)
    config.set("models.chat", model)
    config.save()
    return {"ok": True, "runtime": rt.name, "base_url": rt.base_url, "model": model, "smoke": result}


def setup(config, model_id: str | None = None, runtime: str | Runtime | None = None, accept_license: bool = False,
          progress: Callable[[str], None] = print, hw: hardware.Hardware | None = None, embedding: bool = True,
          smoke=smoketest.run, hub: Hub | None = None) -> dict:
    hw = _hardware(config, hw)
    rt = _runtime(config, hw, runtime)
    if rt is None:
        return {"ok": False, "error": "No local model runtime is installed. " + install_hint(hw)}
    if not rt.available():
        return {"ok": False, "error": f"{rt.name} is not available. {install_hint(hw, rt.name)}"}
    if rt.name not in LOCAL:
        return _external(config, rt, model_id, progress, smoke)

    prism = _prism(config, hw, rt) if runtime is None else None     # only when the runtime is auto-chosen
    if model_id:
        entry = catalog.get(model_id)
        if "chat" not in entry["roles"] or entry["params_b"] < fit.MIN_CHAT_PARAMS_B:
            return {"ok": False, "error": f"{model_id} is not a chat model of at least {fit.MIN_CHAT_PARAMS_B:.0f}B"}
        on = "prism" if prism is not None and catalog.artifact(entry, rt.name) is None \
            and catalog.artifact(entry, "prism") else rt.name
        cand = fit.candidate(entry, hw, config, "chat", on)
        if cand["fit"] == "no":
            progress(f"warning: {model_id} {cand['reasons'][0]}; trying anyway because it was requested")
        queue = [cand]  # an explicit choice is never silently replaced
    else:
        queue = [c for c in _ranked(hw, config, rt, prism) if c["ok"]][:1 + MAX_FALLBACKS]
        if not queue:
            return {"ok": False, "error": plan(config, hw, rt)["notes"][-1]}

    attempts = []
    main = rt
    for cand in queue:
        rt = prism if cand.get("runtime") == "prism" and prism is not None else main
        entry = catalog.get(cand["id"])
        decision, why, link = _license_gate(entry, rt, config, accept_license, hub)
        if decision == "ask":
            return {"ok": False, "needs_license": {"model": entry["id"], "reason": why, "link": link},
                    "attempts": attempts,
                    "error": f"{entry['id']} needs license acceptance: {why}. Re-run with accept_license=True."}
        if decision == "block":
            attempts.append({"model": entry["id"], "error": why})
            progress(f"skipping {entry['id']}: {why}")
            continue
        progress(f"{entry['id']} via {rt.name}: {'; '.join(cand['reasons'][:3])}")
        try:
            if rt.name == "ollama":
                rt.start()  # the Ollama API does the pulling
            if not rt.installed(entry):
                rt.pull(entry, progress)
            model = rt.prepare(entry, cand["ctx"])
            rt.start()
            vision = rt.vision(entry, model)
            progress(f"smoke testing {model}{' with vision' if vision else ''}")
            result = _run_async(smoke(rt.base_url, model, vision=vision, api_key=rt.api_key))
        except Exception as exc:
            attempts.append({"model": entry["id"], "error": str(exc)})
            progress(f"{entry['id']} failed: {exc}")
            continue
        attempts.append({"model": entry["id"], "smoke": result})
        for check in result["checks"]:
            progress(f"  {'pass' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']} ({check['seconds']}s)")
        if not all(c["passed"] for c in result["checks"] if c["name"] != "vision"):
            progress(f"{entry['id']} failed the smoke test")
            continue
        if vision and not result["passed"]:
            vision = False
            progress("vision check failed; continuing with text only")

        config.set("models.runtime", rt.name)
        config.set("models.base_url", rt.base_url)
        config.set("models.chat", model)
        config.set("models.vision", vision)
        config.set("models.context", cand["ctx"])
        if embedding:
            # Prism's build serves only the chat model; embeddings come from Ollama beside it.
            embedder = main if rt is prism and main is not None and main.name == "ollama" else rt
            if embedder is not rt and not embedder.running():
                embedder.start()
            config.set("models.embedding", _setup_embedding(config, embedder, hw, progress))
            config.set("models.embedding_url", embedder.base_url if embedder is not rt else "")
        config.save()
        return {"ok": True, "runtime": rt.name, "base_url": rt.base_url, "model": model, "entry": entry["id"],
                "ctx": cand["ctx"], "vision": vision, "embedding": config.get("models.embedding") or "",
                "smoke": result, "attempts": attempts}
    return {"ok": False, "error": "No model passed the tool-calling smoke test", "attempts": attempts}


def endpoint(config) -> Endpoint:
    base_url = config.get("models.base_url") or ""
    if not base_url:
        rt = choose_runtime(config)
        base_url = rt.base_url if rt else ""
    api_key = os.environ.get("CODERUNNER_MODEL_API_KEY") or config.get("models.api_key") or ""
    return Endpoint(base_url=base_url.rstrip("/"), model=config.get("models.chat") or "", api_key=api_key,
                    vision=bool(config.get("models.vision")), embedding_model=config.get("models.embedding") or "")


def ensure_running(config) -> Endpoint:
    """Start the configured local runtime if it is down; external endpoints are left alone."""
    rt = choose_runtime(config)
    ep = endpoint(config)
    if rt and rt.name in LOCAL and rt.base_url.rstrip("/") == ep.base_url and not rt.running():
        rt.start()
    return ep
