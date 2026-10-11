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
from .runtimes import LlamaCppRuntime, OllamaRuntime, PrismRuntime, Runtime, choose_runtime, install_hint

MAX_FALLBACKS = 2
WORTH_A_NOTE = 0.2          # utility gain (fit.utility) worth suggesting a change to the machine
LOCAL = ("ollama", "llama.cpp", "prism")


def _prism(config, hw: hardware.Hardware, rt: Runtime | None) -> PrismRuntime | None:
    """Prism ML's llama.cpp build as a second runtime to rank models with, for ternary Bonsai models,
    when this platform has a build and the owner has not pinned another runtime."""
    if (rt is not None and rt.name == "prism") or (config.get("models.runtime") or "auto") != "auto":
        return None
    prism = PrismRuntime(config, hw=hw)
    return prism if prism.available() else None


def _llama(config, rt: Runtime | None) -> Runtime | None:
    """llama.cpp next to Ollama, for models Ollama cannot serve: tool formats it does not parse, experts in RAM."""
    if rt is None or rt.name != "ollama":
        return None
    llama = LlamaCppRuntime(config)
    return llama if llama.available() else None


def _discovered(config, hw, rt: Runtime | None, progress: Callable[[str], None] | None = None,
                llama: Runtime | None = None) -> list[dict]:
    """Recent models found on Hugging Face, each with the best build that fits this machine."""
    if not config.get("models.discover", True) or rt is None or rt.name not in ("ollama", "llama.cpp", "mlx"):
        return []
    from . import discover
    known = {(e.get("license") or {}).get("base_repo", "").lower() for e in catalog.models()}
    entries = []
    for model in discover.cached(progress):
        if model["repo"].lower() in known:
            continue                        # curated entry with measured factors wins
        entry, on = discover.best_build(model, hw, config, rt.name), rt.name
        if entry is None and rt.name == "ollama":
            entry, on = discover.best_build(model, hw, config, "llama.cpp"), "llama.cpp"
        if entry:
            entry["runtime"] = on
            entry["missing_runtime"] = on == "llama.cpp" and llama is None     # worth installing llama.cpp
            entries.append(entry)
    return discover.place(entries, catalog.models())


def _ranked(hw, config, rt: Runtime | None, prism: Runtime | None, extra: list[dict] | None = None,
            llama: Runtime | None = None) -> list[dict]:
    """Chat candidates for the main runtime, Prism's build and discovered models, best first."""
    found = fit.recommend(hw, config, "chat", _artifact_runtime(rt), include_rejected=True)
    if prism is not None:
        found += fit.recommend(hw, config, "chat", "prism")
    if llama is not None:
        found += [c for c in fit.recommend(hw, config, "chat", "llama.cpp") if c.get("offload")]
    for entry in extra or []:
        cand = fit.candidate(entry, hw, config, "chat", entry.get("runtime") or _artifact_runtime(rt))
        cand["discovered"] = True
        if entry.get("missing_runtime") and cand["ok"]:
            cand["ok"] = False
            cand["reasons"].insert(0, f"needs {cand['runtime']}, which is not installed")
            cand["missing_runtime"] = True
        found.append(cand)
    return sorted((c for c in found if c["ok"]), key=fit._order) + [c for c in found if not c["ok"]]


def _entry(model_id: str, extra: list[dict]) -> dict:
    for entry in extra:
        if entry["id"] == model_id:
            return entry
    return catalog.get(model_id)


@dataclasses.dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str = ""
    vision: bool = False
    embedding_model: str = ""


def _hardware(config, hw: hardware.Hardware | None) -> hardware.Hardware:
    hw = hw or hardware.detect(float(config.get("sandbox.memory_gb") or 1))
    peers = [node for node in (config.get("models.cluster") or []) if node.get("host")]
    if peers and hw.nodes == 1:
        # Machines pooled over llama.cpp RPC (e.g. several DGX Sparks): each lends most of its memory.
        extra = sum(max(0.0, float(node.get("memory_gb") or 0) * 0.85 - 4) for node in peers)
        hw = dataclasses.replace(hw, budget_gb=hw.budget_gb + extra, nodes=1 + len(peers))
    return hw


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
    llama = _llama(config, rt) if runtime is None else None
    extra = _discovered(config, hw, rt, llama=llama) if runtime is None else []
    candidates = _ranked(hw, config, rt, prism, extra, llama)
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
    if chosen and hw.gpu == "apple" and hw.ram_gb >= 48:
        # macOS lets the GPU wire about 75% of memory; on big Macs the limit can be raised until restart.
        limit = hw.ram_gb - 8
        raised = dataclasses.replace(hw, gpu_limit_gb=limit,
                                     budget_gb=hardware.apple_budget(hw.ram_gb, limit, hw.sandbox_memory_gb))
        bigger = _discovered(config, raised, rt, llama=llama) if runtime is None else []    # builds that now fit
        better = max((c for c in _ranked(raised, config, rt, prism, bigger, llama)
                      if c["ok"] and c["tok_s"] >= fit.MIN_TOK_S and c["ctx"] >= fit.MIN_AGENT_CTX
                      and fit.utility(c) > fit.utility(chosen) + WORTH_A_NOTE), key=fit.utility, default=None)
        if better:
            notes.append(f"Raising the GPU memory limit allows {better['id']} ({better['need_gb']:.0f} GB): "
                         f"sudo sysctl iogpu.wired_limit_mb={int(limit * 1024)} (until restart)")
    unserved = [c for c in candidates if c.get("missing_runtime")]
    if runtime is None and llama is None and rt is not None and rt.name == "ollama":
        # Mixture-of-experts models with their experts in system RAM, which only llama.cpp does.
        unserved += [{**c, "missing_runtime": True}
                     for c in fit.recommend(hw, config, "chat", "llama.cpp") if c.get("offload")]
    missing = max((c for c in unserved if c["tok_s"] >= fit.MIN_TOK_S and c["fit"] != "no"
                   and (chosen is None or fit.utility(c) > fit.utility(chosen) + WORTH_A_NOTE)),
                  key=fit.utility, default=None)
    if missing:
        notes.append(f"{missing['id']} would rank higher but needs {missing['runtime']}: "
                     f"{install_hint(hw, missing['runtime'])}")
    chosen_rt = rt
    if chosen and chosen.get("runtime") == "prism":
        chosen_rt = prism
    elif chosen and chosen.get("runtime") == "llama.cpp" and llama is not None:
        chosen_rt = llama
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
        if art == "ollama" or (chosen_rt is not rt and OllamaRuntime(config).available()) else None,
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
    llama = _llama(config, rt) if runtime is None else None
    extra = _discovered(config, hw, rt, progress, llama) if runtime is None else []
    if model_id:
        entry = _entry(model_id, extra)
        if "chat" not in entry["roles"] or entry["params_b"] < fit.MIN_CHAT_PARAMS_B:
            return {"ok": False, "error": f"{model_id} is not a chat model of at least {fit.MIN_CHAT_PARAMS_B:.0f}B"}
        on = entry.get("runtime") or ("prism" if prism is not None and catalog.artifact(entry, rt.name) is None
                                      and catalog.artifact(entry, "prism") else rt.name)
        if on == "llama.cpp" and rt.name != "llama.cpp" and llama is None:
            return {"ok": False, "error": f"{model_id} needs llama.cpp. {install_hint(hw, 'llama.cpp')}"}
        cand = fit.candidate(entry, hw, config, "chat", on)
        if cand["fit"] == "no":
            progress(f"warning: {model_id} {cand['reasons'][0]}; trying anyway because it was requested")
        queue = [cand]  # an explicit choice is never silently replaced
    else:
        queue = [c for c in _ranked(hw, config, rt, prism, extra, llama) if c["ok"]][:1 + MAX_FALLBACKS]
        if not queue:
            return {"ok": False, "error": plan(config, hw, rt)["notes"][-1]}

    attempts = []
    main = rt
    for cand in queue:
        if cand.get("runtime") == "prism" and prism is not None:
            rt = prism
        elif cand.get("runtime") == "llama.cpp" and llama is not None:
            rt = llama
        else:
            rt = main
        entry = _entry(cand["id"], extra)
        if not cand.get("vision") and "vision" in entry.get("roles", []):
            entry = {**entry, "roles": [r for r in entry["roles"] if r != "vision"]}    # projector does not fit
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
            model = rt.prepare(entry, cand["ctx"], offload=True) if cand.get("offload") else \
                rt.prepare(entry, cand["ctx"])
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
            embedder = main if rt is not main and main is not None and main.name == "ollama" else rt
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
