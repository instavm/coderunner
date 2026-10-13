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

from ..core.llm import LLM
from . import bench, catalog, fit, hardware, licenses, smoketest
from .hub import Hub
from .runtimes import LlamaCppRuntime, OllamaRuntime, PrismRuntime, Runtime, choose_runtime, install_hint

MAX_FALLBACKS = 2
WORTH_A_NOTE = 0.2          # utility gain (fit.utility) worth suggesting a change to the machine
SLOWER_THAN_ESTIMATED = 1.5     # measured step time / estimate past which setup looks at the next model
LOCAL = ("ollama", "llama.cpp", "prism")


def _pinned(config) -> str | None:
    """The runtime the owner chose, if any. Setup writes its own pick to models.runtime as well (the daemon
    serves from it), so a local runtime counts as the owner's choice only with models.runtime_pinned;
    otherwise plans and setup compare models on every installed runtime again."""
    name = config.get("models.runtime") or "auto"
    if name == "auto" or (name in LOCAL and not config.get("models.runtime_pinned")):
        return None
    return name


def _main_runtime(config, hw, runtime: str | Runtime | None) -> Runtime | None:
    """The given or pinned runtime, else the one CodeRunner would pick (Ollama, else llama.cpp), else the
    one setup picked last time."""
    if runtime is not None:
        return _runtime(config, hw, runtime)
    return choose_runtime(config, hw, "auto") or choose_runtime(config, hw)


def _prism(config, hw: hardware.Hardware, rt: Runtime | None) -> PrismRuntime | None:
    """Prism ML's llama.cpp build as a second runtime to rank models with, for ternary Bonsai models,
    when this platform has a build and the owner has not pinned another runtime."""
    if (rt is not None and rt.name == "prism") or _pinned(config) is not None:
        return None
    prism = PrismRuntime(config, hw=hw)
    return prism if prism.available() else None


def _chat_on_llama(hw) -> bool:
    """On Apple silicon chat models run on CodeRunner's own llama.cpp server, Ollama keeping the embedding and
    decision models: Ollama's server keeps up to 8 GB of earlier prompts in memory, with no setting for it (2.4
    GB in one sample-task run on an 18 GB Mac, which then wrote 0.06 tokens/s), where CodeRunner's keeps none
    and sleeps when idle. Elsewhere llama.cpp's official build runs on the CPU only, so Ollama serves them."""
    return hw is not None and hw.gpu == "apple" and hw.arch == "arm64"


def _llama(config, rt: Runtime | None, hw=None) -> Runtime | None:
    """llama.cpp next to Ollama, for models Ollama cannot serve (tool formats it does not parse, experts in RAM)
    and, on Apple silicon, for chat models (setup installs llama.cpp's official build)."""
    if rt is None or rt.name != "ollama":
        return None
    llama = LlamaCppRuntime(config)
    return llama if llama.available() or _chat_on_llama(hw) else None


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
        entry, on = None, rt.name
        if rt.name == "ollama" and llama is not None and _chat_on_llama(hw):
            entry, on = discover.best_build(model, hw, config, "llama.cpp"), "llama.cpp"
        if entry is None:
            entry, on = discover.best_build(model, hw, config, rt.name), rt.name
        if entry is None and rt.name == "ollama":
            entry, on = discover.best_build(model, hw, config, "llama.cpp"), "llama.cpp"
        if entry:
            entry["runtime"] = on
            entry["missing_runtime"] = on == "llama.cpp" and llama is None     # worth installing llama.cpp
            entries.append(entry)
    return discover.place(entries, catalog.models())


def _measured(hw) -> dict[tuple[str, str], dict]:
    """Sample-task results (tasks.py) for this kind of machine, by model and runtime: measured here, else
    shipped from an identical machine. A run that paged from disk counts only until one that did not: its
    times are slower than the machine's, but leaving it out made its model count as unmeasured, which ranks
    on estimates (a compact Bonsai 2 build then beat the measured one on paper after losing to it in the run)."""
    from . import tasks
    machine = bench.fingerprint(hw)
    found: dict[tuple[str, str], dict] = {}
    for record in tasks.shared() + tasks.load():          # this machine's own results win
        if record.get("machine") == machine and record.get("suite") == tasks.SUITE:
            key = (record["model"], record["runtime"])
            if record.get("paging") and key in found and not found[key].get("paging"):
                continue
            found[key] = record
    return found


def _with_measured(c: dict, measured: dict, main: str | None) -> dict:
    record = measured.get((c["id"], c.get("runtime") or main or ""))
    if record is None:
        c["reasons"].append("sample tasks not run on this kind of machine yet (coderunner models bench --tasks)")
        return c
    from . import tasks
    c["tasks"] = {"passed": record["passed"], "total": record["total"], "seconds": record["seconds"],
                  "date": record["date"]}
    c["step_s"] = round(tasks.step_seconds(record), 1)
    c["reasons"].append(f"sample tasks here: {record['passed']} of {record['total']} passed in "
                        f"{record['seconds'] / 60:.0f} min, {c['step_s']:.0f} s a step ({record['date']}"
                        + (", paging from disk: slower than this machine can be" if record.get("paging") else "") + ")")
    return c


def _ranked(hw, config, rt: Runtime | None, prism: Runtime | None, extra: list[dict] | None = None,
            llama: Runtime | None = None) -> list[dict]:
    """Chat candidates for the main runtime, Prism's build and discovered models, best first."""
    found = fit.recommend(hw, config, "chat", _artifact_runtime(rt), include_rejected=True)
    if prism is not None:
        found += fit.recommend(hw, config, "chat", "prism")
    if llama is not None:
        on_llama = fit.recommend(hw, config, "chat", "llama.cpp")
        if _chat_on_llama(hw):
            moved = {c["id"] for c in on_llama}
            found = [c for c in found if not (c["ok"] and c["id"] in moved)] + on_llama
        else:
            found += [c for c in on_llama if c.get("offload")]
    for entry in extra or []:
        cand = fit.candidate(entry, hw, config, "chat", entry.get("runtime") or _artifact_runtime(rt))
        cand["discovered"] = True
        if entry.get("missing_runtime") and cand["ok"]:
            cand["ok"] = False
            cand["reasons"].insert(0, f"needs {cand['runtime']}, which is not installed")
            cand["missing_runtime"] = True
        found.append(cand)
    measured = _measured(hw)
    found = [_with_measured(c, measured, rt.name if rt else None) for c in found]
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
    if not hw.measured:
        hw = bench.apply(hw, bench.load(hw))      # speeds measured on this machine replace spec-sheet ones
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
    runtime = runtime if runtime is not None else _pinned(config)
    rt = _main_runtime(config, hw, runtime)
    art = _artifact_runtime(rt)
    prism = _prism(config, hw, rt) if runtime is None else None     # only when the runtime is auto-chosen
    llama = _llama(config, rt, hw) if runtime is None else None
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
                      if c["ok"] and c["step_s"] <= fit.MAX_STEP_S and c["ctx"] >= fit.MIN_AGENT_CTX
                      and fit.utility(c) > fit.utility(chosen) + WORTH_A_NOTE), key=fit.utility, default=None)
        if better:
            notes.append(f"Raising the GPU memory limit allows {better['id']} ({better['need_gb']:.0f} GB): "
                         f"sudo sysctl iogpu.wired_limit_mb={int(limit * 1024)} (until restart)")
    unserved = [c for c in candidates if c.get("missing_runtime")]
    if runtime is None and llama is None and rt is not None and rt.name == "ollama":
        # Mixture-of-experts models with their experts in system RAM, which only llama.cpp does.
        unserved += [{**c, "missing_runtime": True}
                     for c in fit.recommend(hw, config, "chat", "llama.cpp") if c.get("offload")]
    missing = max((c for c in unserved if c["step_s"] <= fit.MAX_STEP_S and c["fit"] != "no"
                   and (chosen is None or fit.utility(c) > fit.utility(chosen) + WORTH_A_NOTE)),
                  key=fit.utility, default=None)
    if missing:
        notes.append(f"{missing['id']} would rank higher but needs {missing['runtime']}: "
                     f"{install_hint(hw, missing['runtime'])}")
    chosen_rt = rt
    if chosen and chosen.get("runtime") == "prism" and prism is not None:      # else Prism is the main runtime
        chosen_rt = prism
    elif chosen and chosen.get("runtime") == "llama.cpp" and llama is not None:
        chosen_rt = llama
    ollama = art == "ollama" or (chosen_rt is not rt and OllamaRuntime(config).available())
    embedder = next(iter(fit.recommend(hw, config, "embedding", "ollama")), None) if ollama else None
    judge = decision_pick(hw, config, chosen["need_gb"] + (embedder or {}).get("need_gb", 0.0)) \
        if ollama and chosen else None
    return {
        "hardware": hw.to_dict(),
        "runtime": chosen_rt.describe() if chosen_rt else None,
        "model": chosen,
        "alternatives": viable[1:6],
        "rejected": [c for c in candidates if not c["ok"]],
        "license": ({"decision": chosen["license"], "reason": chosen["license_reason"],
                     "link": chosen["license_link"]} if chosen else None),
        # Embeddings and the decision model come from Ollama, also when another runtime serves the chat model.
        "embedding": embedder,
        # None: no decision model fits beside the chat model, which then judges browser steps itself.
        "decision": judge,
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


def _free_gpu(config, keep: Runtime) -> None:
    """Unload the model the assistant used so far before another one loads: two models at once froze an 18 GB
    Mac twice. Ollama keeps its other models (embeddings); a llama.cpp server of another runtime stops (its own
    runtime restarts it with the new model)."""
    before, chat = config.get("models.runtime") or "", config.get("models.chat") or ""
    if before not in LOCAL or not chat:
        return
    old = choose_runtime(config, name=before)
    if old is None or not old.running():
        return
    if old.name == "ollama":
        old.unload(chat)
    elif old.name != keep.name:
        old.stop()


def setup(config, model_id: str | None = None, runtime: str | Runtime | None = None, accept_license: bool = False,
          progress: Callable[[str], None] = print, hw: hardware.Hardware | None = None, embedding: bool = True,
          smoke=smoketest.run, hub: Hub | None = None, measure=None, decision: bool = True) -> dict:
    """measure times the model once it passes the smoke test (default bench.measure; False skips it);
    decision also sets up a decision model to judge browser steps when memory allows."""
    hw = _hardware(config, hw)
    runtime = runtime if runtime is not None else _pinned(config)
    rt = _main_runtime(config, hw, runtime)
    if rt is None:
        return {"ok": False, "error": "No local model runtime is installed. " + install_hint(hw)}
    if not rt.available():
        return {"ok": False, "error": f"{rt.name} is not available. {install_hint(hw, rt.name)}"}
    if rt.name not in LOCAL:
        return _external(config, rt, model_id, progress, smoke)
    if hasattr(rt, "ensure_current"):
        try:
            rt.ensure_current(progress)       # EmbeddingGemma 2 and decision models need a recent Ollama
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    prism = _prism(config, hw, rt) if runtime is None else None     # only when the runtime is auto-chosen
    llama = _llama(config, rt, hw) if runtime is None else None
    extra = _discovered(config, hw, rt, progress, llama) if runtime is None else []
    if model_id:
        entry = _entry(model_id, extra)
        if "chat" not in entry["roles"] or entry["params_b"] < fit.MIN_CHAT_PARAMS_B:
            return {"ok": False, "error": f"{model_id} is not a chat model of at least {fit.MIN_CHAT_PARAMS_B:.0f}B"}
        on = entry.get("runtime") or ("prism" if prism is not None and catalog.artifact(entry, rt.name) is None
                                      and catalog.artifact(entry, "prism") else rt.name)
        if on == "ollama" and llama is not None and _chat_on_llama(hw) and catalog.artifact(entry, "llama.cpp"):
            on = "llama.cpp"                    # as the plan does (_chat_on_llama)
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
    for index, cand in enumerate(queue):
        if cand.get("runtime") == "prism" and prism is not None:
            rt = prism
        elif cand.get("runtime") == "llama.cpp" and llama is not None:
            rt = llama
        else:
            rt = main
        entry = _entry(cand["id"], extra)
        if not cand.get("vision") and "vision" in entry.get("roles", []):
            entry = {**entry, "roles": [r for r in entry["roles"] if r != "vision"]}    # projector does not fit
        verdict, why, link = _license_gate(entry, rt, config, accept_license, hub)
        if verdict == "ask":
            return {"ok": False, "needs_license": {"model": entry["id"], "reason": why, "link": link},
                    "attempts": attempts,
                    "error": f"{entry['id']} needs license acceptance: {why}. Re-run with accept_license=True."}
        if verdict == "block":
            attempts.append({"model": entry["id"], "error": why})
            progress(f"skipping {entry['id']}: {why}")
            continue
        progress(f"{entry['id']} via {rt.name}: {'; '.join(cand['reasons'][:3])}")
        pulled = False
        try:
            if rt.name == "ollama":
                rt.start()  # the Ollama API does the pulling
            if not rt.installed(entry):
                rt.pull(entry, progress)
                # Only a finished download is deleted when the model is then rejected: an interrupted one
                # stays, so running setup again resumes it (7 GB over a flaky connection).
                pulled = True
            built = rt.build_format(entry)
            if built not in (None, "gguf"):
                # Ollama's plain tags became MLX builds on Apple silicon: qwen3.5:9b took 16.3 GB on an 18 GB
                # Mac, against 8.1 GB planned for its GGUF build. The catalog names GGUF tags; this catches
                # a tag that changes again.
                raise RuntimeError(f"{rt.name} serves {entry['id']} as a {built} build, not the GGUF build "
                                   "its memory was planned for")
            _free_gpu(config, rt)
            model = rt.prepare(entry, cand["ctx"], offload=True) if cand.get("offload") else \
                rt.prepare(entry, cand["ctx"])
            rt.start()
            vision = rt.vision(entry, model)
            progress(f"smoke testing {model}{' with vision' if vision else ''}")
            result = _run_async(smoke(rt.base_url, model, vision=vision, api_key=rt.api_key))
        except Exception as exc:
            attempts.append({"model": entry["id"], "error": str(exc)})
            progress(f"{entry['id']} failed: {exc}")
            _discard(rt, entry, pulled, progress)
            continue
        attempts.append({"model": entry["id"], "smoke": result})
        for check in result["checks"]:
            progress(f"  {'pass' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']} ({check['seconds']}s)")
        if not all(c["passed"] for c in result["checks"] if c["name"] != "vision"):
            progress(f"{entry['id']} failed the smoke test")
            _discard(rt, entry, pulled, progress)
            continue
        if vision and not result["passed"]:
            vision = False
            progress("vision check failed; continuing with text only")
        speed = _measure(measure, rt, model, progress)
        if speed:
            attempts[-1]["speed"] = speed
            cal = bench.calibration(hw, entry, cand.get("runtime") or _artifact_runtime(rt), speed, cand["ctx"])
            bench.save(cal)
            progress(f"measured {speed['read_tok_s']:.0f} tok/s reading, {speed['gen_tok_s']:.1f} tok/s writing: "
                     f"{speed['step_s']:.0f} s per agent step (estimated {cand['step_s']:.0f} s)")
            if not model_id and speed["step_s"] > SLOWER_THAN_ESTIMATED * cand["step_s"]:
                # Slower here than the spec sheet said: rank the remaining candidates again with what was
                # measured, and move on if one of them is now clearly better.
                measured = bench.apply(hw, cal)
                now = fit.candidate(entry, measured, config, "chat", cand.get("runtime"))
                # This model's own measurement, not the machine factors (which are kept within a range).
                now.update(tok_s=speed["gen_tok_s"], read_tok_s=speed["read_tok_s"], step_s=speed["step_s"])
                rest = sorted((fit.candidate(_entry(c["id"], extra), measured, config, "chat", c.get("runtime"))
                               for c in queue[index + 1:]), key=fit._order)
                if rest and rest[0]["ok"] and fit._order(rest[0]) < fit._order(now) and (
                        now["step_s"] > fit.MAX_STEP_S or fit.utility(rest[0]) > fit.utility(now) + WORTH_A_NOTE):
                    progress(f"{entry['id']} is slower here than estimated; trying {rest[0]['id']}")
                    attempts[-1]["error"] = "slower on this machine than estimated"
                    queue[index + 1:] = rest
                    _discard(rt, entry, pulled, progress)
                    continue

        config.set("models.runtime", rt.name)
        config.set("models.runtime_pinned", runtime is not None)     # the owner's choice, kept by later plans
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
        if decision:
            judge_rt = main if main is not None and main.name == "ollama" else rt
            _setup_decision(config, judge_rt, rt, hw, cand["need_gb"] + _embedding_need(config, hw), progress)
        config.save()
        return {"ok": True, "runtime": rt.name, "base_url": rt.base_url, "model": model, "entry": entry["id"],
                "ctx": cand["ctx"], "vision": vision, "embedding": config.get("models.embedding") or "",
                "smoke": result, "attempts": attempts}
    return {"ok": False, "error": "No model passed the tool-calling smoke test", "attempts": attempts}


def _discard(rt: Runtime, entry: dict, pulled: bool, progress) -> None:
    """Delete the weights of a model this setup downloaded and then did not use."""
    if pulled:
        try:
            rt.remove(entry)
            progress(f"removed {entry['id']}'s weights")
        except Exception as exc:
            progress(f"could not remove {entry['id']}: {exc}")


def _measure(measure, rt: Runtime, model: str, progress) -> dict | None:
    if measure is False:
        return None
    measure = measure or bench.measure
    progress(f"timing {model} on this machine")
    try:
        return _run_async(measure(rt.base_url, model, api_key=rt.api_key))
    except Exception as exc:
        progress(f"could not time {model}: {exc}")
        return None


def _embedding_need(config, hw) -> float:
    tag = config.get("models.embedding") or ""
    for cand in fit.recommend(hw, config, "embedding", "ollama"):
        if catalog.get(cand["id"])["artifacts"]["ollama"]["tag"] == tag:
            return cand["need_gb"]
    return 0.0


def decision_pick(hw: hardware.Hardware, config, used_gb: float) -> dict | None:
    """The most accurate decision model that fits well beside the models already using `used_gb`."""
    left = dataclasses.replace(hw, budget_gb=max(0.0, hw.budget_gb - used_gb))
    return next((c for c in fit.recommend(left, config, "decision", "ollama")
                 if c["fit"] == "good" and c["license"] == "allow"), None)


def _setup_decision(config, rt: Runtime | None, chat_rt: Runtime, hw, used_gb: float, progress) -> None:
    """A dedicated decision model judges browser steps when memory allows; otherwise the chat model does."""
    for key in ("models.decision", "models.decision_api", "models.decision_url"):
        config.set(key, "")
    if rt is None or rt.name != "ollama":
        return
    pick = decision_pick(hw, config, used_gb)
    if pick is None:
        progress("no decision model fits beside the chat model; the chat model judges browser steps")
        return
    entry = catalog.get(pick["id"])
    tag = entry["artifacts"]["ollama"]["tag"]
    try:
        if not rt.installed(entry):
            rt.pull(entry, progress)
        from ..safety.decision import COMMIT_CRITERIA, COMMIT_QUESTION, DecisionModel

        async def probe():
            model = DecisionModel(rt.base_url, tag, api="systemone", timeout=120)
            try:
                return await model.probability('Site: example.com\nAction: click link "Next page"',
                                               COMMIT_QUESTION, COMMIT_CRITERIA, {"commit", "unclear"})
            finally:
                await model.aclose()
        _run_async(probe())
    except Exception as exc:
        progress(f"decision model {tag} unavailable ({exc}); the chat model judges browser steps")
        return
    config.set("models.decision", tag)
    config.set("models.decision_api", "systemone")
    config.set("models.decision_url", rt.base_url if rt is not chat_rt else "")
    config.set("models.decision_threshold", entry["decision"]["threshold"])
    progress(f"decision model: {tag} ({entry['decision']['measured']})")


DRIVER_CTX = 16384     # a driver keeps its last three screenshots (about 1,300 tokens each) and its steps


def _chat_need(config, hw) -> float | None:
    """Memory the assistant's configured model needs; None when it is not a catalog model."""
    alias = config.get("models.chat") or ""
    model_id = alias.removeprefix("coderunner-")
    try:
        entry = catalog.get(model_id)
    except KeyError:
        return None
    runtime = config.get("models.runtime") or None
    return fit.estimate(entry, hw, ctx=int(config.get("models.context") or 32768),
                        runtime=None if runtime in ("auto", "openai") else runtime)["need_gb"]


def driver_pick(hw: hardware.Hardware, config, used_gb: float | None) -> dict | None:
    """The browser driver for this machine. The best-scoring one that fits beside the assistant's model
    stays loaded with it. Where none does (18 GB Macs and below), the smallest one that fits on its own takes
    turns with it: each turn costs a load (Fara1.5-4B: 10-20 s on an M3 Pro) and the assistant's model
    reloading after, so the quickest to load wins there."""
    drivers = sorted(catalog.by_role("driver"), key=lambda e: -e["driver"]["webvoyager"])
    sized = [(e, fit.estimate(e, hw, ctx=DRIVER_CTX, runtime="driver")["need_gb"]) for e in drivers]
    if used_gb is not None:
        for entry, need in sized:
            if used_gb + need <= hw.budget_gb:
                return {"id": entry["id"], "need_gb": need, "exclusive": False,
                        "reason": f"fits beside the assistant's model ({used_gb + need:.1f} of {hw.budget_gb:.1f} GB)"}
    embedding = _embedding_need(config, hw)
    alone = sorted(((e, need) for e, need in sized if embedding + need <= hw.budget_gb), key=lambda pair: pair[1])
    if not alone:
        return None
    entry, need = alone[0]
    return {"id": entry["id"], "need_gb": need, "exclusive": True,
            "reason": f"takes turns with the assistant's model: both need more than {hw.budget_gb:.1f} GB"}


def setup_driver(config, model_id: str | None = None, progress: Callable[[str], None] = print,
                 hub: Hub | None = None, off: bool = False) -> dict:
    """Download and configure the browser driver (models.driver), or switch it off."""
    if off:
        config.set("models.driver", {})
        config.save()
        return {"ok": True, "driver": None}
    from .runtimes import DriverRuntime
    hw = _hardware(config, None)
    used = _chat_need(config, hw)
    used = None if used is None else used + _embedding_need(config, hw)
    pick = driver_pick(hw, config, used)
    if model_id:
        entry = catalog.get(model_id)
        need = fit.estimate(entry, hw, ctx=DRIVER_CTX, runtime="driver")["need_gb"]
        exclusive = used is None or used + need > hw.budget_gb
        pick = {"id": model_id, "need_gb": need, "exclusive": exclusive,
                "reason": "chosen" + (" (takes turns with the assistant's model)" if exclusive else "")}
    if pick is None:
        return {"ok": False, "error": "no browser driver fits this machine"}
    entry = catalog.get(pick["id"])
    runtime = DriverRuntime(config, hub=hub)
    if not runtime.available():
        runtime.install(progress)
    if not runtime.installed(entry):
        runtime.pull(entry, progress)
    alias = runtime.prepare(entry, DRIVER_CTX)
    settings = dict(config.get("models.driver") or {})       # keeps a hand-set binary
    settings.update(model=entry["id"], exclusive=pick["exclusive"], steps=int(settings.get("steps") or 30))
    config.set("models.driver", settings)
    config.save()
    progress(f"browser driver: {entry['id']} ({pick['reason']})")
    return {"ok": True, "driver": entry["id"], "alias": alias, "base_url": runtime.base_url, **pick}


def bench_here(config, model: str | None = None, measure=None, hw: hardware.Hardware | None = None) -> dict:
    """Time the configured model (or `model` on the configured runtime), save the calibration and re-plan."""
    hw = hw or hardware.detect(float(config.get("sandbox.memory_gb") or 1))
    ep = ensure_running(config)
    model = model or ep.model
    if not model or not ep.base_url:
        return {"ok": False, "error": "No model is configured yet; run `coderunner setup` first."}
    measured = _run_async((measure or bench.measure)(ep.base_url, model, api_key=ep.api_key))
    runtime = config.get("models.runtime") or ""
    entry_id = model.split(":")[0].removeprefix("coderunner-")
    entry = next((e for e in catalog.models() if e["id"] == entry_id), None)
    cal = None
    if entry is not None:
        cal = bench.calibration(hw, entry, runtime if runtime in catalog.ARTIFACT else None, measured,
                                int(config.get("models.context") or 32768))
        bench.save(cal)
    planned = plan(config, bench.apply(hw, cal or bench.load(hw)))
    version = None
    if runtime == "ollama":
        found = OllamaRuntime(config).version()
        version = ".".join(map(str, found)) if found else None
    picks = [c for c in [planned["model"], *planned["alternatives"]] if c][:5]
    return {"ok": True, "model": model, "measured": measured, "calibration": cal, "plan": planned,
            "report": bench.report(bench.apply(hw, cal or bench.load(hw)), cal or bench.load(hw), picks,
                                   {"name": runtime, "version": version})}


def bench_candidates(config, count: int = 2, progress: Callable[[str], None] = print,
                     hw: hardware.Hardware | None = None, run_tasks=None, **setup_options) -> dict:
    """Bench before use: run the sample tasks (tasks.py) with the best candidates for this machine that have
    no result for it yet, one model at a time, then set up the best by the measurements and delete what this
    downloaded for the others."""
    from . import tasks
    run_tasks = run_tasks or tasks.run
    hw = _hardware(config, hw)
    planned = plan(config, hw)
    queue = [c for c in [planned["model"], *planned["alternatives"]] if c and not c.get("tasks")][:count]
    downloaded, runs = [], []
    for cand in queue:
        rt = choose_runtime(config, hw, cand.get("runtime") or "auto")
        entry = _entry(cand["id"], [])
        fresh = rt is not None and not rt.installed(entry)
        progress(f"== {cand['id']}")
        for attempt in (1, 2):
            done = setup(config, model_id=cand["id"], hw=hw, progress=progress,
                         **{"embedding": False, "decision": False, **setup_options})
            cut = any("interrupted" in str(a.get("error", "")) for a in done.get("attempts") or [])
            if done.get("ok") or not cut or attempt == 2:
                break
            progress(f"{cand['id']}: the download was cut off; resuming it")
        if fresh and done.get("ok"):
            downloaded.append((done["runtime"], entry))
        if not done.get("ok"):
            if fresh and rt is not None:
                rt.remove(entry)                    # a partial download nobody will resume
            runs.append({"model": cand["id"], "error": done.get("error", "setup failed")})
            continue
        result = _run_async(run_tasks(config, progress=progress))
        runs.append({"model": cand["id"], **{k: result.get(k) for k in ("passed", "total", "seconds", "paging", "error")}})
    after = plan(config, hw)
    best = after["model"]
    if best and best["id"] != entry_id(config):
        progress(f"== setting up {best['id']}, the best measured here")
        final = setup(config, model_id=best["id"], hw=hw, progress=progress, **setup_options)
        if not final.get("ok"):
            return {"ok": False, "error": f"{best['id']} could not be set up again: {final.get('error')}",
                    "runs": runs, "plan": after}
    for name, entry in downloaded:
        if not best or entry["id"] != best["id"]:
            choose_runtime(config, hw, name).remove(entry)
            progress(f"removed {entry['id']}'s weights")
    return {"ok": True, "runs": runs, "model": best, "plan": after}


def entry_id(config) -> str:
    """The catalog id of the configured chat model."""
    return (config.get("models.chat") or "").split(":")[0].removeprefix("coderunner-")


def endpoint(config) -> Endpoint:
    base_url = config.get("models.base_url") or ""
    if not base_url:
        rt = choose_runtime(config)
        base_url = rt.base_url if rt else ""
    api_key = os.environ.get("CODERUNNER_MODEL_API_KEY") or config.get("models.api_key") or ""
    return Endpoint(base_url=base_url.rstrip("/"), model=config.get("models.chat") or "", api_key=api_key,
                    vision=bool(config.get("models.vision")), embedding_model=config.get("models.embedding") or "")


def restart(config) -> bool:
    """Restart the configured local runtime (it crashed or cannot compute); False for external endpoints."""
    rt = choose_runtime(config)
    if not (rt and rt.name in LOCAL and rt.base_url.rstrip("/") == endpoint(config).base_url):
        return False
    rt.stop()
    rt.start()
    return rt.running()


def ensure_running(config) -> Endpoint:
    """Start the configured local runtime if it is down; external endpoints are left alone."""
    rt = choose_runtime(config)
    ep = endpoint(config)
    if rt and rt.name in LOCAL and rt.base_url.rstrip("/") == ep.base_url and not rt.running():
        rt.start()
    return ep
