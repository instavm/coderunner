"""Detect, choose, fetch, serve and verify a local model.

`plan` explains what would happen, `setup` does it and only writes the config
once the served model passes the tool-calling smoke test, and `endpoint` /
`ensure_running` are what the daemon uses afterwards.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import shutil
import threading
from typing import Callable

from ..core.llm import LLM
from . import bench, catalog, fit, hardware, licenses, smoketest
from .hub import Hub, HubError
from .runtimes import (
    JudgeRuntime,
    LlamaCppRuntime,
    MLXRuntime,
    OllamaRuntime,
    PrismRuntime,
    Runtime,
    VoiceRuntime,
    choose_runtime,
    install_hint,
    models_dir,
)

MAX_FALLBACKS = 2
WORTH_A_NOTE = 0.2          # utility gain (fit.utility) worth suggesting a change to the machine
SLOWER_THAN_ESTIMATED = 1.5     # measured step time / estimate past which setup looks at the next model
MIN_FREE_GB = 5.0           # disk kept free by downloads unless files.min_free_gb says otherwise
MEMORY_MARGIN_GB = 1.0      # memory left free beside a model that loads: with 8.8 GB free, a 9.7 GB model swapped
LOCAL = ("ollama", "llama.cpp", "prism")


def _is_managed_local(rt: Runtime | None) -> bool:
    """True when CodeRunner owns the runtime process and local model lifecycle (including managed MLX)."""
    return rt is not None and (rt.name in LOCAL or (isinstance(rt, MLXRuntime) and rt.managed()))


def _pinned(config) -> str | None:
    """The runtime the owner chose, if any. Setup writes its own pick to models.runtime as well (the daemon
    serves from it), so a local runtime counts as the owner's choice only with models.runtime_pinned;
    otherwise plans and setup compare models on every installed runtime again."""
    name = config.get("models.runtime") or "auto"
    if name == "auto" or (name in LOCAL and not config.get("models.runtime_pinned")):
        return None
    return name


def _main_runtime(config, hw, runtime: str | Runtime | None) -> Runtime | None:
    """The given or pinned runtime. Else, on Apple silicon, CodeRunner's own llama.cpp server (_chat_on_llama):
    one runtime for every model there, as in Ori, with embeddings on Core ML and no Ollama. Elsewhere the one
    CodeRunner would pick (Ollama, else llama.cpp), else the one setup picked last time, else Ollama where
    setup can install its own copy (_own_ollama)."""
    if runtime is not None:
        return _runtime(config, hw, runtime)
    if _chat_on_llama(hw):
        return LlamaCppRuntime(config, hw=hw)
    found = choose_runtime(config, hw, "auto") or choose_runtime(config, hw)
    if found is None:
        own = OllamaRuntime(config)
        return own if own.can_install() else None
    return found


def _own_ollama(rt: Runtime, hw, progress: Callable[[str], None]) -> str | None:
    """A machine with nothing to run models with gets CodeRunner's own copy of Ollama: a pinned release,
    checked against its published checksum, under CODERUNNER_HOME/runtime. Returns what went wrong, if
    anything."""
    if rt.available() or not isinstance(rt, OllamaRuntime):
        return None
    if not rt.can_install():
        return f"Ollama is not installed. {install_hint(hw, 'ollama')}"
    progress(f"no model runtime is installed here: fetching CodeRunner's own copy of Ollama {rt.RELEASE}")
    try:
        rt.install(progress)
    except Exception as exc:
        return f"Ollama {rt.RELEASE} could not be installed ({exc}). {install_hint(hw, 'ollama')}"
    return None


NAMES = {"embeddinggemma-2": "EmbeddingGemma 2", "embeddinggemma-300m": "EmbeddingGemma 300M", "tev1-4b": "tev1 4B",
         "judge-4b": "Safety Judge 4B", "gpt-oss-20b": "gpt-oss 20B", "gpt-oss-120b": "gpt-oss 120B",
         "qwen3-embedding-0.6b": "Qwen3 Embedding 0.6B", "bonsai-2.5-27b": "Bonsai 2.5 27B",
         "bonsai-2.5-27b-compact": "Bonsai 2.5 27B Compact", "bonsai-2-9b": "Bonsai 2 9B", "bonsai-2-4b": "Bonsai 2 4B",
         "inkling-2-mini": "Inkling 2 Mini", "inkling-2-small": "Inkling 2 Small",
         "ornith-1.5-122b": "Ornith 1.5 122B", "mimo-v2.6-flash-rl": "MiMo V2.6 Flash",
         "mimo-v2.6-distill-9b": "MiMo V2.6 Distill 9B", "glm-5.3-flash": "GLM 5.3 Flash",
         "deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash", "qwen3.8-flash-next": "Qwen3.8 Flash Next",
         "qwen3.8-27b-q6": "Qwen3.8 27B Q6", "holo4-35b-a3b": "Holo 4 35B", "mage-vl-cowork": "Mage VL Cowork"}


def pretty(model_id: str) -> str:
    """A model's name as an owner reads it: bonsai-2-27b -> Bonsai 2 27B."""
    import re
    if model_id in NAMES:
        return NAMES[model_id]
    words = []
    for word in model_id.removeprefix("hf-").split("-"):
        if re.fullmatch(r"[ae]?\d+(\.\d+)?[bm]", word):
            words.append(word.upper())                       # 27b, a3b, e4b
        elif re.fullmatch(r"\d+bit", word):
            words.append(word.replace("bit", "-bit"))
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def _cpu_note(hw, chosen: dict | None = None) -> str | None:
    """What an owner should hear before setting up a machine with no GPU CodeRunner can use: it works, slowly."""
    if hw.gpu != "none":
        return None
    speed = (f": about {chosen['tok_s']:.0f} tokens a second and {chosen['step_s']:.0f} s a step with {chosen['id']}"
             if chosen and chosen.get("tok_s") else "")
    return (f"No GPU was found here ({hw.chip or 'unknown processor'}), so models run on the processor, slowly"
            f"{speed}. An Apple silicon Mac or an NVIDIA GPU runs them many times faster.")


def _free_gb(path) -> float:
    """Decimal GB free on the disk that holds `path` (or will, once it exists)."""
    path = pathlib.Path(path).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def _room_gb(config) -> tuple[float, float, float]:
    """(GB downloads may take, GB free, GB kept free) where CodeRunner keeps models. The floor is
    files.min_free_gb, the one file indexing pauses at: a full disk stops macOS swapping, and on a small Mac
    the model with it."""
    floor = float(config.get("files.min_free_gb", MIN_FREE_GB) or 0)
    free = _free_gb(models_dir(config))
    return free - floor, free, floor


def _no_room(config, need_gb: float) -> str | None:
    """Why a download of this size may not start (it would take the disk under its floor); None when it may."""
    room, free, floor = _room_gb(config)
    if need_gb <= room:
        return None
    return (f"a {need_gb:.1f} GB download, and {max(room, 0):.1f} GB can be used ({free:.1f} GB free, "
            f"{floor:.0f} GB kept free)")


def _download_gb(entry: dict, cand: dict, rt: Runtime) -> float:
    """Decimal GB of this model's files for this runtime (its vision projector included when it is used)."""
    art = catalog.artifact(entry, rt.name if rt.name in catalog.ARTIFACT else None) or {}
    return float(art.get("size_gb") or 0) + (float(art.get("mmproj_gb") or 0) if cand.get("vision") else 0.0)


# Decimal GB of the runtime archives setup may fetch (the pinned releases' own sizes).
RUNTIME_GB = {"ollama-darwin.tgz": 0.17, "ollama-linux-amd64.tar.zst": 1.44, "ollama-linux-arm64.tar.zst": 1.56,
              "llama.cpp": 0.03, "prism": 0.01, "coreml": 0.12, "mlx": 0.25}


def downloads(config, planned: dict, hw: hardware.Hardware | None = None) -> list[dict]:
    """What setup still has to fetch for this plan, as [{"what", "gb"}]: runtimes that are not here, then the
    assistant's model, the embedding builds and the decision model. gb is None where the catalog has no size
    (a discovered model's is in its reasons)."""
    from . import coreml
    hw = _hardware(config, hw)
    found: list[dict] = []
    ollama = OllamaRuntime(config)
    name = (planned.get("model") or {}).get("runtime") or ""
    uses_ollama = name == "ollama" or planned.get("embedding") or planned.get("decision")
    if uses_ollama and (ollama.binary() is None or ollama.outdated()) and ollama.can_install():
        found.append({"what": f"Ollama {ollama.RELEASE}", "gb": RUNTIME_GB.get(ollama.build()[0])})
    mlx = MLXRuntime(config)
    local = {"ollama": ollama, "llama.cpp": LlamaCppRuntime(config, hw=hw), "prism": PrismRuntime(config, hw=hw),
             **({"mlx": mlx} if mlx.managed() else {})}
    if name in ("llama.cpp", "prism") and local[name].binary() is None:
        found.append({"what": "llama.cpp" if name == "llama.cpp" else "Prism's llama.cpp build", "gb": RUNTIME_GB[name]})
    elif name == "mlx" and mlx.managed() and mlx.python() is None:
        found.append({"what": "MLX runtime (mlx-lm)", "gb": RUNTIME_GB["mlx"]})
    for role, key in (("assistant model", "model"), ("embedding model (Ollama)", "embedding"),
                      ("decision model", "decision")):
        cand = planned.get(key)
        if not cand:
            continue
        rt = local.get(cand.get("runtime") or "ollama", ollama)
        try:
            entry = catalog.lookup(cand["id"], hw=hw, config=config, runtime=rt.name)
        except KeyError:
            found.append({"what": f"{role}: {cand['id']}", "gb": None})
            continue
        if not rt.installed(entry):
            found.append({"what": f"{role}: {cand['id']}", "gb": round(_download_gb(entry, cand, rt), 2) or None})
    if coreml.supported() and (config.get("models.embedding_runtime") or "auto") != "gpu":
        core = coreml.CoreMLRuntime(config)
        entry = next((e for e in catalog.by_role("embedding") if core.artifact(e)), None)
        if entry and not core.installed(entry):
            found.append({"what": f"embedding model (Core ML): {entry['id']}", "gb": core.artifact(entry).get("size_gb")})
        if entry and core.python() is None:
            found.append({"what": "Core ML packages", "gb": RUNTIME_GB["coreml"]})
    return found


def _embedding_download_gb(config, hw, embedder: Runtime | None) -> float:
    """Decimal GB the embedding models still need: the build in Ollama and, on Apple silicon, the Core ML one.
    An Ollama that is not running yet cannot say what it has, and counts as having nothing."""
    total = 0.0
    if embedder is not None and embedder.name == "ollama":
        pick = next((c for c in fit.recommend(hw, config, "embedding", "ollama") if c["license"] == "allow"), None)
        if pick and not embedder.installed(catalog.get(pick["id"])):
            total += _download_gb(catalog.get(pick["id"]), pick, embedder)
    from . import coreml
    if coreml.supported() and (config.get("models.embedding_runtime") or "auto") != "gpu":
        core = coreml.CoreMLRuntime(config)
        entry = next((e for e in catalog.by_role("embedding") if core.artifact(e)), None)
        if entry and not core.installed(entry):
            total += float(core.artifact(entry).get("size_gb") or 0)
    return total


def _prism(config, hw: hardware.Hardware, rt: Runtime | None) -> PrismRuntime | None:
    """Prism ML's llama.cpp build as a second runtime to rank models with, for ternary Bonsai models,
    when this platform has a build and the owner has not pinned another runtime."""
    if (rt is not None and rt.name == "prism") or _pinned(config) is not None:
        return None
    prism = PrismRuntime(config, hw=hw)
    return prism if prism.available() else None


def _chat_on_llama(hw) -> bool:
    """On Apple silicon every model runs on CodeRunner's own llama.cpp server, and embeddings on Core ML; Ollama
    is used only when the owner names it (--runtime ollama). Ollama's server keeps up to 8 GB of earlier prompts
    in memory, with no setting for it (2.4 GB in one sample-task run on an 18 GB Mac, which then wrote 0.06
    tokens/s), where CodeRunner's keeps none and sleeps when idle; and a first run fetched Ollama (0.17 GB) and
    a second copy of the embedding model through it (1.33 GB) that Core ML does not need. Elsewhere llama.cpp's
    official build runs on the CPU only, so Ollama serves the models."""
    return hw is not None and hw.gpu == "apple" and hw.arch == "arm64"


def _llama(config, rt: Runtime | None, hw=None) -> Runtime | None:
    """llama.cpp next to Ollama, for models Ollama cannot serve (tool formats it does not parse, experts in RAM)
    and, on Apple silicon, for chat models (setup installs llama.cpp's official build)."""
    if rt is None or rt.name != "ollama":
        return None
    llama = LlamaCppRuntime(config, hw=hw)
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
        repo_low = model["repo"].lower()
        # Curated entries carry measured speed factors; on large unified-memory Macs (>= 80 GiB budget),
        # still allow discovered higher-precision builds (Q6_K/Q8_0) of cataloged repos.
        if repo_low in known and hw.budget_gb < 80:
            continue
        entry, on = None, rt.name
        if rt.name == "ollama" and llama is not None and _chat_on_llama(hw):
            entry, on = discover.best_build(model, hw, config, "llama.cpp"), "llama.cpp"
        if entry is None:
            entry, on = discover.best_build(model, hw, config, rt.name), rt.name
        if entry is None and rt.name == "ollama":
            entry, on = discover.best_build(model, hw, config, "llama.cpp"), "llama.cpp"
        if entry and repo_low in known:
            art = catalog.artifact(entry, on) or {}
            if float(art.get("bits") or 0) < 6.0:
                continue
        if entry:
            entry["runtime"] = on
            # Worth installing llama.cpp for, where Ollama is the main runtime and llama.cpp is not beside it.
            entry["missing_runtime"] = on == "llama.cpp" and llama is None and rt.name != "llama.cpp"
            entries.append(entry)
    return discover.place(entries, catalog.models())


def _measured(hw) -> dict[tuple[str, str], dict]:
    """Sample-task results (tasks.py) for this kind of machine, by model and runtime: measured here, else
    shipped from an identical machine. A run in which the model slowed down counts only until one in which it
    did not: its times are slower than the machine's, but leaving it out made its model count as unmeasured,
    which ranks on estimates (a compact Bonsai 2 build then beat the measured one on paper after losing to it)."""
    from . import tasks
    machine = bench.fingerprint(hw)
    found: dict[tuple[str, str], dict] = {}
    for record in tasks.shared() + tasks.load():          # this machine's own results win
        if record.get("machine") == machine and record.get("suite") == tasks.SUITE:
            key = (record["model"], record["runtime"])
            if record.get("slowed") and key in found and not found[key].get("slowed"):
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
                        + (", the model slowed down: slower than this machine can be" if record.get("slowed") else "") + ")")
    return c


def _ranked(hw, config, rt: Runtime | None, prism: Runtime | None, extra: list[dict] | None = None,
            llama: Runtime | None = None) -> list[dict]:
    """Chat candidates for the main runtime, Prism's build and discovered models, best first."""
    found = fit.recommend(hw, config, "chat", _artifact_runtime(rt), include_rejected=True)
    if prism is not None:
        # Prism's fork only for what the main runtime cannot serve: Bonsai 2 27B runs on llama.cpp itself.
        served = {c["id"] for c in found if c["ok"]}
        found += [c for c in fit.recommend(hw, config, "chat", "prism") if c["id"] not in served]
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


def _entry(model_id: str, extra: list[dict], hw=None, config=None, runtime: str | None = None) -> dict:
    return catalog.lookup(model_id, extra, hw=hw, config=config, runtime=runtime)


@dataclasses.dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str = ""
    vision: bool = False
    embedding_model: str = ""


def _hardware(config, hw: hardware.Hardware | None) -> hardware.Hardware:
    hw = hw or hardware.detect(float(config.get("sandbox.memory_gb") or 1))
    from . import probe
    hw = probe.apply(hw, probe.load(hw))          # cached Tier 0 GPU/ANE probe (no-op if not run yet)
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
    if isinstance(rt, OllamaRuntime) and not rt.available() and rt.can_install():
        notes.append(f"No model runtime is installed here: setup fetches its own copy of Ollama {rt.RELEASE}.")
    elif isinstance(rt, LlamaCppRuntime) and not rt.available() and rt.can_install():
        pass                                # setup fetches llama.cpp's build; it is among the downloads
    elif rt is None or not rt.available():
        notes.append(install_hint(hw, rt.name if rt else None))
    if _cpu_note(hw, chosen):
        notes.append(_cpu_note(hw, chosen))
    if chosen is not None:
        room = _room_gb(config)[0]
        chosen_on = prism if chosen.get("runtime") == "prism" and prism is not None else \
            llama if chosen.get("runtime") == "llama.cpp" and llama is not None else rt
        local = _is_managed_local(chosen_on)       # an endpoint the owner runs has its model
        entry = _entry(chosen["id"], extra) if local else None
        need = _download_gb(entry, chosen, chosen_on) if local and not chosen_on.installed(entry) else 0.0
        if need > room:
            notes.append(f"{chosen['id']} is {_no_room(config, need)}: setup uses the best model that fits the "
                         "disk, so free some space first for this one.")
    if chosen is None:
        notes.append(f"No chat model of at least {fit.MIN_CHAT_PARAMS_B:.0f}B parameters fits the "
                     f"{hw.budget_gb:.1f} GB budget. Free memory, lower sandbox.memory_gb, or point "
                     "models.base_url at a bigger machine with models.runtime set to openai.")
    elif chosen["license"] == "ask":
        notes.append(f"{chosen['id']}: {chosen['license_reason']}")
    wired_hint = None
    if chosen and hw.gpu == "apple" and hw.ram_gb >= 48:
        # macOS lets the GPU wire about 75% of memory; on 48-512 GB Macs raising it unlocks bigger models.
        limit = hardware.safe_wired_limit_gb(hw.ram_gb, hw.sandbox_memory_gb)
        if hw.gpu_limit_gb < limit - 1.0:
            raised = dataclasses.replace(hw, gpu_limit_gb=limit,
                                         budget_gb=hardware.apple_budget(hw.ram_gb, limit, hw.sandbox_memory_gb))
            bigger = _discovered(config, raised, rt, llama=llama) if runtime is None else []    # builds that now fit
            better = max((c for c in _ranked(raised, config, rt, prism, bigger, llama)
                          if c["ok"] and c["step_s"] <= fit.MAX_STEP_S and c["ctx"] >= fit.MIN_AGENT_CTX
                          and fit.utility(c) > fit.utility(chosen) + WORTH_A_NOTE), key=fit.utility, default=None)
            if better:
                limit_mb = int(limit * 1024)
                cmd = f"sudo sysctl iogpu.wired_limit_mb={limit_mb}"
                wired_hint = {"limit_gb": round(limit, 1), "limit_mb": limit_mb, "command": cmd,
                              "model": better["id"], "model_name": pretty(better["id"]),
                              "need_gb": round(float(better["need_gb"]), 1)}
                notes.append(f"Raising the GPU memory limit allows {better['id']} ({better['need_gb']:.0f} GB): "
                             f"{cmd} (until restart)")
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
    judge_rt = "ollama" if ollama else ("llama.cpp" if chosen_rt and chosen_rt.name in ("llama.cpp", "prism") else None)
    judge = decision_pick(hw, config, chosen["need_gb"] + (embedder or {}).get("need_gb", 0.0), runtime=judge_rt) \
        if judge_rt and chosen else None
    driver = driver_pick(hw, config, chosen["need_gb"] + (embedder or {}).get("need_gb", 0.0)) if chosen else None
    return {
        "hardware": hw.to_dict(),
        "tier": hardware.ram_tier(hw.ram_gb) if hw.gpu == "apple" and hw.ram_gb else None,
        "runtime": chosen_rt.describe() if chosen_rt else None,
        "model": chosen,
        "presets": fit.presets(hw, viable),
        "alternatives": viable[1:6],
        "rejected": [c for c in candidates if not c["ok"]],
        "license": ({"decision": chosen["license"], "reason": chosen["license_reason"],
                     "link": chosen["license_link"]} if chosen else None),
        "embedding": embedder,
        "decision": judge,
        "driver": driver,
        "wired_limit_hint": wired_hint,
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
    repo = (entry["artifacts"].get("gguf") or {}).get("repo") if rt.name == "llama.cpp" else None
    if repo:            # a build named by its address has no repository of its own: the catalog's licence stands
        sources.append(repo)
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
                short = _no_room(config, _download_gb(entry, cand, rt))
                if short:
                    raise RuntimeError(short)
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


def setup_coreml_embedding(config, hw: hardware.Hardware | None = None, progress: Callable[[str], None] = print,
                           remeasure: bool = False, rt=None, measure=None, warm=None) -> str:
    """On Apple silicon the embedding model runs on the Neural Engine (coreml.py) unless this machine's GPU
    measured faster on text; the Core ML server embeds pictures and sound with the build's own encoders and passes
    them on to its build in Ollama when those are missing. `models.embedding_runtime` (auto | coreml | gpu)
    overrides the measurement.
    Returns the runtime that embeds text: "coreml" or "gpu"."""
    from . import coreml
    wanted = config.get("models.embedding_runtime") or "auto"
    media, media_url = config.get("models.embedding_media") or "", config.get("models.embedding_media_url") or ""
    rt = rt or coreml.CoreMLRuntime(config)
    if (config.get("models.embedding_url") or "").rstrip("/") != rt.base_url:     # a fresh setup: Ollama's build
        media = config.get("models.embedding") or ""
        # No build in Ollama (the default on Apple silicon): Core ML's own encoders read pictures and sound.
        media_url = (config.get("models.embedding_url") or config.get("models.base_url") or "") if media else ""
    entry = next((e for e in catalog.by_role("embedding") if rt.artifact(e)
                  and (not media or (e["artifacts"].get("ollama") or {}).get("tag") == media.split(":")[0])), None)
    if wanted == "gpu" or not coreml.supported() or entry is None or licenses.check(entry, config)[0] != "allow":
        return "gpu"
    hw = _hardware(config, hw)
    record = None if remeasure else coreml.load(hw)
    if wanted == "auto" and media and record and coreml.choice(record)[0] == "gpu":
        progress(f"embeddings stay on the GPU: {coreml.choice(record)[1]}")
        return "gpu"
    try:
        if not rt.installed(entry):
            short = _no_room(config, float(rt.artifact(entry).get("size_gb") or 0))
            if short:
                raise RuntimeError(short)
            rt.pull(entry, progress)
        elif not rt.available():
            rt.install(progress)
        config.set("models.embedding_media", media)
        config.set("models.embedding_media_url", media_url)
        rt.stop()                                   # started again with the media model it passes pictures to
        rt.start()
        progress("preparing the embedding model for the Neural Engine (once, about a minute)")
        _run_async((warm or coreml.warm)(rt.base_url, rt.model_id(entry)))
        if wanted == "auto" and record is None:
            if media:
                progress("timing embeddings on the Neural Engine against the GPU")
                record = coreml.save(hw, _run_async((measure or coreml.compare)(rt.base_url, rt.model_id(entry),
                                                                                   media_url, media)))
            elif measure is None:
                progress("timing embeddings on the Neural Engine")
                record = coreml.save(hw, _run_async(coreml.benchmark_local(rt.base_url, rt.model_id(entry))))
        pick, why = coreml.choice(record) if wanted == "auto" and record else \
            ("coreml", "set by models.embedding_runtime" if wanted == "coreml" else "no GPU build to compare with")
    except Exception as exc:
        progress(f"Core ML embeddings unavailable, keeping the GPU build: {exc}")
        rt.stop()
        config.set("models.embedding", media)
        config.set("models.embedding_url", media_url)
        config.set("models.embedding_media", "")
        config.set("models.embedding_media_url", "")
        return "gpu"
    progress(f"embeddings: {why}")
    if pick == "coreml":
        config.set("models.embedding", rt.model_id(entry))
        config.set("models.embedding_url", rt.base_url)
        return "coreml"
    rt.stop()
    rt.remove(entry)                                # the GPU build does everything here: free the disk
    progress(f"removed the Core ML build of {entry['id']}")
    config.set("models.embedding", media)
    config.set("models.embedding_url", media_url)
    config.set("models.embedding_media", "")
    config.set("models.embedding_media_url", "")
    return "gpu"


def setup_search(config, hw: hardware.Hardware | None = None, progress: Callable[[str], None] = print) -> dict:
    """Search mode: only the embedding model, for the file index and the search tools over MCP. On Apple silicon
    EmbeddingGemma 2 runs on the Neural Engine (Core ML); elsewhere Ollama serves it. No chat model, no sandbox."""
    from . import coreml
    config.set("models.chat", "")
    config.set("sandbox.driver", "none")
    config.set("assistant.mode", "search")
    if coreml.supported():
        config.set("models.embedding_runtime", config.get("models.embedding_runtime") or "coreml")
        where = setup_coreml_embedding(config, hw, progress)
        if where == "coreml":
            config.save()
            return {"ok": True, "mode": "search", "embedding": config.get("models.embedding"),
                    "embedding_url": config.get("models.embedding_url"), "runtime": "coreml"}
    rt = _runtime(config, _hardware(config, hw), "ollama")
    error = _own_ollama(rt, _hardware(config, hw), progress)
    if error or not rt.available():
        return {"ok": False, "error": "Search mode needs the Core ML runtime (Apple silicon, macOS 15+) or Ollama "
                                      f"for the embedding model. {error or ''} Install Ollama "
                                      "(https://ollama.com/download) and run `coderunner setup --search` again."}
    if not rt.running():
        rt.start()
    tag = _setup_embedding(config, rt, _hardware(config, hw), progress)
    if not tag:
        return {"ok": False, "error": "No embedding model could be set up in Ollama."}
    config.set("models.embedding", tag)
    config.set("models.embedding_url", rt.base_url)
    config.save()
    return {"ok": True, "mode": "search", "embedding": tag, "embedding_url": rt.base_url, "runtime": "ollama"}


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


def _memory_short(cand: dict, hw, rt: Runtime) -> tuple[float, float] | None:
    """(GB this model needs, GB free right now) when it would not fit in what is free at this moment; None when
    it fits with a margin, when a server elsewhere runs it, or when free memory cannot be read. Loading Bonsai 2
    27B (9.7 GB) on an 18 GB Mac took swap from 2.2 to 5.9 GB in 25 seconds with 6 GB free, and by 1.6 GB in
    seconds with 8.8 GB free; the plan had said it fits the machine."""
    if not _is_managed_local(rt):
        return None
    free = hardware.available_gb(getattr(hw, "ram_gb", 0) or 0)
    need = float(cand.get("need_gb") or 0)
    if free is None or not need or need + MEMORY_MARGIN_GB <= free:
        return None
    return need, free


def _free_gpu(config, keep: Runtime) -> None:
    """Unload the model the assistant used so far before another one loads: two models at once froze an 18 GB
    Mac twice. Ollama keeps its other models (embeddings); a llama.cpp server of another runtime stops (its own
    runtime restarts it with the new model)."""
    before, chat = config.get("models.runtime") or "", config.get("models.chat") or ""
    if not before or not chat:
        return
    old = choose_runtime(config, name=before)
    if not _is_managed_local(old) or not old.running():
        return
    if old.name == "ollama":
        old.unload(chat)
    elif old.name != keep.name:
        old.stop()


def _stopped(first: str, why: str, entry: dict | None = None, need: float = 0.0, attempts=(), **more) -> dict:
    """Setup's answer when the model it set out with did not work out and it may not go on to another by
    itself: what went wrong in a sentence, and the next choice for the owner to take or leave."""
    out = {"ok": False, "attempts": list(attempts), "error": f"{pretty(first)} could not be set up: {why}", **more}
    if entry is not None:
        out["next"] = {"model": entry["id"], "download_gb": round(need, 1)}
    return out


def setup(config, model_id: str | None = None, runtime: str | Runtime | None = None, accept_license: bool = False,
          progress: Callable[[str], None] = print, hw: hardware.Hardware | None = None, embedding: bool = True,
          smoke=smoketest.run, hub: Hub | None = None, measure=None, decision: bool = True,
          other_downloads: bool = True, insist: bool = True, ctx: int | None = None) -> dict:
    """measure times the model once it passes the smoke test (default bench.measure; False skips it);
    decision also sets up a decision model to judge browser steps when memory allows. With other_downloads
    off (the app: its first screen showed one model and its size), a model that does not work out is not
    replaced by one that has to be downloaded: setup stops and names it for the owner to choose. With insist
    off (the app again), a model named by model_id is still not loaded into memory that is not free."""
    hw = _hardware(config, hw)
    runtime = runtime if runtime is not None else _pinned(config)
    rt = _main_runtime(config, hw, runtime)
    if rt is None:
        return {"ok": False, "error": "No local model runtime is installed. " + install_hint(hw)}
    error = _own_ollama(rt, hw, progress)
    if error:
        return {"ok": False, "error": error}
    if not rt.available() and not (isinstance(rt, (LlamaCppRuntime, MLXRuntime)) and rt.can_install()):
        return {"ok": False, "error": f"{rt.name} is not available. {install_hint(hw, rt.name)}"}
    if not _is_managed_local(rt):
        return _external(config, rt, model_id, progress, smoke)
    if _cpu_note(hw):
        progress("warning: " + _cpu_note(hw))
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
        if ctx is not None and int(ctx) > 0:
            est = fit.estimate(entry, hw, ctx=int(ctx), runtime=on)
            cand = {**cand, "ctx": int(ctx), "need_gb": est["need_gb"]}
        if cand["fit"] == "no":
            progress(f"warning: {model_id} {cand['reasons'][0]}; trying anyway because it was requested")
        queue = [cand]  # an explicit choice is never silently replaced
    else:
        queue = [c for c in _ranked(hw, config, rt, prism, extra, llama) if c["ok"]]
        if not queue:
            return {"ok": False, "error": plan(config, hw, rt)["notes"][-1]}

    attempts = []
    main = rt
    tried, too_big, others = 0, [], None
    for index, cand in enumerate(queue):
        if tried > MAX_FALLBACKS:
            break
        if cand.get("runtime") == "prism" and prism is not None:
            rt = prism
        elif cand.get("runtime") == "llama.cpp" and llama is not None:
            rt = llama
        else:
            rt = main
        entry = _entry(cand["id"], extra)
        tried += 1
        if not cand.get("vision") and "vision" in entry.get("roles", []):
            entry = {**entry, "roles": [r for r in entry["roles"] if r != "vision"]}    # projector does not fit
        try:
            verdict, why, link = _license_gate(entry, rt, config, accept_license, hub)
        except HubError as exc:                 # Hugging Face out of reach, or limiting this connection
            attempts.append({"model": entry["id"], "error": str(exc)})
            progress(f"{entry['id']}: its licence could not be read: {exc}")
            continue
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
                # A download that would fill the disk is not started; the next model that fits is tried (a first
                # run on an 18 GB Mac fetches 11 GB, and the Mac this was measured on had 11.4 GB free).
                if others is None:
                    others = _embedding_download_gb(config, hw, main) if embedding else 0.0
                need = _download_gb(entry, cand, rt) + others
                short = _no_room(config, need)
                if not other_downloads and (attempts or too_big):
                    # An attempt that ran its checks and failed them carries no error of its own.
                    why = (attempts[-1].get("error") or "it did not pass its checks on this computer") if attempts \
                        else too_big[-1][1]
                    return _stopped(queue[0]["id"], why, None if short else entry, need, attempts)
                if short and not other_downloads:
                    room = max(_room_gb(config)[0], 0)
                    return {"ok": False, "attempts": attempts,
                            "disk_short": {"model": entry["id"], "need_gb": round(need, 1), "room_gb": round(room, 1)},
                            "error": f"Not enough disk space: {pretty(entry['id'])} needs {need:.1f} GB and "
                                     f"{room:.1f} GB can be used. Free {max(need - room, 0.1):.1f} GB and try again."}
                if short:
                    too_big.append((need, f"{entry['id']} with its embedding models is {short}"))
                    progress(f"skipping {entry['id']}: with its embedding models it is {short}")
                    tried -= 1                      # not an attempt: the next candidates still get theirs
                    continue
                progress(f"{entry['id']}: {need:.1f} GB to download with its embedding models, "
                         f"{_room_gb(config)[1]:.1f} GB free")
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
            short = _memory_short(cand, hw, rt)
            if short and model_id and insist:
                progress(f"warning: {entry['id']} needs {short[0]:.1f} GB of memory and {short[1]:.1f} GB is free "
                         "right now; loading it anyway because it was requested")
            elif short:
                # The model suits the machine but not this moment (other apps are open). Settling for a smaller
                # one for good because of what happens to be open is the wrong trade: say so, and let the owner
                # close something or choose the smaller model.
                need, free = short
                fits = next((c for c in queue[index + 1:]
                             if float(c.get("need_gb") or 0) + MEMORY_MARGIN_GB <= free), None)
                smaller, more = fits and fits["id"], {}
                if fits:                            # the choice the owner is offered, with what it would download
                    its = prism if fits.get("runtime") == "prism" and prism is not None else \
                        llama if fits.get("runtime") == "llama.cpp" and llama is not None else main
                    other = _entry(smaller, extra)
                    more["next"] = {"model": smaller,
                                    "download_gb": 0.0 if its.installed(other) else round(_download_gb(other, fits, its), 1)}
                progress(f"{entry['id']} needs {need:.1f} GB of memory and {free:.1f} GB is free right now; "
                         "its download is kept")
                return {"ok": False, "attempts": attempts,
                        "memory_short": {"model": entry["id"], "need_gb": round(need, 1), "free_gb": round(free, 1),
                                         "smaller": smaller}, **more,
                        "error": f"Not enough memory free: {pretty(entry['id'])} needs {need:.1f} GB and {free:.1f} GB "
                                 "is free. Close some apps and try again"
                                 + (f", or use {pretty(smaller)} (--model {smaller})." if smaller else ".")}
            if isinstance(rt, LlamaCppRuntime) and not isinstance(rt, PrismRuntime):
                use_no_mmap = bool(hw.gpu == "apple" and hw.ram_gb >= 64 and cand.get("fit") == "good")
                use_kv_q8 = bool(cand.get("ctx", 0) >= 65536 or cand.get("fit") == "tight")
                model = rt.prepare(entry, cand["ctx"], offload=bool(cand.get("offload")),
                                   kv_q8=use_kv_q8, no_mmap=use_no_mmap)
            else:
                model = rt.prepare(entry, cand["ctx"], offload=True) if cand.get("offload") else \
                    rt.prepare(entry, cand["ctx"])
            progress(f"loading {entry['id']} into memory")
            try:
                rt.start()
            except Exception as exc:
                # A server that does not come up says nothing about the model: it may be short of memory right
                # now, or have been given an argument it does not take (--no-mmap, once, on a build that calls
                # it --load-mode; setup then deleted 7.6 GB that had taken two hours to fetch). The weights stay.
                attempts.append({"model": entry["id"], "error": str(exc)})
                progress(f"{entry['id']} did not start: {exc}; its download is kept")
                continue
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
            setup_coreml_embedding(config, hw, progress)
        if decision:
            judge_rt = main if main is not None and main.name == "ollama" else rt
            _setup_decision(config, judge_rt, rt, hw, cand["need_gb"] + _embedding_need(config, hw), progress)
        config.save()
        if too_big:
            # The owner should know the assistant could be better here, and what it takes.
            need, better = too_big[0][0], too_big[0][1].split(" with its", 1)[0]
            progress(f"{better} ranks higher but the disk has no room for it: free "
                     f"{need - _room_gb(config)[0]:.1f} GB more, then run `coderunner models setup` to move to it")
        return {"ok": True, "runtime": rt.name, "base_url": rt.base_url, "model": model, "entry": entry["id"],
                "ctx": cand["ctx"], "vision": vision, "embedding": config.get("models.embedding") or "",
                "smoke": result, "attempts": attempts, "skipped_for_disk": [why for _, why in too_big]}
    if too_big and not any("smoke" in a for a in attempts):
        need, smallest = min(too_big)             # the least there is to free
        return {"ok": False, "attempts": attempts, "too_big": [why for _, why in too_big],
                "error": f"Not enough disk space: {smallest.split(' with its', 1)[0]} needs {need:.1f} GB and "
                         f"{max(_room_gb(config)[0], 0):.1f} GB can be used. Free "
                         f"{max(need - _room_gb(config)[0], 0.1):.1f} GB and run setup again."}
    if attempts and not any("smoke" in a for a in attempts):
        return {"ok": False, "error": f"No model could be set up: {attempts[-1].get('error')}", "attempts": attempts}
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
    tag = config.get("models.embedding_media") or config.get("models.embedding") or ""
    for cand in fit.recommend(hw, config, "embedding", "ollama"):
        if catalog.get(cand["id"])["artifacts"]["ollama"]["tag"] == tag:
            return cand["need_gb"]
    return 0.0


def decision_pick(hw: hardware.Hardware, config, used_gb: float, runtime: str | None = "ollama") -> dict | None:
    """The most accurate decision model that fits well beside the models already using `used_gb`.
    Uses Ollama systemone models when Ollama is active, or a dedicated llama.cpp GGUF judge on 24 GB+ Macs."""
    left = dataclasses.replace(hw, budget_gb=max(0.0, hw.budget_gb - used_gb))
    rt_name = runtime or "ollama"
    return next((c for c in fit.recommend(left, config, "decision", rt_name)
                 if c["fit"] == "good" and c["license"] == "allow"), None)


def _setup_decision(config, rt: Runtime | None, chat_rt: Runtime, hw, used_gb: float, progress) -> None:
    """A dedicated decision model judges browser steps when memory allows; otherwise the chat model does."""
    for key in ("models.decision", "models.decision_api", "models.decision_url"):
        config.set(key, "")
    if rt is None or rt.name not in ("ollama", "llama.cpp", "prism"):
        return
    judge_rt_name = "ollama" if rt.name == "ollama" else "llama.cpp"
    pick = decision_pick(hw, config, used_gb, runtime=judge_rt_name)
    if pick is None:
        progress("no decision model fits beside the chat model; the chat model judges browser steps")
        return
    entry = catalog.get(pick["id"])
    if judge_rt_name == "ollama":
        tag = entry["artifacts"]["ollama"]["tag"]
        try:
            if not rt.installed(entry):
                short = _no_room(config, _download_gb(entry, pick, rt))
                if short:
                    raise RuntimeError(short)
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
        return
    # Dedicated llama.cpp GGUF safety judge on Apple Silicon (24 GB+ Macs) without requiring Ollama.
    from .runtimes import JudgeRuntime
    jrt = JudgeRuntime(config)
    try:
        if not jrt.installed(entry):
            short = _no_room(config, _download_gb(entry, pick, jrt))
            if short:
                raise RuntimeError(short)
            jrt.pull(entry, progress)
        alias = jrt.prepare(entry, 4096)
        jrt.start()
        config.set("models.decision", alias)
        config.set("models.decision_api", "chat")
        config.set("models.decision_url", jrt.base_url)
        config.set("models.decision_threshold", entry["decision"]["threshold"])
        progress(f"decision model: {entry['id']} on llama.cpp ({entry['decision']['measured']})")
    except Exception as exc:
        progress(f"decision model {entry['id']} unavailable ({exc}); the chat model judges browser steps")


DRIVER_CTX = 16384     # a driver keeps its last three screenshots (about 1,300 tokens each) and its steps


def _chat_need(config, hw) -> float | None:
    """Memory the assistant's configured model needs (curated or discovered HF model)."""
    alias = config.get("models.chat") or ""
    model_id = alias.split(":")[0].removeprefix("coderunner-")
    runtime = config.get("models.runtime") or None
    rt_arg = None if runtime in ("auto", "openai") else runtime
    try:
        entry = catalog.lookup(model_id, hw=hw, config=config, runtime=rt_arg)
    except KeyError:
        return None
    return fit.estimate(entry, hw, ctx=int(config.get("models.context") or 32768), runtime=rt_arg)["need_gb"]


def driver_pick(hw: hardware.Hardware, config, used_gb: float | None) -> dict | None:
    """The browser driver for this machine. The best-scoring one that fits beside the assistant's model
    stays loaded with it. Where none does (18 GB Macs and below), the smallest one that fits on its own takes
    turns with it: each turn costs a load (Fara1.5-4B: 10-20 s on an M3 Pro) and the assistant's model
    reloading after, so the quickest to load wins there."""
    # Ranked on the published web-browsing score; a driver without one (Holo3.1) is used only when chosen.
    drivers = sorted((e for e in catalog.by_role("driver") if e["driver"].get("webvoyager") is not None),
                     key=lambda e: -e["driver"]["webvoyager"])
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


VOICE_CTX = VoiceRuntime.VOICE_CTX
VOICE_MODEL = "lfm3-omni-2b"


def setup_voice_omni(config, progress: Callable[[str], None] = print, hub: Hub | None = None,
                     *, accept_license: bool = False, off: bool = False) -> dict:
    """Download and configure LFM3-Omni as a llama.cpp voice sidecar, or remove it."""
    if off:
        rt = VoiceRuntime(config, hub=hub)
        try:
            entry = catalog.get(config.get("voice.omni_model") or VOICE_MODEL)
            if rt.installed(entry):
                rt.remove(entry)
        except KeyError:
            pass
        rt.stop()
        config.set("voice.omni_model", "")
        config.set("voice.omni_exclusive", False)
        config.save()
        return {"ok": True, "omni": None}
    entry = catalog.get(VOICE_MODEL)
    verdict, reason, link = licenses.check(entry, config, hub=hub)
    if verdict == "ask":
        if not accept_license:
            return {"ok": False, "error": reason, "license": "ask", "link": link,
                    "license_name": (entry.get("license") or {}).get("name") or "LFM Open License v1.0"}
        key = licenses.key((entry.get("license") or {}).get("spdx"),
                           (entry.get("license") or {}).get("name"))
        accepted = list(config.get("models.accepted_licenses") or [])
        if key and key not in {str(x).lower() for x in accepted}:
            accepted.append(key)
            config.set("models.accepted_licenses", accepted)
    elif verdict == "block":
        return {"ok": False, "error": reason, "license": "block", "link": link}
    hw = _hardware(config, None)
    used = _chat_need(config, hw)
    # Remote chat (Claude / owner's server) leaves the local GPU free for voice.
    rem = config.get("models.remote") or {}
    if rem.get("provider"):
        used = 0.0
    need = fit.estimate(entry, hw, ctx=VOICE_CTX, runtime="voice")["need_gb"]
    exclusive = used is None or used + need > hw.budget_gb
    runtime = VoiceRuntime(config, hub=hub, hw=hw)
    if not runtime.available():
        runtime.install(progress)
    if not runtime.installed(entry):
        runtime.pull(entry, progress)
    alias = runtime.prepare(entry, VOICE_CTX)
    record = (runtime._state().get("models") or {}).get(entry["id"]) or {}
    if not record.get("mmproj"):
        return {"ok": False, "error": "LFM3-Omni needs its audio projector; the download was incomplete."}
    config.set("voice.omni_model", entry["id"])
    config.set("voice.omni_exclusive", exclusive)
    config.save()
    progress(f"voice: {entry['id']}" + (" (takes turns with the assistant's model)" if exclusive else ""))
    return {"ok": True, "omni": entry["id"], "alias": alias, "base_url": runtime.base_url,
            "exclusive": exclusive, "need_gb": need}


def voice_omni_status(config) -> dict:
    """Readiness of the LFM3-Omni sidecar for the app and Settings."""
    model = config.get("voice.omni_model") or ""
    rt = VoiceRuntime(config)
    ready = bool(model) and rt.ready()
    exclusive = bool(config.get("voice.omni_exclusive"))
    return {
        "model": model or None,
        "ready": ready,
        "running": rt.running() if ready else False,
        "exclusive": exclusive,
        "speak_tasks": bool(config.get("voice.speak_tasks")),
        "voice": config.get("voice.voice") or "female",
        "language": config.get("voice.language") or "",
        "label": ("Ready. Speaks when the assistant is idle." if exclusive else "Ready") if ready else "Not installed",
    }


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
    try:
        entry = catalog.lookup(entry_id, hw=hw, config=config,
                               runtime=runtime if runtime in catalog.ARTIFACT else None)
    except KeyError:
        entry = None
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
        entry = _entry(cand["id"], [], hw=hw, config=config, runtime=cand.get("runtime"))
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
        runs.append({"model": cand["id"], **{k: result.get(k) for k in ("passed", "total", "seconds", "slowed", "error")}})
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
    if not (_is_managed_local(rt) and rt.base_url.rstrip("/") == endpoint(config).base_url):
        return False
    rt.stop()
    rt.start()
    return rt.running()


def restart_decision(config) -> bool:
    """Restart a managed local dedicated JudgeRuntime after a crash or reboot; False for shared/external judges."""
    if not config.get("models.decision"):
        return False
    jrt = JudgeRuntime(config)
    if (config.get("models.decision_url") or "").rstrip("/") != jrt.base_url.rstrip("/"):
        return False
    if not (jrt._state().get("active") or {}).get("model"):
        return False
    jrt.stop()
    jrt.start()
    return jrt.running()


def restart_embedding(config) -> bool:
    """Start the Core ML embedding server again when it is down (it was found killed, likely under memory
    pressure, and file indexing then failed for a day); False for runtimes CodeRunner does not run itself."""
    from .coreml import CoreMLRuntime
    rt = CoreMLRuntime(config)
    if (config.get("models.embedding_url") or "").rstrip("/") != rt.base_url:
        return False
    if not rt.running():
        rt.start()
    return rt.running()


def ensure_running(config) -> Endpoint:
    """Start the configured local runtime if it is down; external endpoints are left alone."""
    # Before setup has run there is no model to serve: starting whichever runtime happens to be installed (an
    # Ollama the owner has for something else, on a Mac that will use llama.cpp) only takes its port and memory.
    set_up = bool(config.get("models.chat") or config.get("models.embedding") or config.get("models.base_url"))
    rt = choose_runtime(config) if set_up else None
    ep = endpoint(config)
    from . import remote
    # In remote mode the planner is not on this computer: its local model stays set up but is not loaded.
    remote_on = remote.active(config) is not None
    ours = _is_managed_local(rt) and rt.base_url.rstrip("/") == ep.base_url
    if ours and remote_on:
        # Its memory goes back to the owner's apps: the model was still held (6.7 GB) after the switch.
        if rt.running() and config.get("models.chat"):
            rt.unload(config.get("models.chat")) if rt.name == "ollama" else rt.stop()
    elif ours and not rt.running():
        rt.start()
    if config.get("models.decision"):
        jrt = JudgeRuntime(config)
        if (config.get("models.decision_url") or "").rstrip("/") == jrt.base_url.rstrip("/"):
            if remote_on:
                if jrt.running():
                    jrt.stop()
            elif not jrt.running() and (jrt._state().get("active") or {}).get("model"):
                try:
                    jrt.start()
                except Exception:
                    pass
    from .coreml import CoreMLRuntime
    embedder = CoreMLRuntime(config)
    if (config.get("models.embedding_url") or "").rstrip("/") == embedder.base_url and not embedder.running():
        embedder.start()            # it loads the model on the first request and releases it when idle
    return ep
