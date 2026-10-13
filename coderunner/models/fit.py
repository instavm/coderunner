"""Memory and speed estimates, and ranked model recommendations.

need = weights + projector + KV cache + ~1 GiB of runtime and compute buffers.
A fit is "good" within 85% of the budget and "tight" within 100%. Context
shrinks (32K -> 16K) before a lower-ranked model is considered; 8K only when
nothing fits at 16K, because browser pages alone fill most of it.
"""

from __future__ import annotations

import math

from . import catalog, evidence, licenses
from .hardware import READ_PER_BANDWIDTH, GiB, Hardware, cpu_budget

GB = 1e9 / GiB                  # decimal GB (artifact sizes) -> GiB (budgets)
OVERHEAD_GIB = 1.0              # runtime buffers (0.5) + compute graph (0.5)
GOOD = 0.85
CONTEXTS = (32768, 16384, 8192)
MIN_CHAT_PARAMS_B = 4.0         # smaller models produced wrong tool arguments in testing
MIN_TOK_S = 8.0                 # generation speed below which a model is called slow
MIN_AGENT_CTX = 16384           # smaller contexts rank below larger ones
# An agent step reads what changed (a page, a tool result) and writes a short tool call; with the
# prompt cache, earlier turns are not read again. Its time is what an owner waits for, so it ranks
# models, not generation speed alone. Measured in the sample tasks (tasks.py) with Bonsai 2 27B on an
# M3 Pro: 580 new tokens read and 76-136 written per model request, plus 2.6 s for the browser action
# and the commit judge.
STEP_READ_TOKENS = 600
STEP_WRITE_TOKENS = 100
STEP_OVERHEAD_S = 2.5
MAX_STEP_S = 90.0               # slower fits rank below faster ones (agent loops are long)
# The balance rule, the same on every machine: ability from published benchmarks (evidence.py), less what
# waiting costs. A step within the budget (models.step_budget_s) costs nothing; each doubling beyond it
# costs SLOW_COST of ability, about what separates Qwen3.5-9B from a 27B model at 98% of its scores. Steps
# are estimated from the hardware until sample tasks have measured them here (tasks.py, manager._measured).
STEP_BUDGET_S = 15.0
SLOW_COST = 0.25
VISION_BONUS = 0.1
UNKNOWN_ABILITY = -3.0          # no published scores: after every model that has some
EMBED_CTX = 8192


def kv_gib(entry: dict, ctx: int) -> float:
    arch = entry.get("arch") or {}
    if not arch.get("full_attn_layers"):
        # Unknown architecture: assume a Llama-8B-like 128 KiB/token, growing with size.
        return ctx * 131072 * max(1.0, (entry.get("params_b", 8) / 8) ** 0.5) / GiB
    full = 2 * arch["full_attn_layers"] * arch["kv_heads"] * arch["head_dim"] * ctx * 2
    swa = 2 * arch.get("swa_layers", 0) * arch.get("swa_kv_heads", 0) * arch.get("swa_head_dim", 0) \
        * min(ctx, arch.get("swa_window", ctx)) * 2
    return (full + swa) / GiB


def estimate(entry: dict, hw: Hardware, ctx: int = 32768, runtime: str | None = None) -> dict:
    art = catalog.artifact(entry, runtime) or {}
    weights = art.get("size_gb", 0.0)
    mmproj = art.get("mmproj_gb", 0.0) if catalog.has_vision(entry, runtime) else 0.0
    kv = kv_gib(entry, ctx) * art.get("kv_bytes", 2) / 2     # some runtimes keep the cache at 8 bits
    need = (weights + mmproj) * GB + kv + OVERHEAD_GIB
    fit = "good" if need <= GOOD * hw.budget_gb else "tight" if need <= hw.budget_gb else "no"
    params = max(float(entry.get("params_b") or 0), 1e-6)
    active = float(entry.get("active_params_b") or params)
    active_gb = weights * active / params
    # Bandwidth-bound estimate, scaled by a factor measured for builds whose kernels fall short of it,
    # and by what this machine measured against the estimate (bench.py).
    tok_s = 0.55 * hw.bandwidth_gbs / active_gb * art.get("tok_s_factor", 1.0) * hw.gen_factor if active_gb else 0.0
    read_k = hw.read_k or READ_PER_BANDWIDTH.get(hw.gpu, READ_PER_BANDWIDTH["none"]) * hw.bandwidth_gbs
    read_tok_s = read_k / max(active, 0.1) * art.get("read_factor", 1.0)
    result = {"need_gb": round(need, 2), "fit": fit, "tok_s": round(tok_s, 1), "ctx": ctx}
    if fit == "no" and _can_offload(entry, hw, runtime):
        # llama.cpp --cpu-moe: attention and shared weights (~10%) on the GPU, experts in system RAM.
        gpu_need = (0.1 * weights + mmproj) * GB + kv + OVERHEAD_GIB
        ram_need = 0.9 * weights * GB
        ram_budget = cpu_budget(hw.ram_gb)
        if gpu_need <= hw.budget_gb and ram_need <= ram_budget:
            fit = "good" if gpu_need <= GOOD * hw.budget_gb and ram_need <= GOOD * ram_budget else "tight"
            dense, experts = 0.1 * weights, max(active_gb - 0.1 * weights, 0.5 * active_gb)
            tok_s = 0.55 / (dense / hw.bandwidth_gbs + experts / hw.ram_bandwidth_gbs) * hw.gen_factor
            read_tok_s /= 2          # experts are copied to the GPU for each batch of prompt tokens
            result = {"need_gb": round(gpu_need, 2), "ram_gb": round(ram_need, 2), "fit": fit,
                      "tok_s": round(tok_s, 1), "ctx": ctx, "offload": True}
    result["read_tok_s"] = round(read_tok_s, 1)
    result["step_s"] = round(step_seconds(read_tok_s, tok_s), 1)
    return result


def step_seconds(read_tok_s: float, tok_s: float) -> float:
    """Seconds for one agent step: read what changed, write a tool call, carry it out."""
    if read_tok_s <= 0 or tok_s <= 0:
        return math.inf
    return STEP_READ_TOKENS / read_tok_s + STEP_WRITE_TOKENS / tok_s + STEP_OVERHEAD_S


def _can_offload(entry: dict, hw: Hardware, runtime: str | None) -> bool:
    """Mixture-of-experts models on a discrete NVIDIA GPU under llama.cpp, with system RAM to spare."""
    params = float(entry.get("params_b") or 0)
    active = float(entry.get("active_params_b") or params)
    return (runtime == "llama.cpp" and hw.gpu == "nvidia" and not hw.unified and params > 0
            and active < 0.35 * params and hw.ram_gb > hw.vram_gb)


def best_context(entry: dict, hw: Hardware, target: int = 32768, runtime: str | None = None,
                 role: str = "chat") -> dict:
    """Largest context that fits well, else the largest that fits at all."""
    if role != "chat":
        return estimate(entry, hw, min(EMBED_CTX, entry.get("ctx_max", EMBED_CTX)), runtime)
    contexts = [c for c in dict.fromkeys((target, *CONTEXTS)) if c <= min(target, entry.get("ctx_max", c))]
    estimates = [estimate(entry, hw, c, runtime) for c in contexts or [8192]]
    for wanted in ("good", "tight"):
        for est in estimates:
            if est["fit"] == wanted:
                return est
    return estimates[-1]


def candidate(entry: dict, hw: Hardware, config, role: str = "chat", runtime: str | None = None) -> dict:
    est = best_context(entry, hw, int(config.get("models.context") or 32768), runtime, role)
    decision, why, link = licenses.check(entry, config)
    vision = catalog.has_vision(entry, runtime)
    reasons = []
    if est["fit"] == "no" and vision:
        # Without its vision projector the model may still fit (8 GB machines): text only, but usable.
        text = best_context({**entry, "roles": [r for r in entry["roles"] if r != "vision"]}, hw,
                            int(config.get("models.context") or 32768), runtime, role)
        if text["fit"] != "no":
            est, vision = text, False
            reasons.append("text only: the vision projector does not fit")
    if est["fit"] == "no":
        reasons.append(f"needs {est['need_gb']:.1f} GB but the budget is {hw.budget_gb:.1f} GB")
    else:
        reasons.append(f"{'fits' if est['fit'] == 'good' else 'tight fit'}: {est['need_gb']:.1f} of "
                       f"{hw.budget_gb:.1f} GB at {est['ctx'] // 1024}K context")
        if est.get("offload"):
            reasons.append(f"experts in system RAM ({est['ram_gb']:.0f} GB)")
    reasons.append(f"~{est['tok_s']:.0f} tok/s, reads ~{est['read_tok_s']:.0f} tok/s, "
                   f"~{est['step_s']:.0f} s per agent step{' (measured here)' if hw.measured else ''}")
    if role == "chat":
        reasons.append("vision" if vision else "text only")
    if entry.get("notes"):
        reasons.append(entry["notes"])
    reasons.append(why)
    ok = est["fit"] != "no" and decision != "block" and catalog.artifact(entry, runtime) is not None
    if runtime and catalog.artifact(entry, runtime) is None:
        reasons.insert(0, f"no {runtime} build")
    if role == "chat" and entry["params_b"] < MIN_CHAT_PARAMS_B:
        ok = False
        reasons.insert(0, f"below {MIN_CHAT_PARAMS_B:.0f}B parameters")
    ability = None
    if role == "chat":
        art = catalog.artifact(entry, runtime) or {}
        bits = art.get("bits") or (8 * art["size_gb"] / entry["params_b"] if art.get("size_gb") else None)
        ability = evidence.ability(entry, bits)
        reasons.append("no published benchmark scores" if ability is None else f"benchmark ability {ability:+.2f}")
    return {"id": entry["id"], "family": entry["family"], "rank": entry["rank"], "params_b": entry["params_b"],
            "runtime": runtime, "vision": vision, **est, "budget_s": float(config.get("models.step_budget_s") or STEP_BUDGET_S), "license": decision, "license_reason": why,
            "license_link": link, "ok": ok, "ability": ability, "reasons": reasons}


def utility(c: dict) -> float:
    """Benchmark ability, less what slow steps cost, plus what vision is worth to an assistant."""
    ability = UNKNOWN_ABILITY if c.get("ability") is None else c["ability"]
    budget = c.get("budget_s") or STEP_BUDGET_S
    step = min(c.get("step_s", math.inf), 64 * budget)
    return ability - SLOW_COST * max(0.0, math.log2(step / budget)) + (VISION_BONUS if c.get("vision") else 0.0)


def passed_share(c: dict) -> float:
    """Share of the sample tasks passed on this machine; a model not measured yet counts as passing them all,
    so it is measured before a weaker one is chosen over it."""
    done = c.get("tasks")
    return done["passed"] / done["total"] if done and done.get("total") else 1.0


def _order(c: dict) -> tuple:
    # Tasks passed come first. Utilities within the same tenth are a tie (estimates are not finer than that),
    # which the faster model wins: an owner waits for every step.
    ability = UNKNOWN_ABILITY if c.get("ability") is None else c["ability"]
    return (c["fit"] != "good", c["ctx"] < MIN_AGENT_CTX, c.get("step_s", math.inf) > MAX_STEP_S,
            c["license"] != "allow", -passed_share(c), -round(utility(c), 1), round(c.get("step_s", math.inf)),
            -ability, c["rank"])


def _order_by_rank(c: dict) -> tuple:
    return (c["fit"] != "good", c["license"] != "allow", c["rank"], not c["vision"])


def recommend(hw: Hardware, config, role: str = "chat", runtime: str | None = None,
              include_rejected: bool = False) -> list[dict]:
    """Ranked candidates for a role. Viable ones come first; rejected ones only on request."""
    found = [candidate(e, hw, config, role, runtime) for e in catalog.by_role(role)]
    ranked = sorted((c for c in found if c["ok"]), key=_order if role == "chat" else _order_by_rank)
    if include_rejected:
        ranked += sorted((c for c in found if not c["ok"]), key=lambda c: c["rank"])
    return ranked
