"""Memory and speed estimates, and ranked model recommendations.

need = weights + projector + KV cache + ~1 GiB of runtime and compute buffers.
A fit is "good" within 85% of the budget and "tight" within 100%. Context
shrinks (32K -> 16K) before a lower-ranked model is considered; 8K only when
nothing fits at 16K, because browser pages alone fill most of it.
"""

from __future__ import annotations

from . import catalog, licenses
from .hardware import GiB, Hardware, cpu_budget

GB = 1e9 / GiB                  # decimal GB (artifact sizes) -> GiB (budgets)
OVERHEAD_GIB = 1.0              # runtime buffers (0.5) + compute graph (0.5)
GOOD = 0.85
CONTEXTS = (32768, 16384, 8192)
MIN_CHAT_PARAMS_B = 4.0         # smaller models produced wrong tool arguments in testing
MIN_TOK_S = 8.0                 # estimated; slower fits rank below faster ones (agent loops are long)
MIN_AGENT_CTX = 16384           # smaller contexts rank below larger ones
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
    active_gb = weights * float(entry.get("active_params_b") or params) / params
    # Bandwidth-bound estimate, scaled by a factor measured for builds whose kernels fall short of it.
    tok_s = 0.55 * hw.bandwidth_gbs / active_gb * art.get("tok_s_factor", 1.0) if active_gb else 0.0
    result = {"need_gb": round(need, 2), "fit": fit, "tok_s": round(tok_s, 1), "ctx": ctx}
    if fit == "no" and _can_offload(entry, hw, runtime):
        # llama.cpp --cpu-moe: attention and shared weights (~10%) on the GPU, experts in system RAM.
        gpu_need = (0.1 * weights + mmproj) * GB + kv + OVERHEAD_GIB
        ram_need = 0.9 * weights * GB
        ram_budget = cpu_budget(hw.ram_gb)
        if gpu_need <= hw.budget_gb and ram_need <= ram_budget:
            fit = "good" if gpu_need <= GOOD * hw.budget_gb and ram_need <= GOOD * ram_budget else "tight"
            dense, experts = 0.1 * weights, max(active_gb - 0.1 * weights, 0.5 * active_gb)
            tok_s = 0.55 / (dense / hw.bandwidth_gbs + experts / hw.ram_bandwidth_gbs)
            result = {"need_gb": round(gpu_need, 2), "ram_gb": round(ram_need, 2), "fit": fit,
                      "tok_s": round(tok_s, 1), "ctx": ctx, "offload": True}
    return result


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
    reasons.append(f"~{est['tok_s']:.0f} tok/s")
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
    return {"id": entry["id"], "family": entry["family"], "rank": entry["rank"], "params_b": entry["params_b"],
            "runtime": runtime, "vision": vision, **est, "license": decision, "license_reason": why,
            "license_link": link, "ok": ok, "reasons": reasons}


def _order(c: dict) -> tuple:
    return (c["fit"] != "good", c["ctx"] < MIN_AGENT_CTX, c["tok_s"] < MIN_TOK_S, c["license"] != "allow",
            c["rank"], not c["vision"])


def recommend(hw: Hardware, config, role: str = "chat", runtime: str | None = None,
              include_rejected: bool = False) -> list[dict]:
    """Ranked candidates for a role. Viable ones come first; rejected ones only on request."""
    found = [candidate(e, hw, config, role, runtime) for e in catalog.by_role(role)]
    ranked = sorted((c for c in found if c["ok"]), key=_order)
    if include_rejected:
        ranked += sorted((c for c in found if not c["ok"]), key=lambda c: c["rank"])
    return ranked
