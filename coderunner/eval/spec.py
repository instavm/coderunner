"""Declarative StackSpec for single models and multi-model combinations across every Mac RAM tier."""

from __future__ import annotations

import dataclasses
from typing import Any

from ..models import catalog, fit, hardware, manager


@dataclasses.dataclass
class StackSpec:
    """One complete assistant configuration: single model or multi-model specialist combination."""

    id: str
    label: str
    planner: str
    runtime: str = "llama.cpp"
    ctx: int = 32768
    escalator: str = ""                 # optional deep model for Fast+Deep combos (64-512 GB Macs)
    driver: str = ""                    # optional screenshot browser specialist (e.g. fara1.5-4b, fara1.5-9b)
    driver_mode: str = "exclusive"      # exclusive | co_resident | off
    decision: str = "self"              # self | judge-4b | tev1-4b
    embedding: str = "coreml"           # coreml | gpu | none
    voice: str = "auto"                 # auto | parakeet-coreml | apple | qwen3-asr | none
    sandbox_memory_gb: float = 2.0
    remote_provider: str = ""           # "" (local) | openai (your server) | anthropic (Claude)
    base_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StackSpec":
        fields = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in fields}
        if not clean.get("id"):
            parts = [clean.get("planner") or "local"]
            if clean.get("driver"):
                parts.append(clean["driver"])
            if clean.get("escalator"):
                parts.append(clean["escalator"])
            clean["id"] = "+".join(parts)
        if not clean.get("label"):
            clean["label"] = manager.pretty(clean.get("planner") or "local") + (
                f" + {manager.pretty(clean['driver'])}" if clean.get("driver") else ""
            )
        return cls(**clean)

    def signature(self) -> str:
        return "|".join([
            self.remote_provider or self.runtime,
            self.planner,
            str(self.ctx),
            self.escalator or "-",
            f"{self.driver}:{self.driver_mode}" if self.driver else "-",
            self.decision or "self",
            self.embedding or "none",
        ])


def estimate_stack_memory(spec: StackSpec, hw: hardware.Hardware, config=None) -> dict[str, Any]:
    """Estimate combined resident memory (GiB), co-residency, and comfort rating for a StackSpec."""
    if spec.remote_provider:
        return {"total_gb": 0.0, "planner_gb": 0.0, "driver_gb": 0.0, "judge_gb": 0.0, "escalator_gb": 0.0,
                "fits": True, "driver_exclusive": False, "comfort": "remote"}
    rt = spec.runtime if spec.runtime in catalog.ARTIFACT else "llama.cpp"

    def model_gb(model_id: str, ctx: int, role_rt: str) -> float:
        if not model_id or model_id == "self":
            return 0.0
        try:
            entry = catalog.lookup(model_id, hw=hw, config=config, runtime=role_rt)
        except KeyError:
            return 0.0
        return float(fit.estimate(entry, hw, ctx=ctx, runtime=role_rt).get("need_gb") or 0.0)

    planner_gb = model_gb(spec.planner, spec.ctx, rt)
    driver_gb = model_gb(spec.driver, manager.DRIVER_CTX, "driver") if spec.driver and spec.driver_mode != "off" else 0.0
    judge_gb = model_gb(spec.decision, 4096, "ollama" if spec.decision == "tev1-4b" else "llama.cpp") \
        if spec.decision and spec.decision != "self" else 0.0
    esc_gb = model_gb(spec.escalator, spec.ctx, rt) if spec.escalator else 0.0

    driver_exclusive = spec.driver_mode == "exclusive" or (
        bool(driver_gb) and planner_gb + judge_gb + esc_gb + driver_gb > hw.budget_gb
    )
    resident = planner_gb + judge_gb + esc_gb + (0.0 if driver_exclusive else driver_gb)
    peak = max(resident, driver_gb + judge_gb)
    fits = peak <= hw.budget_gb and planner_gb > 0
    ratio = peak / max(hw.budget_gb, 0.1)
    comfort = "comfortable" if ratio <= 0.80 else "tight" if ratio <= 1.0 else "over_budget"
    return {
        "total_gb": round(peak, 2),
        "planner_gb": round(planner_gb, 2),
        "driver_gb": round(driver_gb, 2),
        "judge_gb": round(judge_gb, 2),
        "escalator_gb": round(esc_gb, 2),
        "budget_gb": round(hw.budget_gb, 2),
        "fits": fits,
        "driver_exclusive": bool(driver_gb and driver_exclusive),
        "comfort": comfort,
    }


def stack_from_config(config, hw: hardware.Hardware | None = None) -> StackSpec:
    """Build a StackSpec reflecting the currently configured CodeRunner install."""
    hw = hw or hardware.detect(float(config.get("sandbox.memory_gb") or 2))
    remote = config.get("models.remote") or {}
    if remote.get("provider"):
        model = remote.get("model") or config.get("models.chat") or ""
        prov = str(remote["provider"])
        suffix = "Claude" if prov == "anthropic" else "your server"
        return StackSpec(
            id=f"{prov}:{model}",
            label=f"{manager.pretty(model)} ({suffix})",
            planner=model,
            runtime=prov,
            ctx=int(config.get("models.context") or 32768),
            embedding="coreml" if config.get("models.embedding_runtime") != "gpu" else "gpu",
            remote_provider=prov,
            base_url=str(remote.get("base_url") or ""),
        )
    chat = manager.entry_id(config)
    drv = config.get("models.driver") or {}
    drv_model = str(drv.get("model") or "") if isinstance(drv, dict) else ""
    drv_mode = ("exclusive" if drv.get("exclusive") else "co_resident") if drv_model else "off"
    dec = str(config.get("models.decision") or "").split(":")[0].removeprefix("coderunner-") or "self"
    label = manager.pretty(chat) if chat else "Local model"
    if drv_model:
        label += f" + {manager.pretty(drv_model)}"
    return StackSpec(
        id=f"{chat}+{drv_model}" if drv_model else (chat or "local"),
        label=label,
        planner=chat,
        runtime=str(config.get("models.runtime") or "llama.cpp"),
        ctx=int(config.get("models.context") or 32768),
        driver=drv_model,
        driver_mode=drv_mode,
        decision=dec,
        embedding="coreml" if config.get("models.embedding_runtime") != "gpu" else "gpu",
        sandbox_memory_gb=float(config.get("sandbox.memory_gb") or 2),
    )


def default_candidates(config, hw: hardware.Hardware | None = None) -> list[StackSpec]:
    """Generate sensible single-model and multi-model comparison stacks tailored to this Mac's RAM tier."""
    hw = manager._hardware(config, hw)
    planned = manager.plan(config, hw=hw)
    tier = planned.get("tier") or hardware.ram_tier(hw.ram_gb or 16)
    viable = [c for c in (planned.get("model"), *(planned.get("alternatives") or [])) if c]
    stacks: list[StackSpec] = []
    seen: set[str] = set()

    def add(spec: StackSpec) -> None:
        sig = spec.signature()
        if sig not in seen and spec.planner:
            seen.add(sig)
            stacks.append(spec)

    if config.get("models.chat") or (config.get("models.remote") or {}).get("provider"):
        add(stack_from_config(config, hw))

    for preset in planned.get("presets") or []:
        cand = next((c for c in viable if c["id"] == preset["model"]), None)
        if cand:
            add(StackSpec(
                id=cand["id"],
                label=f"{manager.pretty(cand['id'])} ({preset['label']})",
                planner=cand["id"],
                runtime=cand.get("runtime") or "llama.cpp",
                ctx=int(cand.get("ctx") or tier["target_ctx"]),
                decision="judge-4b" if tier.get("dedicated_judge") and hw.ram_gb >= 32 else "self",
                embedding="coreml" if hw.gpu == "apple" else "gpu",
                sandbox_memory_gb=float(tier.get("sandbox_gb") or 2.0),
            ))

    # Multi-model specialist combo: Planner + Fara1.5 browser driver (exclusive on 16-18 GB, co-resident on 24 GB+).
    if viable and hw.ram_gb >= 16:
        top = viable[0]
        driver_id = "fara1.5-9b" if hw.ram_gb >= 48 else "fara1.5-4b"
        driver_mode = "co_resident" if hw.ram_gb >= 24 else "exclusive"
        combo = StackSpec(
            id=f"{top['id']}+{driver_id}",
            label=f"{manager.pretty(top['id'])} + {manager.pretty(driver_id)}",
            planner=top["id"],
            runtime=top.get("runtime") or "llama.cpp",
            ctx=int(top.get("ctx") or tier["target_ctx"]),
            driver=driver_id,
            driver_mode=driver_mode,
            decision="judge-4b" if hw.ram_gb >= 48 else "self",
            embedding="coreml" if hw.gpu == "apple" else "gpu",
            sandbox_memory_gb=float(tier.get("sandbox_gb") or 2.0),
        )
        if estimate_stack_memory(combo, hw, config)["fits"]:
            add(combo)

    # Larger single-model comparison on 64 GB+ Macs (fast MoE vs deeper planner head-to-head).
    if tier.get("dual_resident") and len(viable) >= 2:
        presets_by_key = {p["key"]: p["model"] for p in planned.get("presets") or []}
        smart_id = presets_by_key.get("smartest") or viable[1]["id"]
        smart_cand = next((c for c in viable if c["id"] == smart_id), viable[1])
        if smart_cand:
            deep = StackSpec(
                id=f"{smart_cand['id']}+fara1.5-9b" if hw.ram_gb >= 96 else smart_cand["id"],
                label=(f"{manager.pretty(smart_cand['id'])} + {manager.pretty('fara1.5-9b')}"
                       if hw.ram_gb >= 96 else f"{manager.pretty(smart_cand['id'])} (Deep)"),
                planner=smart_cand["id"],
                runtime=smart_cand.get("runtime") or "llama.cpp",
                ctx=int(smart_cand.get("ctx") or tier["target_ctx"]),
                driver="fara1.5-9b" if hw.ram_gb >= 96 else "",
                driver_mode="co_resident" if hw.ram_gb >= 96 else "off",
                decision="judge-4b",
                embedding="coreml" if hw.gpu == "apple" else "gpu",
                sandbox_memory_gb=float(tier.get("sandbox_gb") or 4.0),
            )
            if estimate_stack_memory(deep, hw, config)["fits"]:
                add(deep)

    return stacks
