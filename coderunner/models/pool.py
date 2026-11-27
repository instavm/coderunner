"""Unified ModelPool role router and host unified-memory lease arbiter.

Coordinates model roles (planner, optional deep escalator on 64-512 GB Macs, screenshot driver,
safety judge, Core ML ANE embedder, and voice STT) under one unified-memory budget and ensures
background file indexing yields when Tier 0 measured GPU/ANE contention > 8% or swap is elevated.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import hardware, manager, probe


@dataclasses.dataclass
class RolePlan:
    planner: str = ""
    escalator: str = ""
    driver: str = ""
    driver_exclusive: bool = True
    decision: str = "self"
    embedding: str = "coreml"
    voice: str = "auto"
    omni: str = ""
    omni_exclusive: bool = False
    yield_indexing: bool = False
    tier_gb: int = 16

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ModelPool:
    """Routes task turns across configured roles and enforces unified-memory safety leases."""

    SWAP_PAUSE_GB = 2.5

    def __init__(self, config, hw: hardware.Hardware | None = None):
        self.config = config
        self.hw = manager._hardware(config, hw)
        self.probed = probe.load(self.hw)
        self._active_turns = 0

    def role_plan(self) -> RolePlan:
        tier = hardware.ram_tier(self.hw.ram_gb or 16)
        drv = self.config.get("models.driver") or {}
        drv_model = str(drv.get("model") or "") if isinstance(drv, dict) else ""
        drv_excl = bool(drv.get("exclusive", True)) if drv_model else True
        esc = str(self.config.get("models.escalator") or "")
        dec = str(self.config.get("models.decision") or "") or "self"
        emb = "gpu" if self.config.get("models.embedding_runtime") == "gpu" else "coreml"
        omni = str(self.config.get("voice.omni_model") or "")
        return RolePlan(
            planner=manager.entry_id(self.config),
            escalator=esc,
            driver=drv_model,
            driver_exclusive=drv_excl,
            decision=dec,
            embedding=emb,
            voice=str(self.config.get("voice.stt") or "auto"),
            omni=omni,
            omni_exclusive=bool(self.config.get("voice.omni_exclusive")) if omni else False,
            yield_indexing=self.should_yield_indexing(llm_busy=True),
            tier_gb=int(tier["tier_gb"]),
        )

    def can_speak_during_task(self) -> bool:
        """Whether LFM3-Omni may load beside the planner without wiping its KV cache."""
        plan = self.role_plan()
        if not plan.omni:
            return False
        if not plan.omni_exclusive:
            return True
        # Exclusive: only speak mid-task when chat is remote (local GPU free).
        rem = self.config.get("models.remote") or {}
        return bool(rem.get("provider"))

    def should_escalate(self, *, stuck: int = 0, failed: bool = False, step: int = 0) -> bool:
        """Whether a Fast+Deep combo should hand this turn to the configured escalator model."""
        if not self.config.get("models.escalator"):
            return False
        return bool(stuck >= 2 or (failed and step >= 3))

    def should_yield_indexing(self, *, llm_busy: bool) -> bool:
        """Pause background file indexing between passages during LLM turns when memory/ANE contention is high."""
        if not llm_busy:
            return False
        setting = self.config.get("files.yield_during_llm_turn", "auto")
        if setting is True or str(setting).lower() == "true":
            return True
        if setting is False or str(setting).lower() == "false":
            return False
        if self.probed and self.probed.get("yield_during_llm_turn"):
            return True
        # Tight 8-18 GB Macs also yield indexing when swap is active.
        if self.hw.gpu == "apple" and 0 < self.hw.ram_gb <= 18.5:
            swap = hardware.swap_used_gb()
            if swap is not None and swap >= 1.5:
                return True
        return False

    def memory_safe_for_load(self, need_gb: float) -> tuple[bool, dict]:
        """Verify live reclaimable RAM and swap before loading another model on the host."""
        mem = hardware.live_memory(self.hw.ram_gb)
        swap = float(mem.get("swap_used_gb") or 0.0)
        free = mem.get("free_gb")
        if swap >= self.SWAP_PAUSE_GB:
            return False, {**mem, "reason": f"swap is high ({swap:.1f} GB in use)"}
        if free is not None and need_gb + manager.MEMORY_MARGIN_GB > free:
            return False, {**mem, "reason": f"needs {need_gb:.1f} GB and {free:.1f} GB is free"}
        return True, mem
