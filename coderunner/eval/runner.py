"""Orchestrate Tier 0, Tier 1, and Tier 2 single-model & multi-model StackSpec benchmarks."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable

from ..core.config import Config
from ..models import bench, hardware, manager, probe
from .report import compare_markdown, plain_summary
from .spec import StackSpec, estimate_stack_memory, stack_from_config
from .suites import ane_workload_bench, tier1_gate, tier2_combo_tasks
from .telemetry import load_history, save_run


def run_tier0(config=None, *, force: bool = False, include_cdn: bool = True,
              progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Run Tier 0 pre-download bare-metal Mac probe (Metal GPU + Core ML ANE + SSD + HF CDN)."""
    sandbox_gb = float(config.get("sandbox.memory_gb") or 2) if config is not None else 2.0
    hw = hardware.detect(sandbox_gb)
    return probe.run(hw=hw, include_cdn=include_cdn, force=force, progress=progress)


async def run_tier1(config, spec: StackSpec | None = None, *, progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Run Tier 1 (90s) gate on the active endpoint (or specified StackSpec if already active)."""
    hw = manager._hardware(config, None)
    spec = spec or stack_from_config(config, hw)
    ep = manager.ensure_running(config)
    if not ep.base_url or not ep.model:
        return {"ok": False, "error": "No model is configured yet."}
    progress(f"testing {spec.label} (tools, vision, prompt cache and memory)")
    t0 = time.monotonic()
    gate = await tier1_gate(ep.base_url, ep.model, vision=ep.vision, api_key=ep.api_key)
    mem = estimate_stack_memory(spec, hw, config)
    ane = probe.load(hw)
    record = {
        "ok": gate["ok"],
        "suite": "tier1",
        "signature": spec.signature(),
        "machine": bench.fingerprint(hw),
        "stack": spec.to_dict(),
        "memory": mem,
        "seconds": round(time.monotonic() - t0, 1),
        "passed": gate["passed"],
        "total": gate["total"],
        "step_s": gate.get("step_s") or 0.0,
        "read_tok_s": gate.get("read_tok_s"),
        "gen_tok_s": gate.get("gen_tok_s"),
        "cache_hit_ratio": gate.get("cache_hit_ratio"),
        "swap_delta_gb": gate.get("swap_delta_gb") or 0.0,
        "missed_commits": gate.get("missed_commits") or 0,
        "ane_offload": bool(ane and ane.get("ane_verified")),
        "checks": gate["checks"],
    }
    save_run(record)
    return record


async def run_stack(config, spec: StackSpec, *, suite: str = "tier2", names: list[str] | None = None,
                    progress: Callable[[str], None] = lambda _: None,
                    setup_model: bool = True) -> dict[str, Any]:
    """Evaluate one StackSpec (single model or multi-model combo) on tier1, tier2, or ane suite."""
    hw = manager._hardware(config, None)
    if suite == "ane":
        ane = await ane_workload_bench(config, hw=hw, progress=progress)
        return {"ok": True, "suite": "ane", "machine": bench.fingerprint(hw), **ane}

    mem = estimate_stack_memory(spec, hw, config)
    if not mem["fits"] and not spec.remote_provider:
        return {"ok": False, "suite": suite, "signature": spec.signature(), "stack": spec.to_dict(),
                "memory": mem, "error": f"{spec.label} needs {mem['total_gb']} GB (budget {mem['budget_gb']} GB)"}

    # Clone config so benchmarking a candidate combo never permanently clobbers the owner's active settings.
    work = Config(copy.deepcopy(config.data), path=config.path)
    if setup_model and not spec.remote_provider and spec.planner and spec.planner != manager.entry_id(config):
        progress(f"preparing {spec.label}")
        done = await asyncio.to_thread(
            manager.setup, work, model_id=spec.planner, runtime=spec.runtime if spec.runtime != "auto" else None,
            hw=hw, progress=progress, embedding=False, decision=(spec.decision not in ("", "self")), measure=False,
        )
        if not done.get("ok"):
            return {"ok": False, "suite": suite, "signature": spec.signature(), "stack": spec.to_dict(),
                    "memory": mem, "error": done.get("error") or "setup failed"}
    if spec.driver and spec.driver_mode != "off" and not spec.remote_provider:
        work.set("models.driver", {"model": spec.driver, "exclusive": mem["driver_exclusive"], "steps": 30})
    else:
        work.set("models.driver", {})

    if suite == "tier1":
        return await run_tier1(work, spec, progress=progress)

    # Tier 2: full deterministic multi-domain tasks with the combo intact.
    swap_before = hardware.swap_used_gb() or 0.0
    result = await tier2_combo_tasks(work, names=names, keep_driver=bool(spec.driver and spec.driver_mode != "off"),
                                     progress=progress)
    swap_after = hardware.swap_used_gb() or 0.0
    if not result.get("ok"):
        return {"ok": False, "suite": suite, "signature": spec.signature(), "stack": spec.to_dict(),
                "memory": mem, "error": result.get("error") or "benchmark failed"}
    ane = probe.load(hw)
    step_s = result.get("median_request_s") or 0.0
    record = {
        "ok": True,
        "suite": suite,
        "signature": spec.signature(),
        "machine": bench.fingerprint(hw),
        "stack": spec.to_dict(),
        "memory": mem,
        "passed": int(result.get("passed") or 0),
        "total": int(result.get("total") or 0),
        "seconds": float(result.get("seconds") or 0.0),
        "step_s": float(step_s),
        "slowed": bool(result.get("slowed")),
        "swap_delta_gb": round(max(0.0, swap_after - swap_before), 2),
        "missed_commits": 0,
        "ane_offload": bool(ane and ane.get("ane_verified")),
        "tasks": result.get("tasks") or [],
    }
    save_run(record)
    return record


async def compare_stacks(config, specs: list[StackSpec], *, suite: str = "tier2",
                         names: list[str] | None = None, preference: str = "balanced",
                         progress: Callable[[str], None] = lambda _: None,
                         reuse_cached: bool = True) -> dict[str, Any]:
    """Compare 2-4 single models or multi-model combinations head-to-head and return a Pareto scoreboard."""
    hw = manager._hardware(config, None)
    history = {r.get("signature"): r for r in load_history(hw=hw, suite=suite)}
    runs: list[dict] = []
    for spec in specs:
        if reuse_cached and spec.signature() in history and history[spec.signature()].get("ok"):
            progress(f"using saved {suite} result for {spec.label}")
            runs.append(history[spec.signature()])
            continue
        progress(f"benchmarking {spec.label} ({suite})")
        res = await run_stack(config, spec, suite=suite, names=names, progress=progress)
        runs.append(res)
    ok_runs = [r for r in runs if r.get("ok")]
    summary = plain_summary(ok_runs, preference=preference)
    return {
        "ok": bool(ok_runs),
        "suite": suite,
        "machine": bench.fingerprint(hw),
        "runs": runs,
        "summary": summary,
        "markdown": compare_markdown(ok_runs, preference=preference) if ok_runs else "",
    }
