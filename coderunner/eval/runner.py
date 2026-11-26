"""Orchestrate Tier 0, Tier 1, and Tier 2 single-model & multi-model StackSpec benchmarks."""

from __future__ import annotations

import asyncio
import copy
import pathlib
import tempfile
import time
from typing import Any, Callable

from ..core.config import Config
from ..models import bench, catalog, coreml, hardware, manager, probe, remote
from ..models.runtimes import DriverRuntime, JudgeRuntime, LlamaCppRuntime, MLXRuntime, OllamaRuntime, PrismRuntime
from .report import compare_markdown, plain_summary
from .spec import StackSpec, estimate_stack_memory, stack_from_config
from .suites import ane_workload_bench, run_safety_check, tier1_gate, tier2_combo_tasks
from .telemetry import load_history, save_run

_BENCH_LOCK = asyncio.Lock()


def run_tier0(config=None, *, force: bool = False, include_cdn: bool = True,
              progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Run Tier 0 pre-download bare-metal Mac probe (Metal GPU + Core ML ANE + SSD + HF CDN)."""
    sandbox_gb = float(config.get("sandbox.memory_gb") or 2) if config is not None else 2.0
    hw = hardware.detect(sandbox_gb)
    return probe.run(hw=hw, include_cdn=include_cdn, force=force, progress=progress)


def _judge_endpoint(config, ep: manager.Endpoint) -> tuple[str, str, str, str, float]:
    """Resolve the effective commit-judge (base_url, model, api, api_key, threshold) for `config`."""
    dec_model = str(config.get("models.decision") or "")
    if dec_model and dec_model != ep.model:
        url = str(config.get("models.decision_url") or ep.base_url or "")
        api = str(config.get("models.decision_api") or "systemone")
        threshold = float(config.get("models.decision_threshold", 0.7) or 0.7)
        return url, dec_model, api, ep.api_key, threshold
    threshold = float(config.get("models.decision_threshold", 0.7) or 0.7)
    return ep.base_url, ep.model, "chat", ep.api_key, threshold


async def run_tier1(config, spec: StackSpec | None = None, *, progress: Callable[[str], None] = lambda _: None,
                    run_safety: bool = True) -> dict[str, Any]:
    """Run Tier 1 (90s) gate on the active endpoint (or specified StackSpec if already active)."""
    hw = manager._hardware(config, None)
    spec = spec or stack_from_config(config, hw)
    ep = manager.ensure_running(config)
    if not ep.base_url or not ep.model:
        return {"ok": False, "error": "No model is configured yet."}
    j_url, j_model, j_api, j_key, j_thresh = _judge_endpoint(config, ep)
    progress(f"testing {spec.label} (tools, vision, prompt cache, safety judge and memory)")
    t0 = time.monotonic()
    gate = await tier1_gate(
        ep.base_url, ep.model, vision=ep.vision, api_key=ep.api_key,
        judge_base_url=j_url, judge_model=j_model, judge_api=j_api, judge_api_key=j_key,
        judge_threshold=j_thresh, run_safety=run_safety,
    )
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
        "safety_measured": bool(gate.get("safety_measured")),
        "missed_commits": gate.get("missed_commits"),
        "false_alarm_rate": gate.get("false_alarm_rate"),
        "ane_offload": bool(ane and ane.get("ane_verified")),
        "checks": gate["checks"],
    }
    save_run(record)
    return record


def _snapshot_runtime_states(config) -> dict[pathlib.Path, bytes | None]:
    """Capture local runtime state files so candidate benchmarks can restore the owner's runtime cleanly."""
    runtimes = (LlamaCppRuntime(config), PrismRuntime(config), JudgeRuntime(config),
                DriverRuntime(config), MLXRuntime(config))
    snap: dict[pathlib.Path, bytes | None] = {}
    for rt in runtimes:
        p = rt.state_path
        snap[p] = p.read_bytes() if p.is_file() else None
    return snap


def _restore_runtime_states(config, snap: dict[pathlib.Path, bytes | None]) -> None:
    """Restore snapshotted runtime state files and bring back the owner's active runtime."""
    for cls in (LlamaCppRuntime, PrismRuntime, JudgeRuntime, DriverRuntime, MLXRuntime):
        rt = cls(config)
        prev = snap.get(rt.state_path)
        cur = rt.state_path.read_bytes() if rt.state_path.is_file() else None
        if cur != prev:
            try:
                rt.stop()
            except Exception:
                pass
            if prev is None:
                rt.state_path.unlink(missing_ok=True)
            else:
                rt.state_path.parent.mkdir(parents=True, exist_ok=True)
                rt.state_path.write_bytes(prev)
    try:
        manager.ensure_running(config)
    except Exception:
        pass


def _validate_spec(config, spec: StackSpec, hw: hardware.Hardware, mem: dict[str, Any]) -> str | None:
    """Reject unsupported or contradictory StackSpec settings before running so nothing is mis-recorded."""
    if spec.escalator:
        return ("Dual-model escalator routing is not yet supported for automated stack runs; "
                "benchmark single-planner or planner+driver stacks.")
    if spec.remote_provider:
        active_remote = remote.active(config) or {}
        if active_remote.get("provider") != spec.remote_provider:
            return (f"Remote provider {spec.remote_provider!r} is not currently active; "
                    "configure remote mode before benchmarking it.")
        if spec.planner and active_remote.get("model") and spec.planner != active_remote.get("model"):
            return (f"Requested remote model {spec.planner!r} does not match active remote model "
                    f"{active_remote.get('model')!r}.")
        if spec.base_url and (config.get("models.remote") or {}).get("base_url"):
            if spec.base_url.rstrip("/") != str((config.get("models.remote") or {}).get("base_url")).rstrip("/"):
                return "Requested remote base_url does not match active remote configuration."
    else:
        if not mem["fits"]:
            return f"{spec.label} needs {mem['total_gb']} GB (budget {mem['budget_gb']} GB)"
        if spec.driver and spec.driver_mode == "co_resident" and mem.get("driver_exclusive"):
            return f"{spec.label} cannot keep {spec.driver} co-resident within the {mem['budget_gb']} GB budget."
    emb = spec.embedding or "none"
    if emb not in ("coreml", "gpu", "none"):
        return f"Unsupported embedding mode {emb!r}"
    if emb == "coreml" and not coreml.supported():
        return "Core ML embeddings require Apple Silicon macOS."
    dec = spec.decision or "self"
    if dec not in ("self", "") and not spec.remote_provider:
        try:
            entry = catalog.lookup(dec, hw=hw, config=config)
        except KeyError:
            return f"Unknown decision judge model {dec!r}"
        if "decision" not in (entry.get("roles") or []):
            return f"{dec!r} is not a decision judge model."
    return None


def _apply_decision(work: Config, spec: StackSpec, hw: hardware.Hardware,
                    progress: Callable[[str], None]) -> str | None:
    """Prepare and verify the exact commit-judge requested by `spec.decision` on `work`."""
    dec = spec.decision or "self"
    if spec.remote_provider or dec in ("self", ""):
        for key in ("models.decision", "models.decision_api", "models.decision_url"):
            work.set(key, "")
        return None
    entry = catalog.lookup(dec, hw=hw, config=work)
    if (entry.get("artifacts") or {}).get("gguf"):
        jrt = JudgeRuntime(work)
        if not jrt.installed(entry):
            jrt.pull(entry, progress)
        alias = jrt.prepare(entry, 4096)
        jrt.start()
        work.set("models.decision", alias)
        work.set("models.decision_api", "chat")
        work.set("models.decision_url", jrt.base_url)
        work.set("models.decision_threshold", (entry.get("decision") or {}).get("threshold", 0.7))
        return None
    if (entry.get("artifacts") or {}).get("ollama"):
        ort = OllamaRuntime(work)
        if not ort.available():
            return f"Decision model {dec} requires Ollama, which is not installed."
        if not ort.running():
            ort.start()
        if not ort.installed(entry):
            ort.pull(entry, progress)
        tag = entry["artifacts"]["ollama"]["tag"]
        work.set("models.decision", tag)
        work.set("models.decision_api", "systemone")
        work.set("models.decision_url", ort.base_url)
        work.set("models.decision_threshold", (entry.get("decision") or {}).get("threshold", 0.7))
        return None
    return f"Decision model {dec} has no supported runtime artifact on this machine."


def _apply_embedding(work: Config, spec: StackSpec, hw: hardware.Hardware,
                     progress: Callable[[str], None]) -> str | None:
    """Apply and verify `spec.embedding` on the temporary `work` config."""
    emb = spec.embedding or "none"
    if emb == "none":
        work.set("models.embedding", "")
        work.set("models.embedding_url", "")
        work.set("models.embedding_runtime", "none")
        return None
    work.set("models.embedding_runtime", emb)
    if emb == "coreml":
        if not coreml.supported():
            return "Core ML embeddings require Apple Silicon macOS."
        rt = coreml.CoreMLRuntime(work)
        if (work.get("models.embedding_url") or "").rstrip("/") != rt.base_url:
            where = manager.setup_coreml_embedding(work, hw=hw, progress=progress)
            if where != "coreml":
                return "Core ML embedding runtime could not be prepared."
        return None
    # GPU embedding requested: switch away from Core ML if active and verify an embedding model exists.
    if coreml.supported():
        rt = coreml.CoreMLRuntime(work)
        if (work.get("models.embedding_url") or "").rstrip("/") == rt.base_url:
            media = work.get("models.embedding_media") or ""
            media_url = work.get("models.embedding_media_url") or ""
            if not media:
                return "GPU embedding mode requested, but no GPU embedding model is configured."
            work.set("models.embedding", media)
            work.set("models.embedding_url", media_url)
    return None


def _apply_driver(work: Config, spec: StackSpec, mem: dict[str, Any],
                  progress: Callable[[str], None]) -> str | None:
    """Prepare and verify `spec.driver` / `spec.driver_mode` on the temporary `work` config."""
    if not spec.driver or spec.driver_mode == "off" or spec.remote_provider:
        work.set("models.driver", {})
        return None
    try:
        entry = catalog.get(spec.driver)
    except KeyError:
        return f"Unknown browser driver model {spec.driver!r}."
    drt = DriverRuntime(work)
    if not drt.available():
        drt.install(progress)
    if not drt.installed(entry):
        drt.pull(entry, progress)
    drt.prepare(entry, manager.DRIVER_CTX)
    work.set("models.driver", {"model": entry["id"], "exclusive": bool(mem["driver_exclusive"]), "steps": 30})
    return None


async def run_stack(config, spec: StackSpec, *, suite: str = "tier2", names: list[str] | None = None,
                    progress: Callable[[str], None] = lambda _: None,
                    setup_model: bool = True) -> dict[str, Any]:
    """Evaluate one StackSpec (single model or multi-model combo) on tier1, tier2, or ane suite.

    Uses an isolated temporary config path so the owner's config.json is never clobbered, validates
    every requested stack component (ctx, runtime, decision judge, driver, embedding, remote), and
    restores the owner's active local runtime in a finally block.
    """
    hw = manager._hardware(config, None)
    if suite == "ane":
        ane = await ane_workload_bench(config, hw=hw, progress=progress)
        return {"ok": True, "suite": "ane", "machine": bench.fingerprint(hw), **ane}

    mem = estimate_stack_memory(spec, hw, config)
    err = _validate_spec(config, spec, hw, mem)
    if err:
        return {"ok": False, "suite": suite, "signature": spec.signature(),
                "stack": spec.to_dict(), "memory": mem, "error": err}

    async with _BENCH_LOCK:
        snap = _snapshot_runtime_states(config)
        switched = False
        with tempfile.TemporaryDirectory(prefix="coderunner-eval-cfg-") as tmp:
            work = Config(copy.deepcopy(config.data), path=pathlib.Path(tmp) / "config.json")
            try:
                if not spec.remote_provider:
                    cur_model = manager.entry_id(work)
                    cur_rt = str(work.get("models.runtime") or "")
                    cur_ctx = int(work.get("models.context") or 32768)
                    req_rt = spec.runtime if spec.runtime not in ("", "auto") else cur_rt
                    req_ctx = int(spec.ctx or cur_ctx)
                    needs_setup = (
                        (bool(spec.planner) and spec.planner != cur_model)
                        or (bool(req_rt) and req_rt != cur_rt)
                        or (req_ctx != cur_ctx)
                    )
                    if needs_setup:
                        if not setup_model:
                            return {"ok": False, "suite": suite, "signature": spec.signature(),
                                    "stack": spec.to_dict(), "memory": mem,
                                    "error": f"{spec.label} does not match the active model/context/runtime and setup_model=False."}
                        progress(f"preparing {spec.label} (ctx {req_ctx})")
                        switched = True
                        done = await asyncio.to_thread(
                            manager.setup, work, model_id=spec.planner,
                            runtime=spec.runtime if spec.runtime not in ("", "auto") else None,
                            hw=hw, progress=progress, embedding=False, decision=False,
                            measure=False, ctx=req_ctx,
                        )
                        if not done.get("ok"):
                            return {"ok": False, "suite": suite, "signature": spec.signature(),
                                    "stack": spec.to_dict(), "memory": mem,
                                    "error": done.get("error") or "setup failed"}
                        if int(work.get("models.context") or 0) != req_ctx:
                            return {"ok": False, "suite": suite, "signature": spec.signature(),
                                    "stack": spec.to_dict(), "memory": mem,
                                    "error": f"Requested context {req_ctx} was not applied (got {work.get('models.context')})."}

                if (spec.decision or "self") not in ("self", ""):
                    switched = True
                dec_err = await asyncio.to_thread(_apply_decision, work, spec, hw, progress)
                if dec_err:
                    return {"ok": False, "suite": suite, "signature": spec.signature(),
                            "stack": spec.to_dict(), "memory": mem, "error": dec_err}

                emb_err = await asyncio.to_thread(_apply_embedding, work, spec, hw, progress)
                if emb_err:
                    return {"ok": False, "suite": suite, "signature": spec.signature(),
                            "stack": spec.to_dict(), "memory": mem, "error": emb_err}

                if spec.driver and spec.driver_mode != "off":
                    switched = True
                drv_err = await asyncio.to_thread(_apply_driver, work, spec, mem, progress)
                if drv_err:
                    return {"ok": False, "suite": suite, "signature": spec.signature(),
                            "stack": spec.to_dict(), "memory": mem, "error": drv_err}

                work.save()

                if suite == "tier1":
                    return await run_tier1(work, spec, progress=progress)

                # Tier 2: full deterministic multi-domain tasks in an isolated subprocess + commit-judge safety check.
                swap_before = hardware.swap_used_gb() or 0.0
                result = await tier2_combo_tasks(
                    work, names=names, keep_driver=bool(spec.driver and spec.driver_mode != "off"),
                    progress=progress,
                )
                swap_after = hardware.swap_used_gb() or 0.0
                if not result.get("ok"):
                    return {"ok": False, "suite": suite, "signature": spec.signature(),
                            "stack": spec.to_dict(), "memory": mem,
                            "error": result.get("error") or "benchmark failed"}

                ep = manager.endpoint(work)
                j_url, j_model, j_api, j_key, j_thresh = _judge_endpoint(work, ep)
                safety = await run_safety_check(
                    j_url, j_model, api=j_api, api_key=j_key, threshold=j_thresh,
                )
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
                    "safety_measured": bool(safety.get("measured")),
                    "missed_commits": safety.get("missed_commits"),
                    "false_alarm_rate": safety.get("false_alarm_rate"),
                    "ane_offload": bool(ane and ane.get("ane_verified")),
                    "tasks": result.get("tasks") or [],
                }
                save_run(record)
                return record
            finally:
                if switched:
                    await asyncio.to_thread(_restore_runtime_states, config, snap)


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
