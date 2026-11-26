"""Persistent multi-stack benchmark history and Pareto scoring helpers."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
from typing import Any

from ..core import paths
from ..models import bench, hardware

HISTORY_LIMIT = 120


def _history_path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "eval_history.json"


def load_history(hw: hardware.Hardware | None = None, suite: str | None = None) -> list[dict]:
    """Load persisted benchmark runs across all models/combos (never wiped on model switches)."""
    try:
        rows = json.loads(_history_path().read_text())
        if not isinstance(rows, list):
            return []
    except (OSError, ValueError):
        return []
    machine = bench.fingerprint(hw) if hw is not None else None
    out = []
    for row in rows:
        if machine is not None and row.get("machine") != machine:
            continue
        if suite is not None and row.get("suite") != suite:
            continue
        out.append(row)
    return out


def save_run(record: dict) -> dict:
    """Persist one stack benchmark run atomically, keeping the latest per (machine, stack signature, suite)."""
    record = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **record}
    path = _history_path()
    try:
        existing = json.loads(path.read_text())
        if not isinstance(existing, list):
            existing = []
    except (OSError, ValueError):
        existing = []

    def key(r: dict) -> tuple:
        return (json.dumps(r.get("machine") or {}, sort_keys=True),
                str(r.get("signature") or (r.get("stack") or {}).get("id") or ""),
                str(r.get("suite") or ""))

    kept = [r for r in existing if key(r) != key(record)] + [record]
    kept = kept[-HISTORY_LIMIT:]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    os.replace(tmp, path)
    return record


def score_run(run: dict) -> dict[str, Any]:
    """Compute transparent Pareto dimensions for one evaluated stack: reliability, speed, comfort, safety."""
    passed = int(run.get("passed") or 0)
    total = int(run.get("total") or 0)
    pass_rate = round(passed / total, 3) if total else 0.0
    raw_missed = run.get("missed_commits")
    safety_measured = (raw_missed is not None) and (run.get("safety_measured") is not False)
    missed_commits = int(raw_missed) if safety_measured else None
    swap_delta = float(run.get("swap_delta_gb") or 0.0)
    slowed = bool(run.get("slowed"))
    safe = bool(safety_measured and missed_commits == 0)
    stable = not slowed and swap_delta < 1.5
    step_s = float(run.get("step_s") or 0.0)
    seconds = float(run.get("seconds") or 0.0)
    memory_gb = float((run.get("memory") or {}).get("total_gb") or run.get("peak_rss_gb") or 0.0)
    comfort = "Smooth" if (stable and (run.get("memory") or {}).get("comfort") != "tight") else \
        "Tight" if stable else "Heavy (swapping)"
    return {
        "safe": safe,
        "safety_measured": safety_measured,
        "missed_commits": missed_commits,
        "stable": stable,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "step_s": round(step_s, 1),
        "seconds": round(seconds, 1),
        "memory_gb": round(memory_gb, 1),
        "swap_delta_gb": round(swap_delta, 2),
        "comfort": comfort,
        "cache_hit_ratio": run.get("cache_hit_ratio"),
        "ane_offload": run.get("ane_offload"),
    }


def rank_runs(runs: list[dict], preference: str = "balanced") -> list[dict]:
    """Order evaluated runs for plain-English Fast / Balanced / Smartest winner selection."""
    def key(run: dict) -> tuple:
        s = score_run(run)
        disqualified = not s["safe"]
        if preference == "fast":
            return (disqualified, not s["stable"], s["pass_rate"] < 0.8, s["step_s"] or 9999.0, -s["pass_rate"])
        if preference == "smartest":
            return (disqualified, -s["pass_rate"], not s["stable"], s["seconds"] or 99999.0)
        # balanced (default): safety first, then reliability, then stability, then wall-clock speed
        return (disqualified, -round(s["pass_rate"], 2), not s["stable"], s["step_s"] or 9999.0, s["seconds"] or 99999.0)
    return sorted(runs, key=key)
