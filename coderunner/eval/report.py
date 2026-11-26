"""Plain-English Pareto summaries and shareable Markdown reports for single vs. multi-model stack runs."""

from __future__ import annotations

from .telemetry import rank_runs, score_run


def plain_summary(runs: list[dict], preference: str = "balanced") -> dict:
    """Turn evaluated stack runs into a calm, human-readable comparison scoreboard (zero jargon)."""
    ordered = rank_runs(runs, preference=preference)
    if not ordered:
        return {"winner": None, "preference": preference, "rows": []}
    winner = ordered[0]
    w_score = score_run(winner)
    rows = []
    for idx, run in enumerate(ordered):
        s = score_run(run)
        stack = run.get("stack") or {}
        speed_rel = (round(s["step_s"] / w_score["step_s"], 1) if (w_score["step_s"] and s["step_s"]) else None)
        rows.append({
            "rank": idx + 1,
            "id": stack.get("id") or run.get("signature") or "",
            "label": stack.get("label") or stack.get("planner") or "Setup",
            "winner": idx == 0 and s["safe"],
            "passed": s["passed"],
            "total": s["total"],
            "pass_rate": s["pass_rate"],
            "step_s": s["step_s"],
            "seconds": s["seconds"],
            "memory_gb": s["memory_gb"],
            "comfort": s["comfort"],
            "safe": s["safe"],
            "cache_hit_ratio": s["cache_hit_ratio"],
            "swap_delta_gb": s["swap_delta_gb"],
            "relative_step": speed_rel,
            "stack": stack,
            "suite": run.get("suite") or "",
            "date": run.get("date") or "",
            "tasks": run.get("tasks") or [],
        })
    return {"winner": rows[0] if rows and rows[0]["safe"] else None, "preference": preference, "rows": rows}


def compare_markdown(runs: list[dict], preference: str = "balanced") -> str:
    """Shareable Markdown table comparing single models and multi-model combos (no paths, names, or secrets)."""
    summary = plain_summary(runs, preference=preference)
    lines = [
        "## CodeRunner Stack Comparison",
        "",
        "| Setup | Passed | Step time | Total | Memory | Mac comfort | Cache hit |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        cache = f"{round(row['cache_hit_ratio'] * 100)}%" if row.get("cache_hit_ratio") is not None else "—"
        step = f"{row['step_s']:.1f} s" if row["step_s"] else "—"
        total = f"{row['seconds'] / 60:.1f} min" if row["seconds"] else "—"
        mem = f"{row['memory_gb']:.1f} GB" if row["memory_gb"] else "Remote"
        badge = " (best)" if row["winner"] else ""
        lines.append(f"| {row['label']}{badge} | {row['passed']}/{row['total']} | {step} | {total} | {mem} | {row['comfort']} | {cache} |")
    return "\n".join(lines) + "\n"
