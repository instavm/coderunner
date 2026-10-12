"""Published benchmark scores turned into one ability per model, so models are ranked on evidence, not size.

Scores come from Hugging Face's structured evaluation results (the .eval_results files on model repos,
for the Hub's official benchmark datasets). Models report different benchmarks, so a one-factor model is
fitted across all of them:

    score[model, benchmark] = offset[benchmark] + scale[benchmark] * ability[model]

Abilities are on a standard scale (mean 0, deviation 1 across the models with scores). A model with
few scores is pulled toward the mean. A bundled snapshot covers the catalog and notable releases;
discovery refreshes it daily. Quantized builds lose ability: measured for 1-bit Bonsai 27B (0.7 below
its base model, Qwen3.6-27B), estimated for the rest.
"""

from __future__ import annotations

import functools
import json
import pathlib
import statistics
from typing import Callable

from .. import paths

SNAPSHOT = pathlib.Path(__file__).with_name("evidence.json")
# Official benchmark datasets ("dataset:task") that separate general assistants. Left out: benchmarks
# run in different settings by different publishers (HLE with or without tools), document parsing,
# and ones that barely separate models.
BENCHMARKS = (
    "Idavidrein/gpqa:diamond",
    "TIGER-Lab/MMLU-Pro:mmlu_pro",
    "harborframework/terminal-bench-2.0:terminalbench_2",
    "harborframework/terminal-bench-2.1:terminalbench_2_1",
    "hkust-nlp/Toolathlon:toolathlon_verified",
    "ScaleAI/SWE-bench_Pro:SWE_Bench_Pro",
    "SWE-bench/SWE-bench_Verified:swe_bench_%_resolved",
    "SWE-bench/SWE-bench_Multilingual:swe_bench_multilingual_%_resolved",
    "MMMU/MMMU_Pro:mmmu_pro_vision",
    "MathArena/aime_2026:MathArena/aime_2026",
)
MIN_MODELS = 3            # a benchmark reported by fewer models cannot place anyone
SHRINK = 1.0              # pulls models with one or two scores toward the mean
ITERATIONS = 200
# Ability lost by builds below the precision their scores were published at: (bits at least, loss).
PRECISION_LOSS = ((4.5, 0.0), (4.0, 0.05), (3.0, 0.15), (2.0, 0.3), (1.5, 0.5), (0.0, 0.7))
# Ability lost per unit of benchmark score a build does not keep, for builds whose makers publish how much
# of the full model's scores they retain. Calibrated so a conventional 2-bit build keeps today's 2-bit loss:
# IQ2_XXS keeps 84.1% of Qwen3.8-27B's average in Prism's comparison, and 1.9 x 0.159 = 0.30.
RETENTION_LOSS = 1.9


def scores_from(eval_results: list[dict]) -> dict[str, float]:
    """Benchmark -> score from the Hub's evalResults records, averaging repeated submissions."""
    found: dict[str, list[float]] = {}
    for record in eval_results or []:
        data = record.get("data") or {}
        dataset = data.get("dataset") or {}
        key = f"{dataset.get('id')}:{dataset.get('task_id')}"
        if key in BENCHMARKS and isinstance(data.get("value"), (int, float)):
            found.setdefault(key, []).append(float(data["value"]))
    return {key: round(statistics.mean(values), 2) for key, values in found.items()}


def fetch(hub, repos, progress: Callable[[str], None] | None = None) -> dict[str, dict[str, float]]:
    """Published scores for each repo; repos that cannot be read are left out."""
    from .hub import HubError, endpoint
    found = {}
    for repo in sorted(set(repos)):
        try:
            info = hub._get_json(f"{endpoint()}/api/models/{repo}?expand%5B%5D=evalResults", f"evals-{repo}")
        except HubError as exc:
            if progress:
                progress(f"no benchmark scores for {repo}: {exc}")
            continue
        scores = scores_from((info or {}).get("evalResults") or [])
        if scores:
            found[repo] = scores
    return found


def _live_path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "evidence.json"


def save(scores: dict[str, dict[str, float]]) -> None:
    path = _live_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scores": scores}, indent=1))


def table() -> dict[str, dict[str, float]]:
    """The bundled snapshot, updated by the latest scores discovery fetched."""
    scores = dict(json.loads(SNAPSHOT.read_text())["scores"])
    live = _live_path()
    if live.exists():
        try:
            scores.update(json.loads(live.read_text())["scores"])
        except (OSError, ValueError, KeyError):
            pass
    return scores


def fit(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """One ability per model from scores on different benchmarks (alternating least squares)."""
    counts: dict[str, int] = {}
    for row in scores.values():
        for key in row:
            counts[key] = counts.get(key, 0) + 1
    obs = {repo: {k: v for k, v in row.items() if counts[k] >= MIN_MODELS} for repo, row in scores.items()}
    obs = {repo: row for repo, row in obs.items() if row}
    if len(obs) < 2:
        return {}
    benches = {key for row in obs.values() for key in row}
    spread = {}
    for key in benches:
        values = [row[key] for row in obs.values() if key in row]
        spread[key] = (statistics.mean(values), statistics.pstdev(values) or 1.0)
    ability = {repo: statistics.mean((v - spread[k][0]) / spread[k][1] for k, v in row.items())
               for repo, row in obs.items()}
    for _ in range(ITERATIONS):
        line = {}
        for key in benches:
            pairs = [(ability[repo], row[key]) for repo, row in obs.items() if key in row]
            mx = statistics.mean(x for x, _ in pairs)
            my = statistics.mean(y for _, y in pairs)
            var = sum((x - mx) ** 2 for x, _ in pairs) + 0.5
            scale = max(sum((x - mx) * (y - my) for x, y in pairs) / var, 1.0)
            line[key] = (my - scale * mx, scale)
        for repo, row in obs.items():
            num = sum(line[k][1] * (v - line[k][0]) for k, v in row.items())
            den = sum(line[k][1] ** 2 for k in row) + SHRINK * min(line[k][1] for k in row) ** 2
            ability[repo] = num / den
        mean = statistics.mean(ability.values())
        dev = statistics.pstdev(ability.values()) or 1.0
        ability = {repo: (x - mean) / dev for repo, x in ability.items()}
    return {repo: round(x, 3) for repo, x in ability.items()}


@functools.lru_cache(maxsize=4)
def _abilities(key: tuple) -> dict[str, float]:
    return fit(table())


def abilities() -> dict[str, float]:
    live = _live_path()
    stamp = live.stat().st_mtime if live.exists() else 0.0
    return _abilities((str(live), stamp))


def precision_loss(bits: float) -> float:
    return next(loss for floor, loss in PRECISION_LOSS if bits >= floor)


def ability(entry: dict, bits: float | None = None) -> float | None:
    """The entry's ability, less what its build loses to quantization; None without published scores.

    Scores normally describe the full-precision model (the catalog's base repo, or a discovered repo).
    An entry's "evidence" can point elsewhere, e.g. a quantized release scored as such ("build": true), or
    give the share of the full model's scores its maker measured the build to keep ("retained": 0.982).
    """
    known = abilities()
    evidence = entry.get("evidence") or {}
    repo = evidence.get("repo") or (entry.get("license") or {}).get("base_repo") or entry.get("repo")
    if repo not in known:
        return None
    if evidence.get("retained") is not None:
        return round(known[repo] - RETENTION_LOSS * (1 - float(evidence["retained"])), 3)
    if evidence.get("build") or bits is None:
        return known[repo]
    return round(known[repo] - precision_loss(bits), 3)
