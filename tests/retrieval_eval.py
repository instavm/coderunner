"""Retrieval quality on the owner's own shared folder, measured without exposing it:
python tests/retrieval_eval.py [BENCH] [--query-url URL --query-model NAME] [--no-code-prompt] [--out FILE]

A bench is a JSON file kept outside the repo (default ~/.cache/coderunner-dev/retrieval-bench/bench.json):
questions in plain language that name no person, number, place or company, each with the files that answer it
as sha256(path relative to the root)[:16]. The search runs through CodeRunner's own index (CODERUNNER_HOME) and
the report shows questions, ranks and the kinds of what came first: never a path or a file's content.

Per kind of question: the share answered by the first result (R@1), within the first five (R@5), the mean
reciprocal rank and the median search time. --query-url embeds the questions with another build of the model
(e.g. Ollama's beside a Core ML index), --no-code-prompt searches code with the general prompt instead.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT = pathlib.Path.home() / ".cache/coderunner-dev/retrieval-bench/bench.json"


def anonymous(path: str, root: str) -> str:
    relative = path[len(root):] if path.startswith(root) else path
    return hashlib.sha256(relative.encode()).hexdigest()[:16]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", nargs="?", default=str(DEFAULT))
    parser.add_argument("--query-url", default="", help="embed the questions here instead (e.g. http://127.0.0.1:11434)")
    parser.add_argument("--query-model", default="", help="with --query-url: the model name there")
    parser.add_argument("--no-code-prompt", action="store_true", help="search code with the general prompt")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="", help="write every question's rank as JSON")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a ranking constant of coderunner.files.search for this run, e.g. WORDS=1.0")
    args = parser.parse_args()
    from coderunner.files import search as ranking
    for item in args.set:
        name, value = item.split("=", 1)
        setattr(ranking, name, type(getattr(ranking, name))(value))

    from coderunner.core import paths
    from coderunner.core.audit import AuditLog
    from coderunner.core.config import Config
    from coderunner.core.store import Store
    from coderunner.files import Embedder, FileIndex

    bench = json.loads(pathlib.Path(args.bench).read_text())
    root = os.path.expanduser(bench.get("root", "~/Downloads")).rstrip("/") + "/"
    config = Config.load()
    url = args.query_url or config.get("models.embedding_url") or "http://127.0.0.1:11434"
    model = args.query_model or config.get("files.embedding") or config.get("models.embedding")
    embedder = Embedder(url, model)

    async def recover() -> bool:                 # the Core ML server exits when idle; start it again
        from coderunner.models import manager
        return await asyncio.to_thread(manager.restart_embedding, config)
    embedder.recover = recover
    if args.no_code_prompt:
        embedder.code_prefix = ""
    index = FileIndex(config, Store(paths.assistant_dir() / "assistant.db"), AuditLog(paths.assistant_dir() / "audit-eval.jsonl"),
                      embedder)
    await asyncio.to_thread(index.vectors.load)
    await index.search("warm up")
    rows, by_kind = [], collections.defaultdict(list)
    for item in bench["questions"]:
        began = time.perf_counter()
        found = await index.search(item["query"], limit=args.limit)
        seconds = time.perf_counter() - began
        # A result stands for its exact copies too ("same file at ...").
        ids = [{anonymous(p, root) for p in [r["path"], *r.get("copies", [])]} for r in found]
        rank = next((i + 1 for i, h in enumerate(ids) if h & set(item["targets"])), None)
        row = {"id": item["id"], "kind": item["kind"], "rank": rank, "ms": round(seconds * 1000),
               "first_kind": found[0]["kind"] if found else None}
        rows.append(row)
        by_kind[item["kind"]].append(row)
        print(f"{item['id']} {item['kind']:10} {('#' + str(rank)) if rank else 'miss':5} {row['ms']:4d} ms  "
              f"{item['query'][:70]:70}  first: {row['first_kind']}", flush=True)

    def summary(group: list[dict]) -> str:
        ranks = [r["rank"] for r in group]
        r1 = sum(r == 1 for r in ranks) / len(ranks)
        r5 = sum(r is not None and r <= 5 for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        return f"R@1 {r1:.2f}  R@5 {r5:.2f}  MRR {mrr:.2f}  median {statistics.median(r['ms'] for r in group):.0f} ms"
    print()
    for kind, group in sorted(by_kind.items()):
        print(f"{kind:10} ({len(group):2d})  {summary(group)}")
    print(f"{'all':10} ({len(rows):2d})  {summary(rows)}  | queries embedded by {model} at {url}"
          f"{', no code prompt' if args.no_code_prompt else ''}{''.join(', ' + item for item in args.set)}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
    await embedder.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
