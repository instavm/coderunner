"""Search quality on a generated folder, with the real embedding model: python tests/files_eval.py

Builds notes, data, PDFs, screenshots, a picture and a spoken memo among distractor notes, indexes them
through Ollama, and asks natural-language questions whose answer is known. Prints the share answered
first (R@1) and in the top three (R@3), the mean reciprocal rank, and warm search latency, so a change
to ranking can be measured before it lands. Needs Ollama with the embedding model, ffmpeg, and on macOS
`say` and the Swift compiler for the spoken memo and the screenshots.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

NOTES = {
    "dentist.md": "Dentist appointment moved to Monday 9am with Dr. Rao. Bring the X-ray from last year.",
    "groceries.txt": "Shopping list: milk, eggs, sourdough bread, tomatoes, coffee beans, washing powder.",
    "trip-plan.md": "Goa holiday: flights on 12 December, beach hotel booked in Candolim, scooter rental.",
    "meeting-notes.md": "Q3 budget review. Marketing spend cut by 10 percent; hiring frozen until January.",
    "recipe.md": "Paneer butter masala: tomatoes, butter, cream, cashews, garam masala, kasuri methi.",
    "car-service.txt": "Car service is due at 40,000 km: oil change, brake pads and tyre rotation.",
}
PDFS = {
    "offer-letter.pdf": ["Offer of employment at Northwind", "Base salary 120000 USD per year plus bonus",
                         "Start date 1 November"],
    "lease.pdf": ["Residential lease agreement", "Monthly rent 1500 USD due on the first of the month",
                  "Pets are allowed with a deposit"],
    "insurance.pdf": ["Health insurance policy", "Policy number HX-55821", "Covers dental and vision care"],
    "boarding pass BLR-DEL.pdf": ["Gate 14 Seat 22A", "Departure 06:10"],
}
SCREENSHOTS = {"error.png": "Error: the disk is almost full. Free up space.",
               "wifi.png": "Home WiFi network sunflower, password tulip-garden-42"}
QUERIES = [
    ("when is my dentist appointment", "dentist.md"),
    ("what do I need to buy at the shop", "groceries.txt"),
    ("beach holiday plans", "trip-plan.md"),
    ("how much is the rent", "lease.pdf"),
    ("my salary in the job offer", "offer-letter.pdf"),
    ("HX-55821", "insurance.pdf"),
    ("screenshot of the disk space error", "error.png"),
    ("wifi password", "wifi.png"),
    ("voice memo about the plumber", "voice-memo.aiff"),
    ("a red square", "red.png"),
    ("boarding pass", "boarding pass BLR-DEL.pdf"),
    ("paneer recipe", "recipe.md"),
    ("budget meeting notes", "meeting-notes.md"),
    ("when is the car service due", "car-service.txt"),
    ("does my insurance cover dental", "insurance.pdf"),
    ("are pets allowed in the apartment", "lease.pdf"),
]
TOPICS = ["project Alpha", "the garden", "tax filing", "a podcast idea", "the school run", "a new laptop",
          "yoga classes", "the book club", "home repairs", "a birthday party", "running shoes", "the quarterly plan",
          "a conference talk", "a photography course", "learning Spanish", "the electricity bill"]
VERBS = ["Thoughts on", "Reminder about", "Draft notes for", "Questions about", "Ideas for", "Follow up on"]


def build(folder: pathlib.Path) -> None:
    from test_files import make_pdf
    for name, text in NOTES.items():
        (folder / name).write_text(text)
    rng = random.Random(3)
    for n in range(70):
        sentence = f"{rng.choice(VERBS)} {rng.choice(TOPICS)}. " + " ".join(
            rng.choice(TOPICS).split()[-1] for _ in range(8))
        (folder / f"note-{n:02d}.md").write_text(sentence)
    (folder / "expenses.csv").write_text("\n".join(f"2026-01-{d:02d},{rng.random() * 900:.2f},{rng.random() * 90:.2f}"
                                                   for d in range(1, 29)))
    for name, pages in PDFS.items():
        (folder / name).write_bytes(make_pdf(pages))
    for name, text in SCREENSHOTS.items():
        source = folder / f".{name}.pdf"
        source.write_bytes(make_pdf([text]))
        subprocess.run(["sips", "-s", "format", "png", str(source), "--out", str(folder / name)], capture_output=True)
        source.unlink()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=256x256", "-frames:v",
                    "1", str(folder / "red.png")], check=True)
    if shutil.which("say"):
        subprocess.run(["say", "-o", str(folder / "voice-memo.aiff"),
                        "Remember to call the plumber about the kitchen sink leak tomorrow morning."], check=True)


async def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp())
    os.environ["CODERUNNER_HOME"] = str(tmp / "crhome")
    from coderunner.audit import AuditLog
    from coderunner.config import Config
    from coderunner.files import Embedder, FileIndex
    from coderunner.store import Store
    folder = tmp / "Eval"
    folder.mkdir()
    build(folder)
    config = Config({"files": {"roots": [str(folder)]}}, path=tmp / "config.json")
    embedder = Embedder(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                        os.environ.get("EMBEDDING_MODEL", "embeddinggemma-2"))
    index = FileIndex(config, Store(":memory:"), AuditLog(tmp / "audit.jsonl"), embedder)
    start = time.monotonic()
    while (await index.index_once(500))["more"]:
        pass
    indexed = time.monotonic() - start
    ranks, latency = [], []
    for query, expected in QUERIES:
        await index.search(query)                       # the query vector is computed once, then cached
        index.searcher.embedder._queries.clear()
        began = time.monotonic()
        found = await index.search(query)
        latency.append(time.monotonic() - began)
        names = [pathlib.Path(r["path"]).name for r in found]
        rank = names.index(expected) + 1 if expected in names else None
        ranks.append(rank)
        print(f"{'ok ' if rank == 1 else 'MISS' if rank is None else f'#{rank} '} {query!r:42} -> "
              f"{names[:3]}{' (weak)' if found and found[0].get('weak') else ''}")
    hits1 = sum(r == 1 for r in ranks) / len(ranks)
    hits3 = sum(r is not None and r <= 3 for r in ranks) / len(ranks)
    mrr = statistics.fmean(1 / r if r else 0 for r in ranks)
    latency.sort()
    print(f"R@1 {hits1:.2f}  R@3 {hits3:.2f}  MRR {mrr:.2f}  | {len(QUERIES)} queries, "
          f"{index.status()['chunks']} pieces indexed in {indexed:.1f}s | search p50 "
          f"{statistics.median(latency) * 1000:.0f} ms, max {latency[-1] * 1000:.0f} ms")
    await index.media.close()
    await embedder.aclose()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
