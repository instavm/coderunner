"""Embeddings on disk and in memory, and the fast first pass of a search.

On disk each vector is kept at half precision (1.5 KB for 768 dimensions) beside a 96-byte code of the
signs of its coordinates. In memory the index holds only the codes, as Python integers, plus a sample of
full vectors per modality. A search compares the query's code with every code (XOR and a bit count, both
run in C), takes the few hundred closest, and scores those exactly with their full vectors read from
disk. That keeps memory at about 100 bytes per piece and a search over 200,000 pieces in tens of
milliseconds, with no numerical library.

The sample serves calibration: for each query, how similar pieces of each modality (names, text, pictures,
sound) are on average, and how much similarities spread. Text-to-picture similarities sit lower than
text-to-text ones for the same query, so raw scores would bury pictures; a score is instead read as
standard deviations above its modality's average.
"""

from __future__ import annotations

import collections
import itertools
import json
import random
import statistics
import struct
from typing import Any, Iterable, Sequence

try:
    from math import sumprod as dot           # Python 3.12+: a C loop
except ImportError:                           # pragma: no cover
    import operator

    def dot(a, b) -> float:
        return sum(map(operator.mul, a, b))

FORMAT = "f16-signs-1"      # changes force a re-index
MODALITIES = ("name", "text", "image", "audio")
SAMPLE = 400                # full vectors kept per modality for calibration
MIN_GROUP = 20              # a modality with fewer pieces is calibrated with all of them
MIN_SPREAD = 0.02


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}e", *vector)


def unpack(value: Any) -> Sequence[float]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return memoryview(bytes(value)).cast("e")
    return json.loads(value)                  # rows written before half-precision storage


def signs(vector: Sequence[float]) -> int:
    code = 0
    for i, x in enumerate(vector):
        if x > 0:
            code |= 1 << i
    return code


def unit_mean(vectors: list[Sequence[float]]) -> list[float]:
    """The mean of vectors, scaled to unit length."""
    total = [sum(column) for column in zip(*vectors)]
    norm = sum(x * x for x in total) ** 0.5 or 1.0
    return [x / norm for x in total]


def code_bytes(vector: Sequence[float]) -> bytes:
    return signs(vector).to_bytes((len(vector) + 7) // 8, "little")


class VectorIndex:
    """Codes of every embedded piece, kept in memory for the first pass of a search."""

    def __init__(self, store):
        self.store = store
        self.loaded = False
        self.ids: list[int] = []
        self.codes: list[int] = []
        self.modality = bytearray()
        self.kind: list[str] = []
        self.row: dict[int, int] = {}
        self.dead: set[int] = set()
        self.sample: dict[str, list[Sequence[float]]] = {}
        self._groups: dict[int, list[int]] | None = None     # positions of each modality, rebuilt on change
        self._grouped = 0

    def __len__(self) -> int:
        return len(self.ids) - len(self.dead)

    def load(self) -> None:
        """Read every code from disk. Built aside and swapped in whole, so a search never sees half of it."""
        fresh = VectorIndex(self.store)
        rows = self.store.query("SELECT c.id, c.code, c.modality, f.kind FROM file_chunks c JOIN files f "
                                "ON f.path = c.path WHERE c.code IS NOT NULL")
        fresh.add((r["id"], r["code"], r["modality"], r["kind"]) for r in rows)
        for modality in MODALITIES:
            picked = self.store.query("SELECT embedding FROM file_chunks WHERE modality = ? AND code IS NOT NULL "
                                      "ORDER BY random() LIMIT ?", (modality, SAMPLE))
            fresh.sample[modality] = [unpack(r["embedding"]) for r in picked]
        (self.ids, self.codes, self.modality, self.kind, self.row, self.dead, self.sample) = (
            fresh.ids, fresh.codes, fresh.modality, fresh.kind, fresh.row, fresh.dead, fresh.sample)
        self._groups = None
        self.loaded = True

    def add(self, rows: Iterable[tuple[int, bytes, str, str]]) -> None:
        for chunk, code, modality, kind in rows:
            self.row[chunk] = len(self.ids)
            self.ids.append(chunk)
            self.codes.append(int.from_bytes(code, "little"))
            self.modality.append(MODALITIES.index(modality) if modality in MODALITIES else 1)
            self.kind.append(kind or "")

    def learn(self, modality: str, vector: Sequence[float]) -> None:
        """Keep a newly indexed vector in the calibration sample while it is small."""
        group = self.sample.setdefault(modality, [])
        if len(group) < SAMPLE:
            group.append(vector)
        elif random.random() < 0.01:
            group[random.randrange(SAMPLE)] = vector

    def remove(self, chunks: Iterable[int]) -> None:
        self.dead.update(c for c in chunks if c in self.row)
        if len(self.dead) > max(1000, len(self.ids) // 4):        # compact once a quarter is gone
            keep = [i for i, chunk in enumerate(self.ids) if chunk not in self.dead]
            self.ids = [self.ids[i] for i in keep]
            self.codes = [self.codes[i] for i in keep]
            self.modality = bytearray(self.modality[i] for i in keep)
            self.kind = [self.kind[i] for i in keep]
            self.row = {chunk: i for i, chunk in enumerate(self.ids)}
            self.dead = set()
            self._groups = None

    def nearest(self, query: Sequence[float], count: int, kinds: set[str] | None = None) -> list[int]:
        """Ids of about `count` pieces of each modality whose sign codes are closest to the query's. Each modality
        gets its own cut: a question's similarity to pictures and sound runs lower than to text, so one shared cut
        let almost only text through (two of 30 pictures in the retrieval benchmark never reached exact scoring)."""
        if not self.ids:
            return []
        distances = list(map(int.bit_count, map(signs(query).__xor__, self.codes)))
        wanted = count + (count * 4 if kinds else 0)
        found: list[int] = []
        for rows in self._by_modality().values():
            near = [distances[i] for i in rows]
            histogram = collections.Counter(near)
            limit, total = 0, 0
            for limit in sorted(histogram):
                total += histogram[limit]
                if total >= wanted + len(self.dead):
                    break
            found += [self.ids[i] for i in itertools.compress(rows, map(limit.__ge__, near))
                      if self.ids[i] not in self.dead and (not kinds or self.kind[i] in kinds)]
        return found

    def _by_modality(self) -> dict[int, list[int]]:
        if self._groups is None or self._grouped != len(self.ids):
            groups: dict[int, list[int]] = collections.defaultdict(list)
            for i, modality in enumerate(self.modality):
                groups[modality].append(i)
            self._groups, self._grouped = dict(groups), len(self.ids)
        return self._groups

    def calibration(self, query: Sequence[float]) -> tuple[dict[str, float], float]:
        """For this query: each modality's average similarity, and the spread of all similarities."""
        scores = {modality: [dot(query, v) for v in group if len(v) == len(query)]
                  for modality, group in self.sample.items()}
        everything = [s for group in scores.values() for s in group]
        if len(everything) < 2:
            return {modality: 0.0 for modality in MODALITIES}, 1.0
        overall = statistics.fmean(everything)
        means = {modality: statistics.fmean(scores[modality]) if len(scores.get(modality, [])) >= MIN_GROUP
                 else overall for modality in MODALITIES}
        centered = [s - means[modality] for modality, group in scores.items() for s in group]
        return means, max(statistics.pstdev(centered), MIN_SPREAD)

    def modality_of(self, chunk: int) -> str:
        return MODALITIES[self.modality[self.row[chunk]]] if chunk in self.row else "text"
