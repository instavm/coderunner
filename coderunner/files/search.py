"""Finding files by what they are about, and by the words in them.

The query is embedded and compared with every piece (vectors.VectorIndex: a fast first pass, then exact
scores). A piece's similarity is read against its modality for this query, in standard deviations above
the average name, passage, picture or sound, so a photo that matches well is not buried under text that
matches a little. Words are matched with SQLite's full-text index over names, passages, PDF pages, text
read from pictures and transcripts, and a file's word score is the share of the query's words it holds.

A file qualifies by standing out in meaning or by holding most of the query's words. Its score adds both,
plus a bonus when it is the kind the query asks for ("photo of...", "...in a video"). Queries made of codes
and numbers ("INV-2026-41") lean on words, which an embedding cannot read. Results far behind the best are
cut: a search returns a few strong answers rather than thirty weak ones. Exact copies of a file count once.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Any

from .discover import in_cloud_only
from .extract import NAME_ONLY
from .kinds import is_code
from .vectors import VectorIndex, dot, unpack

CANDIDATES = 300            # pieces scored exactly after the first pass
Z_GATE = 2.0                # standard deviations above its modality's average for a meaning match to count
MIN_COVERAGE = 0.5          # share of the query's words for a word match to count
# Measured on 64 plain-language questions over a real Downloads folder (tests/retrieval_eval.py): a word weight of 2
# let text that shares a few common words outrank the picture or video that matched in meaning (MRR 0.74);
# 0.5 gives 0.78 (R@1 0.64 -> 0.69). Exact lookups ("HX-55821", a file name) keep a strong word weight.
MEANING, WORDS, HINT = 1.0, 0.5, 1.0
LOOKUP_MEANING, LOOKUP_WORDS = 0.5, 4.0
STRONG = 0.5                # results scoring below this share of the best are cut
SMALL = 50                  # with fewer pieces than this, similarities cannot be calibrated; rank instead
SNIPPET = 300
COPY_SAMPLE = 65536         # bytes compared from the start, middle and end of two files of one size to tell copies
MAX_COPIES = 5              # other places of a copied file named in its result
WORD =re.compile(r"\w+", re.UNICODE)
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "my", "me",
             "i", "is", "are", "was", "were", "it", "this", "that", "what", "where", "when", "which", "who",
             "about", "show", "find", "get", "some", "any", "all", "do", "did", "have", "has", "had"}
KIND_WORDS = {
    "photo": "image", "photos": "image", "picture": "image", "pictures": "image", "pic": "image", "pics": "image",
    "image": "image", "images": "image", "screenshot": "image", "screenshots": "image", "selfie": "image",
    "video": "video", "videos": "video", "clip": "video", "clips": "video", "movie": "video", "footage": "video",
    "audio": "audio", "song": "audio", "songs": "audio", "music": "audio", "track": "audio", "podcast": "audio",
    "recording": "audio", "voice": "audio", "pdf": "doc", "document": "doc", "documents": "doc", "letter": "doc",
    "contract": "doc", "note": "text", "notes": "text",
}


def terms(query: str) -> list[str]:
    words = list(dict.fromkeys(w.lower() for w in WORD.findall(query)))
    kept = [w for w in words if w not in STOPWORDS]
    return kept or words


def kind_hints(query: str) -> set[str]:
    return {KIND_WORDS[w] for w in WORD.findall(query.lower()) if w in KIND_WORDS}


def is_lookup(words: list[str]) -> bool:
    """Codes and numbers ("K7Q2", "4815162342") mean nothing to an embedding and everything to keywords."""
    return bool(words) and all(any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4)
                               for w in words)


def snippet(text: str, words: list[str]) -> str:
    lowered = text.lower()
    first = min((i for i in (lowered.find(w) for w in words) if i >= 0), default=0)
    start = max(0, first - SNIPPET // 3)
    return ("…" if start else "") + " ".join(text[start:start + SNIPPET].split())


class Searcher:
    def __init__(self, store, vectors: VectorIndex, embedder=None):
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self._fingerprints: dict[tuple, bytes] = {}

    async def search(self, query: str, kind: str | None = None, limit: int = 8) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 30))
        kinds = {kind} if kind else None
        words = terms(query)
        # Words are looked up while the query is being embedded.
        meaning, coverage = await asyncio.gather(self._meaning(query, kinds),
                                                 asyncio.to_thread(self._words, words, kinds))
        paths = set(meaning) | set(coverage)
        if not paths:
            return []
        files = {r["path"]: r for r in self._rows(
            "SELECT path, kind, size FROM files WHERE path IN ({})", list(paths))}
        if kinds:
            paths = {p for p in paths if p in files and files[p]["kind"] in kinds}
        calibrated = len(self.vectors) >= SMALL
        lookup = is_lookup(query.split())        # "HX-55821" as typed, not split into hx and 55821
        weight_meaning, weight_words = (LOOKUP_MEANING, LOOKUP_WORDS) if lookup else (MEANING, WORDS)
        hints = kind_hints(query)
        scored: dict[str, float] = {}
        for path in paths:
            z = meaning.get(path, (0.0, None))[0]
            share = coverage.get(path, (0.0, None))[0]
            if calibrated and z < Z_GATE and share < MIN_COVERAGE:
                continue
            score = weight_meaning * (max(z, 0.0) if calibrated else z) + weight_words * share ** 1.5
            if files.get(path, {}).get("kind") in hints:
                score += HINT
            scored[path] = score
        weak = not scored
        if weak:          # nothing stands out: offer the closest few, marked as weak
            scored = {p: meaning[p][0] for p in sorted(meaning, key=lambda p: -meaning[p][0])[:3]}
        ranked = sorted(scored, key=lambda p: (-scored[p], p))
        top = scored[ranked[0]] if ranked else 0.0
        if top > 0 and not weak:
            ranked = [p for p in ranked if scored[p] >= top * STRONG]
        ranked, copies = await asyncio.to_thread(self._copies, ranked, files, limit)
        return self._results(ranked, scored, meaning, coverage, files, words, weak, calibrated, copies)

    def _copies(self, ranked: list[str], files: dict, limit: int) -> tuple[list[str], dict[str, list[str]]]:
        """Exact copies of a file ("file (1).pdf", a second checkout of a repository) count once: the best-ranked
        stands for the others and names them, so they do not fill the results. On 64 questions over a real
        Downloads folder, copies took a place above the right answer in 4."""
        kept: list[str] = []
        copies: dict[str, list[str]] = {}
        for path in ranked:
            if len(kept) >= limit:
                break
            size = files.get(path, {}).get("size")
            twin = next((k for k in kept if size and files.get(k, {}).get("size") == size
                         and self._identical(k, path)), None)
            if twin:
                copies.setdefault(twin, []).append(path)
            else:
                kept.append(path)
        return kept, copies

    def _identical(self, a: str, b: str) -> bool:
        return (fingerprint := self._fingerprint(a)) is not None and fingerprint == self._fingerprint(b)

    def _fingerprint(self, path: str) -> bytes | None:
        """The file's size and bytes from its start, middle and end; None for a file only iCloud holds (reading
        it would download it) or one that is gone."""
        try:
            stat = os.stat(path)
            if in_cloud_only(stat):
                return None
            key = (path, stat.st_mtime_ns, stat.st_size)
            if key not in self._fingerprints:
                with open(path, "rb") as handle:
                    if stat.st_size <= 3 * COPY_SAMPLE:
                        sample = handle.read()
                    else:
                        sample = b""
                        for start in (0, (stat.st_size - COPY_SAMPLE) // 2, stat.st_size - COPY_SAMPLE):
                            handle.seek(start)
                            sample += handle.read(COPY_SAMPLE)
                self._fingerprints[key] = stat.st_size.to_bytes(8, "little") + hashlib.sha256(sample).digest()
                while len(self._fingerprints) > 4096:
                    self._fingerprints.pop(next(iter(self._fingerprints)))
            return self._fingerprints[key]
        except OSError:
            return None

    async def _meaning(self, query: str, kinds: set[str] | None) -> dict[str, tuple[float, int]]:
        """Each file's best calibrated similarity to the query, and the piece that has it."""
        if self.embedder is None:
            return {}
        if not self.vectors.loaded:
            self.vectors.load()
        if not len(self.vectors):
            return {}
        # Code passages are compared with the question as the model's code retrieval prompt puts it; both
        # forms of the question go in one request.
        code_prefix = getattr(self.embedder, "code_prefix", "")
        if code_prefix:
            vector, code = await self.embedder.queries(query, [self.embedder.query_prefix, code_prefix])
        else:
            vector, code = await self.embedder.query(query), None
        chunks = self.vectors.nearest(vector, CANDIDATES, kinds)
        if code is not None:
            chunks = list(dict.fromkeys([*chunks, *self.vectors.nearest(code, CANDIDATES, kinds)]))
        if not chunks:
            return {}
        calibrations = {"text": self.vectors.calibration(vector)}
        if code is not None:
            calibrations["code"] = self.vectors.calibration(code)
        best: dict[str, tuple[float, int]] = {}
        for row in self._rows("SELECT id, path, embedding, modality FROM file_chunks WHERE id IN ({})", chunks):
            stored = unpack(row["embedding"])
            if len(stored) != len(vector):
                continue
            modality = row["modality"] or "text"
            as_code = code is not None and modality == "text" and is_code(row["path"])
            asked = code if as_code else vector
            means, spread = calibrations["code" if as_code else "text"]
            z = (dot(asked, stored) - means.get(modality, 0.0)) / spread
            if z > best.get(row["path"], (float("-inf"), 0))[0]:
                best[row["path"]] = (z, row["id"])
        return best

    def _words(self, words: list[str], kinds: set[str] | None) -> dict[str, tuple[float, int]]:
        """Each file's share of the query's words (over all its matched pieces), and its best-matching piece."""
        if not words:
            return {}
        expression = " OR ".join(f'"{w}"' for w in words[:16])
        rows = self.store.query("SELECT c.id, c.path, c.text FROM file_chunks_fts JOIN file_chunks c "
                                "ON c.id = file_chunks_fts.rowid WHERE file_chunks_fts MATCH ? "
                                "ORDER BY bm25(file_chunks_fts) LIMIT 400", (expression,))
        held: dict[str, set[str]] = {}
        first: dict[str, int] = {}
        wanted = set(words)
        for row in rows:
            held.setdefault(row["path"], set()).update(wanted & set(WORD.findall(row["text"].lower())))
            first.setdefault(row["path"], row["id"])
        return {path: (len(found) / len(wanted), first[path]) for path, found in held.items() if found}

    def _rows(self, sql: str, values: list[Any]) -> list[dict]:
        found: list[dict] = []
        for start in range(0, len(values), 500):               # SQLite caps the number of parameters
            part = values[start:start + 500]
            found += self.store.query(sql.format(",".join("?" * len(part))), part)
        return found

    def _results(self, ranked, scored, meaning, coverage, files, words, weak, calibrated, copies) -> list[dict]:
        chosen = {}
        for path in ranked:
            z, by_meaning = meaning.get(path, (0.0, None))
            by_words = coverage.get(path, (0.0, None))[1]
            # Point at the piece that matched by meaning when it stands out, else at the one with the words.
            chosen[path] = by_meaning if by_meaning is not None and (z >= Z_GATE or not calibrated or by_words is None) \
                else by_words
        pieces = {r["id"]: r for r in self._rows("SELECT id, part, text, loc, modality FROM file_chunks "
                                                 "WHERE id IN ({})", [c for c in chosen.values() if c is not None])}
        results = []
        for path in ranked:
            piece = pieces.get(chosen[path]) or {}
            part = piece.get("part") or ""
            found: dict[str, Any] = {"path": path, "kind": files.get(path, {}).get("kind", ""),
                                     "part": part, "score": round(scored[path], 3)}
            if piece.get("loc") is not None:
                place = "page" if part.startswith("page") else "line" if part.startswith("from line") else "at"
                found[place] = int(piece["loc"]) if place != "at" else round(piece["loc"], 1)
            if piece.get("modality") == "text" and piece.get("text"):
                found["snippet"] = snippet(piece["text"], words)
            if part.startswith(NAME_ONLY):
                found["in_icloud"] = True
            if copies.get(path):
                found["copies"] = copies[path][:MAX_COPIES]
            if weak:
                found["weak"] = True
            results.append(found)
        return results
