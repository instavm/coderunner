"""Finding files by what they are about.

The question is embedded and compared with every piece (vectors.VectorIndex: binary codes, then exact scores
for each modality's closest pieces). Similarities to names, passages, pictures and sound run on different
scales, so each is standardised against its own modality (a z-score over a fixed sample of it), and a file
scores the best of its pieces, plus one standard deviation when it is the kind the question names ("photo
of...", "...in a video"). Code passages are compared with the question in the model's code-retrieval form.

Words (SQLite's full-text index, BM25) rank questions made only of codes and numbers ("HX-55821"), which an
embedding cannot read, and find files with no embedded piece (exports made mostly of figures), after the rest.
Exact copies of a file count once.

Measured on 64 questions over a real Downloads folder and 25 held out from every choice (tests/retrieval_eval.py):
adding word evidence to the meaning score did not help at any weight (MRR 0.77 / 0.70 without, 0.76 / 0.70 at
0.5, 0.75 / 0.70 at 1.0), nor did gates or a cut of weak results; fusing a BM25 ranking with the meaning
ranking, by reciprocal ranks (0.50 / 0.57) or distribution-scaled scores (0.47 / 0.57), buried pictures,
video and sound under text sharing a word with the question. The kind bonus held (0.72 / 0.69 without it).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
from typing import Any

from .discover import in_cloud_only
from .extract import NAME_ONLY
from .kinds import is_code
from .vectors import VectorIndex, dot, unpack

CANDIDATES = 300            # pieces of each modality scored exactly after the binary first pass
HINT = 1.0                  # standard deviations added for the kind the question names
WORD_PIECES = 1000          # pieces read from the BM25 ranking
SMALL = 50                  # with fewer pieces than this, modalities cannot be standardised; raw similarity then
SNIPPET = 300
COPY_SAMPLE = 65536         # bytes compared from the start, middle and end of two files of one size to tell copies
MAX_COPIES = 5              # other places of a copied file named in its result
WORD = re.compile(r"\w+", re.UNICODE)
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
        self._fingerprint_lock = threading.Lock()

    async def search(self, query: str, kind: str | None = None, limit: int = 8) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 30))
        kinds = {kind} if kind else None
        words = terms(query)
        lookup = is_lookup(query.split())        # "HX-55821" as typed, not split into hx and 55821
        # Words are looked up while the query is being embedded.
        meaning, lexical = await asyncio.gather(
            self._meaning(query, kinds) if not lookup else asyncio.sleep(0, {}),
            asyncio.to_thread(self._words, words))
        paths = set(meaning) | {path for path, _ in lexical}
        if not paths:
            return []
        files = {r["path"]: r for r in self._rows(
            "SELECT path, kind, size FROM files WHERE path IN ({})", list(paths))}
        if kinds:
            meaning = {p: m for p, m in meaning.items() if files.get(p, {}).get("kind") in kinds}
            lexical = [(p, c) for p, c in lexical if files.get(p, {}).get("kind") in kinds]
        hints = kind_hints(query)
        scored = {path: z + (HINT if files.get(path, {}).get("kind") in hints else 0.0)
                  for path, (z, _) in meaning.items()}
        ranked = sorted(scored, key=lambda p: (-scored[p], p))
        chosen = {path: chunk for path, (_, chunk) in meaning.items()}
        # Then what only words find, in BM25 order: everything for a code, else files with no embedded piece.
        for path, chunk in lexical:
            if path not in chosen:
                ranked.append(path)
                chosen[path] = chunk
        ranked, copies = await asyncio.to_thread(self._copies, ranked, files, limit)
        return self._results(ranked, chosen, files, words, copies)

    def _copies(self, ranked: list[str], files: dict, limit: int) -> tuple[list[str], dict[str, list[str]]]:
        """Exact copies of a file ("file (1).pdf", a second checkout of a repository) count once: the best-ranked
        stands for the others and names them, so they do not fill the results. On 64 questions over a real
        Downloads folder, copies took a place above the right answer in 4."""
        kept: list[str] = []
        copies: dict[str, list[str]] = {}
        sizes: set = set()
        for path in ranked:                    # past the limit only to find copies of what was kept
            size = files.get(path, {}).get("size")
            if len(kept) >= limit and size not in sizes:
                continue
            twin = next((k for k in kept if size and files.get(k, {}).get("size") == size
                         and self._identical(k, path)), None)
            if twin:
                copies.setdefault(twin, []).append(path)
            elif len(kept) < limit:
                kept.append(path)
                sizes.add(size)
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
            # A file replaced with other bytes of the same size keeps its mtime after cp -p or an unzip; its
            # inode or change time does not.
            key = (path, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            with self._fingerprint_lock:
                known = self._fingerprints.get(key)
            if known is not None:
                return known
            with open(path, "rb") as handle:
                if stat.st_size <= 3 * COPY_SAMPLE:
                    sample = handle.read()
                else:
                    sample = b""
                    for start in (0, (stat.st_size - COPY_SAMPLE) // 2, stat.st_size - COPY_SAMPLE):
                        handle.seek(start)
                        sample += handle.read(COPY_SAMPLE)
            fingerprint = stat.st_size.to_bytes(8, "little") + hashlib.sha256(sample).digest()
            with self._fingerprint_lock:              # searches run this in threads of their own
                self._fingerprints[key] = fingerprint
                while len(self._fingerprints) > 4096:
                    self._fingerprints.pop(next(iter(self._fingerprints)))
            return fingerprint
        except OSError:
            return None

    async def _meaning(self, query: str, kinds: set[str] | None) -> dict[str, tuple[float, int]]:
        """Each file's best similarity, standardised within its piece's modality, and the piece that has it."""
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
        standardise = len(self.vectors) >= SMALL
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
            similarity = dot(code if as_code else vector, stored)
            if standardise:
                means, spread = calibrations["code" if as_code else "text"]
                similarity = (similarity - means.get(modality, 0.0)) / spread
            if similarity > best.get(row["path"], (float("-inf"), 0))[0]:
                best[row["path"]] = (similarity, row["id"])
        return best

    def _words(self, words: list[str]) -> list[tuple[str, int]]:
        """Files in BM25 order (SQLite FTS5) by their best piece, with that piece. Ranked inside the full-text
        index first and only then joined to the pieces: joined in one query, SQLite read the piece behind every
        match (6 s for a question whose words matched 130,000 of 332,000 pieces)."""
        if not words:
            return []
        expression = " OR ".join(f'"{w}"' for w in words[:16])
        ranked = [r["id"] for r in self.store.query(
            "SELECT rowid AS id FROM file_chunks_fts WHERE file_chunks_fts MATCH ? "
            "ORDER BY bm25(file_chunks_fts) LIMIT ?", (expression, WORD_PIECES))]
        owner = {r["id"]: r["path"] for r in self._rows("SELECT id, path FROM file_chunks WHERE id IN ({})", ranked)}
        found: dict[str, int] = {}
        for chunk in ranked:
            if chunk in owner:
                found.setdefault(owner[chunk], chunk)
        return list(found.items())

    def _rows(self, sql: str, values: list[Any]) -> list[dict]:
        found: list[dict] = []
        for start in range(0, len(values), 500):               # SQLite caps the number of parameters
            part = values[start:start + 500]
            found += self.store.query(sql.format(",".join("?" * len(part))), part)
        return found

    def _results(self, ranked, chosen, files, words, copies) -> list[dict]:
        pieces = {r["id"]: r for r in self._rows("SELECT id, part, text, loc, modality FROM file_chunks "
                                                 "WHERE id IN ({})", [chosen[p] for p in ranked if p in chosen])}
        results = []
        for path in ranked:
            piece = pieces.get(chosen.get(path)) or {}
            part = piece.get("part") or ""
            found: dict[str, Any] = {"path": path, "kind": files.get(path, {}).get("kind", ""), "part": part}
            if piece.get("loc") is not None:
                place = "page" if part.startswith("page") else "line" if part.startswith("from line") else "at"
                found[place] = int(piece["loc"]) if place != "at" else round(piece["loc"], 1)
            if piece.get("modality") == "text" and piece.get("text"):
                found["snippet"] = snippet(piece["text"], words)
            if part.startswith(NAME_ONLY):
                found["in_icloud"] = True
            if copies.get(path):
                found["copies"] = copies[path][:MAX_COPIES]
            results.append(found)
        return results
