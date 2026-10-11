"""Long-term memory: short notes the owner wants kept, searchable by words and meaning.

Search is hybrid. SQLite FTS5 (bm25) finds keyword matches; when an embedding
model is configured, cosine similarity over stored vectors finds paraphrases.
The two rankings are merged with reciprocal-rank fusion, so neither score scale
has to be calibrated against the other. If the embedding model is missing or
down, everything keeps working on keywords alone.
"""

from __future__ import annotations

import heapq
import json
import math
import operator
import re
import time
from typing import Awaitable, Callable

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]

MAX_CHARS = 4000
MAX_TERMS = 32
RRF_K = 60
_WORD = re.compile(r"\w+")
_COLUMNS = "id, text, tags, created, source"
_dot = getattr(math, "sumprod", None) or (lambda a, b: sum(map(operator.mul, a, b)))


def fts_query(text: str, *, all_terms: bool = False) -> str:
    """Turn free text into a safe FTS5 expression of quoted terms ('' if none).

    Only word characters survive and every term is double-quoted, so operators
    (AND, NEAR, *, ^, column filters) are matched as plain words, never parsed.
    """
    terms = list(dict.fromkeys(t.lower() for t in _WORD.findall(text or "")))[:MAX_TERMS]
    return (" AND " if all_terms else " OR ").join(f'"{t}"' for t in terms)


def _unit(vector) -> list[float] | None:
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(v * v for v in values))
    if not values or not math.isfinite(norm) or norm == 0:
        return None
    return [round(v / norm, 6) for v in values]


class Memory:
    def __init__(self, store, audit, embedder: Embedder | None = None):
        self.store = store
        self.audit = audit
        self.embedder = embedder
        self._embed_down = False
        self._vectors: dict[int, list[float]] = {}  # id -> unit vector, mirrors the embedding column

    async def save(self, text: str, tags: str | list[str] = "", source: str = "") -> dict:
        """source is empty for the owner's own facts, "untrusted" when saved after reading third-party content."""
        text = (text or "").strip()
        if isinstance(tags, (list, tuple)):
            tags = ", ".join(str(t).strip() for t in tags if str(t).strip())
        tags = (tags or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        if len(text) > MAX_CHARS:
            raise ValueError(f"memory text is {len(text)} characters; the limit is {MAX_CHARS}")
        if existing := self._find(text):
            if existing.get("source") and not source:
                # The owner's own run confirms a note that was saved as unverified.
                self.store.execute("UPDATE memories SET source = '' WHERE id = ?", (existing["id"],))
                existing["source"] = ""
            return {**existing, "duplicate": True}
        vectors = await self._embed([text])
        if existing := self._find(text):  # saved by someone else while we awaited the embedder
            return {**existing, "duplicate": True}
        vector = vectors[0] if vectors else None
        cursor = self.store.execute(
            "INSERT INTO memories(text, tags, created, embedding, source) VALUES (?, ?, ?, ?, ?)",
            (text, tags, time.time(), json.dumps(vector) if vector else None, source),
        )
        memory_id = cursor.lastrowid
        if vector:
            self._vectors[memory_id] = vector
        self.audit.write("memory.save", id=memory_id, chars=len(text), tags=tags, embedded=bool(vector),
                         source=source)
        return {**self._get(memory_id), "duplicate": False}

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 50))
        pool = max(limit * 4, 20)
        rankings: list[list[int]] = []
        if expr := fts_query(query):
            rows = self.store.query(
                "SELECT rowid AS id FROM memories_fts WHERE memories_fts MATCH ? "
                "ORDER BY bm25(memories_fts, 1.0, 0.5) LIMIT ?",
                (expr, pool),
            )
            rankings.append([row["id"] for row in rows])
        if self.embedder and self._refresh_vectors():
            vectors = await self._embed([query])
            if vectors:
                rankings.append(self._nearest(vectors[0], pool))

        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, memory_id in enumerate(ranking, 1):
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(scores, key=lambda i: (-scores[i], -i))[:limit]
        if not top:
            return []
        marks = ",".join("?" * len(top))
        rows = {r["id"]: r for r in self.store.query(f"SELECT {_COLUMNS} FROM memories WHERE id IN ({marks})", top)}
        return [{**rows[i], "score": round(scores[i], 6)} for i in top if i in rows]

    async def recall(self, text: str, limit: int = 5, min_similarity: float = 0.5) -> list[dict]:
        """Memories clearly relevant to text, for the assistant's prompt.

        search() ranks anything that matches at all. recall() returns only memories close in
        meaning to the text, so personal facts do not enter tasks that never needed them.
        """
        text = (text or "").strip()
        if not text:
            return []
        if self.embedder and self._refresh_vectors():
            vectors = await self._embed([text])
            if vectors:
                return self._rows(self._nearest(vectors[0], limit, min_score=min_similarity))
        return []        # without embeddings the assistant decides when to call memory_search

    def _rows(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in self.store.query(f"SELECT {_COLUMNS} FROM memories WHERE id IN ({marks})", ids)}
        return [rows[i] for i in ids if i in rows]

    def forget(self, id: int | None = None, query: str | None = None) -> int:
        """Delete one memory by id, or every memory containing all words of query."""
        if (id is None) == (query is None):
            raise ValueError("pass exactly one of id or query")
        if id is not None:
            ids = [r["id"] for r in self.store.query("SELECT id FROM memories WHERE id = ?", (int(id),))]
        else:
            # AND semantics: "forget my old address" must not wipe every note containing "my".
            expr = fts_query(query, all_terms=True)
            rows = self.store.query("SELECT rowid AS id FROM memories_fts WHERE memories_fts MATCH ?", (expr,)) if expr else []
            ids = [r["id"] for r in rows]
        for memory_id in ids:
            self.store.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._vectors.pop(memory_id, None)
        self.audit.write("memory.forget", by="id" if id is not None else "query", query=query, ids=ids, count=len(ids))
        return len(ids)

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        return self.store.query(f"SELECT {_COLUMNS} FROM memories ORDER BY created DESC, id DESC LIMIT ?", (limit,))

    async def backfill(self, batch: int = 32) -> int:
        """Embed memories saved while the embedding model was unavailable."""
        if not self.embedder:
            return 0
        rows = self.store.query("SELECT id, text FROM memories WHERE embedding IS NULL ORDER BY id LIMIT ?", (batch,))
        vectors = await self._embed([r["text"] for r in rows]) if rows else None
        if not vectors:
            return 0
        for row, vector in zip(rows, vectors):
            # Only the embedding column changes, so the FTS index stays valid.
            self.store.execute("UPDATE memories SET embedding = ? WHERE id = ?", (json.dumps(vector), row["id"]))
            self._vectors[row["id"]] = vector
        return len(rows)

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embedder:
            return None
        try:
            vectors = [_unit(v) for v in await self.embedder(texts)]
            if len(vectors) != len(texts) or any(v is None for v in vectors):
                raise ValueError("embedder returned malformed vectors")
        except Exception as exc:
            if not self._embed_down:  # audit once per outage, not on every call
                self._embed_down = True
                self.audit.write("memory.embedding_unavailable", error=f"{type(exc).__name__}: {exc}")
            return None
        self._embed_down = False
        return vectors

    def _refresh_vectors(self) -> bool:
        """Sync the in-process vector cache with the table; True if any vectors exist."""
        ids = {r["id"] for r in self.store.query("SELECT id FROM memories WHERE embedding IS NOT NULL")}
        for stale in self._vectors.keys() - ids:
            del self._vectors[stale]
        if ids - self._vectors.keys():
            for row in self.store.query("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"):
                if row["id"] not in self._vectors:
                    try:
                        vector = _unit(json.loads(row["embedding"]))
                    except (TypeError, ValueError):
                        vector = None
                    self._vectors[row["id"]] = vector or []  # [] marks unreadable, so we don't re-parse it
        return any(self._vectors.values())

    def _nearest(self, vector: list[float], limit: int, min_score: float = 0.0) -> list[int]:
        scored = (
            (_dot(vector, stored), memory_id)
            for memory_id, stored in self._vectors.items()
            if len(stored) == len(vector)  # skip vectors from a different embedding model
        )
        return [memory_id for score, memory_id in heapq.nlargest(limit, scored) if score > max(min_score, 0)]

    def _find(self, text: str) -> dict | None:
        return self.store.one(f"SELECT {_COLUMNS} FROM memories WHERE text = ? ORDER BY id LIMIT 1", (text,))

    def _get(self, memory_id: int) -> dict | None:
        return self.store.one(f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,))
