"""The indexer: finds what changed in the shared folders, reads it, embeds it and keeps the index current.

Files are read a group at a time, several at once, and the next group is read while the current one is
embedded; each group is embedded in as few requests as fit and written in one transaction. The folders
are walked in a worker thread (a home folder takes tens of seconds), file events (macOS) bring a new or
changed file in within seconds, and indexing waits while the laptop is on a low battery or in Low Power
Mode. Vectors from a different embedding model, or stored in an older format, are replaced by indexing
again, since they cannot be compared with new ones.
"""

from __future__ import annotations

import asyncio
import dataclasses
import collections
import pathlib
import time

import httpx

from . import power
from .discover import Folders, Found
from .embed import Embedder
from .extract import NAME_ONLY, Extractor, Media, Piece, Transcriber
from .kinds import kind_of
from .search import Searcher
from .vectors import FORMAT, VectorIndex, code_bytes, pack

GROUP = 32                  # files per embedding round
PARALLEL_READS = 4          # files converted at once
NAME_BATCH = 512            # names of iCloud-only files embedded per pass, once the rest is done
FINGERPRINT = "files.fingerprint"
PAUSE_CHECK = 60.0


class FileIndex:
    def __init__(self, config, store, audit, embedder: Embedder | None = None, media: Media | None = None,
                 transcriber: Transcriber | None = None):
        self.config = config
        self.store = store
        self.audit = audit
        self.embedder = embedder
        self.media = media or Media()
        self.transcriber = transcriber
        self.folders = Folders(config)
        self.vectors = VectorIndex(store)
        self.searcher = Searcher(store, self.vectors, embedder)
        self._queue: collections.deque[Found] = collections.deque()
        self._files = 0
        self._lock = asyncio.Lock()
        self._fingerprint_checked = False
        self.wake = asyncio.Event()
        self.paused = ""
        self.watching = False

    # ---- sharing -------------------------------------------------------------------

    def roots(self) -> list[pathlib.Path]:
        return self.folders.roots()

    def resolve(self, path: str) -> pathlib.Path:
        return self.folders.resolve(path)

    def add_root(self, folder: str) -> str:
        real = self.folders.add(folder)
        self.wake.set()
        return real

    def remove_root(self, folder: str) -> int:
        self.folders.remove(folder)
        return self._forget_outside()

    @property
    def cloud_only(self) -> int:
        return self.folders.cloud_only

    # ---- indexing ------------------------------------------------------------------

    async def index_once(self, budget: int = 200) -> dict:
        """Index new or changed files (at most `budget` per pass) and forget removed ones. The folders are
        walked once, then the changed files are worked through over as many passes as it takes."""
        async with self._lock:
            await self._check_model()
            removed = 0
            if not self._queue:
                removed = await self._scan()
            batch = [self._queue.popleft() for _ in range(min(budget, len(self._queue)))]
            done, failed = await self._index(batch)
            waiting = False
            if not self._queue and await self._embed_names(max(NAME_BATCH, budget - len(batch))):
                waiting = self.store.one("SELECT 1 AS x FROM file_chunks WHERE modality = 'name' AND code IS NULL "
                                         "LIMIT 1") is not None
            result = {"indexed": done, "failed": failed, "removed": removed, "files": self._files,
                      "more": bool(self._queue) or waiting, "cloud_only": self.cloud_only}
            if done or failed or removed:
                self.audit.write("files.indexed", **result)
            return result

    async def _check_model(self) -> None:
        """Vectors from other weights, or in an older format, cannot be compared with new ones: index again."""
        if self.embedder is None or self._fingerprint_checked:
            return
        model = await self.embedder.fingerprint()
        if "@" not in model:            # the runtime did not say which weights: check again next time
            return
        fingerprint = f"{model}/{FORMAT}"
        stored = self.store.get_kv(FINGERPRINT)
        if stored != fingerprint:
            if self.store.one("SELECT 1 AS x FROM files LIMIT 1"):
                self.audit.write("files.reindex", previous=stored, now=fingerprint)
                with self.store.transaction():
                    self.store.execute("DELETE FROM file_chunks")
                    self.store.execute("DELETE FROM files")
                self._queue.clear()
                self.vectors.loaded = False
            self.store.set_kv(FINGERPRINT, fingerprint)
        self._fingerprint_checked = True

    async def _scan(self) -> int:
        known = {r["path"]: r for r in self.store.query(
            "SELECT path, size, mtime, (SELECT part FROM file_chunks c WHERE c.path = f.path LIMIT 1) "
            "AS part FROM files f")}

        def walk() -> tuple[list[Found], set[str]]:
            changed, seen = [], set()
            for found in self.folders.walk():
                key = str(found.path)
                seen.add(key)
                row = known.get(key)
                # Also redone: a file now on this Mac that was indexed by name while only iCloud had it.
                if not (row and row["size"] == found.stat.st_size and row["mtime"] == found.stat.st_mtime
                        and (row["part"] or "").startswith(NAME_ONLY) == found.remote):
                    changed.append(found)
            return changed, seen

        changed, seen = await asyncio.to_thread(walk)
        self._queue.extend(sorted(changed, key=lambda found: found.remote))     # what is on this Mac first
        self._files = len(seen)
        gone = [p for p in known if p not in seen]
        self._drop(gone)
        return len(gone)

    async def _extractor(self) -> Extractor:
        capabilities = await self.embedder.capabilities() if self.embedder else set()
        return Extractor(self.media, doc_prefix=self.embedder.doc_prefix if self.embedder else "",
                         capabilities=capabilities,
                         media_on=self.embedder is not None and bool(self.config.get("files.media", True)),
                         ocr_on=bool(self.config.get("files.ocr", True)),
                         transcriber=self.transcriber if self.config.get("files.transcribe", False) else None)

    async def _index(self, batch: list[Found]) -> tuple[int, int]:
        if not batch:
            return 0, 0
        extractor = await self._extractor()
        reads = asyncio.Semaphore(PARALLEL_READS)

        async def read(found: Found) -> tuple[Found, list[Piece], str]:
            async with reads:
                if found.remote:
                    # Findable by its words at once; its name is embedded later (_embed_names), after
                    # what is on this Mac. A home folder can name 100,000 files that only iCloud holds.
                    return found, [dataclasses.replace(extractor.name(found.path, remote=True), embed=None)], ""
                try:
                    return found, await extractor.pieces(found.path, kind_of(found.path)), ""
                except Exception as exc:          # the name is still worth finding
                    return found, [extractor.name(found.path)], str(exc)[:300]

        groups = [batch[i:i + GROUP] for i in range(0, len(batch), GROUP)]
        upcoming = asyncio.gather(*map(read, groups[0]))
        done = failed = 0
        try:
            for number in range(len(groups)):
                files = await upcoming
                # Read the next group while this one is embedded.
                upcoming = asyncio.gather(*map(read, groups[number + 1])) if number + 1 < len(groups) else None
                inputs = [piece.embed for _, pieces, _ in files for piece in pieces if piece.embed is not None]
                vectors = await self.embedder.embed(inputs) if self.embedder is not None and inputs else []
                self._write(files, vectors)
                done += sum(1 for _, _, error in files if not error)
                failed += sum(1 for _, _, error in files if error)
        except httpx.HTTPError as exc:
            # The model is unreachable: leave the rest for the next pass rather than index it without vectors.
            if upcoming is not None:
                upcoming.cancel()
            self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
        return done, failed

    def _write(self, files: list[tuple[Found, list[Piece], str]], vectors: list[list[float]]) -> None:
        stream = iter(vectors)
        added = []
        with self.store.transaction():
            self._drop([str(found.path) for found, _, _ in files], transaction=False)
            for found, pieces, error in files:
                kind = kind_of(found.path)
                self.store.execute("INSERT INTO files(path, kind, size, mtime, indexed, chunks, error) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?)", (str(found.path), kind, found.stat.st_size,
                                                                    found.stat.st_mtime, time.time(), len(pieces),
                                                                    error))
                for piece in pieces:
                    vector = next(stream, None) if piece.embed is not None else None
                    code = code_bytes(vector) if vector else None
                    row = self.store.execute(
                        "INSERT INTO file_chunks(path, part, text, embedding, code, modality, loc) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)", (str(found.path), piece.part, piece.text,
                                                         pack(vector) if vector else None, code, piece.modality,
                                                         piece.loc))
                    if vector:
                        added.append((row.lastrowid, code, piece.modality, kind))
                        self.vectors.learn(piece.modality, vector)
        if self.vectors.loaded:
            self.vectors.add(added)

    async def _embed_names(self, limit: int) -> int:
        """Embed names written without a vector (files only iCloud holds), a batch at a time."""
        if self.embedder is None:
            return 0
        rows = self.store.query("SELECT c.id, c.path, f.kind FROM file_chunks c JOIN files f ON f.path = c.path "
                                "WHERE c.modality = 'name' AND c.code IS NULL LIMIT ?", (limit,))
        if not rows:
            return 0
        extractor = await self._extractor()
        try:
            vectors = await self.embedder.embed([extractor.name(pathlib.Path(r["path"])).embed for r in rows])
        except httpx.HTTPError as exc:
            self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
            return 0
        added = []
        with self.store.transaction():
            for row, vector in zip(rows, vectors):
                code = code_bytes(vector)
                self.store.execute("UPDATE file_chunks SET embedding = ?, code = ? WHERE id = ?",
                                   (pack(vector), code, row["id"]))
                added.append((row["id"], code, "name", row["kind"]))
                self.vectors.learn("name", vector)
        if self.vectors.loaded:
            self.vectors.add(added)
        return len(rows)

    def _drop(self, paths: list[str], transaction: bool = True) -> None:
        if not paths:
            return
        if transaction:
            with self.store.transaction():
                self._drop(paths, transaction=False)
            return
        for start in range(0, len(paths), 500):
            part = paths[start:start + 500]
            marks = ",".join("?" * len(part))
            self.vectors.remove(r["id"] for r in self.store.query(
                f"SELECT id FROM file_chunks WHERE path IN ({marks})", part))
            self.store.execute(f"DELETE FROM file_chunks WHERE path IN ({marks})", part)
            self.store.execute(f"DELETE FROM files WHERE path IN ({marks})", part)

    def _forget_outside(self) -> int:
        gone = [r["path"] for r in self.store.query("SELECT path FROM files") if not self.folders.inside(r["path"])]
        self._drop(gone)
        return len(gone)

    async def changed(self, paths: set[str]) -> None:
        """File events: look at these paths again, ahead of everything else."""
        for raw in paths:
            path = pathlib.Path(raw)
            found = await asyncio.to_thread(self.folders.check, path) if path.is_file() else None
            if found is not None:
                self._queue.appendleft(found)
            elif not path.exists():
                self._drop([raw, str(path.parent.resolve() / path.name)])
        self.wake.set()

    async def run_forever(self) -> None:
        """Keep the index current for as long as the daemon runs."""
        if self.embedder is not None:
            await self.embedder.warm()
        await asyncio.to_thread(self.vectors.load)
        watch: asyncio.Task | None = None
        watched: list[pathlib.Path] | None = None
        try:
            while True:
                roots = self.roots()
                if roots != watched:
                    if watch is not None:
                        watch.cancel()
                    watch = asyncio.create_task(self.watcher_run(roots)) if roots else None
                    watched = roots
                self.paused = await power.pause_reason(self.config)
                more = False
                if roots and not self.paused:
                    try:
                        more = (await self.index_once())["more"]
                    except Exception as exc:  # keep indexing alive across a model restart
                        self.audit.write("files.index_error", error=str(exc)[:300])
                if more:
                    continue
                self.wake.clear()
                interval = float(self.config.get("files.scan_interval", 600) or 600)
                if self.paused:
                    interval = PAUSE_CHECK
                elif self.watching:
                    interval *= 6          # events bring changes in; the walk only catches what they miss
                try:
                    await asyncio.wait_for(self.wake.wait(), interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            if watch is not None:
                watch.cancel()
            await self.media.close()

    async def watcher_run(self, roots: list[pathlib.Path]) -> None:
        from .watch import Watcher
        self.watching = True
        try:
            self.watching = await Watcher(self.media).run(roots, self.changed)
        finally:
            self.watching = False

    # ---- search and status ---------------------------------------------------------

    async def search(self, query: str, kind: str | None = None, limit: int = 8) -> list[dict]:
        return await self.searcher.search(query, kind, limit)

    def status(self) -> dict:
        rows = self.store.query("SELECT kind, COUNT(*) AS n, SUM(error != '') AS failed FROM files GROUP BY kind")
        return {"roots": [str(r) for r in self.roots()], "model": self.embedder.model if self.embedder else None,
                "by_kind": {r["kind"]: {"files": r["n"], "failed": r["failed"] or 0} for r in rows},
                "chunks": (self.store.one("SELECT COUNT(*) AS n FROM file_chunks") or {}).get("n", 0),
                "in_memory": len(self.vectors), "queued": len(self._queue), "watching": self.watching,
                "paused": self.paused, "cloud_only_skipped": self.cloud_only}
