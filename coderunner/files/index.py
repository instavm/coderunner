"""The indexer: finds what changed in the shared folders, reads it, embeds it and keeps the index current.

Files are read a group at a time, several at once, and the next group is read while the current one is
embedded; each group is embedded in as few requests as fit and written in one transaction. The folders
are walked in a worker thread (a home folder takes tens of seconds), file events (macOS) bring a new or
changed file in within seconds, and indexing waits while the laptop is on a low battery or in Low Power
Mode. Vectors from a different embedding model, or stored in an older format, are replaced by indexing
again, since they cannot be compared with new ones.

Each file's contents are indexed once: a copy of a file already read (a second checkout, "name (1).pdf", an
export unpacked twice) takes that file's pieces and vectors, and only its name is embedded; in a real
Downloads folder 71% of the files were copies. A file that moved or was renamed keeps its pieces (its
identity on its volume stays the same), and only its name is embedded again when that changed. Folders are
indexed as entries of their own, found by their names and the kinds of file in them. Small files only iCloud
holds are downloaded to be read and given back to iCloud afterwards.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import datetime
import os
import pathlib
import sys
import time

import httpx

from . import power
from .discover import Folders, Found, content_fingerprint, in_cloud_only
from .embed import Embedder
from .extract import NAME_ONLY, Extractor, Media, Piece, Skipped, Transcriber
from .kinds import MEDIA, is_code
from .search import Searcher
from .sources import Sources
from .vectors import FORMAT, VectorIndex, code_bytes, pack, unit_mean

GROUP = 32                  # files per embedding round
PARALLEL_READS = 4          # files converted at once
NAME_BATCH = 512            # names written without a vector, embedded per pass once the rest is done
FINGERPRINT = "files.fingerprint"
PAUSE_CHECK = 60.0
EVICTION_DELAY = 90.0       # seconds after reading before a downloaded file is given back to iCloud (asked right
                            # after the read, iCloud reports the eviction and keeps the file)
DOWNLOAD_WAIT = 120.0       # seconds to wait for iCloud to bring a file's bytes
NEVER = b""                 # the embedding of a piece that gets no vector (code stays NULL; a pending piece has both NULL)


def text_part(fingerprint: str | None) -> str | None:
    """The part of an index fingerprint that text vectors depend on: the model, its text weights and the format.
    None when the fingerprint does not name its parts (every change then means indexing everything again)."""
    if not fingerprint or "|media:" not in fingerprint:
        return None
    model_and_text, _, rest = fingerprint.partition("|media:")
    return model_and_text + "/" + rest.rpartition("/")[2]


def priority(found: Found) -> tuple:
    """Indexing order: what is on this Mac first; notes, documents, screenshots and folders (quick) before
    pictures, sound and video (hours for a big folder), then source code, and git checkouts, mostly other
    people's code, last; newest first within each. In a real Downloads folder, 4,765 exported notes were queued
    behind 146,000 files of cloned projects."""
    heavy = found.kind in MEDIA and not found.screenshot
    return (found.remote or found.evicted, found.in_repo, not found.folder and is_code(found.path), heavy,
            -found.mtime)


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
        self.sources = Sources(config, store, audit, embedder.doc_prefix if embedder else "")
        self._queue: collections.deque[Found] = collections.deque()
        self._waiting: dict[str, Found] = {}      # pictures, sound, video and downloads held back while on battery
        self._downloaded: list[tuple[str, float, int]] = []   # files fetched from iCloud, to give back
        self._files = 0
        self._lock = asyncio.Lock()
        self._fingerprint_checked = False
        self._names_checked = False
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
            self._ensure_names()
            removed = 0
            hold = await self._media_waits()
            if self._waiting and not hold:
                self._queue.extend(self._waiting.values())
                self._waiting.clear()
            if not self._queue and not (hold and self._waiting):     # no walk of every folder while media waits
                removed = await self._scan()
            batch: list[Found] = []
            while self._queue and len(batch) < budget:
                found = self._queue.popleft()
                if hold and ((found.kind in MEDIA and not found.screenshot) or found.evicted):
                    self._waiting[str(found.path)] = found
                else:
                    batch.append(found)
            done, failed = await self._index(batch)
            waiting = False
            if not self._queue and await self._embed_names(max(NAME_BATCH, budget - len(batch))):
                waiting = self.store.one("SELECT 1 AS x FROM file_chunks WHERE modality = 'name' AND code IS NULL "
                                         "AND embedding IS NULL LIMIT 1") is not None
            entries = 0
            if not self._queue and self.embedder is not None:
                entries, removed_entries = await self._index_sources(max(1, budget - len(batch)))
                removed += removed_entries
                waiting = waiting or any(self.sources._more.values())
            await self._release_downloads()
            result = {"indexed": done, "failed": failed, "removed": removed, "files": self._files,
                      "entries": entries, "more": bool(self._queue) or waiting, "cloud_only": self.cloud_only}
            if done or failed or removed:
                self.audit.write("files.indexed", **result)
            return result

    async def _media_waits(self) -> bool:
        """Whether pictures, sound, video and iCloud downloads wait for the charger (files.media_on_battery turns
        this off)."""
        if self.embedder is None or not self.config.get("files.media", True):
            return False
        if self.config.get("files.media_waits", False):     # the owner holds them back (e.g. to keep the GPU idle)
            return True
        if self.config.get("files.media_on_battery", False):
            return False
        return await power.on_battery()

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
                media_only = text_part(stored) is not None and text_part(stored) == text_part(fingerprint)
                self.audit.write("files.reindex", previous=stored, now=fingerprint, media_only=media_only)
                with self.store.transaction():
                    if media_only:
                        # Same text weights: only pictures, sound and video are embedded anew.
                        kinds = ", ".join("?" for _ in MEDIA)
                        self.store.execute(f"DELETE FROM file_chunks WHERE path IN (SELECT path FROM files "
                                           f"WHERE kind IN ({kinds}))", MEDIA)
                        self.store.execute(f"DELETE FROM files WHERE kind IN ({kinds})", MEDIA)
                    else:
                        self.store.execute("DELETE FROM file_chunks")
                        self.store.execute("DELETE FROM files")
                self._queue.clear()
                self.vectors.loaded = False
            self.store.set_kv(FINGERPRINT, fingerprint)
        self._fingerprint_checked = True

    def _ensure_names(self) -> None:
        """The names of an index from before the ranking by names, once: each name piece's own name and kind."""
        if self._names_checked:
            return
        self._names_checked = True
        if self.store.one("SELECT 1 AS x FROM file_names_src LIMIT 1"):
            return
        rows = self.store.query("SELECT c.id, c.path, f.kind FROM file_chunks c JOIN files f ON f.path = c.path "
                                "WHERE c.modality = 'name'")
        if not rows:
            return
        with self.store.transaction():
            for row in rows:
                self.store.execute("INSERT OR REPLACE INTO file_names_src(id, text) VALUES (?, ?)",
                                   (row["id"], Extractor.own_name(pathlib.Path(row["path"]), row["kind"])))

    @staticmethod
    def _matches(row: dict, found: Found) -> bool:
        """Whether an indexed row describes `found` as it is now."""
        if row["kind"] != found.kind or (row["profile"] or "") != found.profile:
            return False
        if found.folder:
            return row["mtime"] == found.mtime
        return row["size"] == found.size and row["mtime"] == found.mtime \
            and (row["part"] or "").startswith(NAME_ONLY) == found.remote

    async def _scan(self) -> int:
        # Entries from the app sources (mail://..., note://...) are not the walk's to find or forget.
        def read_known():
            # On a connection of its own, off the event loop: with 107,000 files this read took 8 s, and through
            # the shared connection every scan (one every ten minutes, and one when a folder is added) stopped
            # the daemon answering for that long.
            with self.store.reader() as db:
                return {r["path"]: dict(r) for r in db.execute(
                    "SELECT path, kind, size, mtime, profile, device, inode, "
                    "(SELECT part FROM file_chunks c WHERE c.path = f.path LIMIT 1) AS part FROM files f "
                    "WHERE path NOT LIKE '%://%'")}
        known = await asyncio.to_thread(read_known)

        def walk():
            changed, seen, identities = [], set(), []
            files = 0
            for found in self.folders.walk():
                key = str(found.path)
                seen.add(key)
                files += not found.folder
                row = known.get(key)
                if row and self._matches(row, found):
                    if not row["inode"] and found.inode:
                        identities.append((found.device, found.inode, key))
                    continue
                changed.append(found)
            gone = {p for p in known if p not in seen}
            # A new path holding a file that is gone from its old path: the same file, moved or renamed.
            by_identity = {}
            for path in gone:
                row = known[path]
                if row["inode"]:
                    by_identity[(row["device"], row["inode"], row["size"], row["mtime"], row["kind"])] = path
            moves = []
            if by_identity:
                kept = []
                for found in changed:
                    key = str(found.path)
                    old = by_identity.get((found.device, found.inode, found.size, found.mtime, found.kind)) \
                        if known.get(key) is None and found.inode else None
                    if old and self._matches(known[old], found):
                        moves.append((old, found))
                        del by_identity[(found.device, found.inode, found.size, found.mtime, found.kind)]
                        gone.discard(old)
                    else:
                        kept.append(found)
                changed = kept
            return changed, files, identities, gone, moves

        changed, files, identities, gone, moves = await asyncio.to_thread(walk)
        self._queue.extend(sorted(changed, key=priority))
        self._files = files
        if identities:
            with self.store.transaction():
                for device, inode, path in identities:
                    self.store.execute("UPDATE files SET device = ?, inode = ? WHERE path = ?", (device, inode, path))
        moved = 0
        for old, found in moves:
            if self._move(old, found):
                moved += 1
            else:
                self._queue.appendleft(found)
                gone.add(old)
        if moved:
            self.audit.write("files.moved", count=moved)
        self._drop(list(gone))
        return len(gone)

    def _move(self, old: str, found: Found) -> bool:
        """A file at a new path takes over its pieces. Only its name is embedded again, and only when it
        changed (a file moved to another folder keeps the vector of its name)."""
        new = str(found.path)
        try:
            with self.store.transaction():
                if self.store.one("SELECT 1 AS x FROM files WHERE path = ?", (new,)):
                    return False
                self.store.execute("UPDATE files SET path = ?, device = ?, inode = ? WHERE path = ?",
                                   (new, found.device, found.inode, old))
                self.store.execute("UPDATE file_chunks SET path = ? WHERE path = ?", (new, old))
                if os.path.basename(old) != found.path.name:
                    rows = self.store.query("SELECT id FROM file_chunks WHERE path = ? AND modality = 'name'", (new,))
                    ids = [r["id"] for r in rows]
                    self.vectors.remove(ids)
                    for chunk in ids:
                        self.store.execute("DELETE FROM file_chunks WHERE id = ?", (chunk,))
                    piece = self._name_piece(found)
                    row = self.store.execute(
                        "INSERT INTO file_chunks(path, part, text, embedding, code, modality, loc) VALUES (?, ?, ?, "
                        "NULL, NULL, 'name', NULL)", (new, piece.part, piece.text))
                    self.store.execute("INSERT OR REPLACE INTO file_names_src(id, text) VALUES (?, ?)",
                                       (row.lastrowid, Extractor.own_name(found.path, found.kind)))
            return True
        except Exception as exc:
            self.audit.write("files.move_failed", error=str(exc)[:200])
            return False

    async def _extractor(self) -> Extractor:
        capabilities = await self.embedder.capabilities() if self.embedder else set()
        return Extractor(self.media, doc_prefix=self.embedder.doc_prefix if self.embedder else "",
                         capabilities=capabilities,
                         media_on=self.embedder is not None and bool(self.config.get("files.media", True)),
                         ocr_on=bool(self.config.get("files.ocr", True)),
                         transcriber=self.transcriber if self.config.get("files.transcribe", False) else None)

    def _name_piece(self, found: Found, extractor: Extractor | None = None) -> Piece:
        extractor = extractor or Extractor(self.media, doc_prefix=self.embedder.doc_prefix if self.embedder else "")
        if found.folder:
            return extractor.folder_name(found.path, found.holds)
        return extractor.name(found.path, remote=found.remote, kind=found.kind)

    def _twin(self, digest: bytes, found: Found) -> list[Piece] | None:
        """The pieces of a file with these contents, read whole with this profile and kind, other than its name:
        what a copy of it holds too, so the copy need not be read and embedded again. None when there is none."""
        twin = self.store.one("SELECT path FROM files WHERE digest = ? AND kind = ? AND profile = ? AND error = '' "
                              "AND path != ? LIMIT 1", (digest, found.kind, found.profile, str(found.path)))
        if twin is None:
            return None
        rows = self.store.query("SELECT part, text, embedding, code, modality, loc FROM file_chunks WHERE path = ? "
                                "AND modality != 'name' ORDER BY id", (twin["path"],))
        return [Piece(r["part"], r["text"], None, r["modality"], r["loc"], copied=(r["embedding"], r["code"]))
                for r in rows]

    async def _index(self, batch: list[Found]) -> tuple[int, int]:
        # A file queued twice (a file event ahead of the walk that also found it) is read once: written twice it
        # broke the batch's transaction and lost the work on every file in it.
        unique: dict[str, Found] = {}
        for found in batch:
            unique.setdefault(str(found.path), found)
        batch = list(unique.values())
        if not batch:
            return 0, 0
        extractor = await self._extractor()
        reads = asyncio.Semaphore(PARALLEL_READS)

        async def read(found: Found) -> tuple[Found, list[Piece], str, bytes | None]:
            async with reads:
                if found.folder:
                    return found, [extractor.folder_name(found.path, found.holds)], "", None
                if found.remote:
                    # Findable by its words at once; its name is embedded later (_embed_names), after
                    # what is on this Mac. A home folder can name 100,000 files that only iCloud holds.
                    return found, [dataclasses.replace(extractor.name(found.path, remote=True, kind=found.kind),
                                                       embed=None)], "", None
                digest = None
                try:
                    if found.evicted:
                        await self._download(found)
                    digest = await asyncio.to_thread(content_fingerprint, str(found.path))
                    twin = self._twin(digest, found) if digest else None
                    if twin is not None:
                        return found, [extractor.name(found.path, kind=found.kind), *twin], "", digest
                    return found, await extractor.pieces(found.path, found.kind, found.repo), "", digest
                except Skipped as exc:            # left out on purpose: still found by its name
                    return found, [extractor.name(found.path, kind=found.kind)], f"skipped: {exc}", digest
                except Exception as exc:          # the name is still worth finding
                    return found, [extractor.name(found.path, kind=found.kind)], str(exc)[:300], digest

        groups = [batch[i:i + GROUP] for i in range(0, len(batch), GROUP)]
        upcoming = asyncio.gather(*map(read, groups[0]))
        done = failed = 0
        try:
            for number in range(len(groups)):
                files = await upcoming
                # Read the next group while this one is embedded.
                upcoming = asyncio.gather(*map(read, groups[number + 1])) if number + 1 < len(groups) else None
                inputs = [piece.embed for _, pieces, _, _ in files for piece in pieces if piece.embed is not None]
                vectors = await self.embedder.embed(inputs) if self.embedder is not None and inputs else []
                self._write(files, vectors)
                # A file skipped on purpose (generated code) is indexed by its name: it counts as done.
                done += sum(1 for found, _, error, _ in files
                            if not found.folder and (not error or error.startswith("skipped")))
                failed += sum(1 for _, _, error, _ in files if error and not error.startswith("skipped"))
        except httpx.HTTPError as exc:
            # The model is unreachable: leave the rest for the next pass rather than index it without vectors.
            if upcoming is not None:
                upcoming.cancel()
                with contextlib.suppress(BaseException):
                    await upcoming
            self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
        return done, failed

    def _write(self, files: list[tuple[Found, list[Piece], str, bytes | None]], vectors: list[list[float]]) -> None:
        stream = iter(vectors)
        added = []
        with self.store.transaction():
            self._drop([str(found.path) for found, _, _, _ in files], transaction=False)
            for found, pieces, error, digest in files:
                kind = found.kind
                self.store.execute(
                    "INSERT INTO files(path, kind, size, mtime, indexed, chunks, error, digest, device, inode, "
                    "profile, holds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(found.path), kind, found.size, found.mtime, time.time(), len(pieces), error, digest,
                     found.device, found.inode, found.profile, " ".join(found.holds)))
                vectors_of = [next(stream, None) if piece.embed is not None else None for piece in pieces]
                for i, piece in enumerate(pieces):
                    if piece.average_of:
                        # A derived piece (the whole video): the mean of the file's pieces it names, normalized.
                        parts = [v for p, v in zip(pieces, vectors_of) if v and p.part.startswith(piece.average_of)]
                        vectors_of[i] = unit_mean(parts) if parts else None
                for piece, vector in zip(pieces, vectors_of):
                    if piece.copied is not None:
                        embedding, code = piece.copied
                    elif vector:
                        embedding, code = pack(vector), code_bytes(vector)
                    elif piece.modality == "name" and piece.embed is None and not found.remote and found.kind != "folder":
                        embedding, code = None, None       # a name written before its vector: embedded later
                    elif piece.embed is None and piece.modality == "name" and found.remote:
                        embedding, code = None, None       # likewise, after what is on this Mac
                    else:
                        embedding, code = NEVER, None      # keywords only
                    row = self.store.execute(
                        "INSERT INTO file_chunks(path, part, text, embedding, code, modality, loc) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)", (str(found.path), piece.part, piece.text, embedding, code,
                                                         piece.modality, piece.loc))
                    if piece.modality == "name":
                        own = getattr(found, "own_name", None) or Extractor.own_name(found.path, kind)
                        self.store.execute("INSERT OR REPLACE INTO file_names_src(id, text) VALUES (?, ?)",
                                           (row.lastrowid, own))
                    if code:
                        added.append((row.lastrowid, code, piece.modality, kind))
                        if vector:
                            self.vectors.learn(piece.modality, vector)
        if self.vectors.loaded:
            self.vectors.add(added)

    async def _index_sources(self, budget: int) -> tuple[int, int]:
        """Mail, Messages, Notes and Calendar entries: read through their sources, embedded and written like files."""
        items, gone = await self.sources.index_once(budget, float(self.config.get("files.sources_interval", 900) or 900))
        if gone:
            self._drop(gone)
        if not items:
            return 0, len(gone)
        for start in range(0, len(items), GROUP):
            group = items[start:start + GROUP]
            inputs = [piece.embed for item in group for piece in item.pieces if piece.embed is not None]
            try:
                vectors = await self.embedder.embed(inputs) if inputs else []
            except httpx.HTTPError as exc:
                self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
                return start, len(gone)
            self._write([(item, item.pieces, "", None) for item in group], vectors)
        self.audit.write("files.sources_indexed", entries=len(items), removed=len(gone))
        return len(items), len(gone)

    async def _embed_names(self, limit: int) -> int:
        """Embed names written without a vector (files only iCloud holds, files renamed), a batch at a time."""
        if self.embedder is None:
            return 0
        rows = self.store.query("SELECT c.id, c.path, f.kind, f.holds FROM file_chunks c JOIN files f ON f.path = c.path "
                                "WHERE c.modality = 'name' AND c.code IS NULL AND c.embedding IS NULL LIMIT ?", (limit,))
        if not rows:
            return 0
        extractor = await self._extractor()
        inputs = []
        for row in rows:
            path = pathlib.Path(row["path"])
            if row["kind"] == "folder":
                inputs.append(extractor.folder_name(path, (row["holds"] or "").split()).embed)
            else:
                inputs.append(extractor.name(path, kind=row["kind"]).embed)
        wanted = [(row, text) for row, text in zip(rows, inputs) if text is not None]
        try:
            vectors = await self.embedder.embed([text for _, text in wanted]) if wanted else []
        except httpx.HTTPError as exc:
            self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
            return 0
        added = []
        with self.store.transaction():
            for row, text in zip(rows, inputs):
                if text is None:
                    self.store.execute("UPDATE file_chunks SET embedding = ? WHERE id = ?", (NEVER, row["id"]))
            for (row, _), vector in zip(wanted, vectors):
                code = code_bytes(vector)
                self.store.execute("UPDATE file_chunks SET embedding = ?, code = ? WHERE id = ?",
                                   (pack(vector), code, row["id"]))
                added.append((row["id"], code, "name", row["kind"]))
                self.vectors.learn("name", vector)
        if self.vectors.loaded:
            self.vectors.add(added)
        return len(rows)

    # ---- iCloud ---------------------------------------------------------------------

    async def _download(self, found: Found) -> None:
        """Asks iCloud for a file it alone holds and waits for its bytes."""
        if sys.platform != "darwin":
            return
        await self._brctl("download", str(found.path))
        began = time.monotonic()
        while time.monotonic() - began < DOWNLOAD_WAIT:
            try:
                if not in_cloud_only(os.stat(found.path)):
                    self._downloaded.append((str(found.path), time.monotonic(), 0))
                    return
            except OSError:
                break
            await asyncio.sleep(1.0)
        raise RuntimeError("iCloud did not bring the file in time")

    async def _release_downloads(self) -> None:
        """Gives back to iCloud the downloaded files whose reading was written EVICTION_DELAY ago, so the Mac's
        storage stays as it was; asked again a few times when one stays."""
        now = time.monotonic()
        due = [item for item in self._downloaded if now - item[1] >= EVICTION_DELAY]
        if not due:
            return
        self._downloaded = [item for item in self._downloaded if now - item[1] < EVICTION_DELAY]
        for path, since, attempts in due:
            ok = await self._brctl("evict", path)
            if not ok and attempts < 3:
                self._downloaded.append((path, now, attempts + 1))

    async def _brctl(self, verb: str, path: str) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec("/usr/bin/brctl", verb, path, stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(proc.wait(), 30)
            return proc.returncode == 0
        except (OSError, asyncio.TimeoutError):
            return False

    # ---- removal ---------------------------------------------------------------------

    def _drop(self, paths: list[str], transaction: bool = True) -> None:
        if not paths:
            return
        if transaction:
            # A transaction per 2,000 files: dropping 25,000 files (1.7 million pieces) in one filled the disk
            # with its write-ahead log.
            for start in range(0, len(paths), 2000):
                with self.store.transaction():
                    self._drop(paths[start:start + 2000], transaction=False)
            return
        for start in range(0, len(paths), 500):
            part = paths[start:start + 500]
            marks = ",".join("?" * len(part))
            self.vectors.remove(r["id"] for r in self.store.query(
                f"SELECT id FROM file_chunks WHERE path IN ({marks})", part))
            self.store.execute(f"DELETE FROM file_chunks WHERE path IN ({marks})", part)
            self.store.execute(f"DELETE FROM files WHERE path IN ({marks})", part)

    def _forget_outside(self) -> int:
        gone = [r["path"] for r in self.store.query("SELECT path FROM files WHERE path NOT LIKE '%://%'")
                if not self.folders.inside(r["path"])]
        self._drop(gone)
        return len(gone)

    async def changed(self, paths: set[str]) -> None:
        """File events: look at these paths again, ahead of everything else. A folder that appeared (moved in,
        unzipped, created) is walked on its own."""
        for raw in paths:
            path = pathlib.Path(raw)
            if path.is_dir():
                start = await asyncio.to_thread(self.folders.folder_start, path)
                if start is not None:
                    queued = {str(f.path) for f in self._queue}
                    fresh = [f for f in await asyncio.to_thread(lambda: list(self.folders.walk([start])))
                             if str(f.path) not in queued]
                    self._queue.extendleft(reversed(sorted(fresh, key=priority)))
                continue
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
                if self.paused or self._waiting or self._downloaded:  # notice the charger, and give files back
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

    def listing(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        """The most recently changed indexed files, for a search without a query."""
        rows = self.store.query("SELECT path, kind, size, mtime FROM files WHERE error = '' AND kind != 'folder' "
                                "AND (? IS NULL OR kind = ?) ORDER BY mtime DESC LIMIT ?",
                                (kind, kind, max(1, min(int(limit), 50))))
        return [{"path": r["path"], "kind": r["kind"], "size": r["size"],
                 "modified": datetime.datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d")} for r in rows]

    def status(self) -> dict:
        rows = self.store.query("SELECT kind, COUNT(*) AS n, SUM(error != '') AS failed FROM files GROUP BY kind")
        return {"roots": [str(r) for r in self.roots()], "model": self.embedder.model if self.embedder else None,
                "by_kind": {r["kind"]: {"files": r["n"], "failed": r["failed"] or 0} for r in rows},
                "chunks": (self.store.one("SELECT COUNT(*) AS n FROM file_chunks") or {}).get("n", 0),
                "in_memory": len(self.vectors), "queued": len(self._queue), "watching": self.watching,
                "paused": self.paused, "cloud_only_skipped": self.cloud_only,
                "waiting_for_power": len(self._waiting), "downloaded": len(self._downloaded),
                "sources": self.sources.status()}
