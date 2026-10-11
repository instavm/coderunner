"""The owner's local files as something the assistant can find and use.

The owner chooses folders (`coderunner files add ~/Documents`). A background
indexer embeds what is in them with a multimodal embedding model: through
Ollama, EmbeddingGemma 2 places text, code, images and audio in one space,
and video is indexed as sampled frames plus its sound. Text also goes into a
keyword index. Nothing leaves the machine.

Only regular files inside the chosen folders are read, after resolving
symlinks. Secrets (key files, .env files, ~/.ssh, keychains, password
databases) and CodeRunner's own state are never indexed or read. File
contents are the owner's private data and may also be third-party text
(downloads, attachments), so the tools that return them mark the task as
holding private data and as having read untrusted content.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import heapq
import json
import math
import os
import pathlib
import shutil
import tempfile
import time
from typing import Any

import httpx

from . import paths

TEXT = {".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".log", ".tex", ".bib", ".srt", ".vtt", ".html", ".htm", ".xml", ".svg", ".css", ".scss",
        ".py", ".ipynb", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h",
        ".cc", ".cpp", ".hpp", ".m", ".mm", ".cs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".scala", ".dart",
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".graphql", ".proto", ".vue", ".svelte", ".zig", ".nim"}
DOCS = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".pages"}         # converted to text by local tools
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tiff", ".tif"}
AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".flac", ".ogg", ".opus", ".caf"}
VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

# Never indexed or read, whatever folder they are in.
SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store", "Keychains", ".coderunner"}
SECRET_FILES = (".env", ".env.*", "*.env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials",
                "credentials.*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "*.pem", "*.key", "*.p12",
                "*.pfx", "*.jks", "*.keystore", "*.kdbx", "*.keychain", "*.keychain-db", "*.ovpn", "*.gpg", "*.asc")
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".tox", "target", "build", "dist", ".next", ".cache", ".Trash", "DerivedData", "Pods"}

TEXT_CHUNK = 1500          # characters per text chunk
TEXT_OVERLAP = 200
MAX_CHUNKS = 200           # per file
AUDIO_SEGMENT = 60         # seconds of sound per embedding
MAX_SEGMENTS = 10
VIDEO_FRAMES = 6
NATIVE_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}   # sent to the embedding model unchanged
NATIVE_IMAGE_BYTES = 20 * 1024 * 1024
# How each embedding model family wants queries and documents phrased, and the similarity above which a
# memory is clearly relevant to a task (measured on labelled examples; scales differ between models).
PROFILES = {
    "embeddinggemma": {"query": "task: search result | query: ", "doc": "title: {title} | text: ", "recall": 0.62},
    "": {"query": "", "doc": "", "recall": 0.5},
}


def embedding_profile(model: str) -> dict:
    family = next((name for name in PROFILES if name and name in (model or "").lower()), "")
    return PROFILES[family]


class FileAccessError(PermissionError):
    pass


def kind_of(path: pathlib.Path) -> str | None:
    suffix = path.suffix.lower()
    for kind, suffixes in (("text", TEXT), ("doc", DOCS), ("image", IMAGES), ("audio", AUDIO), ("video", VIDEO)):
        if suffix in suffixes:
            return kind
    return None


def is_secret(path: pathlib.Path) -> bool:
    if any(part in SECRET_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILES)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class Embedder:
    """Ollama's /api/embed, which takes text, images and audio (base64) in one request."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 120.0, transport=None):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport)
        self._capabilities: set[str] | None = None
        profile = embedding_profile(model)
        self.query_prefix = profile["query"]
        self.doc_prefix = profile["doc"]

    async def capabilities(self) -> set[str]:
        if self._capabilities is None:
            try:
                response = await self._client.post(f"{self.base_url}/api/show", json={"model": self.model})
                response.raise_for_status()
                self._capabilities = set(response.json().get("capabilities") or [])
            except (httpx.HTTPError, ValueError):
                self._capabilities = set()
        return self._capabilities

    async def embed(self, inputs: list[Any]) -> list[list[float]]:
        response = await self._client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": inputs})
        response.raise_for_status()
        return [_unit(vector) for vector in response.json()["embeddings"]]

    async def aclose(self) -> None:
        await self._client.aclose()


class Media:
    """Local conversions with ffmpeg / macOS tools; every input is a path argument, never shell text."""

    def __init__(self):
        self.ffmpeg = shutil.which("ffmpeg")
        self.sips = shutil.which("sips")
        self.textutil = shutil.which("textutil")
        self.pdftotext = shutil.which("pdftotext")

    async def _run(self, *argv: str, timeout: float = 120) -> bytes:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{pathlib.Path(argv[0]).name} timed out")
        if proc.returncode:
            raise RuntimeError(f"{pathlib.Path(argv[0]).name} failed: {err.decode(errors='replace')[-300:]}")
        return out

    async def text(self, path: pathlib.Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if not self.pdftotext:
                raise RuntimeError("install poppler (pdftotext) to read PDFs")
            return (await self._run(self.pdftotext, "-layout", "-q", str(path), "-")).decode(errors="replace")
        if suffix in DOCS:
            if not self.textutil:
                raise RuntimeError("documents need macOS textutil")
            return (await self._run(self.textutil, "-convert", "txt", "-stdout", str(path))).decode(errors="replace")
        return path.read_text(errors="replace")

    async def image(self, path: pathlib.Path, at: float | None = None, size: int = 1024) -> bytes:
        """A JPEG no larger than size pixels: the image itself, or a video frame at `at` seconds."""
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "frame.jpg"
            if self.ffmpeg and (at is not None or path.suffix.lower() not in (".heic", ".heif")):
                argv = [self.ffmpeg, "-loglevel", "error", "-y"]
                if at is not None:
                    argv += ["-ss", f"{at:.2f}"]
                argv += ["-i", str(path), "-frames:v", "1", "-vf",
                         f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease", str(out)]
                await self._run(*argv)
            elif self.sips:
                await self._run(self.sips, "-s", "format", "jpeg", "-Z", str(size), str(path), "--out", str(out))
            else:
                raise RuntimeError("install ffmpeg to read images and video")
            return out.read_bytes()

    async def duration(self, path: pathlib.Path) -> float:
        probe = shutil.which("ffprobe")
        if not probe:
            return 0.0
        out = await self._run(probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path))
        try:
            return float(out.decode().strip() or 0)
        except ValueError:
            return 0.0

    async def sound(self, path: pathlib.Path, start: float, seconds: float) -> bytes:
        """16 kHz mono WAV of part of an audio or video file."""
        if not self.ffmpeg:
            raise RuntimeError("install ffmpeg to read audio and video")
        return await self._run(self.ffmpeg, "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}",
                               "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-")


class FileIndex:
    def __init__(self, config, store, audit, embedder: Embedder | None = None, media: Media | None = None):
        self.config = config
        self.store = store
        self.audit = audit
        self.embedder = embedder
        self.media = media or Media()
        self._vectors: dict[int, list[float]] | None = None
        self._lock = asyncio.Lock()
        self.wake = asyncio.Event()

    # ---- access control ------------------------------------------------------

    def roots(self) -> list[pathlib.Path]:
        return [pathlib.Path(os.path.expanduser(r)).resolve() for r in self.config.get("files.roots", []) or []]

    def resolve(self, path: str) -> pathlib.Path:
        """The real path of an indexed-folder file the assistant may read, or FileAccessError."""
        real = pathlib.Path(os.path.expanduser(str(path))).resolve()
        root = next((r for r in self.roots() if r in real.parents), None)
        if root is None:
            raise FileAccessError(f"{path} is not in a folder the owner shared (coderunner files add <folder>)")
        state = paths.home().resolve()
        hidden = any(part.startswith(".") for part in real.relative_to(root).parts)
        private = any(d == real or d in real.parents for d in self._private_dirs())
        if real == state or state in real.parents or is_secret(real) or hidden or private:
            raise FileAccessError(f"{path} may hold secrets or private app data and is never read")
        if not real.is_file():
            raise FileAccessError(f"{path} is not a file")
        return real

    def add_root(self, folder: str) -> str:
        real = pathlib.Path(os.path.expanduser(folder)).resolve()
        if not real.is_dir():
            raise ValueError(f"{folder} is not a folder")
        state = paths.home().resolve()
        if real == state or state in real.parents:
            raise ValueError("that folder holds CodeRunner's own state")
        if len(real.parts) <= 2 or real in self._private_dirs():
            raise ValueError("choose specific folders, not the whole disk or app data")
        roots = [str(r) for r in self.roots()]
        if str(real) not in roots:
            self.config.set("files.roots", roots + [str(real)])
            self.config.save()
        self.wake.set()
        return str(real)

    def remove_root(self, folder: str) -> int:
        real = pathlib.Path(os.path.expanduser(folder)).resolve()
        self.config.set("files.roots", [str(r) for r in self.roots() if r != real])
        self.config.save()
        return self._forget_outside()

    # ---- indexing ----------------------------------------------------------------

    @staticmethod
    def _private_dirs() -> set[pathlib.Path]:
        # App data under the home folder (mail, messages, keychains, browser profiles) is never indexed.
        return {pathlib.Path.home().resolve() / "Library"}

    def _walk(self):
        limit = int(self.config.get("files.max_file_mb", 200) or 200) * 1024 * 1024
        private = self._private_dirs()
        for root in self.roots():
            for folder, dirs, names in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in SECRET_DIRS and not d.startswith(".")
                           and pathlib.Path(folder, d) not in private]
                for name in names:
                    path = pathlib.Path(folder) / name
                    if name.startswith(".") or is_secret(path) or kind_of(path) is None or path.is_symlink():
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    if 0 < stat.st_size <= limit:
                        yield path, stat

    def _forget_outside(self) -> int:
        roots = self.roots()
        gone = [r["path"] for r in self.store.query("SELECT path FROM files")
                if not any(root in pathlib.Path(r["path"]).parents for root in roots)]
        for path in gone:
            self._drop(path)
        return len(gone)

    def _drop(self, path: str) -> None:
        self.store.execute("DELETE FROM file_chunks WHERE path = ?", (path,))
        self.store.execute("DELETE FROM files WHERE path = ?", (path,))
        self._vectors = None

    async def index_once(self, budget: int = 200) -> dict:
        """Index new or changed files (at most `budget` per pass) and forget removed ones."""
        async with self._lock:
            known = {r["path"]: r for r in self.store.query("SELECT path, size, mtime FROM files")}
            seen, done, failed = set(), 0, 0
            for path, stat in self._walk():
                key = str(path)
                seen.add(key)
                row = known.get(key)
                if row and row["size"] == stat.st_size and row["mtime"] == stat.st_mtime:
                    continue
                if done + failed >= budget:
                    continue
                try:
                    chunks = await self._chunks(path, kind_of(path))
                    vectors = await self._embed_chunks(chunks)
                    error = ""
                    done += 1
                except Exception as exc:
                    chunks, vectors, error = [], [], str(exc)[:300]
                    failed += 1
                self._drop(key)
                self.store.execute("INSERT INTO files(path, kind, size, mtime, indexed, chunks, error) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (key, kind_of(path), stat.st_size, stat.st_mtime, time.time(), len(chunks), error))
                for (part, text, _), vector in zip(chunks, vectors or [None] * len(chunks)):
                    self.store.execute("INSERT INTO file_chunks(path, part, text, embedding) VALUES (?, ?, ?, ?)",
                                       (key, part, text, json.dumps(vector) if vector else None))
            removed = [p for p in known if p not in seen]
            for path in removed:
                self._drop(path)
            self._vectors = None
            result = {"indexed": done, "failed": failed, "removed": len(removed), "files": len(seen),
                      "more": done + failed >= budget}
            if done or failed or removed:
                self.audit.write("files.indexed", **result)
            return result

    async def _chunks(self, path: pathlib.Path, kind: str) -> list[tuple[str, str, Any]]:
        """(part label, keyword text, embedding input) for each piece of a file."""
        title = path.name
        if kind in ("text", "doc"):
            text = await self.media.text(path)
            chunks = []
            for start in range(0, max(len(text), 1), TEXT_CHUNK - TEXT_OVERLAP):
                piece = text[start:start + TEXT_CHUNK]
                if not piece.strip():
                    continue
                line = text.count("\n", 0, start) + 1
                prefix = self.embedder.doc_prefix.format(title=title) if self.embedder else ""
                chunks.append((f"from line {line}", piece, prefix + piece))
                if len(chunks) >= MAX_CHUNKS:
                    break
            return chunks or [("whole file", title, title)]
        media_ok = self.embedder is not None and bool(self.config.get("files.media", True))
        caps = await self.embedder.capabilities() if media_ok else set()
        if kind == "image":
            if "vision" not in caps:
                return [("image", title, title)]
            # The model's own image processing is used as is; only formats it cannot decode (HEIC, TIFF,
            # BMP, GIF) or very large files are converted to JPEG first.
            if path.suffix.lower() in NATIVE_IMAGES and path.stat().st_size <= NATIVE_IMAGE_BYTES:
                raw = path.read_bytes()
            else:
                raw = await self.media.image(path, size=2048)
            return [("image", title, {"image": base64.b64encode(raw).decode()})]
        if kind in ("audio", "video"):
            chunks = []
            length = await self.media.duration(path) if self.media.ffmpeg else 0.0
            if kind == "video" and "vision" in caps and self.media.ffmpeg:
                times = [length * (i + 0.5) / VIDEO_FRAMES for i in range(VIDEO_FRAMES)] if length else [0.0]
                for at in times:
                    data = base64.b64encode(await self.media.image(path, at=at, size=2048)).decode()
                    chunks.append((f"frame at {at:.0f}s", title, {"image": data}))
            if "audio" in caps and self.media.ffmpeg:
                segments = max(1, min(MAX_SEGMENTS, math.ceil(length / AUDIO_SEGMENT))) if length else 1
                for i in range(segments):
                    sound = await self.media.sound(path, i * AUDIO_SEGMENT, AUDIO_SEGMENT)
                    if len(sound) > 1000:
                        chunks.append((f"sound from {i * AUDIO_SEGMENT}s", title,
                                       {"audio": base64.b64encode(sound).decode()}))
            return chunks or [(kind, title, title)]
        return [("file", title, title)]

    async def _embed_chunks(self, chunks: list[tuple[str, str, Any]]) -> list[list[float]] | None:
        if self.embedder is None or not chunks:
            return None
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), 16):
            vectors += await self.embedder.embed([item for _, _, item in chunks[start:start + 16]])
        return vectors

    async def run_forever(self) -> None:
        while True:
            try:
                if self.roots():
                    result = await self.index_once()
                    if result["more"]:
                        continue
            except Exception as exc:  # keep indexing alive across a model restart
                self.audit.write("files.index_error", error=str(exc)[:300])
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), float(self.config.get("files.scan_interval", 600) or 600))
            except asyncio.TimeoutError:
                pass

    # ---- search ------------------------------------------------------------------

    def _load_vectors(self) -> dict[int, list[float]]:
        if self._vectors is None:
            self._vectors = {}
            for row in self.store.query("SELECT id, embedding FROM file_chunks WHERE embedding IS NOT NULL"):
                try:
                    self._vectors[row["id"]] = json.loads(row["embedding"])
                except (TypeError, ValueError):
                    continue
        return self._vectors

    async def search(self, query: str, kind: str | None = None, limit: int = 8) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 30))
        rankings: list[list[int]] = []
        if self.embedder is not None and (vectors := self._load_vectors()):
            try:
                [q] = await self.embedder.embed([self.embedder.query_prefix + query])
                scored = ((sum(a * b for a, b in zip(q, v)), cid) for cid, v in vectors.items() if len(v) == len(q))
                rankings.append([cid for _, cid in heapq.nlargest(limit * 6, scored)])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                self.audit.write("files.embedding_unavailable", error=str(exc)[:200])
        from .memory import fts_query
        # Keywords count only when every word appears: exact lookups (names, codes, phrases) without
        # common words like "in" or "the" outranking what the embedding found.
        if expr := fts_query(query, all_terms=True):
            rows = self.store.query("SELECT rowid AS id FROM file_chunks_fts WHERE file_chunks_fts MATCH ? "
                                    "ORDER BY bm25(file_chunks_fts) LIMIT ?", (expr, limit * 6))
            rankings.append([r["id"] for r in rows])
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, 1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        if not scores:
            return []
        marks = ",".join("?" * len(scores))
        rows = {r["id"]: r for r in self.store.query(
            f"SELECT c.id, c.path, c.part, c.text, f.kind FROM file_chunks c JOIN files f ON f.path = c.path "
            f"WHERE c.id IN ({marks})", list(scores))}
        # One result per file: its best part first, plus a few other parts that matched.
        results: dict[str, dict] = {}
        for cid in sorted(scores, key=lambda i: -scores[i]):
            row = rows.get(cid)
            if row is None or (kind and row["kind"] != kind):
                continue
            found = results.get(row["path"])
            if found is None:
                if len(results) >= limit:
                    continue
                snippet = row["text"][:300] if row["kind"] in ("text", "doc") else ""
                results[row["path"]] = {"path": row["path"], "kind": row["kind"], "part": row["part"],
                                        "score": round(scores[cid], 5), **({"snippet": snippet} if snippet else {})}
            elif len(found.setdefault("also", [])) < 3 and row["part"] not in found["also"]:
                found["also"].append(row["part"])
        return list(results.values())

    def status(self) -> dict:
        rows = self.store.query("SELECT kind, COUNT(*) AS n, SUM(error != '') AS failed FROM files GROUP BY kind")
        return {"roots": [str(r) for r in self.roots()], "model": self.embedder.model if self.embedder else None,
                "by_kind": {r["kind"]: {"files": r["n"], "failed": r["failed"] or 0} for r in rows},
                "chunks": (self.store.one("SELECT COUNT(*) AS n FROM file_chunks") or {}).get("n", 0)}
