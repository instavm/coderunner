"""Minimal Hugging Face Hub client: metadata, licenses and verified downloads.

API responses are cached under assistant_dir()/models/cache for a day. The
llama.cpp manifest endpoint resolves a quant tag to the GGUF and its vision
projector in one call. Downloads are pinned to a commit, resumable through
HTTP Range and checked against the LFS sha256 before they are moved into place;
large files come in several ranges at once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from urllib.parse import quote

import httpx

from ..core import paths

SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def token() -> str:
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value.strip()
    path = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token"
    try:
        return path.read_text().strip()
    except OSError:
        return ""


RESUMES = 20           # times a dropped download carries on from where it stopped
# Large files come in PARALLEL ranges at once. Measured on a home line (Oct 2026): one connection to Hugging
# Face's CDN held 1.5 MB/s through a whole 2.9 GB file while fresh ones got 0.9-6.6 MB/s, and four together
# 7.3 MB/s, what the line carries. Each range is written into its place in the .part file, so a download
# needs no space beyond the file itself.
PARALLEL = 4
PARALLEL_MIN = 256 << 20


class _NoRanges(Exception):
    """The server sends whole files only."""


class HubError(Exception):
    pass


class RateLimited(HubError):
    """Hugging Face answered 429. Callers without an account get 500 API requests per five minutes for their
    address (its RateLimit-Policy header, November 2026); `seconds` is when the window starts again."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        super().__init__(f"Hugging Face is limiting requests from this connection; try again in "
                         f"{max(1, round(seconds / 60))} min, or set HF_TOKEN to an account's token")


class Hub:
    calls = 0           # requests made to the Hub's API by this process (setup_eval reports them)

    def __init__(self, client: httpx.Client | None = None, cache_dir: pathlib.Path | None | bool = None,
                 ttl: float = 24 * 3600):
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, read=120), follow_redirects=True,
                                              headers={"User-Agent": "coderunner"})
        if cache_dir is None:
            cache_dir = paths.assistant_dir() / "models" / "cache"
        self.cache_dir = cache_dir or None
        self.ttl = ttl

    # ---- metadata ---------------------------------------------------------

    def _auth(self) -> dict:
        tok = token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _get_json(self, url: str, key: str, headers: dict | None = None) -> dict:
        cache = self.cache_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", key) + ".json") if self.cache_dir else None
        if cache and cache.exists() and time.time() - cache.stat().st_mtime < self.ttl:
            try:
                return json.loads(cache.read_text())
            except json.JSONDecodeError:
                pass
        Hub.calls += 1
        try:
            response = self.client.get(url, headers={**self._auth(), **(headers or {})})
        except httpx.HTTPError as exc:
            raise HubError(f"Hugging Face is unreachable: {exc}") from exc
        if response.status_code == 429:
            if cache and cache.exists():            # yesterday's answer serves better than none
                with contextlib.suppress(OSError, json.JSONDecodeError):
                    return json.loads(cache.read_text())
            found = re.search(r"\bt=(\d+)", response.headers.get("ratelimit", ""))
            raise RateLimited(int(found.group(1)) if found else int(response.headers.get("retry-after") or 300))
        if response.status_code in (401, 403):
            raise HubError(f"{url} needs access: accept the model terms on Hugging Face and set HF_TOKEN")
        if response.status_code >= 400:
            raise HubError(f"{url} returned {response.status_code}: {response.text[:200]}")
        data = response.json()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        return data

    def model_info(self, repo: str, revision: str | None = None) -> dict:
        path = f"{repo}/revision/{revision}" if revision and revision != "main" else repo
        return self._get_json(f"{endpoint()}/api/models/{path}?blobs=true", f"info-{repo}-{revision or 'main'}")

    def manifest(self, repo: str, quant: str) -> dict:
        # llama.cpp's own resolver: the User-Agent switches the response to GGUF + mmproj.
        return self._get_json(f"{endpoint()}/v2/{repo}/manifests/{quant}", f"manifest-{repo}-{quant}",
                              {"User-Agent": "llama-cpp"})

    def gguf_files(self, repo: str, quant: str, vision: bool = True, revision: str | None = None) -> list[dict]:
        """Every file to fetch for a quant: all shards of a split GGUF, then the projector."""
        data = self.manifest(repo, quant)
        main = data.get("ggufFile") or {}
        if not main.get("rfilename"):
            raise HubError(f"{repo} has no GGUF for {quant}")
        siblings = {s["rfilename"]: s for s in self.model_info(repo, revision).get("siblings", [])}

        def describe(name: str, kind: str, meta: dict | None = None) -> dict:
            sib = siblings.get(name, {})
            meta = meta or {}
            lfs = meta.get("lfs") or sib.get("lfs") or {}
            return {"file": name, "kind": kind, "size": meta.get("size") or sib.get("size"),
                    "sha256": lfs.get("sha256") or lfs.get("oid")}

        files = [describe(main["rfilename"], "model", main)]
        shard = SHARD.search(main["rfilename"])
        if shard:
            total, prefix = int(shard.group(2)), main["rfilename"][:shard.start()]
            files = [describe(f"{prefix}-{i:05d}-of-{total:05d}.gguf", "model") for i in range(1, total + 1)]
        mmproj = data.get("mmprojFile") or {}
        if vision and mmproj.get("rfilename"):
            files.append(describe(mmproj["rfilename"], "mmproj", mmproj))
        return files

    def resolve_license(self, repo: str, depth: int = 3) -> dict:
        """License of a repo; mirrors without metadata inherit it from their base model."""
        result = {"spdx": None, "name": None, "link": f"{endpoint()}/{repo}", "repo": repo, "gated": False}
        seen = set()
        for level in range(depth + 1):
            info = self.model_info(repo)
            card, tags = info.get("cardData") or {}, info.get("tags") or []
            if level == 0:
                result["gated"] = bool(info.get("gated"))
            spdx = card.get("license") or next((t[8:] for t in tags if t.startswith("license:")), None)
            if isinstance(spdx, list):
                spdx = spdx[0] if spdx else None
            if spdx:
                link = card.get("license_link") or f"{endpoint()}/{repo}"
                if not link.startswith("http"):
                    link = f"{endpoint()}/{repo}/blob/main/{link.lstrip('./')}"
                result.update(spdx=str(spdx).lower(), name=card.get("license_name"), link=link, repo=repo)
                return result
            seen.add(repo)
            base = next((t.split(":", 2)[2] for t in tags if t.startswith("base_model:quantized:")), None)
            if not base:
                card_base = card.get("base_model")
                base = card_base[0] if isinstance(card_base, list) and card_base else card_base
            if not base or base in seen:
                break
            repo = base
        return result

    # ---- downloads --------------------------------------------------------

    def download(self, repo: str, file: str, dest: pathlib.Path | str, revision: str = "main",
                 sha256: str | None = None, size: int | None = None,
                 progress: Callable[[str], None] | None = None) -> pathlib.Path:
        dest = pathlib.Path(dest)
        if dest.exists() and (size is None or dest.stat().st_size == size):
            return dest  # files are only renamed into place after verification
        if sha256 is None:
            sib = next((s for s in self.model_info(repo, revision).get("siblings", []) if s["rfilename"] == file), {})
            sha256 = (sib.get("lfs") or {}).get("sha256")
            size = size or sib.get("size")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        state = part.with_name(part.name + ".json")            # what each range has so far
        url = f"{endpoint()}/{repo}/resolve/{revision}/{quote(file)}"
        digest = None
        # A .part without its ranges is an earlier one-stream download: that carries on as it was.
        if size and size >= PARALLEL_MIN and (state.exists() or not part.exists()):
            try:
                self._ranges(url, repo, file, part, state, size, progress)
                digest = _sha256(part)
            except _NoRanges:
                part.unlink(missing_ok=True)
                state.unlink(missing_ok=True)
        if digest is None:
            digest = self._stream(url, repo, file, part, size, progress)
        if size and part.stat().st_size != size:
            raise HubError(f"{file}: expected {size} bytes, got {part.stat().st_size}; run again to resume")
        if sha256 and digest.hexdigest() != sha256:
            part.unlink()
            raise HubError(f"{file}: sha256 mismatch, deleted the download")
        os.replace(part, dest)
        return dest

    def _stream(self, url: str, repo: str, file: str, part: pathlib.Path, size: int | None,
                progress: Callable[[str], None] | None) -> "hashlib._Hash":
        """One stream from the start, or from where an earlier one stopped; hashed as it comes."""
        digest, offset = hashlib.sha256(), 0
        if part.exists():
            with open(part, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
                    offset += len(block)
        last, resumes = 0.0, 0
        while True:
            headers = {**self._auth(), **({"Range": f"bytes={offset}-"} if offset else {})}
            before = offset
            try:
                with self.client.stream("GET", url, headers=headers, timeout=httpx.Timeout(30, read=300)) as response:
                    if response.status_code in (401, 403):
                        raise HubError(f"{repo} needs access: accept its terms on Hugging Face and set HF_TOKEN")
                    if response.status_code == 416 and offset:
                        pass  # the partial file is already complete
                    elif response.status_code == 200 and offset:
                        digest, offset = hashlib.sha256(), 0  # server ignored Range; start over
                    elif response.status_code not in (200, 206):
                        raise HubError(f"Download of {repo}/{file} failed with {response.status_code}")
                    total = size or offset + int(response.headers.get("content-length") or 0)
                    if response.status_code != 416:
                        with open(part, "ab" if offset else "wb") as handle:
                            done = offset
                            for chunk in response.iter_bytes(1 << 20):
                                handle.write(chunk)
                                digest.update(chunk)
                                done += len(chunk)
                                if progress and (time.monotonic() - last > 2 or done == total):
                                    last = time.monotonic()
                                    progress(_progress(file, done, total))
                return digest
            except httpx.HTTPError as exc:
                # Hugging Face's CDN closed connections every 1.4-1.7 GB in testing: carry on from where it stopped
                # while each attempt gets further.
                offset = part.stat().st_size if part.exists() else 0
                resumes += 1
                if offset <= before or resumes > RESUMES:
                    raise HubError(f"Download of {repo}/{file} interrupted ({exc}); run again to resume") from exc
                if progress:
                    progress(f"downloading {file}: the connection dropped at {offset / 1e9:.2f} GB; resuming")
                time.sleep(min(30, 2 * resumes))

    def _ranges(self, url: str, repo: str, file: str, part: pathlib.Path, state: pathlib.Path, size: int,
                progress: Callable[[str], None] | None) -> None:
        """PARALLEL ranges at once, each written into its place in the .part file. What each has so far is
        kept beside it (.part.json, only ever behind what is written), so an interrupted download carries on."""
        try:
            saved = json.loads(state.read_text())
            ranges = [list(map(int, r)) for r in saved["ranges"]] if saved.get("size") == size and part.exists() \
                else None
        except (OSError, ValueError, KeyError, TypeError):
            ranges = None
        if not ranges:
            ranges = [[size * i // PARALLEL, size * (i + 1) // PARALLEL, 0] for i in range(PARALLEL)]
        lock, stop, last = threading.Lock(), threading.Event(), [0.0]

        def save() -> None:
            state.write_text(json.dumps({"size": size, "ranges": ranges}))

        def fetch(r: list[int]) -> None:
            start, end, resumes = r[0], r[1], 0
            while r[2] < end - start and not stop.is_set():
                before = r[2]
                headers = {**self._auth(), "Range": f"bytes={start + r[2]}-{end - 1}"}
                try:
                    with self.client.stream("GET", url, headers=headers,
                                            timeout=httpx.Timeout(30, read=300)) as response:
                        if response.status_code in (401, 403):
                            raise HubError(f"{repo} needs access: accept its terms on Hugging Face and set HF_TOKEN")
                        if response.status_code == 200:
                            raise _NoRanges()
                        if response.status_code != 206:
                            raise HubError(f"Download of {repo}/{file} failed with {response.status_code}")
                        for chunk in response.iter_bytes(1 << 20):
                            if stop.is_set():
                                return
                            chunk = chunk[:end - start - r[2]]
                            os.pwrite(fd, chunk, start + r[2])
                            with lock:
                                r[2] += len(chunk)
                                if time.monotonic() - last[0] > 2:
                                    last[0] = time.monotonic()
                                    save()
                                    if progress:
                                        progress(_progress(file, sum(x[2] for x in ranges), size))
                    if r[2] <= before:
                        raise HubError(f"Download of {repo}/{file} stopped sending; run again to resume")
                except httpx.HTTPError as exc:
                    resumes += 1
                    if r[2] <= before or resumes > RESUMES:
                        raise HubError(f"Download of {repo}/{file} interrupted ({exc}); run again to resume") from exc
                    time.sleep(min(30, 2 * resumes))

        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with ThreadPoolExecutor(PARALLEL) as pool:
                futures = [pool.submit(fetch, r) for r in ranges]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:            # the others stop too; what they have is kept
                        stop.set()
                        errors.append(exc)
        finally:
            os.close(fd)
            with lock:
                save()
        if errors:
            raise next((e for e in errors if isinstance(e, _NoRanges)), errors[0])
        state.unlink(missing_ok=True)
        if progress:
            progress(_progress(file, size, size))


def _sha256(path: pathlib.Path) -> "hashlib._Hash":
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest


def _progress(file: str, done: int, total: int) -> str:
    return f"downloading {file}: " + (f"{100 * done / total:.0f}% of {total / 1e9:.2f} GB" if total
                                      else f"{done / 1e9:.2f} GB")



_default: Hub | None = None


def default() -> Hub:
    global _default
    if _default is None:
        _default = Hub()
    return _default


def model_info(repo: str, revision: str | None = None) -> dict:
    return default().model_info(repo, revision)


def manifest(repo: str, quant: str) -> dict:
    return default().manifest(repo, quant)


def resolve_license(repo: str) -> dict:
    return default().resolve_license(repo)


def download(repo: str, file: str, dest, revision: str = "main", progress=None, **kwargs) -> pathlib.Path:
    return default().download(repo, file, dest, revision=revision, progress=progress, **kwargs)
