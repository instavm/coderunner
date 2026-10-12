"""Minimal Hugging Face Hub client: metadata, licenses and verified downloads.

API responses are cached under assistant_dir()/models/cache for a day. The
llama.cpp manifest endpoint resolves a quant tag to the GGUF and its vision
projector in one call. Downloads are pinned to a commit, resumable through
HTTP Range and checked against the LFS sha256 before they are moved into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
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


class HubError(Exception):
    pass


class Hub:
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
        try:
            response = self.client.get(url, headers={**self._auth(), **(headers or {})})
        except httpx.HTTPError as exc:
            raise HubError(f"Hugging Face is unreachable: {exc}") from exc
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
        digest, offset = hashlib.sha256(), 0
        if part.exists():
            with open(part, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
                    offset += len(block)
        url = f"{endpoint()}/{repo}/resolve/{revision}/{quote(file)}"
        headers = {**self._auth(), **({"Range": f"bytes={offset}-"} if offset else {})}
        last = 0.0
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
                                pct = f"{100 * done / total:.0f}% of {total / 1e9:.2f} GB" if total else f"{done / 1e9:.2f} GB"
                                progress(f"downloading {file}: {pct}")
        except httpx.HTTPError as exc:
            raise HubError(f"Download of {repo}/{file} interrupted ({exc}); run again to resume") from exc
        if size and part.stat().st_size != size:
            raise HubError(f"{file}: expected {size} bytes, got {part.stat().st_size}; run again to resume")
        if sha256 and digest.hexdigest() != sha256:
            part.unlink()
            raise HubError(f"{file}: sha256 mismatch, deleted the download")
        os.replace(part, dest)
        return dest


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
