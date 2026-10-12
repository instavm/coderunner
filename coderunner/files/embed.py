"""The embedding model, through Ollama's native embed API (text, images and audio in one request).

Inputs are sent in batches: one request with 64 short texts costs little more than one with a single
text, and indexing a home folder is mostly short texts (names, passages). The model is kept loaded while
CodeRunner runs, so a search does not wait for it, and recent query vectors are reused.
"""

from __future__ import annotations

import asyncio
import collections
import math
from typing import Any

import httpx

from .kinds import embedding_profile

TEXT_BATCH = 64
MEDIA_BATCH = 8            # pictures and sound are heavy: fewer per request
PARALLEL = 4               # requests in flight: a runtime with several sequence slots embeds them together
KEEP_ALIVE = "1h"
QUERY_CACHE = 256


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class Embedder:
    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0, transport=None,
                 keep_alive: str = KEEP_ALIVE, parallel: int = PARALLEL):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.keep_alive = keep_alive
        self._slots = asyncio.Semaphore(max(1, parallel))
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport)
        self._capabilities: set[str] | None = None
        self._queries: collections.OrderedDict[str, list[float]] = collections.OrderedDict()
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

    async def fingerprint(self) -> str:
        """The model and the exact weights: vectors from different weights cannot be compared."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            wanted = {self.model, f"{self.model}:latest"}
            digest = next((m.get("digest", "") for m in response.json().get("models", [])
                           if m.get("name") in wanted or m.get("model") in wanted), "")
        except (httpx.HTTPError, ValueError):
            digest = ""
        return f"{self.model}@{digest[:12]}" if digest else self.model

    async def _request(self, inputs: list[Any]) -> list[list[float]]:
        response = await self._client.post(f"{self.base_url}/api/embed", json={
            "model": self.model, "input": inputs, "keep_alive": self.keep_alive})
        response.raise_for_status()
        return [_unit(vector) for vector in response.json()["embeddings"]]

    async def embed(self, inputs: list[Any]) -> list[list[float]]:
        """Vectors for texts, {"image": b64} and {"audio": b64} inputs, in order, in as few requests as fit."""
        batches: list[list[Any]] = []
        for item in inputs:
            media = isinstance(item, dict)
            if not batches or len(batches[-1]) >= (MEDIA_BATCH if media else TEXT_BATCH) \
                    or isinstance(batches[-1][0], dict) != media:
                batches.append([])
            batches[-1].append(item)

        async def send(batch: list[Any]) -> list[list[float]]:
            async with self._slots:
                return await self._request(batch)
        return [vector for done in await asyncio.gather(*map(send, batches)) for vector in done]

    async def query(self, text: str) -> list[float]:
        if text in self._queries:
            self._queries.move_to_end(text)
            return self._queries[text]
        [vector] = await self._request([self.query_prefix + text])
        self._queries[text] = vector
        while len(self._queries) > QUERY_CACHE:
            self._queries.popitem(last=False)
        return vector

    async def warm(self) -> None:
        """Load the model now, so the first search does not wait for it."""
        try:
            await self._request([self.query_prefix + "warm up"])
        except httpx.HTTPError:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()
