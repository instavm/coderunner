"""EmbeddingGemma 2's text encoder on the Neural Engine, behind the parts of Ollama's and OpenAI's embedding APIs
that CodeRunner uses (/api/embed, /api/show, /api/tags, /v1/embeddings).

Started by coderunner.models.coreml with the Python of its own environment (coremltools, tokenizers, numpy); it
imports nothing from CodeRunner. Pictures and sound go on to the same model in Ollama (--media-url), whose vectors
share the text encoder's space. The model is released after --idle seconds without a request and loaded again by
the next one.

The Core ML build (FluidInference/embeddinggemma-2-coreml) has one fixed-shape function per length, which is what
keeps it on the Neural Engine: embed_32 ... embed_512. Text over 510 tokens is embedded in windows whose vectors
are averaged by length, as the encoder reads at most 512 tokens with <bos> and <eos>.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Three of the build's lengths (it has 32-512): each is compiled for the Neural Engine on first use, 15-33 s on
# an M3 Pro, so fewer means a shorter wait once; padding a 10-token query to 64 costs a millisecond or two.
SIZES = (64, 256, 512)
WINDOW = SIZES[-1] - 2
URGENT_TEXTS = 2
PACKAGE = "EmbeddingGemma2Text.mlpackage"
COMPILED = "EmbeddingGemma2Text.mlmodelc"


def windows(ids: list[int], bos: int = 2, eos: int = 1) -> list[list[int]]:
    """Token windows the encoder can read, each with its own <bos> and <eos>."""
    return [[bos, *ids[start:start + WINDOW], eos] for start in range(0, max(len(ids), 1), WINDOW)]


def combine(vectors: list[list[float]], weights: list[int]) -> list[float]:
    """The length-weighted mean of window vectors, normalized."""
    total = [sum(w * v[i] for v, w in zip(vectors, weights)) for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(x * x for x in total)) or 1.0
    return [x / norm for x in total]


class Encoder:
    """Tokenizer, token table and the Core ML functions, loaded on first use and released when idle."""

    def __init__(self, folder: pathlib.Path, units: str = "CPU_AND_NE"):
        self.folder = pathlib.Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        self.units = units
        self.lock = threading.Lock()                # one request at a time on the model, a text at a time
        self.turns = threading.Condition()
        self.urgent = 0                             # searches waiting: indexing yields to them between texts
        self.models: dict[int, object] = {}
        self.tokenizer = self.table = None

    @property
    def loaded(self) -> bool:
        return self.tokenizer is not None

    def _compiled(self) -> pathlib.Path:
        import coremltools as ct
        path = self.folder / COMPILED
        if not path.exists():
            staging = self.folder / "compiling"           # coremltools wants the name to end in .mlmodelc
            shutil.rmtree(staging, ignore_errors=True)
            ct.utils.compile_model(str(self.folder / PACKAGE), str(staging / COMPILED))
            (staging / COMPILED).rename(path)
            shutil.rmtree(staging, ignore_errors=True)
        return path

    def _model(self, size: int):
        if size not in self.models:
            import coremltools as ct
            # The first load of each function compiles it for the Neural Engine; Core ML caches that, so later
            # loads take about a second.
            self.models[size] = ct.models.CompiledMLModel(str(self._compiled()), function_name=f"embed_{size}",
                                                          compute_units=getattr(ct.ComputeUnit, self.units))
        return self.models[size]

    def _load(self) -> None:
        if self.tokenizer is None:
            import numpy as np
            from tokenizers import Tokenizer
            table = self.config["token_table"]
            self.table = np.memmap(self.folder / table["file"], dtype=np.uint16, mode="r", shape=tuple(table["shape"]))
            self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))

    def _window(self, ids: list[int]) -> list[float]:
        import numpy as np
        size = next(s for s in SIZES if s >= len(ids))
        rows = (np.asarray(self.table[ids], dtype=np.uint32) << 16).view(np.float32)     # bfloat16 -> float32
        x = np.zeros((1, size, rows.shape[1]), dtype=np.float16)
        x[0, :len(ids)] = rows * np.float32(self.config["embed_scale"])
        mask = np.zeros((1, size), dtype=np.float16)
        mask[0, :len(ids)] = 1
        out = self._model(size).predict({"inputs_embeds": x, "attention_mask": mask})["embedding"]
        return [float(v) for v in out.reshape(-1)]

    def embed(self, texts: list[str], urgent: bool = False) -> list[list[float]]:
        """Vectors for texts. An urgent request (a search) goes before the next text of any other: indexing a
        batch of 64 passages holds the model for ~3 s, which a search used to wait out in full."""
        if urgent:
            with self.turns:
                self.urgent += 1
        try:
            vectors = []
            for text in texts:
                if not urgent:
                    with self.turns:
                        self.turns.wait_for(lambda: self.urgent == 0)
                with self.lock:
                    self._load()
                    bos, eos = self.config["bos_token_id"], self.config["eos_token_id"]
                    ids = self.tokenizer.encode(text, add_special_tokens=False).ids
                    parts = windows(ids, bos, eos)
                    vectors.append(combine([self._window(p) for p in parts], [len(p) for p in parts]))
            return vectors
        finally:
            if urgent:
                with self.turns:
                    self.urgent -= 1
                    self.turns.notify_all()

    def release(self) -> None:
        with self.lock:
            self.models.clear()
            self.tokenizer = self.table = None
            gc.collect()


class Service:
    def __init__(self, encoder, name: str, digest: str = "", media_url: str = "", media_model: str = "",
                 idle: float = 0):
        self.encoder, self.name, self.digest = encoder, name, digest
        self.media_url, self.media_model = media_url.rstrip("/").removesuffix("/v1"), media_model
        self.idle, self.used = idle, time.monotonic()
        self._caps: tuple[float, list[str]] = (0.0, [])

    def _media(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(f"{self.media_url}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def embed(self, inputs: list, keep_alive=None) -> list[list[float]]:
        """Texts here, pictures and sound ({"image": b64} / {"audio": b64}) from the media model, in order."""
        self.used = time.monotonic()
        texts = [(i, item) for i, item in enumerate(inputs) if isinstance(item, str)]
        media = [(i, item) for i, item in enumerate(inputs) if not isinstance(item, str)]
        if media and not (self.media_url and self.media_model):
            raise ValueError("pictures and sound need the embedding model in Ollama, which is not configured")
        out: list = [None] * len(inputs)
        # A request of one or two texts is a search or a memory lookup, someone waiting; a batch is indexing.
        urgent = 0 < len(texts) <= URGENT_TEXTS
        for (i, _), vector in zip(texts, self.encoder.embed([t for _, t in texts], urgent=urgent) if texts else []):
            out[i] = vector
        if media:
            body = {"model": self.media_model, "input": [m for _, m in media]}
            if keep_alive is not None:
                body["keep_alive"] = keep_alive
            for (i, _), vector in zip(media, self._media("/api/embed", body)["embeddings"]):
                out[i] = vector
        self.used = time.monotonic()
        return out

    def tags(self) -> dict:
        """The model with a digest for the text weights and the media model's, so an index notices either change."""
        digest = self.digest
        if self.media_url and self.media_model:
            try:
                with urllib.request.urlopen(f"{self.media_url}/api/tags", timeout=5) as response:
                    models = json.load(response).get("models", [])
                wanted = {self.media_model, f"{self.media_model}:latest"}
                media = next((m.get("digest", "") for m in models if m.get("name") in wanted or m.get("model") in wanted), "")
                digest = f"{digest}+{media[:12]}" if media else digest
            except (OSError, ValueError):
                pass
        return {"models": [{"name": self.name, "model": self.name, "digest": digest}]}

    def capabilities(self) -> list[str]:
        if not (self.media_url and self.media_model):
            return ["embedding"]
        if time.monotonic() - self._caps[0] > 60:
            try:
                found = self._media("/api/show", {"model": self.media_model}).get("capabilities") or []
            except (OSError, ValueError):
                found = []
            self._caps = (time.monotonic(), sorted({"embedding", *found}))
        return self._caps[1]

    def watch(self, every: float = 15.0) -> None:
        """Release the model after `idle` seconds without a request."""
        while self.idle:
            time.sleep(min(every, self.idle))
            if self.encoder.loaded and time.monotonic() - self.used >= self.idle:
                self.encoder.release()
                print(f"released the model after {self.idle:.0f} s idle", flush=True)


def handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:  # one line per request is noise in the log
            pass

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "loaded": service.encoder.loaded})
            elif self.path == "/api/tags":
                self._send(200, service.tags())
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model", "owned_by": "local"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/show":
                    self._send(200, {"capabilities": service.capabilities()})
                    return
                if self.path not in ("/api/embed", "/v1/embeddings"):
                    self._send(404, {"error": "not found"})
                    return
                inputs = body.get("input")
                inputs = [inputs] if isinstance(inputs, (str, dict)) else list(inputs or [])
                vectors = service.embed(inputs, body.get("keep_alive"))
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            except (OSError, urllib.error.URLError) as exc:
                self._send(502, {"error": f"media model: {exc}"})
                return
            except Exception as exc:  # a Core ML failure: report it rather than drop the connection
                traceback.print_exc()
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if self.path == "/api/embed":
                self._send(200, {"model": service.name, "embeddings": vectors})
            else:
                self._send(200, {"object": "list", "model": service.name, "data": [
                    {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]})
    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folder", required=True, help="the downloaded Core ML build")
    p.add_argument("--name", required=True, help="the model name clients send")
    p.add_argument("--port", type=int, default=8084)
    p.add_argument("--digest", default="", help="identifies the weights, so an index knows when vectors change")
    p.add_argument("--media-url", default="", help="Ollama, for pictures and sound")
    p.add_argument("--media-model", default="")
    p.add_argument("--idle", type=float, default=600, help="seconds without requests before the model is released")
    p.add_argument("--units", default="CPU_AND_NE", help="Core ML compute units (CPU_AND_NE, CPU_AND_GPU, ALL)")
    args = p.parse_args(argv)
    service = Service(Encoder(pathlib.Path(args.folder), args.units), args.name, args.digest, args.media_url,
                      args.media_model, args.idle)
    threading.Thread(target=service.watch, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(service))
    print(f"serving {args.name} on 127.0.0.1:{args.port} ({args.units})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
