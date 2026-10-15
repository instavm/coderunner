"""EmbeddingGemma 2's text encoder on the Neural Engine, behind the parts of Ollama's and OpenAI's embedding APIs
that CodeRunner uses (/api/embed, /api/show, /api/tags, /v1/embeddings).

Started by coderunner.models.coreml with the Python of its own environment (coremltools, tokenizers, numpy); it
imports nothing from CodeRunner. Pictures and sound go through the build's own vision and audio encoders when it
has them, on the Neural Engine, and otherwise on to the same model in Ollama (--media-url). The model is released
after --idle seconds without a request and loaded again by the next one.

The Core ML build (FluidInference/embeddinggemma-2-coreml) has one fixed-shape function per length, which is what
keeps it on the Neural Engine: embed_32 ... embed_512. Text over 510 tokens is embedded in windows whose vectors
are averaged by length, as the encoder reads at most 512 tokens with <bos> and <eos>.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import io
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
VISION_TOKENS = 280       # the model's default budget: on 64 questions over ~/Downloads as good as Ollama's build on
                          # the GPU (MRR 0.76 both); 70 lost text-heavy screens. 0.53 s a picture on an M3 Pro
                          # (its GPU: 0.41 s, at three times the energy).
AUDIO_TOKENS = 500        # 20 s of sound (25 tokens a second) per embedding, with 4 special tokens within 512
PACKAGE = "EmbeddingGemma2Text.mlpackage"
COMPILED = "EmbeddingGemma2Text.mlmodelc"


def windows(ids: list[int], bos: int = 2, eos: int = 1) -> list[list[int]]:
    """Token windows the encoder can read, each with its own <bos> and <eos>."""
    return [[bos, *ids[start:start + WINDOW], eos] for start in range(0, max(len(ids), 1), WINDOW)]


def fit(width: int, height: int, pixels: int, block: int) -> tuple[int, int]:
    """Sides keeping the aspect ratio, multiples of `block`, with at most `pixels` pixels."""
    scale = math.sqrt(pixels / max(width * height, 1))
    w, h = max(block, int(width * scale // block) * block), max(block, int(height * scale // block) * block)
    while w * h > pixels:
        if w >= h and w > block:
            w -= block
        elif h > block:
            h -= block
        else:
            break
    return w, h


def combine(vectors: list[list[float]], weights: list[int]) -> list[float]:
    """The length-weighted mean of window vectors, normalized."""
    total = [sum(w * v[i] for v, w in zip(vectors, weights)) for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(x * x for x in total)) or 1.0
    return [x / norm for x in total]


class Encoder:
    """Tokenizer, token table and the Core ML functions, loaded on first use and released when idle. Pictures and
    sound go through the build's own vision and audio encoders when they are there: their tokens are read by the
    text model between <|image> ... <image|> or <|audio> ... <audio|>, as EmbeddingGemma 2 does natively."""

    def __init__(self, folder: pathlib.Path, units: str = "CPU_AND_NE", vision_tokens: int = 0):
        self.folder = pathlib.Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        self.units = units
        self.vision_tokens = vision_tokens or VISION_TOKENS
        self.state = threading.Condition()          # loading, and releasing only while no one uses the model
        self.users = 0
        self.media_lock = threading.Lock()          # a picture or a sound at a time through its encoder
        self.turns = threading.Condition()          # whose turn on the text model, a text at a time
        self.busy = False
        self.urgent = 0                             # searches waiting: indexing yields to them between texts
        self.models: dict[object, object] = {}
        self.tokenizer = self.table = self.positions = None

    @property
    def loaded(self) -> bool:
        return self.tokenizer is not None

    def media(self) -> set[str]:
        """The media this build can embed itself ("vision", "audio")."""
        return {name for name in ("vision", "audio")
                if name in self.config and (self.folder / self.config[name]["package"]).exists()}

    def _compiled(self, package: str = PACKAGE) -> pathlib.Path:
        import coremltools as ct
        path = self.folder / (pathlib.Path(package).stem + ".mlmodelc")
        if not path.exists():
            # coremltools wants the name to end in .mlmodelc; a folder per package, as encoders can compile at once
            staging = self.folder / f"compiling-{path.stem}"
            shutil.rmtree(staging, ignore_errors=True)
            ct.utils.compile_model(str(self.folder / package), str(staging / path.name))
            (staging / path.name).rename(path)
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

    def _encoder(self, name: str, function: str | None = None):
        key = (name, function)
        if key not in self.models:
            import coremltools as ct
            options = {"function_name": function} if function else {}
            self.models[key] = ct.models.CompiledMLModel(str(self._compiled(self.config[name]["package"])),
                                                         compute_units=getattr(ct.ComputeUnit, self.units), **options)
        return self.models[key]

    def _load(self) -> None:
        if self.tokenizer is None:
            import numpy as np
            from tokenizers import Tokenizer
            table = self.config["token_table"]
            self.table = np.memmap(self.folder / table["file"], dtype=np.uint16, mode="r", shape=tuple(table["shape"]))
            self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))

    def _rows(self, ids: list[int]):
        """Token embeddings as the text model reads them: the bf16 table row times sqrt(hidden size)."""
        import numpy as np
        rows = (np.asarray(self.table[ids], dtype=np.uint32) << 16).view(np.float32)     # bfloat16 -> float32
        return rows * np.float32(self.config["embed_scale"])

    def _sequence(self, x) -> list[float]:
        """The embedding of a sequence of input vectors [L, hidden]."""
        import numpy as np
        size = next(s for s in SIZES if s >= len(x))
        padded = np.zeros((1, size, x.shape[1]), dtype=np.float16)
        padded[0, :len(x)] = x
        mask = np.zeros((1, size), dtype=np.float16)
        mask[0, :len(x)] = 1
        out = self._model(size).predict({"inputs_embeds": padded, "attention_mask": mask})["embedding"]
        return [float(v) for v in out.reshape(-1)]

    def _window(self, ids: list[int]) -> list[float]:
        return self._sequence(self._rows(ids))

    def _wrapped(self, tokens, start: int, end: int):
        """<bos> <start> tokens <end> <eos>: media tokens are already in the text model's space."""
        import numpy as np
        bos, eos = self.config["bos_token_id"], self.config["eos_token_id"]
        head, tail = self._rows([bos, start]), self._rows([end, eos])
        return np.concatenate([head, np.asarray(tokens, dtype=np.float32), tail])

    def _turn(self, urgent: bool, work):
        # The turn is taken under one condition: indexing waiting for the model checks for searches again each
        # time it wakes, so a search waits for at most the piece in progress.
        with self.turns:
            self.turns.wait_for(lambda: not self.busy and (urgent or self.urgent == 0))
            self.busy = True
        try:
            with self._using():
                return work()
        finally:
            with self.turns:
                self.busy = False
                self.turns.notify_all()

    @contextlib.contextmanager
    def _using(self):
        with self.state:
            self._load()
            self.users += 1
        try:
            yield
        finally:
            with self.state:
                self.users -= 1

    def embed(self, texts: list[str], urgent: bool = False) -> list[list[float]]:
        """Vectors for texts. An urgent request (a search) goes before the next text of any other: indexing a
        batch of 64 passages holds the model for ~3 s, which a search used to wait out in full."""
        if urgent:
            with self.turns:
                self.urgent += 1
        try:
            def one(text: str) -> list[float]:
                bos, eos = self.config["bos_token_id"], self.config["eos_token_id"]
                parts = windows(self.tokenizer.encode(text, add_special_tokens=False).ids, bos, eos)
                return combine([self._window(part) for part in parts], [len(part) for part in parts])
            return [self._turn(urgent, lambda text=text: one(text)) for text in texts]
        finally:
            if urgent:
                with self.turns:
                    self.urgent -= 1
                    self.turns.notify_all()

    def image(self, data: bytes, tokens: int = 0) -> list[float]:
        """A picture through the vision encoder, prepared as the build's card says (HF Gemma4ImageProcessor):
        resized keeping its aspect ratio to sides that are multiples of 48 px within the token budget, cut into
        16 px patches row by row, three by three patches pooled into one token. The vision encoder runs beside
        the text model, so a search does not wait for a picture (5 ms instead of up to 0.5 s on an M3 Pro); only
        reading its tokens takes a turn on the text model."""
        vision = self.config["vision"]
        with self._using(), self.media_lock:
            found = self._image_tokens(data, tokens or self.vision_tokens)
        return self._turn(False, lambda: self._sequence(self._wrapped(found, vision["boi_token_id"],
                                                                      vision["eoi_token_id"])))

    def _image_tokens(self, data: bytes, tokens: int):
        import numpy as np
        from PIL import Image
        vision = self.config["vision"]
        function = f"vision_{tokens}"
        most = vision["functions"][function]["patches"]
        side, kernel = vision["patch_size"], vision["pooling_kernel"]
        block = side * kernel
        picture = Image.open(io.BytesIO(data)).convert("RGB")
        width, height = picture.size
        width, height = fit(width, height, most * side * side, block)
        picture = picture.resize((width, height), Image.Resampling.BICUBIC, reducing_gap=None)
        pixels = np.asarray(picture, dtype=np.float32) / 255.0
        rows, cols = height // side, width // side
        count = rows * cols
        patches = np.zeros((1, most, side * side * 3), dtype=np.float16)
        patches[0, :count] = pixels.reshape(rows, side, cols, side, 3).transpose(0, 2, 1, 3, 4).reshape(count, -1)
        ys, xs = np.divmod(np.arange(count), cols)
        positions = np.full((1, most, 2), -1, dtype=np.float16)
        positions[0, :count, 0], positions[0, :count, 1] = xs, ys
        if self.positions is None:
            table = vision["position_table"]
            self.positions = np.fromfile(self.folder / table["file"], dtype=np.float16).reshape(table["shape"])
        embedded = np.zeros((1, most, self.positions.shape[2]), dtype=np.float16)
        embedded[0, :count] = (self.positions[0][xs].astype(np.float32) + self.positions[1][ys]).astype(np.float16)
        valid = np.zeros((1, most), dtype=np.float16)
        valid[0, :count] = 1
        groups = (ys // kernel) * (cols // kernel) + xs // kernel
        pool = np.zeros((tokens, most), dtype=np.float16)
        pool[groups, np.arange(count)] = 1.0 / (kernel * kernel)
        out = self._encoder("vision", function).predict({"patches": patches, "positions": positions, "pool": pool,
                                                         "valid": valid, "position_embeddings": embedded})
        return out["image_tokens"][0, :(rows // kernel) * (cols // kernel)].astype(np.float32)

    def audio(self, wav: bytes) -> list[float]:
        """Sound through the audio encoder, 10 s at a time; up to AUDIO_TOKENS of its tokens make one embedding,
        a longer piece is embedded in parts averaged by length, as long text is. The audio encoder runs beside
        the text model, as the vision encoder does."""
        audio = self.config["audio"]
        with self._using(), self.media_lock:
            tokens = self._audio_tokens(wav, audio)

        def work() -> list[float]:
            import numpy as np
            starts = range(0, max(len(tokens), 1), AUDIO_TOKENS)
            parts = [tokens[i:i + AUDIO_TOKENS] for i in starts] or [np.zeros((0, 512), dtype=np.float32)]
            vectors = [self._sequence(self._wrapped(part, audio["boa_token_id"], audio["eoa_token_id"]))
                       for part in parts]
            return combine(vectors, [len(part) + 4 for part in parts])
        return self._turn(False, work)

    def _audio_tokens(self, wav: bytes, audio: dict):
        import numpy as np
        import wave
        with wave.open(io.BytesIO(wav)) as sound:
            if (sound.getframerate(), sound.getnchannels(), sound.getsampwidth()) != (audio["sample_rate"], 1, 2):
                raise ValueError("sound must be 16 kHz mono 16-bit WAV")
            samples = np.frombuffer(sound.readframes(sound.getnframes()), dtype="<i2").astype(np.float32) / 32768
        window, pad, frames = audio["sample_rate"] * audio["window_seconds"], audio["left_pad_samples"], audio["frames"]
        found = []
        for start in range(0, max(len(samples), 1), window):
            chunk = samples[start:start + window]
            waveform = np.zeros((1, audio["waveform_samples"]), dtype=np.float32)
            waveform[0, pad:pad + len(chunk)] = chunk
            mask = (np.arange(frames) * 160 + 321 <= pad + len(chunk)).astype(np.float16)[None]
            out = self._encoder("audio").predict({"waveform": waveform, "frame_mask": mask})["audio_tokens"][0]
            keep = mask[0, ::frames // audio["tokens_per_window"]][:audio["tokens_per_window"]] > 0
            found.append(out[keep].astype(np.float32))
        return np.concatenate(found) if found else np.zeros((0, 512), dtype=np.float32)

    def release(self) -> bool:
        """Free the model's memory, unless a request is using it (then the watcher tries again later)."""
        with self.state:
            if self.users:
                return False
            self.models.clear()
            self.tokenizer = self.table = self.positions = None
            gc.collect()
            return True


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
        own = self.own_media()
        here = [(i, item) for i, item in enumerate(inputs) if isinstance(item, dict)
                and (("image" in item and "vision" in own) or ("audio" in item and "audio" in own))]
        taken = {i for i, _ in here}
        media = [(i, item) for i, item in enumerate(inputs) if not isinstance(item, str) and i not in taken]
        if media and not (self.media_url and self.media_model):
            raise ValueError("pictures and sound need the embedding model in Ollama, which is not configured")
        out: list = [None] * len(inputs)
        for i, item in here:
            out[i] = (self.encoder.image(base64.b64decode(item["image"])) if "image" in item
                      else self.encoder.audio(base64.b64decode(item["audio"])))
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

    def own_media(self) -> set[str]:
        """Pictures ("vision") and sound ("audio") this server embeds itself, on the Neural Engine."""
        media = getattr(self.encoder, "media", None)
        return media() if callable(media) else set()

    def tags(self) -> dict:
        """The model with a digest naming the text weights and where media is embedded ("text:<id>|media:<id>"), so
        an index re-embeds only pictures and sound when only their encoder changes."""
        media = ""
        if self.own_media():
            media = "coreml-" + "-".join(sorted(self.own_media()))
        elif self.media_url and self.media_model:
            try:
                with urllib.request.urlopen(f"{self.media_url}/api/tags", timeout=5) as response:
                    models = json.load(response).get("models", [])
                wanted = {self.media_model, f"{self.media_model}:latest"}
                media = next((m.get("digest", "") for m in models if m.get("name") in wanted or m.get("model") in wanted), "")[:12]
            except (OSError, ValueError):
                media = ""
        digest = f"text:{self.digest}|media:{media or 'none'}"
        return {"models": [{"name": self.name, "model": self.name, "digest": digest}]}

    def capabilities(self) -> list[str]:
        if self.own_media() or not (self.media_url and self.media_model):
            return sorted({"embedding", *self.own_media()})
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
            if self.encoder.loaded and time.monotonic() - self.used >= self.idle and self.encoder.release() is not False:
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
    p.add_argument("--vision-tokens", type=int, default=VISION_TOKENS, help="image encoder budget: 70, 140 or 280")
    args = p.parse_args(argv)
    service = Service(Encoder(pathlib.Path(args.folder), args.units, args.vision_tokens), args.name, args.digest,
                      args.media_url, args.media_model, args.idle)
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
