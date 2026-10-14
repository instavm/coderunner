"""Core ML on Apple silicon: models on the Neural Engine, which leaves the GPU to the assistant's model.

EmbeddingGemma 2's text encoder runs on the Neural Engine from FluidInference's Core ML build, served by
coreml_server.py in an environment of its own (coremltools, tokenizers, numpy) under assistant_dir()/runtimes.
Pictures and sound still go to the same model in Ollama, whose vectors share the space.

Measured on an M3 Pro embedding 821 passages of CodeRunner's own files (macmon for power): 40 s against 53 s for
Ollama's build on the GPU, 3.6 W against 8.8 W above idle (178 against 570 mJ a passage), the GPU 1% busy against
76%, and the same files found for 20 questions. A larger Mac's GPU may be the faster one, so `compare` times both
on this machine and Core ML is used unless it is slower by more than TOLERANCE.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import time

from ..core import paths
from .hardware import Hardware
from .hub import Hub
from .runtimes import Progress, Runtime, _ok, _spawn, _terminate, _wait_ready, idle_seconds

PORT = 8084
PACKAGES = ["coremltools==9.0", "tokenizers==0.23.3", "numpy>=2,<3", "pillow>=11,<13"]
SERVER = pathlib.Path(__file__).with_name("coreml_server.py")
TOLERANCE = 0.10        # Core ML may be this much slower than the GPU build (about run-to-run noise)
PARITY = 0.97           # mean cosine to the GPU build's vectors below this means a broken build
SAMPLE = 96             # passages timed by compare()


def supported() -> bool:
    """Apple silicon with macOS 15 or later (the build's functions share weights, a macOS 15 feature)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 15
    except ValueError:
        return False


class CoreMLRuntime(Runtime):
    """The Core ML embedding server: Ollama's and OpenAI's embedding APIs on 127.0.0.1."""
    name = "coreml"

    def __init__(self, config=None, port: int = PORT, hub: Hub | None = None):
        super().__init__(config)
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.hub = hub
        self.root = paths.assistant_dir() / "models" / "coreml"
        self.env = paths.assistant_dir() / "runtimes" / "coreml"

    def python(self) -> str | None:
        """The environment's Python, once it holds exactly the pinned packages."""
        found, ready = self.env / "bin" / "python", self.env / "installed.txt"
        try:
            return str(found) if found.exists() and ready.read_text() == "\n".join(PACKAGES) else None
        except OSError:
            return None

    def available(self) -> bool:
        return supported() and self.python() is not None

    def install(self, progress: Progress | None = None) -> None:
        """A virtual environment with the pinned packages the server needs (about 120 MB)."""
        if not supported():
            raise RuntimeError("Core ML models need Apple silicon with macOS 15 or later")
        if progress:
            progress("installing Core ML packages (coremltools, tokenizers)")
        shutil.rmtree(self.env, ignore_errors=True)
        for cmd in ([sys.executable, "-m", "venv", str(self.env)],
                    [str(self.env / "bin" / "python"), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                     "--retries", "10", "--timeout", "60", *PACKAGES]):      # a timed-out index read looks like
                                                                                # "no versions found"
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode:
                shutil.rmtree(self.env, ignore_errors=True)
                raise RuntimeError(f"Installing the Core ML runtime failed: {(done.stderr or done.stdout)[-400:]}")
        (self.env / "installed.txt").write_text("\n".join(PACKAGES))

    @staticmethod
    def artifact(entry: dict) -> dict:
        return (entry.get("artifacts") or {}).get("coreml") or {}

    def folder(self, entry: dict) -> pathlib.Path:
        return self.root / self.artifact(entry)["repo"].replace("/", "--")

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        if not self.python():
            self.install(progress)
        art = self.artifact(entry)
        hub = self.hub or Hub()
        for name in art["files"]:
            hub.download(art["repo"], name, self.folder(entry) / name, revision=art["revision"], progress=progress)

    def installed(self, entry: dict) -> bool:
        art = self.artifact(entry)
        return bool(art) and all((self.folder(entry) / name).exists() for name in art["files"])

    def remove(self, entry: dict) -> None:
        if self.artifact(entry):
            folder = self.folder(entry).resolve()
            if self.root.resolve() in folder.parents:
                shutil.rmtree(folder, ignore_errors=True)

    def running(self) -> bool:
        return _ok(f"http://127.0.0.1:{self.port}/health")

    def command(self, entry: dict) -> list[str]:
        config = self.config
        get = config.get if config is not None else (lambda key, default=None: default)
        art = self.artifact(entry)
        return [self.python() or sys.executable, "-I", str(SERVER), "--folder", str(self.folder(entry)),
                "--name", self.model_id(entry), "--port", str(self.port),
                "--digest", art.get("text_weights") or art.get("revision", ""),
                "--idle", str(idle_seconds(config)),
                "--media-url", get("models.embedding_media_url") or "", "--media-model", get("models.embedding_media") or ""]

    def start(self, timeout: float = 120) -> None:
        if self.running():
            return
        from . import catalog
        if supported() and self.env.exists() and not self.python():
            self.install()                  # the pinned packages changed (pictures need Pillow): bring them up to date
        if not self.available():
            raise RuntimeError("The Core ML runtime is not installed; run `coderunner models setup`")
        entry = catalog.get((self.config.get("models.embedding_entry") if self.config else None) or "embeddinggemma-2")
        proc = _spawn("coreml-server", self.command(entry))
        _wait_ready(self.running, proc, timeout, "the Core ML server", str(paths.assistant_dir() / "logs" / "coreml-server.log"))

    def stop(self) -> None:
        _terminate("coreml-server", "python")


def passages(count: int = SAMPLE, prefix: str = "title: {title} | text: ") -> list[str]:
    """Passages cut from CodeRunner's own source as the file index cuts text: the same sample on every machine."""
    from ..files.extract import TEXT_CHUNK, TEXT_OVERLAP
    found: list[str] = []
    for path in sorted(pathlib.Path(__file__).resolve().parent.parent.rglob("*.py")):
        text = path.read_text(errors="replace")
        for start in range(0, len(text), TEXT_CHUNK - TEXT_OVERLAP):
            found.append(prefix.format(title=path.name) + text[start:start + TEXT_CHUNK])
            if len(found) >= count:
                return found
    return found


async def warm(url: str, model: str) -> None:
    """Compile each of the server's lengths for the Neural Engine now, during setup, not on the first search."""
    from ..files.embed import Embedder
    embedder = Embedder(url, model, keep_alive="5m")
    try:
        await embedder.embed(["word " * 20, "word " * 150, "word " * 400])
    finally:
        await embedder.aclose()


async def compare(coreml_url: str, coreml_model: str, gpu_url: str, gpu_model: str, count: int = SAMPLE) -> dict:
    """Time the Core ML build against the GPU build on the same passages, after one untimed pass each (the first
    Core ML pass compiles for the Neural Engine), and check their vectors agree."""
    from ..files.embed import Embedder
    sample = passages(count)
    rates, vectors = {}, {}
    for key, url, model in (("coreml", coreml_url, coreml_model), ("gpu", gpu_url, gpu_model)):
        embedder = Embedder(url, model, keep_alive="5m")
        try:
            await embedder.embed(sample)
            began = time.perf_counter()
            vectors[key] = await embedder.embed(sample)
            rates[key] = len(sample) / (time.perf_counter() - began)
        finally:
            await embedder.aclose()
    cosine = sum(sum(a * b for a, b in zip(u, v)) for u, v in zip(vectors["coreml"], vectors["gpu"])) / len(sample)
    return {"coreml_per_s": round(rates["coreml"], 1), "gpu_per_s": round(rates["gpu"], 1), "cosine": round(cosine, 4),
            "passages": len(sample)}


def choice(measured: dict) -> tuple[str, str]:
    """("coreml" | "gpu", why) from a comparison."""
    if measured["cosine"] < PARITY:
        return "gpu", f"the Core ML vectors differ from the GPU build's (cosine {measured['cosine']:.3f})"
    if measured["coreml_per_s"] < measured["gpu_per_s"] * (1 - TOLERANCE):
        return "gpu", (f"the GPU is faster here ({measured['gpu_per_s']:.0f} against "
                       f"{measured['coreml_per_s']:.0f} passages/s)")
    return "coreml", (f"{measured['coreml_per_s']:.0f} passages/s on the Neural Engine against "
                      f"{measured['gpu_per_s']:.0f} on the GPU, which stays free for the assistant")


def _path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "coreml.json"


def load(hw: Hardware) -> dict | None:
    """The latest comparison on this machine."""
    from .bench import fingerprint
    machine = fingerprint(hw)
    try:
        records = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    return next((r for r in reversed(records) if r.get("machine") == machine), None)


def save(hw: Hardware, measured: dict) -> dict:
    from .bench import fingerprint
    try:
        records = json.loads(_path().read_text())
    except (OSError, ValueError):
        records = []
    record = {"machine": fingerprint(hw), "date": datetime.date.today().isoformat(), **measured}
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps([*records, record][-50:], indent=1))
    return record
