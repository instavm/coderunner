"""Local model runtimes behind one interface.

Every runtime serves an OpenAI-compatible API on 127.0.0.1. Servers that
CodeRunner starts log to assistant_dir()/logs and record their pid in
assistant_dir()/run, and stop() only ever signals a process it started.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable

import httpx

from ..core import paths
from . import catalog
from .hardware import Hardware
from .hub import Hub

Progress = Callable[[str], None]
_procs: dict[str, subprocess.Popen] = {}


def _ok(url: str, headers: dict | None = None, timeout: float = 2.0, trust_env: bool = False) -> bool:
    try:
        return httpx.get(url, headers=headers or {}, timeout=timeout, trust_env=trust_env).status_code == 200
    except httpx.HTTPError:
        return False


def _pidfile(name: str) -> pathlib.Path:
    return paths.assistant_dir() / "run" / f"{name}.pid"


def _spawn(name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    paths.ensure_dirs()
    with open(paths.assistant_dir() / "logs" / f"{name}.log", "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True)
    _procs[name] = proc
    _pidfile(name).write_text(str(proc.pid))
    return proc


def _alive(name: str, pid: int, expect: str) -> bool:
    if name in _procs and _procs[name].pid == pid:
        return _procs[name].poll() is None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True).stdout
    return expect in out  # guards against a recycled pid


def _terminate(name: str, expect: str, timeout: float = 15) -> bool:
    path = _pidfile(name)
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return False
    stopped = False
    if _alive(name, pid, expect):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)  # started with start_new_session, so this includes runner children
            except OSError:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and _alive(name, pid, expect):
                time.sleep(0.2)
            if not _alive(name, pid, expect):
                stopped = True
                break
    path.unlink(missing_ok=True)
    _procs.pop(name, None)
    return stopped


def _wait_ready(check: Callable[[], bool], proc: subprocess.Popen, timeout: float, what: str, log: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited with code {proc.returncode}; see {log}")
        time.sleep(0.5)
    raise RuntimeError(f"{what} did not become ready within {timeout:.0f}s; see {log}")


def _throttled(progress: Progress | None, every: float = 2.0) -> Progress:
    last = [0.0]

    def emit(message: str, force: bool = False) -> None:
        if progress and (force or time.monotonic() - last[0] >= every):
            last[0] = time.monotonic()
            progress(message)
    return emit


def models_dir(config=None) -> pathlib.Path:
    """Where model files are kept: models.dir when the owner chose a folder (another disk, say), else
    CODERUNNER_HOME/assistant/models. What CodeRunner learns about this machine (measurements, the list of
    discovered models) stays in the home either way."""
    chosen = (config.get("models.dir") if config is not None else "") or ""
    return pathlib.Path(chosen).expanduser() if chosen else paths.assistant_dir() / "models"


PORT_BASE = 8081


def server_port(config, offset: int) -> int:
    """The loopback port of a model server CodeRunner runs: llama.cpp at models.port_base (8081), Prism's build,
    the browser driver and the Core ML embedding server on the three after it. Another base lets a second
    CodeRunner home (a test run, a bench) serve its own models beside the first instead of talking to its
    servers."""
    base = config.get("models.port_base", PORT_BASE) if config is not None else PORT_BASE
    return int(base or PORT_BASE) + offset


def idle_seconds(config) -> int:
    """How long a model may sit unused before its runtime frees its memory (models.idle_unload_minutes)."""
    minutes = 10 if config is None else config.get("models.idle_unload_minutes", 10)
    return max(0, int(float(minutes or 0) * 60))


class Runtime:
    name = ""
    base_url = ""
    api_key = ""

    def __init__(self, config=None):
        self.config = config

    def available(self) -> bool:
        raise NotImplementedError

    def running(self) -> bool:
        raise NotImplementedError

    def start(self, timeout: float = 60) -> None:
        pass

    def stop(self) -> None:
        pass

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        pass

    def prepare(self, entry: dict, ctx: int) -> str:
        """Make the pulled model servable at `ctx` tokens and return the id clients should request."""
        return self.model_id(entry)

    def model_id(self, entry: dict) -> str:
        return f"coderunner-{entry['id']}"

    def installed(self, entry: dict) -> bool:
        return False

    def build_format(self, entry: dict) -> str | None:
        """The format of the build the runtime serves for this entry (gguf...), when it can tell."""
        return None

    def remove(self, entry: dict) -> None:
        """Delete the model's weights (setup does this for models it downloaded and then rejected)."""

    def vision(self, entry: dict, model: str | None = None) -> bool:
        return catalog.has_vision(entry, self.name)

    def describe(self) -> dict:
        return {"name": self.name, "base_url": self.base_url, "available": self.available(),
                "running": self.running()}


def parse_version(text: str) -> tuple[int, int, int] | None:
    import re
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(part) for part in match.groups()) if match else None   # type: ignore[return-value]


def _unpack(archive: pathlib.Path, into: pathlib.Path) -> None:
    """Unpack a release archive whose checksum was checked. Python reads .tar.zst itself from 3.14; before that
    the zstandard package does (installed with CodeRunner on Linux), and failing that the system's tar, which
    needs zstd installed (a minimal Debian has neither)."""
    import tarfile
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(into, filter="data")     # refuses absolute paths, links out, devices
        return
    except tarfile.ReadError:
        if not archive.name.endswith(".zst"):
            raise
    try:
        import zstandard
    except ImportError:
        zstandard = None
    if zstandard is not None:
        try:
            with open(archive, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream, \
                    tarfile.open(fileobj=stream, mode="r|") as tar:
                tar.extractall(into, filter="data")
            return
        except (zstandard.ZstdError, tarfile.ReadError) as exc:
            raise RuntimeError(f"{archive.name} could not be unpacked: {exc}") from exc
    tar = shutil.which("tar")
    done = subprocess.run([tar, "--zstd", "-xf", str(archive), "-C", str(into)], capture_output=True,
                          text=True) if tar else None
    if done is None or done.returncode != 0:
        raise RuntimeError(f"{archive.name} could not be unpacked: install zstd (for example `sudo apt install "
                           f"zstd`) and run setup again{': ' + done.stderr.strip()[-200:] if done else ''}")


class OllamaRuntime(Runtime):
    """Ollama, preferring CodeRunner's own copy (installed when the owner has none, or one too old) over the
    system one.

    EmbeddingGemma 2 needs Ollama 0.36.0 and decision models (/v1/systemone) 0.35.0. Where no Ollama is
    installed, or an older one, setup downloads a pinned release, checked against its published SHA-256, into
    CODERUNNER_HOME/runtime; the owner's own Ollama app is left alone.
    """

    name = "ollama"
    APP_BINARY = "/Applications/Ollama.app/Contents/Resources/ollama"
    MIN_VERSION = (0, 36, 0)
    RELEASE = "0.40.3"
    # (os, arch) -> (asset, sha256), from the release's own checksums. The Linux archives carry NVIDIA's CUDA
    # libraries (1.4 GB); AMD's ROCm libraries are a separate archive, which is not fetched.
    BUILDS = {
        ("darwin", "arm64"): ("ollama-darwin.tgz", "a2d1d906a2c8febd9cb18f5c183854d0a6db5dc522940d68b0c0ed048e8dfdea"),
        ("darwin", "x86_64"): ("ollama-darwin.tgz", "a2d1d906a2c8febd9cb18f5c183854d0a6db5dc522940d68b0c0ed048e8dfdea"),
        ("linux", "x86_64"): ("ollama-linux-amd64.tar.zst",
                              "ce6f8b671a699bf1ebafb28d60dbbabf4f37b30b499145fa308a8ba72de021d6"),
        ("linux", "aarch64"): ("ollama-linux-arm64.tar.zst",
                               "5754700747a93ea14db88888f08072386d2da6fbf7dd75be19a684348b9f28b8"),
    }

    def __init__(self, config=None, host: str = "127.0.0.1:11434"):
        super().__init__(config)
        self.host = host
        self.api = f"http://{host}"
        self.base_url = f"{self.api}/v1"
        self.managed_dir = paths.home() / "runtime" / f"ollama-{self.RELEASE}"

    def _managed(self) -> pathlib.Path | None:
        """CodeRunner's own copy: at the top of the macOS archive, under bin/ in the Linux ones."""
        for found in (self.managed_dir / "ollama", self.managed_dir / "bin" / "ollama"):
            if found.is_file() and os.access(found, os.X_OK):
                return found
        return None

    def binary(self) -> str | None:
        managed = self._managed()
        if managed:
            return str(managed)
        return shutil.which("ollama") or (self.APP_BINARY if os.path.exists(self.APP_BINARY) else None)

    def available(self) -> bool:
        return self.binary() is not None

    def running(self) -> bool:
        return _ok(f"{self.api}/api/version")

    def unload(self, model: str) -> None:
        """Free one model's memory now; the server and its other models (embeddings) stay up."""
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(f"{self.api}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30,
                       trust_env=False)

    def version(self) -> tuple[int, int, int] | None:
        """The running server's version (what answers requests), else the binary's."""
        try:
            return parse_version(httpx.get(f"{self.api}/api/version", timeout=2, trust_env=False).json()["version"])
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        binary = self.binary()
        if not binary:
            return None
        try:
            out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        # "ollama version is X"; with a server of another version running it adds "client version is Y".
        found = [parse_version(line) for line in (out.stdout + out.stderr).splitlines() if "client version" in line]
        return found[0] if found and found[0] else parse_version(out.stdout + out.stderr)

    def outdated(self) -> bool:
        found = self.version()
        return found is not None and found < self.MIN_VERSION

    @classmethod
    def build(cls, system: str | None = None, machine: str | None = None) -> tuple[str, str] | None:
        """The pinned release's archive for a platform (this one by default), or None where there is none."""
        system = (system or platform.system()).lower()
        machine = (machine or platform.machine()).lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
        if machine in ("arm64", "aarch64"):         # one chip, named differently by each system
            machine = "aarch64" if system == "linux" else "arm64"
        return cls.BUILDS.get((system, machine))

    def can_install(self, system: str | None = None) -> bool:
        return self.build(system) is not None

    def install(self, progress: Progress | None = None) -> str:
        """Download the pinned Ollama release into CODERUNNER_HOME/runtime (macOS and Linux)."""
        import hashlib
        build = self.build()
        if build is None:
            raise RuntimeError(f"Update Ollama to {'.'.join(map(str, self.MIN_VERSION))} or later. "
                               f"{install_hint(runtime='ollama')}")
        asset, sha256 = build
        url = f"https://github.com/ollama/ollama/releases/download/v{self.RELEASE}/{asset}"
        self.managed_dir.mkdir(parents=True, exist_ok=True)
        archive = self.managed_dir / asset
        emit = _throttled(progress, every=5)
        emit(f"downloading Ollama {self.RELEASE}", True)
        digest, done = hashlib.sha256(), 0
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            with open(archive, "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if total:
                        emit(f"downloading Ollama {self.RELEASE}: {100 * done / total:.0f}% of {total / 1e9:.2f} GB")
        if digest.hexdigest() != sha256:
            archive.unlink()
            raise RuntimeError(f"Ollama {self.RELEASE} ({asset}) does not match its published checksum")
        try:
            _unpack(archive, self.managed_dir)
        finally:
            archive.unlink(missing_ok=True)
        binary = self._managed()
        if binary is None:
            raise RuntimeError(f"Ollama {self.RELEASE} has no ollama binary")
        return str(binary)

    def ensure_current(self, progress: Progress | None = None) -> None:
        """Make sure the Ollama that will serve requests is new enough, installing CodeRunner's copy if needed."""
        if not self.outdated():
            return
        found = ".".join(map(str, self.version() or ()))
        need = ".".join(map(str, self.MIN_VERSION))
        if self.running():
            if not self.stop():
                raise RuntimeError(
                    f"Ollama {found} is running (probably the Ollama app), but EmbeddingGemma 2 and decision "
                    f"models need {need} or later. Update the app (menu bar: Restart to Update) or quit it so "
                    "CodeRunner can start its own copy, then run setup again.")
        if self.can_install():
            self.install(progress)
        if self.outdated():
            raise RuntimeError(f"Ollama {found} is older than {need}. {install_hint(runtime='ollama')}")

    def _env(self) -> dict:
        env = {**os.environ, "OLLAMA_HOST": self.host}
        idle = idle_seconds(self.config)
        if idle:
            env.setdefault("OLLAMA_KEEP_ALIVE", f"{idle}s")       # models unload after this long unused
        return env

    def start(self, timeout: float = 30) -> None:
        if self.running():
            return
        if not self.available():
            raise RuntimeError(f"Ollama is not installed. {install_hint(runtime='ollama')}")
        proc = _spawn("ollama", [self.binary(), "serve"], self._env())
        _wait_ready(self.running, proc, timeout, "ollama serve", str(paths.assistant_dir() / "logs" / "ollama.log"))

    def stop(self) -> bool:
        return _terminate("ollama", "ollama")     # only a server CodeRunner started

    def tag(self, entry: dict) -> str:
        return entry["artifacts"]["ollama"]["tag"]

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        tag, emit = self.tag(entry), _throttled(progress)
        emit(f"ollama pull {tag}", True)
        with httpx.stream("POST", f"{self.api}/api/pull", json={"model": tag, "stream": True},
                          timeout=httpx.Timeout(30, read=None), trust_env=False) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"ollama pull {tag} failed: {response.read().decode()[:300]}")
            for line in response.iter_lines():
                event = json.loads(line) if line.strip() else {}
                if event.get("error"):
                    raise RuntimeError(f"ollama pull {tag} failed: {event['error']}")
                total, done = event.get("total"), event.get("completed")
                if total and done is not None:
                    emit(f"pulling {tag}: {100 * done / total:.0f}% of {total / 1e9:.2f} GB", done == total)
                elif event.get("status") == "success":
                    emit(f"pulled {tag}", True)

    def installed(self, entry: dict) -> bool:
        try:
            names = {m["name"] for m in httpx.get(f"{self.api}/api/tags", timeout=5, trust_env=False).json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return False
        tag = self.tag(entry)
        return tag in names or f"{tag}:latest" in names

    def build_format(self, entry: dict) -> str | None:
        try:
            shown = httpx.post(f"{self.api}/api/show", json={"model": self.tag(entry)}, timeout=10, trust_env=False)
            return (shown.json().get("details") or {}).get("format") or None
        except (httpx.HTTPError, ValueError):
            return None

    def remove(self, entry: dict) -> None:
        for name in (self.model_id(entry), self.tag(entry)):
            try:
                httpx.request("DELETE", f"{self.api}/api/delete", json={"model": name}, timeout=30, trust_env=False)
            except httpx.HTTPError:
                pass

    @staticmethod
    def modelfile(tag: str, ctx: int) -> str:
        # The OpenAI-compatible endpoint ignores per-request num_ctx, so bake it into an alias.
        return f"FROM {tag}\nPARAMETER num_ctx {ctx}\n"

    def alias_command(self, entry: dict, modelfile_path: os.PathLike | str) -> list[str]:
        return [self.binary() or "ollama", "create", self.model_id(entry), "-f", str(modelfile_path)]

    def prepare(self, entry: dict, ctx: int) -> str:
        folder = paths.assistant_dir() / "models" / "ollama"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.model_id(entry)}.Modelfile"
        path.write_text(self.modelfile(self.tag(entry), ctx))
        result = subprocess.run(self.alias_command(entry, path), env=self._env(), capture_output=True,
                                text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ollama create failed: {(result.stderr or result.stdout).strip()[-300:]}")
        return self.model_id(entry)

    def capabilities(self, model: str) -> list[str]:
        try:
            data = httpx.post(f"{self.api}/api/show", json={"model": model}, timeout=10, trust_env=False).json()
        except (httpx.HTTPError, ValueError):
            return []
        return data.get("capabilities") or []  # null for some models

    def vision(self, entry: dict, model: str | None = None) -> bool:
        if not catalog.has_vision(entry, self.name):
            return False
        return "vision" in self.capabilities(model or self.model_id(entry))


# llama.cpp's official build, checked against the SHA-256 GitHub publishes for each asset.
LLAMA_RELEASE = "b12776"
LLAMA_BUILDS = {  # (os, arch) -> (asset, sha256)
    ("darwin", "arm64"): ("macos-arm64", "cb0c5c97a1a4ad50a0d64e228a67694de5c3a876d1ae0077b39e3c2381f3a039"),
    ("darwin", "x86_64"): ("macos-x64", "e644d574b6800879163352cea41b4ebbe03d88a6704adab28dddbe44805f4918"),
    ("linux", "x86_64"): ("ubuntu-x64", "27e18f608c7cd1e2f86d70f91bde4299e795113c0065322f6339cbe5135e79c2"),
    ("linux", "aarch64"): ("ubuntu-arm64", "69a37e8fc18bf06214052d39fd83605846273ba08d7a6d88651e95788ca926db"),
}


def llama_dir() -> pathlib.Path:
    return paths.assistant_dir() / "runtimes" / f"llama-{LLAMA_RELEASE}"


def llama_build_for_host() -> tuple[str, str] | None:
    arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
    if sys.platform == "linux" and arch == "arm64":
        arch = "aarch64"
    return LLAMA_BUILDS.get((sys.platform, arch))


def install_llama_release(progress: Progress | None = None, *, release: str | None = None,
                          builds: dict | None = None, build: tuple[str, str] | None = None,
                          target: pathlib.Path | None = None) -> pathlib.Path:
    """Download and checksum-verify llama.cpp's pinned release archive into assistant_dir()/runtimes."""
    import hashlib
    import tarfile

    rel = release or LLAMA_RELEASE
    if build is None:
        table = builds if builds is not None else LLAMA_BUILDS
        arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
        if sys.platform == "linux" and arch == "arm64":
            arch = "aarch64"
        build = table.get((sys.platform, arch))
    if build is None:
        raise RuntimeError("llama.cpp publishes no build for this platform")
    asset, sha256 = build
    name = f"llama-{rel}-bin-{asset}.tar.gz"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{rel}/{name}"
    dest = target or (paths.assistant_dir() / "runtimes" / f"llama-{rel}")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / name
    if progress:
        progress(f"downloading {name}")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as response:
        response.raise_for_status()
        with open(archive, "wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                digest.update(chunk)
                handle.write(chunk)
    if digest.hexdigest() != sha256:
        archive.unlink()
        raise RuntimeError(f"{name} does not match its published checksum")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")     # refuses absolute paths, links out, devices
    archive.unlink()
    return dest


class LlamaCppRuntime(Runtime):
    name = "llama.cpp"

    PORT_OFFSET = 0

    def __init__(self, config=None, port: int | None = None, hub: Hub | None = None, hw: Hardware | None = None):
        super().__init__(config)
        self.port = port = port or server_port(config, self.PORT_OFFSET)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.hub = hub
        self.hw = hw
        self.root = models_dir(config)
        self.state_path = self.root / "llama.cpp.json"

    def binary(self) -> str | None:
        # Prefer the pinned managed build over PATH. Homebrew's llama-server is often older than
        # LLAMA_RELEASE and cannot load LFM3 (unknown architecture: 'lfm3').
        managed = list((paths.assistant_dir() / "runtimes").glob("llama-b*/**/llama-server"))
        pinned = [path for path in managed if f"llama-{LLAMA_RELEASE}" in path.parts]
        if pinned:
            return str(sorted(pinned)[-1])
        if managed:
            return str(sorted(managed)[-1])
        return shutil.which("llama-server")

    def available(self) -> bool:
        return self.binary() is not None

    def can_install(self) -> bool:
        """Whether llama.cpp publishes a build for this platform (pull fetches it when no llama-server is here)."""
        return llama_build_for_host() is not None

    def running(self) -> bool:
        return _ok(f"http://127.0.0.1:{self.port}/health")

    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        if not self.binary():
            self.install(progress)
        art = entry["artifacts"]["gguf"]
        hub = self.hub or Hub()
        if art.get("url"):
            # A file named by its address and checksum, not by a Hub repository: Bonsai 2 27B in the packing
            # llama.cpp itself reads (Q2_0) is published only in Ollama's registry, which serves it as a plain
            # download. Its vision projector is the maker's, from the Hub.
            folder = self.root / art["folder"]
            record = {"repo": art["folder"], "commit": art["sha256"][:12], "mmproj": None, "model": str(hub.fetch(
                art["url"], folder / art["file"], art["sha256"], art.get("bytes"), progress, art.get("streams")))}
            if art.get("mmproj") and catalog.needs_mmproj(entry, self.name):
                record["mmproj"] = str(hub.download(art["mmproj_repo"], art["mmproj"], folder / art["mmproj"],
                                                    sha256=art.get("mmproj_sha256"), progress=progress))
            state = self._state()
            state.setdefault("models", {})[entry["id"]] = record
            self._save(state)
            return
        if art.get("repo") and art.get("file") and art.get("sha256"):
            # Pinned direct HF file + SHA-256 (e.g., Ternary Bonsai 2.5 Q2_0): skips manifest lookup.
            folder = self.root / art["repo"].replace("/", "--")
            record = {"repo": art["repo"], "commit": "main", "mmproj": None, "model": str(hub.download(
                art["repo"], art["file"], folder / art["file"], sha256=art["sha256"],
                size=art.get("bytes"), progress=progress))}
            if art.get("mmproj") and catalog.needs_mmproj(entry, self.name):
                record["mmproj"] = str(hub.download(art.get("mmproj_repo") or art["repo"], art["mmproj"],
                                                    folder / art["mmproj"], sha256=art.get("mmproj_sha256"),
                                                    progress=progress))
            state = self._state()
            state.setdefault("models", {})[entry["id"]] = record
            self._save(state)
            return
        commit = hub.model_info(art["repo"]).get("sha") or "main"
        files = hub.gguf_files(art["repo"], art["quant"], vision=catalog.needs_mmproj(entry, self.name))
        folder = self.root / art["repo"].replace("/", "--")
        record = {"repo": art["repo"], "commit": commit, "model": None, "mmproj": None}
        for f in files:
            dest = hub.download(art["repo"], f["file"], folder / f["file"], revision=commit, sha256=f["sha256"],
                                size=f["size"], progress=progress)
            if f["kind"] == "mmproj":
                record["mmproj"] = str(dest)
            elif record["model"] is None:
                record["model"] = str(dest)  # first shard; llama.cpp finds the rest beside it
        state = self._state()
        state.setdefault("models", {})[entry["id"]] = record
        self._save(state)

    def installed(self, entry: dict) -> bool:
        record = self._state().get("models", {}).get(entry["id"]) or {}
        return bool(record.get("model")) and os.path.exists(record["model"])

    def remove(self, entry: dict) -> None:
        state = self._state()
        record = state.get("models", {}).pop(entry["id"], None) or {}
        # Builds of one model share files (both Bonsai 2 builds use one vision projector): keep those in use.
        in_use = {other.get(key) for other in state.get("models", {}).values() for key in ("model", "mmproj")}
        root = self.root.resolve()
        for key in ("model", "mmproj"):
            path = pathlib.Path(record[key]).resolve() if record.get(key) and record[key] not in in_use else None
            if path and root in path.parents:          # only files CodeRunner downloaded
                for part in sorted(path.parent.glob(path.name.split("-00001-of-")[0] + "*")) \
                        if "-00001-of-" in path.name else [path]:
                    part.unlink(missing_ok=True)
        # An interrupted download of this build, kept for a resume that will not come.
        art = (entry.get("artifacts") or {}).get(catalog.ARTIFACT.get(self.name, "gguf")) or {}
        name = art.get("file") or art.get("quant")
        if (art.get("repo") or art.get("folder")) and name:
            folder = self.root / (art.get("folder") or art["repo"].replace("/", "--"))
            for part in [*folder.glob(f"*{name}*.part"), *folder.glob(f"*{name}*.part.json")]:
                part.unlink(missing_ok=True)
        if (state.get("active") or {}).get("id") == entry["id"]:
            state.pop("active")
        self._save(state)

    def prepare(self, entry: dict, ctx: int, offload: bool = False, *, kv_q8: bool = False,
                no_mmap: bool = False) -> str:
        state = self._state()
        record = state.get("models", {}).get(entry["id"])
        if not record:
            raise RuntimeError(f"{entry['id']} has not been downloaded for llama.cpp")
        art = (entry.get("artifacts") or {}).get(catalog.ARTIFACT.get(self.name, "gguf")) or {}
        args = list(art.get("server_args") or [])
        if (kv_q8 or ctx >= 65536) and "-ctk" not in args:
            args += ["-ctk", "q8_0", "-ctv", "q8_0"]
        if no_mmap and "--no-mmap" not in args and "--load-mode" not in args:
            args += ["--no-mmap"]
        active = {"id": entry["id"], "alias": self.model_id(entry), "ctx": ctx,
                  "model": record["model"], "mmproj": record.get("mmproj"), "offload": offload,
                  **({"args": args} if args else {})}
        if state.get("active") != active:
            self.stop()  # restarts with the new model on the next start()
        state["active"] = active
        self._save(state)
        return active["alias"]

    def install(self, progress: Progress | None = None) -> None:
        """llama.cpp's official build (pinned, checksum-checked)."""
        install_llama_release(progress)
        if not self.binary():
            raise RuntimeError("llama.cpp's build has no llama-server")

    def command(self, active: dict) -> list[str]:
        cmd = [self.binary() or "llama-server", "-m", active["model"]]
        if active.get("mmproj"):
            cmd += ["--mmproj", active["mmproj"]]
        # --cache-ram 0 avoids a prompt-cache bug that leaks state across conversations. Two slots over one
        # shared cache: the step judge gets its own slot instead of evicting the agent's cached conversation.
        # Measured with Qwen3.5-9B on an M3 Pro: with one slot every step after a judge call re-read the
        # whole prompt (6 s growing to 19 s per step); with two, steps read only what was new (3.6 s).
        # A slot for each task that may run at once (hardware-tiered or agent.parallel_tasks) and one for the judge:
        # each task's requests name its slot, so its conversation stays read while the other has the model.
        # The first slot's reading of the instructions is copied to the others through a file (LLM.copy_slot).
        from . import hardware
        slots = hardware.parallel_tasks(self.config, self.hw) + 1
        saved = paths.assistant_dir() / "run" / "slots"
        saved.mkdir(parents=True, exist_ok=True)
        for old in saved.glob("*.bin"):     # what another model read is of no use to this one
            if old.name != f"instructions-{active['alias']}.bin":
                old.unlink(missing_ok=True)
        cmd += ["--jinja", "-c", str(active["ctx"]), "--host", "127.0.0.1", "--port", str(self.port),
                "--cache-ram", "0", "-np", str(slots), "--kv-unified", "--slot-save-path", str(saved),
                "--alias", active["alias"]]
        idle = idle_seconds(self.config)
        if idle:
            # The server frees the model's memory when idle and loads it again on the next request: a 27B
            # held 10 GB of an 18 GB Mac (0.5 GB asleep); waking took 4 s plus re-reading the instructions.
            cmd += ["--sleep-idle-seconds", str(idle)]
        if active.get("offload"):
            cmd += ["--cpu-moe"]          # experts in system RAM, the rest on the GPU
        cmd += active.get("args") or []   # what this build needs, from the catalog (server_args)
        peers = [node for node in (self.config.get("models.cluster") or []) if node.get("host")] if self.config else []
        if peers:
            # Other machines running llama.cpp's rpc-server (models.cluster); layers are spread across them.
            cmd += ["--rpc", ",".join(f"{node['host']}:{int(node.get('port') or 50052)}" for node in peers)]
        return cmd

    def start(self, timeout: float = 600) -> None:
        if self.running():
            return
        if not self.available():
            raise RuntimeError(f"llama-server is not installed. {install_hint(runtime='llama.cpp')}")
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No llama.cpp model is configured; run setup first")
        proc = _spawn("llama-server", self.command(active))
        _wait_ready(self.running, proc, timeout, "llama-server", str(paths.assistant_dir() / "logs" / "llama-server.log"))

    def stop(self) -> None:
        _terminate("llama-server", "llama-server")

    def vision(self, entry: dict, model: str | None = None) -> bool:
        active = self._state().get("active") or {}
        return catalog.has_vision(entry, self.name) and bool(active.get("mmproj")) and active.get("id") == entry["id"]


class PrismRuntime(LlamaCppRuntime):
    """Prism ML's llama.cpp build, needed for ternary Bonsai models (PQ2_0 / PTQ1_0 packings).

    The build is pinned to a release and checked against its published SHA-256 before it is unpacked.
    """

    name = "prism"
    RELEASE = "prism-b12232-f702476"
    BUILDS = {  # (os, arch, gpu) -> (asset, sha256); gpu "" matches any
        ("darwin", "arm64", ""): ("macos-arm64",
                                  "d39332891793dfcaf690ed86a783bd3eda30c5a855c453c179a197236243e19d"),
        ("linux", "x86_64", "nvidia"): ("linux-cuda-12.4-x64",
                                        "b5750a2e4eb7369093d5ec7a1e82bf3e3772fefd5e998a46e8f245d4fb3490ed"),
        ("linux", "x86_64", ""): ("ubuntu-vulkan-x64",
                                  "c65f0134336b8c60cf18998cfd5ac0dd863340e756bcf9ed255c3e463d64f263"),
        ("linux", "aarch64", ""): ("ubuntu-arm64",
                                   "fd96b0a238c8efe51abfe1f27d025d83db80583761efa860c2ac3faab5f0b3de"),
    }

    PORT_OFFSET = 1

    def __init__(self, config=None, port: int | None = None, hub: Hub | None = None, hw: Hardware | None = None):
        super().__init__(config, port, hub)
        self.state_path = self.root / "prism.json"
        self.bin_dir = paths.assistant_dir() / "runtimes" / self.RELEASE
        self.hw = hw

    @classmethod
    def build_for(cls, hw: Hardware | None) -> tuple[str, str] | None:
        system = hw.os if hw else platform.system().lower()
        arch = (hw.arch if hw else platform.machine()).lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
        gpu = hw.gpu if hw else ""
        if (system, arch) == ("linux", "aarch64") and gpu == "nvidia":
            return None     # the arm64 build has no CUDA: on a DGX Spark it would run on the CPU
        return cls.BUILDS.get((system, arch, gpu)) or cls.BUILDS.get((system, arch, ""))

    def binary(self) -> str | None:
        found = sorted(self.bin_dir.glob("**/llama-server")) if self.bin_dir.exists() else []
        return str(found[0]) if found and os.access(found[0], os.X_OK) else None

    def available(self) -> bool:
        return self.binary() is not None or self.build_for(self.hw) is not None    # can be installed

    def install(self, progress: Progress | None = None) -> str:
        build = self.build_for(self.hw)
        if build is None:
            raise RuntimeError("Prism ML publishes no llama.cpp build for this platform")
        import hashlib
        import tarfile
        asset, sha256 = build
        name = f"llama-{self.RELEASE}-bin-{asset}.tar.gz"
        url = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{self.RELEASE}/{name}"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        archive = self.bin_dir / name
        if progress:
            progress(f"downloading {name}")
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as response:
            response.raise_for_status()
            with open(archive, "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sha256:
            archive.unlink()
            raise RuntimeError(f"{name} does not match its published checksum")
        with tarfile.open(archive) as tar:
            tar.extractall(self.bin_dir, filter="data")     # refuses absolute paths, links out, devices
        archive.unlink()
        binary = self.binary()
        if not binary:
            raise RuntimeError(f"{name} has no llama-server")
        return binary

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        if not self.binary():
            self.install(progress)
        art = entry["artifacts"]["prism"]
        hub = self.hub or Hub()
        folder = self.root / art["repo"].replace("/", "--")
        record = {"repo": art["repo"], "model": None, "mmproj": None}
        record["model"] = str(hub.download(art["repo"], art["file"], folder / art["file"], sha256=art["sha256"],
                                           progress=progress))
        if art.get("mmproj"):
            record["mmproj"] = str(hub.download(art["repo"], art["mmproj"], folder / art["mmproj"],
                                                sha256=art.get("mmproj_sha256"), progress=progress))
        state = self._state()
        state.setdefault("models", {})[entry["id"]] = record
        self._save(state)

    def command(self, active: dict) -> list[str]:
        # Medium reasoning effort: the maker's advice for API use (the default thinks at length and can
        # run out of output budget). The context cache is kept at 8 bits, as the memory estimate assumes.
        # Weights are read into the server's own memory ("none"): memory-mapped, macOS evicted them under
        # pressure and re-read them from disk (0.27 tokens/s on an 18 GB Mac); locked ("mlock"), they could
        # not be reclaimed at all, and with a second model loaded the Mac stopped responding until the
        # watchdog restarted it. models.lock_weights brings the lock back for machines with room to spare.
        # The judge's own slot matters most here: a re-read conversation goes at about 70 tokens/s.
        mode = "mlock" if self.config is not None and self.config.get("models.lock_weights", False) else "none"
        return super().command(active) + ["--chat-template-kwargs", json.dumps({"reasoning_effort": "medium"}),
                                          "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "--load-mode", mode]

    def start(self, timeout: float = 600) -> None:
        if self.running():
            return
        if not self.binary():
            self.install()
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No Bonsai model is configured; run setup first")
        proc = _spawn("prism-server", self.command(active))
        _wait_ready(self.running, proc, timeout, "prism llama-server", str(paths.assistant_dir() / "logs" / "prism-server.log"))

    def stop(self) -> None:
        _terminate("prism-server", "llama-server")


class DriverRuntime(LlamaCppRuntime):
    """A second llama.cpp server for the browser driver: a model that acts on web pages from screenshots
    (Fara1.5). It has its own port, state and process, so it can run beside the assistant's model where
    memory allows, or take turns with it where it does not (models.driver.exclusive)."""

    name = "driver"

    PORT_OFFSET = 2

    def __init__(self, config=None, port: int | None = None, hub: Hub | None = None):
        super().__init__(config, port, hub)
        self.state_path = self.root / "driver.json"

    def binary(self) -> str | None:
        configured = (self.config.get("models.driver.binary") if self.config else "") or ""
        if configured and os.access(configured, os.X_OK):
            return configured
        return super().binary()

    def command(self, active: dict) -> list[str]:
        cmd = [self.binary() or "llama-server", "-m", active["model"]]
        if active.get("mmproj"):
            cmd += ["--mmproj", active["mmproj"]]
        idle = idle_seconds(self.config)
        # Two slots over one shared cache, as for the assistant's model: where the driver runs alone the step
        # judge asks it too, and with one slot each judge call evicted the driver's conversation. Measured with
        # Fara1.5-4B on an M3 Pro: every step re-read its whole prompt, 10,000 tokens and 40 s.
        return cmd + ["--jinja", "-c", str(active["ctx"]), "--host", "127.0.0.1", "--port", str(self.port),
                      "--cache-ram", "0", "-np", "2", "--kv-unified", "-ngl", "99", "--alias", active["alias"],
                      # Qwen-VL grounding needs at least 1,024 image tokens (llama.cpp's warning, H's settings):
                      # a 1440 x 900 page is already more; a 1280 x 800 one is brought up to it.
                      "--image-min-tokens", "1024"] + (
                          ["--sleep-idle-seconds", str(idle)] if idle else [])

    def start(self, timeout: float = 300) -> None:
        if self.running():
            return
        if not self.available():
            raise RuntimeError("The browser driver needs llama.cpp's llama-server: run coderunner models driver "
                               "again, which installs it")
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No browser driver is set up; run coderunner models driver")
        proc = _spawn("llama-driver", self.command(active))
        _wait_ready(self.running, proc, timeout, "the browser driver",
                    str(paths.assistant_dir() / "logs" / "llama-driver.log"))

    def stop(self) -> None:
        _terminate("llama-driver", "llama-server")


class JudgeRuntime(LlamaCppRuntime):
    """A dedicated lightweight llama.cpp server for safety commit judging and injection screening on Apple
    silicon Macs (24 GB+) without requiring Ollama, keeping the main chat model's KV cache untouched."""

    name = "judge"
    PORT_OFFSET = 4

    def __init__(self, config=None, port: int | None = None, hub: Hub | None = None):
        super().__init__(config, port, hub)
        self.state_path = self.root / "judge.json"

    def command(self, active: dict) -> list[str]:
        cmd = [self.binary() or "llama-server", "-m", active["model"]]
        idle = idle_seconds(self.config)
        return cmd + ["--jinja", "-c", str(min(int(active.get("ctx") or 4096), 8192)), "--host", "127.0.0.1",
                      "--port", str(self.port), "--cache-ram", "0", "-np", "1", "-ngl", "99",
                      "--alias", active["alias"]] + (["--sleep-idle-seconds", str(idle)] if idle else [])

    def start(self, timeout: float = 180) -> None:
        if self.running():
            return
        if not self.available():
            self.install()
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No safety judge model is prepared")
        proc = _spawn("llama-judge", self.command(active))
        _wait_ready(self.running, proc, timeout, "the safety judge",
                    str(paths.assistant_dir() / "logs" / "llama-judge.log"))

    def stop(self) -> None:
        _terminate("llama-judge", "llama-server")


class VoiceRuntime(LlamaCppRuntime):
    """A loopback llama.cpp server for LFM3-Omni: spoken lines while tasks work and spoken replies (POST /tts).

    Keeps its own port and state so the planner's chat model and KV cache stay untouched. On tight Macs it
    takes turns with chat (voice.omni_exclusive); otherwise it idle-unloads beside chat.
    """

    name = "voice"
    PORT_OFFSET = 6
    VOICE_CTX = 8192

    def __init__(self, config=None, port: int | None = None, hub: Hub | None = None, hw: Hardware | None = None):
        super().__init__(config, port, hub, hw)
        self.state_path = self.root / "voice.json"

    def binary(self) -> str | None:
        """LFM3 needs the pinned release; never fall back to an older PATH binary."""
        pinned = sorted(llama_dir().glob("**/llama-server"))
        return str(pinned[-1]) if pinned else None

    def command(self, active: dict) -> list[str]:
        cmd = [self.binary() or "llama-server", "-m", active["model"]]
        if active.get("mmproj"):
            cmd += ["--mmproj", active["mmproj"]]
        idle = idle_seconds(self.config)
        args = list(active.get("args") or [])
        if "--reasoning" not in args:
            args += ["--reasoning", "off"]      # spoken lines come at once, never after a hidden think
        return cmd + ["--jinja", "-c", str(active.get("ctx") or self.VOICE_CTX),
                      "--host", "127.0.0.1", "--port", str(self.port), "--cache-ram", "0", "-np", "1",
                      "-ngl", "99", "--alias", active["alias"]] + (
                          ["--sleep-idle-seconds", str(idle)] if idle else []) + args

    def start(self, timeout: float = 300) -> None:
        if self.running():
            return
        if not self.available():
            self.install()
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No voice model is set up; run coderunner voice omni")
        if not active.get("mmproj"):
            raise RuntimeError("The voice model is missing its audio projector; run coderunner voice omni again")
        proc = _spawn("llama-voice", self.command(active))
        _wait_ready(self.running, proc, timeout, "the voice model",
                    str(paths.assistant_dir() / "logs" / "llama-voice.log"))

    def stop(self) -> None:
        _terminate("llama-voice", "llama-server")

    def ready(self) -> bool:
        """Weights prepared and a process can be started (or is already up)."""
        active = self._state().get("active") or {}
        return bool(active.get("model") and active.get("mmproj") and os.path.exists(active["model"])
                    and os.path.exists(active["mmproj"]))

    def ensure(self, timeout: float = 300) -> None:
        """Lazy-start the server when weights are ready."""
        if not self.ready():
            raise RuntimeError("LFM3-Omni is not set up; run coderunner voice omni")
        self.start(timeout)

    def alias(self) -> str:
        return (self._state().get("active") or {}).get("alias") or "lfm3-omni-2b"

    def tts_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/tts"


class OpenAIRuntime(Runtime):
    """An endpoint the owner runs themselves (LM Studio, vLLM, a remote server...)."""

    name = "openai"
    default_url = ""

    def __init__(self, config=None, base_url: str | None = None):
        super().__init__(config)
        get = config.get if config is not None else (lambda key, default=None: default)
        self.base_url = (base_url or get("models.base_url") or self.default_url).rstrip("/")
        self.api_key = os.environ.get("CODERUNNER_MODEL_API_KEY") or get("models.api_key") or ""
        self._chat = get("models.chat") or ""

    def available(self) -> bool:
        return bool(self.base_url)

    def running(self) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return bool(self.base_url) and _ok(f"{self.base_url}/models", headers, timeout=5, trust_env=True)

    def model_id(self, entry: dict) -> str:
        return self._chat or entry["id"]

    def vision(self, entry: dict, model: str | None = None) -> bool:
        return catalog.has_vision(entry)


class MLXRuntime(OpenAIRuntime):
    """Managed Apple Silicon MLX runtime (`mlx-lm` / `mlx-vlm` under assistant_dir()/runtimes/mlx),
    falling back seamlessly to an already-running owner MLX server on base_url."""

    name = "mlx"
    PORT_OFFSET = 5
    PACKAGES = ("mlx-lm", "huggingface_hub")
    default_url = "http://127.0.0.1:8080/v1"

    def __init__(self, config=None, base_url: str | None = None, port: int | None = None, hub: Hub | None = None):
        super().__init__(config, base_url=base_url)
        self.hub = hub
        self.port = port or server_port(config, self.PORT_OFFSET)
        self.root = models_dir(config) / "mlx"
        self.env = paths.assistant_dir() / "runtimes" / "mlx"
        self.state_path = self.root / "mlx.json"
        self._managed_url = f"http://127.0.0.1:{self.port}/v1"
        explicit = (base_url or (config.get("models.base_url") if config is not None else "") or "").strip().rstrip("/")
        self._external = bool(explicit and explicit != self._managed_url)
        if not self._external:
            self.base_url = self._managed_url

    @staticmethod
    def supported() -> bool:
        return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "arm64e")

    def managed(self) -> bool:
        """True when CodeRunner owns the local mlx-lm server; False for explicitly owner-managed endpoints."""
        return not self._external

    def python(self) -> str | None:
        found, ready = self.env / "bin" / "python", self.env / "installed.txt"
        try:
            return str(found) if found.exists() and ready.read_text().strip() == "\n".join(self.PACKAGES) else None
        except OSError:
            return None

    def can_install(self) -> bool:
        return self.managed() and self.supported()

    def available(self) -> bool:
        return (self.python() is not None or self.running()) if self.managed() else self.running()

    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def install(self, progress: Progress | None = None) -> None:
        if not self.supported():
            raise RuntimeError("MLX models require Apple silicon")
        if progress:
            progress("installing MLX runtime packages (mlx-lm)")
        shutil.rmtree(self.env, ignore_errors=True)
        for cmd in ([sys.executable, "-m", "venv", str(self.env)],
                    [str(self.env / "bin" / "python"), "-m", "pip", "install", "-q",
                     "--disable-pip-version-check", "--retries", "5", "--timeout", "60", *self.PACKAGES]):
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode:
                shutil.rmtree(self.env, ignore_errors=True)
                raise RuntimeError(f"Installing MLX failed: {(done.stderr or done.stdout)[-400:]}")
        (self.env / "installed.txt").write_text("\n".join(self.PACKAGES))

    def model_folder(self, entry: dict) -> pathlib.Path:
        art = (entry.get("artifacts") or {}).get("mlx") or {}
        return self.root / str(art.get("repo") or entry["id"]).replace("/", "--")

    def installed(self, entry: dict) -> bool:
        rec = self._state().get("models", {}).get(entry["id"]) or {}
        return bool(rec.get("path")) and os.path.isdir(rec["path"])

    def pull(self, entry: dict, progress: Progress | None = None) -> None:
        if not self.python():
            self.install(progress)
        art = (entry.get("artifacts") or {}).get("mlx") or {}
        repo = art.get("repo")
        if not repo:
            raise RuntimeError(f"{entry['id']} has no MLX artifact")
        folder = self.model_folder(entry)
        folder.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"downloading {repo} for MLX")
        cmd = [self.python(), "-c",
               "from huggingface_hub import snapshot_download; import sys; "
               "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])",
               repo, str(folder)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"Downloading MLX model {repo} failed: {(done.stderr or done.stdout)[-400:]}")
        state = self._state()
        state.setdefault("models", {})[entry["id"]] = {"repo": repo, "path": str(folder)}
        self._save(state)

    def remove(self, entry: dict) -> None:
        state = self._state()
        rec = state.get("models", {}).pop(entry["id"], None) or {}
        if rec.get("path"):
            folder = pathlib.Path(rec["path"]).resolve()
            if self.root.resolve() in folder.parents:
                shutil.rmtree(folder, ignore_errors=True)
        if (state.get("active") or {}).get("id") == entry["id"]:
            state.pop("active", None)
        self._save(state)

    def prepare(self, entry: dict, ctx: int, offload: bool = False) -> str:
        state = self._state()
        rec = state.get("models", {}).get(entry["id"])
        if not rec:
            raise RuntimeError(f"{entry['id']} has not been downloaded for MLX")
        active = {"id": entry["id"], "alias": self.model_id(entry), "path": rec["path"], "ctx": ctx}
        if state.get("active") != active:
            self.stop()
        state["active"] = active
        self._save(state)
        return active["alias"]

    def command(self, active: dict) -> list[str]:
        return [self.python() or sys.executable, "-m", "mlx_lm.server",
                "--model", active["path"], "--host", "127.0.0.1", "--port", str(self.port)]

    def start(self, timeout: float = 300) -> None:
        if self.running():
            return
        if not self.python():
            raise RuntimeError("MLX runtime is not installed")
        active = self._state().get("active")
        if not active:
            raise RuntimeError("No MLX model is prepared")
        proc = _spawn("mlx-server", self.command(active))
        _wait_ready(self.running, proc, timeout, "mlx-lm server", str(paths.assistant_dir() / "logs" / "mlx-server.log"))

    def stop(self) -> None:
        _terminate("mlx-server", "python")

    def model_id(self, entry: dict) -> str:
        return f"coderunner-{entry['id']}" if self.managed() else super().model_id(entry)


RUNTIMES: dict[str, type[Runtime]] = {
    "ollama": OllamaRuntime, "llama.cpp": LlamaCppRuntime, "openai": OpenAIRuntime, "mlx": MLXRuntime,
    "prism": PrismRuntime,
}


def choose_runtime(config, hw: Hardware | None = None, name: str | None = None) -> Runtime | None:
    """The configured runtime, else Ollama, else llama.cpp; None when nothing is installed."""
    name = name or (config.get("models.runtime") if config is not None else None) or "auto"
    if name != "auto":
        if name not in RUNTIMES:
            raise ValueError(f"Unknown runtime {name!r}; choose one of {', '.join(RUNTIMES)} or auto")
        cls = RUNTIMES[name]
        return cls(config, hw=hw) if cls in (LlamaCppRuntime, PrismRuntime) else cls(config)
    for cls in (OllamaRuntime, LlamaCppRuntime):
        runtime = cls(config, hw=hw) if cls is LlamaCppRuntime else cls(config)
        if runtime.available():
            return runtime
    return None


def install_hint(hw: Hardware | None = None, runtime: str | None = None) -> str:
    system = hw.os if hw else platform.system().lower()
    if runtime == "llama.cpp":
        if system == "darwin":
            return "Install llama.cpp with `brew install llama.cpp`."
        return "Install llama.cpp from https://github.com/ggml-org/llama.cpp/releases (or your package manager)."
    if runtime == "mlx":
        return "Run `pip install mlx-lm` and start `mlx_lm.server`, then set models.base_url."
    if system == "darwin":
        return ("Install Ollama with `brew install ollama` (or the app from https://ollama.com/download), "
                "or llama.cpp with `brew install llama.cpp`.")
    if system == "linux":
        return ("Install Ollama with the official script after reviewing it: "
                "curl -fsSL https://ollama.com/install.sh | sh  "
                "(or llama.cpp from https://github.com/ggml-org/llama.cpp/releases).")
    return "Install Ollama from https://ollama.com/download."
