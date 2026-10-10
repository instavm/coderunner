"""Local model runtimes behind one interface.

Every runtime serves an OpenAI-compatible API on 127.0.0.1. Servers that
CodeRunner starts log to assistant_dir()/logs and record their pid in
assistant_dir()/run, and stop() only ever signals a process it started.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import time
from typing import Callable

import httpx

from .. import paths
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

    def vision(self, entry: dict, model: str | None = None) -> bool:
        return catalog.has_vision(entry, self.name)

    def describe(self) -> dict:
        return {"name": self.name, "base_url": self.base_url, "available": self.available(),
                "running": self.running()}


class OllamaRuntime(Runtime):
    name = "ollama"
    APP_BINARY = "/Applications/Ollama.app/Contents/Resources/ollama"

    def __init__(self, config=None, host: str = "127.0.0.1:11434"):
        super().__init__(config)
        self.host = host
        self.api = f"http://{host}"
        self.base_url = f"{self.api}/v1"

    def binary(self) -> str | None:
        return shutil.which("ollama") or (self.APP_BINARY if os.path.exists(self.APP_BINARY) else None)

    def available(self) -> bool:
        return self.binary() is not None

    def running(self) -> bool:
        return _ok(f"{self.api}/api/version")

    def _env(self) -> dict:
        return {**os.environ, "OLLAMA_HOST": self.host}

    def start(self, timeout: float = 30) -> None:
        if self.running():
            return
        if not self.available():
            raise RuntimeError(f"Ollama is not installed. {install_hint(runtime='ollama')}")
        proc = _spawn("ollama", [self.binary(), "serve"], self._env())
        _wait_ready(self.running, proc, timeout, "ollama serve", str(paths.assistant_dir() / "logs" / "ollama.log"))

    def stop(self) -> None:
        _terminate("ollama", "ollama")

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


class LlamaCppRuntime(Runtime):
    name = "llama.cpp"

    def __init__(self, config=None, port: int = 8081, hub: Hub | None = None):
        super().__init__(config)
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.hub = hub
        self.root = paths.assistant_dir() / "models"
        self.state_path = self.root / "llama.cpp.json"

    def binary(self) -> str | None:
        return shutil.which("llama-server")

    def available(self) -> bool:
        return self.binary() is not None

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
        art = entry["artifacts"]["gguf"]
        hub = self.hub or Hub()
        commit = hub.model_info(art["repo"]).get("sha") or "main"
        files = hub.gguf_files(art["repo"], art["quant"], vision=catalog.has_vision(entry, self.name))
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

    def prepare(self, entry: dict, ctx: int) -> str:
        state = self._state()
        record = state.get("models", {}).get(entry["id"])
        if not record:
            raise RuntimeError(f"{entry['id']} has not been downloaded for llama.cpp")
        active = {"id": entry["id"], "alias": self.model_id(entry), "ctx": ctx,
                  "model": record["model"], "mmproj": record.get("mmproj")}
        if state.get("active") != active:
            self.stop()  # restarts with the new model on the next start()
        state["active"] = active
        self._save(state)
        return active["alias"]

    def command(self, active: dict) -> list[str]:
        cmd = [self.binary() or "llama-server", "-m", active["model"]]
        if active.get("mmproj"):
            cmd += ["--mmproj", active["mmproj"]]
        # --cache-ram 0 avoids a prompt-cache bug that leaks state across conversations.
        return cmd + ["--jinja", "-c", str(active["ctx"]), "--host", "127.0.0.1", "--port", str(self.port),
                      "--cache-ram", "0", "-np", "1", "--alias", active["alias"]]

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
    """mlx_lm.server / mlx_vlm.server started by the owner; treated as an external endpoint."""

    name = "mlx"
    default_url = "http://127.0.0.1:8080/v1"


RUNTIMES: dict[str, type[Runtime]] = {
    "ollama": OllamaRuntime, "llama.cpp": LlamaCppRuntime, "openai": OpenAIRuntime, "mlx": MLXRuntime,
}


def choose_runtime(config, hw: Hardware | None = None, name: str | None = None) -> Runtime | None:
    """The configured runtime, else Ollama, else llama.cpp; None when nothing is installed."""
    name = name or (config.get("models.runtime") if config is not None else None) or "auto"
    if name != "auto":
        if name not in RUNTIMES:
            raise ValueError(f"Unknown runtime {name!r}; choose one of {', '.join(RUNTIMES)} or auto")
        return RUNTIMES[name](config)
    for cls in (OllamaRuntime, LlamaCppRuntime):
        runtime = cls(config)
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
