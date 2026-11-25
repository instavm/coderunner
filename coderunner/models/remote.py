"""Remote mode: the assistant's model on Anthropic's API, with the owner's own key.

A setting the owner turns on knowingly. It is not private: each request sends the conversation, the text of
the pages the assistant reads and the contents of files it opens to Anthropic. In exchange a task takes
seconds a step on any computer, without a model download. While it is on, Claude plans, sees pictures, judges
browser steps and screens untrusted text; file search and voice stay on this computer (their models are
local), and the browser driver is not used.

The local model's settings are left as they are, so switching back is one action. The key is kept in a file
only the owner can read, never in the config, the logs or the audit trail.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

from ..core import paths

PROVIDER = "anthropic"
MODELS = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
          "claude-opus-5-5": "Claude Opus 5.5"}
DEFAULT = "claude-haiku-5-5"
WARNING = "Your requests, the pages it reads and the files it opens are sent to Anthropic. This is not private."
KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")


def key_path() -> pathlib.Path:
    return paths.assistant_dir() / "anthropic-key"


def key() -> str:
    try:
        return key_path().read_text().strip()
    except OSError:
        return ""


def keep_key(text: str) -> None:
    """Keep the key found in this text (a pasted key, or the contents of a file that holds one) where only the
    owner can read it. ValueError when the text holds no Anthropic key; the text is never repeated."""
    found = KEY.search(text or "")
    if not found:
        raise ValueError("That is not an Anthropic API key (they start with sk-ant-).")
    path = key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(found.group(0) + "\n")
    os.replace(tmp, path)


def keep_key_file(path: str) -> None:
    source = pathlib.Path(os.path.expanduser(path))
    try:
        if source.stat().st_size > 4096:
            raise ValueError("That file is too large to be a key.")
        text = source.read_text(errors="replace")
    except OSError as exc:
        raise ValueError(f"That file could not be read: {exc.strerror or exc}") from exc
    keep_key(text)


def forget_key() -> None:
    key_path().unlink(missing_ok=True)


# The owner's own model server (an OpenAI-compatible API, such as llama.cpp's on another computer of theirs):
# chosen like Anthropic, so a Mac short of memory loads no model of its own. Its key, when it has one, is kept
# like the Anthropic key.
SERVER = "openai"


def server_key_path() -> pathlib.Path:
    return paths.assistant_dir() / "server-key"


def server_key() -> str:
    try:
        return server_key_path().read_text().strip()
    except OSError:
        return ""


def server_warning(host: str) -> str:
    return f"Your requests, the pages it reads and the files it opens are sent to your model server at {host}."


def use_server(config, base_url: str, model: str, api_key: str = "") -> dict:
    """Remote mode on the owner's own server: base_url is its OpenAI-compatible API (ending in /v1)."""
    from urllib.parse import urlsplit
    base_url = (base_url or "").strip().rstrip("/")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Give the server's address, such as http://100.64.0.5:8080/v1.")
    if not (model or "").strip():
        raise ValueError("Give the name of the model the server serves.")
    path = server_key_path()
    if api_key.strip():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(api_key.strip() + "\n")
        os.replace(tmp, path)
    else:
        path.unlink(missing_ok=True)
    config.set("models.remote", {"provider": SERVER, "base_url": base_url, "model": model.strip()})
    config.save()
    return active(config)


def active(config) -> dict | None:
    """{"provider", "model", "name"} while remote mode is on and its key is here, else None. On the owner's own
    server also "base_url" and "host"."""
    chosen = (config.get("models.remote") if config is not None else None) or {}
    if chosen.get("provider") == SERVER and chosen.get("base_url") and chosen.get("model"):
        from urllib.parse import urlsplit
        return {"provider": SERVER, "model": chosen["model"], "name": chosen["model"], "base_url": chosen["base_url"],
                "host": urlsplit(chosen["base_url"]).netloc, "effort": "medium",
                "keep_slots": chosen.get("keep_slots") is not False}
    if chosen.get("provider") != PROVIDER or not key():
        return None
    model = chosen.get("model") or DEFAULT
    return {"provider": PROVIDER, "model": model, "name": MODELS.get(model, model),
            "effort": chosen.get("effort") or "medium"}


def ready(config) -> bool:
    """Whether the assistant has a model to answer with: one set up here, or remote mode."""
    return bool(config.get("models.chat")) or active(config) is not None


def sdk_installed() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def install_sdk(progress=print) -> None:
    """Anthropic's own client library, into the environment CodeRunner runs from (it is not needed, and not
    installed, until remote mode is turned on)."""
    if sdk_installed():
        return
    progress("installing Anthropic's client library")
    done = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "anthropic>=1,<2"],
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError("Anthropic's client library could not be installed: "
                           + (done.stderr or done.stdout).strip().splitlines()[-1][:200])


def turn_on(config, model: str = DEFAULT, progress=print) -> dict:
    """Remote mode on (the key must be kept already). The local model's settings stay untouched."""
    if model not in MODELS:
        raise ValueError(f"Choose one of {', '.join(MODELS)}.")
    if not key():
        raise ValueError("No Anthropic key is kept yet.")
    install_sdk(progress)
    config.set("models.remote", {"provider": PROVIDER, "model": model})
    config.save()
    return active(config)


def turn_off(config, forget: bool = False) -> None:
    """Back to the model on this computer; the key is kept for next time unless forget."""
    config.set("models.remote", {})
    config.save()
    if forget:
        forget_key()


def client(config, **options):
    """The model client for remote mode, or None when it is off."""
    chosen = active(config)
    if chosen is None:
        return None
    if chosen["provider"] == SERVER:
        return server_client(chosen)
    from ..core.claude import Claude
    return Claude(key(), chosen["model"], effort=chosen["effort"], **options)


def claude_model(config) -> str:
    """The Claude model a single task sent to Anthropic runs on: remote mode's choice when that is Claude."""
    chosen = (config.get("models.remote") if config is not None else None) or {}
    return chosen.get("model") if chosen.get("provider") == PROVIDER and chosen.get("model") in MODELS else DEFAULT


def successor(served: list[str], current: str) -> str | None:
    """The name to use when the server no longer serves current: the newest Bonsai it serves (Bonsai2.5 after
    Bonsai2), one of the same size first; else its only model. None while current is still served."""
    if current in served or not served:
        return None

    def version(name: str) -> float:
        found = re.search(r"bonsai[\s_-]*v?(\d+(?:[._]\d+)?)", name, re.I)
        return float(found.group(1).replace("_", ".")) if found else 0.0

    sizes = set(re.findall(r"\d+b\b", current.lower()))
    bonsai = [name for name in served if "bonsai" in name.lower()]
    if bonsai:
        return max(bonsai, key=lambda name: (version(name), bool(sizes & set(re.findall(r"\d+b\b", name.lower())))))
    return served[0] if len(served) == 1 else None


def server_client(chosen: dict):
    """The owner's model server as the assistant's model: no slots of ours to pin, no runtime of ours to restart.
    Its context per task is read from llama.cpp's /props where it has one, else 32K."""
    import httpx
    from ..core.llm import LLM
    llm = LLM(chosen["base_url"], chosen["model"], server_key())
    llm.remote, llm.provider = True, SERVER
    root = chosen["base_url"][:-3] if chosen["base_url"].endswith("/v1") else chosen["base_url"]

    async def recover() -> bool:
        """Not ours to restart: a request that failed (the network between, the server busy or restarting) is
        tried once more when the server says it is healthy, within a minute. A task 75 steps in ended on one
        dropped request."""
        import asyncio
        headers = {"Authorization": f"Bearer {server_key()}"} if server_key() else {}
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            for _ in range(12):
                try:
                    if (await client.get(f"{root}/health", headers=headers)).status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)
        return False
    llm.recover = recover

    async def rename() -> str | None:
        try:
            return successor(await llm.list_models(), llm.model)
        except Exception:
            return None
    llm.rename = rename
    context = 32768
    try:
        headers = {"Authorization": f"Bearer {server_key()}"} if server_key() else {}
        props = httpx.get(f"{root}/props", headers=headers, timeout=5, trust_env=False).json()
        n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
        slots = max(1, int(props.get("total_slots") or 1))
        if n_ctx:
            # Whether the slots share it cannot be told from here (both kinds report the whole figure): each task
            # keeps to its share.
            context = n_ctx // slots
        llm.slots = slots
    except Exception:
        llm.slots = 1
    llm.context = context
    if llm.slots > 1 and chosen.get("keep_slots", True):
        # The step judge's request in the last slot made the server drop the task's slot (core/llm.py,
        # SlotKeeper); kept in a file around it where the server allows (--slot-save-path). Off with
        # models.remote.keep_slots: false.
        llm.keep_slots()
    # A larger model than the Mac's reads more of a page at once and keeps links whole enough to cite.
    llm.page_chars, llm.link_chars = 16_000, 200
    return llm
