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


def active(config) -> dict | None:
    """{"provider", "model", "name"} while remote mode is on and its key is here, else None."""
    chosen = (config.get("models.remote") if config is not None else None) or {}
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
    from ..core.claude import Claude
    return Claude(key(), chosen["model"], effort=chosen["effort"], **options)
