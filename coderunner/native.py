"""Small native helpers built from Swift source shipped with CodeRunner, once, on first use (macOS).

Apple's on-device frameworks (Vision, PDFKit, FSEvents, Speech) have no Python bindings in the standard
library, so a feature that needs one ships a short Swift program and runs it as a subprocess. The binary
is cached under the assistant's folder, keyed by a hash of its source, so an update rebuilds it. Without
the Swift compiler (Xcode Command Line Tools) the feature falls back or stays off; nothing is downloaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import threading

from . import paths

BUILD_TIMEOUT = 600
_lock = threading.Lock()


def _compiler() -> list[str] | None:
    if shutil.which("swiftc"):
        return ["swiftc"]
    if shutil.which("xcrun"):
        return ["xcrun", "swiftc"]
    return None


def build(source: pathlib.Path, name: str) -> str | None:
    """Path of the helper built from source, building it now if needed; None where it cannot be built."""
    if sys.platform != "darwin":
        return None
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    folder = paths.assistant_dir() / "bin"
    binary, failed = folder / f"{name}-{digest}", folder / f"{name}-{digest}.failed"
    if binary.exists():
        return str(binary)
    if failed.exists():                     # do not retry a failed build on every call
        return None
    compiler = _compiler()
    if compiler is None:
        return None
    with _lock:
        if binary.exists():
            return str(binary)
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / f".{name}-{digest}.{os.getpid()}"
        try:
            done = subprocess.run([*compiler, "-O", "-o", str(partial), str(source)], capture_output=True,
                                  text=True, timeout=BUILD_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failed.write_text(str(exc))
            return None
        if done.returncode != 0 or not partial.exists():
            failed.write_text(done.stderr[-4000:])
            partial.unlink(missing_ok=True)
            return None
        os.replace(partial, binary)
        for old in folder.glob(f"{name}-*"):
            if old != binary:
                old.unlink(missing_ok=True)
    return str(binary)


async def ensure(source: pathlib.Path, name: str) -> str | None:
    return await asyncio.to_thread(build, source, name)
