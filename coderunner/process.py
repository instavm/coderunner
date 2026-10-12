"""Finding and running local tools (ffmpeg, speech engines, chat bridges, Swift helpers).

Commands are argument lists, never shell strings, and a child is killed when its time runs out or
the caller is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import shutil

# Daemons started by launchd get a minimal PATH, so also look where Homebrew, pipx and uv install tools.
EXTRA_PATHS = ["/opt/homebrew/bin", "/usr/local/bin", str(pathlib.Path.home() / ".local" / "bin")]


def find_binary(name: str, override: str | None = None) -> str | None:
    """Resolve an executable by name (PATH plus common install dirs) or by explicit path."""
    candidate = os.path.expanduser(override or name)
    if os.sep in candidate:
        return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None
    return shutil.which(candidate, path=os.pathsep.join([os.environ.get("PATH", ""), *EXTRA_PATHS]))


async def run(argv: list[str], timeout: float, stdin: bytes | None = None) -> tuple[int, str, str]:
    """Run argv directly (never through a shell); the process is killed on timeout or cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def terminate(proc: asyncio.subprocess.Process | None, grace: float = 5) -> None:
    """Stop a long-running child: SIGTERM, then SIGKILL after `grace` seconds."""
    if proc is None or proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
