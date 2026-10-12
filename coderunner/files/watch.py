"""Changes in the shared folders as they happen (FSEvents on macOS), so new files are found in seconds.

Elsewhere, or without the macOS helper, the indexer walks the folders every files.scan_interval seconds
instead. File events only say which paths to look at again; every path still goes through the same rules
as a walk (discover.Folders.check).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Awaitable, Callable

from .extract import Media

SETTLE = 0.5                # seconds of quiet before a burst of events is handled


class Watcher:
    def __init__(self, media: Media):
        self.media = media
        self.running = False

    async def run(self, roots: list[pathlib.Path], changed: Callable[[set[str]], Awaitable[None]]) -> bool:
        """Call `changed` with paths as they change, until cancelled. False when events are unavailable."""
        helper = await self.media.helper()
        if not helper or not roots:
            return False
        proc = await asyncio.create_subprocess_exec(helper, "watch", *map(str, roots), stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        self.running = True
        pending: set[str] = set()
        try:
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), SETTLE if pending else None)
                except asyncio.TimeoutError:
                    batch, pending = pending, set()
                    await changed(batch)
                    continue
                if not line:
                    return False
                try:
                    pending.update(json.loads(line).get("paths") or [])
                except ValueError:
                    continue
        finally:
            self.running = False
            if proc.returncode is None:
                proc.stdin.close()
                proc.kill()
                await proc.wait()
