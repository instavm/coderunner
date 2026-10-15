"""Whether indexing should wait: a laptop on a low battery or in Low Power Mode is left alone."""

from __future__ import annotations

import asyncio
import pathlib
import re
import shutil
import sys
import time

from ..core import paths

CHECK_EVERY = 60.0
_cache: tuple[float, tuple[bool, int | None, bool]] = (0.0, (False, None, False))


async def _pmset(*args: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec("pmset", *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 5)
        return out.decode(errors="replace")
    except (OSError, asyncio.TimeoutError):
        return ""


async def state() -> tuple[bool, int | None, bool]:
    """(on battery, battery percent, low power mode)."""
    global _cache
    if time.monotonic() - _cache[0] < CHECK_EVERY:
        return _cache[1]
    on_battery, percent, low = False, None, False
    if sys.platform == "darwin" and shutil.which("pmset"):
        battery = await _pmset("-g", "batt")
        on_battery = "Battery Power" in battery
        if match := re.search(r"(\d+)%", battery):
            percent = int(match.group(1))
        low = bool(re.search(r"lowpowermode\s+1", await _pmset("-g")))
    else:
        for supply in pathlib.Path("/sys/class/power_supply").glob("BAT*"):
            try:
                on_battery = (supply / "status").read_text().strip() == "Discharging"
                percent = int((supply / "capacity").read_text().strip())
            except (OSError, ValueError):
                continue
            break
    _cache = (time.monotonic(), (on_battery, percent, low))
    return _cache[1]


async def pause_reason(config) -> str:
    """Why indexing should wait now, or "" to go ahead."""
    on_battery, percent, low = await state()
    if low and config.get("files.pause_in_low_power", True):
        return "Low Power Mode is on"
    minimum = int(config.get("files.min_battery", 30) or 0)
    if on_battery and percent is not None and percent < minimum:
        return f"on battery at {percent}% (below {minimum}%)"
    return disk_reason(config)


def disk_reason(config) -> str:
    """Indexing stops before it fills the disk: a large folder of source code grew the index by 4 GB in half
    an hour and left 2.5 GB free."""
    floor = float(config.get("files.min_free_gb", 5) or 0)
    try:
        free = shutil.disk_usage(paths.assistant_dir()).free / 1e9
    except OSError:
        return ""
    return f"only {free:.1f} GB free on disk (below {floor:g} GB)" if free < floor else ""


async def on_battery() -> bool:
    return (await state())[0]
