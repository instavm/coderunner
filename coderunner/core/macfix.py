"""Fixes on the owner's Mac that need an administrator, run through macOS's own password prompt.

A short fixed list. Each fix knows how to tell it is needed (detect), the one command that fixes it (a constant
here, never built from anything a model, a page or a person typed), and how to tell it worked (check). The owner
starts a fix from the app; the command runs as `osascript -e 'do shell script "..." with administrator privileges'`
from an argument list (no shell), so macOS shows its own dialog and the password never reaches CodeRunner. The
assistant has no tool for this: only the owner's own request to the app runs a fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Awaitable, Callable

RUN_TIMEOUT = 300           # the owner may take a while to type their password
CHECK_TIMEOUT = 90
SETTLE_S = 1.5            # launchd brings the stopped services back when Vision next asks
CHECK_PICTURE = pathlib.Path(__file__).with_name("macfix-check.png")
CHECK_WORD = "lighthouse"   # the word in that picture


@dataclass(frozen=True)
class Fix:
    id: str
    title: str                                   # what the owner notices, in their words
    detect: Callable[[], bool]
    command: str                                 # constant: run as root through macOS's prompt
    check: Callable[[], Awaitable[bool]]
    not_fixed: str = "Not fixed."


# ---- the Neural Engine: text in pictures read with the fast model ----------------

def _ocr_fell_back() -> bool:
    from ..files import extract
    return extract.ocr_fell_back()


async def _accurate_ocr_works() -> bool:
    """A fresh helper reads the built-in picture: True when Vision's accurate model read its word."""
    from ..files import extract
    helper = await extract.Media().helper()
    if not helper or not CHECK_PICTURE.is_file():
        return False
    try:
        proc = await asyncio.create_subprocess_exec(helper, "ocr", stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(str(CHECK_PICTURE).encode() + b"\n"), CHECK_TIMEOUT)
        reply = json.loads(out.splitlines()[0]) if out.strip() else {}
    except (OSError, ValueError, asyncio.TimeoutError):
        return False
    return not reply.get("fallback") and CHECK_WORD in str(reply.get("text", "")).lower()


def _ocr_fixed() -> None:
    from ..files import extract
    extract.ocr_reset()


FIXES = (
    Fix(id="neural-engine", title="Text in pictures is being read at lower quality.", detect=_ocr_fell_back,
        command="killall ANECompilerService aned", check=_accurate_ocr_works,
        not_fixed="Not fixed. Restarting your Mac also fixes this."),
)
AFTER = {"neural-engine": _ocr_fixed}            # what changes here once a fix worked

for _fix in FIXES:          # nothing in a command could break out of the AppleScript string it is put in
    assert not set(_fix.command) & set('"\\\'$`;|&<>\n'), _fix.id


def available() -> tuple[Fix, ...]:
    return FIXES if sys.platform == "darwin" else ()


def find(fix_id: str) -> Fix | None:
    return next((fix for fix in available() if fix.id == fix_id), None)


def detected() -> list[Fix]:
    found = []
    for fix in available():
        try:
            if fix.detect():
                found.append(fix)
        except Exception:
            continue
    return found


def admin_argv(command: str) -> list[str]:
    """osascript asking macOS to run the constant command as an administrator, with its own password dialog."""
    return ["osascript", "-e", f'do shell script "{command}" with administrator privileges']


def cancelled(stderr: str) -> bool:
    return "-128" in stderr or "User canceled" in stderr or "User cancelled" in stderr


_running: set[str] = set()


async def run(fix: Fix, audit=None, spawn=None) -> dict:
    """Run a fix through the admin prompt, then check it. result: fixed, not_fixed, cancelled or busy."""
    if _running:                       # one dialog at a time
        return {"id": fix.id, "result": "busy", "message": "Already fixing."}
    _running.add(fix.id)
    spawn = spawn or asyncio.create_subprocess_exec
    try:
        try:
            proc = await spawn(*admin_argv(fix.command), stdin=asyncio.subprocess.DEVNULL,
                               stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, err = await asyncio.wait_for(proc.communicate(), RUN_TIMEOUT)
            stderr, code = err.decode(errors="replace"), proc.returncode
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            stderr, code = "timed out", None
        except OSError as exc:
            stderr, code = str(exc), None
        if code != 0 and cancelled(stderr):
            result = {"id": fix.id, "result": "cancelled", "message": "Not done."}
        else:
            # killall says it found nothing to stop when one of the two was not running: the check decides.
            ok = False
            try:
                if code == 0:
                    await asyncio.sleep(SETTLE_S)
                ok = await fix.check()
            except Exception:
                ok = False
            if ok and fix.id in AFTER:
                AFTER[fix.id]()
            result = {"id": fix.id, "result": "fixed" if ok else "not_fixed",
                      "message": "Fixed." if ok else fix.not_fixed}
        if audit is not None:
            audit.write("macfix.run", id=fix.id, result=result["result"])
        return result
    finally:
        _running.discard(fix.id)
