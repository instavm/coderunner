"""Keeps the assistant's computer up by itself, the way an assistant's own computer should be: nobody presses a
button to start it, set it up or retry it.

The daemon asks at start, the app whenever it wants the screen, and a task whenever it needs the VM. One attempt:

  1. A Mac whose disk is nearly full is said as such and nothing is tried: a full disk is what broke the VM's
     own disk on 2026-11-25 (I/O errors inside, `container list` saying running while exec and stop failed).
  2. The VM is started when it is down, and its server must answer.
  3. Its desktop is brought up (and the assistant's Chrome on it, in browser mode "computer"); a VM without a
     desktop gets one installed (sandbox/computer.py), once.
  4. Something broken is repaired, a step at a time, each audited: the desktop session restarted inside the VM;
     else the VM stopped and started; else the container service itself restarted (at most once an hour).

Attempts repeat with a growing wait; after the last one fails it rests, and the app says one plain sentence.

It also keeps the VM's disk small (watch_disk). That disk is a file on the Mac that only grows: blocks freed inside
the VM are never given back. Once an hour it is measured; past its budget (sandbox.disk_budget_gb) the VM is created
again the way `coderunner sandbox reset` does it, with the owner's data carried over on the data volume, which is
never deleted. Only with no task running and the owner away from the Mac, at most once a day, never when the VM
has not grown much since it was last made (a fresh VM above the budget would be made again for nothing), and never
when the Mac lacks the room to make it. Each decision is audited, and the app says it once, quietly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

READY, WORKING, FAILED, DISK_FULL, ABSENT = "ready", "working", "failed", "disk_full", "absent"
_DISK = "Your Mac's disk" if sys.platform == "darwin" else "This computer's disk"
MESSAGES = {
    READY: "",
    WORKING: "Getting the computer ready…",
    FAILED: "The assistant's computer could not be started. It will try again shortly.",
    DISK_FULL: f"{_DISK} is almost full, so the assistant's computer can't run. Free up some space.",
    ABSENT: "This install has no computer for the assistant.",
}
VM_DISK_FULL = "The assistant's computer has run out of disk space."
LOCAL_NETWORK_BLOCKED = (
    "macOS Local Network access is off for CodeRunner. "
    "Turn it on in System Settings → Privacy & Security → Local Network."
)
TIDYING = "Tidying the assistant's computer…"
TIDIED = "Tidied the assistant's computer"
NOTE_KEPT_S = 86400                 # the app shows the note once, if it opens within a day


def mac_idle_seconds() -> float | None:
    """Seconds since the owner last touched the Mac's keyboard, mouse or trackpad (HIDIdleTime); None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem", "-d", "4", "-r", "-k", "HIDIdleTime"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out or "")
    return int(found.group(1)) / 1e9 if found else None

# Inside the VM: can it write to its own disk, and is the desktop installed? Exit 3 is a disk that fails writes.
PROBE = ("touch /var/tmp/.coderunner-probe && rm -f /var/tmp/.coderunner-probe || exit 3; "
         "if command -v Xvnc >/dev/null && [ -x /usr/bin/google-chrome-stable ] "
         "&& id -u \"${CODERUNNER_DESKTOP_USER:-box}\" >/dev/null 2>&1; "
         "then echo desktop-installed; else echo desktop-missing; fi")
# The desktop session, ended so the sandbox server starts a fresh one (Chrome goes with its display).
# The bracketed letters keep `pkill -f` from matching (and killing) the shell that runs this line.
END_SESSION = "pkill -x Xvnc; pkill -f '[x]fce4-session'; pkill -f '[d]bus-launch --exit-with-session'; true"

STEPS = ("session", "container", "service")
FIRST_STEP = {"session": "session", "screen": "session", "unhealthy": "container", "unavailable": "container",
              "stuck": "container", "io": "container"}
# The container service is everyone's: restarted only when commands into the VM themselves fail (exec answering
# "container is not running" while it is listed as running, or its disk failing). A server that answers badly
# is not that: a lite sandbox's health check said 503 for ever, and its container and then the service were
# restarted though the VM was fine.
NEEDS_SERVICE = ("stuck", "io")


def classify(code: int | None, output: str) -> str:
    """What the probe inside the VM found: installed, missing, io, vm_disk_full or stuck."""
    text = output or ""
    if "No space left on device" in text:
        return "vm_disk_full"
    if "Input/output error" in text or "Read-only file system" in text or code == 3:
        return "io"
    if code == 0 and "desktop-installed" in text:
        return "installed"
    if code == 0 and "desktop-missing" in text:
        return "missing"
    return "stuck"          # exec failed while the VM is listed as running ("container is not running")


async def _vnc_answers(host: str, port: int) -> bool:
    """The screen's VNC server greets a new connection with its version ("RFB 003.008")."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 5)
    except (OSError, asyncio.TimeoutError):
        return False
    try:
        return (await asyncio.wait_for(reader.read(12), 5)).startswith(b"RFB ")
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        writer.close()


class Keeper:
    BACKOFF = (5, 15, 45, 120)      # seconds between attempts
    REST_S = 300                    # after the attempts fail, asking again only tries after this (doubling)
    MAX_REST_S = 3600
    RECHECK_S = 30                  # a full disk or a missing sandbox is looked at again this often
    MIN_GAP_S = 5                   # a computer that was just found ready is not probed again sooner
    SERVICE_EVERY_S = 3600          # the container service is restarted at most this often
    HEALTH_WAIT_S = 60
    LOW_DISK_GB = 2.0
    BROWSER_GAP_S = 30              # Chrome is reopened on a screen without it at most this often
    DISK_EVERY_S = 3600             # the VM's disk is measured this often
    TIDY_GAP_S = 86400              # and made again at most this often
    TIDY_GROWTH_GB = 1.0            # only when it has grown at least this much since it was last made
    OWNER_AWAY_S = 900              # the owner counts as away after this long without touching the Mac

    def __init__(self, config, driver, sandbox, audit, *, installer: Callable | None = None,
                 on_mode: Callable[[], Any] | None = None, disk_free_gb: Callable[[], float] | None = None,
                 vnc_check: Callable[[str, int], Awaitable[bool]] | None = None,
                 sleep: Callable[[float], Awaitable] | None = None, clock: Callable[[], float] = time.monotonic,
                 busy: Callable[[], bool] | None = None, owner_away: Callable[[], bool] | None = None,
                 recreate: Callable[[Any], bool] | None = None, tidy_file: pathlib.Path | None = None,
                 wall: Callable[[], float] = time.time):
        """busy: whether a task is running; owner_away: whether the owner is away from the Mac (both required for
        the disk budget: without them the VM is never made again); recreate: makes the VM again (sandbox/reset.py);
        tidy_file: where the last time it was made, and its size then, are kept across restarts."""
        self.config, self.driver, self.sandbox, self.audit = config, driver, sandbox, audit
        self._busy, self._owner_away = busy, owner_away
        self._recreate = recreate
        self._tidy_file = tidy_file
        self._wall = wall
        self._skipped = ""
        self.note: dict | None = None       # {"text", "at"}: said once by the app after the VM was made again
        self._installer = installer
        self._on_mode = on_mode
        self._disk_free_gb = disk_free_gb or self._mac_disk_free_gb
        self._vnc_check = vnc_check or _vnc_answers
        self._sleep = sleep or asyncio.sleep
        self._clock = clock
        self.state = "idle"
        self.message = ""
        self.want_desktop = True
        self._task: asyncio.Task | None = None
        self._healthy: asyncio.Event | None = None
        self._last_start = -math.inf
        self._rest_until = -math.inf
        self._service_at = -math.inf
        self._installed = False
        self._failures = 0
        self._browser_task: asyncio.Task | None = None
        self._browser_at = -math.inf

    # ---- what the app is told ----------------------------------------------

    def public(self) -> dict:
        state = self.state if self.state in (READY, FAILED, DISK_FULL, ABSENT) else WORKING
        return {"state": state, "message": self.message or MESSAGES[state]}

    def saw_ready(self) -> None:
        """The gateway reached the screen itself: whatever an earlier attempt concluded, it is up now."""
        if self._task is None or self._task.done():
            self._set(READY)

    def screen_without_browser(self) -> asyncio.Task | None:
        """The screen is up but the assistant's Chrome is not on it: the sandbox brought its display back by
        itself (Chrome goes with the display), so nothing here repaired anything and Chrome was not reopened."""
        if self._browser_task is not None and not self._browser_task.done():
            return self._browser_task
        now = self._clock()
        if now - self._browser_at < self.BROWSER_GAP_S:
            return None
        self._browser_at = now
        self._browser_task = asyncio.get_running_loop().create_task(self._browser())
        return self._browser_task

    def _set(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message

    # ---- asking for it ------------------------------------------------------

    def kick(self, desktop: bool | None = None) -> asyncio.Task | None:
        """Make sure of the computer in the background; the running attempt when one is under way, None when it
        is resting (failed a moment ago, a full disk, no sandbox, or just found ready)."""
        if desktop:
            self.want_desktop = True
        if self._task is not None and not self._task.done():
            return self._task
        now = self._clock()
        if self.state == FAILED and now < self._rest_until:
            return None
        if self.state in (DISK_FULL, ABSENT) and now - self._last_start < self.RECHECK_S:
            return None
        if self.state == READY and now - self._last_start < self.MIN_GAP_S:
            return None
        self._last_start = now
        self._healthy_event().clear()
        self._task = asyncio.get_running_loop().create_task(self._run())
        return self._task

    async def ensure(self, desktop: bool = False, timeout: float = 120) -> bool:
        """For a task that needs the VM (desktop=False: its server answering) or its screen: True once it is
        there, False when it could not be had within `timeout`."""
        task = self.kick(desktop or None)
        if task is not None:
            waiters = {asyncio.ensure_future(asyncio.shield(task))}
            if not desktop:
                waiters.add(asyncio.ensure_future(self._healthy_event().wait()))
            done, pending = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for waiter in pending:
                waiter.cancel()
        if self.state == READY:
            return True
        return not desktop and self._healthy_event().is_set()

    def _healthy_event(self) -> asyncio.Event:
        if self._healthy is None:
            self._healthy = asyncio.Event()
        return self._healthy

    # ---- attempts -----------------------------------------------------------

    async def _run(self) -> str:
        self._installed = False
        for wait in (0, *self.BACKOFF):
            if wait:
                await self._sleep(wait)
            try:
                outcome = await self._attempt()
            except Exception as exc:          # never let a surprise end the daemon's care for it
                self.audit.write("computer.error", error=f"{type(exc).__name__}: {exc}"[:200])
                outcome = "broken"
            if outcome in (READY, DISK_FULL, ABSENT) or (outcome == FAILED and self.message == LOCAL_NETWORK_BLOCKED):
                return outcome
            if self.state != WORKING:
                self._set(WORKING)
        self._failures += 1
        rest = min(self.REST_S * 2 ** (self._failures - 1), self.MAX_REST_S)
        self._rest_until = self._clock() + rest
        self._set(FAILED)
        self.audit.write("computer.unavailable", rest_s=rest)
        return FAILED

    async def _attempt(self) -> str:
        if self._disk_low():
            return self._disk_full()
        try:
            status = await asyncio.to_thread(self.driver.status)
        except FileNotFoundError:             # the platform's tool is not installed
            status = "unavailable"
        except Exception as exc:              # a hung service: tried as a VM that is down
            self.audit.write("computer.call_failed", call="status", error=str(exc)[:200])
            status = "unknown"
        if status in ("missing", "unavailable"):
            if self.state != ABSENT:
                self.audit.write("computer.absent", driver=getattr(self.driver, "name", "?"), status=status)
            self._set(ABSENT)
            return ABSENT
        if self.state not in (READY, WORKING):
            self._set(WORKING)
        if status != "running":
            self._set(WORKING)
            ok = await self._call(self.driver.start, default=False)
            self.audit.write("sandbox.started" if ok else "sandbox.start_failed",
                             driver=getattr(self.driver, "name", "?"), was=status)
        problem = await self._probe()
        if problem is None:
            return self._ready()
        self._set(WORKING)
        if problem == "missing":
            return await self._install()
        return await self._repair(problem)

    async def _probe(self) -> str | None:
        """None when the computer is as wanted; otherwise what is wrong."""
        if not await self._answers(self.HEALTH_WAIT_S):
            kind = await self._inside()
            if kind in ("installed", "missing") and self._local_network_blocked():
                return "local_network"
            return kind if kind in ("io", "vm_disk_full", "stuck") else "unhealthy"
        self._healthy_event().set()
        if not self.want_desktop:
            return None
        try:
            info = await self.sandbox.desktop(start=True)
        except Exception:
            kind = await self._inside()
            return kind if kind in ("io", "vm_disk_full", "stuck") else "unhealthy"
        if not info.get("available"):        # 409, or a server older than the desktop (404)
            kind = await self._inside()
            if kind == "missing" or (kind == "installed" and "predates" in str(info.get("error", ""))):
                return "missing"             # an older server is replaced along with the install
            return "unavailable" if kind == "installed" else kind
        if not info.get("running"):
            return "session"
        host = urlsplit(getattr(self.sandbox, "base_url", "") or "").hostname or "127.0.0.1"
        if not await self._vnc_check(host, int(info.get("port") or 5900)):
            return "screen"
        await self._browser()
        return None

    async def _browser(self) -> None:
        """The assistant's Chrome on the screen, so the owner sees the browser rather than a bare desktop. One
        Chrome (the one the tasks use); a failure here is not the computer's, and is left to the first task."""
        if self.config.get("browser.mode") != "computer":
            return
        try:
            info = await self.sandbox.desktop(start=True, chrome=True)
            ok = bool(info.get("chrome", info.get("running")))
        except Exception:
            ok = False
        if not ok:
            self.audit.write("computer.chrome_failed")

    async def _repair(self, problem: str) -> str:
        if problem == "vm_disk_full":
            self.audit.write("computer.disk_full", where="vm")
            self._set(DISK_FULL, VM_DISK_FULL)
            return DISK_FULL
        if problem == "local_network":
            self.audit.write("computer.local_network_blocked", url=getattr(self.sandbox, "base_url", ""))
            self._set(FAILED, LOCAL_NETWORK_BLOCKED)
            return FAILED
        for step in STEPS[STEPS.index(FIRST_STEP.get(problem, "container")):]:
            if self._disk_low():
                return self._disk_full()
            if step == "service" and problem not in NEEDS_SERVICE:
                self.audit.write("computer.repair_skipped", step=step, reason="the VM takes commands")
                break
            if step == "service" and self._clock() - self._service_at < self.SERVICE_EVERY_S:
                self.audit.write("computer.repair_skipped", step=step, reason="restarted recently")
                break
            ok = await self._step(step)
            self.audit.write("computer.repair", step=step, problem=problem, ok=ok)
            if not ok:
                continue
            again = await self._probe()
            if again is None:
                self.audit.write("computer.repaired", step=step, problem=problem)
                return self._ready()
            if again == "missing":
                return await self._install()
            if again == "vm_disk_full":
                return await self._repair(again)
            problem = again
        return "broken"

    async def _step(self, step: str) -> bool:
        if step == "session":
            code, _ = await self._call(self.driver.run_in, ["sh", "-c", END_SESSION], 20, default=(None, ""))
            if code is None:
                return False
            await self._sleep(1)
            return True
        if step == "container":
            await self._call(self.driver.stop, default=False)         # fails on a stuck VM; start says
            return bool(await self._call(self.driver.start, default=False))
        self._service_at = self._clock()
        if not await self._call(self.driver.restart_service, default=False):
            return False
        return bool(await self._call(self.driver.start, default=False))

    async def _install(self) -> str:
        if self._installed:                   # installed in this run and still missing: not again
            return "broken"
        self._installed = True
        if self._disk_low():
            return self._disk_full()
        lines: list[str] = []
        from . import computer
        installer = self._installer or computer.install
        ok = bool(await self._call(installer, self.driver, lines.append, default=False))
        self.audit.write("computer.installed", ok=ok, last=" | ".join(lines[-3:])[-300:])
        if not ok:
            if any("No space left on device" in line for line in lines):
                return await self._repair("vm_disk_full")
            return "broken"
        # The sandbox server picks up its desktop routes when it starts again.
        await self._call(self.driver.stop, default=False)
        await self._call(self.driver.start, default=False)
        if self.config.get("browser.mode") != "chrome":     # the owner's own browser stays their choice
            self.config.set("browser.mode", "computer")
            with contextlib.suppress(Exception):
                self.config.save()
            if self._on_mode is not None:
                self._on_mode()
        again = await self._probe()
        if again is None:
            return self._ready()
        return "broken" if again == "missing" else await self._repair(again)

    # ---- the VM's disk on the Mac -----------------------------------------------

    def budget_gb(self) -> float:
        try:
            return max(0.0, float(self.config.get("sandbox.disk_budget_gb", 8) or 0))
        except (TypeError, ValueError):
            return 8.0

    def disk_gb(self) -> float | None:
        """What the VM's disk takes on the Mac now (a stat of its file), or None where it is not measured."""
        try:
            used = self.driver.disk_bytes()
        except Exception:
            return None
        return round(used / 1e9, 1) if used else None

    def public_note(self) -> dict | None:
        """The quiet note for the app, while it is fresh."""
        if self.note and self._wall() - self.note["at"] < NOTE_KEPT_S:
            return dict(self.note)
        return None

    async def watch_disk(self) -> None:
        """The daemon's slow loop: the VM's disk looked at once an hour."""
        while True:
            await self._sleep(self.DISK_EVERY_S)
            try:
                await self.check_disk()
            except Exception as exc:          # never let a surprise end the watch
                self.audit.write("computer.error", error=f"disk: {type(exc).__name__}: {exc}"[:200])

    async def check_disk(self) -> str:
        """One look at the VM's disk; what was decided. "tidying" starts making the VM again (in self._task)."""
        budget = self.budget_gb()
        used = await asyncio.to_thread(self.disk_gb)
        if used is None or budget <= 0:
            return "unmeasured" if used is None else "off"
        kept = self._tidy_state()
        if kept.get("pending") and self.state == READY:
            # The size of the VM as made, its desktop and Chrome included: growth is counted from here.
            kept = {**kept, "fresh_gb": used}
            kept.pop("pending")
            self._save_tidy_state(kept)
        if used <= budget:
            self._skipped = ""
            return "within"
        image_gb = None
        fresh = kept.get("fresh_gb")
        if fresh is None:
            image_gb = await asyncio.to_thread(self._image_gb)
            fresh = image_gb or 0.0
        if used - fresh < self.TIDY_GROWTH_GB:
            return self._skip("fresh", used, budget, fresh_gb=fresh)
        if self._wall() - float(kept.get("at") or 0) < self.TIDY_GAP_S:
            return self._skip("today", used, budget)
        if self._task is not None and not self._task.done():
            return self._skip("working", used, budget)
        if self._busy is None or self._owner_away is None or self._recreate is None:
            return self._skip("unwired", used, budget)
        if self._busy():
            return self._skip("task", used, budget)
        if not self._owner_away():
            return self._skip("owner", used, budget)
        if image_gb is None:
            image_gb = await asyncio.to_thread(self._image_gb)
        need = (image_gb or fresh or used) + self.LOW_DISK_GB
        try:
            free = self._disk_free_gb()
        except Exception:
            free = 0.0
        if free < need:
            return self._skip("low_disk", used, budget, free_gb=round(free, 1), need_gb=round(need, 1))
        self._skipped = ""
        self._last_start = self._clock()
        self._healthy_event().clear()
        self._set(WORKING, TIDYING)
        self._task = asyncio.get_running_loop().create_task(self._tidy(used, budget))
        return "tidying"

    async def _tidy(self, used: float, budget: float) -> str:
        # Kept before anything is deleted: a daemon that stops midway does not try again within the day.
        kept = {"at": self._wall(), "pending": True, "before_gb": used}
        self._save_tidy_state(kept)
        self.audit.write("sandbox.tidy", used_gb=used, budget_gb=budget)
        ok = bool(await self._call(self._recreate, self.driver, default=False))
        self.audit.write("sandbox.tidied", ok=ok, was_gb=used, used_gb=await asyncio.to_thread(self.disk_gb))
        if ok:
            self.note = {"text": TIDIED, "at": self._wall()}
        else:                                # nothing to count growth from: the image's size stands in for it
            kept.pop("pending")
            self._save_tidy_state(kept)
        outcome = await self._run()          # up again, with its desktop and Chrome
        fresh = await asyncio.to_thread(self.disk_gb) if ok and outcome == READY else None
        if fresh is not None:                # its size as made, desktop included: growth counts from here
            kept.pop("pending")
            self._save_tidy_state({**kept, "fresh_gb": fresh})
        return outcome

    def _skip(self, reason: str, used: float, budget: float, **more) -> str:
        if reason != self._skipped:          # audited when the reason changes, not every hour
            self.audit.write("sandbox.tidy_skipped", reason=reason, used_gb=used, budget_gb=budget, **more)
            self._skipped = reason
        return reason

    def _image_gb(self) -> float | None:
        try:
            image = (self.driver.settings() or {}).get("image") or ""
            size = self.driver.image_bytes(image) if image else None
        except Exception:
            return None
        return round(size / 1e9, 1) if size else None

    def _tidy_path(self) -> pathlib.Path:
        if self._tidy_file is None:
            from ..core import paths
            self._tidy_file = paths.assistant_dir() / "sandbox-tidy.json"
        return self._tidy_file

    def _tidy_state(self) -> dict:
        try:
            kept = json.loads(self._tidy_path().read_text())
        except (OSError, ValueError):
            return {}
        return kept if isinstance(kept, dict) else {}

    def _save_tidy_state(self, kept: dict) -> None:
        with contextlib.suppress(OSError):
            path = self._tidy_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(kept))

    # ---- pieces -------------------------------------------------------------

    def _ready(self) -> str:
        self._failures = 0
        if self.state != READY:
            self.audit.write("computer.ready", desktop=self.want_desktop)
        self._set(READY)
        return READY

    def _disk_full(self) -> str:
        if self.state != DISK_FULL:
            self.audit.write("computer.disk_full", where="host", free_gb=round(self._disk_free_gb(), 2))
        self._set(DISK_FULL)
        return DISK_FULL

    def _disk_low(self) -> bool:
        try:
            return self._disk_free_gb() < self.LOW_DISK_GB
        except Exception:
            return False

    @staticmethod
    def _mac_disk_free_gb() -> float:
        return shutil.disk_usage(pathlib.Path.home()).free / 1e9

    def _local_network_blocked(self) -> bool:
        """macOS Local Network privacy rejects connections to 192.168.x.x / *.local with immediate EHOSTUNREACH."""
        if sys.platform != "darwin":
            return False
        host = (urlsplit(getattr(self.sandbox, "base_url", "") or "").hostname or "").lower()
        if not host or host in ("127.0.0.1", "localhost", "::1"):
            return False
        exc = getattr(self.sandbox, "last_error", None)
        while exc is not None:
            if getattr(exc, "errno", None) == 65 or "no route to host" in str(exc).lower():
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    async def _answers(self, seconds: float) -> bool:
        no_route_streak = 0
        for attempt in range(max(1, int(seconds // 2))):
            with contextlib.suppress(Exception):
                if await self.sandbox.health():
                    return True
            if self._local_network_blocked():
                no_route_streak += 1
                if no_route_streak >= 3:
                    return False
            else:
                no_route_streak = 0
            await self._sleep(2)
        return False

    async def _inside(self) -> str:
        code, output = await self._call(self.driver.run_in, ["sh", "-c", PROBE], 20, default=(None, ""))
        return classify(code, output)

    async def _call(self, fn, *args, default=None):
        """A driver call (they block) off the event loop; its exceptions count as failure."""
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:
            self.audit.write("computer.call_failed", call=getattr(fn, "__name__", "?"), error=str(exc)[:200])
            return default
