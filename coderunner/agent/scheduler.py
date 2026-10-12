"""Recurring and one-shot tasks the assistant runs on the owner's behalf.

Specs are small and human-readable (case-insensitive):

    every 30m              fixed interval, minimum 5 minutes
    daily 08:00            every day at a wall-clock time
    weekdays 09:00         Monday to Friday
    weekly mon 09:00       one day a week
    at 2026-12-24 18:00    once, at a local date and time
    in 2h                  once, after a delay

Wall-clock times are resolved in the owner's timezone so they stay put across
DST changes; intervals and delays are elapsed time. Runs are at-most-once: a
schedule is claimed (next_run advanced or, for one-shots, disabled) before it
runs, so a crash or a concurrent tick never repeats it, and runs missed while
the daemon was off collapse into a single catch-up run.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timedelta, tzinfo
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

RunTask = Callable[[dict], Awaitable[str]]
Deliver = Callable[[dict, str], Awaitable[None]]

MIN_INTERVAL = 5 * 60
MAX_DURATION = 3650 * 86400
USAGE = "use one of: every 30m | daily 08:00 | weekdays 09:00 | weekly mon 09:00 | at 2026-12-24 18:00 | in 2h"

_UNITS = {"m": 60, "h": 3600, "d": 86400}
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_CLOCK = r"(?:at )?(\d{1,2}):(\d{2})"
_DURATION = re.compile(r"(every|in) (\d{1,9}) ?(m|mins?|minutes?|h|hrs?|hours?|d|days?)")
_DAILY = re.compile(rf"(daily|weekdays) {_CLOCK}")
_WEEKLY = re.compile(rf"weekly ([a-z]+) {_CLOCK}")
_AT = re.compile(r"at (\d{4})-(\d{1,2})-(\d{1,2})[ t](\d{1,2}):(\d{2})")


def _clock(hour: str, minute: str) -> tuple[int, int]:
    h, m = int(hour), int(minute)
    if h > 23 or m > 59:
        raise ValueError(f"invalid time {hour}:{minute}; use 24-hour HH:MM")
    return h, m


def _weekday(word: str) -> int:
    for index, name in enumerate(_DAYS):
        if len(word) >= 3 and name.startswith(word):
            return index
    raise ValueError(f"unknown weekday {word!r}; use mon, tue, wed, thu, fri, sat or sun")


def _parse(spec: str) -> tuple:
    text = " ".join(str(spec or "").lower().split())
    if m := _DURATION.fullmatch(text):
        seconds = int(m[2]) * _UNITS[m[3][0]]
        if m[1] == "every" and seconds < MIN_INTERVAL:
            raise ValueError(f"{spec!r} is too frequent; the shortest interval is every 5m")
        if seconds <= 0:
            raise ValueError(f"{spec!r} needs a positive delay")
        if seconds > MAX_DURATION:
            raise ValueError(f"{spec!r} is too far out; the longest is 3650d")
        return m[1], seconds
    if m := _DAILY.fullmatch(text):
        return m[1], _clock(m[2], m[3])
    if m := _WEEKLY.fullmatch(text):
        return "weekly", _clock(m[2], m[3]), _weekday(m[1])
    if m := _AT.fullmatch(text):
        try:
            return "at", datetime(*(int(part) for part in m.groups()))
        except ValueError as exc:
            raise ValueError(f"invalid date or time in {spec!r}: {exc}") from None
    raise ValueError(f"unrecognised schedule {spec!r}; {USAGE}")


def parse_spec(spec: str, now: datetime) -> datetime | None:
    """Next run strictly after now, in now's timezone; None for a one-shot in the past."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    rule = _parse(spec)
    tz, after = now.tzinfo, now.timestamp()
    # Compare epoch seconds, never aware datetimes: same-tz comparison ignores
    # fold, which would misorder the repeated hour when DST ends.
    if rule[0] in ("every", "in"):
        return datetime.fromtimestamp(after + rule[1], tz)
    if rule[0] == "at":
        at = rule[1].replace(tzinfo=tz).timestamp()
        return datetime.fromtimestamp(at, tz) if at > after else None
    hour, minute = rule[1]
    days = {"daily": range(7), "weekdays": range(5), "weekly": rule[2:]}[rule[0]]
    for offset in range(8):
        day = now.date() + timedelta(days=offset)
        if day.weekday() not in days:
            continue
        # fold=0 picks the first of a repeated time and shifts a skipped one
        # forward by the gap (02:30 on spring-forward day runs at 03:30).
        candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).timestamp()
        if candidate > after:
            return datetime.fromtimestamp(candidate, tz)
    raise AssertionError("unreachable: every weekly rule matches within 8 days")


def is_one_shot(spec: str) -> bool:
    return _parse(spec)[0] in ("at", "in")


def _interval(spec: str) -> int | None:
    rule = _parse(spec)
    return rule[1] if rule[0] == "every" else None


def _zone(name: str | None) -> tzinfo:
    if name:
        try:
            return ZoneInfo(name)
        except (KeyError, ValueError, OSError):
            raise ValueError(f"unknown timezone {name!r}; use an IANA name like Europe/Berlin") from None
    env = os.environ.get("TZ", "").lstrip(":")
    if env:
        try:
            return ZoneInfo(env)
        except (KeyError, ValueError, OSError):
            pass
    try:
        with open("/etc/localtime", "rb") as handle:
            return ZoneInfo.from_file(handle, key="localtime")
    except (OSError, ValueError):
        return datetime.now().astimezone().tzinfo  # fixed offset; no DST awareness


class Scheduler:
    def __init__(self, store, audit, run_task: RunTask, deliver: Deliver, tz: str | None = None,
                 run_timeout: float | None = 900):
        self.store = store
        self.audit = audit
        self.run_task = run_task
        self.deliver = deliver
        self.tz = _zone(tz)
        self.run_timeout = run_timeout
        self._running: set[int] = set()

    def _now(self, now: datetime | None = None) -> datetime:
        if now is None:
            return datetime.now(self.tz)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(self.tz)

    def add(self, name: str, spec: str, prompt: str, channel: str = "", chat_id: str = "", *,
            now: datetime | None = None) -> dict:
        name, prompt = (name or "").strip(), (prompt or "").strip()
        spec = " ".join((spec or "").split())
        if not name:
            raise ValueError("a schedule needs a name")
        if not prompt:
            raise ValueError("a schedule needs a prompt")
        current = self._now(now)
        next_run = parse_spec(spec, current)
        if next_run is None:
            raise ValueError(f"{spec!r} is in the past")
        cursor = self.store.execute(
            "INSERT INTO schedules(name, spec, prompt, channel, chat_id, next_run, last_status, enabled, created) "
            "VALUES (?, ?, ?, ?, ?, ?, '', 1, ?)",
            (name, spec, prompt, channel or "", str(chat_id or ""), next_run.timestamp(), current.timestamp()),
        )
        row = self.get(cursor.lastrowid)
        self.audit.write("schedule.add", id=row["id"], name=name, spec=spec, channel=channel,
                         next_run=next_run.isoformat())
        return row

    def get(self, id: int) -> dict | None:
        row = self.store.one("SELECT * FROM schedules WHERE id = ?", (id,))
        return {**row, "enabled": bool(row["enabled"])} if row else None

    def list(self) -> list[dict]:
        return [{**row, "enabled": bool(row["enabled"])} for row in self.store.query("SELECT * FROM schedules ORDER BY id")]

    def remove(self, id: int) -> bool:
        removed = self.store.execute("DELETE FROM schedules WHERE id = ?", (id,)).rowcount > 0
        if removed:
            self.audit.write("schedule.remove", id=id)
        return removed

    def set_enabled(self, id: int, enabled: bool, *, now: datetime | None = None) -> bool:
        row = self.get(id)
        if not row:
            return False
        if enabled and not row["enabled"]:
            # Recompute from now so a stale next_run doesn't fire immediately.
            next_run = parse_spec(row["spec"], self._now(now))
            if next_run is None:
                raise ValueError(f"{row['spec']!r} is in the past; add a new schedule instead")
            self.store.execute("UPDATE schedules SET enabled = 1, next_run = ? WHERE id = ?", (next_run.timestamp(), id))
        elif not enabled:
            self.store.execute("UPDATE schedules SET enabled = 0 WHERE id = ?", (id,))
        self.audit.write("schedule.enable" if enabled else "schedule.disable", id=id)
        return True

    async def tick(self, now: datetime | None = None) -> int:
        """Run every due schedule once, sequentially. Returns how many ran."""
        current = self._now(now)
        due = self.store.query(
            "SELECT id FROM schedules WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ? "
            "ORDER BY next_run, id",
            (current.timestamp(),),
        )
        ran = 0
        for item in due:
            try:
                row = self._claim(item["id"], current)
                if row is None:
                    continue
                self._running.add(row["id"])
                try:
                    await self._run(row, current.timestamp())
                finally:
                    self._running.discard(row["id"])
                ran += 1
            except Exception as exc:  # bookkeeping failure; keep going with the rest
                self.audit.write("schedule.error", id=item["id"], error=f"{type(exc).__name__}: {exc}")
        return ran

    async def run_forever(self, interval: float = 20) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:
                self.audit.write("schedule.error", error=f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(interval)

    def _claim(self, id: int, current: datetime) -> dict | None:
        # No awaits in here: re-check and compare-and-swap happen atomically for
        # this event loop, and the SQL guard covers other processes.
        if id in self._running:
            return None  # still running from an earlier tick
        row = self.get(id)
        now_ts = current.timestamp()
        if not row or not row["enabled"] or row["next_run"] is None or row["next_run"] > now_ts:
            return None
        try:
            one_shot = is_one_shot(row["spec"])
            step = None if one_shot else _interval(row["spec"])
            if one_shot:
                next_run = None
            elif step:
                # Keep the interval's phase; skip (don't replay) missed slots.
                next_run = row["next_run"] + step * (int((now_ts - row["next_run"]) // step) + 1)
            else:
                next_run = parse_spec(row["spec"], current).timestamp()
        except ValueError as exc:
            self.store.execute("UPDATE schedules SET enabled = 0, last_status = ? WHERE id = ?", (f"error: {exc}", id))
            self.audit.write("schedule.invalid", id=id, spec=row["spec"], error=str(exc))
            return None
        claimed = self.store.execute(
            "UPDATE schedules SET next_run = ?, enabled = ? WHERE id = ? AND enabled = 1 AND next_run = ?",
            (next_run, 0 if one_shot else 1, id, row["next_run"]),
        ).rowcount
        if not claimed:
            return None
        return {**row, "scheduled_for": row["next_run"], "next_run": next_run, "enabled": not one_shot}

    async def _run(self, row: dict, started: float) -> None:
        clock, stage = time.monotonic(), "run"
        try:
            text = await asyncio.wait_for(self.run_task(row), self.run_timeout)
            text = "" if text is None else str(text)
            if text.strip():
                stage = "delivery"
                await asyncio.wait_for(self.deliver(row, text), self.run_timeout)
            status = "ok"
        except Exception as exc:
            reason = f"timed out after {self.run_timeout}s" if isinstance(exc, asyncio.TimeoutError) else (
                str(exc) or type(exc).__name__)
            status = f"error: {reason}" if stage == "run" else f"error: delivery failed: {reason}"
        status = status[:500]
        self.store.execute("UPDATE schedules SET last_run = ?, last_status = ? WHERE id = ?", (started, status, row["id"]))
        self.audit.write("schedule.run", id=row["id"], name=row["name"], status=status,
                         seconds=round(time.monotonic() - clock, 3))
