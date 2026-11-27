"""A task's journal, so the task goes on after the daemon restarts (an update, a crash, the Mac sleeping).

What a task was started with is written when it is asked for; its conversation as the model last saw it, with the
run's own state (plan, steps, time left, what it has read), after each model reply and after each step's results:
one row, replaced. Each tool call is written before it runs and again with its result, so a restart in the middle
of a call is known. On the next start the task goes on from there (server.tasks): a reply the model was writing is
asked for again; a call that had no result runs again only if the tool is replay-safe (it only reads), and
otherwise the model is told it was interrupted and to look before repeating it. Writing the journal never stops a
task: a failure to write is only a restart that goes back further.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

RESTARTS_ALLOWED = 3        # resumed this many times without finishing, a task ends as failed
INTERRUPTED = ("Error: the assistant restarted while this step was running, so it is not known whether it took "
               "effect. Check the current state (look at the page, the file or the result) before repeating it.")
# Said to the model once it goes on, when it had used the browser: another task or the restart may have moved it.
BROWSER_MOVED = ("[Not a new request] The assistant restarted while you worked on this. The browser may not be where "
                 "you left it: look at the page again before acting on it.")
# The run's own state kept with its conversation: sets as lists, the long lists cut to their latest entries.
CTX_FIELDS = {"tainted": None, "content_tainted": None, "private_sources": set, "shared": set, "plan": None,
              "shown": None, "told": None, "found": None, "said": 60, "read": 40, "unanswered": None, "known": set,
              "found_files": 100, "interim": None, "trail": 100, "sites_noted": set, "heard": None}


def ctx_state(ctx) -> dict:
    state = {}
    for name, kind in CTX_FIELDS.items():
        value = getattr(ctx, name, None)
        if kind is set:
            value = sorted(value or ())
        elif isinstance(kind, int) and isinstance(value, list):
            value = value[-kind:]
        state[name] = value
    return state


def restore_ctx(ctx, state: dict) -> None:
    for name, kind in CTX_FIELDS.items():
        if name in (state or {}) and state[name] is not None:
            setattr(ctx, name, set(state[name]) if kind is set else state[name])


def portable(messages: list[dict]) -> list[dict]:
    """The conversation as kept: pictures left out (the run drops all but the latest anyway, and they are large)."""
    return [{**m, "content": "[earlier screenshot removed]"} if isinstance(m.get("content"), list) else m
            for m in messages]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Journal:
    def __init__(self, store, task: str):
        self.store, self.task = store, task
        self.current: tuple[int, int] | None = None     # the call running now: (step, place in the step)

    def _write(self, sql: str, params: tuple) -> None:
        with contextlib.suppress(Exception):
            self.store.execute(sql, params)

    # ---- the task -------------------------------------------------------------

    def begin(self, job: dict) -> None:
        self._write("INSERT OR REPLACE INTO task_journal(task, job, state, restarts, stopped, updated) "
                    "VALUES (?, ?, NULL, 0, 0, ?)", (self.task, _dump(job), time.time()))

    def row(self) -> dict | None:
        try:
            row = self.store.one("SELECT * FROM task_journal WHERE task = ?", (self.task,))
        except Exception:
            return None
        if row is None:
            return None
        for key in ("job", "state"):
            try:
                row[key] = json.loads(row[key]) if row[key] else None
            except ValueError:
                row[key] = None
        row["job"] = row["job"] or {}
        return row

    def update_job(self, **changes) -> None:
        row = self.row()
        if row is not None:
            self._write("UPDATE task_journal SET job = ? WHERE task = ?", (_dump({**row["job"], **changes}), self.task))

    def state(self) -> dict | None:
        row = self.row()
        return row["state"] if row else None

    def restarted(self) -> int:
        """One more restart, and how many there have been."""
        self._write("UPDATE task_journal SET restarts = restarts + 1 WHERE task = ?", (self.task,))
        return int((self.row() or {}).get("restarts") or 0)

    def stop(self) -> None:
        """The owner stopped it: never resumed, even if the daemon goes before the task has ended."""
        self._write("UPDATE task_journal SET stopped = 1 WHERE task = ?", (self.task,))

    def end(self) -> None:
        self._write("DELETE FROM task_calls WHERE task = ?", (self.task,))
        self._write("DELETE FROM task_journal WHERE task = ?", (self.task,))

    # ---- the run ----------------------------------------------------------------

    def save(self, state: dict) -> None:
        """The run's conversation and state, replacing the last; the calls of steps before it are let go."""
        self._write("UPDATE task_journal SET state = ?, updated = ? WHERE task = ?",
                    (_dump(state), time.time(), self.task))
        self._write("DELETE FROM task_calls WHERE task = ? AND step < ?", (self.task, int(state.get("acts") or 0)))

    def intent(self, step: int, pos: int, name: str) -> None:
        self.current = (step, pos)
        self._write("INSERT OR REPLACE INTO task_calls(task, step, pos, name, result, flags, approval, at) "
                    "VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)", (self.task, step, pos, name, time.time()))

    def asking(self, approval: str | None) -> None:
        """The running call waits for the owner's approval (its id), or not any more (None): a call that was
        waiting when the daemon went did nothing yet, and is run again so the owner is asked again."""
        if self.current is not None:
            self._write("UPDATE task_calls SET approval = ? WHERE task = ? AND step = ? AND pos = ? AND result IS NULL",
                        (approval, self.task, *self.current))

    def result(self, step: int, pos: int, name: str, result: str, flags: dict) -> None:
        self.current = None
        self._write("INSERT OR REPLACE INTO task_calls(task, step, pos, name, result, flags, approval, at) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)", (self.task, step, pos, name, result, _dump(flags), time.time()))

    def calls(self, step: int) -> dict[int, dict]:
        try:
            rows = self.store.query("SELECT * FROM task_calls WHERE task = ? AND step = ?", (self.task, step))
        except Exception:
            return {}
        for row in rows:
            try:
                row["flags"] = json.loads(row["flags"]) if row["flags"] else None
            except ValueError:
                row["flags"] = None
        return {int(row["pos"]): row for row in rows}
