"""Tasks started from the app: each runs in the background, and everything it does is published as it happens.

A task is one message to the assistant in a conversation. While it runs, the agent loop reports its tool calls,
notes and plan; the audit trail adds what the browser, the screenshot driver and the approvals did. Each event is
kept (so a reopened app shows the task as it went) and sent to every app watching the event stream.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import Any

from ..safety.policy import current_run

# Audit events worth showing as a task's activity, and the fields of each that the app may show.
ACTIVITY = ("browser.", "driver.", "approval.", "sandbox.", "files.", "connector.")
SHOWN_FIELDS = ("url", "page", "target", "summary", "kind", "status", "ok", "error", "reason", "text", "name",
                "ref", "key", "tab", "id", "risk", "action", "step", "thought", "completed", "label", "option",
                "direction", "submit", "goal", "report", "model", "x", "y")
FINAL = ("done", "failed", "cancelled")


class EventBus:
    def __init__(self, store):
        self.store = store
        self.subscribers: set[asyncio.Queue] = set()

    def publish(self, event: dict) -> dict:
        event = {"ts": time.time(), **event}
        cursor = self.store.execute("INSERT INTO task_events(task, conversation, ts, data) VALUES (?, ?, ?, ?)",
                                    (event.get("task", ""), event.get("conversation", ""), event["ts"],
                                     json.dumps(event, default=str)))
        event["seq"] = cursor.lastrowid
        for queue in list(self.subscribers):
            if queue.qsize() < 2000:            # a stalled reader loses events rather than holding memory
                queue.put_nowait(event)
        return event

    def latest(self) -> int:
        row = self.store.one("SELECT MAX(seq) AS seq FROM task_events")
        return int((row or {}).get("seq") or 0)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    def since(self, seq: int, conversation: str | None = None, limit: int = 2000) -> list[dict]:
        if conversation:
            rows = self.store.query("SELECT seq, data FROM task_events WHERE conversation = ? AND seq > ? "
                                    "ORDER BY seq LIMIT ?", (conversation, seq, limit))
        else:
            rows = self.store.query("SELECT seq, data FROM task_events WHERE seq > ? ORDER BY seq LIMIT ?",
                                    (seq, limit))
        return [{**json.loads(r["data"]), "seq": r["seq"]} for r in rows]


class Tasks:
    def __init__(self, daemon, bus: EventBus):
        self.daemon = daemon
        self.bus = bus
        self.store = daemon.svc.store
        self.running: dict[str, asyncio.Task] = {}
        daemon.svc.audit.listeners.append(self._on_audit)
        # A daemon that stopped mid-task left it "running" for ever.
        self.store.execute("UPDATE tasks SET status = 'cancelled', finished = ? WHERE status = 'running'",
                           (time.time(),))

    # ---- starting and stopping -------------------------------------------

    def start(self, conversation: str, prompt: str, origin: str) -> dict:
        task_id = secrets.token_hex(6)
        now = time.time()
        self.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                           "VALUES (?, ?, ?, 'running', '', ?, NULL)", (task_id, conversation, prompt, now))
        base = {"task": task_id, "conversation": conversation}

        def emit(event: dict) -> None:
            self.bus.publish({**base, **event})

        emit({"type": "started", "prompt": prompt})
        self.running[task_id] = asyncio.create_task(self._run(task_id, conversation, prompt, origin, emit))
        return self.get(task_id)

    async def _run(self, task_id: str, conversation: str, prompt: str, origin: str, emit) -> None:
        status, reply = "done", ""
        try:
            channel, _, chat_id = conversation.partition(":")
            reply = await self.daemon.handle_message(channel, chat_id, prompt, origin=origin, events=emit,
                                                     task=task_id)
        except asyncio.CancelledError:
            status, reply = "cancelled", "Stopped."
        except Exception as exc:        # the app shows the failure instead of waiting for ever
            status, reply = "failed", f"The task failed: {exc}"
        finally:
            self.running.pop(task_id, None)
            self.store.execute("UPDATE tasks SET status = ?, reply = ?, finished = ? WHERE id = ?",
                               (status, reply, time.time(), task_id))
            emit({"type": status, "reply": reply})

    def cancel(self, task_id: str) -> bool:
        task = self.running.get(task_id)
        if task is None:
            return False
        task.cancel()
        return True

    # ---- reading -----------------------------------------------------------

    def get(self, task_id: str) -> dict | None:
        return self.store.one("SELECT * FROM tasks WHERE id = ?", (task_id,))

    def conversations(self, prefix: str = "app:", limit: int = 100) -> list[dict]:
        return self.store.query(
            "SELECT conversation, MIN(created) AS created, MAX(created) AS updated, "
            "(SELECT prompt FROM tasks t2 WHERE t2.conversation = t.conversation ORDER BY created LIMIT 1) AS title, "
            "(SELECT status FROM tasks t3 WHERE t3.conversation = t.conversation ORDER BY created DESC LIMIT 1) "
            "AS status FROM tasks t WHERE conversation LIKE ? GROUP BY conversation ORDER BY updated DESC LIMIT ?",
            (prefix + "%", limit))

    def tasks(self, conversation: str) -> list[dict]:
        return self.store.query("SELECT * FROM tasks WHERE conversation = ? ORDER BY created", (conversation,))

    # ---- activity from the audit trail -------------------------------------

    def _on_audit(self, record: dict) -> None:
        ctx = current_run.get()
        emit = getattr(ctx, "events", None)
        event = record.get("event", "")
        if emit is None or not event.startswith(ACTIVITY):
            return
        fields = {k: v for k, v in record.items() if k in SHOWN_FIELDS and v not in (None, "")}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > 300:
                fields[key] = value[:300] + "…"
            elif isinstance(value, (dict, list)):
                fields[key] = json.dumps(value, default=str)[:300]
        emit({"type": "activity", "event": event, **fields})
