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

from ..core import paths
from ..safety.policy import current_run

# Audit events worth showing as a task's activity, and the fields of each that the app may show.
ACTIVITY = ("browser.", "driver.", "approval.", "sandbox.", "files.", "connector.")
SHOWN_FIELDS = ("url", "page", "target", "summary", "kind", "status", "ok", "error", "reason", "text", "name",
                "ref", "key", "tab", "id", "risk", "action", "step", "thought", "completed", "label", "option",
                "direction", "submit", "goal", "report", "model", "x", "y")
FINAL = ("done", "failed", "cancelled")
# A picture of the page goes into the task's thread where it tells the owner something words do not: a page the
# assistant has just opened, the page a step it asks approval for is on, and how the page stood at the end.
SHOT_AFTER = ("browser.open", "browser.navigate")
SHOT_SETTLE_S = 1.5         # a page just opened is still drawing
SHOT_APART_S = 20           # the same page is not shown again sooner than this
MAX_SHOTS = 8               # for one task
SHOTS_KEPT_DAYS = 14


def shots_dir():
    return paths.assistant_dir() / "shots"


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
        # A daemon that stopped mid-task left it "running" for ever: its thread says how it ended.
        for row in self.store.query("SELECT id, conversation FROM tasks WHERE status = 'running'"):
            reply = "Stopped: the assistant was restarted while this was running."
            self.store.execute("UPDATE tasks SET status = 'cancelled', reply = ?, finished = ? WHERE id = ?",
                               (reply, time.time(), row["id"]))
            bus.publish({"task": row["id"], "conversation": row["conversation"], "type": "cancelled", "reply": reply})
        for old in shots_dir().glob("*.jpg") if shots_dir().exists() else []:
            if time.time() - old.stat().st_mtime > SHOTS_KEPT_DAYS * 86400:
                old.unlink(missing_ok=True)

    # ---- starting and stopping -------------------------------------------

    def start(self, conversation: str, prompt: str, origin: str, attachments: list[str] | None = None) -> dict:
        """attachments: ids of files the owner attached in the app (files/attached.py)."""
        from ..files import attached
        task_id = secrets.token_hex(6)
        now = time.time()
        self.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                           "VALUES (?, ?, ?, 'running', '', ?, NULL)", (task_id, conversation, prompt, now))
        base = {"task": task_id, "conversation": conversation}

        def emit(event: dict) -> None:
            self.bus.publish({**base, **event})

        emit.task = task_id
        emit.shots = {"count": 0, "page": "", "at": 0.0, "since": 0, "busy": False}
        files = [info for info in map(attached.describe, (attachments or [])[:attached.MAX_PER_TASK]) if info]
        emit({"type": "started", "prompt": prompt, **({"attachments": files} if files else {})})
        self.running[task_id] = asyncio.create_task(
            self._run(task_id, conversation, prompt, origin, emit, [info["id"] for info in files]))
        return self.get(task_id)

    async def _run(self, task_id: str, conversation: str, prompt: str, origin: str, emit,
                   attachments: list[str] | None = None) -> None:
        status, reply = "done", ""
        from ..files import attached as kept
        outputs = kept.outputs_now()            # to tell afterwards what this task made
        try:
            channel, _, chat_id = conversation.partition(":")
            text, images = prompt, None
            if attachments:
                # The model is told where each file is, and shown the pictures among them straight away.
                from ..files import attached
                svc = self.daemon.svc
                note, images, _ = await attached.for_the_model(
                    attachments, getattr(getattr(svc, "files", None), "media", None), bool(getattr(svc, "vision", False)),
                    sandbox=svc.config.get("sandbox.driver") != "none")
                text = f"{prompt}\n\n{note}".strip() if note else prompt
            reply = await self.daemon.handle_message(channel, chat_id, text, origin=origin, events=emit,
                                                     task=task_id, **({"images": images} if images else {}))
        except asyncio.CancelledError:
            status, reply = "cancelled", "Stopped."
        except Exception as exc:        # the app shows the failure instead of waiting for ever
            status, reply = "failed", f"The task failed: {exc}"
        finally:
            self.running.pop(task_id, None)
            if status == "done" and emit.shots["since"]:
                await self._capture(emit, 0, again=True)        # how the page stood when it finished
            self.store.execute("UPDATE tasks SET status = ?, reply = ?, finished = ? WHERE id = ?",
                               (status, reply, time.time(), task_id))
            emit({"type": status, "reply": reply})
            made = kept.produced(outputs)       # a chart, a sheet, a document its code wrote: shown under the answer
            if made:
                emit({"type": "files", "files": made})

    # ---- pictures of the page ----------------------------------------------

    def _shoot(self, emit, delay: float, again: bool = False) -> None:
        state = getattr(emit, "shots", None)
        if state is None or state["busy"] or state["count"] >= MAX_SHOTS:
            return
        state["busy"] = True
        try:
            asyncio.get_running_loop().create_task(self._capture(emit, delay, again, held=True))
        except RuntimeError:            # an audit record written off the event loop: no picture for that one
            state["busy"] = False

    async def _capture(self, emit, delay: float, again: bool = False, held: bool = False) -> None:
        state = emit.shots
        try:
            await asyncio.sleep(delay)
            browser = getattr(self.daemon.svc, "browser", None)
            found = await browser.peek() if browser is not None and state["count"] < MAX_SHOTS else None
            if not found:
                return
            image, page = found
            if not again and page == state["page"] and time.time() - state["at"] < SHOT_APART_S:
                return
            state.update(count=state["count"] + 1, page=page, at=time.time(), since=0)
            name = f"{emit.task}-{state['count']}.jpg"
            shots_dir().mkdir(parents=True, exist_ok=True)
            (shots_dir() / name).write_bytes(image)
            emit({"type": "shot", "file": name, "page": page})
        except Exception:               # a picture is an extra: the task goes on without it
            pass
        finally:
            if held:
                state["busy"] = False

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
        state = getattr(emit, "shots", None)
        if state is None:
            return
        if event.startswith("browser."):
            state["since"] += 1
        if event in SHOT_AFTER and record.get("ok", True) is not False:
            self._shoot(emit, SHOT_SETTLE_S)
        elif event == "approval.requested" and str(record.get("kind", "")).startswith("browser."):
            self._shoot(emit, 0, again=True)        # the page the owner is asked about, as it is now
