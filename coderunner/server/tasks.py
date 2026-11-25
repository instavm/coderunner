"""Tasks started from the app: each runs in the background, and everything it does is published as it happens.

A task is one message to the assistant in a conversation. While it runs, the agent loop reports its tool calls,
notes and plan; the audit trail adds what the browser, the screenshot driver and the approvals did. Each event is
kept (so a reopened app shows the task as it went) and sent to every app watching the event stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time

from ..core import paths
from ..safety.policy import current_run
from . import titles

# Audit events worth showing as a task's activity, and the fields of each that the app may show.
ACTIVITY = ("browser.", "driver.", "approval.", "sandbox.", "files.", "connector.")
SHOWN_FIELDS = ("url", "page", "target", "summary", "kind", "status", "ok", "error", "reason", "text", "name",
                "ref", "key", "tab", "id", "risk", "action", "step", "thought", "completed", "label", "option",
                "direction", "submit", "goal", "report", "model", "x", "y")
FINAL = ("done", "failed", "cancelled")
# A picture of the page is kept with the task's events: a page the assistant has just opened, the page a step it
# asks approval for is on, and how the page stood at the end. The app does not put them in the thread (the live
# screen is beside it); the check reads the pages a task went to from them.
SHOT_AFTER = ("browser.open", "browser.navigate")
SHOT_SETTLE_S = 1.5         # a page just opened is still drawing
SHOT_APART_S = 20           # the same page is not shown again sooner than this
MAX_SHOTS = 8               # for one task
APPROVAL_CHARS = 20000      # of what an approval asks about (the text of a message to send)
SHOTS_KEPT_DAYS = 14


def shots_dir():
    return paths.assistant_dir() / "shots"


def by_owner(origin: str) -> bool:
    """A message the owner sent from the app (gateway.principal), not a routine's, a channel's or the check's."""
    return origin in ("owner", "admin-token") or origin.startswith("device:")


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


def _run_ids() -> dict:
    """The task an audit record is written in, when it is one of the app's (its trace reads them by it)."""
    task = getattr(getattr(current_run.get(), "events", None), "task", None)
    return {"task": task} if isinstance(task, str) and task else {}


def limit(svc) -> int:
    """How many tasks run at once: two on a local model (they take turns at it, see Turns), more where the
    model is not on this computer. agent.parallel_tasks sets it."""
    set_by_owner = svc.config.get("agent.parallel_tasks") if hasattr(svc, "config") else None
    if set_by_owner:
        return max(1, int(set_by_owner))
    away = getattr(svc, "remote", None)
    if away and away.get("provider") == "openai":
        # The owner's own server: a task in each of its slots but the last, which the step judge keeps (with two
        # slots, one task at a time; start it with -np 3 for two).
        return max(1, int(getattr(getattr(svc, "llm", None), "slots", 2) or 2) - 1)
    return 4 if away else 2


class Turns:
    """One model turn at a time, shared between the tasks that are running.

    A local model writes about twenty tokens a second for everyone: two tasks generating at once each get
    half, and take each other's place in its cache. A task's step is a model turn and then a wait that needs no
    model (a page loading, code running, the owner's answer), so tasks take turns: while one waits on its page
    the other has the model. The task the owner is looking at goes first, then whoever asked first."""

    def __init__(self):
        self.focus = ""                     # the conversation open in the app
        self.sizes: dict[str, int] = {}     # tokens each running task's conversation holds in the shared context
        self._busy = False
        self._order = 0
        self._waiting: list[tuple[int, int, asyncio.Future]] = []

    @contextlib.asynccontextmanager
    async def turn(self, conversation: str = ""):
        if self._busy:
            self._order += 1
            waiter = asyncio.get_running_loop().create_future()
            entry = (0 if conversation and conversation == self.focus else 1, self._order, waiter)
            self._waiting.append(entry)
            try:
                await waiter
            except asyncio.CancelledError:
                if entry in self._waiting:
                    self._waiting.remove(entry)
                elif waiter.done() and not waiter.cancelled():
                    self._next()            # the turn had just been handed over: pass it on
                raise
        self._busy = True
        try:
            yield
        finally:
            self._next()

    def _next(self) -> None:
        self._waiting.sort(key=lambda entry: entry[:2])
        while self._waiting:
            _, _, waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result(None)     # the turn passes straight to it: still busy
                return
        self._busy = False


class Tasks:
    def __init__(self, daemon, bus: EventBus):
        self.daemon = daemon
        self.bus = bus
        self.store = daemon.svc.store
        self.running: dict[str, asyncio.Task] = {}
        self.waiting: list[tuple] = []          # tasks asked for while as many as may run are running
        self.slots: dict[str, int] = {}         # running task -> its place (its model slot and code session)
        self._asked: dict[str, object] = {}     # approval id -> the thread of the task that asked
        self._ended: dict[str, list[asyncio.Future]] = {}   # task id -> whoever waits for it to end (wait)
        self._emits: dict[str, object] = {}     # task id -> its event sender, while it has not ended
        self._untitled: list[str] = []          # threads whose first task has ended, still to be named by the model
        self._naming: asyncio.Task | None = None
        self._named_recent = False             # the newest older threads queued for names this run
        daemon.svc.audit.listeners.append(self._on_audit)
        daemon.svc.audit.context = _run_ids
        # A daemon that stopped mid-task left it "running" for ever: its thread says how it ended.
        for row in self.store.query("SELECT id, conversation FROM tasks WHERE status IN ('running', 'queued')"):
            reply = "Stopped: the assistant was restarted while this was running."
            self.store.execute("UPDATE tasks SET status = 'cancelled', reply = ?, finished = ? WHERE id = ?",
                               (reply, time.time(), row["id"]))
            bus.publish({"task": row["id"], "conversation": row["conversation"], "type": "cancelled", "reply": reply})
        for old in shots_dir().glob("*.jpg") if shots_dir().exists() else []:
            if time.time() - old.stat().st_mtime > SHOTS_KEPT_DAYS * 86400:
                old.unlink(missing_ok=True)

    # ---- starting and stopping -------------------------------------------

    def start(self, conversation: str, prompt: str, origin: str, attachments: list[str] | None = None,
              remote: str = "", time_limit: float = 0, tainted: bool = False, routine: dict | None = None) -> dict:
        """attachments: ids of files the owner attached in the app (files/attached.py). remote: the name of the
        Anthropic model this one task runs on, when the owner asked for it (a retry of a task the local model
        did not finish); empty for the assistant's own model. time_limit: seconds from when it starts running
        (the check's limit): it posts an interim answer before then and carries on, and is stopped at
        HARD_STOP times it; 0 for none.
        routine: {"id", "name", "spec"} for a run of a routine made in this conversation, which the thread shows
        by the routine's name; tainted: its prompt was written after reading untrusted content, so the run starts
        untrusted.
        The owner's message to a thread whose task of theirs is still running (or queued) is not a task of its
        own: it goes to that task, which reads it at its next step (steer)."""
        from ..files import attached
        if by_owner(origin) and not remote and not routine and not time_limit:
            busy = self._steerable(conversation)
            if busy is not None:
                return self.steer(busy, prompt, attachments)
        task_id = secrets.token_hex(6)
        now = time.time()
        queued = len(self.running) >= self.limit()
        self.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                           "VALUES (?, ?, ?, ?, '', ?, NULL)",
                           (task_id, conversation, prompt, "queued" if queued else "running", now))
        base = {"task": task_id, "conversation": conversation}

        def emit(event: dict) -> None:
            if event.get("type") == "interim":
                emit.interim = str(event.get("text") or "")     # what a caller grades at the time limit (check)
            elif event.get("type") == "steer_taken":
                emit.taken.update(event.get("ids") or [])
            self.bus.publish({**base, **event})

        emit.task, emit.conversation = task_id, conversation
        emit.shots = {"count": 0, "page": "", "at": 0.0, "since": 0, "busy": False}
        emit.remote, emit.incomplete, emit.tainted = remote, False, tainted
        emit.time_limit = max(0.0, float(time_limit or 0))
        emit.run = None                 # the run's context, set by the agent loop once it starts
        emit.interim = ""               # its latest interim answer
        emit.origin = origin
        # The owner's messages sent while it works: all of them, the ones ready for the run (RunContext.inbox, None
        # once it has ended), and the ones it has read.
        emit.steers, emit.inbox, emit.taken = [], [], set()
        self._emits[task_id] = emit
        files = [info for info in map(attached.describe, (attachments or [])[:attached.MAX_PER_TASK]) if info]
        emit({"type": "started", "prompt": prompt, **({"attachments": files} if files else {}),
              **({"remote": remote} if remote else {}), **({"routine": routine} if routine else {}),
              **({"time_limit": emit.time_limit} if emit.time_limit else {})})
        job = (task_id, conversation, prompt, origin, emit, [info["id"] for info in files])
        if queued:
            # As many tasks as this computer runs well at once are running: this one starts when one ends.
            self.waiting.append(job)
            emit({"type": "queued", "ahead": len(self.waiting) - 1})
        else:
            self._launch(job)
        return self.get(task_id)

    def limit(self) -> int:
        return limit(self.daemon.svc)

    # ---- the owner's messages while a task works ------------------------------

    def _steerable(self, conversation: str):
        """The event sender of the owner's latest task in this thread that is running or queued and still reads
        messages, or None."""
        for emit in reversed(list(self._emits.values())):
            if emit.conversation == conversation and emit.inbox is not None and by_owner(emit.origin):
                return emit
        return None

    def steer(self, emit, prompt: str, attachments: list[str] | None = None) -> dict:
        """The owner's message to a running task: kept with its events (a reopened app shows it under the task),
        and read by the run before its next model request. One it never reads (it ended first) starts a task of
        its own when it ends (_follow_up)."""
        from ..files import attached
        files = [info for info in map(attached.describe, (attachments or [])[:attached.MAX_PER_TASK]) if info]
        item = {"id": secrets.token_hex(4), "prompt": prompt, "files": [info["id"] for info in files]}
        emit.steers.append(item)
        emit({"type": "steer", "id": item["id"], "text": prompt, **({"attachments": files} if files else {})})
        if files:
            asyncio.get_running_loop().create_task(self._post(emit, item))
        else:
            emit.inbox.append({"id": item["id"], "text": prompt})
        return {**(self.get(emit.task) or {}), "steer": item["id"]}

    async def _post(self, emit, item: dict) -> None:
        """A message with files: the model is told where each is, and shown the pictures, as with a request."""
        from ..files import attached
        svc = self.daemon.svc
        note, images = "", []
        with contextlib.suppress(Exception):
            note, images, _ = await attached.for_the_model(
                item["files"], getattr(getattr(svc, "files", None), "media", None), bool(getattr(svc, "vision", False)),
                sandbox=svc.config.get("sandbox.driver") != "none")
        if emit.inbox is not None:          # still running: else it has become a task of its own
            emit.inbox.append({"id": item["id"], "text": f"{item['prompt']}\n\n{note}".strip(),
                               **({"images": images} if images else {})})

    def _follow_up(self, emit, status: str) -> None:
        """What the owner sent that the task ended without reading: a task of its own in the thread, so nothing
        is lost. Not after a stop: Stop stops."""
        left = [item for item in emit.steers if item["id"] not in emit.taken]
        if not left or status == "cancelled":
            return
        for item in left:
            started = self.start(emit.conversation, item["prompt"], emit.origin, attachments=item["files"])
            emit({"type": "steer_moved", "id": item["id"], "to": started.get("id", "")})

    def _launch(self, job: tuple) -> None:
        task_id, emit = job[0], job[4]
        emit.slot = next(n for n in range(64) if n not in self.slots.values())
        self.slots[task_id] = emit.slot
        # Which model and place it runs on, for its trace (server/trace.py).
        llm = getattr(self.daemon.svc, "llm", None)
        with contextlib.suppress(Exception):
            self.daemon.svc.audit.write("task.running", task=task_id, slot=emit.slot,
                                        model=emit.remote or str(getattr(llm, "model", "") or
                                                                 self.daemon.svc.config.get("models.chat") or ""))
        self.running[task_id] = asyncio.create_task(self._run(*job))

    def _next(self) -> None:
        while self.waiting and len(self.running) < self.limit():
            job = self.waiting.pop(0)
            self.store.execute("UPDATE tasks SET status = 'running' WHERE id = ? AND status = 'queued'", (job[0],))
            job[4]({"type": "resumed"})
            self._launch(job)

    async def _run(self, task_id: str, conversation: str, prompt: str, origin: str, emit,
                   attachments: list[str] | None = None) -> None:
        status, reply = "done", ""
        from ..files import attached as kept
        from ..agent.loop import stopped_reply
        outputs = kept.outputs_now()            # to tell afterwards what this task made
        emit.hard = None
        if emit.time_limit:
            from ..agent.loop import HARD_STOP
            emit.deadline = time.monotonic() + emit.time_limit      # counted from when it runs, as the caller does
            # The limit is soft (the run answers before it and carries on); here it is stopped for good, and
            # answers from what it had found, as a task the owner stops.
            emit.stop_at = time.monotonic() + HARD_STOP * emit.time_limit
            loop, me = asyncio.get_running_loop(), asyncio.current_task()
            emit.hard = loop.call_later(HARD_STOP * emit.time_limit, me.cancel)

            def extend(seconds: float) -> None:
                """Time the task spent waiting for the owner (RunContext.pause): its limit moves on by as much."""
                if seconds <= 0 or emit.hard is None:
                    return
                emit.deadline += seconds
                emit.stop_at += seconds
                when = emit.hard.when() + seconds
                emit.hard.cancel()
                emit.hard = loop.call_at(when, me.cancel)
            emit.extend = extend
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
            await self._fresh_slot(emit.slot)
            reply = await self.daemon.handle_message(channel, chat_id, text, origin=origin, events=emit,
                                                     task=task_id, **({"images": images} if images else {}),
                                                     **({"remote": True} if emit.remote else {}),
                                                     **({"tainted": True} if getattr(emit, "tainted", False) else {}))
        except asyncio.CancelledError:
            # Stopped by the owner or at a caller's time limit: it answers at once with what it had found.
            status, reply = "cancelled", stopped_reply(getattr(emit, "run", None))
        except Exception as exc:        # the app shows the failure instead of waiting for ever
            # Said as a person would say it. What went wrong is in the audit trail for whoever looks.
            self.daemon.svc.audit.write("task.failed", task=task_id, error=f"{type(exc).__name__}: {exc}"[:400])
            status, reply = "failed", ("I couldn't finish this: something went wrong on my side partway through. "
                                       "Anything said above still stands. You can ask again.")
        finally:
            emit.inbox = None                   # it reads no more: what the owner sends now is a new task
            if emit.hard is not None:
                emit.hard.cancel()
                emit.hard = None
            self.running.pop(task_id, None)
            if status == "done" and emit.shots["since"]:
                await self._capture(emit, 0, again=True)        # how the page stood when it finished
            self.slots.pop(task_id, None)
            turns = getattr(self.daemon.svc, "turns", None)
            if turns is not None:
                turns.sizes.pop(conversation, None)
            await self._fresh_slot(emit.slot)
            browser = getattr(self.daemon.svc, "browser", None)
            if browser is not None and hasattr(browser, "tidy"):
                with contextlib.suppress(Exception):            # its own pages are not left to pile up
                    await asyncio.wait_for(browser.tidy(scope=conversation), 10)
            self.store.execute("UPDATE tasks SET status = ?, reply = ?, finished = ? WHERE id = ?",
                               (status, reply, time.time(), task_id))
            # An answer that says the task was not done, or a failure: the app can offer it again on Claude.
            undone = status == "failed" or (status == "done" and getattr(emit, "incomplete", False))
            emit({"type": status, "reply": reply, **({"incomplete": True} if undone else {}),
                  **({"remote": emit.remote} if emit.remote else {})})
            made = kept.produced(outputs)       # a chart, a sheet, a document its code wrote: shown under the answer
            if made:
                emit({"type": "files", "files": made})
            self._end(task_id)
            self._next()                        # a task that was waiting for a place starts now
            self._follow_up(emit, status)
            with contextlib.suppress(Exception):            # a name is an extra: never in the way of a task
                self._name_later(conversation)

    async def _fresh_slot(self, slot) -> None:
        """A task's place in the local runtime holds only the instructions when it starts and when it ends
        (LLM.reset_slot): the context is shared, so what a finished task read would crowd the others."""
        llm = getattr(self.daemon.svc, "llm", None)
        if slot is None or llm is None or getattr(llm, "remote", False) or not hasattr(llm, "reset_slot"):
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(llm.reset_slot(slot), 10)

    # ---- names in the list ----------------------------------------------------

    def _name_later(self, conversation: str) -> None:
        """Once a thread's first task has ended, the model names it (titles.py), when no task needs the model.
        A thread that already has a name, and the check's own, are left as they are."""
        if getattr(self.daemon.svc, "llm", None) is None:
            return
        if not conversation.startswith(self.CHECKS) and conversation not in self._untitled and \
                not self.store.one("SELECT 1 AS x FROM conversation_titles WHERE conversation = ?", (conversation,)):
            self._untitled = [*self._untitled, conversation][-20:]
        if self._untitled and self._model_free() and (self._naming is None or self._naming.done()):
            self._naming = asyncio.get_running_loop().create_task(self._name_untitled())

    RECENT_NAMED = 20

    def _name_recent(self, rows: list[dict]) -> None:
        """Once a run, the newest threads from before names existed are named too, one at a time while the model
        is free: they are the ones the owner sees first, and each costs one short request."""
        if self._named_recent or getattr(self.daemon.svc, "llm", None) is None:
            return
        self._named_recent = True
        named = {r["conversation"] for r in self.store.query("SELECT conversation FROM conversation_titles")}
        for row in reversed([r for r in rows if r["conversation"] not in named
                             and not r["conversation"].startswith(self.CHECKS)][:self.RECENT_NAMED]):
            self._name_later(row["conversation"])

    def _model_free(self) -> bool:
        """Whether a side question can go to the model now. Anthropic's API takes it any time; a model server
        or a local runtime only when no task is running or waiting for it: on the owner's server a request in
        any slot made a running task read its whole conversation again."""
        llm = getattr(self.daemon.svc, "llm", None)
        if llm is None or getattr(llm, "concurrent", False):
            return llm is not None
        turns = getattr(self.daemon.svc, "turns", None)
        return not self.running and not self.waiting and not getattr(turns, "_busy", False)

    def _name_slot(self, llm) -> int | None:
        """The step judge's slot (after the tasks'), so a name never lands in a task's slot."""
        if getattr(llm, "concurrent", False):
            return None
        if int(getattr(llm, "slots", 0) or 0) > 1:          # the owner's model server: its last slot
            return int(llm.slots) - 1
        decider = getattr(getattr(self.daemon.svc, "browser", None), "decider", None)
        judge = getattr(decider, "model", None)
        if getattr(judge, "slot", None) is not None and \
                str(getattr(judge, "base_url", "")).rstrip("/") == str(getattr(llm, "base_url", "")).rstrip("/"):
            return judge.slot
        return None

    async def _name_untitled(self) -> None:
        while self._untitled and self._model_free():
            conversation = self._untitled.pop()             # the newest first
            await self._name(conversation)

    async def _name(self, conversation: str) -> None:
        if self.store.one("SELECT 1 AS x FROM conversation_titles WHERE conversation = ?", (conversation,)):
            return                              # named meanwhile (another of its tasks ended while it was named)
        first =self.store.one("SELECT prompt FROM tasks WHERE conversation = ? ORDER BY created LIMIT 1",
                               (conversation,))
        prompt = str((first or {}).get("prompt") or "")
        title, source = "", "model"
        llm = getattr(self.daemon.svc, "llm", None)
        if prompt.strip() and llm is not None and hasattr(llm, "ask"):
            try:
                title = await asyncio.wait_for(titles.from_model(llm, prompt, self._name_slot(llm)), titles.TIMEOUT_S)
            except Exception as exc:        # away, too slow, an error: the name made from the request stays
                self.daemon.svc.audit.write("title.failed", conversation=conversation,
                                            error=f"{type(exc).__name__}: {exc}"[:200])
        if not title:
            title, source = titles.heuristic(prompt), "heuristic"
        self.store.execute("INSERT OR IGNORE INTO conversation_titles(conversation, title, source, created) "
                           "VALUES (?, ?, ?, ?)", (conversation, title, source, time.time()))
        if source == "model":
            # The app reads the list again on its next load and shows the new name.
            self.bus.publish({"conversation": conversation, "type": "title", "title": title})

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
            found = await self._peek(browser, getattr(emit, "conversation", None)) \
                if browser is not None and state["count"] < MAX_SHOTS else None
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

    @staticmethod
    async def _peek(browser, conversation):
        """The page this task is on (its own tab, not another task's)."""
        import inspect
        if "scope" in inspect.signature(browser.peek).parameters:
            return await browser.peek(scope=conversation)
        return await browser.peek()

    def cancel(self, task_id: str) -> bool:
        task = self.running.get(task_id)
        if task is None:
            job = next((job for job in self.waiting if job[0] == task_id), None)
            if job is None:
                return False
            self.waiting.remove(job)            # stopped before it started
            job[4].inbox = None
            from ..agent.loop import stopped_reply
            reply = stopped_reply(None)
            self.store.execute("UPDATE tasks SET status = 'cancelled', reply = ?, finished = ? WHERE id = ?",
                               (reply, time.time(), task_id))
            job[4]({"type": "cancelled", "reply": reply})
            self._end(task_id)
            return True
        task.cancel()
        return True

    def wait(self, task_id: str) -> asyncio.Future:
        """A future for the task's row once it has ended (done, failed or cancelled); at once for one that has."""
        future = asyncio.get_running_loop().create_future()
        row = self.get(task_id)
        if row is None or row["status"] in FINAL:
            future.set_result(row)
        else:
            self._ended.setdefault(task_id, []).append(future)
        return future

    def so_far(self, task_id: str) -> tuple[str, bool]:
        """What a running task has answered so far: its latest interim answer (True), or else what it would answer
        if it were stopped now (False). The check grades this at the time limit."""
        emit = self._emits.get(task_id)
        if emit is not None and getattr(emit, "interim", ""):
            return emit.interim, True
        from ..agent.loop import stopped_reply
        return stopped_reply(getattr(emit, "run", None)), False

    def _end(self, task_id: str) -> None:
        self._emits.pop(task_id, None)
        row = self.get(task_id)
        for future in self._ended.pop(task_id, []):
            if not future.done():
                future.set_result(row)

    # ---- reading -----------------------------------------------------------

    def get(self, task_id: str) -> dict | None:
        return self.store.one("SELECT * FROM tasks WHERE id = ?", (task_id,))

    def conversations(self, prefix: str = "app:", limit: int = 100) -> list[dict]:
        """Newest first, with each one's latest task and whether that task waits for the owner (an approval or
        a turn on the computer): a task asking from another thread showed "Working…" in the list."""
        rows = self._conversations(prefix, limit)
        own = [row["conversation"] for row in rows if not row["conversation"].startswith(self.CHECKS)]
        named = {r["conversation"]: r["title"] for r in self.store.query(
            f"SELECT conversation, title FROM conversation_titles WHERE conversation IN ({','.join('?' * len(own))})",
            own)} if own else {}
        with contextlib.suppress(RuntimeError):        # listed from outside the event loop: nothing queued
            asyncio.get_running_loop()
            self._name_recent(rows)
        for row in rows:
            # A short name (titles.py), the whole first request kept as prompt. The check's threads, and those
            # not named yet, are named from the request itself.
            row["title"] = named.get(row["conversation"]) or titles.heuristic(row["prompt"] or "")
            row["waiting"] = False
            if row["status"] == "running":
                asked: set = set()
                for found in self.store.query("SELECT data FROM task_events WHERE conversation = ? AND task = ? "
                                              "AND data LIKE '%\"approval.%' ORDER BY seq",
                                              (row["conversation"], row["task"])):
                    e = json.loads(found["data"])
                    if e.get("event") == "approval.requested":
                        asked.add(e.get("id"))
                    elif e.get("event") == "approval.decided":
                        asked.discard(e.get("id"))
                row["waiting"] = bool(asked)
        return rows

    def tray(self, limit: int = 12) -> dict:
        """The menu bar's view of the owner's threads (the check's left out): those that need the owner first,
        then those running or queued, then the most recent. Each has its status line (what its running task
        said last), when its latest task started and finished, the start of its answer, and whether it has used
        the assistant's computer (its browser or screen); with how many run, how many wait for the owner, and
        whether a running one uses the computer (the popover then shows its screen live)."""
        rows = [row for row in self.conversations(limit=max(limit, 30))
                if not row["conversation"].startswith(self.CHECKS)]
        rank = lambda row: 0 if row["waiting"] else 1 if row["status"] in ("running", "queued") else 2  # noqa: E731
        rows = sorted(rows, key=lambda row: (rank(row), -(row["updated"] or 0)))
        shown = rows[:limit]
        ids = [row["task"] for row in shown if row["task"]]
        found = {r["id"]: r for r in self.store.query(
            f"SELECT id, created, finished, reply FROM tasks WHERE id IN ({','.join('?' * len(ids))})", ids)} if ids else {}
        threads = []
        for row in shown:
            task = found.get(row["task"]) or {}
            line, computer = "", False
            if row["status"] == "running":
                computer = self.store.one("SELECT 1 AS used FROM task_events WHERE task = ? AND (data LIKE "
                                          "'%\"event\": \"browser.%' OR data LIKE '%\"event\": \"driver.%') LIMIT 1",
                                          (row["task"],)) is not None
            if row["status"] == "running" and not row["waiting"]:
                latest = self.store.one("SELECT data FROM task_events WHERE task = ? AND (data LIKE '%\"type\": \"note\"%' "
                                        "OR data LIKE '%\"type\": \"paused\"%' OR data LIKE '%\"type\": \"resumed\"%') "
                                        "ORDER BY seq DESC LIMIT 1", (row["task"],))
                said = json.loads(latest["data"]) if latest else {}
                if said.get("type") in ("note", "paused"):
                    line = " ".join(str(said.get("text") or "").split())
                    line = line.split(". ")[0].rstrip(".")
                    line = line if len(line) <= 90 else line[:89] + "…"
            reply = " ".join(str(task.get("reply") or "").split())
            threads.append({"conversation": row["conversation"], "title": row["title"], "status": row["status"],
                            "waiting": row["waiting"], "line": line, "task": row["task"], "computer": computer,
                            "started": task.get("created"), "finished": task.get("finished"),
                            "reply": reply if len(reply) <= 160 else reply[:159] + "…"})
        return {"threads": threads,
                "running": sum(1 for row in rows if row["status"] in ("running", "queued")),
                "waiting": sum(1 for row in rows if row["waiting"]),
                "computer": any(thread["computer"] for thread in threads)}

    CHECKS = "app:check-"

    def _conversations(self, prefix: str, limit: int) -> list[dict]:
        # The check's own threads have their own share of the list: after a few long checks they filled all of it
        # and the owner's threads were gone from the app.
        select = ("SELECT conversation, MIN(created) AS created, MAX(created) AS updated, "
                  "(SELECT prompt FROM tasks t2 WHERE t2.conversation = t.conversation ORDER BY created LIMIT 1) AS prompt, "
                  "(SELECT status FROM tasks t3 WHERE t3.conversation = t.conversation ORDER BY created DESC LIMIT 1) "
                  "AS status, (SELECT id FROM tasks t4 WHERE t4.conversation = t.conversation ORDER BY created DESC "
                  "LIMIT 1) AS task FROM tasks t WHERE conversation LIKE ? ESCAPE '\\' {} "
                  "GROUP BY conversation ORDER BY updated DESC LIMIT ?")
        like = lambda text: text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"  # noqa: E731
        own = self.store.query(select.format("AND conversation NOT LIKE ? ESCAPE '\\'"),
                               (like(prefix), like(self.CHECKS), limit))
        checks = self.store.query(select.format(""), (like(self.CHECKS), limit)) \
            if self.CHECKS.startswith(prefix) else []
        return sorted(own + checks, key=lambda row: row["updated"], reverse=True)

    def tasks(self, conversation: str) -> list[dict]:
        return self.store.query("SELECT * FROM tasks WHERE conversation = ? ORDER BY created", (conversation,))

    # ---- activity from the audit trail -------------------------------------

    def _on_audit(self, record: dict) -> None:
        ctx = current_run.get()
        emit = getattr(ctx, "events", None)
        event = record.get("event", "")
        if event == "approval.decided" and emit is None:
            # Answered from the app (or timed out elsewhere): outside the task that asked, so it is put into
            # that task's thread here. The thread then says how each approval ended when it is opened later.
            emit = self._asked.pop(record.get("id", ""), None)
        if emit is None or not event.startswith(ACTIVITY):
            return
        if event == "approval.requested" and record.get("id"):
            self._asked[record["id"]] = emit
            for old in list(self._asked)[:-200]:
                del self._asked[old]
        fields = {k: v for k, v in record.items() if k in SHOWN_FIELDS and v not in (None, "")}
        # What an approval asks about reaches the app whole: a message to be sent, a setting to be changed, must
        # be read in full before it is allowed, never cut at 300 characters.
        whole = "summary" if event == "approval.requested" else ""
        for key, value in fields.items():
            if key == whole:
                fields[key] = value[:APPROVAL_CHARS] if isinstance(value, str) else value
            elif isinstance(value, str) and len(value) > 300:
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
