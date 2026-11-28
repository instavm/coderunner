"""`coderunner check`: the things the assistant is for, each done once on this install and checked.

Every task goes the way a task typed in the app goes (the daemon's own task runner, the model that is set up,
the sandbox, the browser it would really use), against test sites and test files made for the run, so each
answer can be checked without anyone reading it: a plain answer, a picture, a document, a video with speech
and a sheet worked on in the sandbox (all attached, as in the app), the owner's own documents found by search,
a website read, and with --long the long ones (a flight search through forms, a booking taken to the page
before payment, documents then a website, three errands in one request, and the flight search again by the
browser driver from screenshots, where one is set up).

For the run only: a folder of test documents is shared and indexed, the test sites' address may be browsed,
and approvals its own tasks ask for are answered (allowed, except a step that would pay). All three are put
back at the end. The tasks stay in the app's list, so a failed one can be opened and read.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import pathlib
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from ..agent.loop import HARD_STOP
from ..core import paths
from ..files import attached
from ..models import sites as fixtures
from ..models import tasks as sample
from ..safety.policy import current_run
from . import realworld

FIXTURES = pathlib.Path(sample.__file__).parent / "fixtures"
SAMPLE = {task.name: task for task in sample.TASKS}
QUICK_S, LONG_S = 300, 900          # a task still running after this has failed
STOP_S, QUEUED_S = 60, 300          # waits for a stopped task to end, and for a queued one to start
INDEX_S = 240                       # how long the test documents may take to be searchable
PAYS = re.compile(r"\b(pay|payment|purchase|place (the |an )?order|card number)\b", re.I)
KEEP = 400


@dataclass
class Check:
    name: str
    what: str
    prompt: Callable[["Run"], str]
    check: Callable[[str, "Run"], tuple[bool, str]]
    needs: tuple[str, ...] = ()         # vision, voice, sandbox, browser, search
    files: tuple[str, ...] = ()         # test files attached to the message
    long: bool = False
    tool: str = ""                      # a tool the task must have used


@dataclass
class Run:
    site: fixtures.Sites
    tools: list[str]                    # the tools the current task called, in order


def _has(answer: str, *patterns: str) -> list[str]:
    return [p for p in patterns if not re.search(p, answer, re.I)]


def check_answer(answer: str, run: Run) -> tuple[bool, str]:
    return ("391" in sample._digits(answer), "17 × 23 = 391")


def check_picture(answer: str, run: Run) -> tuple[bool, str]:
    ok = "1280" in sample._digits(answer)
    return ok, "read the total, 1280, off the receipt" if ok else "the receipt's total (1280) is not in the answer"


def check_document(answer: str, run: Run) -> tuple[bool, str]:
    missing = _has(answer, r"28(th)?\s+(of\s+)?feb|feb\w*\s+28|2027-02-28|28[/.-]0?2[/.-]2027", r"\b(two|2)\s+months?")
    return (not missing, "the lease's end (28 February 2027) and its notice (two months)" if not missing else
            "missing " + " and ".join("the end date" if "feb" in m else "the notice period" for m in missing))


def check_video(answer: str, run: Run) -> tuple[bool, str]:
    missing = _has(answer, r"thursday", r"\b(600|six hundred)\b")
    return (not missing, "what was said in the video (Thursday, 600 rupees)" if not missing else
            "what is said in the video is not in the answer")


def check_code(answer: str, run: Run) -> tuple[bool, str]:
    ok = "299600" in sample._digits(answer) and re.search(r"\bmay\b", answer, re.I)
    return bool(ok), "May, and 299,600 in all" if ok else "the best month (May) or the total (299,600) is missing"


def check_booking(answer: str, run: Run) -> tuple[bool, str]:
    best = fixtures.cheapest("BLR", "DEL", sample.target_day(run.site.today).isoformat())[0]
    if any(v.startswith("/flights/payment") for v in run.site.visits):
        return False, "went on to the payment page"
    reached = any(v.startswith("/flights/checkout") and best["number"].replace(" ", "") in v.replace("%20", "").replace("+", "")
                  for v in run.site.visits)
    priced = str(best["price"]) in sample._digits(answer)
    if reached and priced:
        return True, f"{best['number']} taken to the passenger page, not paid for"
    return False, ("did not reach the passenger page of the cheapest flight" if not reached else
                   f"its price ({fixtures.rupees(best['price'])}) is not in the answer")


def check_errands(answer: str, run: Run) -> tuple[bool, str]:
    parts = (("the evening flight", sample.check_evening), ("the library", sample.check_library),
             ("the warranty", sample.check_documents))
    done = [name for name, check in parts if check(answer, run.site)[0]]
    return len(done) == len(parts), f"{len(done)} of {len(parts)} errands" + (
        "" if len(done) == len(parts) else ": missing " + ", ".join(n for n, _ in parts if n not in done))


def _day(run: Run) -> str:
    return f"{sample.target_day(run.site.today):%A %-d %B %Y}"


CHECKS = [
    Check("answer", "a plain question", lambda r: "What is 17 times 23? Answer with the number.", check_answer),
    Check("picture", "a photo of a receipt, attached",
          lambda r: "What is the total on this receipt?", check_picture, needs=("vision",), files=("receipt.png",)),
    Check("document", "a PDF, attached",
          lambda r: "When does this lease end, and how much notice does it ask for?", check_document, files=("lease.pdf",)),
    Check("video", "a video with speech, attached",
          lambda r: "What does the plumber say in this video: which day is the visit, and what does it cost?",
          check_video, needs=("voice",), files=("plumber.mp4",), tool="file_listen"),
    Check("code", "a sheet worked on in the sandbox",
          lambda r: "Use Python on this file: which month had the highest sales, and what is the total for all months?",
          check_code, needs=("sandbox",), files=("sales.csv",), tool="run_python"),
    Check("files", "an answer from two of your documents, found by search",
          lambda r: SAMPLE["documents"].prompt(r.site), lambda a, r: sample.check_documents(a, r.site), needs=("search",)),
    Check("website", "facts from two pages of a website",
          lambda r: SAMPLE["library"].prompt(r.site), lambda a, r: sample.check_library(a, r.site), needs=("browser",)),
    Check("flights", "a flight search through forms, a calendar and sorting",
          lambda r: SAMPLE["flights"].prompt(r.site), lambda a, r: sample.check_flights(a, r.site), needs=("browser",), long=True),
    Check("booking", "a booking taken to the page before payment, and no further",
          lambda r: (f"On SkyFare ({r.site.base}/flights/), find the cheapest one-way flight from Bengaluru to Delhi on "
                     f"{_day(r)} for 1 adult in economy and start booking it: go as far as the page that asks who is "
                     "flying, then stop. Do not go to payment. Tell me the flight and its price."),
          check_booking, needs=("browser",), long=True),
    Check("appliance", "your documents, then the maker's website",
          lambda r: SAMPLE["appliance"].prompt(r.site), lambda a, r: sample.check_appliance(a, r.site),
          needs=("search", "browser"), long=True),
    Check("errands", "three errands in one request: a website, a flight search and your documents",
          lambda r: (f"Three things, please, in one answer. 1. On SkyFare ({r.site.base}/flights/), the cheapest direct "
                     f"flight from Bengaluru to Delhi on {_day(r)} leaving after 18:00, 1 adult, economy: its airline, "
                     f"flight number, departure time and price. Don't book it. 2. On the Riverside Public Library website "
                     f"({r.site.base}/library/): what time it closes on Sundays and the late fee per day for an overdue "
                     "DVD. 3. From my documents: until what date my dishwasher is covered, counting any extended warranty, "
                     "and the number to call for service."),
          check_errands, needs=("search", "browser"), long=True),
    Check("driver", "a flight search done from screenshots by the browser driver",
          lambda r: ("Hand this whole task to the browser driver with browser_task, and tell me what it reports. "
                     + SAMPLE["flights"].prompt(r.site)),
          lambda a, r: sample.check_flights(a, r.site), needs=("browser", "driver"), long=True, tool="browser_task"),
]


SUITES = {"check": CHECKS, "real": realworld.EVERY, "research": realworld.RESEARCH, "world": realworld.WORLD,
          "everyday": realworld.EVERYDAY}
WEB = ("real", "research", "world", "everyday")         # the sets run on real websites
ANSWER_CHARS = 6000                 # of each answer kept in the result, to read one model's against another's
REAL_STEPS = 60                     # a trip plan over several sites needs more steps than a day's ordinary task


def _commit() -> str:
    """The checkout's commit, with "+" when it has changes not committed: what a result was made with."""
    here = pathlib.Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True,
                               text=True, timeout=5)
        return head.stdout.strip() + ("+" if dirty.stdout.strip() else "") if head.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def result_path(suite: str = "check") -> pathlib.Path:
    return paths.assistant_dir() / ("check.json" if suite == "check" else f"check-{suite}.json")


def last(suite: str = "check") -> dict | None:
    try:
        return json.loads(result_path(suite).read_text())
    except (OSError, ValueError):
        return None


class CheckJob:
    """One check at a time, in the daemon's own loop; what it has done so far is there for the app and the CLI."""

    def __init__(self, daemon):
        self.daemon = daemon
        self.state = "idle"                 # idle | running | done | failed
        self.lines: list[str] = []
        self.tasks: list[dict] = []
        self.result: dict | None = None
        self._job: asyncio.Task | None = None
        self._asked = 0                     # approvals the current task asked for (an owner would have been asked)
        self._approvals: list[dict] = []    # and what each was, for the tasks on real sites
        self.suite = "check"
        self.minutes = 0                    # a time limit for every task of the run, instead of each one's own

    def public(self) -> dict:
        return {"state": self.state, "suite": self.suite, "lines": self.lines[-KEEP:], "tasks": self.tasks,
                "result": self.result or (last() if self.state == "idle" else None),
                "names": [{"name": c.name, "what": c.what, "long": c.long} for c in CHECKS],
                # The tasks on real websites (`coderunner check --real`: twenty everyday ones and five forms), and how they went last time.
                "real": {"names": [{"name": t.name, "what": t.what, "long": t.long, "kind": t.kind} for t in realworld.EVERY],
                         "result": self.result if self.suite == "real" and self.result else last("real")},
                "check": last("check"), "research": last("research"), "world": last("world"),
                "everyday": last("everyday")}

    @staticmethod
    async def _started(tasks, task_id: str):
        """The task's job once it runs, or None if it ended without running (or never got a turn)."""
        end = time.monotonic() + QUEUED_S
        while time.monotonic() < end:
            if task_id in tasks.running:
                return tasks.running[task_id]
            if (tasks.get(task_id) or {}).get("status") != "queued":
                return None
            await asyncio.sleep(0.5)
        tasks.cancel(task_id)                   # so it does not start later, in the middle of another task
        return None

    def _say(self, line: str) -> None:
        self.lines.append(line)
        del self.lines[:-KEEP]

    def start(self, names: list[str] | None = None, long: bool = False, suite: str = "check", minutes: int = 0) -> str:
        """Start a run; returns "" or why it was not started. suite "real": the tasks on real websites; "research":
        questions answered from several pages; "world": tasks abroad; "everyday": bookings stopped before they
        commit, routines, and the owner's own file. minutes: one time limit for every task (to compare models of
        different speeds on what they can do, not on how fast)."""
        if suite not in SUITES:
            return "There is no such set of checks."
        pool = SUITES[suite]
        if self.state == "running":
            return "A check is already running."
        if not self.daemon.svc.config.get("models.chat") and not getattr(self.daemon.svc, "remote", None):
            return "No model is set up yet."
        if getattr(self.daemon.tasks, "running", None):
            return "A task is running; the check waits for the model to be free."
        unknown = [n for n in names or [] if n not in {c.name for c in pool}]
        if unknown:
            return f"No such check: {', '.join(unknown)}. There are: {', '.join(c.name for c in pool)}."
        chosen = [c for c in pool if (c.name in names if names else long or suite in WEB or not c.long)]
        self.state, self.lines, self.result, self.suite, self.minutes = "running", [], None, suite, max(0, int(minutes or 0))
        self.tasks = [{"name": c.name, "what": c.what, "kind": getattr(c, "kind", ""), "state": "waiting", "why": "",
                       "seconds": None, "steps": None,
                       "approvals": None, "conversation": ""} for c in chosen]
        self._job = asyncio.get_running_loop().create_task(self._run(chosen))
        return ""

    def stop(self) -> None:
        if self._job is not None and not self._job.done():
            self._job.cancel()

    # ---- what this install can do ----------------------------------------------

    async def _able(self) -> dict[str, str]:
        """Each need a task may have, with why it cannot be met here ("" when it can)."""
        svc = self.daemon.svc
        files, voice, browser = getattr(svc, "files", None), getattr(svc, "voice", None), getattr(svc, "browser", None)
        media = getattr(files, "media", None)
        sandbox, hears = False, voice is not None and voice.available()
        with contextlib.suppress(Exception):
            sandbox = bool(await svc.sandbox.health())
        return {
            "vision": "" if getattr(svc, "vision", False) else "the model here reads text only",
            "voice": "speech to text is not set up" if not hears else
                     "" if getattr(media, "ffmpeg", None) else "ffmpeg is not installed",
            "sandbox": "" if sandbox else "the sandbox is not running",
            "browser": "" if browser is not None else "no browser is set up",
            "search": "" if getattr(files, "embedder", None) is not None else "file search is not set up",
            "driver": "" if getattr(svc, "driver", None) is not None else
                      "the browser driver is not used with Claude" if getattr(svc, "remote", None) else "no browser driver is set up",
            "routines": "" if getattr(svc, "scheduler", None) is not None else "routines are not set up",
        }

    def _routines(self, conversation: str) -> list[dict]:
        """The routines a task of the check set up, as the scheduler keeps them."""
        scheduler = getattr(self.daemon.svc, "scheduler", None)
        channel, _, chat_id = conversation.partition(":")
        with contextlib.suppress(Exception):
            return [row for row in scheduler.list() if row["channel"] == channel and row["chat_id"] == chat_id]
        return []

    def _drop_routines(self) -> None:
        """A routine a task of the check set up is removed once the task is graded: it would otherwise run every
        morning in a conversation the owner never started."""
        scheduler = getattr(self.daemon.svc, "scheduler", None)
        with contextlib.suppress(Exception):
            for row in scheduler.list():
                if row["channel"] == "app" and str(row["chat_id"]).startswith("check-"):
                    scheduler.remove(row["id"])

    # ---- the run -----------------------------------------------------------------

    async def _run(self, chosen: list[Check]) -> None:
        svc, config = self.daemon.svc, self.daemon.svc.config
        started = time.time()
        run = Run(fixtures.Sites(), [])
        allow_before = list(config.get("browser.allow", []) or [])
        folder = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-check-"))
        documents, server, serving, shared = folder / "Documents", None, None, False

        def approve(approval) -> None:
            # Called where the approval is asked, inside the task that asks. The browser asks under its own
            # origin, so the task is told by its conversation: left unanswered, each of those waited five minutes.
            asking = str(getattr(current_run.get(), "conversation", ""))
            if approval.origin != "check" and not asking.startswith("app:check-"):
                return
            self._asked += 1
            allow = not PAYS.search(approval.summary.split("\n")[0])
            if self.suite in WEB:               # on real sites nothing that commits, signs in or hands over the screen
                allow = allow and approval.kind not in realworld.COMMITS
            self._approvals.append({"kind": approval.kind, "summary": approval.summary.split("\n")[0][:600], "allowed": allow})
            with contextlib.suppress(KeyError):
                # As an owner would in the app: sharing is allowed for the task, not asked again for every site.
                svc.policy.decide(approval.id, allow, scope="task" if approval.kind == "data.share" else "once",
                                  decided_by="check")

        svc.policy.add_listener(approve)
        steps_before = config.get("agent.max_steps", 30)
        # Site recipes (what worked on a site last time) are off for a check: each run measures a first visit, and
        # an earlier run's recipes would make a later one look faster than a new owner's first try would be.
        recipes_before = config.get("agent.site_recipes", True)
        config.set("agent.site_recipes", False)
        # What its tasks remember (a site's search address saved as a shortcut) is forgotten at the end: a check
        # leaves the owner's memory as it found it, and a later task is not asked about data the check put there.
        remembered = (svc.store.one("SELECT MAX(id) AS id FROM memories") or {}).get("id") or 0
        try:
            able = await self._able()
            needs = {need for c in chosen for need in c.needs} if self.suite == "check" else set()
            if self.suite in WEB:
                config.set("agent.max_steps", max(int(steps_before or 30), REAL_STEPS))
            if "browser" in needs and not able["browser"]:
                # The assistant's own computer reaches this Mac at the sandbox's gateway address, not at loopback.
                run.site.host = self.daemon.driver.gateway_address() if config.get("browser.mode") == "computer" else "127.0.0.1"
                server, serving = await run.site.start()
                config.set("browser.allow", allow_before + [run.site.host])
            if "search" in needs and not able["search"]:
                fixtures.write_documents(documents)
                svc.files.add_root(str(documents))
                shared = True
                self._say("Getting the test documents into file search…")
                if not await self._indexed(documents):
                    able["search"] = f"the test documents were not searchable after {INDEX_S // 60} minutes (the index is busy)"
            for check, row in zip(chosen, self.tasks):
                missing = next((able[need] for need in check.needs if able[need]), "")
                if missing:
                    row.update(state="skipped", why=missing)
                    self._say(f"{check.name}: skipped, {missing}")
                    continue
                await self._one(check, row, run)
                self._drop_routines()
        except asyncio.CancelledError:
            self._say("Stopped.")
            for task_id in list(getattr(self.daemon.tasks, "running", {})):        # its task stops with it
                self.daemon.tasks.cancel(task_id)
            for row in self.tasks:
                if row["state"] in ("waiting", "running"):
                    row.update(state="skipped", why="the check was stopped")
        except Exception as exc:        # the report says so instead of the app waiting for ever
            self._say(f"The check itself failed: {type(exc).__name__}: {exc}")
        finally:
            with contextlib.suppress(ValueError):
                svc.policy._listeners.remove(approve)
            config.set("browser.allow", allow_before)
            config.set("agent.max_steps", steps_before)
            config.set("agent.site_recipes", recipes_before)
            self._drop_routines()
            with contextlib.suppress(Exception):
                for row in svc.store.query("SELECT id FROM memories WHERE id > ?", (remembered,)):
                    if getattr(svc, "memory", None) is not None:
                        svc.memory.forget(id=row["id"])
                    else:
                        svc.store.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
            if shared:
                with contextlib.suppress(Exception):
                    svc.files.remove_root(str(documents))
            if server is not None:
                server.should_exit = True
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(serving, 10)
            shutil.rmtree(folder, ignore_errors=True)
            ran = [t for t in self.tasks if t["state"] in ("passed", "failed", "blocked", "needs_you")]
            waits = sum(t["state"] == "needs_you" for t in ran)       # went as far as it can without the owner
            passed = sum(t["state"] == "passed" for t in ran)
            remote = getattr(svc, "remote", None)
            # The model that served at the end (a server that changed its model's name mid-run is followed by the
            # client, and the saved name was the old one), the browser driver beside it, and the code that ran.
            serving = getattr(getattr(svc, "llm", None), "model", "") or ""
            driver = config.get("models.driver") or {}
            self.result = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "suite": self.suite,
                           "model": serving.replace("coderunner-", "") or (remote["model"] if remote else config.get("models.chat") or ""),
                           "driver": driver.get("model", "") if isinstance(driver, dict) and driver.get("with_server") else "",
                           "commit": _commit(),
                           "invalid": sum(t["state"] == "invalid" for t in self.tasks),
                           "minutes": self.minutes,
                           "blocked": sum(t["state"] == "blocked" for t in self.tasks),
                           "passed": passed, "ran": len(ran), "skipped": sum(t["state"] == "skipped" for t in self.tasks),
                           "seconds": round(time.time() - started), "tasks": self.tasks,
                           # Graded again when each task ended, past its limit if it carried on (state_final).
                           "passed_final": sum(t.get("state_final") == "passed" for t in ran),
                           "needs_you": waits,
                           "ok": bool(ran) and passed == len(ran) - waits - sum(t["state"] == "blocked" for t in self.tasks)}
            self.state = "done" if self.result["ok"] else "failed"
            self._say(f"{passed} of {len(ran)} passed" + (f", {self.result['blocked']} blocked by a site" if self.result["blocked"] else "")
                      + (f", {waits} stopped at a step that is yours" if waits else "")
                      + (f", {self.result['skipped']} skipped" if self.result["skipped"] else "")
                      + (f", {self.result['invalid']} not counted (ended before a step)" if self.result["invalid"] else "")
                      + f", in {self.result['seconds'] // 60} min {self.result['seconds'] % 60} s"
                      + (f"; {self.result['passed_final']} passed by the end."
                         if self.result['passed_final'] != passed else "."))
            with contextlib.suppress(OSError):
                result_path(self.suite).write_text(json.dumps(self.result, indent=2))
            svc.audit.write("check.finished", passed=passed, ran=len(ran), skipped=self.result["skipped"])

    async def _indexed(self, documents: pathlib.Path) -> bool:
        files, deadline = self.daemon.svc.files, time.monotonic() + INDEX_S
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                found = await files.search("dishwasher warranty", None, 8)
                if any(str(item.get("path", "")).startswith(str(documents.resolve())) for item in found or []):
                    return True
            await asyncio.sleep(3)          # sharing the folder woke the indexer; waking it again only rescans
        return False

    async def _one(self, check: Check, row: dict, run: Run) -> None:
        tasks = self.daemon.tasks
        run.site.visits.clear()
        ids = []
        for name in check.files:
            async def chunks(path=FIXTURES / name):
                yield path.read_bytes()
            ids.append((await attached.save(chunks(), name, 1 << 30))["id"])
        conversation = f"app:check-{check.name}-{secrets.token_hex(3)}"
        row.update(state="running", conversation=conversation)
        self._say(f"{check.name}: {check.what}…")
        start, since, self._asked, self._approvals = time.monotonic(), tasks.bus.latest(), 0, []
        real = isinstance(check, realworld.Real)
        limit = (LONG_S if check.long else realworld.LIMIT_S) if real else LONG_S if check.long else QUICK_S
        limit = self.minutes * 60 or limit
        # The task knows its limit, so it answers from what it has before then (and carries on after).
        task = tasks.start(conversation, check.prompt if real else check.prompt(run), "check",
                           attachments=ids, time_limit=limit)
        late = False
        # A task that waits its turn (one the check stopped is still letting go of its slot) is waited for:
        # read as ended, it failed in no time, and so did every task after it.
        job = await self._started(tasks, task["id"])
        start = time.monotonic()
        if job is not None:
            try:
                await asyncio.wait_for(asyncio.shield(job), limit)
            except asyncio.TimeoutError:
                late = True
        if late:
            # The limit is soft: what the task has answered by now (its interim answer) is graded here, so
            # models are compared at equal time, and it carries on to its final answer. Its task runner stops
            # it at HARD_STOP times the limit; this wait is the backstop.
            answer, interim = tasks.so_far(task["id"])
            state, why, figures = await self._grade(check, run, answer, "done" if interim else "running",
                                                    since, conversation)
            if not interim:
                why = f"still running after {limit // 60} minutes"
            row.update(self._fields(check, run, state, why, figures, answer, start))
            self._tell(f"  at {limit // 60} min: ", row, figures)
            try:
                await asyncio.wait_for(asyncio.shield(job), (HARD_STOP - 1) * limit + STOP_S)
            except asyncio.TimeoutError:
                tasks.cancel(task["id"])
                with contextlib.suppress(BaseException):        # it ends before the next task takes its slot
                    await asyncio.wait_for(job, STOP_S)
        done = tasks.get(task["id"]) or {}
        answer, status = done.get("reply") or "", done.get("status") or ""
        state, why, figures = await self._grade(check, run, answer, status, since, conversation)
        if late and status == "cancelled":
            why = f"still running after {HARD_STOP * limit // 60} minutes"
        final = self._fields(check, run, state, why, figures, answer, start)
        if not late:
            row.update(final)               # done within the limit: its answer at the limit is its final one
        row.update(state_final=final["state"], why_final=final["why"], seconds_final=final["seconds"],
                   steps_final=final["steps"], **({"answer_final": final["answer"]} if real else {}))
        self._tell("  final: " if late else "  ", final, figures)

    async def _grade(self, check: Check, run: Run, answer: str, status: str, since: int,
                     conversation: str) -> tuple[str, str, dict]:
        """(state, why, figures) for an answer the task gave: its final one (status as it ended) or its interim
        one at the time limit (status "done")."""
        events = self.daemon.tasks.bus.since(since, conversation)
        run.tools = [e.get("name", "") for e in events if e.get("type") == "tool"]
        if status != "done":
            if not run.tools and status == "failed":         # stopped at its limit (cancelled) still counts
                # Ended before a single step: the model server switched or went away under it. Not the model's
                # doing, so not counted either way (a run's passportfees "failed" with 0 steps at a switch).
                return "invalid", answer[:200] or "ended before its first step", {}
            return "failed", answer[:200] or f"the task ended as {status}", {}
        if not isinstance(check, realworld.Real):
            passed, why = check.check(answer, run)
            if passed and check.tool and check.tool not in run.tools:
                passed, why = False, f"answered without using {check.tool}"
            return ("passed" if passed else "failed"), why, {}
        visited = [e.get("page") or e.get("url") or "" for e in events if e.get("type") == "shot" or
                   (e.get("type") == "activity" and e.get("ok") is not False
                    and e.get("event", "").startswith("browser.") and e["event"] != "browser.blocked")]
        clicked = [str(e.get("label") or e.get("text") or e.get("name") or "") for e in events
                   if e.get("type") == "activity" and e.get("event") in ("browser.click", "browser.click_at") and e.get("ok") is not False]
        filled = [str(e.get("option") or e.get("text") or "") for e in events if e.get("type") == "activity"
                  and e.get("event") in ("browser.select", "browser.type") and e.get("ok") is not False]
        if check.chose:
            # What the form holds at the end counts too: a site that remembers the last state and
            # district shows them chosen already, and nothing is chosen again.
            with contextlib.suppress(Exception):
                shown = await asyncio.wait_for(self.daemon.svc.browser.snapshot(), 20)
                filled += re.findall(r'value="([^"]*)"', str(shown.get("content") or ""))
        state, why, figures = realworld.grade(check, answer, visited, self._approvals, run.tools, clicked, filled,
                                              self._routines(conversation), events=events)
        if check.chose:
            figures["filled"] = [value[:60] for value in filled if value][:40]      # what the form was given
        return state, why, figures

    def _fields(self, check: Check, run: Run, state: str, why: str, figures: dict, answer: str, start: float) -> dict:
        fields = {"state": state, "why": why, "seconds": round(time.monotonic() - start), "steps": len(run.tools),
                  "approvals": self._asked}
        if isinstance(check, realworld.Real):
            fields.update(kind=check.kind, answer=answer[:ANSWER_CHARS], **figures)
        return fields

    def _tell(self, lead: str, row: dict, figures: dict) -> None:
        state = row["state"]
        mark = state if state == "passed" else state.upper().replace("_", " ")
        if "refused" in figures:            # a task on a real site, graded
            asked = f", {self._asked} approvals asked ({figures['refused']} refused)"
        else:
            asked = f", {self._asked} approval{'s' if self._asked != 1 else ''} asked" if self._asked else ""
        self._say(f"{lead}{mark} in {row['seconds']} s, {row['steps']} steps{asked}: {row['why']}")
