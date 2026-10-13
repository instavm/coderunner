"""Sample tasks that measure a served model end to end on this machine.

The real assistant (its instructions, tools, browser and file search) does each task against the local test
sites and documents in sites.py, in a headless browser and a throwaway home folder, so nothing touches the
owner's browser, memories or files. Each answer is checked automatically. The result records, for this
machine profile and this exact model build, which tasks passed, how long each took and how many model
steps it needed; `coderunner models bench --tasks` runs it and saves it next to the speed calibration.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import pathlib
import re
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from ..core import paths
from . import sites as fixtures
from .bench import fingerprint

SUITE = 2                 # change when tasks change: results of different suites are not compared
TASK_SECONDS = 900        # a task still running after this counts as failed
COMMITTING = re.compile(r"\b(pay|book|reserve|purchase|checkout|check out|place order|confirm)", re.I)


def target_day(today: datetime.date) -> datetime.date:
    """The Friday after five weeks: a month change in the calendar, as most trips are booked."""
    day = today + datetime.timedelta(days=35)
    return day + datetime.timedelta(days=(4 - day.weekday()) % 7)


def _digits(text: str) -> str:
    """Numbers without grouping, so ₹6,214, 6214 and 6 214 all read 6214."""
    return re.sub(r"(?<=\d)[,\s.](?=\d)", "", text)


def check_flights(answer: str, site: fixtures.Sites) -> tuple[bool, str]:
    day = target_day(site.today).isoformat()
    want = fixtures.cheapest("BLR", "DEL", day)
    plain, lower = _digits(answer), answer.lower()
    missing = [f"{f['airline']} {fixtures.rupees(f['price'])}" for f in want
               if str(f["price"]) not in plain or f["airline"].lower() not in lower]
    if any(v.startswith("/flights/checkout") for v in site.visits):
        return False, "opened a booking page"
    return (not missing, "named the three cheapest fares" if not missing else "missing " + ", ".join(missing))


def check_evening(answer: str, site: fixtures.Sites) -> tuple[bool, str]:
    want = fixtures.evening("BLR", "DEL", target_day(site.today).isoformat())
    if any(v.startswith("/flights/checkout") for v in site.visits):
        return False, "opened a booking page"
    flight = want["number"].replace(" ", "") in re.sub(r"[\s-]", "", answer)
    price = str(want["price"]) in _digits(answer)
    return (flight and price, f"{want['number']} at {fixtures.rupees(want['price'])}"
            + ("" if flight and price else " not named"))


def check_library(answer: str, site: fixtures.Sites) -> tuple[bool, str]:
    closes = re.search(r"\b14[:.]00\b|\b2(?::00)?\s*(?:pm|p\.m\.)", answer, re.I)
    fee = re.search(r"(?<![\d:.])15(?![\d:])", answer)
    missing = [what for what, ok in (("Sunday closing time 14:00", closes), ("DVD fee ₹15", fee)) if not ok]
    return (not missing, "both facts" if not missing else "missing " + ", ".join(missing))


def check_documents(answer: str, site: fixtures.Sites) -> tuple[bool, str]:
    until = re.search(r"2030", answer) and re.search(r"\bmar(ch)?\b|\b0?3[/-]2030|2030-03", answer, re.I)
    phone = "18002661880" in re.sub(r"[\s-]", "", answer)
    missing = [what for what, ok in (("cover until March 2030", until), ("service line 1800 266 1880", phone)) if not ok]
    return (not missing, "both facts" if not missing else "missing " + ", ".join(missing))


def check_appliance(answer: str, site: fixtures.Sites) -> tuple[bool, str]:
    every = re.search(r"\b(3|three)[\s-]*months?\b|\bquarterly\b|\bevery quarter\b", answer, re.I)
    guide = any(v.lower().startswith("/kelvin/guide/dw-412") for v in site.visits)
    return (bool(every and guide), "every 3 months, from the DW-412 guide" if every and guide else
            "did not open the DW-412 guide" if not guide else "missing every 3 months")


@dataclass
class Task:
    name: str
    what: str                                   # what it stands for, in the report
    prompt: Callable[[fixtures.Sites], str]
    check: Callable[[str, fixtures.Sites], tuple[bool, str]]
    needs_files: bool = False


TASKS = [
    Task("flights", "flight search on a booking site (forms, suggestions, calendar, sorting)",
         lambda s: (f"Find one-way flights from Bengaluru to Delhi on {target_day(s.today):%A %-d %B %Y} for 1 adult, "
                    f"economy, on SkyFare ({s.base}/flights/). Don't book anything. Tell me the three cheapest flights "
                    "with airline, departure time and price."),
         check_flights),
    Task("evening", "a flight search with conditions (direct, leaving after 18:00)",
         lambda s: (f"On SkyFare ({s.base}/flights/), find the cheapest direct flight from Bengaluru to Delhi on "
                    f"{target_day(s.today):%A %-d %B %Y} that leaves after 18:00, for 1 adult in economy. Don't book "
                    "anything. Tell me its airline, flight number, departure time and price."),
         check_evening),
    Task("library", "facts from two pages of a website",
         lambda s: (f"On the Riverside Public Library website ({s.base}/library/), find what time the library closes "
                    "on Sundays and the late fee per day for an overdue DVD."),
         check_library),
    Task("documents", "an answer that needs two of the owner's documents",
         lambda s: ("Check my documents: until what date is my dishwasher covered, counting any extended warranty "
                    "I bought? And which number do I call for service?"),
         check_documents, needs_files=True),
    Task("appliance", "the owner's documents, then the maker's website",
         lambda s: (f"How often should I descale my dishwasher? Find its model in my documents, then check the "
                    f"maker's support site ({s.base}/kelvin/)."),
         check_appliance, needs_files=True),
]


class Meter:
    """Counts the model requests of a task: how many, what they read and wrote, and how long they took."""

    def __init__(self, llm):
        self.llm, self._chat = llm, llm.chat
        llm.chat = self.chat
        self.all: list[float] = []               # request times of every task so far
        self.reset()

    def reset(self) -> None:
        self.all.extend(r["seconds"] for r in getattr(self, "requests", []))
        self.requests: list[dict] = []
        self.steps: list[str] = []

    async def chat(self, messages, tools=None, *args, **kwargs):
        start = time.monotonic()
        reply = await self._chat(messages, tools, *args, **kwargs)
        timings, usage = reply.get("timings") or {}, reply.get("usage") or {}
        self.requests.append({"seconds": time.monotonic() - start,
                              "read": int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0),
                              "wrote": int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)})
        for call in reply.get("tool_calls") or []:
            fn = call.get("function") or {}
            self.steps.append(f"{fn.get('name')}({str(fn.get('arguments') or '')[:100]})")
        return reply

    def summary(self) -> dict:
        times = [r["seconds"] for r in self.requests]
        return {"requests": len(times), "tool_calls": len(self.steps),
                "read_tokens": sum(r["read"] for r in self.requests), "wrote_tokens": sum(r["wrote"] for r in self.requests),
                "model_s": round(sum(times), 1), "median_request_s": round(statistics.median(times), 1) if times else 0.0}


@contextlib.contextmanager
def _home(folder: pathlib.Path):
    """CODERUNNER_HOME pointed at a throwaway folder while the tasks run."""
    before = os.environ.get("CODERUNNER_HOME")
    os.environ["CODERUNNER_HOME"] = str(folder)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = before


def _config(owner, endpoint, site: fixtures.Sites, documents: pathlib.Path, visible: bool):
    from ..core.config import Config

    data = json.loads(json.dumps(owner.data))
    data["models"].update(base_url=endpoint.base_url, chat=endpoint.model, driver={})
    data["browser"].update(mode="dedicated", headless=not visible, allow=["127.0.0.1"])
    data["policy"] = {"rules": [{"kind": "browser.interact", "target": "127.0.0.1", "decision": "allow"}]}
    data["files"]["roots"] = [str(documents)]
    data["owner"]["desktop_notifications"] = False
    data.update(channels={}, connectors={"enabled": []}, remote={"provider": "", "owners": []})
    return Config(data, path=paths.config_path())


def served(config) -> dict:
    """The model being measured, down to the file: the same name can mean another quantization elsewhere."""
    from .runtimes import LlamaCppRuntime, OllamaRuntime, choose_runtime

    rt = choose_runtime(config)
    info = {"runtime": rt.name if rt else config.get("models.runtime") or "", "runtime_version": "",
            "model": config.get("models.chat") or "", "build": config.get("models.chat") or ""}
    if isinstance(rt, LlamaCppRuntime):
        active = rt._state().get("active") or {}
        info.update(model=active.get("id") or info["model"], build=pathlib.Path(active.get("model") or "").name or info["build"])
        release = getattr(rt, "RELEASE", "") if rt.name == "prism" else ""
        found = re.search(r"llama-(b\d+)", rt.binary() or "")
        info["runtime_version"] = release or (found.group(1) if found else "")
    elif isinstance(rt, OllamaRuntime):
        version = rt.version()
        info["runtime_version"] = ".".join(map(str, version)) if version else ""
    return info


async def run(config, names: list[str] | None = None, visible: bool = False,
              progress: Callable[[str], None] = print) -> dict:
    """Run the tasks with the served model; returns the result (also saved, see save())."""
    from ..agent import RunContext
    from . import hardware, manager

    tasks = [t for t in TASKS if not names or t.name in names]
    hw = hardware.detect(float(config.get("sandbox.memory_gb") or 1))
    endpoint = manager.ensure_running(config)
    if not endpoint.model or not endpoint.base_url:
        return {"ok": False, "error": "No model is configured yet; run `coderunner setup` first."}
    record = {"suite": SUITE, "date": datetime.date.today().isoformat(), "machine": fingerprint(hw),
              **served(config), "context": int(config.get("models.context") or 32768),
              "thinking": str(config.get("agent.thinking", "auto")), "tasks": []}
    site = fixtures.Sites()
    with tempfile.TemporaryDirectory(prefix="coderunner-tasks-") as tmp, _home(pathlib.Path(tmp)):
        from ..server.daemon import Daemon

        server, serving = await site.start()
        documents = pathlib.Path(tmp) / "Documents"
        fixtures.write_documents(documents)
        paths.ensure_dirs()
        svc = Daemon(_config(config, endpoint, site, documents, visible)).svc
        svc.llm.recover = None                   # a crash is a result here, not something to hide
        meter = Meter(svc.llm)
        asked: list[dict] = []

        def approve(approval) -> None:
            # Nothing here needs approval: a request is an interruption the owner would have had, and a
            # committing step is refused as the owner would.
            allow = not COMMITTING.search(approval.summary.split("\n")[0])
            asked.append({"kind": approval.kind, "summary": approval.summary[:120], "allowed": allow})
            svc.policy.decide(approval.id, allow, decided_by="bench")

        svc.policy.add_listener(approve)
        try:
            if any(t.needs_files for t in tasks):
                if svc.files.embedder is None:
                    progress("No embedding model for file search: the documents task is skipped.")
                    tasks = [t for t in tasks if not t.needs_files]
                else:
                    while (await svc.files.index_once())["more"]:
                        pass
            start = time.monotonic()
            await svc.agent.warm()
            record["warm_s"] = round(time.monotonic() - start, 1)
            for task in tasks:
                progress(f"{task.name}: {task.what}...")
                meter.reset()
                asked.clear()
                site.visits.clear()
                for row in svc.store.query("SELECT id FROM memories"):    # each task starts as the first one
                    svc.memory.forget(id=row["id"])
                start = time.monotonic()
                try:
                    answer = await asyncio.wait_for(
                        svc.agent.run(RunContext(conversation=f"bench-{task.name}", origin="bench"), task.prompt(site)),
                        TASK_SECONDS)
                    passed, why = task.check(answer, site)
                except asyncio.TimeoutError:
                    answer, passed, why = "", False, f"still running after {TASK_SECONDS // 60} min"
                result = {"name": task.name, "passed": passed, "why": why, "seconds": round(time.monotonic() - start, 1),
                          **meter.summary(), "approvals": [f"{'allowed' if a['allowed'] else 'refused'}: {a['summary']}" for a in asked],
                          "steps": meter.steps[:40], "answer": answer[:1500]}
                record["tasks"].append(result)
                progress(f"  {'passed' if passed else 'FAILED'} in {result['seconds']:.0f} s, {result['tool_calls']} tool "
                         f"calls, {result['requests']} model requests ({why})")
                await svc.browser.stop()
        finally:
            with contextlib.suppress(Exception):
                await svc.browser.stop()
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(serving, 10)
    meter.reset()
    done = record["tasks"]
    record.update(passed=sum(t["passed"] for t in done), total=len(done),
                  seconds=round(sum(t["seconds"] for t in done), 1),
                  median_request_s=round(statistics.median(meter.all), 1) if meter.all else 0.0)
    save(record)
    return {"ok": True, **record}


# ---- results --------------------------------------------------------------------------------------------

def _path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "tasks.json"


def load() -> list[dict]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return []


def save(record: dict) -> None:
    """Keep the latest result per machine, model build and runtime version."""
    key = lambda r: (json.dumps(r["machine"], sort_keys=True), r["build"], r["runtime"], r["runtime_version"], r["suite"])
    results = [r for r in load() if key(r) != key(record)] + [record]
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False))


def report(results: list[dict]) -> str:
    """A table to compare models on this machine, without answers, paths or names."""
    lines = ["| model build | runtime | passed | total time | median request | per task |", "| --- | --- | --- | --- | --- | --- |"]
    for r in sorted(results, key=lambda r: (-r["passed"], r["seconds"])):
        per = ", ".join(f"{t['name']} {'ok' if t['passed'] else 'fail'} {t['seconds']:.0f}s/{t['tool_calls']} calls"
                        for t in r["tasks"])
        lines.append(f"| {r['build']} | {r['runtime']} {r['runtime_version']} | {r['passed']}/{r['total']} | "
                     f"{r['seconds'] / 60:.1f} min | {r['median_request_s']:.1f} s | {per} |")
    return "\n".join(lines)
