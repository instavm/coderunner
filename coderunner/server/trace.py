"""A thread's trace for the developers: what each of its tasks was asked, planned, did and answered, in time order.

A test user in developer mode exports it from the app (GET /api/conversations/{id}/trace) and sends it. It is made
from the task's events (tasks.py), the audit trail (core/audit.py, read by task id or, for older records, by time)
and the tasks table, with a header saying what it ran on. Markdown by default, so anyone can open it; ?format=json
for the same as data, ?format=zip for both.

Everything in it is scrubbed before it is written (scrub): keys, tokens, cookies, passwords and codes; email
addresses, phone, card and account numbers; IP addresses and the model server's address; the home folder's user
name. Page texts and answers stay, cut short.
"""

from __future__ import annotations

import datetime
import getpass
import io
import json
import os
import pathlib
import platform
import re
import time
import zipfile
from typing import Any
from urllib.parse import urlsplit

from ..connectors.redact import redact as mail_redact
from ..core import audit as audit_module

FORMAT = "coderunner-trace/1"
ABOUT = ("CodeRunner trace: what the assistant saw and did in one thread, for debugging. Keys, tokens, passwords, "
         "codes, email addresses, phone, card and account numbers, IP and server addresses and home folder names "
         "were removed; page texts and answers are cut short. Check it before sending.")

ARGS_CHARS = 1000
RESULT_CHARS = 2000
TEXT_CHARS = 2000           # a page's text, an approval's summary, an interim answer
ANSWER_CHARS = 8000
LINE_CHARS = 500            # a status line, a plan step, an error
SLACK_S = 5                 # audit records written just after a task's row says it finished

# Audit records the trace takes (the task's events already have the browser's and the approvals' as activity).
AUDIT_TAKEN = ("agent.", "task.", "decision.", "guard.", "egress.", "browser.handoff", "browser.handed_back",
               "browser.owner_control", "approval.")
# Records about the whole computer, not one task: taken by time alone.
GLOBAL = ("browser.owner_control",)


# ---- scrubbing -------------------------------------------------------------------------------------------------

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
KEY_SHAPES = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[abpors]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{20,}|"
    r"glpat-[A-Za-z0-9_-]{16,}|tskey-[A-Za-z0-9-]{16,}|(?:pk|sk|rk)_(?:live|test)_[A-Za-z0-9]{16,})")
BEARER = re.compile(r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{12,}")
NAMED = re.compile(r"(?i)\b((?:api[_-]?key|access[_-]?key|secret|token|passw(?:or)?d|pwd|passcode|cookie|session[_-]?id|"
                   r"auth(?:orization)?|otp|pin)\b[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&]{3,})")
QUERY = re.compile(r"(?i)([?&](?:[\w-]*(?:token|key|secret|sig|signature|password|passwd|auth|session|code|otp)[\w-]*)=)"
                   r"[^&#\s\"'<>]+")
IPV4 = re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])")
IPV6 = re.compile(r"(?i)(?<![\w:])(?:(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}|(?:[0-9a-f]{1,4}:){1,6}:(?:[0-9a-f]{1,4}"
                  r"(?::[0-9a-f]{1,4}){0,5})?|::1)(?![\w:])")
LOCAL_URL = re.compile(r"(?i)\b(?:https?|wss?)://(?:localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0|[\w-]+\.local)(?::\d+)?")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
DIGITS = re.compile(r"(?<![\w.,/-])\+?\d(?:[ ().-]{0,2}\d){8,22}(?![\w/-]|[.,]\d)")
# A field named for a secret is dropped whole (as the audit trail does, core/audit.py); "step_tokens" is a count.
SENSITIVE_KEY = re.compile(r"(?i)secret|passw|authorization|cookie|api_?key|otp|token(?!s)|"
                           + "|".join(re.escape(word) for word in audit_module._SENSITIVE if word != "token"))
OWN_FIELDS = {"about", "format", "at", "exported", "started", "finished"}      # written here: times, not content
HOME = re.compile(r"(?<![\w.])(?:/Users|/home|C:\\Users)[/\\]([^/\\\s\"'<>]+)")


def _luhn(digits: str) -> bool:
    total, odd = 0, False
    for char in reversed(digits):
        n = int(char)
        if odd:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        odd = not odd
    return total % 10 == 0


def _number(match: re.Match) -> str:
    text = match.group(0)
    digits = re.sub(r"\D", "", text)
    if not 9 <= len(digits) <= 19:
        return text
    if re.fullmatch(r"\d{4}[-./]\d{2}[-./]\d{2}.*", text.lstrip("+")):     # a date, maybe with a time
        return text
    if 13 <= len(digits) <= 19 and _luhn(digits):
        return "[card number]"
    return "[number]"


class Scrubber:
    """Removes what must not leave the computer from every string in a trace. literals: values known to be secret
    here (the admin token, keys, the owner's name), servers: addresses of the model servers and the sandbox."""

    def __init__(self, literals: list[tuple[str, str]] | None = None, servers: list[str] | None = None):
        self.literals = sorted({(value, mask) for value, mask in (literals or []) if value and len(value) >= 3},
                               key=lambda pair: -len(pair[0]))
        self.servers = sorted({s for s in (servers or []) if s and len(s) >= 4}, key=len, reverse=True)
        user = ""
        try:
            user = getpass.getuser()
        except Exception:
            pass
        self.user = user if len(user) >= 3 else ""
        self.home = str(pathlib.Path.home())

    def text(self, value: str) -> str:
        if not value:
            return value
        # First what the mail connectors already mask: one-time codes near a code word, sign-in and reset links
        # whole, long random tokens.
        text = mail_redact(value)
        for literal, mask in self.literals:
            text = text.replace(literal, mask)
        for server in self.servers:
            text = text.replace(server, "[model server]")
        if self.home and len(self.home) > 1:
            text = text.replace(self.home, "~")
        text = HOME.sub("~", text)
        if self.user:
            text = re.sub(rf"(?<![\w.]){re.escape(self.user)}(?![\w])", "[user]", text)
        text = LOCAL_URL.sub("[local server]", text)
        text = JWT.sub("[redacted token]", text)
        text = KEY_SHAPES.sub("[redacted key]", text)
        text = BEARER.sub(lambda m: f"{m.group(1)} [redacted]", text)
        text = QUERY.sub(lambda m: m.group(1) + "[redacted]", text)
        text = NAMED.sub(lambda m: m.group(1) + "[redacted]", text)
        text = EMAIL.sub("[email]", text)
        text = IPV4.sub("[ip]", text)
        text = IPV6.sub("[ip]", text)
        text = IBAN.sub(lambda m: "[account number]" if sum(c.isdigit() for c in m.group(0)) >= 10 else m.group(0), text)
        return DIGITS.sub(_number, text)

    def value(self, value: Any, key: str = "") -> Any:
        if key and SENSITIVE_KEY.search(key):
            return "[redacted]"
        if key in OWN_FIELDS:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.value(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v) for v in value]
        return value


def scrubber_for(svc) -> Scrubber:
    config = getattr(svc, "config", None)
    get = (lambda key, default=None: config.get(key, default)) if config is not None else (lambda key, default=None: default)
    literals: list[tuple[str, str]] = []
    for name in ("admin_token", "api_token"):
        try:
            literals.append((str(getattr(svc, name)() or ""), "[redacted token]"))
        except Exception:
            pass
    literals.append((str(getattr(svc, "llm_api_key", "") or ""), "[redacted key]"))
    literals.append((str(get("models.api_key") or ""), "[redacted key]"))
    remote = get("models.remote") or {}
    if isinstance(remote, dict):
        literals += [(str(v), "[redacted key]") for k, v in remote.items() if "key" in str(k).lower() and v]
    try:
        from ..models import remote as remote_models
        literals.append((remote_models.key(), "[redacted key]"))
    except Exception:
        pass
    store = getattr(svc, "secrets", None)
    if store is not None:
        try:
            literals += [(str(store.get(name) or ""), "[redacted secret]") for name in store.names()]
        except Exception:
            pass
    owner = str(get("owner.name") or "").strip()
    if owner:
        literals.append((owner, "[owner]"))
        literals += [(part, "[owner]") for part in owner.split() if len(part) >= 3]
    servers: list[str] = []
    urls = [get("models.base_url"), get("models.embedding_url"), get("models.embedding_media_url"),
            get("sandbox.url"), (remote or {}).get("base_url") if isinstance(remote, dict) else "",
            getattr(getattr(svc, "llm", None), "base_url", ""),
            getattr(getattr(svc, "sandbox", None), "base_url", ""),
            (getattr(svc, "remote", None) or {}).get("base_url", "")]
    for url in urls:
        if not url or not isinstance(url, str):
            continue
        servers.append(url.rstrip("/"))
        host = urlsplit(url).netloc
        if host:
            servers.append(host)
    return Scrubber(literals, servers)


# ---- reading ---------------------------------------------------------------------------------------------------

def _audit_between(path, start: float, end: float) -> list[dict]:
    """Audit records written between start and end. The file is in time order: the first one is found by halving
    (a long-used audit file is tens of megabytes), then read on until end."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    found: list[dict] = []
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        low, high = 0, size
        while high - low > 4096:
            middle = (low + high) // 2
            handle.seek(middle)
            handle.readline()                       # to the start of the next record
            line = handle.readline()
            ts = _ts(line)
            if ts is None or ts >= start:
                high = middle
            else:
                low = middle
        handle.seek(low)
        if low:
            handle.readline()
        for line in handle:
            ts = _ts(line)
            if ts is None:
                continue
            if ts > end:
                break
            if ts >= start:
                try:
                    found.append(json.loads(line))
                except ValueError:
                    continue
    return found


def _ts(line: bytes) -> float | None:
    match = re.match(rb'\{"ts": ([0-9.]+)', line or b"")
    return float(match.group(1)) if match else None


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def _cut(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more characters]"


def _args(raw: Any) -> Any:
    """A tool call's arguments as data where they parse, each long value cut."""
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return _cut(raw, ARGS_CHARS)
    if isinstance(value, dict):
        out = {k: (_cut(v, ARGS_CHARS // 2) if isinstance(v, (str, list, dict)) and len(json.dumps(v, default=str)) > ARGS_CHARS // 2
                   else v) for k, v in value.items()}
        return out
    return _cut(value, ARGS_CHARS)


def header(svc, daemon=None) -> dict:
    config = getattr(svc, "config", None)
    get = (lambda key, default=None: config.get(key, default)) if config is not None else (lambda key, default=None: default)
    from .. import __version__
    try:
        from .check import _commit
        commit = _commit()
    except Exception:
        commit = ""
    system = platform.system()
    os_name = (f"macOS {platform.mac_ver()[0]}" if system == "Darwin" else f"{system} {platform.release()}").strip()
    remote = getattr(svc, "remote", None) or {}
    llm = getattr(svc, "llm", None)

    def name(value: Any) -> str:
        text = str(value or "")
        return pathlib.PurePath(text).name if text.startswith(("/", "~", ".")) or "\\" in text else text
    driver = get("models.driver") or {}
    models = {"chat": name(remote.get("model") or get("models.chat") or getattr(llm, "model", "")),
              "local_chat": name(get("models.chat")),
              "decision": name(get("models.decision")), "guard": name(get("models.guard")),
              "driver": name(driver.get("model") if isinstance(driver, dict) else driver),
              "embedding": name(get("models.embedding") or getattr(svc, "embedding_model", "")),
              "runtime": str(get("models.runtime") or ""), "vision": bool(getattr(svc, "vision", False)),
              "remote": ({"provider": remote.get("provider", ""), "model": name(remote.get("model"))} if remote else None)}
    browser = getattr(svc, "browser", None)
    limits = {"max_steps": get("agent.max_steps"), "step_tokens": get("agent.step_tokens"),
              "approval_timeout_s": get("agent.approval_timeout"), "parallel_tasks": get("agent.parallel_tasks"),
              "context_tokens": get("models.context"), "step_budget_s": get("models.step_budget_s"),
              "tool_output_chars": get("agent.tool_output_chars")}
    try:
        from ..agent.loop import HARD_STOP
        limits["hard_stop_times_limit"] = HARD_STOP
    except Exception:
        pass
    count = lambda key: len(get(key) or [])          # noqa: E731  how many, never which
    settings = {"agent.thinking": get("agent.thinking", "auto"), "agent.transcript": bool(get("agent.transcript", False)),
                "agent.recall_min_similarity": get("agent.recall_min_similarity"),
                "models.decision_api": get("models.decision_api"), "models.decision_threshold": get("models.decision_threshold"),
                "models.discover": get("models.discover"), "models.idle_unload_minutes": get("models.idle_unload_minutes"),
                "sandbox.driver": get("sandbox.driver"), "sandbox.network": get("sandbox.network"),
                "sandbox.memory_gb": get("sandbox.memory_gb"),
                "browser.headless": get("browser.headless"), "browser.allow": count("browser.allow"),
                "browser.block": count("browser.block"), "browser.sensitive_hosts": count("browser.sensitive_hosts"),
                "connectors.enabled": list(get("connectors.enabled") or []), "connectors.send": list(get("connectors.send") or []),
                "policy.rules": count("policy.rules"), "egress.allow": count("egress.allow"),
                "files.roots": count("files.roots"), "files.media": get("files.media"),
                "channels": sorted((get("channels") or {}).keys()) if isinstance(get("channels"), dict) else []}
    return {"app": {"version": __version__, "commit": commit, "os": os_name, "machine": platform.machine(),
                    "python": platform.python_version()},
            "models": models,
            "browser": {"mode": getattr(browser, "mode", None) or get("browser.mode")},
            "limits": limits, "settings": settings}


def _taken(record: dict) -> bool:
    event = str(record.get("event", ""))
    # The policy's own rulings, where it did not simply allow (asked or refused).
    return event.startswith(AUDIT_TAKEN) or (event == "policy" and record.get("decision") not in (None, "allow"))


def _owns(record: dict, task: dict, others: list[dict]) -> bool:
    """Whether an audit record belongs to this task: by its task id; else (records from before ids were kept) by
    conversation and time, or by time alone when no other task ran then."""
    start, end = task["created"] or 0, (task["finished"] or time.time()) + SLACK_S
    if not start <= record.get("ts", 0) <= end:
        return False
    if record.get("task"):
        return record["task"] == task["id"]
    if record.get("event", "").startswith(GLOBAL):
        return True
    if record.get("conversation"):
        return record["conversation"] == task["conversation"]
    return not any((o["created"] or 0) < end and (o["finished"] or time.time()) + SLACK_S > start for o in others)


def build(svc, store, conversation: str, now: float | None = None) -> dict | None:
    """The trace of one thread, scrubbed, as data. None when it has no tasks."""
    tasks = store.query("SELECT * FROM tasks WHERE conversation = ? ORDER BY created", (conversation,))
    if not tasks:
        return None
    now = now or time.time()
    rows = store.query("SELECT seq, task, ts, data FROM task_events WHERE conversation = ? ORDER BY seq", (conversation,))
    events: dict[str, list[dict]] = {}
    for row in rows:
        try:
            event = json.loads(row["data"])
        except ValueError:
            continue
        event.setdefault("ts", row["ts"])
        events.setdefault(row["task"] or "", []).append(event)
    start = min(t["created"] or now for t in tasks) - 1
    end = max((t["finished"] or now) for t in tasks) + SLACK_S
    path = getattr(getattr(svc, "audit", None), "path", None)
    records = _audit_between(path, start, end) if path else []
    overlapping = store.query("SELECT id, conversation, created, finished FROM tasks WHERE created < ? AND "
                              "(finished IS NULL OR finished > ?)", (end, start))
    title_row = store.one("SELECT title FROM conversation_titles WHERE conversation = ?", (conversation,))
    from . import titles
    title = (title_row or {}).get("title") or titles.heuristic(tasks[0].get("prompt") or "")
    trace = {"about": ABOUT, "format": FORMAT, "exported": _iso(now), **header(svc),
             "thread": {"title": title, "tasks": len(tasks), "started": _iso(tasks[0]["created"]),
                        "finished": _iso(tasks[-1]["finished"])},
             "tasks": []}
    for number, task in enumerate(tasks, 1):
        others = [o for o in overlapping if o["id"] != task["id"]]
        asked = {e.get("id") for e in events.get(task["id"], []) if str(e.get("event", "")).startswith("approval.")}
        own = [r for r in records if _taken(r) and (_owns(r, task, others) or
                                                    (r["event"].startswith("approval.") and r.get("id") in asked))]
        trace["tasks"].append(_task(number, task, events.get(task["id"], []), own))
    return scrubber_for(svc).value(trace)


def _task(number: int, task: dict, events: list[dict], records: list[dict]) -> dict:
    t0 = task["created"] or (events[0]["ts"] if events else time.time())
    timeline: list[dict] = []

    def add(ts: float, kind: str, **fields: Any) -> dict:
        entry = {"at": _iso(ts), "t": round((ts or t0) - t0, 2), "kind": kind,
                 **{k: v for k, v in fields.items() if v not in (None, "", [], {})}}
        entry["_ts"] = ts or t0
        timeline.append(entry)
        return entry

    info: dict[str, Any] = {"number": number, "id": task["id"], "status": task["status"],
                            "started": _iso(task["created"]), "finished": _iso(task["finished"]),
                            "seconds": round(task["finished"] - task["created"], 1) if task["finished"] and task["created"] else None,
                            "request": _cut(task.get("prompt") or "", ANSWER_CHARS)}
    calls: dict[str, dict] = {}
    approvals: dict[str, dict] = {}
    plans = 0
    totals = {"steps": 0, "failed_steps": 0, "turns": 0, "read": 0, "cached": 0, "wrote": 0, "read_ms": 0,
              "wrote_ms": 0, "approvals": 0, "interim_answers": 0}
    for e in events:
        kind, ts = e.get("type"), e.get("ts") or t0
        if kind == "started":
            files = [f.get("name") for f in e.get("attachments") or [] if isinstance(f, dict)]
            info.update({"attachments": files or None, "time_limit_s": e.get("time_limit"),
                         "remote": e.get("remote"), "routine": (e.get("routine") or {}).get("name")})
            add(ts, "request", text=info["request"], attachments=files)
        elif kind in ("steer", "steered", "steering"):
            info.setdefault("steers", []).append(_cut(e.get("text") or e.get("message") or "", TEXT_CHARS))
            add(ts, "steer", text=_cut(e.get("text") or e.get("message") or "", TEXT_CHARS))
        elif kind in ("queued", "resumed"):
            add(ts, kind, ahead=e.get("ahead"))
        elif kind == "note":
            add(ts, "status", text=_cut(e.get("text") or "", LINE_CHARS))
        elif kind == "plan":
            plans += 1
            steps = [{"step": _cut(s.get("step") or s.get("text") or s.get("title") or "", LINE_CHARS),
                      "status": s.get("status", "")} if isinstance(s, dict) else {"step": _cut(s, LINE_CHARS)}
                     for s in e.get("steps") or []]
            add(ts, "plan", version=plans, steps=steps)
        elif kind == "tool":
            totals["steps"] += 1
            entry = add(ts, "step", number=totals["steps"], tool=e.get("name"), arguments=_args(e.get("arguments")))
            calls[e.get("id") or f"#{totals['steps']}"] = entry
        elif kind == "tool_done":
            entry = calls.pop(e.get("id") or "", None)
            if entry is None:                       # no id: the latest open call of that tool
                key = next((k for k, v in reversed(list(calls.items())) if v.get("tool") == e.get("name")), None)
                entry = calls.pop(key) if key else add(ts, "step", tool=e.get("name"))
            entry["ok"] = bool(e.get("ok", True))
            entry["result"] = _cut(e.get("result") or e.get("preview") or "", RESULT_CHARS)
            entry["seconds"] = round(ts - entry["_ts"], 2)
            if not entry["ok"]:
                totals["failed_steps"] += 1
        elif kind == "activity":
            _activity(e, ts, add, approvals)
        elif kind == "paused":
            add(ts, "control", what="paused while the owner has the screen", text=_cut(e.get("text") or "", LINE_CHARS))
        elif kind == "interim":
            totals["interim_answers"] += 1
            add(ts, "interim", text=_cut(e.get("text") or "", ANSWER_CHARS))
        elif kind in ("done", "failed", "cancelled"):
            info["answer"] = _cut(e.get("reply") or task.get("reply") or "", ANSWER_CHARS)
            info["ended"] = {"how": kind, "incomplete": bool(e.get("incomplete"))}
            add(ts, "end", how=kind, incomplete=bool(e.get("incomplete")) or None, answer=info["answer"])
        elif kind == "files":
            add(ts, "files", names=[f.get("name") for f in e.get("files") or [] if isinstance(f, dict)])
        elif kind == "items":
            add(ts, "items", count=len(e.get("items") or []))
        elif kind == "routine":
            add(ts, "routine", action=e.get("action"), name=e.get("name"), spec=e.get("spec"))
        elif kind == "shot":
            add(ts, "picture", page=_cut(e.get("page") or "", LINE_CHARS))
    for record in records:
        _record(record, add, approvals, info, totals)
    totals["approvals"] = len(approvals)
    for entry in calls.values():                    # started and never finished (stopped mid-step)
        entry["ok"] = None
        entry["result"] = "(no result: the task ended during this step)"
    if "ended" not in info:
        if task["status"] in ("done", "failed", "cancelled"):
            info["answer"] = _cut(task.get("reply") or "", ANSWER_CHARS)
            info["ended"] = {"how": task["status"]}
        else:
            info["ended"] = {"how": f"still {task['status']} when exported"}
    timeline.sort(key=lambda entry: entry["_ts"])
    for entry in timeline:
        entry.pop("_ts", None)
    totals["read_s"] = round(totals.pop("read_ms") / 1000, 1)
    totals["wrote_s"] = round(totals.pop("wrote_ms") / 1000, 1)
    info["totals"] = totals
    info["timeline"] = timeline
    return info


def _activity(e: dict, ts: float, add, approvals: dict) -> None:
    event = str(e.get("event") or "")
    fields = {k: v for k, v in e.items() if k not in ("type", "event", "task", "conversation", "ts", "seq")}
    side = {"approval.requested": "asked", "approval.decided": "decided"}.get(event)
    if side and e.get("id") and side in approvals.get(e["id"], {}):
        return                                      # the same approval told twice
    if event == "approval.requested":
        entry = add(ts, "approval", what="asked", id=e.get("id"), action=e.get("kind"), target=e.get("target"),
                    risk=e.get("risk"), summary=_cut(e.get("summary") or "", TEXT_CHARS))
        approvals.setdefault(e.get("id") or f"#{len(approvals)}", {})["asked"] = entry
    elif event == "approval.decided":
        entry = add(ts, "approval", what="decided", id=e.get("id"), status=e.get("status"))
        approvals.setdefault(e.get("id") or f"#{len(approvals)}", {})["decided"] = entry
    elif event in ("browser.handoff", "browser.handed_back"):
        add(ts, "control", what="handed to the owner" if event == "browser.handoff" else "handed back",
            reason=_cut(e.get("reason") or "", LINE_CHARS), page=e.get("page"), waited_s=e.get("waited"))
    else:
        failed = e.get("ok") is False or bool(e.get("error"))
        add(ts, "error" if failed and event.endswith(("fault", "crashed", "failed")) else "activity", event=event,
            **{k: (_cut(v, LINE_CHARS) if isinstance(v, str) else v) for k, v in fields.items()})


def _record(r: dict, add, approvals: dict, info: dict, totals: dict) -> None:
    event, ts = r.get("event", ""), r.get("ts") or 0
    if event == "agent.turn":
        totals["turns"] += 1
        for key in ("read", "cached", "wrote", "read_ms", "wrote_ms"):
            totals[key] += int(r.get(key) or 0)
        add(ts, "turn", step=r.get("step"), think=r.get("think"), read=r.get("read"), cached=r.get("cached"),
            wrote=r.get("wrote"), read_ms=r.get("read_ms"), wrote_ms=r.get("wrote_ms"))
    elif event == "task.running":
        info["model"], info["slot"] = r.get("model") or None, r.get("slot")
    elif event.startswith(("decision.", "guard.", "egress.")) or event == "policy":
        add(ts, "judge", event=event, **{k: (_cut(v, LINE_CHARS) if isinstance(v, str) else v)
                                         for k, v in r.items() if k not in ("ts", "event", "task", "conversation")})
    elif event.startswith("approval."):
        # The app's events have the approval itself; the audit adds who decided and for how long.
        known = approvals.get(r.get("id") or "", {})
        side = "decided" if event == "approval.decided" else "asked"
        if side in known:
            for key in ("scope", "decided_by", "status"):
                if r.get(key) not in (None, ""):
                    known[side].setdefault(key, r[key])
        else:
            entry = add(ts, "approval", what=side, id=r.get("id"), action=r.get("kind"), target=r.get("target"),
                        status=r.get("status"), scope=r.get("scope"), decided_by=r.get("decided_by"),
                        summary=_cut(r.get("summary") or "", TEXT_CHARS))
            approvals.setdefault(r.get("id") or f"#{len(approvals)}", {})[side] = entry
    elif event in ("browser.handoff", "browser.handed_back"):
        pass                                        # in the task's own events already
    elif event == "browser.owner_control":
        add(ts, "control", what="owner took the screen" if r.get("on") else "owner gave the screen back")
    elif event in ("task.failed", "agent.model_failed") or r.get("error"):
        add(ts, "error", event=event, error=_cut(r.get("error") or "", LINE_CHARS))
    elif event in ("agent.start", "agent.finish"):
        add(ts, "agent", event=event, origin=r.get("origin"), tainted=r.get("tainted"))
    elif event.startswith(("agent.", "task.")):
        add(ts, "agent", event=event, **{k: (_cut(v, LINE_CHARS) if isinstance(v, str) else v)
                                         for k, v in r.items() if k not in ("ts", "event", "task", "conversation")})


# ---- writing ---------------------------------------------------------------------------------------------------

def filename(trace: dict, extension: str, now: float | None = None) -> str:
    title = re.sub(r"[^a-z0-9]+", "-", str(trace.get("thread", {}).get("title") or "thread").lower()).strip("-")
    title = title[:40].rstrip("-") or "thread"
    date = datetime.datetime.fromtimestamp(now or time.time()).strftime("%Y-%m-%d")
    return f"coderunner-trace-{title}-{date}.{extension}"


def to_json(trace: dict) -> str:
    return json.dumps(trace, ensure_ascii=False, indent=2, default=str) + "\n"


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|])")


def _inline(value: Any, limit: int = LINE_CHARS) -> str:
    text = " ".join(str(value if value is not None else "").split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return _MD_SPECIAL.sub(r"\\\1", text)


def _block(value: Any, indent: str = "  ", lang: str = "") -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    body = "\n".join(indent + line for line in text.splitlines()) or indent
    return f"{indent}{fence}{lang}\n{body}\n{indent}{fence}"


def _clock(iso: str | None) -> str:
    return iso[11:23] if iso and len(iso) >= 23 else (iso or "")


def _num(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def to_markdown(trace: dict) -> str:
    out = [f"> {trace['about']}", "", f"# Trace: {_inline(trace['thread']['title'], 120)}", ""]
    app, models = trace["app"], trace["models"]
    model_line = " · ".join(f"{k}: {_inline(v, 80)}" for k, v in models.items()
                            if k not in ("remote", "vision") and v)
    if models.get("remote"):
        model_line += f" · remote: {_inline(models['remote'].get('provider'))} {_inline(models['remote'].get('model'))}"
    limits = " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in trace["limits"].items() if v not in (None, ""))
    out += ["| | |", "|---|---|",
            f"| Exported | {trace['exported']} |",
            f"| App | CodeRunner {_inline(app['version'])}" + (f" ({_inline(app['commit'])})" if app.get("commit") else "")
            + f" · {_inline(app['os'])} {_inline(app['machine'])} · Python {_inline(app['python'])} |",
            f"| Models | {model_line or '-'}" + (" · sees pictures" if models.get("vision") else "") + " |",
            f"| Browser | {_inline(trace['browser'].get('mode') or '-')} |",
            f"| Limits | {limits or '-'} |",
            f"| Thread | {trace['thread']['tasks']} task{'s' if trace['thread']['tasks'] != 1 else ''}, "
            f"started {trace['thread']['started']} |", ""]
    settings = ", ".join(f"`{k}` {_inline(json.dumps(v, default=str), 120)}" for k, v in trace["settings"].items()
                         if v not in (None, "", [], 0, False))
    if settings:
        out += [f"Settings: {settings}", ""]
    for task in trace["tasks"]:
        out += _md_task(task)
    out.append("")
    return "\n".join(out)


def _md_task(task: dict) -> list[str]:
    ended = task.get("ended") or {}
    secs = f" · {task['seconds']} s" if task.get("seconds") is not None else ""
    out = [f"## Task {task['number']} · {_inline(ended.get('how') or task['status'])}"
           + (" (said it did not finish)" if ended.get("incomplete") else "") + secs, ""]
    meta = [f"Started {task['started']}"]
    if task.get("model"):
        meta.append(f"model {_inline(task['model'], 80)}" + (f", slot {task['slot']}" if task.get("slot") is not None else ""))
    if task.get("remote"):
        meta.append(f"on {_inline(task['remote'], 60)}")
    meta.append(f"time limit {task['time_limit_s']:.0f} s" if task.get("time_limit_s") else "no time limit")
    if task.get("routine"):
        meta.append(f"routine {_inline(task['routine'], 80)}")
    out += [" · ".join(meta), "", "**Request**", "", _block(task.get("request") or "", "")]
    if task.get("attachments"):
        out += ["", "Attached: " + ", ".join(_inline(name, 80) for name in task["attachments"])]
    for steer in task.get("steers") or []:
        out += ["", "**Steer**", "", _block(steer, "")]
    tot = task["totals"]
    out += ["", f"**Totals** {tot['steps']} steps ({tot['failed_steps']} failed) · {tot['turns']} model turns · "
                f"read {_num(tot['read'])} new tokens + {_num(tot['cached'])} cached in {tot['read_s']} s · "
                f"wrote {_num(tot['wrote'])} in {tot['wrote_s']} s · {tot['approvals']} approvals", "",
            "### Timeline", ""]
    for e in task["timeline"]:
        out += _md_entry(e)
    out += ["", "### Answer", "", _block(task.get("answer") or "(none)", ""), ""]
    return out


def _md_entry(e: dict) -> list[str]:
    head = f"- `{_clock(e.get('at'))}` +{e.get('t', 0):.1f} s "
    kind = e["kind"]
    if kind == "request":
        return [head + "**Request**" + (f" with {len(e['attachments'])} attached" if e.get("attachments") else "")]
    if kind == "steer":
        return [head + "**Steer**", _block(e.get("text") or "")]
    if kind == "status":
        return [head + "Status: " + _inline(e.get("text"))]
    if kind == "plan":
        marks = {"done": "x", "completed": "x"}
        lines = [head + f"**Plan v{e.get('version')}**"]
        lines += [f"  - [{marks.get(step.get('status'), ' ')}] {_inline(step.get('step'))}"
                  + (f" ({_inline(step['status'])})" if step.get("status") not in (None, "", "done", "completed", "pending") else "")
                  for step in e.get("steps") or []]
        return lines
    if kind == "step":
        ok = {True: "ok", False: "failed", None: "unfinished"}.get(e.get("ok"), "")
        took = f" · {e['seconds']} s" if e.get("seconds") is not None else ""
        lines = [head + f"**Step {e.get('number', '')} · {_inline(e.get('tool'))}** {ok}{took}"]
        if e.get("arguments") not in (None, "", {}):
            lines.append(_block(e["arguments"], lang="json" if isinstance(e["arguments"], (dict, list)) else ""))
        if e.get("result"):
            lines.append(_block(e["result"]))
        return lines
    if kind == "turn":
        return [head + f"Turn (step {e.get('step')}): read {_num(e.get('read', 0))} new tokens + "
                       f"{_num(e.get('cached', 0))} cached in {(e.get('read_ms') or 0) / 1000:.1f} s · wrote {_num(e.get('wrote', 0))} in "
                       f"{(e.get('wrote_ms') or 0) / 1000:.1f} s" + (" · thinking" if e.get("think") is not False else "")]
    if kind == "judge":
        rest = ", ".join(f"{k} {_inline(v, 160)}" for k, v in e.items() if k not in ("at", "t", "kind", "event"))
        return [head + f"Judge `{_inline(e.get('event'))}`: {rest}"]
    if kind == "approval":
        bits = [f"{k} {_inline(e[k], 160)}" for k in ("action", "target", "risk", "status", "scope", "decided_by") if e.get(k)]
        lines = [head + f"**Approval {_inline(e.get('what'))}** " + ", ".join(bits)]
        if e.get("summary"):
            lines.append(_block(e["summary"]))
        return lines
    if kind == "control":
        rest = ", ".join(f"{k} {_inline(e[k], 200)}" for k in ("reason", "text", "page", "waited_s") if e.get(k))
        return [head + f"**Control**: {_inline(e.get('what'))}" + (f". {rest}" if rest else "")]
    if kind == "interim":
        return [head + "**Interim answer**", _block(e.get("text") or "")]
    if kind == "end":
        return [head + f"**Ended: {_inline(e.get('how'))}**" + (" (incomplete)" if e.get("incomplete") else "")]
    if kind == "error":
        return [head + f"**Error** `{_inline(e.get('event'))}` " + _inline(e.get("error") or e.get("reason") or "")]
    same_page = e.get("page") and e.get("page") == e.get("url")
    rest = ", ".join(f"{k} {_inline(v, 160)}" for k, v in e.items()
                     if k not in ("at", "t", "kind", "event") and not (k == "page" and same_page))
    label = f" `{_inline(e['event'])}`" if e.get("event") else ""
    return [head + _inline(kind.capitalize()) + label + (f": {rest}" if rest else "")]


def to_zip(trace: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trace.md", to_markdown(trace))
        archive.writestr("trace.json", to_json(trace))
    return buffer.getvalue()
