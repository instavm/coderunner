"""Allow / deny / ask decisions and out-of-band approvals.

Every action goes through Policy.check(). Decisions come from, in order:
explicit deny rules, built-in hard denies, owner rules and grants, then
per-kind defaults. When the answer is "ask", an approval request is created
and the caller waits until the owner decides through the CLI, web UI or a
paired messaging channel. The model can never answer its own approval:
decisions only arrive through the admin API.

The owner is asked only about actions that commit something (paying, booking,
sending, posting, deleting, changing settings) or that could leak their data.
Ordinary steps such as opening pages, clicking through a site and typing a
search run without prompts.

Taint: once an agent run has read untrusted content (web pages, mail,
messages from others, sandbox output), actions in TAINT_SENSITIVE ignore
standing grants and need a fresh approval.

Leaks: injected instructions can only leak the owner's data when one run both
holds private data (memories, mail, calendar...) and has read third-party
content. In such a run, the first time data could leave towards a destination
(typing into a site, opening a URL the model chose, running code with open
network access), the owner approves sharing with that destination once for
the run.
"""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from ..core.audit import AuditLog
from ..core.config import Config
from ..core.store import Store


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# "task": for a data-sharing approval, every destination for the rest of that task (a research task that opens
# six sites asked six times, once a site); for any other approval it is "once".
SCOPES = {"once": 0, "task": 0, "session": None, "hour": 3600, "day": 86400, "always": None}
ANYWHERE = "*"                          # in a run's `shared`: the owner allowed every destination for this task

# Defaults when no rule or grant matches.
DEFAULTS: dict[str, Decision] = {
    "sandbox.exec": Decision.ALLOW,
    "llm.chat": Decision.ALLOW,
    "memory.read": Decision.ALLOW,
    "memory.write": Decision.ALLOW,       # after untrusted content, saved as unverified instead of asking
    "memory.forget": Decision.ALLOW,
    "files.read": Decision.ALLOW,         # only inside folders the owner shared (files.FileIndex.resolve)
    "browser.read": Decision.ALLOW,
    "browser.navigate": Decision.ALLOW,   # local network, file: and blocked hosts are refused separately
    "browser.interact": Decision.ALLOW,   # ordinary clicks and typing; committing ones are browser.sensitive
    "browser.sensitive": Decision.ASK,    # pay, book, send, post, delete, change settings, grant access
    "browser.password": Decision.DENY,    # the owner types passwords, never the model
    "egress.connect": Decision.ASK,
    "secret.use": Decision.DENY,          # allowed only for hosts configured per secret
    "connector.read": Decision.DENY,      # allowed only for enabled connectors
    "connector.write": Decision.ALLOW,    # creates events, reminders, notes and drafts; nothing is sent
    "channel.reply": Decision.ALLOW,      # reply in the conversation the owner started
    "channel.send": Decision.ASK,         # message anyone else
    "connector.send": Decision.ASK,       # send mail from the owner's account (connectors.send must list it)
    "system.setting": Decision.ASK,       # change one of a few reversible settings of this Mac
    "phone.call": Decision.ASK,           # dial a number for the owner, who then speaks
    # A routine the owner asks for in their own conversation is set up without a prompt; one written after
    # reading untrusted content, or asked for by someone else (their runs are tainted), needs approval.
    "schedule.create": Decision.ALLOW,
    "schedule.update": Decision.ALLOW,
    "data.share": Decision.ASK,           # see the module docstring
}

# High-risk kinds never accept standing grants, only one-time approval.
HIGH_RISK = {"browser.sensitive", "channel.send", "connector.send", "system.setting", "phone.call"}
# After untrusted content these ask, so injected text cannot plant lasting changes or
# reuse the owner's earlier grants.
TAINT_SENSITIVE = {"browser.sensitive", "channel.send", "connector.write", "egress.connect",
                   "schedule.create", "schedule.update", "memory.forget", "connector.send", "system.setting", "phone.call"}

AWAY_WAIT = 30.0    # seconds an approval waits once an earlier one of the same task expired unanswered

# The agent run an action belongs to (an agent.loop.RunContext), so the leak check can see
# what that run has read without every caller passing it along.
current_run: contextvars.ContextVar[Any] = contextvars.ContextVar("coderunner_run", default=None)


@dataclass(frozen=True)
class Action:
    kind: str
    target: str
    summary: str
    detail: dict = field(default_factory=dict)
    tainted: bool = False

    def risk(self) -> str:
        return "high" if self.kind in HIGH_RISK else "medium"


@dataclass
class Approval:
    id: str
    kind: str
    target: str
    summary: str
    detail: dict
    risk: str
    origin: str
    status: str
    created: float
    scope: str = "once"
    decided: float | None = None
    decided_by: str = ""

    def public(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "kind", "target", "summary", "detail", "risk", "origin", "status",
            "created", "scope", "decided", "decided_by")}


def _matches(pattern: str, target: str) -> bool:
    pattern, target = pattern.lower(), target.lower()
    if pattern == target or pattern == "*":
        return True
    if pattern.startswith("*.") and (target == pattern[2:] or target.endswith(pattern[1:])):
        return True
    return fnmatch.fnmatchcase(target, pattern)


class Policy:
    def __init__(self, config: Config, store: Store, audit: AuditLog):
        self.config = config
        self.store = store
        self.audit = audit
        self._session_grants: list[tuple[str, str]] = []
        self._waiters: dict[str, asyncio.Future] = {}
        self._listeners: list[Callable[[Approval], Awaitable[None] | None]] = []

    # ---- decisions -------------------------------------------------------

    def evaluate(self, action: Action) -> Decision:
        rules = self.config.get("policy.rules", []) or []
        for rule in rules:
            if rule.get("kind") == action.kind and rule.get("decision") == "deny" \
                    and _matches(rule.get("target", "*"), action.target):
                return Decision.DENY
        builtin = self._builtin(action)
        if builtin is Decision.DENY:
            return builtin
        if action.tainted and action.kind in TAINT_SENSITIVE:
            return Decision.ASK
        if builtin is not None:
            return builtin
        if action.kind not in HIGH_RISK:
            for rule in rules:
                if rule.get("kind") == action.kind and rule.get("decision") == "allow" \
                        and _matches(rule.get("target", "*"), action.target):
                    return Decision.ALLOW
            if self._granted(action):
                return Decision.ALLOW
        # "ask" rules make an owner stricter than the defaults, e.g. approve every new site.
        if any(rule.get("kind") == action.kind and rule.get("decision") == "ask"
               and _matches(rule.get("target", "*"), action.target) for rule in rules):
            return Decision.ASK
        return DEFAULTS.get(action.kind, Decision.ASK)

    def _builtin(self, action: Action) -> Decision | None:
        if action.kind == "connector.read":
            enabled = self.config.get("connectors.enabled", []) or []
            return Decision.ALLOW if action.target in enabled else Decision.DENY
        if action.kind == "connector.write":
            enabled = self.config.get("connectors.enabled", []) or []
            if action.target.split(":")[0] not in enabled:
                return Decision.DENY
        if action.kind == "connector.send":
            # Sending is switched on per connector, apart from enabling it: reading mail does not allow sending it.
            name = action.target.split(":")[0]
            if name not in (self.config.get("connectors.enabled", []) or []) \
                    or name not in (self.config.get("connectors.send", []) or []):
                return Decision.DENY
        if action.kind in ("system.setting", "phone.call"):
            name = {"system.setting": "settings", "phone.call": "phone"}[action.kind]
            if name not in (self.config.get("connectors.enabled", []) or []):
                return Decision.DENY
        if action.kind == "secret.use":
            name, _, host = action.target.partition("@")
            spec = (self.config.get("secrets", {}) or {}).get(name)
            if not spec:
                return Decision.DENY
            hosts = spec.get("hosts", [])
            return Decision.ALLOW if any(_matches(h, host) for h in hosts) else Decision.DENY
        if action.kind == "egress.connect" and not action.tainted:
            host = action.target.rsplit(":", 1)[0]
            if any(_matches(p, host) for p in self.config.get("egress.allow", []) or []):
                return Decision.ALLOW
        if action.kind in ("browser.navigate", "browser.interact", "browser.read"):
            if any(_matches(p, action.target) for p in self.config.get("browser.block", []) or []):
                return Decision.DENY
            if action.kind == "browser.navigate" and any(
                    _matches(p, action.target) for p in self.config.get("browser.allow", []) or []):
                return Decision.ALLOW
        return None

    def _granted(self, action: Action) -> bool:
        if any(k == action.kind and _matches(t, action.target) for k, t in self._session_grants):
            return True
        now = time.time()
        rows = self.store.query(
            "SELECT target FROM grants WHERE kind = ? AND decision = 'allow' AND (expires IS NULL OR expires > ?)",
            (action.kind, now),
        )
        return any(_matches(row["target"], action.target) for row in rows)

    async def check(self, action: Action, origin: str = "agent", wait: float | None = None) -> bool:
        decision = self.evaluate(action)
        timeout = wait if wait is not None else float(self.config.get("agent.approval_timeout", 300))
        run, destination = current_run.get(), self._leak_destination(action)
        if wait is None and getattr(run, "unanswered", 0):
            # The owner let an approval of this task expire: they are away. The next ones are still shown, but
            # the task waits a short while for each, not the full time again (four expired one after another
            # held a real task for twenty minutes). Declining is the safe side.
            timeout = min(timeout, AWAY_WAIT)
        if decision is not Decision.DENY and destination is not None and destination not in run.shared \
                and ANYWHERE not in run.shared:
            sources = sorted(run.private_sources)
            note = (f"This task has read untrusted content and holds data from your {', '.join(sources)}; "
                    f"allow it to use that data on {destination} for the rest of the task.")
            if decision is Decision.ASK:
                # One prompt covers both: approving the step also approves sharing with its destination.
                action = Action(action.kind, action.target, f"{action.summary}\n\n{note}",
                                {**action.detail, "shares": sources}, action.tainted)
            else:
                share = Action("data.share", destination, f"{note}\nFirst step: {action.summary}",
                               {"sources": sources, "step": action.kind}, tainted=True)
                asked = self.evaluate(share)
                if asked is Decision.ASK:
                    approval = await self.request(share, origin)
                    if not await self.wait(approval.id, timeout):
                        return False
                    if (self.get(approval.id) or approval).scope == "task":
                        run.shared.add(ANYWHERE)
                elif not await self._decide(share, asked, origin, timeout):
                    return False
                run.shared.add(destination)
        allowed = await self._decide(action, decision, origin, timeout)
        if allowed and destination is not None:
            run.shared.add(destination)
        return allowed

    async def _decide(self, action: Action, decision: Decision, origin: str, timeout: float) -> bool:
        if decision is not Decision.ASK:
            self.audit.write("policy", kind=action.kind, target=action.target, decision=decision.value,
                             origin=origin, tainted=action.tainted)
            return decision is Decision.ALLOW
        approval = await self.request(action, origin)
        return await self.wait(approval.id, timeout)

    def _leak_destination(self, action: Action) -> str | None:
        """Where this action could carry the owner's data, if the current run could be steered to leak it."""
        run = current_run.get()
        if run is None or not run.content_tainted or not run.private_sources:
            return None
        if action.kind == "browser.navigate" and action.detail.get("chosen"):
            return action.target  # the model wrote this URL; its path and query can carry data
        if action.kind in ("browser.interact", "browser.sensitive") and (
                "text" in action.detail or "field" in action.detail):
            return action.target  # typing
        if action.kind == "sandbox.exec" and self.config.get("sandbox.network") == "default":
            return "the internet (sandbox code)"
        return None

    # ---- approvals -------------------------------------------------------

    async def request(self, action: Action, origin: str) -> Approval:
        approval = Approval(
            id=secrets.token_hex(4), kind=action.kind, target=action.target, summary=action.summary,
            detail=action.detail, risk=action.risk(), origin=origin, status="pending", created=time.time(),
        )
        self.store.execute(
            "INSERT INTO approvals(id, kind, target, summary, detail, risk, origin, status, scope, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 'once', ?)",
            (approval.id, approval.kind, approval.target, approval.summary, json.dumps(approval.detail),
             approval.risk, origin, approval.created),
        )
        self._waiters[approval.id] = asyncio.get_running_loop().create_future()
        self.audit.write("approval.requested", **approval.public())
        for listener in list(self._listeners):
            try:
                result = listener(approval)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as exc:  # a broken channel must not block approvals elsewhere
                self.audit.write("approval.listener_error", error=str(exc))
        return approval

    async def wait(self, approval_id: str, timeout: float) -> bool:
        future = self._waiters.get(approval_id)
        if future is None:
            row = self.get(approval_id)
            return bool(row and row.status == "approved")
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except asyncio.TimeoutError:
            self._finish(approval_id, "expired", "once", "timeout")
            run = current_run.get()
            if run is not None and hasattr(run, "unanswered"):
                run.unanswered += 1
            return False
        finally:
            self._waiters.pop(approval_id, None)

    def decide(self, approval_id: str, allow: bool, scope: str = "once", decided_by: str = "owner") -> Approval:
        approval = self.get(approval_id)
        if approval is None or approval.status != "pending":
            raise KeyError(f"No pending approval {approval_id}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
        if not allow or approval.kind in HIGH_RISK:
            scope = "once"
        if scope == "task" and approval.kind != "data.share":
            scope = "once"
        if allow and scope == "task":
            pass                                # the run that asked remembers it (check), nothing is stored
        elif allow and scope == "session":
            self._session_grants.append((approval.kind, approval.target))
        elif allow and scope != "once":
            ttl = SCOPES[scope]
            self.store.execute(
                "INSERT INTO grants(kind, target, decision, expires, created, source) VALUES (?, ?, 'allow', ?, ?, ?)",
                (approval.kind, approval.target, time.time() + ttl if ttl else None, time.time(), decided_by),
            )
        self._finish(approval_id, "approved" if allow else "denied", scope, decided_by)
        future = self._waiters.get(approval_id)
        if future is not None and not future.done():
            future.get_loop().call_soon_threadsafe(future.set_result, allow)
        return self.get(approval_id)

    def _finish(self, approval_id: str, status: str, scope: str, decided_by: str) -> None:
        self.store.execute(
            "UPDATE approvals SET status = ?, scope = ?, decided = ?, decided_by = ? WHERE id = ? AND status = 'pending'",
            (status, scope, time.time(), decided_by, approval_id),
        )
        self.audit.write("approval.decided", id=approval_id, status=status, scope=scope, decided_by=decided_by)

    def get(self, approval_id: str) -> Approval | None:
        row = self.store.one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            return None
        row["detail"] = json.loads(row["detail"] or "{}")
        return Approval(**row)

    def pending(self) -> list[Approval]:
        rows = self.store.query("SELECT id FROM approvals WHERE status = 'pending' ORDER BY created")
        live = [self.get(r["id"]) for r in rows]
        # Requests from a previous daemon run can no longer be resumed.
        for approval in live:
            if approval.id not in self._waiters:
                self._finish(approval.id, "expired", "once", "restart")
        return [a for a in live if a.id in self._waiters]

    def grants(self) -> list[dict]:
        rows = self.store.query("SELECT * FROM grants WHERE expires IS NULL OR expires > ? ORDER BY created",
                                (time.time(),))
        return rows + [{"kind": k, "target": t, "decision": "allow", "scope": "session"}
                       for k, t in self._session_grants]

    def revoke(self, grant_id: int) -> None:
        self.store.execute("DELETE FROM grants WHERE id = ?", (grant_id,))
        self.audit.write("grant.revoked", id=grant_id)

    def add_listener(self, callback: Callable[[Approval], Any]) -> None:
        self._listeners.append(callback)
