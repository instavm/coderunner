"""Guarded control of the user's local browser.

The broker drives a Chromium-family browser on the host, with a dedicated
profile and (normally) a visible window the owner can watch and take over.
The model gets a small action API (open, read, click, type, ...) and never
raw CDP, script evaluation, cookies or storage. Every action passes Policy:

  browser.navigate   opening a URL or going back
  browser.read       snapshot, screenshot, tab list, scrolling
  browser.interact   clicks, typing and key presses
  browser.sensitive  the same when the step commits something: the decision
                     model (or the assistant model) judges each click, typed
                     entry and Enter in its page context; payment fields, web
                     chat clients and labels too long to show are always
                     sensitive (one-time approval only)
  browser.password   typing into password / one-time-code fields (denied by
                     default: the owner does that through handoff)

Navigations inside attached tabs (links, redirects, form posts) are checked
too: document requests to localhost or private networks fail unless the host
is in browser.allow, so a page cannot steer the browser into the owner's LAN.
Page content is untrusted and returned as data only.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import hashlib
import ipaddress
import json
import re
import socket
import subprocess
import time
import urllib.parse
from typing import Any, Awaitable, Callable

from ..safety import decision
from ..core import paths
from ..safety.policy import Action, Decision, _matches
from . import launcher as default_launcher
from .cdp import CDPError, CDPPipe
from .errors import BrowserError
from .page import PAGE_CHARS, ax_value, cut, page_changes, render_ax, squash
from .scripts import (ANCESTORS_JS, CONTAINS_JS, DESCRIBE_JS, FOCUSED_JS, SCROLL_JS, SELECT_ALL_JS, SELECT_OPTION_JS,
                      SETTLE_JS)
from .urls import is_local_ip, local_problem, page_host, parse_url, same_site



NAV_TIMEOUT = 20.0
NAV_START_WINDOW = 0.5      # how long a click or key press gets to start a navigation
HANDOFF_WAIT = 600
SPAWN_WINDOW = 10.0         # popups opened by an action follow its navigation rules for this long

# Web chat clients: anything typed or clicked there could message people or answer
# an approval request on a paired channel, so every interaction is sensitive.
MESSAGING_HOSTS = ("web.telegram.org", "*.web.telegram.org", "web.whatsapp.com", "messages.google.com",
                   "signal.org", "*.signal.org", "discord.com", "*.discord.com", "discordapp.com",
                   "*.discordapp.com", "slack.com", "*.slack.com")
FIND_SCAN_CHARS = 400_000   # browser_find reads the whole page, not just the first snapshot screenful
FIND_LIMIT = 40             # matching lines returned
CLICK_ATTEMPTS = 3          # clicks nobody approved are re-checked when the page moves under them
CLICK_SETTLE = 0.4          # seconds to let overlays and hover effects settle before re-checking
SECRET_AUTOCOMPLETE = frozenset({"current-password", "new-password", "one-time-code"})
FRAME_TAGS = frozenset({"iframe", "frame", "object", "embed"})
KEYS = {  # name -> (windows virtual key code, text)
    "Enter": (13, "\r"), "Tab": (9, ""), "Escape": (27, ""), "ArrowDown": (40, ""), "ArrowUp": (38, ""),
    "ArrowLeft": (37, ""), "ArrowRight": (39, ""), "Home": (36, ""), "End": (35, ""), "Backspace": (8, ""),
    "Delete": (46, ""), "PageDown": (34, ""), "PageUp": (33, ""), "Space": (32, " "),
}
ACTIVATING_KEYS = frozenset({"Enter", "Space"})   # press the focused control, or submit its form
# Before reading the page after an action: how long to wait for it to start changing (typed text often brings
# suggestions about a second later), then for it to stay still, and at most in all (seconds).
SETTLE_START = {"type": 1.5, "load": 1.5}
SETTLE_START_DEFAULT = 0.6
SETTLE_QUIET = 0.3
SETTLE_MOST = 3.0
SETTLE_MOST_LOAD = 8.0      # after a navigation: results pages (flights, shops) fill in for several seconds
NO_CHANGE = "No visible change on the page."
BLANK_WAIT = 10.0           # a page with no text yet (a bot check, a script-drawn app) gets this long to show some
DENIED_PERMISSIONS = ("geolocation", "notifications", "camera", "microphone", "midi", "clipboard-read",
                      "display-capture", "idle-detection", "window-management", "local-fonts",
                      "local-network-access", "payment-handler", "nfc")
PASSWORD_HINT = ("This is a password or one-time-code field. The owner types those, never the assistant: "
                 "call handoff(reason) so they can finish it in the CodeRunner browser window.")

def field_kind(node: dict) -> str | None:
    """'password' for secret fields, 'payment' for card fields, from a DOM.describeNode result."""
    raw = node.get("attributes") or []
    attrs = {raw[i].lower(): raw[i + 1] for i in range(0, len(raw) - 1, 2)}
    tokens = set(attrs.get("autocomplete", "").lower().split())
    if (node.get("nodeName", "").upper() == "INPUT" and attrs.get("type", "").lower() == "password") \
            or tokens & SECRET_AUTOCOMPLETE:
        return "password"
    if any(token.startswith("cc-") for token in tokens):
        return "payment"
    return None


CARD_FIELDS = {"cc-number": "card number", "cc-csc": "card security code", "cc-exp": "card expiry date",
               "cc-exp-month": "card expiry month", "cc-exp-year": "card expiry year", "cc-name": "name on card"}


def card_field(node: dict) -> str:
    raw = node.get("attributes") or []
    tokens = {raw[i].lower(): raw[i + 1] for i in range(0, len(raw) - 1, 2)}.get("autocomplete", "").lower().split()
    return next((CARD_FIELDS[t] for t in tokens if t in CARD_FIELDS), "card details")


def _fingerprint(plan: dict) -> tuple:
    return plan["hit"], plan["frame"], plan["label"], plan["kind"]


# ---- broker ---------------------------------------------------------------

class BrowserBroker:
    def __init__(self, config, policy, audit, launcher=None, decider=None, chrome=None):
        self.config = config
        self.policy = policy
        self.audit = audit
        self.launcher = launcher    # None: chosen by browser.mode each time the browser starts
        self.chrome = chrome        # extension.ChromeBridge, for browser.mode "chrome"
        self._launched_by: Any = None
        self.decider = decider      # optional decision.ActionCheck: a second opinion on committing steps
        self.origin = "agent"
        self.blank_wait = BLANK_WAIT
        self._process: Any = None
        self._cdp: CDPPipe | None = None
        self._start_lock = asyncio.Lock()
        self._tab_counter = 0
        self._tab_ids: dict[str, str] = {}        # targetId -> short tab id shown to the model
        self._tab_targets: dict[str, str] = {}    # short tab id -> targetId
        self._reset()

    def _reset(self) -> None:
        self._sessions: dict[str, str] = {}       # targetId -> sessionId
        self._targets: dict[str, str] = {}        # sessionId -> targetId
        self._active: str | None = None
        self._opened: set[str] = set()            # tabs the broker opened itself
        self._refs: dict[str, dict[int, dict]] = {}
        self._numbers: dict[str, dict] = {}       # tab -> element numbers kept across renderings (render_ax)
        self._lines: dict[str, list[str]] = {}    # tab -> the page as the model last saw it, line by line
        self._read_urls: dict[str, str] = {}      # tab -> the address of the page as the model last saw it
        self._snapshots: dict[str, str] = {}
        self._shots: dict[str, str] = {}
        self._loads: dict[str, collections.deque] = {}
        self._dialogs: dict[str, dict] = {}
        self._dismissed: dict[str, dict] = {}
        self._blocked: dict[str, str] = {}
        self._busy: dict[str, bool] = {}          # tabs the assistant is acting in -> tainted flag
        self._handoff: set[str] = set()
        self._expected: dict[str, str] = {}       # tab -> host of the approved navigation in progress
        self._committed: dict[str, str] = {}      # tab -> URL of the last committed main-frame document
        self._page_targets: dict[str, dict] = {}  # every page target -> {"opener": ...}
        self._spawned: dict[str, tuple[bool, str, float]] = {}   # popup -> (tainted, opener, until)
        self._new_tabs: dict[str, list[str]] = {}         # tab -> popups it opened during the action
        self._dns: dict[str, tuple[float, str | None]] = {}

    @property
    def mode(self) -> str:
        """dedicated: a browser with the assistant's own profile; chrome: a tab group in the owner's browser."""
        return "chrome" if self.config.get("browser.mode") == "chrome" else "dedicated"

    @property
    def headless(self) -> bool:
        return self.mode == "dedicated" and bool(self.config.get("browser.headless", False))

    def _launcher(self):
        if self.launcher is not None:
            return self.launcher
        if self.mode == "chrome":
            if self.chrome is None:
                raise BrowserError("browser.mode is chrome, but the extension bridge is not running")
            from .extension import ExtensionLauncher
            return ExtensionLauncher(self.chrome)
        return default_launcher

    def running(self) -> bool:
        return (self._process is not None and self._process.poll() is None
                and self._cdp is not None and not self._cdp.closed)

    # ---- public actions -------------------------------------------------

    async def start(self) -> dict:
        async def body() -> dict:
            await self._ensure()
            return {"ok": True, "pid": self._process.pid, "headless": self.headless}
        return await self._run("start", {}, body)

    async def stop(self) -> dict:
        async with self._start_lock:
            if self._process is None:
                return {"ok": True, "running": False}
            await self._shutdown()
        self.audit.write("browser.stop", ok=True)
        return {"ok": True, "running": False}

    async def tabs(self, tainted: bool = False) -> dict:
        return await self._run("tabs", {"tainted": tainted}, lambda: self._tabs(tainted))

    async def open(self, url: str, tainted: bool = False) -> dict:
        return await self._run("open", {"url": url, "tainted": tainted}, lambda: self._open(url, tainted))

    async def navigate(self, url: str, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("navigate", {"url": url, "tab": tab, "tainted": tainted},
                               lambda: self._navigate(url, tab, tainted))

    async def back(self, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("back", {"tab": tab, "tainted": tainted}, lambda: self._back(tab, tainted))

    async def close_tab(self, tab: str, tainted: bool = False) -> dict:
        return await self._run("close_tab", {"tab": tab, "tainted": tainted}, lambda: self._close_tab(tab, tainted))

    async def snapshot(self, tab: str | None = None, max_chars: int = PAGE_CHARS, tainted: bool = False) -> dict:
        return await self._run("snapshot", {"tab": tab, "tainted": tainted},
                               lambda: self._snapshot(tab, max_chars, tainted))

    async def click(self, ref: int, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("click", {"ref": ref, "tab": tab, "tainted": tainted},
                               lambda: self._click(ref, tab, tainted))

    async def type(self, ref: int, text: str, submit: bool = False, tab: str | None = None,
                   tainted: bool = False, clear: bool = True) -> dict:
        return await self._run("type", {"ref": ref, "chars": len(str(text)), "submit": submit, "tab": tab,
                                        "tainted": tainted},
                               lambda: self._type(ref, text, submit, tab, tainted, clear))

    async def click_at(self, x: float, y: float, tab: str | None = None, tainted: bool = False) -> dict:
        """Click what is at viewport point (x, y) of a screenshot: the element there is found first and clicked
        through the same checks as click(ref), so a point never bypasses policy, the judge or approvals."""
        return await self._run("click_at", {"x": round(float(x)), "y": round(float(y)), "tab": tab, "tainted": tainted},
                               lambda: self._click_at(float(x), float(y), tab, tainted))

    async def type_focused(self, text: str, submit: bool = False, tab: str | None = None,
                           tainted: bool = False) -> dict:
        """Type into the field that has the focus (after a click on it), with the same checks as type(ref)."""
        return await self._run("type_focused", {"chars": len(str(text)), "submit": submit, "tab": tab,
                                                "tainted": tainted},
                               lambda: self._type_focused(str(text), submit, tab, tainted))

    async def find(self, text: str, tab: str | None = None, tainted: bool = False) -> dict:
        """The lines of the whole page (not just the first part) that mention text, with element refs."""
        return await self._run("find", {"query": str(text)[:200], "tab": tab, "tainted": tainted},
                               lambda: self._find(str(text), tab, tainted))

    async def select(self, ref: int, option: str, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("select", {"ref": ref, "option": str(option)[:200], "tab": tab, "tainted": tainted},
                               lambda: self._select(ref, str(option), tab, tainted))

    async def press(self, key: str, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("press", {"key": key, "tab": tab, "tainted": tainted},
                               lambda: self._press(key, tab, tainted))

    async def scroll(self, direction: str = "down", amount: int = 600, tab: str | None = None,
                     tainted: bool = False) -> dict:
        return await self._run("scroll", {"direction": direction, "amount": amount, "tab": tab, "tainted": tainted},
                               lambda: self._scroll(direction, amount, tab, tainted))

    async def screenshot(self, tab: str | None = None, full_page: bool = False, format: str = "jpeg",
                         quality: int = 70, tainted: bool = False) -> dict:
        return await self._run("screenshot", {"tab": tab, "full_page": full_page, "tainted": tainted},
                               lambda: self._screenshot(tab, full_page, format, quality, tainted))

    async def handoff(self, reason: str, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._run("handoff", {"reason": reason, "tab": tab, "tainted": tainted},
                               lambda: self._handoff_to_owner(reason, tab, tainted))

    # ---- lifecycle --------------------------------------------------------

    async def _run(self, name: str, fields: dict, body: Callable[[], Awaitable[dict]]) -> dict:
        try:
            result = await body()
        except (BrowserError, CDPError) as exc:
            result = {"ok": False, "error": str(exc)}
        extra = result.pop("_audit", {})     # e.g. typed text, only when it is not a secret
        self.audit.write(f"browser.{name}", **fields, **extra, ok=bool(result.get("ok")), error=result.get("error"),
                         page=result.get("url"))
        return result

    async def _ensure(self) -> CDPPipe:
        if self.running():
            return self._cdp
        async with self._start_lock:
            if self.running():
                return self._cdp
            if self._process is not None:          # closed by the owner, crashed or lost the pipe
                await self._shutdown()
                self.audit.write("browser.restart")
            launcher = self._launcher()
            executable = launcher.find_browser(self.config)
            if not executable:
                raise BrowserError("No Chrome, Brave, Edge or Chromium found. Install one or set browser.executable.")
            profile = paths.assistant_dir() / "browser-profile"
            try:
                self._process, self._cdp = await launcher.launch(executable, profile, self.headless)
            except (OSError, CDPError) as exc:
                raise BrowserError(f"Could not start the browser: {exc}") from None
            self._launched_by = launcher
            try:
                await self._prepare(self._cdp)
            except CDPError as exc:              # never run without the hardening in place
                await self._shutdown()
                raise BrowserError(f"Could not prepare the browser: {exc}") from None
            self.audit.write("browser.started", executable=executable, pid=self._process.pid,
                             headless=self.headless)
            return self._cdp

    async def _prepare(self, cdp: CDPPipe) -> None:
        cdp.on("Target.targetCreated", lambda p, _: self._on_target(p.get("targetInfo") or {}))
        cdp.on("Target.targetInfoChanged", lambda p, _: self._on_target(p.get("targetInfo") or {}))
        cdp.on("Target.targetDestroyed", lambda p, _: self._forget_target(p.get("targetId")))
        cdp.on("Target.detachedFromTarget", lambda p, _: self._forget_session(p.get("sessionId")))
        cdp.on("Page.lifecycleEvent", self._on_lifecycle)
        cdp.on("Page.frameNavigated", self._on_navigated)
        cdp.on("Page.javascriptDialogOpening", self._on_dialog)
        cdp.on("Page.javascriptDialogClosed", lambda _, sid: self._dialogs.pop(self._targets.get(sid), None))
        cdp.on("Page.fileChooserOpened", lambda p, sid: self.audit.write(
            "browser.file_chooser_blocked", tab=self._tab_ids.get(self._targets.get(sid or "", ""))))
        cdp.on("Fetch.requestPaused", self._on_request)
        cdp.on_close(self._on_closed)
        await cdp.send("Target.setDiscoverTargets", {"discover": True})
        # Browser-wide, so every tab (popups and new windows too) is covered before its first request.
        await cdp.send("Fetch.enable", {"patterns": [
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Request"}]})
        await cdp.send("Browser.setDownloadBehavior", {"behavior": "deny"})
        for name in DENIED_PERMISSIONS:      # best effort: names vary between versions
            with contextlib.suppress(CDPError):
                await cdp.send("Browser.setPermission", {"permission": {"name": name}, "setting": "denied"})

    async def _shutdown(self) -> None:
        process, cdp = self._process, self._cdp
        if cdp is not None and not cdp.closed:
            with contextlib.suppress(CDPError):
                await cdp.send("Browser.close", timeout=5)
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(process.wait, 5)
        if cdp is not None:
            cdp.close()
        await asyncio.to_thread((self._launched_by or self.launcher or default_launcher).terminate, process)
        self._process, self._cdp, self._launched_by = None, None, None
        self._reset()

    def _on_closed(self) -> None:
        self._sessions.clear()
        self._targets.clear()

    def _forget_session(self, sid: str | None) -> None:
        target = self._targets.pop(sid or "", None)
        if target is not None and self._sessions.get(target) == sid:
            del self._sessions[target]
        self._loads.pop(sid or "", None)

    def _forget_target(self, target: str | None) -> None:
        if not target:
            return
        self._forget_session(self._sessions.get(target))
        for table in (self._refs, self._numbers, self._lines, self._read_urls, self._snapshots, self._shots,
                      self._dialogs, self._dismissed, self._blocked,
                      self._committed, self._page_targets, self._spawned, self._new_tabs, self._expected):
            table.pop(target, None)
        self._opened.discard(target)
        if self._active == target:
            self._active = None

    # ---- tabs and sessions --------------------------------------------------

    def _tab_id(self, target: str) -> str:
        if target not in self._tab_ids:
            self._tab_counter += 1
            self._tab_ids[target] = f"t{self._tab_counter}"
            self._tab_targets[self._tab_ids[target]] = target
        return self._tab_ids[target]

    async def _pages(self) -> list[dict]:
        infos = (await self._cdp.send("Target.getTargets"))["targetInfos"]
        pages = [info for info in infos if info.get("type") == "page"]
        for info in pages:
            self._tab_id(info["targetId"])
            self._page_targets.setdefault(info["targetId"], {"opener": info.get("openerId")})
        return pages

    async def _target(self, tab: str | None) -> str:
        cdp = await self._ensure()
        pages = await self._pages()
        alive = {info["targetId"] for info in pages}
        if tab is not None:
            tab = str(tab)
            target = self._tab_targets.get(tab if not tab.isdigit() else f"t{tab}")
            if target not in alive:
                raise BrowserError(f"No open tab {tab}; call tabs to list them")
        elif self._active in alive:
            target = self._active
        elif pages:
            target = pages[0]["targetId"]
        else:
            target = (await cdp.send("Target.createTarget", {"url": "about:blank"}))["targetId"]
        self._active = target
        self._tab_id(target)
        return target

    async def _session(self, target: str) -> str:
        sid = self._sessions.get(target)
        if sid:
            return sid
        sid = (await self._cdp.send("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
        self._sessions[target], self._targets[sid] = sid, target
        try:
            await self._send(sid, "Page.enable")
            await self._send(sid, "Page.setLifecycleEventsEnabled", {"enabled": True})
            await self._send(sid, "Page.setInterceptFileChooserDialog", {"enabled": True})
            # The page behaves as if its window had the focus. Otherwise, whenever the owner works in another
            # window, pop-up widgets ignore clicks: booking.com's destination box opened only with this on.
            await self._send(sid, "Emulation.setFocusEmulationEnabled", {"enabled": True})
            self._committed.setdefault(target, (await self._info(target)).get("url", ""))
        except CDPError:
            self._forget_session(sid)
            with contextlib.suppress(CDPError):
                await self._cdp.send("Target.detachFromTarget", {"sessionId": sid})
            raise
        return sid

    async def _send(self, sid: str, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        if self._cdp is None:
            raise BrowserError("The browser is not running")
        for attempt in range(20):
            try:
                return await self._cdp.send(method, params, sid, timeout)
            except CDPError as exc:
                # The page is briefly detached while a navigation swaps processes.
                if "Not attached to an active page" not in str(exc) or attempt == 19:
                    raise
                await asyncio.sleep(0.1)
        raise AssertionError("unreachable")

    async def _info(self, target: str) -> dict:
        return (await self._cdp.send("Target.getTargetInfo", {"targetId": target}))["targetInfo"]

    async def _web_page(self, target: str) -> tuple[str, str]:
        """URL and host of a tab, refusing browser-internal and blocked pages."""
        url = (await self._info(target)).get("url", "")
        host = page_host(url)
        if host is None:
            scheme = url.split(":", 1)[0] if ":" in url else "unknown"
            raise BrowserError(f"Tab {self._tab_id(target)} shows a browser page ({scheme}:), not a web page; "
                               "open a web page first")
        if host != "about:blank":
            await self._vet(url)
        return url, host

    async def _result(self, target: str, **extra: Any) -> dict:
        info = await self._info(target)
        result = {"ok": True, "tab": self._tab_id(target), "url": info.get("url", ""),
                  "title": info.get("title", ""), **extra}
        dialog = self._dismissed.pop(target, None)
        if dialog:
            result["dialog"] = dialog
        popups = [self._tab_id(t) for t in self._new_tabs.pop(target, []) if t in self._page_targets]
        if popups:
            result["new_tabs"] = popups
        blocked = self._blocked.pop(target, None)
        if blocked:
            result["blocked"] = blocked
        return result

    @contextlib.asynccontextmanager
    async def _acting(self, target: str, sid: str, tainted: bool = False):
        self._busy[target] = tainted
        self._blocked.pop(target, None)
        self._new_tabs.pop(target, None)
        try:
            dialog = self._dialogs.get(target)
            if dialog and target not in self._handoff:
                await self._dismiss(sid, target, dialog)
            yield
        finally:
            self._busy.pop(target, None)
            if self._cdp is not None and not self._cdp.closed:
                with contextlib.suppress(CDPError):
                    await self._cdp.send("Runtime.releaseObjectGroup", {"objectGroup": "coderunner"}, sid, 5)

    # ---- policy -------------------------------------------------------------

    async def _allow(self, kind: str, host: str, summary: str, tainted: bool, **detail: Any) -> None:
        if kind == "browser.interact" and self._messaging(host):
            kind = "browser.sensitive"
        if not await self.policy.check(Action(kind, host, summary, detail, tainted), origin=self.origin):
            raise BrowserError(f"Not permitted: {summary} ({kind} was denied by policy or not approved by the owner)")

    async def _commits(self, host: str, url: str, action: str, role: str, name: str, **extra: str) -> bool:
        """Whether the judge (decision or assistant model) thinks this step commits something."""
        if self.decider is None:
            return True                  # no judge: every click, typed entry and Enter needs approval
        path = urllib.parse.urlsplit(url).path or "/"
        return await self.decider.commits(decision.describe(host, path, action, role, name, **extra))

    def _messaging(self, host: str) -> bool:
        patterns = list(MESSAGING_HOSTS) + list(self.config.get("browser.sensitive_hosts", []) or [])
        return any(_matches(pattern, host) for pattern in patterns)

    def _allowed_local(self, host: str) -> bool:
        return any(_matches(pattern, host) for pattern in self.config.get("browser.allow", []) or [])

    async def _vet(self, url: str) -> tuple[str, str]:
        """parse_url plus the local-network checks; returns (url, host) or raises."""
        url, host = parse_url(url)
        if host == "about:blank":
            return url, host
        if local_problem(host) and self._admin_port(url):
            raise BrowserError("Blocked: the CodeRunner admin interface is never opened in the assistant's browser")
        if self._allowed_local(host):
            return url, host
        problem = local_problem(host) or await self._dns_problem(host)
        if problem:
            raise BrowserError(f"Blocked: {problem}. Add it to browser.allow if the owner wants this.")
        return url, host

    def _admin_port(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return port == int(self.config.get("gateway.admin_port", 8224) or 8224)

    async def _dns_problem(self, host: str) -> str | None:
        try:
            ipaddress.ip_address(host)
            return None                              # literals were already checked
        except ValueError:
            pass
        now = time.monotonic()
        cached = self._dns.get(host)
        if cached and cached[0] > now:
            return cached[1]
        problem = None
        try:
            for address in await asyncio.wait_for(self._resolve(host), 5):
                ip = ipaddress.ip_address(address.split("%", 1)[0])
                if is_local_ip(ip):
                    problem = f"{host} resolves to a local or private address ({ip})"
                    break
        except (OSError, ValueError, asyncio.TimeoutError):
            pass                                     # unresolvable: the browser will fail on its own
        self._dns[host] = (now + 60, problem)
        return problem

    async def _resolve(self, host: str) -> list[str]:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return [info[4][0] for info in infos]

    # ---- events ---------------------------------------------------------------

    def _on_lifecycle(self, params: dict, sid: str | None) -> None:
        if params.get("name") == "load" and params.get("frameId") == self._targets.get(sid or ""):
            self._loads.setdefault(sid, collections.deque(maxlen=16)).append(params.get("loaderId"))

    async def _on_dialog(self, params: dict, sid: str | None) -> None:
        target = self._targets.get(sid or "")
        if target is None:
            return
        self._dialogs[target] = params
        self.audit.write("browser.dialog", tab=self._tab_id(target), type=params.get("type"),
                         message=str(params.get("message", ""))[:500])
        # Dialogs are left to the owner unless the assistant is acting in that tab.
        if target in self._busy and target not in self._handoff:
            await self._dismiss(sid, target, params)

    async def _dismiss(self, sid: str, target: str, dialog: dict) -> None:
        accept = dialog.get("type") in ("alert", "beforeunload")
        with contextlib.suppress(CDPError):
            await self._cdp.send("Page.handleJavaScriptDialog", {"accept": accept}, sid, 5)
        self._dialogs.pop(target, None)
        self._dismissed[target] = {"type": dialog.get("type"), "message": str(dialog.get("message", ""))[:500],
                                   "accepted": accept}

    def _on_target(self, info: dict) -> None:
        target = info.get("targetId")
        if info.get("type") != "page" or not target:
            return
        if target not in self._page_targets:
            opener = info.get("openerId")
            self._page_targets[target] = {"opener": opener}
            if opener in self._busy and opener not in self._handoff:
                # Opened by something the assistant just did: same rules as the opener until it loads.
                self._spawned[target] = (self._busy[opener], opener, time.monotonic() + SPAWN_WINDOW)
                self._new_tabs.setdefault(opener, []).append(target)

    def _on_navigated(self, params: dict, sid: str | None) -> None:
        frame = params.get("frame") or {}
        target = self._targets.get(sid or "")
        if target and frame.get("id") == target and not frame.get("parentId"):
            self._committed[target] = frame.get("url", "")
            self._refs.pop(target, None)            # element refs belong to the old document
            self._lines.pop(target, None)
            if target in self._numbers:             # new numbers continue, so an old one cannot hit a new element
                self._numbers[target]["ids"] = {}

    async def _on_request(self, params: dict, sid: str | None) -> None:
        """Browser-wide guard for every document request, in every tab."""
        url = (params.get("request") or {}).get("url", "")
        frame = params.get("frameId")
        reason = None
        if frame and frame not in self._page_targets and self._busy and self._cdp is not None:
            # In case a popup's first request beats its targetCreated event.
            with contextlib.suppress(CDPError):
                self._on_target((await self._cdp.send("Target.getTargetInfo", {"targetId": frame}))["targetInfo"])
        try:
            if urllib.parse.urlsplit(url).scheme in ("http", "https"):
                _, host = await self._vet(url)
                if self.policy.evaluate(Action("browser.read", host, url)) is Decision.DENY:
                    reason = f"{host} is blocked by policy"
                elif frame in self._page_targets:     # a tab's main frame (its frame id is the target id)
                    reason = self._navigation_problem(frame, url, host)
        except Exception as exc:                     # fail closed: a paused request must not slip through
            reason = str(exc) or "the request could not be checked"
        spawned = self._spawned.get(frame)
        cdp = self._cdp
        if cdp is None or cdp.closed:
            return
        with contextlib.suppress(CDPError):
            if not reason:
                await cdp.send("Fetch.continueRequest", {"requestId": params["requestId"]}, sid)
                return
            for target in {frame, spawned[1] if spawned else None} & set(self._page_targets):
                self._blocked[target] = reason
            self.audit.write("browser.blocked", url=url, reason=reason)
            await cdp.send("Fetch.failRequest", {"requestId": params["requestId"],
                                                 "errorReason": "BlockedByClient"}, sid)
            if spawned:                              # do not leave the blocked popup around
                self._spawned.pop(frame, None)
                await cdp.send("Target.closeTarget", {"targetId": frame})

    def _navigation_problem(self, target: str, url: str, host: str) -> str | None:
        """While the assistant acts in a tab, leaving its site needs browser.navigate to be allowed outright."""
        if target in self._spawned and self._spawned[target][2] > time.monotonic():
            tainted, opener, _ = self._spawned[target]
            reference = {page_host(self._committed.get(opener, "")), self._expected.get(opener)}
        elif target in self._busy and target not in self._handoff:
            tainted = self._busy[target]
            reference = {page_host(self._committed.get(target, "")), self._expected.get(target)}
        else:
            return None                              # the owner's own browsing
        if any(same_site(host, other) for other in reference if other):
            return None
        action = Action("browser.navigate", host, f"Open {url}", {"url": url}, tainted)
        if self.policy.evaluate(action) is Decision.ALLOW:
            return None
        return f"Opening {url} needs the owner's approval; use browser_open with that URL if it is really needed"

    # ---- navigation helpers -------------------------------------------------

    async def _wait_load(self, sid: str, loader_id: str | None, timeout: float = NAV_TIMEOUT) -> bool:
        if not loader_id:
            return True                              # same-document navigation
        if loader_id in self._loads.get(sid, ()):
            return True
        event = self._cdp.wait_for_event(
            "Page.lifecycleEvent", lambda p: p.get("name") == "load" and p.get("loaderId") == loader_id,
            timeout, sid)
        return await event is not None

    def _watch_navigation(self, sid: str, target: str) -> tuple[asyncio.Future, asyncio.Future]:
        started = self._cdp.wait_for_event("Page.frameStartedLoading", lambda p: p.get("frameId") == target,
                                           NAV_TIMEOUT, sid)
        committed = self._cdp.wait_for_event("Page.frameNavigated",
                                             lambda p: (p.get("frame") or {}).get("id") == target, NAV_TIMEOUT, sid)
        return started, committed

    async def _settle(self, sid: str, watch: tuple[asyncio.Future, asyncio.Future],
                      window: float = NAV_START_WINDOW) -> bool:
        """Wait for a navigation the last input started, if any. Returns whether one happened."""
        started, committed = watch
        done, _ = await asyncio.wait({started, committed}, timeout=window, return_when=asyncio.FIRST_COMPLETED)
        if not any(future.result() is not None for future in done):
            started.cancel()
            committed.cancel()
            await asyncio.sleep(0.1)                 # let page handlers finish
            return False
        started.cancel()
        navigated = await committed
        if navigated and navigated.get("type") != "BackForwardCacheRestore":
            await self._wait_load(sid, (navigated.get("frame") or {}).get("loaderId"))
        return True

    async def _goto(self, target: str, url: str, host: str, tainted: bool) -> dict:
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            self._expected[target] = host           # approved already; its redirects stay on that site
            try:
                result = await self._send(sid, "Page.navigate", {"url": url}, timeout=NAV_TIMEOUT + 10)
                if result.get("errorText"):
                    reason = self._blocked.pop(target, None)
                    raise BrowserError(f"Could not open {url}: {reason or result['errorText']}")
                loaded = await self._wait_load(sid, result.get("loaderId"))
            finally:
                self._expected.pop(target, None)
            return await self._observe(target, sid, await self._result(target, loaded=loaded), tainted, "load")

    # ---- action bodies ------------------------------------------------------

    async def _tabs(self, tainted: bool) -> dict:
        await self._allow("browser.read", "tabs", "List browser tabs", tainted)
        if not self.running():
            return {"ok": True, "tabs": [], "running": False}
        tabs = []
        for info in await self._pages():
            target, url = info["targetId"], info.get("url", "")
            host = page_host(url)
            visible = host is not None and (host == "about:blank" or self.policy.evaluate(
                Action("browser.read", host, url, tainted=tainted)) is not Decision.DENY)
            tabs.append({"id": self._tab_id(target), "url": url if visible else "(hidden)",
                         "title": info.get("title", "") if visible else "", "active": target == self._active})
        return {"ok": True, "tabs": tabs}

    async def _open(self, url: str, tainted: bool) -> dict:
        url, host = await self._vet(url)
        await self._allow("browser.navigate", host, f"Open {url} in a new tab", tainted, url=url, chosen=True)
        cdp = await self._ensure()
        target = (await cdp.send("Target.createTarget", {"url": "about:blank"}))["targetId"]
        self._opened.add(target)
        self._active = target
        return await self._goto(target, url, host, tainted)

    async def _navigate(self, url: str, tab: str | None, tainted: bool) -> dict:
        url, host = await self._vet(url)
        # chosen: the model wrote this URL, so it could carry data out (see Policy._leak_destination).
        await self._allow("browser.navigate", host, f"Open {url}", tainted, url=url, chosen=True)
        return await self._goto(await self._target(tab), url, host, tainted)

    async def _back(self, tab: str | None, tainted: bool) -> dict:
        target = await self._target(tab)
        sid = await self._session(target)
        history = await self._send(sid, "Page.getNavigationHistory")
        index = history.get("currentIndex", 0)
        if index <= 0:
            raise BrowserError("There is no previous page in this tab")
        entry = history["entries"][index - 1]
        url, host = await self._vet(entry.get("url", ""))
        await self._allow("browser.navigate", host, f"Go back to {url}", tainted, url=url)
        async with self._acting(target, sid, tainted):
            self._expected[target] = host
            try:
                watch = self._watch_navigation(sid, target)
                await self._send(sid, "Page.navigateToHistoryEntry", {"entryId": entry["id"]})
                await self._settle(sid, watch, window=NAV_TIMEOUT)
            finally:
                self._expected.pop(target, None)
            return await self._observe(target, sid, await self._result(target), tainted, "load")

    async def _close_tab(self, tab: str, tainted: bool) -> dict:
        if tab is None:
            raise BrowserError("Say which tab to close")
        target = await self._target(tab)
        info = await self._info(target)
        if target not in self._opened:               # the owner's tabs need permission
            host = page_host(info.get("url", "")) or "browser"
            await self._allow("browser.interact", host, f'Close tab {tab} ("{info.get("title", "")}")', tainted)
        await self._cdp.send("Target.closeTarget", {"targetId": target})
        self._forget_target(target)
        return {"ok": True, "closed": str(tab)}

    async def _snapshot(self, tab: str | None, max_chars: int, tainted: bool, whole: bool = False) -> dict:
        target = await self._target(tab)
        url, host = await self._web_page(target)
        await self._allow("browser.read", host, f"Read {url}", tainted)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            content = await self._render(target, sid, url, whole)
            result = await self._result(target)
        if not whole:                   # a search through the whole page is not the page as the model reads it
            result["changed"] = self._seen(target, result["url"], content)
        result["content"] = cut(content, max(100, int(max_chars)))
        return result

    async def _render(self, target: str, sid: str, url: str, whole: bool = False) -> str:
        nodes = (await self._send(sid, "Accessibility.getFullAXTree", timeout=60)).get("nodes", [])
        secret = await self._secret_fields(sid, nodes)
        content, refs = render_ax(nodes, url, secret, self._numbers.setdefault(target, {"next": 1, "ids": {}}),
                                  whole=whole)
        self._refs[target] = refs
        return content

    def _seen(self, target: str, url: str, content: str) -> bool:
        """Record the page as the model now knows it; returns whether it differs from what it saw before."""
        digest = hashlib.sha256(f"{url}\n{content}".encode()).hexdigest()
        changed = self._snapshots.get(target) != digest
        self._snapshots[target] = digest
        self._lines[target] = content.split("\n")
        self._read_urls[target] = url
        return changed

    async def _observe(self, target: str, sid: str, result: dict, tainted: bool, action: str = "") -> dict:
        """Add the page after an action to its result: only what changed since the model last read it, or the
        whole page after a navigation or a large change. Saves the model a step (and re-reading the page)
        after every click. Left out when reading the page would need the owner's approval."""
        try:
            url, host = await self._web_page(target)
        except BrowserError:
            return result
        if self.policy.evaluate(Action("browser.read", host, f"Read {url}", tainted=tainted)) is not Decision.ALLOW:
            return result
        before = self._lines.get(target)
        start = SETTLE_START.get(action, SETTLE_START_DEFAULT)
        most = SETTLE_MOST_LOAD if action == "load" else SETTLE_MOST
        with contextlib.suppress(CDPError):
            await self._send(sid, "Runtime.callFunctionOn", {
                "functionDeclaration": SETTLE_JS, "executionContextId": await self._world(sid, target),
                "arguments": [{"value": start * 1000}, {"value": SETTLE_QUIET * 1000}, {"value": most * 1000}],
                "awaitPromise": True, "returnByValue": True}, timeout=most + 5)
        try:
            content = await self._render(target, sid, url)
            deadline = time.monotonic() + self.blank_wait
            while not content.strip() and time.monotonic() < deadline:
                # Bot checks redirect to the real page after a few seconds; apps draw theirs late.
                await asyncio.sleep(0.5)
                url, now = await self._web_page(target)
                if now != host:                     # went elsewhere: reading that is a new decision
                    result.update(url=url, note="The page moved to another site; take a snapshot to read it.")
                    return result
                content = await self._render(target, sid, url)
        except (CDPError, BrowserError):
            return result
        info = await self._info(target)
        result.update(url=info.get("url", url), title=info.get("title", result.get("title", "")))
        if not content.strip():
            result["note"] = "The page shows no text yet; it may still be loading. Take a snapshot in a moment."
            return result
        self._seen(target, url, content)
        changes = page_changes(before, content.split("\n")) if before is not None else None
        if changes is None:
            result["content"] = cut(content, PAGE_CHARS)
        else:
            result["changes"] = cut(changes, PAGE_CHARS) if changes else NO_CHANGE
        return result

    async def _find(self, text: str, tab: str | None, tainted: bool) -> dict:
        result = await self._snapshot(tab, FIND_SCAN_CHARS, tainted, whole=True)   # footer included
        lines = result.pop("content").split("\n")
        whole, words = text.strip().lower(), re.findall(r"\w+", text.lower())
        hits = [i for i, line in enumerate(lines)
                if whole in line.lower() or (words and all(word in line.lower() for word in words))]
        shown: list[int] = []
        for i in hits[:FIND_LIMIT]:
            shown += [j for j in range(max(0, i - 1), min(len(lines), i + 2)) if j not in shown]
        result["matches"] = len(hits)
        result["content"] = "\n".join(lines[j] for j in shown) or f"Nothing on this page mentions {text!r}."
        return result

    async def _select(self, ref: int, option: str, tab: str | None, tainted: bool) -> dict:
        target = await self._target(tab)
        url, host = await self._web_page(target)
        element = self._ref(target, ref)
        sid = await self._session(target)
        label = f'{element["role"]} {json.dumps(element["name"], ensure_ascii=False)}'
        summary = f"Choose {json.dumps(option, ensure_ascii=False)} in {label} on {host}"
        risky = await self._commits(host, url, "choose an option in", element["role"], element["name"], typed=option)
        await self._allow("browser.sensitive" if risky else "browser.interact", host, summary, tainted,
                          url=url, ref=ref, label=label, option=option)
        async with self._acting(target, sid, tainted):
            context = await self._world(sid, target)
            outcome = await self._call(sid, await self._object(sid, element["backend"], context),
                                       SELECT_OPTION_JS, {"value": option}) or {}
            if not outcome.get("ok"):
                choices = ", ".join(json.dumps(o, ensure_ascii=False) for o in outcome.get("options", [])[:30])
                raise BrowserError(f"[{ref}]: {outcome.get('error', 'could not choose')}"
                                   + (f"; options are {choices}" if choices else ""))
            result = await self._observe(target, sid, await self._result(target), tainted)
            result["selected"] = outcome["selected"]
            return result

    async def _secret_fields(self, sid: str, nodes: list[dict]) -> set[int]:
        candidates = [node["backendDOMNodeId"] for node in nodes
                      if not node.get("ignored") and node.get("backendDOMNodeId")
                      and ax_value(node.get("role")) in ("textbox", "searchbox", "combobox")][:100]

        async def check(backend: int) -> int | None:
            try:
                node = (await self._send(sid, "DOM.describeNode", {"backendNodeId": backend}))["node"]
            except CDPError:
                return None
            return backend if field_kind(node) == "password" else None

        return {backend for backend in await asyncio.gather(*map(check, candidates)) if backend}

    def _ref(self, target: str, ref: Any) -> dict:
        try:
            return self._refs[target][int(ref)]
        except (KeyError, ValueError, TypeError):
            raise BrowserError(f"Unknown element [{ref}] in this tab; take a new snapshot") from None

    async def _click(self, ref: int, tab: str | None, tainted: bool) -> dict:
        target = await self._target(tab)
        url, host = await self._web_page(target)
        read = self._read_urls.get(target, "")      # the page as the model last saw it
        element = self._ref(target, ref)
        sid = await self._session(target)
        for attempt in range(CLICK_ATTEMPTS):
            async with self._acting(target, sid, tainted):
                plan = await self._click_plan(sid, target, element)
            kind = plan["kind"]
            if kind == "browser.interact" and await self._commits(
                    host, url, "click", plan["role"], plan["name"] or plan["text"], nearby=plan["label"]):
                kind = "browser.sensitive"
            summary = f"Click {plan['label']} on {host}"
            # Whether the owner is shown this exact plan; if so it must not change at all before the click.
            approved = self.policy.evaluate(Action(kind, host, summary, {"url": url}, tainted)) is not Decision.ALLOW
            await self._allow(kind, host, summary, tainted, url=url, ref=ref, label=plan["label"])
            async with self._acting(target, sid, tainted):
                # The click goes to exactly what was checked: the same element under the pointer, the same
                # label and frame. Anything different means the page changed, so nothing is clicked.
                now = await self._click_plan(sid, target, element)
                moved = _fingerprint(now) != _fingerprint(plan)
                if not moved:
                    x, y = now["x"], now["y"]
                    await self._send(sid, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
                    moved = _fingerprint(await self._probe(sid, target, element, x, y)) != _fingerprint(plan)
                if not moved:
                    watch = self._watch_navigation(sid, target)
                    for event in ("mousePressed", "mouseReleased"):
                        await self._send(sid, "Input.dispatchMouseEvent", {
                            "type": event, "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1})
                    navigated = await self._settle(sid, watch)
                    result = await self._observe(target, sid, await self._result(target, navigated=navigated),
                                                 tainted, "load" if navigated else "")
                    result["_audit"] = {"clicked": now["label"]}    # what was under the pointer, as checked
                    if result.get("changes") == NO_CHANGE:
                        # Say what the click landed on: covered by a banner or an overlay, or a page that has
                        # moved on (a late redirect) explains a click that did nothing better than a retry.
                        result["note"] = f"The click landed on {now['label']}."
                        if read and result.get("url") and result["url"].split("#")[0] != read.split("#")[0]:
                            result["note"] += " The page's address changed since you read it; take a new snapshot."
                    return result
            if approved or attempt == CLICK_ATTEMPTS - 1:
                break
            # Overlays fading in or out and hover effects settle quickly; check the click again from scratch.
            await asyncio.sleep(CLICK_SETTLE)
        raise BrowserError("The page changed under the pointer; nothing was clicked. Take a new snapshot")

    async def _click_at(self, x: float, y: float, tab: str | None, tainted: bool) -> dict:
        target = await self._target(tab)
        url, host = await self._web_page(target)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            try:
                hit = await self._send(sid, "DOM.getNodeForLocation", {
                    "x": int(x), "y": int(y), "includeUserAgentShadowDOM": True, "ignorePointerEventsNone": True})
            except CDPError:
                raise BrowserError(f"Nothing is at ({x:.0f}, {y:.0f}); take a new screenshot") from None
            if hit.get("frameId") not in (None, target):
                raise BrowserError("That point is inside an embedded frame, which the assistant cannot use")
            ref = await self._numbered(sid, target, url, hit.get("backendNodeId"))
        if ref is None:
            raise BrowserError(f"Nothing the assistant can click is at ({x:.0f}, {y:.0f}); take a new screenshot")
        return await self._click(ref, tab, tainted)

    async def _type_focused(self, text: str, submit: bool, tab: str | None, tainted: bool) -> dict:
        target = await self._target(tab)
        url, _ = await self._web_page(target)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            found = await self._send(sid, "Runtime.evaluate", {
                "expression": FOCUSED_JS, "contextId": await self._world(sid, target), "objectGroup": "coderunner"})
            object_id = (found.get("result") or {}).get("objectId")
            backend = None
            if object_id:
                backend = (await self._send(sid, "DOM.describeNode", {"objectId": object_id}))["node"].get(
                    "backendNodeId")
            ref = await self._numbered(sid, target, url, backend) if backend else None
        if ref is None:
            raise BrowserError("No text field has the focus; click the field first")
        return await self._type(ref, text, submit, tab, tainted, clear=False)

    async def _numbered(self, sid: str, target: str, url: str, backend: int | None) -> int | None:
        """The number of the nearest element at or above a node that the page text numbers (a button around an
        icon, say). The page is read again first, so the number is current."""
        if not backend:
            return None
        await self._render(target, sid, url, whole=True)        # a point may land on a footer link
        numbers = {info["backend"]: ref for ref, info in self._refs.get(target, {}).items()}
        if backend in numbers:
            return numbers[backend]
        context = await self._world(sid, target)
        chain = await self._send(sid, "Runtime.callFunctionOn", {
            "functionDeclaration": ANCESTORS_JS, "objectId": await self._object(sid, backend, context),
            "returnByValue": False, "objectGroup": "coderunner"})
        array = (chain.get("result") or {}).get("objectId")
        if not array:
            return None
        items = (await self._send(sid, "Runtime.getProperties", {"objectId": array, "ownProperties": True}))["result"]
        for item in sorted((i for i in items if i.get("name", "").isdigit()), key=lambda i: int(i["name"])):
            object_id = (item.get("value") or {}).get("objectId")
            if not object_id:
                continue
            node = (await self._send(sid, "DOM.describeNode", {"objectId": object_id}))["node"]
            if node.get("backendNodeId") in numbers:
                return numbers[node["backendNodeId"]]
        return None

    async def _click_plan(self, sid: str, target: str, element: dict) -> dict:
        with contextlib.suppress(CDPError):
            await self._send(sid, "DOM.scrollIntoViewIfNeeded", {"backendNodeId": element["backend"]})
        x, y = await self._center(sid, element["backend"])
        return {"x": x, "y": y, **await self._probe(sid, target, element, x, y)}

    async def _probe(self, sid: str, target: str, element: dict, x: float, y: float) -> dict:
        """What a click at (x, y) would hit, described in full for the approval."""
        backend = element["backend"]
        hit = await self._send(sid, "DOM.getNodeForLocation", {
            "x": int(x), "y": int(y), "includeUserAgentShadowDOM": True, "ignorePointerEventsNone": True})
        own = await self._describe(sid, target, backend) or {}
        name, text = element["name"], own.get("text", "")
        label = f'{element["role"]} {json.dumps(name or text, ensure_ascii=False)}'
        if name and text and text != name:
            label += f" (page text {json.dumps(text, ensure_ascii=False)})"
        cover = await self._covering(sid, target, backend, hit)
        if cover is not None:
            label += f' (covered by {cover.get("tag")} {json.dumps(cover.get("text", ""), ensure_ascii=False)})'
        # Structural signals only; whether the click commits something is judged in _click.
        risky = own.get("truncated") or (cover is not None and (
            cover.get("truncated") or cover.get("tag") in FRAME_TAGS))
        # When the click lands on the element itself, which of its own child nodes is under the
        # pointer does not matter (menus restyle them on hover); a different covering element does.
        return {"hit": "element" if cover is None else hit.get("backendNodeId"), "frame": hit.get("frameId"),
                "label": label, "kind": "browser.sensitive" if risky else "browser.interact",
                "text": text, "name": name, "role": element["role"]}

    async def _center(self, sid: str, backend: int) -> tuple[float, float]:
        try:
            quad = (await self._send(sid, "DOM.getBoxModel", {"backendNodeId": backend}))["model"]["border"]
        except CDPError:
            raise BrowserError("That element is not visible on the page; take a new snapshot") from None
        xs, ys = quad[0::2], quad[1::2]
        metrics = await self._send(sid, "Page.getLayoutMetrics")
        viewport = metrics.get("cssVisualViewport") or metrics.get("visualViewport") or {}
        width, height = viewport.get("clientWidth", 1e9), viewport.get("clientHeight", 1e9)
        left, right = max(min(xs), 0), min(max(xs), width)
        top, bottom = max(min(ys), 0), min(max(ys), height)
        if right - left < 1 or bottom - top < 1:
            raise BrowserError("That element has no visible area; scroll or take a new snapshot")
        return (left + right) / 2, (top + bottom) / 2

    async def _world(self, sid: str, target: str) -> int:
        world = await self._send(sid, "Page.createIsolatedWorld", {"frameId": target, "worldName": "coderunner"})
        return world["executionContextId"]

    async def _object(self, sid: str, backend: int, context: int) -> str:
        resolved = await self._send(sid, "DOM.resolveNode", {
            "backendNodeId": backend, "executionContextId": context, "objectGroup": "coderunner"})
        return resolved["object"]["objectId"]

    async def _call(self, sid: str, object_id: str, function: str, *args: dict) -> Any:
        result = await self._send(sid, "Runtime.callFunctionOn", {
            "functionDeclaration": function, "objectId": object_id, "arguments": list(args),
            "returnByValue": True, "objectGroup": "coderunner"})
        if result.get("exceptionDetails"):
            raise BrowserError("The page did not respond as expected; take a new snapshot")
        return (result.get("result") or {}).get("value")

    async def _describe(self, sid: str, target: str, backend: int | None = None, closest: bool = False) -> dict | None:
        """Tag, accessible text and form submit label of an element (or the focused one)."""
        context = await self._world(sid, target)
        if backend is None:
            found = await self._send(sid, "Runtime.evaluate", {
                "expression": FOCUSED_JS, "contextId": context, "objectGroup": "coderunner"})
            object_id = (found.get("result") or {}).get("objectId")
            if not object_id:
                return None
        else:
            object_id = await self._object(sid, backend, context)
        return await self._call(sid, object_id, DESCRIBE_JS, {"value": closest})

    async def _covering(self, sid: str, target: str, backend: int, hit: dict) -> dict | None:
        """What a hit test really found, when that is not the element or something inside it."""
        other = hit.get("backendNodeId")
        if other == backend:
            return None
        if hit.get("frameId") not in (None, target):
            return {"tag": "iframe", "text": "content of an embedded frame"}
        context = await self._world(sid, target)
        mine, theirs = await self._object(sid, backend, context), await self._object(sid, other, context)
        if await self._call(sid, mine, CONTAINS_JS, {"objectId": theirs}):
            return None
        return await self._describe(sid, target, other, closest=True) or {"tag": "unknown", "text": ""}

    async def _type(self, ref: int, text: str, submit: bool, tab: str | None, tainted: bool, clear: bool) -> dict:
        text = str(text)
        target = await self._target(tab)
        url, host = await self._web_page(target)
        element = self._ref(target, ref)
        backend = element["backend"]
        sid = await self._session(target)
        label = f'{element["role"]} {json.dumps(element["name"], ensure_ascii=False)}'
        async with self._acting(target, sid, tainted):
            node = (await self._send(sid, "DOM.describeNode", {"backendNodeId": backend}))["node"]
            field = "password" if element.get("password") else field_kind(node)
            own = {} if field == "password" else await self._describe(sid, target, backend) or {}
        if field == "password":
            if not await self.policy.check(Action("browser.password", host, f"Type into {label} on {host}",
                                                  {"url": url, "ref": ref}, tainted), origin=self.origin):
                raise BrowserError(PASSWORD_HINT)
        else:
            # The owner approves exactly what will be typed; card values are never shown or logged.
            shown = f"[{card_field(node)}]" if field == "payment" else json.dumps(text, ensure_ascii=False)
            summary = f"Type {shown} into {label} on {host}"
            if submit:
                summary += " and press Enter" + (
                    f" (submits {json.dumps(own['submit'], ensure_ascii=False)})" if own.get("submit") else "")
            detail = {"url": url, "ref": ref, "label": label, "submit": submit}
            if field == "payment":
                detail["field"] = card_field(node)
            else:
                detail["text"] = text
            risky = field == "payment" or own.get("truncated") or await self._commits(
                host, url, "type and press Enter" if submit else "type", element["role"], element["name"],
                typed=text, nearby=f"field text: {own.get('text', '')}; submit button: {own.get('submit', '')}")
            await self._allow("browser.sensitive" if risky else "browser.interact", host, summary, tainted, **detail)
        async with self._acting(target, sid, tainted):
            try:
                await self._send(sid, "DOM.focus", {"backendNodeId": backend})
            except CDPError:
                raise BrowserError(f"[{ref}] cannot take text input; click it first or pick another element") from None
            focused = await self._describe(sid, target) or {}
            if focused.get("tag") in FRAME_TAGS:
                raise BrowserError("Focus moved into an embedded frame; the assistant cannot type there")
            tokens = set(focused.get("autocomplete", "").split())
            if field != "password" and (focused.get("type") == "password" or tokens & SECRET_AUTOCOMPLETE):
                raise BrowserError(PASSWORD_HINT)
            if field is None and any(token.startswith("cc-") for token in tokens):
                raise BrowserError("Focus moved to a payment card field; take a new snapshot")
            if clear:
                context = await self._world(sid, target)
                await self._call(sid, await self._object(sid, backend, context), SELECT_ALL_JS)
                if not text:
                    await self._key(sid, "Backspace")
            if text:
                await self._send(sid, "Input.insertText", {"text": text})
            navigated = False
            if submit:
                watch = self._watch_navigation(sid, target)
                await self._key(sid, "Enter")
                navigated = await self._settle(sid, watch)
            result = await self._observe(target, sid, await self._result(target, navigated=navigated), tainted,
                                         "load" if navigated else "type")
            if field is None:
                result["_audit"] = {"text": text}
            return result

    async def _key(self, sid: str, key: str) -> None:
        code, text = KEYS[key]
        down = {"type": "keyDown" if text else "rawKeyDown", "key": key, "code": key,
                "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code}
        if text:
            down.update(text=text, unmodifiedText=text)
        await self._send(sid, "Input.dispatchKeyEvent", down)
        await self._send(sid, "Input.dispatchKeyEvent", {"type": "keyUp", "key": key, "code": key,
                                                         "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code})

    async def _press(self, key: str, tab: str | None, tainted: bool) -> dict:
        if key not in KEYS:
            raise BrowserError(f"Unsupported key {key!r}; use one of {', '.join(KEYS)}")
        target = await self._target(tab)
        url, host = await self._web_page(target)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            focused = await self._describe(sid, target)
        summary = f"Press {key} on {host}"
        risky = False
        if focused:
            summary += f' (focus: {focused.get("tag")} {json.dumps(focused.get("text", ""), ensure_ascii=False)})'
            if key in ACTIVATING_KEYS:
                # Enter and Space activate the focused control; Enter also submits its form.
                risky = focused.get("tag") in FRAME_TAGS or focused.get("truncated") or await self._commits(
                    host, url, f"press {key} in", focused.get("tag", ""), focused.get("text", ""),
                    nearby=f"submit button: {focused['submit']}" if focused.get("submit") else "")
                if focused.get("submit"):
                    summary += f" which may submit {json.dumps(focused['submit'], ensure_ascii=False)}"
        await self._allow("browser.sensitive" if risky else "browser.interact", host, summary, tainted,
                          url=url, key=key)
        async with self._acting(target, sid, tainted):
            if await self._describe(sid, target) != focused:
                raise BrowserError("Focus changed while waiting for approval; no key was pressed")
            watch = self._watch_navigation(sid, target)
            await self._key(sid, key)
            navigated = await self._settle(sid, watch)
            return await self._observe(target, sid, await self._result(target, navigated=navigated), tainted,
                                              "load" if navigated else "")

    async def _scroll(self, direction: str, amount: int, tab: str | None, tainted: bool) -> dict:
        deltas = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}
        if direction not in deltas:
            raise BrowserError("direction must be up, down, left or right")
        amount = min(max(int(amount), 1), 10000)
        target = await self._target(tab)
        url, host = await self._web_page(target)
        await self._allow("browser.read", host, f"Scroll {direction} on {host}", tainted)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            dx, dy = deltas[direction]
            moved = await self._send(sid, "Runtime.callFunctionOn", {
                "functionDeclaration": SCROLL_JS, "executionContextId": await self._world(sid, target),
                "arguments": [{"value": dx * amount}, {"value": dy * amount}], "returnByValue": True})
            outcome = (moved.get("result") or {}).get("value") or {}
            await asyncio.sleep(0.25)               # pages load more content as they scroll
            result = await self._result(target, scrolled=bool(outcome.get("moved")))
            if "percent" in outcome:
                result["position"] = f"{outcome['percent']}% of the way down"
            return await self._observe(target, sid, result, tainted)

    async def _screenshot(self, tab: str | None, full_page: bool, format: str, quality: int, tainted: bool) -> dict:
        format = {"jpg": "jpeg"}.get(str(format).lower(), str(format).lower())
        if format not in ("jpeg", "png", "webp"):
            raise BrowserError("format must be jpeg, png or webp")
        target = await self._target(tab)
        url, host = await self._web_page(target)
        await self._allow("browser.read", host, f"Screenshot {url}", tainted)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            params: dict[str, Any] = {"format": format}
            if format != "png":
                params["quality"] = min(max(int(quality), 1), 100)
            metrics = await self._send(sid, "Page.getLayoutMetrics")
            view = metrics.get("cssVisualViewport") or metrics.get("visualViewport") or {}
            if full_page:
                size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
                params["captureBeyondViewport"] = True
                params["clip"] = {"x": 0, "y": 0, "width": min(size.get("width", 1280), 4096),
                                  "height": min(size.get("height", 900), 16384), "scale": 1}
            data = (await self._send(sid, "Page.captureScreenshot", params, timeout=60))["data"]
            result = await self._result(target)
        digest = hashlib.sha256(base64.b64decode(data)).hexdigest()
        result.update(mime=f"image/{format}", data_b64=data, changed=self._shots.get(target) != digest,
                      # in CSS pixels, the space click_at takes points in, whatever the screen's pixel density
                      viewport=[view.get("clientWidth", 0), view.get("clientHeight", 0)])
        self._shots[target] = digest
        return result

    async def _handoff_to_owner(self, reason: str, tab: str | None, tainted: bool) -> dict:
        if self.headless:
            raise BrowserError("The browser is headless, so the owner cannot take over; "
                               "set browser.headless to false")
        target = await self._target(tab)
        info = await self._info(target)
        url = info.get("url", "")
        host = page_host(url) or "browser"
        reason = squash(str(reason))[:300] or "please take over"
        action = Action("browser.handoff", host, f"Please finish in the CodeRunner browser: {reason}",
                        {"url": url, "tab": self._tab_id(target)}, tainted)
        if self.policy.evaluate(action) is Decision.DENY:
            raise BrowserError("Handing the browser to the owner is disabled by policy")
        sid = await self._session(target)
        self._handoff.add(target)
        try:
            # The owner may need to pick files while in control.
            with contextlib.suppress(CDPError):
                await self._send(sid, "Page.setInterceptFileChooserDialog", {"enabled": False})
            with contextlib.suppress(CDPError):
                await self._send(sid, "Page.bringToFront")
            # Always a fresh request: a standing grant must not mark the hand-off as done.
            approval = await self.policy.request(action, self.origin)
            completed = await self.policy.wait(approval.id, HANDOFF_WAIT)
        finally:
            self._handoff.discard(target)
            if self._sessions.get(target) == sid:
                with contextlib.suppress(CDPError, BrowserError):
                    await self._send(sid, "Page.setInterceptFileChooserDialog", {"enabled": True})
        self._snapshots.pop(target, None)
        self._refs.pop(target, None)
        if target in self._sessions:
            return await self._result(target, completed=completed)
        return {"ok": True, "completed": completed, "tab": self._tab_id(target)}
