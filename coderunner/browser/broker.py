"""Guarded control of the user's local browser.

The broker drives a Chromium-family browser on the host, with a dedicated
profile and (normally) a visible window the owner can watch and take over.
The model gets a small action API (open, read, click, type, ...) and never
raw CDP, script evaluation, cookies or storage. Every action passes Policy:

  browser.navigate   opening a URL or going back
  browser.read       snapshot, screenshot, tab list, scrolling
  browser.interact   clicks, typing and key presses
  browser.sensitive  the same when the target looks like buying, paying,
                     sending, deleting, publishing... (one-time approval only)
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

from .. import decision, paths
from ..policy import Action, Decision, _matches
from . import launcher as default_launcher
from .cdp import CDPError, CDPPipe


class BrowserError(Exception):
    """An action could not be done; the message is written for the agent."""


NAV_TIMEOUT = 20.0
NAV_START_WINDOW = 0.5      # how long a click or key press gets to start a navigation
HANDOFF_WAIT = 600
SPAWN_WINDOW = 10.0         # popups opened by an action follow its navigation rules for this long

INTERACTIVE = frozenset({
    "link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "option", "switch", "slider", "spinbutton", "listbox",
})
INLINE = frozenset({"strong", "emphasis", "mark", "subscript", "superscript", "time", "code", "abbr",
                    "insertion", "deletion"})
CHECKABLE = frozenset({"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"})
# Controls that commit something. Nouns that merely name a section ("Your orders", "Manage your
# booking", "Check-out date") are left out; the optional decision model catches what words miss.
SENSITIVE = re.compile(
    r"\b(?:buy(?:ing)?|pay(?:s|ing|ment|ments)?|purchas(?:e|es|ing)|check\s*-?\s*out(?!\s*-?\s*(?:date|time|day))"
    r"|(?:place|submit|confirm)\s+(?:your\s+|my\s+|the\s+)?order|order\s+now|place\s+(?:a\s+)?bid"
    r"|(?:un)?subscri(?:be|bing|ption)|donat(?:e|es|ing|ion)|transfer(?:s|ring)?|send(?:ing)?"
    r"|delet(?:e|es|ing)|remov(?:e|es|ing)|trash|merge|submit|invit(?:e|es|ing)"
    r"|confirm(?:s|ing|ation)?|book|complete\s+(?:booking|purchase|payment)|reserv(?:e|es|ing|ation)"
    r"|finish\s+cancellation|cancel\s+(?:my\s+|your\s+)?(?:membership|subscription|plan|order|booking|account)"
    r"|(?:sign|log)\s*-?\s*out\s+(?:of\s+)?(?:all|every)"
    r"|post(?:ing)?|publish(?:ing)?|shar(?:e|es|ing)"
    # Words that could answer one of the owner's approval prompts in a chat client.
    r"|approv(?:e|es|ed|ing|al|als)|allow(?:s|ed|ing)?|authori[sz](?:e|es|ed|ing|ation)"
    r"|grant(?:s|ed|ing)?|den(?:y|ies|ied|ying)|permit(?:s|ted|ting)?)\b", re.I)
# Web chat clients: anything typed or clicked there could message people or answer
# an approval request on a paired channel, so every interaction is sensitive.
MESSAGING_HOSTS = ("web.telegram.org", "*.web.telegram.org", "web.whatsapp.com", "messages.google.com",
                   "signal.org", "*.signal.org", "discord.com", "*.discord.com", "discordapp.com",
                   "*.discordapp.com", "slack.com", "*.slack.com")
TEXT_LIMIT = 10000          # longer page text is cut, and the action is then treated as sensitive
SECRET_AUTOCOMPLETE = frozenset({"current-password", "new-password", "one-time-code"})
FRAME_TAGS = frozenset({"iframe", "frame", "object", "embed"})
KEYS = {  # name -> (windows virtual key code, text)
    "Enter": (13, "\r"), "Tab": (9, ""), "Escape": (27, ""), "ArrowDown": (40, ""), "ArrowUp": (38, ""),
    "Backspace": (8, ""), "PageDown": (34, ""), "PageUp": (33, ""),
}
DENIED_PERMISSIONS = ("geolocation", "notifications", "camera", "microphone", "midi", "clipboard-read",
                      "display-capture", "idle-detection", "window-management", "local-fonts",
                      "local-network-access", "payment-handler", "nfc")
LOCAL_NETS = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.168.0.0/16", "224.0.0.0/4", "255.255.255.255/32",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8"))
PASSWORD_HINT = ("This is a password or one-time-code field. The owner types those, never the assistant: "
                 "call handoff(reason) so they can finish it in the CodeRunner browser window.")

# Broker-owned constant scripts. They run in an isolated world, so page scripts
# cannot tamper with the prototypes they use; the model never supplies script.
DESCRIBE_JS = r"""function(closest) {
  let el = this.nodeType === 1 ? this : this.parentElement;
  if (el && closest) el = el.closest('a,button,input,select,textarea,summary,[role],[onclick]') || el;
  if (!el) return null;
  let truncated = false;
  const clean = s => {
    s = String(s || '').replace(/\s+/g, ' ').trim();
    if (s.length > LIMIT) { truncated = true; s = s.slice(0, LIMIT); }
    return s;
  };
  const text = e => [e.getAttribute('aria-label'), e.getAttribute('title'), e.innerText,
                     e.tagName === 'INPUT' ? e.value : ''].map(clean).filter(Boolean).join(' / ');
  const type = (el.getAttribute('type') || '').toLowerCase();
  const field = el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && !/^(submit|button|reset|image)$/.test(type));
  const form = el.form || el.closest('form');
  const submit = form && form.querySelector('button[type=submit],input[type=submit],input[type=image],button:not([type])');
  return {tag: el.tagName.toLowerCase(), type, autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
          text: field ? clean(el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.name) : text(el),
          submit: submit ? text(submit) : '', truncated};
}""".replace("LIMIT", str(TEXT_LIMIT))
FOCUSED_JS = r"""(() => {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  return el && el !== document.body && el !== document.documentElement ? el : null;
})()"""
CONTAINS_JS = "function(other) { for (let n = other; n; n = n.parentNode || n.host) if (n === this) return true; return false; }"
SELECT_ALL_JS = r"""function() {
  try { if (typeof this.select === 'function') { this.select(); return; } } catch (e) {}
  const range = document.createRange(); range.selectNodeContents(this);
  const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
}"""


# ---- URLs and hosts -------------------------------------------------------

def _ipv4_number(host: str) -> ipaddress.IPv4Address | None:
    """WHATWG IPv4 parsing, so 127.1 or 0x7f.0.0.1 are seen the way the browser sees them."""
    labels = host.split(".")
    if not re.fullmatch(r"\d+|0x[0-9a-f]*", labels[-1]):
        return None
    if len(labels) > 4:
        raise BrowserError(f"Invalid IPv4 address {host}")
    numbers = []
    for label in labels:
        try:
            if label.startswith("0x"):
                numbers.append(int(label[2:] or "0", 16))
            elif len(label) > 1 and label.startswith("0"):
                numbers.append(int(label, 8))
            else:
                numbers.append(int(label))
        except ValueError:
            raise BrowserError(f"Invalid IPv4 address {host}") from None
    if any(n > 255 for n in numbers[:-1]) or numbers[-1] >= 256 ** (5 - len(numbers)):
        raise BrowserError(f"Invalid IPv4 address {host}")
    return ipaddress.IPv4Address(numbers[-1] + sum(n << 8 * (3 - i) for i, n in enumerate(numbers[:-1])))


def parse_url(url: str) -> tuple[str, str]:
    """Normalise a URL to (url, host). Only http(s) and about:blank are accepted.

    The URL is rebuilt from the parsed parts so the browser opens exactly the
    host that policy saw.
    """
    url = (url or "").strip()
    if not url:
        raise BrowserError("URL is empty")
    if url.lower() == "about:blank":
        return "about:blank", "about:blank"
    if "\\" in url or any(ord(c) <= 0x20 or ord(c) == 0x7f for c in url):
        raise BrowserError("URL must not contain spaces, backslashes or control characters; percent-encode them")
    scheme = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    if scheme is None or re.fullmatch(r"[^/:?#@]+:\d+(?:[/?#].*)?", url, re.S):
        url = "https://" + url                      # bare host such as example.com or example.com:8080/x
    elif scheme.group(1).lower() not in ("http", "https"):
        raise BrowserError(f"Only http and https URLs are allowed, not {scheme.group(1).lower()}:")
    parts = urllib.parse.urlsplit(url)
    if "@" in parts.netloc:
        raise BrowserError("URLs with embedded credentials are not allowed")
    try:
        port = parts.port
    except ValueError:
        raise BrowserError("URL has an invalid port") from None
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise BrowserError("URL has no host")
    if ":" in host:
        try:
            host = ipaddress.IPv6Address(host).compressed
        except ValueError:
            raise BrowserError(f"Invalid host {host}") from None
        netloc = f"[{host}]"
    else:
        try:
            host = host.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise BrowserError(f"Invalid host name {host}") from None
        if not re.fullmatch(r"[a-z0-9_.-]+", host):
            raise BrowserError(f"Invalid host name {host}")
        ip = _ipv4_number(host)
        if ip is not None:
            host = str(ip)
        netloc = host
    if port is not None:
        netloc += f":{port}"
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query,
                                    parts.fragment)), host


def _local_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in LOCAL_NETS if net.version == ip.version)


def local_problem(host: str) -> str | None:
    """Why a host must not be opened without an explicit browser.allow entry, if it must not."""
    if host == "localhost" or host.endswith(".localhost"):
        return f"{host} is this computer"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    return f"{host} is a local or private network address" if _local_ip(ip) else None


def page_host(url: str) -> str | None:
    """Policy target for a page URL; None for pages that are not ordinary web pages."""
    if url.startswith("about:blank"):
        return "about:blank"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme in ("http", "https") and parts.hostname:
        return parts.hostname.rstrip(".")
    return None


def _same_site(a: str, b: str) -> bool:
    return a == b or a.removeprefix("www.") == b.removeprefix("www.")


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


def sensitive(*texts: str) -> bool:
    return any(text and SENSITIVE.search(text) for text in texts)


# ---- accessibility snapshot -----------------------------------------------

def _value(field: Any) -> Any:
    return field.get("value") if isinstance(field, dict) else None


def _squash(text: str) -> str:
    return " ".join(text.split())


def _short_url(url: str, page_url: str) -> str:
    target, page = urllib.parse.urlsplit(url), urllib.parse.urlsplit(page_url)
    if target.scheme in ("http", "https") and (target.scheme, target.netloc) == (page.scheme, page.netloc):
        url = urllib.parse.urlunsplit(("", "", target.path or "/", target.query, target.fragment))
    return url if len(url) <= 120 else url[:119] + "…"


def _element_line(ref: int, role: str, name: str, node: dict, props: dict, page_url: str, secret: bool) -> str:
    line = f"[{ref}] {role}"
    if name:
        line += " " + json.dumps(name[:200], ensure_ascii=False)
    value = _value(node.get("value"))
    if secret:
        line += " (password field: the owner fills it in, use handoff)"
    elif value not in (None, "") and str(value) != name:
        line += " value=" + json.dumps(str(value)[:200], ensure_ascii=False)
    checked = props.get("checked")
    if role in CHECKABLE and checked is not None:
        line += {"true": " checked", "mixed": " mixed"}.get(str(checked).lower(), " unchecked")
    if props.get("expanded") is not None:
        line += " expanded" if props["expanded"] in (True, "true") else " collapsed"
    for flag in ("selected", "disabled"):
        if props.get(flag) in (True, "true"):
            line += f" {flag}"
    if role == "link" and props.get("url"):
        line += " -> " + _short_url(str(props["url"]), page_url)
    return line


def render_ax(nodes: list[dict], page_url: str = "", secret: frozenset | set = frozenset()) -> tuple[str, dict]:
    """Compact text for an Accessibility.getFullAXTree result.

    Headings become "# name", interactive elements "[N] role "name" ...", and
    other text is joined per block and deduplicated. Returns (text, refs)
    where refs maps N to the element's backendDOMNodeId, role and name.
    """
    by_id = {node["nodeId"]: node for node in nodes}
    roots = [node["nodeId"] for node in nodes if node.get("parentId") not in by_id]
    lines: list[str] = []
    refs: dict[int, dict] = {}
    seen: set[str] = set()
    parts: list[str] = []
    owner: list[Any] = [None]

    def flush() -> None:
        text = _squash("".join(parts))
        parts.clear()
        owner[0] = None
        if text and text not in seen:
            seen.add(text)
            lines.append(text)

    # Iterative depth-first walk: (node id, text block owner, name that already covers the text).
    stack: list[tuple[str, Any, str]] = [(root, None, "") for root in reversed(roots)]
    while stack:
        node_id, block, covered = stack.pop()
        node = by_id.get(node_id)
        if node is None:
            continue
        children = node.get("childIds") or []
        role = str(_value(node.get("role")) or "")
        raw_name = str(_value(node.get("name")) or "")
        name = _squash(raw_name)

        def descend(new_block: Any, new_covered: str) -> None:
            stack.extend((child, new_block, new_covered) for child in reversed(children))

        if node.get("ignored"):
            descend(block, covered)
        elif role == "StaticText":
            if name and not (covered and name in covered):
                if block != owner[0]:
                    flush()
                    owner[0] = block
                parts.append(raw_name)
        elif role == "LineBreak":
            flush()
        elif role == "InlineTextBox":
            continue
        elif role == "heading":
            flush()
            props = {p.get("name"): _value(p.get("value")) for p in node.get("properties", [])}
            if name:
                try:
                    level = min(max(int(props.get("level") or 1), 1), 6)
                except (TypeError, ValueError):
                    level = 1
                lines.append("#" * level + " " + name)
            descend(node_id, name)
        elif role in INTERACTIVE and node.get("backendDOMNodeId"):
            flush()
            props = {p.get("name"): _value(p.get("value")) for p in node.get("properties", [])}
            backend = node["backendDOMNodeId"]
            ref = len(refs) + 1
            is_secret = backend in secret
            refs[ref] = {"backend": backend, "role": role, "name": name, "password": is_secret}
            lines.append(_element_line(ref, role, name, node, props, page_url, is_secret))
            if role not in ("textbox", "searchbox") and not props.get("editable"):
                descend(node_id, name)      # the value is already shown; skip editor internals
        elif role in INLINE:
            descend(block, covered)
        else:
            descend(node_id, covered)
    flush()
    return "\n".join(lines), refs


# ---- broker ---------------------------------------------------------------

class BrowserBroker:
    def __init__(self, config, policy, audit, launcher=None, decider=None):
        self.config = config
        self.policy = policy
        self.audit = audit
        self.launcher = launcher or default_launcher
        self.decider = decider      # optional decision.ActionCheck: a second opinion on committing steps
        self.origin = "agent"
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
    def headless(self) -> bool:
        return bool(self.config.get("browser.headless", False))

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

    async def snapshot(self, tab: str | None = None, max_chars: int = 12000, tainted: bool = False) -> dict:
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
            executable = self.launcher.find_browser(self.config)
            if not executable:
                raise BrowserError("No Chrome, Brave, Edge or Chromium found. Install one or set browser.executable.")
            profile = paths.assistant_dir() / "browser-profile"
            try:
                self._process, self._cdp = await self.launcher.launch(executable, profile, self.headless)
            except (OSError, CDPError) as exc:
                raise BrowserError(f"Could not start the browser: {exc}") from None
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
        await asyncio.to_thread(self.launcher.terminate, process)
        self._process, self._cdp = None, None
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
        for table in (self._refs, self._snapshots, self._shots, self._dialogs, self._dismissed, self._blocked,
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
        """The decision model's view, for steps the keyword rules consider ordinary."""
        if self.decider is None:
            return False
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
                if _local_ip(ip):
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
        if any(_same_site(host, other) for other in reference if other):
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
            return await self._result(target, loaded=loaded)

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
            return await self._result(target)

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

    async def _snapshot(self, tab: str | None, max_chars: int, tainted: bool) -> dict:
        target = await self._target(tab)
        url, host = await self._web_page(target)
        await self._allow("browser.read", host, f"Read {url}", tainted)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            nodes = (await self._send(sid, "Accessibility.getFullAXTree", timeout=60)).get("nodes", [])
            secret = await self._secret_fields(sid, nodes)
            content, refs = render_ax(nodes, url, secret)
            self._refs[target] = refs
            result = await self._result(target)
        digest = hashlib.sha256(f"{result['url']}\n{content}".encode()).hexdigest()
        result["changed"] = self._snapshots.get(target) != digest
        self._snapshots[target] = digest
        limit = max(100, int(max_chars))
        if len(content) > limit:
            content = content[:limit] + f"\n… [truncated {len(content) - limit} characters]"
        result["content"] = content
        return result

    async def _secret_fields(self, sid: str, nodes: list[dict]) -> set[int]:
        candidates = [node["backendDOMNodeId"] for node in nodes
                      if not node.get("ignored") and node.get("backendDOMNodeId")
                      and _value(node.get("role")) in ("textbox", "searchbox", "combobox")][:100]

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
        element = self._ref(target, ref)
        sid = await self._session(target)
        async with self._acting(target, sid, tainted):
            plan = await self._click_plan(sid, target, element)
        kind = plan["kind"]
        if kind == "browser.interact" and await self._commits(
                host, url, "click", plan["role"], plan["name"] or plan["text"], nearby=plan["text"]):
            kind = "browser.sensitive"
        await self._allow(kind, host, f"Click {plan['label']} on {host}", tainted,
                          url=url, ref=ref, label=plan["label"])
        async with self._acting(target, sid, tainted):
            # The approval covers exactly what was shown: the same element under the pointer, the same
            # label and frame. Anything different means the page changed, so nothing is clicked.
            now = await self._click_plan(sid, target, element)
            if _fingerprint(now) != _fingerprint(plan):
                raise BrowserError("The page changed while waiting for approval; nothing was clicked. "
                                   "Take a new snapshot")
            x, y = now["x"], now["y"]
            await self._send(sid, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
            if _fingerprint(await self._probe(sid, target, element, x, y)) != _fingerprint(plan):
                raise BrowserError("The page changed under the pointer; nothing was clicked. Take a new snapshot")
            watch = self._watch_navigation(sid, target)
            for event in ("mousePressed", "mouseReleased"):
                await self._send(sid, "Input.dispatchMouseEvent", {
                    "type": event, "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1})
            navigated = await self._settle(sid, watch)
            return await self._result(target, navigated=navigated)

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
        risky = sensitive(name, text) or own.get("truncated") or (cover is not None and (
            sensitive(cover.get("text", "")) or cover.get("truncated") or cover.get("tag") in FRAME_TAGS))
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
            risky = field == "payment" or own.get("truncated") or (
                submit and sensitive(own.get("submit", ""), own.get("text", ""))) or await self._commits(
                host, url, "type and press Enter" if submit else "type", element["role"], element["name"],
                typed=text, nearby=f"submits: {own.get('submit')}" if submit and own.get("submit") else "")
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
            result = await self._result(target, navigated=navigated)
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
            if key == "Enter":
                # Enter activates the focused control or submits its form.
                risky = focused.get("tag") in FRAME_TAGS or focused.get("truncated") or \
                    sensitive(focused.get("text", ""), focused.get("submit", "")) or await self._commits(
                        host, url, "press Enter in", focused.get("tag", ""), focused.get("text", ""),
                        nearby=f"submits: {focused['submit']}" if focused.get("submit") else "")
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
            return await self._result(target, navigated=navigated)

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
            metrics = await self._send(sid, "Page.getLayoutMetrics")
            viewport = metrics.get("cssVisualViewport") or metrics.get("visualViewport") or {}
            dx, dy = deltas[direction]
            await self._send(sid, "Input.dispatchMouseEvent", {
                "type": "mouseWheel", "x": viewport.get("clientWidth", 800) / 2,
                "y": viewport.get("clientHeight", 600) / 2, "deltaX": dx * amount, "deltaY": dy * amount})
            await asyncio.sleep(0.25)
            return await self._result(target)

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
            if full_page:
                metrics = await self._send(sid, "Page.getLayoutMetrics")
                size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
                params["captureBeyondViewport"] = True
                params["clip"] = {"x": 0, "y": 0, "width": min(size.get("width", 1280), 4096),
                                  "height": min(size.get("height", 900), 16384), "scale": 1}
            data = (await self._send(sid, "Page.captureScreenshot", params, timeout=60))["data"]
            result = await self._result(target)
        digest = hashlib.sha256(base64.b64decode(data)).hexdigest()
        result.update(mime=f"image/{format}", data_b64=data, changed=self._shots.get(target) != digest)
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
        reason = _squash(str(reason))[:300] or "please take over"
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
