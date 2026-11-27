"""A blocked site goes on in the browser on the owner's Mac, with their OK.

The assistant browses in Chrome on the sandbox VM (browser.mode "computer"). When a page there turns out to be a
bot wall (walls.py; the broker marks the step's result with "wall" after giving it a few seconds to let the
browser through by itself), the owner is asked once, in the thread:

    "example.com blocks the sandbox browser. Continue in CodeRunner's built-in browser on your Mac?"
    Allow once / Always for this site / No

Allow: the page opens in a new tab of the Mac browser (app.py: the CodeRunner app's own browser, or a Chrome
window of the assistant's profile when the app is closed), and this task's steps go on there until it opens a
page on another site. Always: the same, now and for every later task, without asking; it is an ordinary grant
(browser.mac for that site) the owner sees and clears in Models and privacy. No: the owner clears the check in
the sandbox's browser themselves (the existing hand-off), and the task is not asked about that site again.

BrowserSwitch stands where the broker stood (svc.browser): every action goes to the broker of the lane the task
is on, so policy, the judge and approvals apply in both browsers exactly as before. Nothing here solves a
challenge, and nothing makes either browser look like something it is not.
"""

from __future__ import annotations

import contextlib
from typing import Any

from ..safety.policy import Action, Decision, current_run
from .broker import site
from .urls import page_host
from .walls import describe

MAC, VM = "mac", "vm"
QUESTION = "{site} blocks the sandbox browser. Continue in CodeRunner's built-in browser on your Mac?"
MOVED = ("{site} blocked the sandbox browser, so with the owner's OK this page is open in the browser on their Mac. "
         "Carry on here as usual.")
STILL_WALLED = ("This page is {wall}. It is the owner's step, never yours: call browser_handoff so they clear it, "
                "then carry on.")
DECLINED = ("This page is {wall}. The owner did not clear it; get what you need from another site, or say in the "
            "answer that this site turned the browser away.")
CLEARED = "The owner cleared the site's check. Read the page again before going on."
HANDOFF = "{site} is checking for a person. Clear the check, then click I'm done."
# Steps that can land on a new page, and so on a wall.
MOVING = frozenset({"navigate", "open", "click", "press", "type", "back", "select", "type_focused", "click_at",
                    "press_keys"})
# Steps that work on the page the task is on: they go to the browser of its lane.
ON_PAGE = frozenset({"snapshot", "close_tab", "items", "find", "wheel", "hover", "scroll", "screenshot", "picture",
                     "handoff", "tabs", "_label", "fixed_viewport", "screens_only"})


class BrowserSwitch:
    def __init__(self, primary, mac, policy, audit, launcher=None, bridge=None, config=None):
        self.primary = primary          # the sandbox's broker (browser.mode "computer")
        self.mac = mac                  # a BrowserBroker in mode "mac" with app.MacLauncher
        self.policy, self.audit = policy, audit
        self.mac_launcher = launcher    # app.MacLauncher: whether a Mac browser is there at all
        self.bridge = bridge            # app.AppBridge, for the import offer and the app's status
        self.config = config if config is not None else primary.config
        self._lanes: dict[str, str] = {}          # task -> MAC while its steps run in the Mac browser
        self._sites: dict[str, set[str]] = {}     # task -> sites the owner let it continue on the Mac
        self._declined: dict[str, set[str]] = {}  # task -> sites the owner said no for (not asked again)

    OWN = frozenset({"primary", "mac", "policy", "audit", "mac_launcher", "bridge", "config", "_lanes", "_sites",
                     "_declined"})

    def __setattr__(self, name: str, value: Any) -> None:
        # What the daemon sets on its browser later (restart, computer) is the sandbox broker's.
        if name in self.OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.primary, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self.OWN:
            raise AttributeError(name)
        if name in MOVING or name in ON_PAGE:
            broker = self._broker()
            attr = getattr(broker, name)
            if name in MOVING and broker is self.primary:
                return self._watched(attr)
            if name in MOVING:
                return self._mac_watched(attr)
            return attr
        return getattr(self.primary, name)

    # ---- lanes -----------------------------------------------------------------------------

    def _scope(self) -> str:
        return str(getattr(current_run.get(), "conversation", "") or "")

    def _broker(self, scope: str | None = None):
        return self.mac if self._lanes.get(self._scope() if scope is None else scope) == MAC else self.primary

    def lane(self, scope: str | None = None) -> str:
        return self._lanes.get(self._scope() if scope is None else scope, VM)

    def mac_available(self) -> bool:
        if self.mac.running():
            return True
        return self.mac_launcher is not None and self.mac_launcher.available(self.config)

    def _always(self, site_name: str) -> bool:
        return self.policy.evaluate(Action("browser.mac", site_name, QUESTION.format(site=site_name))) \
            is Decision.ALLOW

    def _goes_to_mac(self, url: str) -> bool:
        host = page_host(str(url or ""))
        if not host or host == "about:blank":
            return self.lane() == MAC
        name = site(host)
        return (name in self._sites.get(self._scope(), set()) or self._always(name)) and self.mac_available()

    async def navigate(self, url: str, tab: str | None = None, tainted: bool = False) -> dict:
        return await self._opening("navigate", url, tainted, tab=tab)

    async def open(self, url: str, tainted: bool = False) -> dict:
        return await self._opening("open", url, tainted)

    async def _opening(self, how: str, url: str, tainted: bool, **extra: Any) -> dict:
        """An address the model opens goes to the Mac browser when its site is let through there, and back to the
        sandbox's when it is not: the switch is for that site, not for the rest of the task."""
        scope = self._scope()
        if str(url or "").strip().startswith("/"):          # a path on the page the task is on
            broker = self._broker()
        elif self._goes_to_mac(url):
            if how == "navigate" and self._lanes.get(scope) != MAC:
                how = "open"            # a tab of its own there: never one of the owner's, nor another task's
            self._lanes[scope] = MAC
            broker = self.mac
        else:
            self._lanes.pop(scope, None)
            broker = self.primary
        if broker is self.mac:
            result = await (self.mac.open(url, tainted=tainted) if how == "open" else
                            self.mac.navigate(url, tainted=tainted, **extra))
            return self._on_mac(result)
        call = self.primary.open(url, tainted=tainted) if how == "open" else \
            self.primary.navigate(url, tainted=tainted, **extra)
        return await self._after(await call, tainted)

    def _watched(self, action):
        async def watched(*args, **kwargs):
            return await self._after(await action(*args, **kwargs), bool(kwargs.get("tainted")))
        return watched

    def _mac_watched(self, action):
        async def watched(*args, **kwargs):
            return self._on_mac(await action(*args, **kwargs))
        return watched

    def _on_mac(self, result: Any) -> Any:
        if isinstance(result, dict) and result.get("wall"):
            result["note"] = STILL_WALLED.format(wall=describe(result["wall"]))
        return result

    # ---- the owner's OK ---------------------------------------------------------------------

    async def _after(self, result: Any, tainted: bool) -> Any:
        """A step in the sandbox's browser that ended on a bot wall: the owner is asked whether the site may go on
        on the Mac; on yes it does, on no they clear the check in the sandbox (hand-off)."""
        if not (isinstance(result, dict) and result.get("ok") is not False and result.get("wall")):
            return result
        url, wall = str(result.get("url") or ""), str(result["wall"])
        host = page_host(url)
        if not host or host == "about:blank":
            return result
        name, scope = site(host), self._scope()
        if name in self._declined.get(scope, set()):
            result["note"] = DECLINED.format(wall=describe(wall))
            return result
        if not self.mac_available():
            return await self._hand_off(result, name, wall, tainted)
        answer = await self._ask(name, url, wall, tainted)
        if answer == "approved":
            moved = await self._move(name, url, tainted)
            if moved is not None:
                return moved
            return await self._hand_off(result, name, wall, tainted)
        self._declined.setdefault(scope, set()).add(name)
        if answer == "denied":
            return await self._hand_off(result, name, wall, tainted)
        result["note"] = DECLINED.format(wall=describe(wall))    # not answered: the owner is away
        return result

    async def _ask(self, name: str, url: str, wall: str, tainted: bool) -> str:
        """approved, denied or expired. A standing grant (Always for this site) answers without asking; a rule
        that denies browser.mac for the site answers no."""
        detail: dict = {"url": url, "wall": wall, "site": name}
        if self.bridge is not None and self.bridge.connected():
            import inspect
            from . import cookies
            found = cookies.installed()
            if found:
                # Offer opt-in sign-in import on every site switch unless the Mac browser already holds cookies for it.
                list_fn = getattr(self.bridge, "cookie_domains", None)
                existing = await list_fn() if inspect.iscoroutinefunction(list_fn) else []
                has_site = any(
                    str(d.get("domain") or "") == name or str(d.get("domain") or "").endswith("." + name)
                    for d in (existing or [])
                )
                if not has_site:
                    detail["sign_in"] = found[0]
                    detail["sign_in_browsers"] = found
        action = Action("browser.mac", name, QUESTION.format(site=name), detail, tainted)
        decision = self.policy.evaluate(action)
        if decision is not Decision.ASK:
            self.audit.write("policy", kind=action.kind, target=name, decision=decision.value,
                             origin=self.primary.origin, tainted=tainted)
            return "approved" if decision is Decision.ALLOW else "denied"
        approval = await self.policy.request(action, self.primary.origin)
        timeout = float(self.config.get("agent.approval_timeout", 300))
        await self.policy.wait(approval.id, timeout)
        status = getattr(self.policy.get(approval.id), "status", "expired")
        return status if status in ("approved", "denied") else "expired"

    async def _move(self, name: str, url: str, tainted: bool) -> dict | None:
        """Open the page in a new tab of the Mac browser and put the task there. None when it cannot."""
        from . import app
        scope = self._scope()
        self._sites.setdefault(scope, set()).add(name)
        result = await self.mac.open(url, tainted=tainted)
        if not isinstance(result, dict) or result.get("ok") is False:
            self._sites[scope].discard(name)
            self.audit.write("browser.switch_failed", site=name, error=str((result or {}).get("error"))[:200])
            return None
        self._lanes[scope] = MAC
        app.mark_used()
        self.audit.write("browser.switched", site=name, to=self.mac_where())
        result = self._on_mac(result)
        if not result.get("note"):
            result["note"] = MOVED.format(site=name)
        return result

    def mac_where(self) -> str:
        launcher = self.mac_launcher
        return "app" if launcher is not None and launcher.app_open() else "chrome"

    async def _hand_off(self, result: dict, name: str, wall: str, tainted: bool) -> dict:
        """The existing hand-off: the owner clears the check in the sandbox's browser, or skips it."""
        done = await self.primary.handoff(HANDOFF.format(site=name), tainted=tainted)
        if isinstance(done, dict) and done.get("completed"):
            return {**{k: v for k, v in result.items() if k not in ("content", "changes", "wall")},
                    **{k: v for k, v in done.items() if k != "completed"}, "note": CLEARED}
        result["note"] = DECLINED.format(wall=describe(wall))
        return result

    # ---- the rest of the broker's surface, for both browsers ----------------------------------

    async def stop(self) -> dict:
        with contextlib.suppress(Exception):
            await self.mac.stop()
        return await self.primary.stop()

    async def peek(self, *args, scope: str | None = None, **kwargs):
        return await self._broker(scope).peek(*args, scope=scope, **kwargs)

    async def focus(self, scope: str) -> bool:
        return await self._broker(scope).focus(scope)

    async def tidy(self, keep: int = 1, scope: str | None = None) -> int:
        scope = self._scope() if scope is None else scope
        closed = await self.primary.tidy(keep, scope)
        if self.mac.running():
            closed += await self.mac.tidy(keep, scope)
        else:
            self.mac.release(scope)
        self._forget(scope)
        if (self.mac.running() and self.mac_launcher is not None
                and callable(getattr(self.mac, "stop", None))
                and getattr(self.mac_launcher, "using_fallback", lambda: False)() is True
                and getattr(self.mac_launcher, "app_open", lambda: False)() is True
                and MAC not in self._lanes.values()):
            await self.mac.stop()
        return closed

    def release(self, scope: str) -> None:
        self.primary.release(scope)
        self.mac.release(scope)
        self._forget(scope)

    def _forget(self, scope: str) -> None:
        """A finished task's switch ends with it; "Always" sites stay, as grants."""
        self._lanes.pop(scope, None)
        self._sites.pop(scope, None)
        self._declined.pop(scope, None)
