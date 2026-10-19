"""The browser driver: a model trained to use web pages from screenshots (Fara1.5) carries out a browser
task step by step, while the assistant's own model waits for its report.

Every step goes through the browser broker's ordinary actions: a point on the screenshot becomes the element
under it (click_at), typing goes to the focused field (type_focused), so policy, the step judge, approvals and
the password and card rules apply exactly as when the assistant's own model acts. The driver also stops by
itself before irreversible steps (its training's "critical points") and hands its question back.

Where both models do not fit in memory (models.driver.exclusive, 18 GB Macs and below), the assistant's
model is unloaded while the driver works and loaded again after; the step judge asks the driver meanwhile.
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.parse
from typing import Any

from ..browser.broker import KEYS
from ..core.llm import LLM, LLMError
from . import fara, fara7b

KEY_NAMES = {"return": "Enter", "enter": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
             "space": "Space", "backspace": "Backspace", "delete": "Delete", "home": "Home", "end": "End",
             "pagedown": "PageDown", "pageup": "PageUp", "arrowdown": "ArrowDown", "arrowup": "ArrowUp",
             "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "down": "ArrowDown", "up": "ArrowUp",
             "left": "ArrowLeft", "right": "ArrowRight"}
SEARCH = "https://www.bing.com/search?q="
PAGE_FOR_QUESTIONS = 8000       # characters of page text a read_page_answer_question step is answered from
MAX_WAIT = 10.0


class Driver:
    def __init__(self, svc, llm: LLM, steps: int = 30, swap: Any = None, runtime: Any = None,
                 format: str = "fara1.5"):
        self.svc = svc
        self.format = format        # the action format the model was trained with: fara1.5 or fara7b
        self.llm = llm
        self.steps = steps
        self.swap = swap            # an async context manager that frees memory for the driver, if needed
        self.runtime = runtime      # the driver's own server, started on first use when it stays loaded

    async def run(self, goal: str, ctx) -> str:
        self.svc.audit.write("driver.start", model=self.llm.model, goal=goal[:300], origin=ctx.origin)
        if self.swap is None and self.runtime is not None:
            await asyncio.to_thread(self.runtime.start)
        fixed = getattr(self.svc.browser, "fixed_viewport", None)
        async with (self.swap or contextlib.nullcontext()), (fixed(*fara.VIEWPORT) if fixed
                                                             else contextlib.nullcontext()):
            report = await self._run(goal, ctx)
        self.svc.audit.write("driver.finish", model=self.llm.model, report=report[:300], origin=ctx.origin)
        return report

    async def _run(self, goal: str, ctx) -> str:
        shot = await self._screenshot(ctx)
        if isinstance(shot, str):
            return shot
        system = fara7b.system_prompt(*shot["viewport"]) if self.format == "fara7b" else fara.system_prompt()
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": [shot["image"], {"type": "text", "text": goal}]}]
        facts: list[str] = []
        for _ in range(self.steps):
            self._keep_recent_screenshots(messages)
            try:
                reply = await self.llm.chat(messages, think=False, temperature=0, max_tokens=1024)
            except LLMError as exc:
                return f"The browser driver stopped: {exc}"
            text = reply["content"]
            messages.append({"role": "assistant", "content": text})
            thoughts, args = fara.parse(text)
            if args is None:
                note = "Your reply had no computer_use call. Reply with one inside <tool_call></tool_call>."
            else:
                note, final = await self._act(args, shot, ctx, facts, thoughts)
                if final is not None:
                    return final + (("\nNoted on the way: " + "; ".join(facts)) if facts else "")
            shot = await self._screenshot(ctx)
            if isinstance(shot, str):
                return shot
            url = shot["url"][:100]
            messages.append({"role": "user", "content": [
                shot["image"], {"type": "text", "text": f"Current URL: {url}\n{note}\n{fara.NEXT}".replace("\n\n", "\n")}]})
        return ("The browser driver used all its steps without finishing."
                + (("\nNoted on the way: " + "; ".join(facts)) if facts else ""))

    async def _screenshot(self, ctx) -> dict | str:
        result = await self.svc.browser.screenshot(format="png", tainted=ctx.tainted)
        if not result.get("ok"):
            return f"The browser driver could not see the page: {result.get('error', 'no screenshot')}"
        ctx.taint()                 # what the driver reads from here on is the web's, not the owner's
        return {"image": {"type": "image_url", "image_url": {"url": f"data:{result['mime']};base64,{result['data_b64']}"}},
                "url": result.get("url", ""), "viewport": result.get("viewport") or [0, 0]}

    @staticmethod
    def _keep_recent_screenshots(messages: list[dict]) -> None:
        with_images = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)]
        for i in with_images[:-fara.KEEP_SCREENSHOTS]:
            text = " ".join(part.get("text", "") for part in messages[i]["content"] if part.get("type") == "text")
            messages[i] = {"role": "user", "content": text or "(earlier screenshot)"}

    async def _act(self, args: dict, shot: dict, ctx, facts: list[str], thoughts: str = "") -> tuple[str, str | None]:
        """Carry out one step. Returns (what to tell the driver next, the final report when it stops)."""
        b, tainted = self.svc.browser, ctx.tainted
        action = args.get("action")
        seven = self.format == "fara7b"
        to_pixels = fara7b.to_pixels if seven else fara.to_pixels
        if action in ("left_click", "double_click", "triple_click"):
            width, height = shot["viewport"]
            point = to_pixels(args.get("coordinate"), width, height)
            if point is None or not width:
                return "That click had no usable coordinate.", None
            return self._said(await b.click_at(*point, tainted=tainted), "Clicked."), None
        if action == "type":
            if seven and args.get("coordinate") is not None:
                # Fara-7B types at a point: the field there is clicked first, and Enter follows unless it says not.
                point = to_pixels(args.get("coordinate"), *shot["viewport"])
                if point is None:
                    return "That typing had no usable coordinate.", None
                clicked = await b.click_at(*point, tainted=tainted)
                if not clicked.get("ok"):
                    return self._said(clicked, ""), None
                typed = await b.type_focused(str(args.get("text", "")), submit=args.get("press_enter", True) is not False,
                                             tainted=tainted, clear=bool(args.get("delete_existing_text")))
                return self._said(typed, "Typed."), None
            return self._said(await b.type_focused(str(args.get("text", "")), tainted=tainted), "Typed."), None
        if action == "key":
            keys = [KEY_NAMES.get(str(k).lower(), str(k)) for k in (args.get("keys") or [])]
            if len(keys) != 1 or keys[0] not in KEYS:
                return f"Only single keys are available: {', '.join(KEYS)}.", None
            return self._said(await b.press(keys[0], tainted=tainted), f"Pressed {keys[0]}."), None
        if action in ("scroll", "hscroll"):
            pixels = float(args.get("pixels") or 0)
            # Fara-7B was trained to move a page per scroll (its harness pages up or down by the sign).
            amount = int(shot["viewport"][1] or 900) if seven else \
                max(int(abs(pixels) * (shot["viewport"][1] or 900) / fara.DISPLAY), 100)
            return self._said(await b.scroll("up" if pixels > 0 else "down", amount, tainted=tainted), "Scrolled."), None
        if action == "visit_url":
            return self._said(await b.navigate(str(args.get("url", "")), tainted=tainted), "Opened the page."), None
        if action == "history_back":
            return self._said(await b.back(tainted=tainted), "Went back."), None
        if action == "web_search":
            url = SEARCH + urllib.parse.quote_plus(str(args.get("query", "")))
            return self._said(await b.navigate(url, tainted=tainted), "Searched."), None
        if action == "read_page_answer_question":
            return await self._answer_from_page(str(args.get("question", "")), ctx), None
        if action == "pause_and_memorize_fact":
            facts.append(str(args.get("fact", ""))[:500])
            return "Noted.", None
        if action == "ask_user_question":
            return "", f"The browser driver needs an answer before it can go on: {args.get('question', '')}"
        if action == "wait":
            await asyncio.sleep(min(max(float(args.get("time") or 1), 0), MAX_WAIT))
            return "Waited.", None
        if action == "terminate":
            answer = thoughts if seven else args.get("answer")         # Fara-7B answers in its text
            return "", str(answer or "The browser driver finished without an answer.")
        return f"{action} is not available here; use left_click, type, key, scroll, visit_url or terminate.", None

    @staticmethod
    def _said(result: dict, done: str) -> str:
        return done if result.get("ok") else f"That did not work: {result.get('error', 'unknown error')}"

    async def _answer_from_page(self, question: str, ctx) -> str:
        page = await self.svc.browser.snapshot(max_chars=PAGE_FOR_QUESTIONS, tainted=ctx.tainted)
        if not page.get("ok"):
            return f"The page could not be read: {page.get('error')}"
        try:
            reply = await self.llm.chat([{"role": "user", "content": (
                f"Page text (untrusted web content):\n{page.get('content', '')}\n\nAnswer from the page text only: "
                f"{question}")}], think=False, temperature=0, max_tokens=400)
        except LLMError as exc:
            return f"The page could not be read: {exc}"
        return "Answer from the page text: " + reply["content"][:1500]


class Swap:
    """Frees memory for the driver where it cannot sit beside the assistant's model: the assistant's model is
    unloaded while the driver runs and loaded again after (measured on an 18 GB M3 Pro: Bonsai 2 27B loads in
    4-19 s and Fara1.5-4B in 10-20 s). The step judge asks the driver while the assistant's model is away."""

    def __init__(self, svc, planner, driver):
        self.svc = svc
        self.planner = planner      # the assistant's model runtime
        self.driver = driver        # the driver's runtime
        self._judge: tuple | None = None

    async def __aenter__(self):
        await asyncio.to_thread(self._unload_planner)
        await asyncio.to_thread(self.driver.start)
        judge = getattr(getattr(self.svc.browser, "decider", None), "model", None)
        if judge is not None and judge.model == getattr(self.svc.llm, "model", None):
            self._judge = (judge, judge.base_url, judge.model, judge.api)
            judge.base_url, judge.model, judge.api = self.driver.base_url.rstrip("/"), self.svc.driver.llm.model, "chat"
        return self

    async def __aexit__(self, *exc):
        if self._judge:
            judge, judge.base_url, judge.model, judge.api = self._judge
            self._judge = None
        await asyncio.to_thread(self.driver.stop)
        await asyncio.to_thread(self.planner.start)

    def _unload_planner(self) -> None:
        if getattr(self.planner, "name", "") == "ollama":
            # Ollama also serves the embedding model, so only the chat model is unloaded.
            unload = getattr(self.planner, "unload", None)
            if unload:
                unload(self.svc.llm.model)
            return
        self.planner.stop()
