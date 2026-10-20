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
from datetime import datetime
from typing import Any

from ..browser.broker import KEYS
from ..core.llm import LLM, LLMError
from . import fara, fara7b, holo

# Key names as Fara writes them, to the browser's (Microsoft's harness maps them so: key_mapping.py).
KEY_NAMES = {"return": "Enter", "enter": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
             "space": "Space", "backspace": "Backspace", "delete": "Delete", "home": "Home", "end": "End",
             "pagedown": "PageDown", "pageup": "PageUp", "arrowdown": "ArrowDown", "arrowup": "ArrowUp",
             "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "down": "ArrowDown", "up": "ArrowUp",
             "left": "ArrowLeft", "right": "ArrowRight", "ctrl": "Control", "control": "Control", "alt": "Alt",
             "option": "Alt", "cmd": "Meta", "command": "Meta", "meta": "Meta", "super": "Meta", "win": "Meta",
             "shift": "Shift", **{f"f{n}": f"F{n}" for n in range(1, 13)}}
SEARCH = "https://www.bing.com/search?q={}&FORM=QBLH"
PAGE_FOR_QUESTIONS = 20000      # characters of page text a read_page_answer_question step is answered from
OBSERVATION_CHARS = 1000        # an answer read from the page, cut in the middle past this (as in training)
URL_CHARS = 100
WAIT = 3.0                      # a wait without a time, as in Microsoft's harness
MAX_WAIT = 10.0
FARA7B_SCROLL = 400             # pixels a Fara-7B scroll moves in Microsoft's harness
EXTRACT = ("You are given the markdown content of a web page. Answer the user's question based solely on this "
           "content. Be concise and extract only the relevant information.")
# The same step again and again (a Google Maps run clicked "Reverse starting point and destination" 18 times in
# a row): from the third the driver is told, from the sixth the step is not taken. A "+" pressed five times
# still works; scrolling and waiting are left alone.
REPEAT_NOTE, REPEAT_STOP = 3, 6
REPEATING = ("You have taken this same step {n} times in a row and are no closer: do something different, or stop "
             "and say what you found.")
SAME_POINT = 25                 # coordinates this close (grid units or pixels) count as the same point
UNPARSED_TRIES = 3              # a step whose tool call the runtime could not read is asked again, twice at most


class Driver:
    def __init__(self, svc, llm: LLM, steps: int = 30, swap: Any = None, runtime: Any = None,
                 format: str = "fara1.5"):
        self.svc = svc
        self.format = format        # the action format the model was trained with: fara1.5, fara7b or holo
        self.llm = llm
        self.steps = steps
        self.swap = swap            # an async context manager that frees memory for the driver, if needed
        self.runtime = runtime      # the driver's own server, started on first use when it stays loaded

    async def run(self, goal: str, ctx) -> str:
        self.svc.audit.write("driver.start", model=self.llm.model, goal=goal[:300], origin=ctx.origin)
        if self.swap is None and self.runtime is not None:
            await asyncio.to_thread(self.runtime.start)
        fixed = getattr(self.svc.browser, "fixed_viewport", None)
        quiet = getattr(self.svc.browser, "screens_only", None)
        viewport = holo.VIEWPORT if self.format == "holo" else fara.VIEWPORT
        async with (self.swap or contextlib.nullcontext()), (fixed(*viewport) if fixed
                                                             else contextlib.nullcontext()), \
                (quiet() if quiet else contextlib.nullcontext()):
            report = await self._run(goal, ctx)
        self.svc.audit.write("driver.finish", model=self.llm.model, report=report[:300], origin=ctx.origin)
        return report

    async def _run(self, goal: str, ctx) -> str:
        self.tab: str | None = None            # a tab the page opened is followed, as Microsoft's harness does
        self.pointer: tuple[float, float] | None = None     # where the mouse last was: the wheel turns there
        if self.format == "holo":
            return await self._run_holo(goal, ctx)
        shot = await self._screenshot(ctx)
        if isinstance(shot, str):
            return shot
        system = fara7b.system_prompt(*shot["viewport"]) if self.format == "fara7b" else fara.system_prompt()
        # The date, which the model cannot know: without it "released 4 days ago" became April, and a 14-day
        # forecast's "tomorrow" the 27th.
        task = f"{goal}\n\nToday is {datetime.now().astimezone().strftime('%A %-d %B %Y')}."
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": [shot["image"], {"type": "text", "text": task}]}]
        facts: list[str] = []
        last, repeats = None, 0
        for _ in range(self.steps):
            self._keep_recent_screenshots(messages)
            try:
                reply = await self.llm.chat(messages, think=False, temperature=0, max_tokens=2048)
            except LLMError as exc:
                return f"The browser driver stopped: {exc}"
            text = reply["content"]
            messages.append({"role": "assistant", "content": text})
            thoughts, args = fara.parse(text)
            if args is None:
                note = "ERROR: Your reply had no computer_use call. Reply with one inside <tool_call></tool_call>."
            else:
                key = self._step_key(args)
                repeats, last = (repeats + 1 if key is not None and key == last else 1), key
                if key is not None and repeats >= REPEAT_STOP:
                    note = REPEATING.format(n=repeats) + " That step was not taken."
                else:
                    note, final = await self._act(args, shot, ctx, facts, thoughts)
                    if final is not None:
                        return final + (("\nNoted on the way: " + "; ".join(facts)) if facts else "")
                    if key is not None and repeats >= REPEAT_NOTE:
                        note = f"{note} {REPEATING.format(n=repeats)}".strip()
            shot = await self._screenshot(ctx)
            if isinstance(shot, str):
                return shot
            # As in training: the address (without its query, cut at 100 characters), then only what the model
            # must hear (an answer it asked the page for, an error), then the same request every step.
            lines = [f"Current URL: {self._trimmed(shot['url'])}"] if shot["url"] else []
            messages.append({"role": "user", "content": [shot["image"], {"type": "text", "text": "\n".join(
                lines + ([note] if note else []) + [fara.NEXT])}]})
        return ("The browser driver used all its steps without finishing."
                + (("\nNoted on the way: " + "; ".join(facts)) if facts else ""))

    async def _run_holo(self, goal: str, ctx) -> str:
        """Holo's own harness (see holo.py): its tools in every request, the screen in <observation> tags, a tool
        result with the address after each action, thinking on; it stops by calling answer."""
        shot = await self._screenshot(ctx)
        if isinstance(shot, str):
            return shot
        task = f"{goal}\n\nToday is {datetime.now().astimezone().strftime('%A %-d %B %Y')}."
        messages: list[dict] = [{"role": "system", "content": holo.SYSTEM}, {"role": "user", "content": task}]
        facts: list[str] = []
        last, repeats = None, 0
        for _ in range(self.steps):
            messages.append(holo.observation(shot["image"]))
            holo.keep_recent_screenshots(messages)
            for attempt in range(UNPARSED_TRIES):
                try:
                    # llama.cpp answers 500 when it cannot read a reply's tool call ("does not match the expected
                    # ... format", a call cut short by a long reasoning, say): the step is asked again without
                    # requiring a call, so a call it cannot read comes back as text, which is read here.
                    reply = await self.llm.chat(messages, tools=holo.TOOLS,
                                                tool_choice="required" if attempt == 0 else "auto",
                                                temperature=holo.TEMPERATURE, max_tokens=holo.MAX_TOKENS,
                                                chat_template_kwargs={"enable_thinking": True})
                    break
                except LLMError as exc:
                    if "does not match" not in str(exc) or attempt == UNPARSED_TRIES - 1:
                        return f"The browser driver stopped: {exc}"
            call = holo.call_of(reply)
            # Only the parsed reply goes back into the history, never the reasoning (core concepts: reasoning).
            messages.append({"role": "assistant", "content": reply["content"],
                             **({"tool_calls": reply["tool_calls"][:1]} if call else {})})
            if call is None:
                continue                             # a step without a call is not a stop
            name, args, call_id = call
            if name == "answer":
                return (str(args.get("content") or "The browser driver finished without an answer.")
                        + (("\nNoted on the way: " + "; ".join(facts)) if facts else ""))
            step = holo.as_fara(name, args)
            key = self._step_key(step)
            repeats, last = (repeats + 1 if key is not None and key == last else 1), key
            if key is not None and repeats >= REPEAT_STOP:
                note = REPEATING.format(n=repeats) + " That step was not taken."
            else:
                note, _ = await self._act(step, shot, ctx, facts)
                if key is not None and repeats >= REPEAT_NOTE:
                    note = f"{note} {REPEATING.format(n=repeats)}".strip()
            shot = await self._screenshot(ctx)
            if isinstance(shot, str):
                return shot
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": f"{note + ' ' if note else 'Done. '}Current URL: {shot['url']}"})
        return "The browser driver used all its steps without finishing."

    @staticmethod
    def _trimmed(url: str) -> str:
        url = url.split("?", 1)[0]
        return url[:URL_CHARS] + " ..." if len(url) > URL_CHARS else url

    async def _screenshot(self, ctx) -> dict | str:
        result = await self.svc.browser.screenshot(format="png", tainted=ctx.tainted, tab=self.tab)
        if not result.get("ok"):
            return f"The browser driver could not see the page: {result.get('error', 'no screenshot')}"
        ctx.taint()                 # what the driver reads from here on is the web's, not the owner's
        return {"image": {"type": "image_url", "image_url": {"url": f"data:{result['mime']};base64,{result['data_b64']}"}},
                "url": result.get("url", ""), "viewport": result.get("viewport") or [0, 0]}

    @staticmethod
    def _step_key(args: dict) -> tuple | None:
        """What makes two steps the same, for the steps that can go round in circles."""
        action = args.get("action")
        if action not in ("left_click", "double_click", "triple_click", "type", "key", "visit_url", "history_back",
                          "web_search"):
            return None
        point = args.get("coordinate")
        try:
            near = tuple(round(float(v) / SAME_POINT) for v in point) if point else None
        except (TypeError, ValueError):
            near = None
        return (action, near, str(args.get("text", "")), str(args.get("keys", "")), str(args.get("url", "")),
                str(args.get("query", "")))

    @staticmethod
    def _keep_recent_screenshots(messages: list[dict]) -> None:
        """At most KEEP_SCREENSHOTS screenshots, as in training, but dropped all but the newest at once.

        Fara's own harness drops the oldest of three at every step, and the two kept after it must then be read
        again, every step; on a hybrid model such as Fara1.5, whose state cannot be wound back to the change,
        everything after the instructions is. Dropped together once a fourth arrives, the model sees one to three
        screenshots, as in the first steps of every task it was trained on, and two steps in three only add the
        new one. Measured with Fara1.5-4B on 12 real-site tasks (M3 Pro): 30 minutes and 11 answered right,
        against 70 minutes and 10, steps reading about 1,400 prompt tokens instead of 6,300-8,900.

        A dropped step goes entirely, as in Microsoft's harness, which leaves the model's own replies one after
        another; only the task keeps its text."""
        with_images = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)]
        if len(with_images) <= fara.KEEP_SCREENSHOTS:
            return
        task = next(i for i, m in enumerate(messages) if m["role"] == "user")
        dropped = set(with_images[:-1])
        kept = []
        for i, message in enumerate(messages):
            if i == task and i in dropped:
                text = " ".join(part.get("text", "") for part in message["content"] if part.get("type") == "text")
                kept.append({"role": "user", "content": text})
            elif i not in dropped:
                kept.append(message)
        messages[:] = kept

    async def _act(self, args: dict, shot: dict, ctx, facts: list[str], thoughts: str = "") -> tuple[str, str | None]:
        """Carry out one step as Microsoft's harness does, through CodeRunner's checks. Returns (what the model
        must hear before the next screenshot: nothing, an error or an answer it asked for; the final report when
        it stops)."""
        b, tainted, tab = self.svc.browser, ctx.tainted, self.tab
        action = args.get("action")
        seven = self.format == "fara7b"
        to_pixels = fara7b.to_pixels if seven else fara.to_pixels
        width, height = shot["viewport"]
        point = to_pixels(args.get("coordinate"), width, height) if args.get("coordinate") is not None else None
        if action in ("left_click", "double_click", "triple_click", "mouse_move"):
            if point is None or not width:
                return "ERROR: That action had no usable coordinate.", None
            self.pointer = point
            if action == "mouse_move":
                return self._said(await b.hover(*point, tab=tab, tainted=tainted)), None
            clicks = {"left_click": 1, "double_click": 2, "triple_click": 3}[action]
            return self._said(await b.click_at(*point, tab=tab, tainted=tainted, clicks=clicks)), None
        if action == "type":
            text = str(args.get("text", ""))
            if seven and point is not None:
                # Fara-7B types at a point: the field there is clicked first, and Enter follows unless it says not.
                self.pointer = point
                clicked = await b.click_at(*point, tab=tab, tainted=tainted)
                if not clicked.get("ok"):
                    return self._said(clicked), None
                return self._said(await b.type_focused(text, submit=args.get("press_enter", True) is not False,
                                                       tab=tab, tainted=tainted, keystrokes=True,
                                                       clear=bool(args.get("delete_existing_text")))), None
            submit = text.endswith("\n")             # a line typed to its end is entered, as a keyboard would
            return self._said(await b.type_focused(text.rstrip("\n"), submit=submit, tab=tab, tainted=tainted,
                                                   keystrokes=True)), None
        if action == "key":
            keys = [KEY_NAMES.get(str(k).lower(), str(k)) for k in (args.get("keys") or [])]
            if len(keys) == 1 and keys[0] in KEYS:
                return self._said(await b.press(keys[0], tab=tab, tainted=tainted)), None
            return self._said(await b.press_keys(keys, tab=tab, tainted=tainted)), None
        if action in ("scroll", "hscroll"):
            # The wheel turns where the mouse is, so what is under it scrolls (a results list beside a map).
            pixels = float(args.get("pixels") or 0)
            at = self.pointer or (width / 2, height / 2)
            if action == "hscroll":
                return self._said(await b.wheel(*at, pixels * (width or 1440) / fara.DISPLAY, 0, tab=tab,
                                                tainted=tainted)), None
            moved = (FARA7B_SCROLL if pixels > 0 else -FARA7B_SCROLL) if seven else \
                pixels * (height or 900) / fara.DISPLAY
            return self._said(await b.wheel(*at, 0, -moved, tab=tab, tainted=tainted)), None
        if action == "visit_url":
            url = str(args.get("url", "")).strip()
            if " " in url and not url.startswith(("http://", "https://")):
                url = SEARCH.format(urllib.parse.quote_plus(url))       # words, not an address: a search
            elif not url.startswith(("http://", "https://")):
                url = "https://" + url
            return self._said(await b.navigate(url, tab=tab, tainted=tainted)), None
        if action == "history_back":
            return self._said(await b.back(tab=tab, tainted=tainted)), None
        if action == "web_search":
            url = SEARCH.format(urllib.parse.quote_plus(str(args.get("query", ""))))
            return self._said(await b.navigate(url, tab=tab, tainted=tainted)), None
        if action == "read_page_answer_question":
            return await self._answer_from_page(str(args.get("question", "")), ctx), None
        if action == "pause_and_memorize_fact":
            facts.append(str(args.get("fact", ""))[:500])
            return "", None
        if action == "ask_user_question":
            return "", f"The browser driver needs an answer before it can go on: {args.get('question', '')}"
        if action == "wait":
            await asyncio.sleep(min(max(float(args.get("time") or WAIT), 0), MAX_WAIT))
            return "", None
        if action == "terminate":
            answer = thoughts if seven else args.get("answer")         # Fara-7B answers in its text
            return "", str(answer or "The browser driver finished without an answer.")
        return f"ERROR: {action} is not available here.", None

    def _said(self, result: dict) -> str:
        """Nothing when an action worked, as in training; an error line (as Magentic-UI sends) when it did not.
        A tab the action opened is where the driver goes on."""
        if result.get("new_tabs"):
            self.tab = result["new_tabs"][-1]
        if not result.get("ok"):
            return f"ERROR: {result.get('error', 'unknown error')}"
        if result.get("blocked"):
            return f"ERROR: The page tried to open {result['blocked']}, which is not allowed."
        return ""

    async def _answer_from_page(self, question: str, ctx) -> str:
        """Microsoft's read_page_answer_question: the page's text and the question to the same model, the answer
        cut in the middle past 1,000 characters."""
        page = await self.svc.browser.snapshot(max_chars=PAGE_FOR_QUESTIONS, tab=self.tab, tainted=ctx.tainted)
        if not page.get("ok"):
            return f"ERROR: The page could not be read: {page.get('error')}"
        try:
            reply = await self.llm.chat([
                {"role": "system", "content": EXTRACT},
                {"role": "user", "content": f"Page content:\n{page.get('content', '')}\n\nQuestion: {question}"}],
                think=False, temperature=0, max_tokens=1024)
        except LLMError as exc:
            return f"ERROR: The page could not be read: {exc}"
        said = f"I read the page to answer: {question}\nAnswer: {reply['content']}"
        if len(said) > OBSERVATION_CHARS:
            half = OBSERVATION_CHARS // 2
            said = f"{said[:half]}\n... [truncated {len(said) - OBSERVATION_CHARS} chars] ...\n{said[-half:]}"
        return said


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
