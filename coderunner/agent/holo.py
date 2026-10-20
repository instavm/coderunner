"""H Company's Holo (Holo3.1) as a browser driver: the agent harness its documentation describes (hub.hcompany.ai,
Models API: core concepts, function calling, browser-agent cookbook), which the model was trained in.

Tools in every request with tool_choice "required"; the screen as a user message wrapped in <observation> tags,
at most the last three, older ones replaced by "[screenshot evicted]"; each action answered by a tool result with
the address; coordinates are integers in [0, 1000] over the screenshot; thinking on at temperature 0.8; the task
ends when the model calls answer. Only the parsed reply goes back into the history, never its reasoning.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM = "You are a web agent. You see the browser through screenshots and act with your tools."
VIEWPORT = (1280, 800)          # the cookbook's page size; coordinates are relative to the screenshot anyway
TEMPERATURE = 0.8               # agent loops, thinking on (core concepts: sampling)
MAX_TOKENS = 4096               # its reasoning comes first
KEEP_SCREENSHOTS = 3            # "more degrades accuracy"
EVICTED = "[screenshot evicted]"
SCROLL = 0.8                    # of the page's height per scroll, as the cookbook scrolls


def _fn(name: str, description: str, **properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)}}}


_INT = {"type": "integer", "description": "Coordinate as integer in [0, 1000]"}
TOOLS = [
    _fn("click", "Click at (x, y) coordinates",
        element={"type": "string", "description": "Detailed description of the target UI element"}, x=_INT, y=_INT),
    _fn("type", "Type text into the focused element, optionally pressing Enter",
        text={"type": "string"}, press_enter={"type": "boolean"}),
    _fn("scroll", "Scroll the page", direction={"type": "string", "enum": ["up", "down"]}),
    _fn("goto", "Navigate to a URL", url={"type": "string"}),
    _fn("answer", "Provide a final answer", content={"type": "string", "description": "The answer content"}),
]


def observation(image: dict) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": "<observation>\n"}, image,
                                        {"type": "text", "text": "\n</observation>"}]}


def keep_recent_screenshots(messages: list[dict]) -> None:
    """At most KEEP_SCREENSHOTS screenshots, older ones replaced by the placeholder inside their <observation>
    wrapper, as the documentation's helper does; but all but the newest go at once when a fourth arrives, so
    two steps in three only add a screenshot (see Driver._keep_recent_screenshots for the measurement)."""
    images = [(i, j) for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)
              for j, part in enumerate(m["content"]) if part.get("type") == "image_url"]
    if len(images) <= KEEP_SCREENSHOTS:
        return
    for i, j in images[:-1]:
        messages[i] = {**messages[i], "content": list(messages[i]["content"])}
        messages[i]["content"][j] = {"type": "text", "text": EVICTED}


def call_of(reply: dict) -> tuple[str, dict, str] | None:
    """The first tool call of a reply: (name, arguments, call id)."""
    calls = reply.get("tool_calls") or []
    if not calls:
        return None
    function = calls[0].get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except (TypeError, ValueError):
        args = {}
    return str(function.get("name", "")), args if isinstance(args, dict) else {}, str(calls[0].get("id") or "call_0")


def as_fara(name: str, args: dict) -> dict[str, Any]:
    """A Holo call as the same step in Fara1.5's form, whose coordinates share the [0, 1000] grid, so it is carried
    out through the same checked actions."""
    if name == "click":
        return {"action": "left_click", "coordinate": [args.get("x"), args.get("y")]}
    if name == "type":
        return {"action": "type", "text": str(args.get("text", "")) + ("\n" if args.get("press_enter") else "")}
    if name == "scroll":
        # Fara1.5's scroll is pixels on the grid, positive up: the cookbook's 0.8 of a page.
        return {"action": "scroll", "pixels": SCROLL * 1000 * (1 if args.get("direction") == "up" else -1)}
    if name == "goto":
        return {"action": "visit_url", "url": str(args.get("url", ""))}
    return {"action": name}
