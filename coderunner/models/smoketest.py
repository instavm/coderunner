"""Verify a served model can drive the agent before it is configured.

Small models often call the right tool with the wrong arguments (a 1.7B model
added 3 and 7 and reported 13), so arguments are checked exactly.
"""

from __future__ import annotations

import base64
import json
import struct
import time
import zlib

from ..llm import LLM

TOOLS = [
    {"type": "function", "function": {
        "name": "add_numbers", "description": "Add two integers and return the sum.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def solid_png(size: int = 64, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = (b"\x00" + bytes(rgb) * size) * size  # filter byte 0 per scanline
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _args(call: dict) -> dict:
    raw = call.get("function", {}).get("arguments") or {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _calls(reply: dict) -> list[tuple[str, dict]]:
    return [(c.get("function", {}).get("name"), _args(c)) for c in reply.get("tool_calls") or []]


def _is_int(value, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def check_add(reply: dict) -> tuple[bool, str]:
    calls = _calls(reply)
    if not calls:
        return False, f"no tool call; replied {reply.get('content', '')[:120]!r}"
    name, args = calls[0]
    if name != "add_numbers":
        return False, f"called {name} instead of add_numbers"
    if _is_int(args.get("a"), 3) and _is_int(args.get("b"), 7):
        return True, "add_numbers(a=3, b=7)"
    return False, f"wrong arguments {json.dumps(args)}; expected a=3, b=7"


def check_weather(reply: dict) -> tuple[bool, str]:
    calls = _calls(reply)
    if not calls:
        return False, f"no tool call; replied {reply.get('content', '')[:120]!r}"
    name, args = calls[0]
    city = str(args.get("city", ""))
    if name == "get_weather" and city.strip().lower().startswith("paris"):
        return True, f"get_weather(city={city!r})"
    return False, f"called {name}({json.dumps(args)}); expected get_weather(city='Paris')"


def check_ready(reply: dict) -> tuple[bool, str]:
    if reply.get("tool_calls"):
        return False, f"called a tool: {_calls(reply)[0][0]}"
    content = reply.get("content", "")
    return ("ready" in content.lower()), f"replied {content[:120]!r}"


def check_red(reply: dict) -> tuple[bool, str]:
    content = reply.get("content", "")
    return ("red" in content.lower()), f"replied {content[:120]!r}"


def cases(vision: bool) -> list[tuple[str, list[dict], list[dict] | None, object]]:
    user = lambda content: [{"role": "user", "content": content}]  # noqa: E731
    out = [
        ("tool_arguments", user("Use a tool to add 3 and 7."), TOOLS, check_add),
        ("tool_selection", user("What's the weather in Paris?"), TOOLS, check_weather),
        ("no_tool", user("Reply with the single word: ready"), TOOLS, check_ready),
    ]
    if vision:
        image = "data:image/png;base64," + base64.b64encode(solid_png()).decode()
        out.append(("vision", user([
            {"type": "text", "text": "What colour is this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": image}},
        ]), None, check_red))
    return out


async def run(base_url: str, model: str, vision: bool = False, api_key: str = "", timeout: float = 300) -> dict:
    llm = LLM(base_url, model, api_key=api_key, timeout=timeout)
    checks, tokens, seconds = [], 0, 0.0
    try:
        for name, messages, tools, verify in cases(vision):
            started = time.monotonic()
            try:
                reply = await llm.chat(messages, tools=tools, temperature=0)
                passed, detail = verify(reply)
                tokens += int((reply.get("usage") or {}).get("completion_tokens") or 0)
            except Exception as exc:  # report any runtime failure as a failed check
                passed, detail = False, f"{type(exc).__name__}: {exc}"[:300]
            elapsed = time.monotonic() - started
            seconds += elapsed
            checks.append({"name": name, "passed": passed, "detail": detail, "seconds": round(elapsed, 2)})
    finally:
        await llm.aclose()
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "tok_s": round(tokens / seconds, 1) if tokens and seconds else None}
