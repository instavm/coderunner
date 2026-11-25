"""A long task's conversation made shorter by a summary the model writes, with its latest steps kept as they are.

Any edit to earlier messages makes a local runtime read the conversation again (a hybrid model cannot resume from
the middle): at ~510 tokens a second, 20,000 tokens is 40 s. The extractive trims (loop._trim) came every few
steps once the context was 60% full. Here, as in Pi's coding agent, the conversation is left alone until it fills
COMPACT_AT of its room; then the model writes a structured summary of the work so far (one extra request with the
instruction appended, so the runtime reads only the instruction and writes up to SUMMARY_TOKENS), and the
conversation becomes: the instructions and the request, the summary, the plan and the owner's words, and the
latest steps verbatim (about KEEP_TAIL of the room). The next one is far away. The summary passes forward: the
next summary request sees it and refines it.

The cut is made at a message the model or the owner wrote, never between a tool call and its result."""

from __future__ import annotations

import json
import re

# What a compaction leaves (instructions, summary, tail) is read again, and the next one comes when the room between
# that and COMPACT_AT has filled, so the tail is kept short: on a fake 60-step task of long pages at 32K, a third of
# the room kept at 75% compacted 13 times and read 225,000 tokens again (more than the trims' 16 times and 184,000),
# 15% kept at 80% compacted 7 times and read 78,000 again; at 64K, 2 times and 31,000 (trims: 4 and 48,000).
COMPACT_AT = 0.8            # share of the room past which the conversation is compacted
KEEP_TAIL = 0.15            # share of the room kept verbatim at the end (the latest step always is, whole)
KEEP_MOST = 20000           # tokens kept verbatim at most (Pi's keepRecent), so a large context re-reads little
MIN_SUMMARIZED = 0.1        # a compaction that would summarize less than this share of the room is not made
HUGE_RESULT = 0.12          # a single result in the kept tail larger than this share of the room is digested
SUMMARY_TOKENS = 700
# Below this room per task a local model's summaries cost more than they save: at 32K on Bonsai 2.5 (~32 tokens a
# second written) a summary took 28-38 s and came every 3-4 steps, nine in one task, and the task took 24 min
# against 13.6 with the extractive trim; Pi's design assumes a large window (16K reserved of 200K). A fast model
# (Claude) writes one in a second or two and summarizes at any size.
SUMMARY_MIN_ROOM = 48_000
SUMMARY_TIMEOUT_S = 60
SUMMARY_MIN_CHARS = 200
SUMMARY_FAILS = 2           # failed summaries in a run before it keeps to the extractive trim
CHARS_PER_TOKEN = 3.5
IMAGE_TOKENS = 1200

SUMMARY_HEADER = "[Summary of the work so far, written by you earlier in this task; not a new request]"
SECTIONS = ("## Goal", "## Constraints & Preferences", "## Progress", "### Done", "### In progress", "### Blocked",
            "## Key Decisions", "## Key facts found", "## Next Steps")
SUMMARIZE = (
    "[Not a new request] This conversation is about to be shortened: your earlier steps will be replaced by a "
    "summary that you write now, and only your latest steps kept as they are. Without calling any tools, write that "
    "summary in exactly this form, in short lines:\n\n"
    "## Goal\nWhat I asked for, in a line or two.\n"
    "## Constraints & Preferences\nLimits and preferences I gave (budget, dates, sizes, sites), or (none).\n"
    "## Progress\n### Done\n### In progress\n### Blocked\n"
    "## Key Decisions\nChoices made and why, in a few words each.\n"
    "## Key facts found\nEvery figure, name, date and link you will need for the answer, copied exactly, each with "
    "the page it came from (its address).\n"
    "## Next Steps\nWhat is left to do, in order.\n\n"
    "If an earlier summary is above, start from it: keep every fact in it that still holds, add what was found "
    "since, and drop only what no longer matters. Keep it under 450 words.")
OVERFLOW = re.compile(r"exceeds? (?:the )?(?:available )?context|context[_ ](?:length|size|window)[_ ]exceeded|"
                      r"exceed_context_size|maximum context length|prompt is too long|too many tokens|"
                      r"context (?:is )?full|input is too long", re.I)


def message_tokens(message: dict) -> int:
    """About how many tokens one message holds (its text, its calls' arguments, its pictures)."""
    content = message.get("content")
    tokens = 0.0
    if isinstance(content, str):
        tokens += len(content) / CHARS_PER_TOKEN
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                tokens += len(part.get("text") or "") / CHARS_PER_TOKEN
            else:
                tokens += IMAGE_TOKENS
    if message.get("tool_calls"):
        tokens += len(json.dumps(message["tool_calls"], ensure_ascii=False)) / CHARS_PER_TOKEN
    return int(tokens) + 4


def tokens_of(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def threshold(base: float) -> float:
    """Tokens past which a conversation with room for `base` tokens is compacted."""
    return COMPACT_AT * base


def keep_tokens(base: float) -> int:
    return int(min(KEEP_TAIL * base, KEEP_MOST))


def cut_point(messages: list[dict], head: int, keep: int) -> int | None:
    """Where the verbatim tail starts: walking back from the newest message until `keep` tokens are reached, then
    back to a message the model or the owner wrote (a tool's result stays with the call that asked for it), and
    never after the latest step. None when nothing would be left to summarize."""
    start, total = len(messages), 0
    while start > head and total < keep:
        start -= 1
        total += message_tokens(messages[start])
    while start > head and messages[start].get("role") == "tool":
        start -= 1
    steps = [i for i in range(head, len(messages)) if messages[i].get("tool_calls")]
    if steps:
        start = min(start, steps[-1])
    while start > head and messages[start].get("role") == "tool":     # (an empty step's results, say)
        start -= 1
    return start if start > head else None


KEY_HEADER = "[Key lines from pages and results read earlier in this task, as they gave them; for figures and links]"
KEY_TOKENS = 0.06           # share of the room the key lines may take (oldest go first)


def is_key_lines(message: dict) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith(KEY_HEADER)


def key_lines_message(blocks: list[str], room: float) -> dict | None:
    """The key lines of the results a summary replaced, verbatim, newest kept when they do not all fit: the model's
    summary keeps the story, and these keep the figures and links as the pages gave them (on Bonsai the summaries
    gave two products the same link and a 90 g jar as 100 g)."""
    budget = int(KEY_TOKENS * room * CHARS_PER_TOKEN)
    kept: list[str] = []
    for block in reversed([b for b in blocks if b.strip()]):
        if sum(len(b) for b in kept) + len(block) > budget:
            break
        kept.insert(0, block)
    if not kept:
        return None
    body = "\n".join(kept)
    return {"role": "user", "content": KEY_HEADER + '\n<untrusted source="earlier pages">\n' + body + "\n</untrusted>"}


def key_blocks(message: dict) -> list[str]:
    """The blocks a key-lines message holds, to be carried into the next one."""
    text = str(message.get("content") or "")
    inner = text.split('<untrusted source="earlier pages">\n', 1)[-1].rsplit("\n</untrusted>", 1)[0]
    return [b for b in inner.split("\n\n") if b.strip()]


def is_summary(message: dict) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith(SUMMARY_HEADER)


def summary_message(text: str) -> dict:
    return {"role": "user", "content": SUMMARY_HEADER + "\n" + text.strip()}


def usable(reply: dict) -> tuple[str, str]:
    """The summary in a reply, and "" when it is usable; else ("", why not)."""
    if reply.get("tool_calls"):
        return "", "junk"
    text = re.sub(r"<think>.*?</think>", "", str(reply.get("content") or ""), flags=re.S).strip()
    found = sum(1 for heading in SECTIONS if heading.lower() in text.lower())
    if len(text) < SUMMARY_MIN_CHARS:
        return "", "too_short"
    if found < 3:
        return "", "junk"
    return text, ""


def ran_out_of_room(reply: dict, cap: int) -> bool:
    """A reply that stopped for length well before its own limit: the runtime's context was full."""
    usage = reply.get("usage") or {}
    if reply.get("finish_reason") != "length" or not cap or not usage.get("prompt_tokens"):
        return False
    return int(usage.get("completion_tokens") or 0) < 0.9 * cap
