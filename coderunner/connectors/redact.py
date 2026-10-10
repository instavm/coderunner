"""Mask credentials that commonly arrive by mail before the model reads them.

One-time codes, password-reset / magic sign-in links and long random tokens
are useless to the assistant and dangerous if injected instructions get it to
forward them. Ordinary numbers (rooms, prices, order numbers) are left alone:
a number is only treated as a code when a word like "code" or "OTP" is nearby.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit

CODE_MASK = "[redacted code]"
LINK_MASK = "[redacted link]"
TOKEN_MASK = "[redacted token]"

URL = re.compile(r"https?://[^\s<>\"'\]\)]+", re.I)
LINK_MARKERS = ("reset", "verify", "verification", "magic", "token=", "code=", "login?", "signin", "sign-in",
                "sign_in", "otp", "passwordless", "confirm")
TOKEN = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])")
# 4-8 digits, or 3+3 split ("123 456", "123-456"), optionally with a letter prefix like Google's "G-".
CODE = re.compile(r"(?<![\w$€£¥.,/#-])(?:[A-Z]-)?(?:\d{3}[ -]\d{3}|\d{4,8})(?!\w|[.,:/-]\d)")
CODE_WORDS = re.compile(r"\b(?:codes?|otp|verification|verify|passcode|pin|one[- ]?time|2fa|two[- ]factor)\b", re.I)
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
BEFORE, AFTER = 60, 40


def _entropy(text: str) -> float:
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def _is_token(text: str) -> bool:
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    # Random tokens have long separator-free runs; slugs like "q3-2026-annual-report-final" do not.
    longest_run = max(len(part) for part in re.split(r"[-_]", text))
    return has_letter and has_digit and longest_run >= 16 and _entropy(text) >= 3.0


def _sensitive_link(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return True
    target = (parts.path + ("?" + parts.query if parts.query else "")).lower()
    return any(marker in target for marker in LINK_MARKERS)


def _link(match: re.Match) -> str:
    url = match.group(0)
    trailing = ""
    while url and url[-1] in ".,;:!?":
        url, trailing = url[:-1], url[-1] + trailing
    return (LINK_MASK if _sensitive_link(url) else url) + trailing


def _code(match: re.Match, text: str) -> str:
    # Look for a code word nearby, but only within the same sentence.
    start, end = match.span()
    before = SENTENCE_END.split(text[max(0, start - BEFORE):start])[-1]
    after = SENTENCE_END.split(text[end:end + AFTER], maxsplit=1)[0]
    return CODE_MASK if CODE_WORDS.search(before + " " + after) else match.group(0)


def redact(text: str) -> str:
    if not text:
        return text or ""
    text = URL.sub(_link, text)
    text = TOKEN.sub(lambda m: TOKEN_MASK if _is_token(m.group(0)) else m.group(0), text)
    return CODE.sub(lambda m: _code(m, text), text)
