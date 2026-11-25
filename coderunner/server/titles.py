"""Short names for the app's threads, as the list shows them: "Mauritius capital", not the whole first request.

A thread's name is written by the model once, after its first task has ended (never before: the start is not
slowed, and the request does not take a running task's place at the model). Until then, and whenever the model
is busy, away or answers something unusable, the name is made here from the request itself.
"""

from __future__ import annotations

import re
import urllib.parse

MAX_CHARS = 40              # of a name made here, cut on a word
MODEL_MAX_CHARS = 60        # a longer answer from the model is not a name
PROMPT_CHARS = 300          # of the request the model is shown
MAX_TOKENS = 16
TIMEOUT_S = 20
SYSTEM = ("Name the request in 2 to 5 words, in the language it is written in. No address, no quotes, no full "
          "stop. Reply with the name only. Examples: Mauritius capital; Noise-cancelling headphones; "
          "Bengaluru weather tomorrow")

URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
DOMAIN = re.compile(r"\b[\w-]+\.(?:com|org|net|io|in|de|co|uk|ai|dev|app|gov|edu|info)\b(?:/\S*)?", re.I)
# Openings that say how to do it, or ask politely, rather than what it is about; dropped from the front, again
# and again ("Please can you open ...").
FILLER = re.compile(
    r"^(?:please|pls|kindly|hey|hi|hello|ok(?:ay)?|so|"
    r"(?:can|could|would|will) you(?: please)?|i (?:want|need|would like|'d like) you to|i want to|"
    r"help me(?: to)?|tell me|show me|let me know|find out|"
    r"(?:use|using|with) (?:the )?[\w.+#-]+(?: [\w.+#-]+)?(?: to|,)|"
    r"open(?: up)?|go to|visit|navigate to|browse to|look at|check|"
    r"bitte|(?:kannst du|können sie|könntest du)(?: mir| uns)?|öffne|"           # German
    r"por favor|puedes|abre|"                                     # Spanish
    r"s'il (?:te|vous) pla[iî]t|peux-tu|pouvez-vous|ouvre|"       # French
    r"कृपया|ज़रा|जरा"                                              # Hindi
    # Not \b: a Hindi word ends on a vowel sign, which is no word character to the regex.
    r")(?=[\s,:;.!-]|$)[\s,:;.!-]*", re.I)
# Words a name made by cutting is not left ending on.
DANGLING = {"a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "at", "with", "by", "from", "its",
            "is", "are", "me", "my", "und", "oder", "der", "die", "das", "von", "mit", "für", "y", "de", "la",
            "el", "et", "le", "les", "du", "और", "का", "की", "के", "में"}
TRAILING = " \t\n.,;:!?।…-–—\"'`*"


def _from_url(url: str) -> str:
    """What an address is about: its last meaningful path segment (wiki/Mauritius), else its site."""
    if not re.match(r"https?://", url, re.I):
        url = "http://" + url
    parts = urllib.parse.urlsplit(url.rstrip(".,;:!?"))
    for segment in reversed([s for s in parts.path.split("/") if s]):
        segment = urllib.parse.unquote(segment)
        segment = re.sub(r"\.(?:html?|php|aspx?|jsp)$", "", segment, flags=re.I)
        words = re.sub(r"[_+-]+", " ", segment).strip()
        # An id, a page number or a site's own section word says nothing.
        if words and not re.fullmatch(r"[\d\s]+|[0-9a-f]{8,}|index|default|home|wiki|en|www|search|dp|p|s|item|"
                                      r"issues?|pulls?|blob|tree|comments?|status",
                                      words, re.I) and len(words) <= 60:
            return words
    host = (parts.hostname or "").removeprefix("www.")
    return host


def heuristic(prompt: str) -> str:
    """A short name made from the request alone: addresses as what they point to, polite or how-to openings
    dropped, cut at MAX_CHARS on a word, no closing full stop. Kept in the request's own language."""
    text = " ".join(str(prompt or "").split())
    if not text:
        return ""
    text = URL.sub(lambda m: _from_url(m.group(0)), text)
    text = DOMAIN.sub(lambda m: _from_url(m.group(0)), text)
    text = " ".join(text.split())
    for _ in range(6):
        shorter = FILLER.sub("", text, count=1).strip()
        if shorter == text or not shorter:
            break
        text = shorter
    text = text.strip(TRAILING)
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS + 1]
        cut = cut[:cut.rfind(" ")] if " " in cut else text[:MAX_CHARS]
        words = cut.split()
        while len(words) > 1 and words[-1].lower().strip(TRAILING) in DANGLING:
            words.pop()
        text = " ".join(words).strip(TRAILING)
    return text[:1].upper() + text[1:]


def clean(answer: str) -> str:
    """The model's answer as a name, or "" when it is not one (empty, too long, an address, a sentence)."""
    lines = [line for line in str(answer or "").strip().splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0].strip(TRAILING + "“”«»").strip()
    text = re.sub(r"^(?:title|name|titel|शीर्षक)\s*[:：-]\s*", "", text, flags=re.I)
    text = text.strip(TRAILING + "“”«»").strip()
    if not text or len(text) > MODEL_MAX_CHARS or len(text.split()) > 8:
        return ""
    if URL.search(text) or DOMAIN.search(text) or "://" in text:
        return ""
    return text[:1].upper() + text[1:]


async def from_model(llm, prompt: str, slot: int | None = None) -> str:
    """One short request for a name, reasoning off; "" when the model gives nothing usable. Errors pass on."""
    turns = [{"role": "user", "content": " ".join(str(prompt).split())[:PROMPT_CHARS]}]
    options = {"slot": slot} if slot is not None else {}
    return clean(await llm.ask(SYSTEM, turns, max_tokens=MAX_TOKENS, **options))
