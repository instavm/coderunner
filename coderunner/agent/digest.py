"""What a task keeps of a tool result it lets go, and of the pages it has read.

When the conversation fills up, old results are cut (loop._trim). A result cut to nothing lost the prices and
facts it held: on the owner's server a 27B went back to the same product pages again and again (one five times,
another four) for figures it had read. So a cut result keeps its key lines in its place, picked without a model
call: a page's address and title and its lines with prices, ratings, dates, sizes, headings and the request's
own words; other output its first and last lines and those with numbers. The same key lines answer a page the
task opens again (Readings), unless the model asks for it fresh or acted on it."""

from __future__ import annotations

import json
import re
import urllib.parse

from ..browser.page import ELEMENT

PAGE_CHARS = 400            # key lines kept of one page (its address and title come on top)
OUTPUT_CHARS = 400          # of other output (a shell's, a script's)
LINE_CHARS = 160            # one line at most
HEADINGS = 3                # headings kept, the first ones: what the page is and its main parts
SHORT_MONEY = 30            # a price line this short takes the line before it (the item it is the price of)
TAG = re.compile(r'^<untrusted source="([\w.-]+)">$')
END = "</untrusted>"
MARK = re.compile(r"^\[(?:page read at|output of) step \d+, (?:cut to its key lines|cut short|let go)\]$")
SHORT_CHARS = 150           # of a digest cut again, when the digests together pass their share
MOST = 1500                 # a result longer than this is never taken for a digest, whatever its second line says
HEADER = re.compile(r"^(tab|url|title|navigated|note|list|dialog|new_tabs|picked|tab_note): ?(.*)$")
CURRENCY = r"(?:rs|inr|usd|eur|gbp|aed|aud|cad|sgd|jpy|cny|chf|dollars?|euros?|rupees?|pounds?)"
MONEY = re.compile(rf"[₹$€£¥]\s?\d|\d\s?[₹$€£¥]|\b{CURRENCY}\b\.?\s?\d|\d[\d,.]*\s?{CURRENCY}\b|\d\s?/-", re.I)
RATING = re.compile(r"\b\d(?:[.,]\d)?\s?(?:/\s?(?:5|10)\b|out of (?:5|10)\b|stars?\b)|\b(?:rated|rating|ratings|"
                    r"reviews?|score)\b\D{0,12}\d|\d[\d,.]*\s?(?:ratings|reviews)\b", re.I)
MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
DATE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s(?:{MONTHS})\b|\b(?:{MONTHS})\.?\s\d{{1,2}}\b|\b(?:19|20)\d\d-\d\d-\d\d\b|"
                  r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{1,2}:\d\d\b|\b(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|"
                  r"sat(?:ur)?|sun)(?:day)?\b", re.I)
SIZE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:kg|g|gm|grams?|lbs?|oz|mg|ml|l|litres?|liters?|gb|tb|mb|mm|cm|km|inch(?:es)?|"
                  r"ft|mah|wh|w|hz|mp|nights?|days?|hours?|hrs?|mins?|minutes?|sq\.?\s?ft|bhk)\b", re.I)
DIGIT = re.compile(r"\d")
JUNK = re.compile(r"^(?:\[(?:menu|footer)\b|…|\.\.\.)")
STOP = frozenset("a an and are as at be by can for from get how i in is it me my of on or show tell that the this "
                 "to was what when where which who why will with you your find give list best".split())
# Parameters that only say where a click came from: dropped from a page's address, so the same page opened from
# two links is one page (the lists of browsers' clean-link features; no site's own parameters).
TRACKING = re.compile(r"^(?:utm_\w+|gclid|gclsrc|dclid|fbclid|msclkid|yclid|twclid|igshid|srsltid|mc_cid|mc_eid|_ga|"
                      r"_gl|_hsenc|_hsmi|ref|ref_|ref_src|referrer|spm|trk|si|cmpid)$", re.I)
LONG_VALUE = 40             # as tools._short_address cuts a page's address, so both sides compare alike


def request_words(text: str) -> frozenset[str]:
    """The words of a request that a line about it would carry: names, models, places, not filler."""
    return frozenset(w for w in re.findall(r"[^\W_]+", (text or "").lower()) if len(w) >= 3 and w not in STOP)


def normal_url(url: str) -> str:
    """A page's address as one key: no fragment, no tracking parameters, no "www.", no trailing slash, the
    parameters in order and long values cut as the page header cuts them."""
    url = str(url or "").strip()
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return url.split("#")[0].rstrip("/")
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    query = []
    for piece in parts.query.split("&"):
        if not piece:
            continue
        name, sep, value = piece.partition("=")
        if TRACKING.match(name):
            continue
        value = value.rstrip("…")[:LONG_VALUE]
        query.append(f"{name}={value}" if sep else name)
    return urllib.parse.urlunsplit(("https" if parts.scheme == "https" else "http", host,
                                    parts.path.rstrip("/"), "&".join(sorted(query)), ""))


def is_cut(text: str) -> bool:
    """A result already cut to its key lines (or to nothing, as before digests)."""
    if not isinstance(text, str) or len(text) > MOST:
        return False
    if text.endswith("take a new snapshot if you need it]\n" + END):
        return True
    lines = text.split("\n", 2)
    return any(MARK.match(line) for line in lines[:2])


def _clean(line: str) -> str:
    """A page line without its element number (stale once the page has moved on) or its role word."""
    line = " ".join(line.split())
    match = ELEMENT.match(line)
    if match:
        name = ""
        if match.group(3):
            try:
                name = str(json.loads(match.group(3)))
            except ValueError:
                name = match.group(3).strip('"')
        line = (name + line[match.end():]).strip()
    elif line.startswith("- ") or line.startswith("* "):
        line = line[2:]
    return line


def _shorten(line: str) -> str:
    if len(line) <= LINE_CHARS:
        return line
    if " -> " in line:                       # a list item's link is what opens it again
        text, link = line.rsplit(" -> ", 1)
        link = link[:60]
        return text[:max(40, LINE_CHARS - len(link) - 5)].rstrip() + "… -> " + link
    return line[:LINE_CHARS - 1].rstrip() + "…"


def _score(line: str, words: frozenset[str], page: bool) -> int:
    score = 3 * bool(MONEY.search(line)) + 2 * bool(RATING.search(line)) + 2 * bool(DATE.search(line))
    score += bool(SIZE.search(line))
    if words:
        score += min(3, len(words & set(re.findall(r"[^\W_]+", line.lower()))))
    if not page and not score and DIGIT.search(line):
        score = 1
    return score


def _pick(lines: list[str], words: frozenset[str], page: bool, limit: int, keep: set[int]) -> list[str]:
    """The highest scoring lines within `limit` characters, without repeats, in the order the page has them;
    the lines in `keep` first."""
    cleaned: list[tuple[int, str, int]] = []
    headings = 0
    previous = ""
    for index, raw in enumerate(lines):
        line = _clean(raw) if page else " ".join(raw.split())
        if not line or JUNK.match(line) or not re.search(r"[^\W_]", line):
            continue
        score = _score(line, words, page)
        if page and line.startswith("#"):
            if headings < HEADINGS:
                score += 2 if headings == 0 else 1
            headings += 1
        named = len(re.findall(r"[^\W\d_]{2,}", line)) >= 2
        bare = not re.search(r"[^\W\d_]", re.sub(rf"\b{CURRENCY}\b", "", line, flags=re.I))
        if page and score and MONEY.search(line) and len(line) < SHORT_MONEY and bare and previous:
            shown = f"{previous[:80]} — {line}"
        else:
            shown = line
        if index in keep:
            score += 100
        if score:
            cleaned.append((score, _shorten(shown), index))
        if line.startswith("#"):
            previous = ""                       # a price under a heading is the heading's, which is kept
        elif named:
            previous = line
    chosen, seen, used = [], set(), 0
    for score, line, index in sorted(cleaned, key=lambda item: (-item[0], item[2])):
        key = re.sub(r"[\W_]+", " ", line.lower()).strip()
        if key in seen:
            continue
        if used + len(line) + 1 > limit:
            continue
        seen.add(key)
        chosen.append((index, line))
        used += len(line) + 1
    return [line for _, line in sorted(chosen)]


def digest(text: str, step: int, words: frozenset[str] = frozenset(), short: bool = False) -> str:
    """A tool result cut to its key lines, marked as such, inside the same untrusted tag as the result was.
    short: fewer of them (a digest cut again, when the digests together pass their share)."""
    lines = text.split("\n")
    tag = TAG.match(lines[0]) if lines else None
    source = tag.group(1) if tag else ""
    body = [line for line in (lines[1:] if tag else lines) if not MARK.match(line)]
    if tag and body and body[-1].strip() == END:
        body = body[:-1]
    how = "cut short" if short else "cut to its key lines"
    if source.startswith("browser_"):
        url = title = ""
        rest = []
        for line in body:
            header = HEADER.match(line)
            if header:
                if header.group(1) == "url":
                    url = header.group(2).strip()      # the last one: where a batch of steps ended
                elif header.group(1) == "title" and not title:
                    title = header.group(2).strip()
                continue
            rest.append(line)
        kept = _pick(rest, words, True, SHORT_CHARS if short else PAGE_CHARS, set())
        if not kept:
            kept = [_shorten(_clean(line)) for line in rest if _clean(line) and not JUNK.match(_clean(line))][:2]
        out = [f"[page read at step {step}, {how}]"]
        out += [f"url: {url}"] if url else []
        out += [f"title: {title[:60 if short else 100]}"] if title else []
        out += kept
    else:
        filled = [i for i, line in enumerate(body) if line.strip()]
        ends = set(filled[:1] + filled[-1:]) if short else set(filled[:2] + filled[-2:])
        out = [f"[output of step {step}, {how}]"] + _pick(body, words, False, SHORT_CHARS if short else OUTPUT_CHARS,
                                                         ends)
    return "\n".join(([lines[0]] if tag else []) + out + ([END] if tag else []))


def cut_level(text: str) -> int:
    """How far a result was cut: 0 not at all, 1 to its key lines, 2 short, 3 to its address (or to nothing,
    as before digests)."""
    if not is_cut(text):
        return 0
    head = text.split("\n", 2)[:2]
    if any(line.endswith(", cut short]") for line in head):
        return 2
    return 1 if any(line.endswith(", cut to its key lines]") for line in head) else 3


def let_go(text: str, step: int) -> str:
    """A short digest let go too, when even those pass their share: its address is what is left."""
    lines = text.split("\n")
    tag = lines[0] if TAG.match(lines[0]) else ""
    url = next((line for line in lines if line.startswith("url: ")), "")
    kind = "page read at" if tag.startswith('<untrusted source="browser_') else "output of"
    return "\n".join(filter(None, [tag, f"[{kind} step {step}, let go]", url, END if tag else ""]))


# ---- pages read in this task -----------------------------------------------

REVISIT = ("You read this page at step {step}; here is what it said (the browser stayed where it was). Open it "
           "with fresh=true if it may have changed or you need to act on it.")
ACTIONS = frozenset({"browser_click", "browser_type", "browser_select", "browser_press", "browser_batch",
                     "browser_scroll"})
# Actions that, made in place, may change a site's other pages (a cart); typing and scrolling change only the page.
SITE_CHANGES = frozenset({"browser_click", "browser_select", "browser_press", "browser_batch"})
READS = frozenset({"browser_open", "browser_snapshot", "browser_back", "browser_list", "browser_search"})
ELSEWHERE = frozenset({"browser_task", "browser_handoff"})      # someone else moved the browser on
SNAPSHOT_GAP = 2            # a snapshot of a page read in the step just before is a look again (still loading)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc if "://" in url else ""


def _result_url(result: str) -> str:
    found = re.findall(r"^url: (\S+)", result, re.M)
    return normal_url(found[-1]) if found else ""


class Readings:
    """The pages one task has read, by address, with the step and key lines of the latest reading. A page the
    model acted on (typed, clicked, filtered there) is never answered from here: it may have changed. A change
    made in place (one that did not move to another page: a filter, an add-to-cart button) lets go of what was
    read of the same site before it, since its other pages (a cart) may show it."""

    def __init__(self, words: frozenset[str] = frozenset()):
        self.words = words
        self.pages: dict[str, tuple[int, str]] = {}
        self.acted: set[str] = set()
        self.alias: dict[str, str] = {}     # an address asked for and the one it ended at (a redirect)
        self.current = ""

    def kept(self, name: str, args: dict, step: int) -> str:
        """For a call that would read a page this task has read and not acted on since: what it said, or ""."""
        if not isinstance(args, dict) or args.get("fresh") is True:
            return ""
        if name == "browser_open":
            url = normal_url(str(args.get("url") or ""))
        elif name == "browser_snapshot" and not args.get("start") and not args.get("more"):
            url = self.current
        else:
            return ""
        if not url or url in self.acted or self.alias.get(url, url) in self.acted or url not in self.pages:
            return ""
        at, said = self.pages[url]
        if name == "browser_snapshot" and step - at < SNAPSHOT_GAP:
            return ""
        return REVISIT.format(step=at) + "\n" + said

    def record(self, name: str, args: dict, result: str, step: int) -> None:
        """After a browser call that worked: where the browser is now, and what it read there."""
        if name in ELSEWHERE:
            self.pages.clear()
            self.current = ""
            return
        if not name.startswith("browser_"):
            return
        url = _result_url(result)
        moved = bool(url) and url != self.current
        if name in ACTIONS:
            before = self.current
            if before:
                self.acted.add(before)
                self.pages.pop(before, None)
                if not moved and name in SITE_CHANGES:
                    host = _host(before)
                    for other in [u for u in self.pages if _host(u) == host]:
                        del self.pages[other]
        if not url:
            return
        self.current = url
        whole = name in READS or (name in ACTIONS and moved and "\n\n" in result)
        if name == "browser_snapshot" and (args.get("start") or args.get("more")):
            whole = False                       # a part of the page: the whole reading stays
        if name == "browser_open" and isinstance(args, dict) and args.get("url"):
            asked = normal_url(str(args["url"]))
            if asked != url and asked not in self.acted:
                self.alias[asked] = url
                self.pages[asked] = (step, digest(result, step, self.words))
        if whole and url not in self.acted:
            self.pages[url] = (step, digest(result, step, self.words))
