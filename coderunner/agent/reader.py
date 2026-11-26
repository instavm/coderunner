"""Fast parallel HTTP clean reader & extractor for static web pages.

Fetches 1-4 static pages concurrently over httpx (using the same URL & private-IP guard as BrowserBroker)
and distills clean readable text/tables/prices/links without spinning up full Chrome tabs. Pages that
require JavaScript rendering or present a bot wall are marked `needs_browser=True` so the planner falls
back to `browser_open` seamlessly.
"""

from __future__ import annotations

import asyncio
import html
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Any

import httpx

from ..browser.identity import chrome_headers, desktop_user_agent
from ..browser.urls import local_problem, page_host, parse_url
from ..browser.walls import classify, probe_from_html
from ..safety.policy import _matches

MAX_URLS = 4
FETCH_TIMEOUT = 12.0
MAX_BYTES = 1_500_000
DEFAULT_CHARS = 3500
USER_AGENT = desktop_user_agent()

SKIP_TAGS = frozenset({
    "script", "style", "noscript", "svg", "canvas", "iframe", "template",
    "nav", "footer", "aside", "form", "button",
})
BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "li", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dt", "dd",
})
JS_SHELL_RE = re.compile(
    r"enable javascript|javascript is required|you need to enable javascript|please enable cookies|"
    r"loading\.\.\.$|__NEXT_DATA__.*\"page\"",
    re.I,
)


class _ArticleExtractor(HTMLParser):
    """Lightweight HTML-to-readable-text/markdown extractor (zero third-party dependencies)."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_a = False
        self._a_href = ""
        self._a_buf: list[str] = []
        self._heading = 0
        self._in_td = False
        self._row_cells: list[str] = []
        self._cell_buf: list[str] = []

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split()).strip()
        self._buf.clear()
        if not text:
            return
        if self._heading:
            self.lines.append(f"{'#' * min(self._heading, 3)} {text}")
        else:
            self.lines.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = True
            return
        if self._skip_depth:
            if t in SKIP_TAGS:
                self._skip_depth += 1
            return
        if t in SKIP_TAGS:
            self._skip_depth = 1
            return
        if t in ("h1", "h2", "h3", "h4"):
            self._flush()
            self._heading = int(t[1])
        elif t == "tr":
            self._flush()
            self._row_cells = []
        elif t in ("td", "th"):
            self._in_td = True
            self._cell_buf = []
        elif t == "li":
            self._flush()
            self._buf.append("- ")
        elif t == "a":
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self._in_a = True
                self._a_href = urllib.parse.urljoin(self.base_url, href)
                self._a_buf = []
        elif t in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
            return
        if self._skip_depth:
            if t in SKIP_TAGS:
                self._skip_depth -= 1
            return
        if t == "a" and self._in_a:
            label = " ".join("".join(self._a_buf).split()).strip()
            self._in_a = False
            if label:
                piece = label
                target = self._cell_buf if self._in_td else self._buf
                target.append(piece)
        elif t in ("td", "th") and self._in_td:
            cell = " ".join("".join(self._cell_buf).split()).strip()
            self._in_td = False
            if cell:
                self._row_cells.append(cell[:120])
        elif t == "tr" and self._row_cells:
            self.lines.append(" | ".join(self._row_cells))
            self._row_cells = []
        elif t in ("h1", "h2", "h3", "h4"):
            self._flush()
            self._heading = 0
        elif t in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth:
            return
        if self._in_a:
            self._a_buf.append(data)
        elif self._in_td:
            self._cell_buf.append(data)
        else:
            self._buf.append(data)

    def result(self) -> tuple[str, list[str]]:
        self._flush()
        title = " ".join("".join(self.title_parts).split()).strip()
        # Deduplicate adjacent identical lines and drop 1-char noise
        clean: list[str] = []
        for line in self.lines:
            if len(line) < 2:
                continue
            if clean and clean[-1] == line:
                continue
            clean.append(line)
        return title, clean


def extract_readable(markup: str, url: str, focus: str = "", max_chars: int = DEFAULT_CHARS) -> tuple[str, str]:
    """Extract (title, focused_text) from raw HTML. When `focus` is given, prioritizes matching lines."""
    parser = _ArticleExtractor(url)
    parser.feed(markup)
    parser.close()
    title, lines = parser.result()
    if not lines:
        return title, ""
    if not focus.strip():
        out, used = [], 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                break
            out.append(line)
            used += len(line) + 1
        return title, "\n".join(out)

    # Focus-weighted extraction: keep headings + top lines matching focus tokens + surrounding context
    terms = {w for w in re.findall(r"[a-zA-Z0-9]{3,}", focus.lower())}
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        hits = sum(1 for t in terms if t in low)
        score = hits * 4 + (3 if line.startswith("#") else 0) + (2 if "|" in line else 0)
        if re.search(r"[$€£₹¥]\s?\d|\d+\s?%|\b(?:19|20)\d{2}\b", line):
            score += 2
        if idx < 4:
            score += 2
        scored.append((score, idx, line))
    chosen_idx: set[int] = set()
    used = 0
    for score, idx, line in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used + len(line) + 1 > max_chars:
            continue
        chosen_idx.add(idx)
        used += len(line) + 1
    return title, "\n".join(lines[i] for i in sorted(chosen_idx))


def _host_allowed(host: str, allow: list[str], block: list[str]) -> str:
    """Return error string if `host` is blocked or private without an explicit allow rule, else ''."""
    if any(_matches(p, host) for p in block):
        return f"Host {host} is blocked by browser rules"
    prob = local_problem(host)
    if prob and not any(_matches(p, host) for p in allow):
        return f"Private or local address blocked: {prob}"
    return ""


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    *,
    allow: list[str],
    block: list[str],
    focus: str,
    max_chars: int,
    distiller=None,
) -> dict[str, Any]:
    try:
        clean_url, host = parse_url(url)
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc), "needs_browser": False}
    if host == "about:blank":
        return {"url": url, "ok": False, "error": "Invalid web URL", "needs_browser": False}
    err = _host_allowed(host, allow, block)
    if err:
        return {"url": clean_url, "ok": False, "error": err, "needs_browser": False}

    try:
        resp = await client.get(clean_url, follow_redirects=True)
    except Exception as exc:
        return {"url": clean_url, "ok": False, "error": f"Could not fetch: {exc}", "needs_browser": True}

    final_url = str(resp.url)
    final_host = page_host(final_url) or ""
    if final_host:
        err2 = _host_allowed(final_host, allow, block)
        if err2:
            return {"url": final_url, "ok": False, "error": err2, "needs_browser": False}

    ctype = (resp.headers.get("content-type") or "").lower()
    if resp.status_code >= 400:
        return {"url": final_url, "ok": False, "status": resp.status_code,
                "error": f"HTTP {resp.status_code}", "needs_browser": resp.status_code in (401, 403, 429, 503)}
    if "html" not in ctype and "text/" not in ctype and "xml" not in ctype:
        return {"url": final_url, "ok": False, "error": f"Non-text content ({ctype[:40]})", "needs_browser": True}

    raw = resp.text[:MAX_BYTES]
    wall = classify(probe_from_html(raw, status=resp.status_code))
    if wall:
        return {"url": final_url, "ok": False, "error": f"Blocked by {wall}", "needs_browser": True, "wall": wall}

    title, content = extract_readable(raw, final_url, focus=focus, max_chars=max_chars)
    if len(content) < 160 or (len(content) < 400 and JS_SHELL_RE.search(raw[:4000])):
        return {"url": final_url, "ok": False, "title": title,
                "error": "Page requires JavaScript rendering; use browser_open.", "needs_browser": True}

    distilled = False
    if distiller is not None and focus and len(content) > 1200:
        try:
            short = await distiller(content, focus, title)
            if short and len(short) >= 60:
                content = short
                distilled = True
        except Exception:
            pass

    return {
        "url": final_url,
        "ok": True,
        "title": title,
        "content": html.unescape(content),
        "distilled": distilled,
        "needs_browser": False,
    }


async def read_many(
    urls: list[str],
    *,
    focus: str = "",
    allow: list[str] | None = None,
    block: list[str] | None = None,
    max_chars_each: int = DEFAULT_CHARS,
    distiller=None,
    transport=None,
) -> list[dict[str, Any]]:
    """Fetch and extract up to MAX_URLS web pages in parallel."""
    picked = [str(u).strip() for u in (urls or []) if str(u or "").strip()][:MAX_URLS]
    if not picked:
        return []
    per_page = max(1800, min(max_chars_each, int(10_000 // max(1, len(picked)))))
    headers = chrome_headers()
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers=headers, trust_env=False, transport=transport) as client:
        return list(await asyncio.gather(*(
            _fetch_one(client, u, allow=allow or [], block=block or [], focus=focus,
                       max_chars=per_page, distiller=distiller)
            for u in picked
        )))
