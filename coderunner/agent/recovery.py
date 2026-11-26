"""Adaptive failure recovery, dead-page detection, dynamic strategy pivots, and plan-stagnation guards.

Local models (9B-27B) frequently get trapped in silent failure loops that standard tool error checks miss:
- Guessing URLs that return HTTP 404 / "Page Not Found" or redirect to an unrelated product/city.
- Retrying hosts that already presented a bot wall ("Just a moment...", "Access Denied").
- Spinning on plan_update without taking any concrete action after hitting an impasse.
- Staying stuck on step 1 of their plan for 20+ steps without switching sites or tactics.

RunMonitor tracks these signals per run with zero extra model calls on healthy steps, and escalates
thinking + targeted replan/pivot nudges only when concrete trouble occurs.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from .digest import normal_url

# Pages that are clearly dead / error pages even though the browser loaded them without throwing an exception.
NOT_FOUND_TITLE = re.compile(
    r"^\s*(?:404\b|error\s*404\b|page\s+not\s+found\b|fehler\s*404\b|seite\s+nicht\s+gefunden\b|"
    r"not\s+found\b|error\s*[-–—|]|oops[!.]?\s*.*not\s+found|page\s+introuvable\b|página\s+no\s+encontrada\b|"
    r"página\s+não\s+encontrada\b|お探しのページは見つかりません)",
    re.I,
)
NOT_FOUND_PATH = re.compile(r"/(?:404(?:\.html?)?|not[-_]?found|page[-_]?not[-_]?found)/?$", re.I)
NOT_FOUND_BODY = re.compile(
    r"\b(?:404\s+(?:error|not\s+found|page\s+not\s+found)|page\s+not\s+found|seite\s+nicht\s+gefunden|"
    r"this\s+page\s+(?:doesn.?t|does\s+not|cannot|could\s+not)\s+(?:exist|be\s+found)|"
    r"we\s+couldn.?t\s+find\s+(?:that|the)\s+page|sorry,?\s+we\s+can.?t\s+find\s+that\s+page|"
    r"cette\s+page\s+n.?existe\s+pas|no\s+se\s+encontró\s+la\s+página)\b",
    re.I,
)

BOT_WALL_TITLE = re.compile(
    r"^\s*(?:just\s+a\s+moment\.{0,3}|attention\s+required!?|access\s+(?:to\s+this\s+page\s+has\s+been\s+)?denied|"
    r"pardon\s+our\s+interruption\.{0,3}|one\s+more\s+step|robot\s+or\s+human\??|are\s+you\s+a\s+(?:robot|human)\??|"
    r"security\s+check|human\s+verification|verify\s+you\s+are\s+human)\b",
    re.I,
)
BOT_WALL_BODY = re.compile(
    r"\b(?:verify\s+(?:that\s+)?you\s+are\s+(?:a\s+)?human|unusual\s+traffic\s+from\s+your|"
    r"access\s+to\s+this\s+page\s+has\s+been\s+denied|press\s+(?:&|and)\s+hold|"
    r"checking\s+if\s+the\s+site\s+connection\s+is\s+secure|enable\s+javascript\s+and\s+cookies\s+to\s+continue|"
    r"automated\s+(?:access|queries|requests)\s+.*blocked)\b",
    re.I,
)

SEARCH_HOSTS = frozenset({
    "bing.com", "duckduckgo.com", "html.duckduckgo.com", "google.com", "search.brave.com",
    "search.yahoo.com", "startpage.com", "ecosia.org",
})

# Generic URL path tokens that do not identify a specific entity/product/location.
GENERIC_SLUG_TOKENS = frozenset({
    "home", "homes", "city", "county", "state", "unit", "product", "products", "item", "items",
    "page", "pages", "shop", "store", "site", "search", "find", "list", "listings", "detail",
    "details", "view", "index", "info", "about", "category", "categories", "filter", "sort",
    "sale", "rent", "html", "aspx", "default", "main", "public", "local", "market", "online",
    "with", "from", "that", "this", "best", "under", "near", "inch", "series", "edition",
    "groceries", "haushalt", "suche", "recherche", "location", "appartement", "immobilier",
    "transport", "flights", "hotels", "jobs", "review", "reviews", "price", "prices",
})

OWN_TOOLS = frozenset({"plan_update", "tell_owner", "show_items"})
BROWSER_PAGE_TOOLS = frozenset({
    "browser_open", "browser_navigate", "browser_snapshot", "browser_click",
    "browser_batch", "browser_back", "browser_type", "browser_select",
})

PLAN_STUCK_ANSWER = (
    "You updated the plan repeatedly without taking a new action. Without calling any tools, "
    "write your final reply to me now from what you have found: lead with the answer, include a "
    "link to each page your facts came from, and add one plain sentence if anything could not be finished."
)


def reg_host(url: str) -> str:
    """Registrable-ish host: www.amazon.in -> amazon.in, m.redfin.com -> redfin.com."""
    host = (urllib.parse.urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def is_search_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if host in SEARCH_HOSTS or reg_host(url) in SEARCH_HOSTS:
        return True
    return False


def _extract_headers(result: str) -> tuple[str, str, str]:
    """Extract (url, title, body) from a browser tool result string."""
    lines = result.splitlines()
    url = title = ""
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        s = line.strip()
        if s.startswith("<untrusted ") or s == "</untrusted>":
            continue
        if not in_body and s.startswith("url: "):
            url = s[5:].strip()
            continue
        if not in_body and s.startswith("title: "):
            title = s[7:].strip()
            continue
        if not in_body and re.match(r"^(?:tab|navigated|note|list|dialog|new_tabs|picked|tab_note):\s", s):
            continue
        if s or in_body:
            in_body = True
            body_lines.append(line)
    return url, title, "\n".join(body_lines).strip()


def _slug_tokens(text: str) -> set[str]:
    """Meaningful alpha/alphanumeric slug tokens (>= 4 chars, not generic boilerplate)."""
    tokens = set()
    for tok in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()):
        if tok not in GENERIC_SLUG_TOKENS:
            tokens.add(tok)
    return tokens


def _redirect_mismatched(requested_url: str, landed_url: str, title: str) -> bool:
    """True when browser_open(requested_url) silently redirected to an unrelated page or dropped to homepage."""
    if not requested_url or not landed_url:
        return False
    req = urllib.parse.urlsplit(requested_url)
    got = urllib.parse.urlsplit(landed_url)
    if reg_host(requested_url) != reg_host(landed_url):
        return False
    req_path = req.path.strip("/")
    got_path = got.path.strip("/")
    if not req_path or req_path == got_path:
        return False
    # Deep search/product URL bounced straight back to bare homepage "/" (e.g. otto.de/suche/?s=... -> otto.de/)
    if not got_path and not got.query and (len(req_path.split("/")) >= 1 and (req.query or len(req_path) >= 8)):
        return True
    req_segs = [seg for seg in req_path.split("/") if seg]
    got_segs = [seg for seg in got_path.split("/") if seg]
    if len(req_segs) < 2 or len(got_segs) < 2:
        return False
    req_tokens = _slug_tokens(req_path)
    if len(req_tokens) < 2:
        return False
    landed_tokens = _slug_tokens(got_path) | _slug_tokens(title)
    if not landed_tokens:
        return False
    return len(req_tokens & landed_tokens) == 0


def classify_dead_page(name: str, args: dict[str, Any], result: str) -> tuple[str, str, str]:
    """Inspect a browser tool result for silent dead ends.

    Returns (kind, url, title) where kind is "" (healthy) or one of:
    "not_found" | "bot_wall" | "redirect_mismatch" | "empty_page".
    """
    if name not in BROWSER_PAGE_TOOLS or not isinstance(result, str) or result.startswith("Error:"):
        return "", "", ""
    url, title, body = _extract_headers(result)
    if not url and not title:
        return "", "", ""
    if is_search_url(url):
        return "", url, title

    path = urllib.parse.urlsplit(url).path if url else ""
    short_body = len(body) < 1600

    if BOT_WALL_TITLE.search(title) or (short_body and BOT_WALL_BODY.search(body)):
        return "bot_wall", url, title

    if NOT_FOUND_PATH.search(path) or NOT_FOUND_TITLE.search(title) or (short_body and NOT_FOUND_BODY.search(body)):
        return "not_found", url, title

    if name == "browser_open" and isinstance(args.get("url"), str):
        if _redirect_mismatched(str(args["url"]), url, title):
            return "redirect_mismatch", url, title

    # A newly opened page with almost no visible text (empty SPA shell or silent block).
    if name == "browser_open" and len(body) < 80 and not title:
        return "empty_page", url, title

    return "", url, title


@dataclass
class PageRecord:
    url: str
    host: str
    title: str
    step: int


@dataclass
class RunMonitor:
    """Tracks per-run dead ends, verified opened pages, plan stagnation, and no-op loops."""

    host_dead: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)
    dead_urls: dict[str, tuple[str, int]] = field(default_factory=dict)
    opened_pages: dict[str, PageRecord] = field(default_factory=dict)   # normalized_url -> PageRecord
    own_only_streak: int = 0
    active_plan_step: str = ""
    plan_step_turns: int = 0
    plan_step_troubles: int = 0
    last_replan_turn: int = 0
    pivots: list[dict[str, Any]] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)

    def precheck_open(self, url: str) -> str:
        """If the exact URL already returned a 404/wall twice, short-circuit instead of wasting a reload."""
        norm = normal_url(url)
        prev = self.dead_urls.get(norm)
        if prev is not None and prev[1] >= 2:
            kind = prev[0]
            host = reg_host(url)
            return (
                f"Error: {url} already returned {kind.replace('_', ' ')} twice in this task. "
                f"Do not reopen it — use browser_search to find a working page or switch away from {host}."
            )
        return ""

    def record_call(
        self, name: str, args: dict[str, Any], result: str, step: int, failed: bool
    ) -> dict[str, Any]:
        """Inspect one completed tool call and return adaptive guidance signals.

        Returns {"note": str, "needs_think": bool, "dead_kind": str, "pivot_host": str}.
        """
        self.tools_used.add(name)
        out = {"note": "", "needs_think": False, "dead_kind": "", "pivot_host": ""}

        if failed:
            self.plan_step_troubles += 1
            out["needs_think"] = True
            return out

        dead_kind, url, title = classify_dead_page(name, args, result)
        if dead_kind:
            out["dead_kind"] = dead_kind
            out["needs_think"] = True
            self.plan_step_troubles += 1
            req_url = str(args.get("url") or url) if name == "browser_open" else url
            host = reg_host(req_url or url)
            # Record BOTH the requested URL and the landed URL (e.g. /missing -> /404) as dead,
            # and evict any prior ledger entry so error pages are never offered as verified citations.
            for candidate in {u for u in (req_url, url) if u}:
                norm = normal_url(candidate)
                prev_kind, prev_n = self.dead_urls.get(norm, (dead_kind, 0))
                self.dead_urls[norm] = (dead_kind, prev_n + 1)
                self.opened_pages.pop(norm, None)
            entries = self.host_dead.setdefault(host, [])
            entries.append((step, dead_kind, req_url or url))
            count = len(entries)
            if count >= 2:
                kinds = ", ".join(sorted({k.replace("_", " ") for _, k, _ in entries}))
                out["pivot_host"] = host
                self.pivots.append({"step": step, "host": host, "count": count, "kinds": kinds})
                out["note"] = (
                    f"\nNote: {host} has failed {count} times ({kinds}). Stop guessing URLs or retrying {host} — "
                    f"use browser_search to find a different source/site and update your plan with plan_update."
                )
            elif dead_kind == "not_found":
                out["note"] = (
                    f"\nNote: that URL returned a 404 / error page on {host}. Do not guess another URL path — "
                    f"use browser_search for a working link or use the site's own search box."
                )
            elif dead_kind == "redirect_mismatch":
                landed = title[:60] if title else (urllib.parse.urlsplit(url).path or "/")
                out["note"] = (
                    f"\nNote: {host} redirected to a different page ({landed!r}) instead of what you requested. "
                    f"That guessed URL was not valid — open the page from browser_search or browser_list instead."
                )
            elif dead_kind == "bot_wall":
                out["note"] = (
                    f"\nNote: {host} showed a bot check / access block. Switch to another site using browser_search "
                    f"rather than retrying {host}."
                )
            else:
                out["note"] = (
                    f"\nNote: {host} returned an empty page. Try browser_search or another site."
                )
            return out

        # Healthy page read: record its exact full URL in our verified opened-pages ledger.
        if url and url.startswith(("http://", "https://")) and not is_search_url(url):
            norm = normal_url(url)
            self.opened_pages[norm] = PageRecord(
                url=url,
                host=reg_host(url),
                title=(title or "").strip()[:120],
                step=step,
            )
        return out

    def record_step_tools(
        self, tool_names: list[str], any_no_change: bool = False, plan_changed: bool = False
    ) -> dict[str, Any]:
        """Update end-of-step loop counters; detect degenerate plan_update-only spins.

        Breaks immediately after 3 consecutive no-op plan/note steps (unchanged plan) or 5 consecutive
        own-only steps with no external tool action.
        """
        if any_no_change:
            self.plan_step_troubles += 1
        if tool_names and all(n in OWN_TOOLS for n in tool_names):
            self.own_only_streak += 1
            no_op_plan = set(tool_names) == {"plan_update"} and not plan_changed
            if no_op_plan:
                self.plan_step_troubles += 1
            if (self.own_only_streak >= 3 and no_op_plan) or self.own_only_streak >= 5:
                return {"force_answer": True, "needs_think": True, "note": ""}
            if self.own_only_streak == 2 and (no_op_plan or set(tool_names) == {"plan_update"}):
                return {
                    "force_answer": False,
                    "needs_think": True,
                    "note": (
                        "\nNote: two steps in a row only updated your plan without taking action. "
                        "Either execute the next concrete tool step now, or if you have what you need "
                        "(or the remaining step belongs to the owner), write your final answer now without calling tools."
                    ),
                }
        else:
            self.own_only_streak = 0
        return {"force_answer": False, "needs_think": False, "note": ""}

    def update_plan_progress(self, plan: list[dict[str, Any]]) -> str:
        """Track how long the active plan step has been stuck; return a replan hint when stagnating."""
        doing = next(
            (str(item.get("step") or "").strip() for item in (plan or [])
             if item.get("status") in ("doing", "in_progress", "active")),
            "",
        )
        if not doing:
            self.active_plan_step = ""
            self.plan_step_turns = 0
            self.plan_step_troubles = 0
            return ""
        if doing != self.active_plan_step:
            self.active_plan_step = doing
            self.plan_step_turns = 1
            self.plan_step_troubles = 0
            self.last_replan_turn = 0
            return ""
        self.plan_step_turns += 1
        if (
            self.plan_step_turns >= 6
            and self.plan_step_troubles >= 2
            and self.plan_step_turns - self.last_replan_turn >= 5
        ):
            self.last_replan_turn = self.plan_step_turns
            bad = [f"{h} ({len(v)}x)" for h, v in self.host_dead.items() if len(v) >= 2]
            bad_note = f" Dead ends so far: {', '.join(bad)}." if bad else ""
            return (
                f"\nYou have spent {self.plan_step_turns} steps on the current plan step with "
                f"{self.plan_step_troubles} setbacks.{bad_note} Rethink your approach: call plan_update "
                f"to pivot to a different source or method, or move to the next step with what you already have."
            )
        return ""

    def verified_pages(self, trail: list[dict[str, Any]] | None = None, limit: int | None = None) -> list[PageRecord]:
        """Deduplicated verified content URLs opened in this run (newest first).

        Prefers raw un-truncated URLs from `trail` over shortened (`…`) tool-output URLs, and strictly
        excludes any URL (requested or landed) recorded in `dead_urls`.
        """
        by_norm: dict[str, PageRecord] = {
            k: PageRecord(url=v.url, host=v.host, title=v.title, step=v.step)
            for k, v in self.opened_pages.items()
            if k not in self.dead_urls
        }
        for entry in trail or ():
            if not entry.get("ok", True):
                continue
            raw_url = str(entry.get("url") or "").strip()
            if not raw_url.startswith(("http://", "https://")) or is_search_url(raw_url):
                continue
            norm = normal_url(raw_url)
            if norm in self.dead_urls:
                continue
            if norm in by_norm:
                # Replace shortened query-value URL (`…`) with the raw full URL from the browser trail.
                if "…" in by_norm[norm].url or len(raw_url) > len(by_norm[norm].url):
                    by_norm[norm].url = raw_url
            else:
                by_norm[norm] = PageRecord(url=raw_url, host=reg_host(raw_url), title="", step=0)
        pages = sorted(by_norm.values(), key=lambda p: p.step, reverse=True)
        return pages[:limit] if limit is not None else pages
