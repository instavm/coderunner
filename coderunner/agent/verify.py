"""Deterministic pre-flight answer verification and surgical single-pass repair.

Local models often complete 90% of a multi-step task accurately and then stumble on the final reply:
- Dropping links to pages they actually opened, or citing a URL from a site they never opened.
- Forgetting an explicit deliverable asked by the user (a comparison table, a recommendation,
  a currency symbol, or answering in the language the user requested).
- Skipping run_python for arithmetic/totals when the prompt asked for it.
- Writing a vague stub ("the results are at the top of the page") instead of extracting the items.

Every check here is fast, deterministic, general-purpose, and zero-cost when the answer is already
complete. When a defect is caught, at most ONE append-only repair turn runs (preserving KV cache).
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Iterable

from .digest import normal_url
from .recovery import PageRecord, is_search_url, reg_host

URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
MONEY_RE = re.compile(
    r"(?:₹|Rs\.?|INR|MUR|US\$|\$|USD|€|EUR|£|GBP|¥|JPY|CHF|R\$)\s?\d[\d,.]*|"
    r"\d[\d,.]*\s?(?:rupees|INR|MUR|USD|dollars|euros|€|EUR|£|GBP|¥|円|JPY|yen|CHF)",
    re.I,
)
TABLE_ASK_RE = re.compile(
    r"\b(?:in a table|as a table|give a table|comparison table|als tabelle|en un tableau|en una tabla|em uma tabela)\b|表で",
    re.I,
)
RECOMMEND_ASK_RE = re.compile(
    r"\b(?:recommend one|recommend\b|which one to get|which is cheapest|which is better value|"
    r"which is best|best value|which day is best|empfiehl(?:e)?\s+einen|mais barat[oa]|say which is cheapest|"
    r"tell me which)\b",
    re.I,
)
RECOMMEND_HAS_RE = re.compile(
    r"\b(?:recommend|pick|best\s+(?:choice|value|option|buy|bet|day|fit)|cheapest|lowest\s+price|"
    r"go\s+with|choose|my\s+choice|get\s+the|I.d\s+(?:get|buy|take|go)|better\s+value|"
    r"empfehl|meine\s+wahl|am\s+besten|tipp|günstigste|mais\s+barat|menor\s+preço|mais\s+em\s+conta|"
    r"better\s+option|top\s+pick|clear\s+winner)\b",
    re.I,
)
PRICE_ASK_RE = re.compile(
    r"\b(?:price|prices|cost|costs|rent|loyer|nightly|fee|fees|fare|fares|total|budget|preis|preise|precio|preço)\b|料金",
    re.I,
)
MULTI_LINK_ASK_RE = re.compile(
    r"\b(?:link(?:s)?\s+(?:for|to)\s+each|each\s+with\s+(?:a|its)\s+link|with\s+(?:the\s+|their\s+)?links|"
    r"mit\s+(?:den\s+)?links|avec\s+(?:le\s+|les\s+)?liens?|com\s+os\s+links|source\s+link\s+for\s+each|"
    r"from\s+(?:two|three|2|3)\s+(?:different\s+)?(?:sites|sources|shops|retailers|review\s+sites)|"
    r"compare\s+three|find\s+three|find\s+two|three\s+\w+.*link)\b|リンク",
    re.I,
)
MULTI_COUNT_RE = re.compile(r"\b(?:three|3|drei|trois|três|tres)\b|3本", re.I)
TWO_COUNT_RE = re.compile(r"\b(?:two|2|zwei|deux|dois|duas|dos)\b|2つ", re.I)

PYTHON_EXPLICIT_RE = re.compile(r"\b(?:with|using|use|in)\s+python\b", re.I)
CALC_ASK_RE = re.compile(
    r"\b(?:what.?s\s+the\s+total|estimated\s+total|total\s+for\s+all|per\s+month\s+and\s+per\s+year|"
    r"work\s+out\s+(?:the|what)|effective\s+rate|share\s+of\s+monthly|best\s+value\s+per\s+square\s+foot|"
    r"until\s+what\s+date|counting\s+any\s+extended|how\s+many\s+days\s+(?:until|between)|notice\s+period)\b",
    re.I,
)
DATE_ARITH_RE = re.compile(
    r"\b(?:adds?\s+\d+\s*(?:years?|yrs?|months?|weeks?|days?)|"
    r"\d+[- ]?(?:year|yr|month)s?\s+(?:warranty|extended|cover|coverage|after|later|before|notice|extension)|"
    r"extended\s+(?:warranty|cover|by\s+\d+)|plus\s+\d+\s*(?:years?|months?|days?))\b",
    re.I,
)
STUB_PHRASES_RE = re.compile(
    r"\b(?:are\s+at\s+the\s+top\s+of\s+the\s+results|sont\s+en\s+haut\s+des\s+résultats|"
    r"results?\s+(?:are\s+)?displayed\s+on\s+the\s+page|shown\s+at\s+the\s+top|"
    r"you\s+can\s+see\s+them\s+on\s+the\s+page|listed\s+on\s+the\s+page)\b",
    re.I,
)
GAVE_UP_RE = re.compile(
    r"\b(?:I\s+)?(?:couldn.?t|could\s+not|was\s+unable\s+to|wasn.?t\s+able\s+to|did\s+not\s+manage\s+to|"
    r"was\s+blocked|can.?t|cannot)\s+(?:complete|finish|do|find|get|reach|load|open|confirm|verify)|"
    r"^I\s+had\s+to\s+stop|^I\s+stopped\s+after|^I\s+got\s+stuck",
    re.I | re.M,
)

# Explicit response-language directives and lightweight language detectors.
def _word_count(pattern: str, text: str) -> int:
    return len(re.findall(rf"\b(?:{pattern})\b", text, re.I))


LANG_CHECKS = (
    (re.compile(r"\bantworte\s+auf\s+deutsch\b|\bauf\s+deutsch\b", re.I),
     "German (auf Deutsch)",
     lambda t: _word_count("und|mit|für|ist|nicht|der|die|das|den|dem|bei|auf|ein|eine|liegt|kostet|unter", t) >= 5),
    (re.compile(r"\bréponds?\s+en\s+fran[çc]ais\b|\ben\s+fran[çc]ais\b", re.I),
     "French (en français)",
     lambda t: _word_count("avec|pour|dans|est|sont|le|la|les|des|du|une|et|au|aux|sur|par|mois|voici", t) >= 5),
    (re.compile(r"\bresponde\s+en\s+español\b|\ben\s+español\b", re.I),
     "Spanish (en español)",
     lambda t: _word_count("con|para|por|sale|llega|el|los|las|del|una|que|es|y", t) >= 5),
    (re.compile(r"\bresponda\s+em\s+portugu[êe]s\b|\bem\s+portugu[êe]s\b", re.I),
     "Portuguese (em português)",
     lambda t: _word_count("não|é|mais|com|para|os|da|uma|que|em|são|preço", t) >= 4),
    (re.compile(r"日本語で"),
     "Japanese (日本語)",
     lambda t: len(re.findall(r"[\u3040-\u30ff]", t)) >= 4),
)


def needs_extraction_nudge(request: str, final: str, read_pages: bool) -> str:
    """Catch a premature stub reply that points at open search results instead of listing the actual items."""
    if not read_pages or not final:
        return ""
    text = final.strip()
    if len(text) >= 400 or GAVE_UP_RE.search(text):
        return ""
    if "|" in text or MONEY_RE.search(text):
        return ""
    if len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)) >= 2:
        return ""
    if STUB_PHRASES_RE.search(text) and (MULTI_COUNT_RE.search(request) or TWO_COUNT_RE.search(request)):
        return (
            "You have not extracted the actual items and figures from the results page yet. "
            "Use browser_list or browser_snapshot to read the listings (prices, details, links), "
            "and then write the complete answer."
        )
    return ""


def needs_python_nudge(request: str, final: str, tools_used: set[str], has_python: bool) -> str:
    """Catch tasks that asked for Python, multi-item totals, or date/duration math where the model computed in its head."""
    if not has_python or "run_python" in tools_used or not final or GAVE_UP_RE.search(final):
        return ""
    explicit = bool(PYTHON_EXPLICIT_RE.search(request))
    calc = bool(CALC_ASK_RE.search(request)) and len(re.findall(r"\d+(?:[.,]\d+)?", final)) >= 4
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", final))
    date_math = len(years) >= 2 and bool(DATE_ARITH_RE.search(final))
    if explicit or calc or date_math:
        return (
            "Use run_python now to compute and verify every date addition, arithmetic step, or total from your "
            "figures (do not compute in your head), then write your complete final answer."
        )
    return ""


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)")


def _url_path_key(url: str) -> tuple[str, str]:
    """(registrable_host, normalized_path) for matching opened pages ignoring fragments/tracking params."""
    parts = urllib.parse.urlsplit(url)
    return reg_host(url), parts.path.rstrip("/") or "/"


def _is_opened_url(url: str, opened_norms: set[str], opened_paths: set[tuple[str, str]]) -> bool:
    """True only when `url` matches an actually opened URL (full normalized URL or exact host+path)."""
    if normal_url(url) in opened_norms:
        return True
    return _url_path_key(url) in opened_paths


def build_repair_prompt(
    request: str,
    final: str,
    opened_pages: Iterable[PageRecord],
    read_pages: bool,
    min_chars_for_links: int = 200,
    tools_used: set[str] | None = None,
    add_links_prefix: str = "",
) -> tuple[str, bool]:
    """Return (repair_prompt, needs_thinking) if the final answer has concrete defects, else ("", False).

    Never interpolates website-controlled titles into the user prompt (only verified URLs), preventing
    untrusted page metadata from steering the rewrite. Combines link grounding with completeness checks
    into a single repair prompt so one turn fixes everything.
    """
    text = (final or "").strip()
    if not text or GAVE_UP_RE.search(text[:450]):
        return "", False

    pages = [p for p in opened_pages if p.url.startswith(("http://", "https://")) and not is_search_url(p.url)]
    opened_norms = {normal_url(p.url) for p in pages}
    opened_paths = {_url_path_key(p.url) for p in pages}
    issues: list[str] = []
    include_urls = False
    needs_think = False

    # 1. Full-URL link grounding and coverage (rejects invented paths on opened hosts as well as unopened hosts)
    raw_urls = [_clean_url(u) for u in URL_RE.findall(text)]
    valid_urls = [u for u in raw_urls if _is_opened_url(u, opened_norms, opened_paths)] if pages else raw_urls
    unopened_urls = [u for u in raw_urls if not _is_opened_url(u, opened_norms, opened_paths)] if pages else []

    wants_links = bool(re.search(r"\b(?:links?|urls?|sources?|cite|liens?)\b|リンク", request, re.I))
    no_links_issue = False
    if read_pages and (pages or add_links_prefix):
        if (add_links_prefix or wants_links or len(text) >= min_chars_for_links) and not raw_urls:
            no_links_issue = True
            include_urls = bool(pages)
            if not add_links_prefix:
                issues.append(
                    "Add inline links [name](https://...) to the pages your facts came from, using ONLY the "
                    "verified URLs opened in this task listed below."
                )
        elif pages and raw_urls and not valid_urls:
            issues.append(
                "Replace every unverified or invented URL with exact verified URLs opened in this task from the list below."
            )
            include_urls = True
        elif unopened_urls:
            issues.append(
                "Replace any URL whose page or path was not actually opened with exact verified URLs from the list below."
            )
            include_urls = True
        elif pages and MULTI_LINK_ASK_RE.search(request):
            content_pages = [p for p in pages if len(p.url.split("/", 3)[-1].strip("/")) > 0]
            want = 3 if MULTI_COUNT_RE.search(request) else 2
            target = min(want, len({normal_url(p.url) for p in content_pages}))
            if target >= 2 and len({normal_url(u) for u in valid_urls}) < target:
                issues.append(
                    f"Include inline links for each item/source (at least {target} distinct opened pages) "
                    "using ONLY the exact URLs listed below."
                )
                include_urls = True

    # 2. Table explicitly requested
    if TABLE_ASK_RE.search(request) and not re.search(r"\|.+\|", text):
        issues.append("Format the comparison as a clean Markdown table.")

    # 3. Recommendation / pick explicitly requested
    if RECOMMEND_ASK_RE.search(request) and not RECOMMEND_HAS_RE.search(text):
        issues.append("Add a clear, decisive recommendation stating which option is best and why.")

    # 4. Explicit response language
    for ask_re, lang_name, ok_fn in LANG_CHECKS:
        if ask_re.search(request) and not ok_fn(text):
            issues.append(f"Rewrite the entire reply in {lang_name} as requested.")
            break

    # 5. Prices with currency
    if PRICE_ASK_RE.search(request) and len(text) >= 120 and re.search(r"\b\d{2,}", text) and not MONEY_RE.search(text):
        issues.append("Include the explicit currency symbol or code (e.g., $, €, £, ₹, ¥, R$) on every price.")

    # 6. Date/duration arithmetic performed without run_python (verify step-by-step with reasoning enabled)
    used = tools_used or set()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    if "run_python" not in used and len(years) >= 2 and DATE_ARITH_RE.search(text):
        issues.append(
            "Carefully verify every date and duration addition step by step (e.g., start year + base duration + "
            "extension = final end year) and fix any arithmetic error."
        )
        needs_think = True

    if not issues and not no_links_issue:
        return "", False

    parts: list[str] = []
    if no_links_issue and add_links_prefix:
        parts.append(add_links_prefix)
        if issues:
            parts.append("Also fix the following in the same rewrite:")
            parts.extend(f"- {item}" for item in issues)
    else:
        parts.append(
            "[Quality check before sending — not a new request] Without calling any tools, rewrite your reply "
            "to fix only the following:"
        )
        parts.extend(f"- {item}" for item in issues)
        parts.append("Keep every verified figure and fact accurate; do not invent data, headings or extra offers.")

    if include_urls and pages:
        # Verified URLs only (no website-controlled titles, so untrusted page text cannot steer the rewrite).
        url_lines = [f"- {p.url}" for p in pages[:12]]
        parts.append("\nVerified URLs opened in this task (use ONLY these):\n" + "\n".join(url_lines))
    return "\n".join(parts), needs_think
