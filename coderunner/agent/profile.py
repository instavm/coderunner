"""Structured, editable owner profile ("About You") persisted locally in SQLite kv.

Provides compact, trusted personal preferences (city, home airport, currency, dietary/travel notes) so
the assistant remembers durable context without repeating it across tasks. Never stores secrets; every
assistant-initiated profile update requires owner approval via Policy.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROFILE_KEY = "owner.profile"

FIELDS: tuple[tuple[str, str, str], ...] = (
    ("name", "Name", "How the assistant addresses you"),
    ("city", "Home city", "Default city for local searches, weather, and transit (e.g. Bengaluru, San Francisco)"),
    ("home_airport", "Home airport", "Default departure airport code (e.g. BLR, SFO, LHR)"),
    ("currency", "Preferred currency", "Currency for comparisons and totals (e.g. INR, USD, EUR, GBP)"),
    ("dietary", "Dietary preferences", "Allergies or meal preferences (e.g. vegetarian, vegan, nut-free)"),
    ("travel_prefs", "Travel & booking", "Non-stop preference, cabin class, loyalty programs, seating"),
    ("shopping_sites", "Preferred retailers", "Shops to check first (e.g. amazon.in, flipkart.com, bestbuy.com)"),
    ("work_role", "Work & background", "Role, tools, or languages for tailored answers"),
    ("notes", "Other lasting notes", "Any other durable preferences (never passwords, keys, or card numbers)"),
)
FIELD_KEYS = frozenset(k for k, _, _ in FIELDS)
MAX_LEN = {"notes": 800}
DEFAULT_MAX = 200

SECRET_RE = re.compile(
    r"(?:sk-ant-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"\b(?:password|passcode|otp|cvv|pin)\s*[:=]\s*\S+|\b(?:\d[ -]?){13,19}\b)",
    re.I,
)


def load(store) -> dict[str, str]:
    """Load the saved profile dict (only known non-empty fields)."""
    if store is None:
        return {}
    data = store.get_kv(PROFILE_KEY)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    return {k: str(v).strip() for k, v in data.items() if k in FIELD_KEYS and str(v or "").strip()}


def save(store, updates: dict[str, Any], *, merge: bool = True) -> dict[str, str]:
    """Validate and persist owner profile fields. Rejects secret-like values."""
    current = load(store) if merge else {}
    for key, val in (updates or {}).items():
        if key not in FIELD_KEYS:
            continue
        text = " ".join(str(val or "").split()) if key != "notes" else str(val or "").strip()
        if text and SECRET_RE.search(text):
            raise ValueError(f"Profile field {key!r} looks like a secret or card number and was not saved.")
        limit = MAX_LEN.get(key, DEFAULT_MAX)
        if text:
            current[key] = text[:limit]
        else:
            current.pop(key, None)
    store.set_kv(PROFILE_KEY, current)
    return current


def render_context(store) -> str:
    """Compact trusted profile block for Agent._context (~30-90 tokens), or '' when empty."""
    prof = load(store)
    if not prof:
        return ""
    labels = {k: lbl for k, lbl, _ in FIELDS}
    lines = [f"- {labels[k]}: {prof[k]}" for k, _, _ in FIELDS if k in prof]
    return "\nAbout the owner (saved profile — use as defaults unless the request specifies otherwise):\n" + "\n".join(lines)
