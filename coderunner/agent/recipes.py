"""Site recipes: how a website was used in a task that ended with an answer, offered the next time the site comes up.

A local model spends most of a website task finding its way (the home page, a form field at a time, the results)
at 16-18 s a step; once it has been there, the address that held the results usually takes the next search
directly with new values. A recipe keeps, per site, that address with its variable parts named ({from}, {date})
and the few steps that led to it. It never keeps what was typed (only a named placeholder), sign-in or one-time
values, or parameters that look like sessions, tracking or personal data. Newest wins; a stale one is ignored by
the model, replaced by the next task that gets there another way, and dropped when its address fails to open.
"""

from __future__ import annotations

import re
import time
import urllib.parse

from ..connectors.redact import TOKEN, _is_token, redact
from .driver import SEARCH

KEY = "site_recipes"
MAX_SITES = 100
MAX_STEPS = 8
MAX_CHARS = 600
OFFER_AT_ONCE = 3
LABEL_CHARS = 30
ACTIVE = ("type", "select", "press", "click")
HEADER = ("Site notes from earlier tasks: the address that held the results, its {named} parts to fill with this "
          "task's values, and the steps that led there. One may be out of date: ignore it if it fails.")
SEARCH_HOSTS = frozenset({urllib.parse.urlsplit(SEARCH).hostname.removeprefix("www."), "html.duckduckgo.com",
                          "duckduckgo.com"})
# Parameters that carry sessions, sign-ins, tracking or the owner's identity: never kept, whatever their value.
DROP_PARAM = re.compile(r"^(utm_.*|gclid|fbclid|msclkid|yclid|igshid|mc_[ce]id|_ga|_gl|srsltid|spm|ref_?|referr?er|"
                        r"sid|session.*|.*token.*|auth.*|key|api_?key|sig|signature|code|otp|state|nonce|password|"
                        r"pass|pwd|e?mail|phone|mobile|user.*|login.*|account.*|cookie.*|ticket)$", re.I)
EMAIL = re.compile(r"[^\s@/]+@[^\s@/]+\.[a-z]{2,}", re.I)
DATE = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
                  r"(20)?\d\d(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01]))$")
LONG_NUMBER = re.compile(r"\d{5,}")
ROLES = {"button", "link", "textbox", "combobox", "searchbox", "checkbox", "radio", "option", "tab", "menuitem",
         "listbox", "spinbutton", "slider", "switch", "cell", "row", "heading", "img"}


def host_of(url: str) -> str:
    """The site of an address, without www. or m.: "booking.com"."""
    try:
        host = (urllib.parse.urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)
    return host


def site_name(host: str) -> str:
    """The name a person calls a site by: "example" for example.com or example.co.in."""
    labels = host.split(".")[:-1]
    if len(labels) > 1 and labels[-1] in {"co", "com", "org", "net", "gov", "ac", "edu", "ne", "or"}:
        labels = labels[:-1]
    return labels[-1] if labels else host


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", urllib.parse.unquote_plus(str(text)).lower()))


def _slug(text: str, fallback: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    if len(words) > 1 and words[0] in ROLES:
        words = words[1:]
    slug = "_".join(words)[:20].strip("_")
    return slug if slug and not slug[0].isdigit() else fallback


class Values:
    """What this task typed or the owner named: a part of an address or a step equal to one of them is kept only as
    a named placeholder. From the request, the words a person writes as names or figures (capitalised, with a
    digit, or quoted): its lower-case words are as often a site's own ("flights", "news") as the owner's."""

    def __init__(self, request: str, typed: list[str]):
        self.typed = [_norm(t) for t in typed if _norm(t)]
        request = str(request or "")
        named = (re.findall(r"\b[A-Z][\w-]*|\b\w*\d[\w-]*", request)
                 + re.findall(r"[\"“']([^\"”']{2,40})[\"”']", request))
        self.named = {_norm(n) for n in named if _norm(n)}
        self.words = {w for n in self.named for w in n.split()}

    def match(self, value: str) -> bool:
        value = _norm(value)
        if not value:
            return False
        for typed in self.typed:
            if value == typed or (len(value) >= 3 and f" {value} " in f" {typed} ") or \
                    (len(typed) >= 3 and f" {typed} " in f" {value} "):
                return True
        return value in self.named or value in self.words


def _personal(value: str) -> bool:
    """A value that may identify the owner or a session: an email, a phone or long number, a random token."""
    text = urllib.parse.unquote_plus(str(value))
    return bool(EMAIL.search(text) or (LONG_NUMBER.search(text) and not DATE.match(text))
                or any(_is_token(m.group(0)) for m in TOKEN.finditer(text)))


class Names:
    """Placeholder names, each used once ({text}, {text2})."""

    def __init__(self, taken=()):
        self.used: set[str] = set(taken)

    def __call__(self, name: str) -> str:
        base, n = name, 2
        while name in self.used:
            name, n = f"{base}{n}", n + 1
        self.used.add(name)
        return "{" + name + "}"


def pattern(url: str, values: Values, typed_names: dict[str, str]) -> str | None:
    """The address with what varies named and what must not be kept left out; None when it must not be kept."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return None
    if redact(url) != url:                  # a sign-in, reset or verification link, or a token in it
        return None
    names = Names(typed_names.values())
    typed_in = {name: Values("", [value for value, owner in typed_names.items() if owner == name])
                for name in dict.fromkeys(typed_names.values())}

    def named(value: str, fallback: str) -> str | None:
        """The placeholder for a value that varies, "" for one left out, None for one kept as it is."""
        field = next((name for name, typed in typed_in.items() if typed.match(value)), "")
        if field:           # what was typed in a field, or the suggestion it picked ("New Delhi (DEL)" for del)
            return "{" + field + "}"
        if DATE.match(urllib.parse.unquote(value)):
            return names("date" if fallback == "term" else fallback)
        if values.match(value):
            return names(fallback)
        if _personal(value):
            return names("id" if fallback == "term" else fallback)
        if len(value) > 40:
            return names("id") if fallback == "term" else ""
        return None

    segments = [(named(segment, "term") if segment else None) or segment for segment in parts.path.split("/")]
    query = []
    for name, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        if DROP_PARAM.match(name):
            continue
        placeholder = named(value, _slug(name, "value"))
        if placeholder != "":
            query.append(f"{urllib.parse.quote(name)}={placeholder or urllib.parse.quote(value, safe=',:+-._~')}")
    return f"{parts.scheme}://{parts.netloc}{'/'.join(segments)}" + ("?" + "&".join(query) if query else "")


def _label(label: str) -> str:
    """An element's name as it was on screen, kept only when short and plainly not personal."""
    label = " ".join(str(label or "").split())[:LABEL_CHARS].strip()
    if not label or _personal(label) or redact(label) != label or re.search(r"\d{4}|[{}<>]", label):
        return ""
    return label


def build(trail: list[dict], host: str, request: str) -> dict | None:
    """The recipe for `host` from one task's browser steps: where its results were and the shortest way there."""
    done = [e for e in trail if e.get("ok") and e.get("url")]
    on = [i for i, e in enumerate(done) if host_of(e["url"]) == host]
    if not on:
        return None
    # The results: the latest page on the site whose address carries a search (a query), else its latest page.
    end = next((i for i in reversed(on) if urllib.parse.urlsplit(done[i]["url"]).query), on[-1])
    start = end
    while start > 0 and host_of(done[start - 1]["url"]) == host:
        start -= 1
    # Opened directly later on (a constructed address): the way starts there.
    start = next((i for i in range(end, start - 1, -1) if done[i].get("action") == "open"), start)
    typed = [str(e.get(k) or "") for e in trail for k in ("text", "picked", "option") if e.get(k)]
    values, names = Values(request, typed), Names()
    typed_names: dict[str, str] = {}
    steps: list[str] = []
    for e in done[start + 1:end + 1]:
        action, label = e.get("action"), _label(e.get("label", ""))
        where = f" in {label}" if label else ""
        step = ""
        if action == "type":
            name = names(_slug(label, "text")).strip("{}")
            for value in (e.get("text"), e.get("picked")):
                if _norm(value or ""):
                    typed_names.setdefault(_norm(value), name)
            step = f"type {{{name}}}{where}" + (" + Enter" if e.get("submit") else "")
        elif action == "select":
            option = str(e.get("option") or "")
            shown = (f'"{_label(option)}"' if _label(option) and not values.match(option)
                     else names(_slug(label, "option")))
            step = f"choose {shown}{where}"
        elif action == "press":
            key = str(e.get("key") or "")
            step = f"press {key}" if re.fullmatch(r"[A-Za-z]{1,12}", key) else ""
        elif action == "click":
            step = f"click {label}" if label and not values.match(label) else ""
        if step and (not steps or steps[-1] != step):
            steps.append(step)
    steps = steps[:MAX_STEPS]
    result = pattern(done[end]["url"], values, typed_names)
    start_url = pattern(done[start]["url"], values, typed_names) if steps else ""
    if result is None or start_url is None:
        return None
    recipe = {"host": host, "url": result, "start": start_url, "steps": steps, "saved": time.time()}
    while len(render(recipe)) > MAX_CHARS and recipe["steps"]:
        recipe["steps"] = recipe["steps"][:-1]
    return recipe if len(render(recipe)) <= MAX_CHARS else None


def render(recipe: dict) -> str:
    text = f"- {recipe['host']}: results at {recipe['url']}"
    if recipe.get("steps"):
        text += f"; reached by: open {recipe['start']}, " + ", ".join(recipe["steps"])
    return text


def matches(template: str, url: str) -> bool:
    """Whether an address is the recipe's own one filled in (same site and path shape; the query may differ)."""
    try:
        want, got = urllib.parse.urlsplit(template), urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if host_of(template) != host_of(url):
        return False
    regex = "".join(r"[^/]+" if re.fullmatch(r"\{\w+\}", piece) else re.escape(piece)
                    for piece in re.split(r"(\{\w+\})", want.path.rstrip("/")))
    return re.fullmatch(regex, got.path.rstrip("/")) is not None


class Recipes:
    """Per-site recipes in the store, newest per site, at most MAX_SITES sites."""

    def __init__(self, store):
        self.store = store

    def all(self) -> dict[str, dict]:
        found = self.store.get_kv(KEY, {})
        return found if isinstance(found, dict) else {}

    def get(self, host: str) -> dict | None:
        return self.all().get(host)

    def put(self, recipe: dict) -> None:
        saved = self.all()
        saved[recipe["host"]] = recipe
        if len(saved) > MAX_SITES:
            saved = dict(sorted(saved.items(), key=lambda item: item[1].get("saved", 0))[-MAX_SITES:])
        self.store.set_kv(KEY, saved)

    def drop(self, host: str) -> None:
        saved = self.all()
        if saved.pop(host, None) is not None:
            self.store.set_kv(KEY, saved)

    def named_in(self, text: str) -> list[str]:
        """The sites with a recipe that a request names, by address or by name ("on Example")."""
        lowered = str(text or "").lower()
        words = set(re.findall(r"[a-z0-9-]+", lowered))
        domains = {host_of("https://" + d) for d in re.findall(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", lowered)}
        return [host for host in self.all() if host in domains
                or (len(site_name(host)) >= 3 and site_name(host) in words)][:OFFER_AT_ONCE]

    def block(self, hosts: list[str]) -> str:
        """The note offered to the model for these sites, "" when none has a recipe."""
        saved = self.all()
        lines = [render(saved[host]) for host in hosts if host in saved]
        return HEADER + "\n" + "\n".join(lines) if lines else ""


def save_from_run(store, trail: list[dict], request: str, answer: str) -> list[str]:
    """After a task answered: a recipe for each site the answer links to (the pages its facts came from), or, when
    it links none, for the site the task ended on. Returns the sites saved."""
    used = {host_of(e["url"]) for e in trail if e.get("ok") and e.get("url")} - SEARCH_HOSTS - {""}
    sites: list[str] = []
    for link in re.findall(r"https?://[^\s)\]>\"']+", answer or ""):
        host = host_of(link)
        if host in used and host not in sites:
            sites.append(host)
    if not sites:
        last = next((host_of(e["url"]) for e in reversed(trail) if e.get("ok") and host_of(e.get("url", "")) in used),
                    "")
        sites = [last] if last else []
    recipes, saved = Recipes(store), []
    for host in sites[:OFFER_AT_ONCE]:
        recipe = build(trail, host, request)
        if recipe is not None:
            recipes.put(recipe)
            saved.append(host)
    return saved
