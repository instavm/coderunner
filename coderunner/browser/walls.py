"""Bot walls: a page that stands between the browser and the site (a challenge, a CAPTCHA, "Access denied").

Sites behind Cloudflare, DataDome, PerimeterX (HUMAN), Akamai or Imperva sometimes answer the sandbox's Chrome
with such a page instead of the site: the VM honestly looks like ARM Linux with software WebGL, which their
scoring dislikes. The broker looks at a page once it has loaded (WALL_JS, in its isolated world: no page script
sees it) and classify() says which wall it is, or None. The browser/switch module then asks the owner whether
that site may continue in the browser on their Mac. Nothing here solves or touches a challenge.

What counts is a wall *instead of* the site, not a site that mentions one: a page of its own with little text
and few links, marked by the vendor's own interstitial (an element id, a script address, its title), or an
error status with "verify you are human" words. A long page that talks about CAPTCHAs, a sign-in form with a
reCAPTCHA or Turnstile box in it (the owner's step, handed over as before), and Cloudflare's bot-detection
script that rides along on ordinary pages (/cdn-cgi/challenge-platform/scripts/jsd/) are not walls.
"""

from __future__ import annotations

import html as html_lib
import re

# What the page gives the check: its status, title, visible text (cut), link count and the start of its markup.
# The markup is what the vendors' interstitials are known by; it is read here, never shown to the model.
WALL_JS = """(() => {
  const nav = performance.getEntriesByType('navigation')[0] || {};
  const text = document.body ? document.body.innerText || '' : '';
  return {status: nav.responseStatus || 0, title: document.title || '', text: text.slice(0, 3000),
          length: text.length, links: document.links.length,
          html: document.documentElement ? document.documentElement.outerHTML.slice(0, 60000) : ''};
})()"""

SHORT_TEXT = 2500           # characters of visible text a wall page has at most (Cloudflare's: ~250, DataDome's: ~0)
FEW_LINKS = 20              # and links (a wall links its vendor and maybe a help page; a site's page has dozens)
ERROR_STATUS = frozenset({401, 403, 405, 429, 503})

# Markup only the vendors' own interstitials carry. A page with one of these and little else is their wall.
MARKERS = (
    ("Cloudflare", re.compile(r"_cf_chl_opt|id=[\"']?challenge-(?:form|running|stage|error-text|body-text)\b|"
                              r"cf-browser-verification|cf_chl_", re.I)),
    ("DataDome", re.compile(r"captcha-delivery\.com", re.I)),
    ("PerimeterX", re.compile(r"px-captcha|_pxCaptcha|captcha\.px-cdn\.net", re.I)),
    ("Akamai", re.compile(r"/_sec/cp_challenge/|sec-if-cpt-container|sec-cpt-if|errors\.edgesuite\.net", re.I)),
    ("Imperva", re.compile(r"_Incapsula_Resource|Incapsula incident ID|imperva_challenge", re.I)),
)
# Titles the walls show (Cloudflare's in English; it translates the rest of its page, not always the title).
TITLES = (
    ("Cloudflare", re.compile(r"^\s*(just a moment\.{0,3}|attention required!? \| cloudflare|one more step)\s*$", re.I)),
    ("PerimeterX", re.compile(r"^\s*access to this page has been denied\.?\s*$", re.I)),
    ("Akamai", re.compile(r"^\s*access denied\s*$", re.I)),
    ("Imperva", re.compile(r"^\s*pardon our interruption\.{0,3}\s*$", re.I)),
    ("", re.compile(r"^\s*(are you a (robot|human)\??|human verification|bot check|security check|"
                    r"verify(ing)? you are (a )?human\.{0,3})\s*$", re.I)),
)
# Words of a wall, counted only on a short page that also has an error status or a wall's title. HUMAN alone
# with an error status will do; DENIED needs the title too (a site's own short 403 page says "access denied").
HUMAN = re.compile(
    r"verify(ing)? (that )?you are (a )?human|confirm you are (a )?human|are you a robot|not a robot|"
    r"press (&|and) hold|checking (if the site connection is secure|your browser)|enable javascript and cookies to "
    r"continue|automated (queries|requests|access)|you have been blocked", re.I)
DENIED = re.compile(r"you don.t have permission to access|has been blocked|access (is )?denied|request unsuccessful|"
                    r"reference #\s*\d|ray id", re.I)
GOOGLE = re.compile(r"unusual traffic from your computer network", re.I)


def classify(probe: dict | None) -> str | None:
    """The wall a page is ("Cloudflare", "DataDome", ..., or "a bot check" when no vendor is named), or None for
    a page of the site itself. probe is what WALL_JS returned."""
    if not isinstance(probe, dict):
        return None
    text = str(probe.get("text") or "")
    length = int(probe.get("length") or len(text))
    links = int(probe.get("links") or 0)
    if length > SHORT_TEXT or links > FEW_LINKS:
        return None                 # a page of the site: whatever it mentions, the site answered
    markup, title = str(probe.get("html") or ""), str(probe.get("title") or "")
    status = int(probe.get("status") or 0)
    for vendor, pattern in MARKERS:
        if pattern.search(markup):
            return vendor
    if GOOGLE.search(text):
        return "Google"
    titled = next((vendor for vendor, pattern in TITLES if pattern.search(title)), None)
    human = bool(HUMAN.search(text))
    if titled is not None and (human or DENIED.search(text) or status in ERROR_STATUS):
        return titled or "a bot check"
    if status in ERROR_STATUS and human:
        return "a bot check"
    return None


def describe(vendor: str) -> str:
    """The wall in a few words for the model and the audit trail."""
    return "a bot check" if vendor == "a bot check" else f"a {vendor} bot check"


# ---- for tests and tools: a probe from a saved page -------------------------------------------------

_DROP = re.compile(r"<(script|style|noscript|template)\b.*?</\1\s*>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title\s*>", re.I | re.S)
_LINKS = re.compile(r"<a\b[^>]*\bhref=", re.I)


def probe_from_html(markup: str, status: int = 200) -> dict:
    """What WALL_JS would return for this page, near enough: a page's text without its scripts and tags."""
    title = _TITLE.search(markup)
    body = re.sub(r"<head\b.*?</head\s*>", " ", markup, flags=re.I | re.S)
    text = " ".join(html_lib.unescape(_TAGS.sub(" ", _DROP.sub(" ", body))).split())
    return {"status": status, "title": html_lib.unescape(title.group(1)).strip() if title else "",
            "text": text[:3000], "length": len(text), "links": len(_LINKS.findall(markup)), "html": markup[:60000]}
