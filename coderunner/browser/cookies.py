"""The owner's sign-ins for a few sites, brought from their own browser into the app's browser, when they ask.

Only ever on the owner's word (the first switch to the Mac browser offers it, unticked; Models and privacy has it
too), only for the sites they tick, never for every site. The cookies are read here and handed to the app's
browser session (app.AppBridge.set_cookies); nothing is kept, and no cookie value is logged or written anywhere.

Chrome (and Brave, Edge, Chromium) keeps its cookies in an SQLite file, their values encrypted with a key in the
login Keychain ("Chrome Safe Storage"); reading that key makes macOS ask the owner to allow it, and that prompt
is theirs to answer. The key is stretched with PBKDF2 (salt "saltysalt", 1003 rounds) into an AES-128-CBC key,
and since Chrome's database version 24 each value starts with the SHA-256 of its host. Firefox keeps its cookies
unencrypted (cookies.sqlite). Safari's are behind Full Disk Access and are not read.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sqlite3
import subprocess
import urllib.parse

CHROME_EPOCH = 11644473600          # seconds from 1601-01-01 (Chrome's time zero) to 1970-01-01
# Where each browser keeps its profiles on a Mac, and its Keychain entry (service, account).
BROWSERS = {
    "chrome": ("~/Library/Application Support/Google/Chrome", ("Chrome Safe Storage", "Chrome")),
    "brave": ("~/Library/Application Support/BraveSoftware/Brave-Browser", ("Brave Safe Storage", "Brave")),
    "edge": ("~/Library/Application Support/Microsoft Edge", ("Microsoft Edge Safe Storage", "Microsoft Edge")),
    "chromium": ("~/Library/Application Support/Chromium", ("Chromium Safe Storage", "Chromium")),
    "firefox": ("~/Library/Application Support/Firefox/Profiles", None),
}
NAMES = {"chrome": "Chrome", "brave": "Brave", "edge": "Edge", "chromium": "Chromium", "firefox": "Firefox"}
SAME_SITE = {-1: "unspecified", 0: "no_restriction", 1: "lax", 2: "strict"}
FIREFOX_SAME_SITE = {0: "no_restriction", 1: "lax", 2: "strict", 256: "unspecified"}
KEYCHAIN_WAIT = 120                 # seconds the owner has to answer macOS's prompt


class CookieError(Exception):
    """Something the owner can act on, said plainly."""


def root(browser: str, home: pathlib.Path | None = None) -> pathlib.Path:
    folder = BROWSERS[browser][0]
    return (home / folder[2:]) if home is not None else pathlib.Path(os.path.expanduser(folder))


def installed(home: pathlib.Path | None = None) -> list[str]:
    """The browsers here that have cookies to offer, Chrome first."""
    return [name for name in BROWSERS if cookie_files(name, home)]


def cookie_files(browser: str, home: pathlib.Path | None = None) -> list[pathlib.Path]:
    base = root(browser, home)
    if not base.is_dir():
        return []
    if browser == "firefox":
        return sorted(base.glob("*/cookies.sqlite"))
    found = []
    for profile in ["Default", *sorted(p.name for p in base.glob("Profile *"))]:
        for place in (base / profile / "Network" / "Cookies", base / profile / "Cookies"):
            if place.is_file():
                found.append(place)
                break
    return found


def clean_domains(domains: list[str]) -> list[str]:
    """Sites as the owner ticked them, made plain (example.com); refuses an empty list and anything that is not
    one site (a bare suffix like "com", a wildcard), so "everything" can never be asked for."""
    cleaned = []
    for domain in domains or []:
        text = str(domain or "").strip().lower()
        host = urllib.parse.urlsplit(text if "://" in text else "https://" + text).hostname or ""
        host = host.lstrip(".")
        if host.startswith("www."):
            host = host[4:]
        if "*" in text or "." not in host or host.endswith("."):
            raise CookieError(f"{domain!r} is not one site")
        if host not in cleaned:
            cleaned.append(host)
    if not cleaned:
        raise CookieError("Tick at least one site")
    return cleaned


def matches(host_key: str, domains: list[str]) -> bool:
    """Whether a cookie's domain belongs to one of the sites: the site itself or a host under it."""
    host = host_key.lower().lstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


# ---- Chrome ----------------------------------------------------------------------------------------

def keychain_password(browser: str) -> bytes:
    """The browser's Safe Storage password from the login Keychain. macOS asks the owner first."""
    service, account = BROWSERS[browser][1]
    try:
        done = subprocess.run(["/usr/bin/security", "find-generic-password", "-w", "-s", service, "-a", account],
                              capture_output=True, timeout=KEYCHAIN_WAIT)
    except (OSError, subprocess.TimeoutExpired):
        raise CookieError(f"{NAMES[browser]}'s sign-ins could not be read: the Keychain did not answer") from None
    if done.returncode != 0:
        raise CookieError(f"{NAMES[browser]}'s sign-ins were not shared (the Keychain prompt was declined or "
                           "there is no key)")
    return done.stdout.strip()


def chrome_key(password: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, 16)


def decrypt(encrypted: bytes, key: bytes, host_key: str, version: int) -> str | None:
    """A cookie value from Chrome's encrypted_value, or None when it cannot be read (another scheme)."""
    if not encrypted.startswith((b"v10", b"v11")):
        return None
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        raise CookieError("Reading Chrome's sign-ins needs the cryptography package") from None
    body = encrypted[3:]
    if not body or len(body) % 16:
        return None
    decryptor = Cipher(algorithms.AES(key), modes.CBC(b" " * 16)).decryptor()
    plain = decryptor.update(body) + decryptor.finalize()
    pad = plain[-1] if plain else 0
    if not 1 <= pad <= 16 or plain[-pad:] != bytes([pad]) * pad:
        return None
    plain = plain[:-pad]
    if version >= 24:
        if plain[:32] != hashlib.sha256(host_key.encode()).digest():
            return None
        plain = plain[32:]
    try:
        return plain.decode()
    except UnicodeDecodeError:
        return None


def _read_only(path: pathlib.Path) -> sqlite3.Connection:
    # immutable: read where the browser keeps it, without its lock and without a copy of the values on disk
    return sqlite3.connect(f"file:{urllib.parse.quote(str(path))}?mode=ro&immutable=1", uri=True)


def read_chrome(path: pathlib.Path, domains: list[str], key: bytes) -> list[dict]:
    db = _read_only(path)
    try:
        version = 0
        try:
            row = db.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()
            version = int(row[0]) if row else 0
        except sqlite3.Error:
            pass
        rows = db.execute("SELECT host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly, "
                          "samesite FROM cookies").fetchall()
    finally:
        db.close()
    found = []
    for host_key, name, value, encrypted, cookie_path, expires, secure, http_only, same_site in rows:
        if not matches(host_key, domains):
            continue
        if not value and encrypted:
            value = decrypt(bytes(encrypted), key, host_key, version)
            if value is None:
                continue
        expiry = (expires / 1_000_000 - CHROME_EPOCH) if expires else None
        found.append(electron_cookie(host_key, name, value or "", cookie_path, bool(secure), bool(http_only),
                                     expiry, SAME_SITE.get(same_site, "unspecified")))
    return found


# ---- Firefox -----------------------------------------------------------------------------------------

def read_firefox(path: pathlib.Path, domains: list[str]) -> list[dict]:
    db = _read_only(path)
    try:
        rows = db.execute("SELECT host, name, value, path, expiry, isSecure, isHttpOnly, sameSite "
                          "FROM moz_cookies").fetchall()
    finally:
        db.close()
    return [electron_cookie(host, name, value or "", cookie_path, bool(secure), bool(http_only),
                            float(expiry) if expiry else None, FIREFOX_SAME_SITE.get(same_site, "unspecified"))
            for host, name, value, cookie_path, expiry, secure, http_only, same_site in rows if matches(host, domains)]


# ---- both ------------------------------------------------------------------------------------------------

def electron_cookie(host_key: str, name: str, value: str, path: str, secure: bool, http_only: bool,
                    expiry: float | None, same_site: str) -> dict:
    """A cookie as Electron's session.cookies.set takes it. A host-only cookie (no leading dot) is set without a
    domain, as it was; a __Host- cookie never has one."""
    host = host_key.lstrip(".")
    cookie = {"url": f"{'https' if secure else 'http'}://{host}{path or '/'}", "name": name, "value": value,
              "path": path or "/", "secure": secure, "httpOnly": http_only, "sameSite": same_site}
    if host_key.startswith(".") and not name.startswith("__Host-"):
        cookie["domain"] = host_key
    if expiry:
        cookie["expirationDate"] = expiry
    return cookie


def read(browser: str, domains: list[str], home: pathlib.Path | None = None, password: bytes | None = None) -> list[dict]:
    """The cookies of these sites in the owner's browser (every profile of it), for the app's browser. password:
    the Safe Storage password, for tests; otherwise the Keychain is asked (and macOS asks the owner)."""
    if browser not in BROWSERS:
        raise CookieError(f"{browser} is not a browser this can read")
    domains = clean_domains(domains)
    files = cookie_files(browser, home)
    if not files:
        raise CookieError(f"No {NAMES[browser]} sign-ins were found on this Mac")
    if browser == "firefox":
        cookies = [c for f in files for c in read_firefox(f, domains)]
    else:
        key = chrome_key(password if password is not None else keychain_password(browser))
        cookies = [c for f in files for c in read_chrome(f, domains, key)]
    unique: dict[tuple, dict] = {}
    for cookie in cookies:                      # the first profile's copy of a cookie wins
        unique.setdefault((cookie.get("domain") or cookie["url"], cookie["path"], cookie["name"]), cookie)
    return list(unique.values())
