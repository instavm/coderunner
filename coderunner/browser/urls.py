"""Which URLs and hosts the browser may open: web schemes only, canonical hosts, no local networks."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse

from .errors import BrowserError

LOCAL_NETS = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.168.0.0/16", "224.0.0.0/4", "255.255.255.255/32",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8"))


def _ipv4_number(host: str) -> ipaddress.IPv4Address | None:
    """WHATWG IPv4 parsing, so 127.1 or 0x7f.0.0.1 are seen the way the browser sees them."""
    labels = host.split(".")
    if not re.fullmatch(r"\d+|0x[0-9a-f]*", labels[-1]):
        return None
    if len(labels) > 4:
        raise BrowserError(f"Invalid IPv4 address {host}")
    numbers = []
    for label in labels:
        try:
            if label.startswith("0x"):
                numbers.append(int(label[2:] or "0", 16))
            elif len(label) > 1 and label.startswith("0"):
                numbers.append(int(label, 8))
            else:
                numbers.append(int(label))
        except ValueError:
            raise BrowserError(f"Invalid IPv4 address {host}") from None
    if any(n > 255 for n in numbers[:-1]) or numbers[-1] >= 256 ** (5 - len(numbers)):
        raise BrowserError(f"Invalid IPv4 address {host}")
    return ipaddress.IPv4Address(numbers[-1] + sum(n << 8 * (3 - i) for i, n in enumerate(numbers[:-1])))


def parse_url(url: str) -> tuple[str, str]:
    """Normalise a URL to (url, host). Only http(s) and about:blank are accepted.

    The URL is rebuilt from the parsed parts so the browser opens exactly the
    host that policy saw.
    """
    url = (url or "").strip()
    if not url:
        raise BrowserError("URL is empty")
    if url.lower() == "about:blank":
        return "about:blank", "about:blank"
    if "\\" in url or any(ord(c) <= 0x20 or ord(c) == 0x7f for c in url):
        raise BrowserError("URL must not contain spaces, backslashes or control characters; percent-encode them")
    scheme = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    if scheme is None or re.fullmatch(r"[^/:?#@]+:\d+(?:[/?#].*)?", url, re.S):
        url = "https://" + url                      # bare host such as example.com or example.com:8080/x
    elif scheme.group(1).lower() not in ("http", "https"):
        raise BrowserError(f"Only http and https URLs are allowed, not {scheme.group(1).lower()}:")
    parts = urllib.parse.urlsplit(url)
    if "@" in parts.netloc:
        raise BrowserError("URLs with embedded credentials are not allowed")
    try:
        port = parts.port
    except ValueError:
        raise BrowserError("URL has an invalid port") from None
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise BrowserError("URL has no host")
    if ":" in host:
        try:
            host = ipaddress.IPv6Address(host).compressed
        except ValueError:
            raise BrowserError(f"Invalid host {host}") from None
        netloc = f"[{host}]"
    else:
        try:
            host = host.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise BrowserError(f"Invalid host name {host}") from None
        if not re.fullmatch(r"[a-z0-9_.-]+", host):
            raise BrowserError(f"Invalid host name {host}")
        ip = _ipv4_number(host)
        if ip is not None:
            host = str(ip)
        netloc = host
    if port is not None:
        netloc += f":{port}"
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query,
                                    parts.fragment)), host


def is_local_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in LOCAL_NETS if net.version == ip.version)


def local_problem(host: str) -> str | None:
    """Why a host must not be opened without an explicit browser.allow entry, if it must not."""
    if host == "localhost" or host.endswith(".localhost"):
        return f"{host} is this computer"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    return f"{host} is a local or private network address" if is_local_ip(ip) else None


def page_host(url: str) -> str | None:
    """Policy target for a page URL; None for pages that are not ordinary web pages."""
    if url.startswith("about:blank"):
        return "about:blank"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme in ("http", "https") and parts.hostname:
        return parts.hostname.rstrip(".")
    return None


def same_site(a: str, b: str) -> bool:
    return a == b or a.removeprefix("www.") == b.removeprefix("www.")
