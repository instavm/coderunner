"""Private remote access to the admin UI over Tailscale Serve (never Funnel).

`enable()` runs `tailscale serve --bg --yes --https=443 http://127.0.0.1:<admin_port>`,
which publishes the admin app at https://<device>.<tailnet>.ts.net/ to devices
on the owner's tailnet only. Funnel (public internet exposure) is never used,
and enable() refuses to run while Funnel is on for the port it would use.

Threat model:
- Serve strips client-supplied Tailscale-User-* headers and sets
  Tailscale-User-Login / -Name for requests that pass through it. That only
  holds for traffic through serve. The admin app also listens on 127.0.0.1,
  where any local process, and Docker Desktop containers via
  host.docker.internal, can connect directly and forge those headers (and
  X-Forwarded-For).
- So identity headers are never sufficient. Every request still needs the
  paired-device cookie (or the admin token); config `remote.owners` and the
  headers only narrow which tailnet users may present a cookie. check_owner()
  cross-checks the login against `tailscale whois` of X-Forwarded-For when
  present.
- Restrict who can reach the device at all with a tailnet grant. Tag this
  machine `tag:coderunner` and allow only the owner to reach tcp:443:

    {"grants": [{"src": ["owner@example.com"], "dst": ["tag:coderunner"], "ip": ["tcp:443"]}]}
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from email.header import decode_header, make_header
from typing import Any, Callable

APP_BINARY = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
HTTPS_PORT = 443
LOGIN_HEADER = "Tailscale-User-Login"
WHOIS_TTL = 60.0

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class RemoteError(Exception):
    pass


def run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=30)


def find_binary(app_binary: str | None = None) -> str | None:
    """`tailscale` on PATH, else the CLI inside the macOS app bundle."""
    found = shutil.which("tailscale")
    if found:
        return found
    app_binary = app_binary or APP_BINARY
    return app_binary if os.path.isfile(app_binary) else None


def _tailscale(args: list[str], runner: Runner | None, binary: str | None) -> subprocess.CompletedProcess:
    binary = binary or find_binary()
    if not binary:
        raise RemoteError("Tailscale is not installed. Install it from https://tailscale.com/download and sign in.")
    try:
        return (runner or run)([binary, *args])
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemoteError(f"Could not run tailscale: {exc}") from None


def _json(result: subprocess.CompletedProcess) -> Any:
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None


def _detail(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or "").strip() or f"exit status {result.returncode}"


# ---- tailnet state -------------------------------------------------------

def tailscale_status(*, runner: Runner | None = None, binary: str | None = None) -> dict | None:
    """`tailscale status --json`, or None when Tailscale is missing or not answering."""
    try:
        result = _tailscale(["status", "--json"], runner, binary)
    except RemoteError:
        return None
    data = _json(result)
    return data if isinstance(data, dict) and data else None


def dns_name(status: dict | None = None, *, runner: Runner | None = None, binary: str | None = None) -> str | None:
    status = status if status is not None else tailscale_status(runner=runner, binary=binary)
    name = ((status or {}).get("Self") or {}).get("DNSName") or ""
    return name.rstrip(".") or None


def serve_config(*, runner: Runner | None = None, binary: str | None = None) -> dict:
    """`tailscale serve status --json`; raises when it cannot be read, so callers fail closed."""
    result = _tailscale(["serve", "status", "--json"], runner, binary)
    data = _json(result)
    if result.returncode != 0 or not isinstance(data, dict):
        raise RemoteError(f"Could not read the Tailscale serve configuration: {_detail(result)}")
    return data


def _configs(config: dict) -> list[dict]:
    """The background config plus any foreground `tailscale serve` sessions."""
    return [config] + [c for c in (config.get("Foreground") or {}).values() if isinstance(c, dict)]


def _port_of(host_port: str) -> int | None:
    port = host_port.rsplit(":", 1)[-1]
    return int(port) if port.isdigit() else None


def _proxy_port(target: str) -> int | None:
    """Local port a serve proxy target points at, e.g. http://127.0.0.1:8224 -> 8224."""
    rest = target.split("://", 1)[-1].rstrip("/")
    host, _, port = rest.partition("/")[0].rpartition(":")
    if host in ("127.0.0.1", "localhost", "[::1]") and port.isdigit():
        return int(port)
    return int(rest) if rest.isdigit() else None


def _handlers(config: dict) -> list[tuple[str, str, str]]:
    """(host:port, path, proxy target) for every web handler."""
    found = []
    for cfg in _configs(config):
        for host_port, web in (cfg.get("Web") or {}).items():
            for path, handler in ((web or {}).get("Handlers") or {}).items():
                found.append((host_port, path, (handler or {}).get("Proxy") or ""))
    return found


def funnel_ports(config: dict, admin_port: int) -> list[str]:
    """Funnel entries that would expose us: on the HTTPS port we use, or proxying to the admin port."""
    ours = {hp for hp, _, target in _handlers(config) if _proxy_port(target) == admin_port}
    exposed = []
    for cfg in _configs(config):
        for host_port, allowed in (cfg.get("AllowFunnel") or {}).items():
            if allowed and (_port_of(host_port) == HTTPS_PORT or host_port in ours):
                exposed.append(host_port)
    return exposed


def _root_handler(config: dict, name: str) -> str | None:
    for host_port, path, target in _handlers(config):
        if host_port == f"{name}:{HTTPS_PORT}" and path == "/":
            return target
    return None


# ---- enable / disable / status -------------------------------------------

def enable(admin_port: int, *, force: bool = False, runner: Runner | None = None,
           binary: str | None = None) -> str:
    """Serve the admin app on https://<device>.ts.net/ inside the tailnet and return that URL."""
    port = int(admin_port)
    if not 0 < port < 65536:
        raise RemoteError(f"Invalid admin port {admin_port}")
    status = tailscale_status(runner=runner, binary=binary)
    if status is None:
        raise RemoteError("Tailscale is not installed or not running. Install it, open it and sign in.")
    state = status.get("BackendState")
    if state != "Running":
        raise RemoteError(f"Tailscale is not connected (state: {state or 'unknown'}). Open Tailscale and sign in.")
    name = dns_name(status)
    if not name:
        raise RemoteError("This device has no tailnet DNS name. Enable MagicDNS and HTTPS certificates in the "
                          "Tailscale admin console.")
    config = serve_config(runner=runner, binary=binary)
    exposed = funnel_ports(config, port)
    if exposed:
        raise RemoteError(f"Tailscale Funnel is on for {', '.join(exposed)}, which would put CodeRunner on the "
                          f"public internet. Turn it off (`tailscale funnel --https={HTTPS_PORT} off`) and retry.")
    existing = _root_handler(config, name)
    if existing and _proxy_port(existing) != port and not force:
        raise RemoteError(f"https://{name}/ already serves {existing}. Remove it with "
                          f"`tailscale serve --https={HTTPS_PORT} off`, or pass force to replace it.")
    result = _tailscale(["serve", "--bg", "--yes", f"--https={HTTPS_PORT}", f"http://127.0.0.1:{port}"],
                        runner, binary)
    if result.returncode != 0:
        raise RemoteError(f"tailscale serve failed: {_detail(result)}")
    return f"https://{name}/"


def disable(*, admin_port: int | None = None, runner: Runner | None = None, binary: str | None = None) -> str:
    """Stop serving on :443. With admin_port, leaves a handler that points somewhere else alone."""
    config = serve_config(runner=runner, binary=binary)
    name = dns_name(runner=runner, binary=binary) or ""
    existing = _root_handler(config, name) if name else None
    if not any(_port_of(hp) == HTTPS_PORT for hp, _, _ in _handlers(config)):
        return "Remote access was not enabled."
    if admin_port is not None and existing and _proxy_port(existing) != int(admin_port):
        return f"Left https://{name}/ alone: it serves {existing}, not CodeRunner."
    result = _tailscale(["serve", f"--https={HTTPS_PORT}", "off"], runner, binary)
    if result.returncode != 0:
        raise RemoteError(f"tailscale serve off failed: {_detail(result)}")
    return "Remote access disabled."


def status(admin_port: int | None = None, *, runner: Runner | None = None, binary: str | None = None) -> dict:
    binary = binary or find_binary()
    info: dict[str, Any] = {"installed": bool(binary), "running": False, "dns_name": None, "serving": False,
                            "url": None, "target": None, "funnel": False}
    if not binary:
        return info
    ts = tailscale_status(runner=runner, binary=binary)
    info["running"] = bool(ts) and ts.get("BackendState") == "Running"
    info["dns_name"] = dns_name(ts) if ts else None
    try:
        config = serve_config(runner=runner, binary=binary)
    except RemoteError as exc:
        info["error"] = str(exc)
        return info
    target = _root_handler(config, info["dns_name"]) if info["dns_name"] else None
    info["target"] = target
    info["serving"] = bool(target) and (admin_port is None or _proxy_port(target) == int(admin_port))
    info["url"] = f"https://{info['dns_name']}/" if info["serving"] else None
    info["funnel"] = bool(funnel_ports(config, int(admin_port))) if admin_port is not None else any(
        allowed for cfg in _configs(config) for allowed in (cfg.get("AllowFunnel") or {}).values())
    return info


# ---- identity --------------------------------------------------------------

def _header_values(headers: Any, name: str) -> list[str]:
    if hasattr(headers, "getlist"):
        return list(headers.getlist(name.lower()))
    items = headers.items() if isinstance(headers, Mapping) else headers
    return [str(v) for k, v in items if str(k).lower() == name.lower()]


def decode_identity(headers: Any, name: str = LOGIN_HEADER) -> str | None:
    """Tailscale-User-Login, decoding RFC 2047 (Tailscale encodes non-ASCII values that way).

    None means the header is absent. A repeated header yields "" and an undecodable one stays raw,
    so neither can match a configured owner by accident.
    """
    values = _header_values(headers, name)
    if not values:
        return None
    if len(values) > 1:
        return ""
    raw = values[0]
    try:
        return str(make_header(decode_header(raw))).strip()
    except Exception:
        return raw.strip()


def whois(ip: str, *, runner: Runner | None = None, binary: str | None = None) -> str | None:
    """Login name of the tailnet user behind an address, via `tailscale whois --json`."""
    try:
        address = str(ipaddress.ip_address(str(ip).strip()))
    except ValueError:
        return None
    try:
        result = _tailscale(["whois", "--json", address], runner, binary)
    except RemoteError:
        return None
    data = _json(result)
    if result.returncode != 0 or not isinstance(data, dict):
        return None
    return ((data.get("UserProfile") or {}).get("LoginName")) or None


_whois_cache: dict[str, tuple[float, str | None]] = {}


def cached_whois(ip: str) -> str | None:
    now = time.monotonic()
    hit = _whois_cache.get(ip)
    if hit and hit[0] > now:
        return hit[1]
    login = whois(ip)
    _whois_cache[ip] = (now + WHOIS_TTL, login)
    return login


def check_owner(headers: Any, owners: list[str], *, lookup: Callable[[str], str | None] | None = None) -> bool:
    """Whether serve-provided identity (if any) belongs to a configured owner.

    This only narrows who may use a paired-device cookie; it never authenticates on its own.
    Requests without identity headers (direct local access) pass, since the cookie still decides.
    The lookup runs a subprocess; call this from a worker thread in async code.
    """
    if not owners:
        return True
    login = decode_identity(headers)
    forwarded = _header_values(headers, "X-Forwarded-For")
    if login is None and not forwarded:
        return True
    if login is not None and login not in owners:
        return False
    if forwarded:
        who = (lookup or cached_whois)(forwarded[0].split(",")[0].strip())
        if who is None or who not in owners or (login is not None and who != login):
            return False
    return True
