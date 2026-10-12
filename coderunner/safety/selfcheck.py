"""A security self-check of this installation, shown by `coderunner doctor`.

Each check looks at configuration and files only (nothing is changed or sent anywhere) and says what to do
when it fails. They cover the ways self-hosted assistants have most often been exposed: an owner API
reachable from the network, credentials other users can read, approvals switched off for committing steps,
and data leaving the machine through a remote model endpoint.
"""

from __future__ import annotations

import ipaddress
import os
import stat
import urllib.parse

from ..core import paths
from .policy import HIGH_RISK

LOOPBACK = ("127.0.0.1", "::1", "localhost")
ALWAYS_ASK = HIGH_RISK | {"browser.password"}     # kinds a rule must never allow outright


def _private(path) -> bool:
    try:
        return not stat.S_IMODE(os.stat(path).st_mode) & 0o077
    except FileNotFoundError:
        return True


def _local_host(host: str) -> bool:
    if host in LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check(name: str, ok: bool, detail: str, fix: str = "", severity: str = "high") -> dict:
    """severity high: a failure exposes the owner; info: a trade-off worth knowing about."""
    return {"check": name, "ok": ok, "severity": severity, "detail": detail,
            **({"fix": fix} if not ok and fix else {})}


def run(config) -> list[dict]:
    checks = []
    admin = str(config.get("gateway.admin_bind", "127.0.0.1"))
    checks.append(_check(
        "owner API on this machine only", _local_host(admin), f"gateway.admin_bind is {admin}",
        "set gateway.admin_bind to 127.0.0.1; reach it from your devices with `coderunner remote`"))

    sandbox_bind = str(config.get("gateway.sandbox_bind", "auto"))
    open_bind = sandbox_bind in ("0.0.0.0", "::")
    checks.append(_check(
        "sandbox API not on every network", not open_bind, f"gateway.sandbox_bind is {sandbox_bind}",
        "set gateway.sandbox_bind to auto (the sandbox network's own address)"))

    folder = paths.assistant_dir()
    exposed = [str(p) for p in (folder, folder / "admin-token", folder / "secrets.json", paths.config_path(),
                                paths.api_token_path()) if not _private(p)]
    checks.append(_check(
        "credentials readable only by you", not exposed,
        "ok" if not exposed else f"readable by other users: {', '.join(exposed)}",
        "chmod 600 the files and 700 the folder (the daemon fixes token files when it starts)"))

    loose = [r for r in config.get("policy.rules", []) or []
             if r.get("decision") == "allow" and r.get("kind") in ALWAYS_ASK]
    checks.append(_check(
        "committing steps still need your approval", not loose,
        "ok" if not loose else f"rules allow {', '.join(sorted({r['kind'] for r in loose}))} outright",
        "remove those rules from policy.rules (paying, sending and passwords should always ask)"))

    base = str(config.get("models.base_url") or "")
    host = urllib.parse.urlsplit(base).hostname or ""
    remote_model = bool(base) and not _local_host(host)
    checks.append(_check(
        "model runs on this machine", not remote_model,
        f"models.base_url is {base}" if base else "local runtime",
        "conversations, page text and file contents go to that endpoint; point models.base_url at a local "
        "runtime unless you trust it"))

    network = str(config.get("sandbox.network", "default"))
    checks.append(_check(
        "sandbox network goes through the egress proxy", network in ("proxy", "none"),
        f"sandbox.network is {network}",
        "code in the sandbox can reach any website (never your files or the owner API); set sandbox.network "
        "to proxy so it reaches only allowed hosts", severity="info"))

    owners = config.get("remote.owners", []) or []
    if config.get("remote.provider"):
        checks.append(_check(
            "remote access limited to your own login", bool(owners),
            f"remote.owners: {', '.join(owners) or 'none'}",
            "set remote.owners to your Tailscale login so other people on the tailnet are refused"))
    return checks
