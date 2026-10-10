"""Policy-enforced network egress for the sandbox.

In `proxy` network mode the sandbox VM sits on a host-only network and its
only route out is this HTTP(S) proxy on the host. Each destination host is
checked against the owner's allow list and grants; unknown hosts become an
approval request. Destinations that resolve to non-public addresses (LAN,
loopback, link-local, CGNAT/tailnet ranges) are refused unless explicitly
allowed, and the proxy connects to the address it checked so a DNS answer
cannot change between the check and the connection.

`fetch()` serves requests that need an owner secret: the sandbox sends a
header placeholder and the host substitutes the value for allowed hosts only.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import ipaddress
import re
import socket
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .audit import AuditLog
from .config import Config
from .policy import Action, Decision, Policy, _matches
from .secrets_store import PLACEHOLDER, SecretStore

HOSTNAME = re.compile(r"^(?=.{1,253}$)([A-Za-z0-9-_]{1,63}\.)*[A-Za-z0-9-_]{1,63}\.?$")
MAX_HEAD = 64 * 1024
MAX_FETCH_BYTES = 10 * 1024 * 1024


class EgressDenied(Exception):
    pass


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class EgressGuard:
    MAX_PENDING = 5
    DENIAL_MEMORY = 300

    def __init__(self, config: Config, policy: Policy, audit: AuditLog, resolver=None):
        self.config = config
        self.policy = policy
        self.audit = audit
        self._resolver = resolver
        self._tainted_until = 0.0
        self._denied: dict[str, float] = {}

    def taint(self, seconds: float = 1800) -> None:
        """Code written after untrusted content ran: allow-listed hosts need approval for a while."""
        self._tainted_until = max(self._tainted_until, time.time() + seconds)
        self.audit.write("egress.tainted", until=self._tainted_until)

    @property
    def tainted(self) -> bool:
        return time.time() < self._tainted_until

    async def _resolve(self, host: str, port: int) -> list[str]:
        if self._resolver:
            return await self._resolver(host, port)
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        return list(dict.fromkeys(info[4][0] for info in infos))

    async def check(self, host: str, port: int, origin: str = "sandbox") -> list[str]:
        """Returns addresses that may be connected to, or raises EgressDenied."""
        host = host.strip("[]").rstrip(".").lower()
        try:
            ipaddress.ip_address(host)
            literal = True
        except ValueError:
            literal = False
            if not HOSTNAME.match(host):
                raise EgressDenied(f"Invalid host name: {host!r}")
        if port not in (self.config.get("egress.ports", [80, 443]) or []):
            raise EgressDenied(f"Port {port} is not allowed")
        try:
            addresses = [host] if literal else await self._resolve(host, port)
        except OSError as exc:
            raise EgressDenied(f"Could not resolve {host}: {exc}") from exc
        if not addresses:
            raise EgressDenied(f"Could not resolve {host}")
        private_ok = any(_matches(p, host) for p in self.config.get("egress.allow_private", []) or [])
        if not private_ok and not all(_is_public(a) for a in addresses):
            self.audit.write("egress", host=host, port=port, allowed=False, reason="non-public address")
            raise EgressDenied(f"{host} resolves to a non-public address")
        if self._denied.get(host, 0) > time.time():
            raise EgressDenied(f"Connection to {host} was recently denied")
        action = Action("egress.connect", host, f"Sandbox wants to connect to {host}:{port}", {"port": port},
                        tainted=self.tainted)
        if self.policy.evaluate(action) is Decision.ASK and sum(
                1 for a in self.policy.pending() if a.kind == "egress.connect") >= self.MAX_PENDING:
            # Code in the sandbox must not be able to bury the owner in prompts.
            raise EgressDenied("Too many network approvals are already waiting")
        allowed = await self.policy.check(action, origin=origin,
                                          wait=float(self.config.get("egress.approval_wait", 60)))
        if not allowed:
            self._denied[host] = time.time() + self.DENIAL_MEMORY
            raise EgressDenied(f"Connection to {host} was not approved")
        return addresses


def _response(status: int, reason: str, body: str = "", extra: str = "") -> bytes:
    payload = body.encode()
    return (f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n{extra}\r\n").encode() + payload


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, counter: list[int], index: int):
    try:
        while data := await reader.read(65536):
            counter[index] += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


class EgressProxy:
    """Minimal HTTP/1.1 forward proxy supporting CONNECT and absolute-form requests."""

    def __init__(self, guard: EgressGuard, audit: AuditLog, token_provider):
        self.guard = guard
        self.audit = audit
        self.token_provider = token_provider

    async def start(self, host: str, port: int) -> asyncio.base_events.Server:
        return await asyncio.start_server(self.handle, host, port)

    def _authorized(self, headers: dict[str, str]) -> bool:
        value = headers.get("proxy-authorization", "")
        scheme, _, encoded = value.partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            _, _, password = base64.b64decode(encoded, validate=True).decode().partition(":")
        except (ValueError, UnicodeDecodeError):
            return False
        expected = self.token_provider()
        return bool(expected) and hmac.compare_digest(password.encode(), expected.encode())

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
            writer.close()
            return
        if len(head) > MAX_HEAD:
            writer.write(_response(431, "Request Header Fields Too Large"))
            writer.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, version = lines[0].split(" ", 2)
        except ValueError:
            writer.write(_response(400, "Bad Request"))
            writer.close()
            return
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        if not self._authorized(headers):
            writer.write(_response(407, "Proxy Authentication Required", "Proxy credential required\n",
                                   'Proxy-Authenticate: Basic realm="coderunner"\r\n'))
            await writer.drain()
            writer.close()
            return

        if method.upper() == "CONNECT":
            host, _, port_text = target.rpartition(":")
            port = int(port_text) if port_text.isdigit() else 443
        else:
            parts = urlsplit(target)
            if parts.scheme != "http" or not parts.hostname:
                writer.write(_response(400, "Bad Request", "Only absolute http:// URLs or CONNECT are supported\n"))
                writer.close()
                return
            host, port = parts.hostname, parts.port or 80

        try:
            addresses = await self.guard.check(host, port)
        except EgressDenied as exc:
            writer.write(_response(403, "Forbidden", f"CodeRunner egress policy: {exc}\n"))
            await writer.drain()
            writer.close()
            return

        upstream = None
        for address in addresses:
            try:
                upstream = await asyncio.wait_for(asyncio.open_connection(address, port), 15)
                break
            except (OSError, asyncio.TimeoutError):
                continue
        if upstream is None:
            writer.write(_response(502, "Bad Gateway", f"Could not connect to {host}\n"))
            writer.close()
            return
        up_reader, up_writer = upstream

        if method.upper() == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            parts = urlsplit(target)
            path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            kept = [line for line in lines[1:] if line and line.split(":", 1)[0].strip().lower()
                    not in ("proxy-authorization", "proxy-connection", "connection", "keep-alive")]
            request = "\r\n".join([f"{method} {path} {version}", *kept, "Connection: close", "", ""])
            up_writer.write(request.encode("latin-1"))
            await up_writer.drain()

        counter = [0, 0]
        await asyncio.gather(_pipe(reader, up_writer, counter, 0), _pipe(up_reader, writer, counter, 1))
        self.audit.write("egress", host=host, port=port, allowed=True, sent=counter[0], received=counter[1])


async def fetch(guard: EgressGuard, secrets: SecretStore, policy: Policy, request: dict[str, Any]) -> dict:
    """Perform one HTTP request on behalf of the sandbox, substituting secrets in headers.

    The request is sent to the address the guard checked (TLS still verifies the
    host name), so a second DNS answer cannot redirect it to the LAN or loopback.
    """
    url = str(request.get("url", ""))
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise EgressDenied("url must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise EgressDenied("Credentials in the URL are not allowed")
    host = parts.hostname.lower()
    port = parts.port or (443 if parts.scheme == "https" else 80)
    method = str(request.get("method", "GET")).upper()
    headers = {str(k): str(v) for k, v in (request.get("headers") or {}).items() if str(k).lower() != "host"}
    body = request.get("body")
    if body_b64 := request.get("body_b64"):
        body = base64.b64decode(body_b64)
    elif isinstance(body, str):
        body = body.encode()
    if PLACEHOLDER.search(url) or (body and PLACEHOLDER.search(body.decode("latin-1"))):
        raise EgressDenied("Secret placeholders are only substituted in headers")

    used: dict[str, str] = {}
    for key, value in list(headers.items()):
        for name in PLACEHOLDER.findall(value):
            if parts.scheme != "https":
                raise EgressDenied("Secrets are only sent over https")
            allowed_header = ((guard.config.get("secrets", {}) or {}).get(name) or {}).get("header")
            if allowed_header and allowed_header.lower() != key.lower():
                raise EgressDenied(f"Secret {name} may only be used in the {allowed_header} header")
            if not await policy.check(Action("secret.use", f"{name}@{host}", f"Use secret {name} for {host}"),
                                      origin="sandbox"):
                raise EgressDenied(f"Secret {name} is not allowed for {host}")
            secret = secrets.get(name)
            if secret is None:
                raise EgressDenied(f"Secret {name} is not set")
            used[name] = secret
        headers[key] = PLACEHOLDER.sub(lambda m: used.get(m.group(1), m.group(0)), value)

    addresses = await guard.check(host, port)
    address = addresses[0]
    netloc = f"[{address}]" if ":" in address else address
    pinned = parts._replace(netloc=f"{netloc}:{port}").geturl()
    default_port = 443 if parts.scheme == "https" else 80
    headers["Host"] = host if port == default_port else f"{host}:{port}"
    timeout = min(float(request.get("timeout", 60)), 300)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout, trust_env=False) as client:
        async with client.stream(method, pinned, headers=headers, content=body,
                                 extensions={"sni_hostname": host}) as response:
            chunks, size, truncated = [], 0, False
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_FETCH_BYTES:
                    truncated = True
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
            response_headers = dict(response.headers)
    # Never hand a secret back to the sandbox, even if the server echoes it.
    for secret in used.values():
        data = data.replace(secret.encode(), b"[secret]")
        response_headers = {k: v.replace(secret, "[secret]") for k, v in response_headers.items()}
    result: dict[str, Any] = {"status": response.status_code, "headers": response_headers, "truncated": truncated}
    try:
        result["body"] = data.decode("utf-8")
    except UnicodeDecodeError:
        result["body_b64"] = base64.b64encode(data).decode()
    return result
