"""HTTP surfaces of the daemon.

sandbox_app: what code inside the sandbox VM may call, bound only to the host
address the VM can reach and protected by the sandbox credential. It offers
the local model (OpenAI-compatible) and secret-injecting fetch. Nothing here
can approve actions, read memories, drive the browser or touch connectors.

admin_app: the owner's API and web UI, bound to loopback. Requests need the
admin token (CLI, MCP bridge) or a paired-device cookie (web UI, including
remote access through a private tailnet). Approvals are only decided here.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import pathlib
import secrets
import shutil
import tempfile
import time
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

from .egress import EgressDenied, fetch
from .policy import current_run

WEB_DIR = pathlib.Path(__file__).parent / "web"
DEVICE_COOKIE = "cr_device"
PAIR_TTL = 600
VOICE_LIMIT = 25 * 1024 * 1024      # a recorded message; about 100 minutes of Opus


def _bearer(request: Request) -> str:
    values = request.headers.getlist("authorization")
    if len(values) != 1:
        return ""
    scheme, _, token = values[0].partition(" ")
    return token if scheme.lower() == "bearer" else ""


def _equal(supplied: str, expected: str) -> bool:
    return bool(supplied) and bool(expected) and hmac.compare_digest(supplied.encode(), expected.encode())


class TokenAuth:
    """Bearer check for every route except GET /health."""

    def __init__(self, app, token_provider):
        self.app = app
        self.token_provider = token_provider

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == "/health" and scope["method"] in ("GET", "HEAD"):
            return await self.app(scope, receive, send)
        request = Request(scope)
        if not _equal(_bearer(request), self.token_provider()):
            response = JSONResponse({"error": "Authentication required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


# ---- sandbox-facing ------------------------------------------------------

def sandbox_app(svc) -> Starlette:
    upstream = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), trust_env=False)

    def _target(path: str) -> str:
        return f"{svc.llm.base_url}{path}"

    def _headers() -> dict:
        key = getattr(svc, "llm_api_key", "")
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    async def models(request: Request):
        return JSONResponse({"object": "list", "data": [
            {"id": "coderunner", "object": "model", "owned_by": "local"},
            {"id": svc.llm.model, "object": "model", "owned_by": "local"},
        ]})

    async def proxy(request: Request, path: str):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Body must be JSON"}, status_code=400)
        if body.get("model") in (None, "", "coderunner", "default"):
            body["model"] = svc.embedding_model if path == "/embeddings" and svc.embedding_model else svc.llm.model
        svc.audit.write("llm.proxy", path=path, model=body.get("model"), stream=bool(body.get("stream")),
                        messages=len(body.get("messages", []) or []), origin="sandbox")
        if body.get("stream"):
            req = upstream.build_request("POST", _target(path), json=body, headers=_headers())
            resp = await upstream.send(req, stream=True)

            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(relay(), status_code=resp.status_code,
                                     media_type=resp.headers.get("content-type", "text/event-stream"))
        resp = await upstream.post(_target(path), json=body, headers=_headers())
        return Response(resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"))

    async def chat(request: Request):
        return await proxy(request, "/chat/completions")

    async def embeddings(request: Request):
        return await proxy(request, "/embeddings")

    async def egress_fetch(request: Request):
        if svc.config.get("sandbox.network") == "none":
            return JSONResponse({"error": "The sandbox has no network access in 'none' mode"}, status_code=403)
        try:
            payload = await request.json()
            return JSONResponse(await fetch(svc.egress, svc.secrets, svc.policy, payload))
        except EgressDenied as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except (json.JSONDecodeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"Upstream error: {exc}"}, status_code=502)

    app = Starlette(routes=[
        Route("/health", health, methods=["GET", "HEAD"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/v1/egress/fetch", egress_fetch, methods=["POST"]),
    ], on_shutdown=[upstream.aclose])
    return TokenAuth(app, svc.api_token)


# ---- owner-facing --------------------------------------------------------

def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class OwnerAuth:
    """Admin token or paired-device cookie; CSRF header for cookie requests."""

    OPEN = {("GET", "/"), ("GET", "/pair"), ("POST", "/pair"), ("GET", "/health")}

    def __init__(self, app, svc):
        self.app = app
        self.svc = svc

    def _device(self, request: Request) -> str | None:
        cookie = request.cookies.get(DEVICE_COOKIE, "")
        if not cookie:
            return None
        row = self.svc.store.one("SELECT id FROM devices WHERE id = ?", (_hash(cookie),))
        if row is None:
            return None
        owners = self.svc.config.get("remote.owners", []) or []
        login = request.headers.get("tailscale-user-login")
        if owners and login is not None:
            from .remote import decode_identity
            if decode_identity(request.headers) not in owners:
                return None
        self.svc.store.execute("UPDATE devices SET last_seen = ? WHERE id = ?", (time.time(), row["id"]))
        return row["id"]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        if (request.method, request.url.path) in self.OPEN:
            return await self.app(scope, receive, send)
        if _equal(_bearer(request), self.svc.admin_token()):
            scope.setdefault("state", {})["principal"] = "admin-token"
            return await self.app(scope, receive, send)
        device = self._device(request)
        origin = request.headers.get("origin")
        same_origin = origin is None or origin.split("://", 1)[-1] == request.headers.get("host", "")
        if device and same_origin and (request.method in ("GET", "HEAD")
                                       or request.headers.get("x-coderunner") == "1"):
            scope.setdefault("state", {})["principal"] = f"device:{device[:8]}"
            return await self.app(scope, receive, send)
        response = JSONResponse({"error": "Authentication required"}, status_code=401)
        return await response(scope, receive, send)


MCP_TAINT_SECONDS = 1800


def admin_app(svc, daemon) -> Starlette:
    failures: dict[str, int] = {}
    mcp_taint: dict[str, tuple[float, float]] = {}
    # Per MCP client: (expiry, private data sources read, destinations the owner allowed sharing with).
    mcp_private: dict[str, tuple[float, set[str], set[str]]] = {}

    def principal(request: Request) -> str:
        return request.scope.get("state", {}).get("principal", "unknown")

    async def index(request: Request):
        return HTMLResponse((WEB_DIR / "index.html").read_text(),
                            headers={"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; "
                                     "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                     "frame-ancestors 'none'", "Cache-Control": "no-store"})

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    async def status(request: Request):
        return JSONResponse(await daemon.status())

    async def approvals(request: Request):
        return JSONResponse([a.public() for a in svc.policy.pending()])

    async def decide(request: Request):
        body = await request.json()
        try:
            approval = svc.policy.decide(request.path_params["id"], bool(body.get("allow")),
                                         body.get("scope", "once"), principal(request))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse(approval.public())

    async def grants(request: Request):
        return JSONResponse(svc.policy.grants())

    async def revoke(request: Request):
        svc.policy.revoke(int(request.path_params["id"]))
        return JSONResponse({"revoked": True})

    async def audit(request: Request):
        return JSONResponse(svc.audit.tail(int(request.query_params.get("limit", 100))))

    async def chat(request: Request):
        body = await request.json()
        message = str(body.get("message", "")).strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)
        conversation = str(body.get("conversation") or "web")
        reply = await daemon.handle_message("web", conversation, message, origin=principal(request))
        return JSONResponse({"reply": reply})

    async def voice_message(request: Request):
        """Transcribe a recording from the web page (the client then sends the text as a chat message)."""
        from .voice import STT_HINT
        voice = getattr(svc, "voice", None)
        if voice is None or not voice.available():
            return JSONResponse({"error": STT_HINT}, status_code=503)
        work = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-voice-"))
        try:
            note, size = work / "message", 0
            with open(note, "wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > VOICE_LIMIT:
                        return JSONResponse({"error": "The recording is too long."}, status_code=413)
                    handle.write(chunk)
            if not size:
                return JSONResponse({"error": "The recording is empty."}, status_code=400)
            try:
                text = await voice.transcribe(note)     # ffmpeg reads the format from the content
            except Exception as exc:
                svc.audit.write("voice.error", error=str(exc)[:300])
                return JSONResponse({"error": "I couldn't transcribe that recording."}, status_code=500)
            return JSONResponse({"text": text})
        finally:
            shutil.rmtree(work, ignore_errors=True)

    async def history(request: Request):
        conversation = f"web:{request.path_params['conversation']}"  # matches handle_message("web", ...)
        rows = svc.store.query("SELECT role, content, created FROM messages WHERE conversation = ? "
                               "ORDER BY id DESC LIMIT 50", (conversation,))
        return JSONResponse([{"role": r["role"], "content": json.loads(r["content"]), "created": r["created"]}
                             for r in reversed(rows)])

    async def tools(request: Request):
        return JSONResponse(svc.registry.describe())

    async def call_tool(request: Request):
        """Lets local MCP clients use the assistant's guarded tools.

        Each call is separate, so what a client has read is remembered for a while: after it
        reads untrusted content, its sensitive actions need fresh approval, and after it also
        reads private data, sharing that data somewhere new needs approval.
        """
        from .agent import RunContext
        body = await request.json()
        client = str(body.get("client", "local"))[:64]
        ctx = RunContext(conversation=f"mcp:{client}", origin=f"mcp:{principal(request)}")
        tainted_until, content_until = mcp_taint.get(client, (0.0, 0.0))
        private_until, sources, shared = mcp_private.get(client, (0.0, set(), set()))
        now = time.time()
        ctx.tainted, ctx.content_tainted = tainted_until > now, content_until > now
        if private_until > now:
            ctx.private_sources, ctx.shared = set(sources), set(shared)
        token = current_run.set(ctx)
        try:
            result = await svc.registry.invoke(request.path_params["name"], body.get("arguments", {}), ctx)
        finally:
            current_run.reset(token)
        if ctx.tainted:
            mcp_taint[client] = (now + MCP_TAINT_SECONDS,
                                 now + MCP_TAINT_SECONDS if ctx.content_tainted else content_until)
        if ctx.private_sources:
            mcp_private[client] = (now + MCP_TAINT_SECONDS, ctx.private_sources, ctx.shared)
        images = [{"mime": mime, "data_b64": data} for mime, data in ctx.images]
        return JSONResponse({"result": result, "images": images})

    async def files(request: Request):
        index = svc.files
        if index is None:
            return JSONResponse({"error": "file index unavailable"}, status_code=503)
        if request.method == "GET":
            query = request.query_params.get("q")
            if query:
                try:
                    limit = int(request.query_params.get("limit") or 8)
                except ValueError:
                    limit = 8
                return JSONResponse(await index.search(query, request.query_params.get("kind") or None, limit))
            return JSONResponse(index.status())
        body = await request.json()
        try:
            if body.get("add"):
                return JSONResponse({"added": index.add_root(str(body["add"])), **index.status()})
            if body.get("remove"):
                return JSONResponse({"forgotten": index.remove_root(str(body["remove"])), **index.status()})
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(await index.index_once(int(body.get("budget", 200))))

    async def schedules(request: Request):
        if svc.scheduler is None:
            return JSONResponse([])
        if request.method == "POST":
            body = await request.json()
            try:
                created = svc.scheduler.add(body["name"], body["spec"], body["prompt"], body.get("channel", ""),
                                            body.get("chat_id", ""))
            except (KeyError, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return JSONResponse(created)
        return JSONResponse(svc.scheduler.list())

    async def delete_schedule(request: Request):
        return JSONResponse({"removed": svc.scheduler.remove(int(request.path_params["id"]))})

    async def pair_start(request: Request):
        code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        svc.store.set_kv("pair_code", {"hash": _hash(code), "expires": time.time() + PAIR_TTL})
        failures.clear()
        svc.audit.write("remote.pair_started", by=principal(request))
        return JSONResponse({"code": code, "expires_in": PAIR_TTL})

    async def pair(request: Request):
        if request.method == "GET":
            return HTMLResponse((WEB_DIR / "index.html").read_text())
        if request.headers.get("x-coderunner") != "1":
            # A custom header forces a CORS preflight, so other websites cannot post here.
            return JSONResponse({"error": "Missing X-CodeRunner header"}, status_code=400)
        body = await request.json()
        pending = svc.store.get_kv("pair_code")
        client = request.client.host if request.client else "?"
        if not pending or pending["expires"] < time.time() or failures.get("total", 0) >= 5:
            return JSONResponse({"error": "No active pairing code. Run `coderunner remote pair`."}, status_code=403)
        if not hmac.compare_digest(_hash(str(body.get("code", "")).strip().upper()), pending["hash"]):
            failures["total"] = failures.get("total", 0) + 1
            if failures["total"] >= 5:
                svc.store.set_kv("pair_code", None)
            await asyncio.sleep(1)
            return JSONResponse({"error": "Wrong code"}, status_code=403)
        from .remote import decode_identity
        identity = decode_identity(request.headers)
        owners = svc.config.get("remote.owners", []) or []
        if owners and identity is not None and identity not in owners:
            return JSONResponse({"error": "This tailnet login may not pair"}, status_code=403)
        svc.store.set_kv("pair_code", None)
        device = secrets.token_urlsafe(32)
        name = str(body.get("name") or identity or client)[:80]
        svc.store.execute("INSERT INTO devices(id, name, created, last_seen) VALUES (?, ?, ?, ?)",
                          (_hash(device), name, time.time(), time.time()))
        svc.audit.write("remote.paired", device=name)
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response = JSONResponse({"paired": True})
        response.set_cookie(DEVICE_COOKIE, device, max_age=180 * 86400, httponly=True, secure=secure,
                            samesite="strict")
        return response

    async def devices(request: Request):
        return JSONResponse(svc.store.query("SELECT substr(id, 1, 8) AS id, name, created, last_seen FROM devices"))

    async def remove_device(request: Request):
        svc.store.execute("DELETE FROM devices WHERE substr(id, 1, 8) = ?", (request.path_params["id"],))
        return JSONResponse({"removed": True})

    async def logout(request: Request):
        cookie = request.cookies.get(DEVICE_COOKIE, "")
        if cookie:
            svc.store.execute("DELETE FROM devices WHERE id = ?", (_hash(cookie),))
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(DEVICE_COOKIE)
        return response

    app = Starlette(routes=[
        Route("/", index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/pair", pair, methods=["GET", "POST"]),
        Route("/api/status", status, methods=["GET"]),
        Route("/api/approvals", approvals, methods=["GET"]),
        Route("/api/approvals/{id}", decide, methods=["POST"]),
        Route("/api/grants", grants, methods=["GET"]),
        Route("/api/grants/{id:int}", revoke, methods=["DELETE"]),
        Route("/api/audit", audit, methods=["GET"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/chat/{conversation}", history, methods=["GET"]),
        Route("/api/voice", voice_message, methods=["POST"]),
        Route("/api/tools", tools, methods=["GET"]),
        Route("/api/tools/{name}", call_tool, methods=["POST"]),
        Route("/api/schedules", schedules, methods=["GET", "POST"]),
        Route("/api/files", files, methods=["GET", "POST"]),
        Route("/api/schedules/{id:int}", delete_schedule, methods=["DELETE"]),
        Route("/api/pair", pair_start, methods=["POST"]),
        Route("/api/devices", devices, methods=["GET"]),
        Route("/api/devices/{id}", remove_device, methods=["DELETE"]),
        Route("/logout", logout, methods=["POST"]),
    ])
    return OwnerAuth(app, svc)


def admin_request(method: str, path: str, *, base_url: str, token: str, json_body: Any = None,
                  timeout: float = 900) -> Any:
    """Small synchronous client used by the CLI and the MCP bridge."""
    # trust_env=False: never route the admin token through a system or environment proxy.
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        response = client.request(method, f"{base_url}{path}", json=json_body,
                                  headers={"Authorization": f"Bearer {token}"})
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:500]}")
    return response.json()
