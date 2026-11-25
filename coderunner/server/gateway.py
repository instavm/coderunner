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
import contextlib
import hashlib
import hmac
import json
import mimetypes
import pathlib
import re
import secrets
import shutil
import tempfile
import time
import urllib.parse
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..safety.egress import EgressDenied, fetch
from ..safety.policy import current_run

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
        if path == "/embeddings" and getattr(svc, "embedding_url", ""):
            return f"{svc.embedding_url}{path}"      # e.g. the Core ML server, beside a chat model on llama.cpp
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

    import contextlib

    @contextlib.asynccontextmanager
    async def lifespan(app):          # on_shutdown is gone from Starlette 1.x; a lifespan works on every version
        try:
            yield
        finally:
            await upstream.aclose()

    app = Starlette(lifespan=lifespan, routes=[
        Route("/health", health, methods=["GET", "HEAD"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/v1/egress/fetch", egress_fetch, methods=["POST"]),
    ])
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
        if scope["type"] == "websocket":
            return await self._websocket(scope, receive, send)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        if (request.method, request.url.path) in self.OPEN or request.url.path.startswith("/app/"):
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

    async def _websocket(self, scope, receive, send):
        """The same rule for websockets: the admin token, or a paired device's cookie from this origin."""
        request = Request({**scope, "type": "http", "method": "GET"})
        if _equal(_bearer(request), self.svc.admin_token()):
            scope.setdefault("state", {})["principal"] = "admin-token"
            return await self.app(scope, receive, send)
        origin = request.headers.get("origin")
        same_origin = origin is not None and origin.split("://", 1)[-1] == request.headers.get("host", "")
        device = self._device(request)
        if device and same_origin:
            scope.setdefault("state", {})["principal"] = f"device:{device[:8]}"
            return await self.app(scope, receive, send)
        await send({"type": "websocket.close", "code": 1008})


MCP_TAINT_SECONDS = 1800


class _AppFiles(StaticFiles):
    """The app's page, script and styles change together with the daemon, so the browser is told to ask each
    time whether its copy is current: an old page kept beside a new script left the Computer panel dead."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


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

    async def setup(request: Request):
        """What `coderunner setup` does, for the app: its state, the plan for this machine, and starting it."""
        job = getattr(daemon, "setup", None)
        if job is None:
            return JSONResponse({"error": "setup is not available here"}, status_code=503)
        if request.method == "GET":
            return JSONResponse(job.public())
        body = await request.json()
        if body.get("action") == "plan":
            try:
                return JSONResponse(await asyncio.to_thread(job.plan, str(body.get("model") or "") or None))
            except Exception as exc:
                return JSONResponse({"error": f"The plan could not be made: {exc}"}, status_code=500)
        if body.get("action") == "location":       # where model files are kept; the plan comes back for it
            try:
                return JSONResponse(await asyncio.to_thread(job.set_location, str(body.get("path") or ""),
                                                            str(body.get("model") or "") or None))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:
                return JSONResponse({"error": f"The plan could not be made: {exc}"}, status_code=500)
        if body.get("action") == "start":
            started = job.start(model=str(body.get("model") or "") or None,
                                accept_license=bool(body.get("accept_license")))
            svc.audit.write("setup.started" if started else "setup.already_running", origin=principal(request))
            return JSONResponse({**job.public(), "started": started}, status_code=200 if started else 409)
        return JSONResponse({"error": "action must be plan, location or start"}, status_code=400)

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
        from ..voice import STT_HINT
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

    async def invoke_for_client(client: str, who: str, name: str, arguments: Any) -> tuple[Any, list[dict]]:
        """A guarded tool call on behalf of an MCP client.

        Each call is separate, so what a client has read is remembered for a while: after it
        reads untrusted content, its sensitive actions need fresh approval, and after it also
        reads private data, sharing that data somewhere new needs approval.
        """
        from ..agent import RunContext
        client = str(client or "local")[:64]
        ctx = RunContext(conversation=f"mcp:{client}", origin=f"mcp:{who}")
        tainted_until, content_until = mcp_taint.get(client, (0.0, 0.0))
        private_until, sources, shared = mcp_private.get(client, (0.0, set(), set()))
        now = time.time()
        ctx.tainted, ctx.content_tainted = tainted_until > now, content_until > now
        if private_until > now:
            ctx.private_sources, ctx.shared = set(sources), set(shared)
        token = current_run.set(ctx)
        try:
            result = await svc.registry.invoke(name, arguments or {}, ctx)
        finally:
            current_run.reset(token)
        if ctx.tainted:
            mcp_taint[client] = (now + MCP_TAINT_SECONDS,
                                 now + MCP_TAINT_SECONDS if ctx.content_tainted else content_until)
        if ctx.private_sources:
            mcp_private[client] = (now + MCP_TAINT_SECONDS, ctx.private_sources, ctx.shared)
        return result, [{"mime": mime, "data_b64": data} for mime, data in ctx.images]

    async def call_tool(request: Request):
        """Lets local MCP clients (the stdio bridge) use the assistant's guarded tools."""
        body = await request.json()
        result, images = await invoke_for_client(body.get("client", "local"), principal(request),
                                                 request.path_params["name"], body.get("arguments", {}))
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
            if body.get("source_add"):
                index.sources.add(str(body["source_add"]))
                index.wake.set()
                return JSONResponse(index.sources.status())
            if body.get("source_remove"):
                source = str(body["source_remove"])
                index.sources.remove(source)
                from ..files.sources import SCHEMES
                if source in SCHEMES:
                    index._drop([r["path"] for r in svc.store.query("SELECT path FROM files WHERE path LIKE ?",
                                                                     (SCHEMES[source] + "://%",))])
                return JSONResponse(index.sources.status())
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


    # ---- the app: tasks in the background, their live activity, and the sandbox's own computer ----------

    async def app_page(request: Request):
        return RedirectResponse("/app/", status_code=307)

    async def start_task(request: Request):
        from ..files import attached
        body = await request.json()
        message = str(body.get("message", "")).strip()
        given = body.get("attachments") or []
        files = [str(item) for item in given if attached.path_of(str(item))] if isinstance(given, list) else []
        if not message and not files:
            return JSONResponse({"error": "message is required"}, status_code=400)
        if len(files) > attached.MAX_PER_TASK:
            return JSONResponse({"error": f"At most {attached.MAX_PER_TASK} files to a message."}, status_code=400)
        # {"remote": true}: this one task on Claude with the kept key (the owner's retry of a task the local model
        # did not finish). Not private, which the app says on the button that sends it.
        on_claude = ""
        from ..models import remote
        if body.get("remote") and (getattr(svc, "remote", None) or {}).get("provider") != remote.PROVIDER:
            if not remote.key():
                return JSONResponse({"error": "No Anthropic key is kept.", "needs_key": True}, status_code=409)
            on_claude = remote.MODELS[remote.claude_model(svc.config)]
        elif not svc.config.get("models.chat") and not getattr(svc, "remote", None):
            return JSONResponse({"error": "No model is set up yet."}, status_code=409)
        conversation = str(body.get("conversation") or "")
        if not conversation.startswith("app:"):
            conversation = "app:" + secrets.token_hex(6)
        return JSONResponse(daemon.tasks.start(conversation, message, principal(request), attachments=files,
                                               **({"remote": on_claude} if on_claude else {})))

    async def remote_mode(request: Request):
        """Remote mode: the assistant's model on Anthropic's API with the owner's key (models/remote.py). GET says
        how it stands; POST {"action": "on", "key" or "key_file", "model"} turns it on, {"action": "off"} goes
        back to the model on this computer. The daemon starts again in place after either. {"action": "key",
        "key"} only keeps the key, for single tasks sent to Claude; nothing restarts. {"action": "server",
        "base_url", "model", "key"} uses the owner's own OpenAI-compatible model server instead."""
        from ..models import remote

        def state() -> dict:
            now = remote.active(svc.config)
            warning = remote.server_warning(now["host"]) if now and now["provider"] == remote.SERVER else remote.WARNING
            return {"on": now is not None, **(now or {}), "has_key": bool(remote.key()), "warning": warning,
                    "models": [{"id": key, "name": name} for key, name in remote.MODELS.items()],
                    "local": svc.config.get("models.chat") or ""}
        if request.method == "GET":
            return JSONResponse(state())
        body = await request.json()
        try:
            if body.get("action") == "key":
                # Only kept, for a task sent to Claude now and then; remote mode stays as it is.
                if body.get("key"):
                    remote.keep_key(str(body["key"]))
                else:
                    remote.keep_key_file(str(body.get("key_file") or ""))
                svc.audit.write("remote.key_kept", origin=principal(request))
                return JSONResponse(state())
            if body.get("action") == "on":
                if body.get("key"):
                    remote.keep_key(str(body["key"]))
                elif body.get("key_file"):
                    remote.keep_key_file(str(body["key_file"]))
                await asyncio.to_thread(remote.turn_on, svc.config, str(body.get("model") or remote.DEFAULT),
                                        lambda line: None)
            elif body.get("action") == "server":
                # The owner's own OpenAI-compatible server: it has to answer, with that model, before it is used.
                base_url, model = str(body.get("base_url") or "").rstrip("/"), str(body.get("model") or "")
                key = str(body.get("key") or "")
                try:
                    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                        listed = (await client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}
                                                   if key else {})).json()
                    served = [m.get("id") for m in listed.get("data", [])]
                except Exception:
                    raise ValueError(f"No model server answered at {base_url}.") from None
                # A name the server dropped in an upgrade (bonsai2 to bonsai2.5) is taken as its successor; the
                # answer names the model used.
                if model not in served and "bonsai" in model.lower():
                    model = next((name for name in [remote.successor(served, model)] if name and "bonsai" in name.lower()), model)
                if model not in served:
                    raise ValueError(f"That server serves {', '.join(map(str, served)) or 'no model'}, not {model}.")
                remote.use_server(svc.config, base_url, model, key)
            elif body.get("action") == "off":
                remote.turn_off(svc.config, forget=bool(body.get("forget")))
            else:
                return JSONResponse({"error": "action must be on, off, key or server"}, status_code=400)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({**state(), "error": str(exc)}, status_code=400)
        svc.audit.write("remote.changed", on=state()["on"], model=state().get("model", ""), origin=principal(request))
        restart = getattr(daemon, "restart_soon", None)
        if restart is not None:
            restart()
        return JSONResponse({**state(), "restarting": restart is not None})

    async def check(request: Request):
        """`coderunner check`: the core tasks run on this install and checked (server/check.py)."""
        job = getattr(daemon, "check", None)
        if job is None:
            return JSONResponse({"error": "the check is not available here"}, status_code=503)
        if request.method == "GET":
            return JSONResponse(job.public())
        body = await request.json()
        if body.get("action") == "stop":
            job.stop()
            return JSONResponse(job.public())
        names = [str(n) for n in body.get("names") or []] if isinstance(body.get("names"), list) else []
        suite = str(body.get("suite") or ("real" if body.get("real") else "check"))
        try:
            minutes = max(0, int(body.get("minutes") or 0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "minutes must be a whole number"}, status_code=400)
        problem = job.start(names or None, long=bool(body.get("long")), suite=suite, minutes=minutes)
        if problem:
            return JSONResponse({**job.public(), "error": problem}, status_code=409)
        svc.audit.write("check.started", origin=principal(request), names=names, long=bool(body.get("long")))
        return JSONResponse(job.public())

    async def upload(request: Request):
        """A file the owner attaches to a task: the body is the file, X-File-Name its name (URL-encoded). It is kept where the
        file tools and the sandbox can reach it (files/attached.py), and named in a task by the id returned."""
        from ..files import attached
        try:
            name = urllib.parse.unquote(request.headers.get("x-file-name") or "") or request.query_params.get("name") or ""
            kept = await attached.save(request.stream(), name, attached.limit_bytes(svc.config))
        except attached.TooLarge as exc:
            return JSONResponse({"error": f"That file is {exc}."}, status_code=413)
        except ValueError as exc:
            return JSONResponse({"error": f"That file was not kept: {exc}."}, status_code=400)
        svc.audit.write("files.attached", name=kept["name"], kind=kept["kind"], size=kept["size"],
                        origin=principal(request))
        return JSONResponse(kept)

    async def output_file(request: Request):
        """A file a task's code wrote to the sandbox's outputs folder, for the app to show under its answer: the
        same rule as an attached file (pictures, video, sound and PDFs in place, anything else as a download)."""
        from ..files import attached
        path = attached.output_path(request.path_params["name"])
        if path is None:
            return JSONResponse({"error": "no such file"}, status_code=404)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {"Cache-Control": "private, no-cache", "X-Content-Type-Options": "nosniff",
                   "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'"}
        if attached.INLINE.fullmatch(mime):
            return FileResponse(path, media_type=mime, headers=headers, content_disposition_type="inline", filename=path.name)
        return FileResponse(path, media_type="application/octet-stream", headers=headers, filename=path.name)

    async def uploaded(request: Request):
        """An attached file, for the app to show: pictures, video, sound and PDFs in place, anything else only
        as a download (an attached HTML page must never run as the daemon's own)."""
        from ..files import attached
        info, path = attached.describe(request.path_params["id"]), attached.path_of(request.path_params["id"])
        if info is None:
            return JSONResponse({"error": "no such file"}, status_code=404)
        headers = {"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff",
                   "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'"}
        if info["inline"]:
            return FileResponse(path, media_type=info["mime"], headers=headers, content_disposition_type="inline",
                                filename=info["name"])
        return FileResponse(path, media_type="application/octet-stream", headers=headers, filename=info["name"])

    async def cancel_task(request: Request):
        return JSONResponse({"cancelled": daemon.tasks.cancel(request.path_params["id"])})

    async def shot(request: Request):
        """A picture of a page from a task's thread (tasks.py), by the name its event carries."""
        from .tasks import shots_dir
        name = request.path_params["name"]
        path = shots_dir() / name
        if not re.fullmatch(r"[0-9a-f]{12}-i?\d{1,2}\.jpg", name) or not path.is_file():
            return JSONResponse({"error": "no such picture"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    async def conversations(request: Request):
        return JSONResponse(daemon.tasks.conversations())

    async def conversation(request: Request):
        name = request.path_params["conversation"]
        return JSONResponse({"conversation": name, "tasks": daemon.tasks.tasks(name),
                             "events": daemon.bus.since(0, conversation=name)})

    async def events(request: Request):
        """Server-sent events: what every task does, as it happens (after `since`, so a reconnect misses none)."""
        try:
            since = int(request.query_params.get("since") or request.headers.get("last-event-id") or 0)
        except ValueError:
            since = 0
        if since < 0:                       # only what happens from now on
            since = daemon.bus.latest()
        queue = daemon.bus.subscribe()

        async def stream():
            try:
                last = since
                for event in daemon.bus.since(since):
                    last = event["seq"]
                    yield f"id: {event['seq']}\ndata: {json.dumps(event, default=str)}\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if event["seq"] <= last:
                        continue
                    last = event["seq"]
                    yield f"id: {event['seq']}\ndata: {json.dumps(event, default=str)}\n\n"
            finally:
                daemon.bus.unsubscribe(queue)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    async def focus(request: Request):
        """The task the owner has open in the app: its window comes to the front of the assistant's computer,
        and it goes first when tasks take turns at the model."""
        body = await request.json()
        conversation = str(body.get("conversation") or "")
        if getattr(svc, "turns", None) is not None:
            svc.turns.focus = conversation
        shown = False
        if conversation and getattr(svc, "browser", None) is not None and hasattr(svc.browser, "focus"):
            shown = await svc.browser.focus(conversation)
        return JSONResponse({"conversation": conversation, "shown": bool(shown)})

    async def control(request: Request):
        """The owner took the screen in the app ("Take control") or handed it back ("Watching", "I'm done"). While
        they have it, the tasks' browser steps wait for them, so nothing clicks under their hands."""
        browser = getattr(svc, "browser", None)
        if request.method == "POST":
            body = {}
            with contextlib.suppress(Exception):
                body = await request.json()
            on = bool(body.get("on"))
            if on:                      # only with a screen to take: otherwise the tasks would wait on nothing
                screen = {}
                with contextlib.suppress(Exception):
                    screen = await svc.sandbox.desktop(start=False)
                on = bool(screen.get("available"))
            if browser is not None and hasattr(browser, "set_owner_control"):
                browser.set_owner_control(on, origin=principal(request))
        return JSONResponse({"on": bool(getattr(browser, "owner_control", False))})

    async def computer(request: Request):
        """The sandbox's screen for the app; POST brings it up. The password opens the screen through
        /api/computer/screen. Nothing here asks the owner to start or set anything up: while the screen is not
        there, the keeper (sandbox/keeper.py) gets the computer ready by itself and the app says only how that
        stands, in its own words: never the sandbox's error, a command to type, or a button."""
        keeper = getattr(svc, "computer", None)
        info = None
        with contextlib.suppress(Exception):
            info = await asyncio.wait_for(svc.sandbox.desktop(start=request.method == "POST"), 20)
        if info and info.get("available") and (info.get("running") or request.method == "GET"):
            if keeper is not None:
                keeper.saw_ready()
            shown = {**info, "state": "ready", "message": ""}
            shown.pop("error", None)
        else:
            if keeper is not None:
                keeper.kick(desktop=True)
                told = keeper.public()
            else:
                told = {"state": "absent", "message": "This install has no computer for the assistant."}
            if told["state"] == "ready":         # it was a moment ago; the next look finds out more
                told = {"state": "working", "message": "Getting the computer ready…"}
            shown = {"available": False, "running": False, **told}
        shown["mode"] = svc.browser.mode
        shown["browser_running"] = svc.browser.running()
        return JSONResponse(shown)

    async def computer_screen(websocket: WebSocket):
        """The VM's VNC server, relayed as a binary websocket for the app's viewer (noVNC)."""
        from urllib.parse import urlsplit
        offered = websocket.scope.get("subprotocols") or []
        await websocket.accept(subprotocol="binary" if "binary" in offered else None)
        host = urlsplit(svc.sandbox.base_url).hostname or "127.0.0.1"
        try:
            info = await svc.sandbox.desktop(start=True)
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(info.get("port") or 5900)), 10)
        except Exception:
            keeper = getattr(svc, "computer", None)
            if keeper is not None:
                keeper.kick(desktop=True)
            await websocket.close(code=1011, reason="The computer's screen is not ready.")
            return
        svc.audit.write("computer.screen_opened", by=websocket.scope.get("state", {}).get("principal", "?"))

        async def upstream():
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    data = message.get("bytes") or (message.get("text") or "").encode()
                    writer.write(data)
                    await writer.drain()
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                return

        async def downstream():
            try:
                while chunk := await reader.read(65536):
                    await websocket.send_bytes(chunk)
            except (ConnectionError, RuntimeError):
                return

        tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        writer.close()
        with contextlib.suppress(Exception):
            await websocket.close()

    mcp_all, mcp_search, mcp_lifespan = mcp_apps(svc, invoke_for_client)
    app = Starlette(lifespan=mcp_lifespan, routes=[
        # Routes with ASGI endpoints rather than mounts: a mount redirects /mcp to /mcp/, which MCP clients do
        # not follow.
        Route("/mcp", mcp_all),
        Route("/mcp/", mcp_all),
        Route("/mcp/search", mcp_search),
        Route("/mcp/search/", mcp_search),
        Route("/", index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/pair", pair, methods=["GET", "POST"]),
        Route("/api/status", status, methods=["GET"]),
        Route("/api/setup", setup, methods=["GET", "POST"]),
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
        Route("/app", app_page, methods=["GET"]),
        Mount("/app/", app=_AppFiles(directory=WEB_DIR / "app", html=True, check_dir=False)),
        Route("/api/tasks", start_task, methods=["POST"]),
        Route("/api/tasks/{id}/cancel", cancel_task, methods=["POST"]),
        Route("/api/shots/{name}", shot, methods=["GET"]),
        Route("/api/check", check, methods=["GET", "POST"]),
        Route("/api/remote", remote_mode, methods=["GET", "POST"]),
        Route("/api/uploads", upload, methods=["POST"]),
        Route("/api/uploads/{id}", uploaded, methods=["GET"]),
        Route("/api/outputs/{name}", output_file, methods=["GET"]),
        Route("/api/conversations", conversations, methods=["GET"]),
        Route("/api/conversations/{conversation}", conversation, methods=["GET"]),
        Route("/api/events", events, methods=["GET"]),
        Route("/api/computer", computer, methods=["GET", "POST"]),
        Route("/api/control", control, methods=["GET", "POST"]),
        Route("/api/focus", focus, methods=["POST"]),
        WebSocketRoute("/api/computer/screen", computer_screen),
    ])
    return OwnerAuth(app, svc)


def mcp_apps(svc, invoke):
    """The assistant's tools as an MCP server over Streamable HTTP (stateless: every request carries the token),
    at /mcp with every tool and /mcp/search with search, reading and memory only, for agents that run elsewhere
    (OpenClaw, LM Studio, a tool on another machine over the tailnet). Each principal is its own client for the
    taint and private-data memory."""
    import contextlib

    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from .mcp_bridge import tool_names

    def build(name: str, selection: str) -> Server:
        server = Server(name)

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["parameters"])
                    for t in tool_names(selection, svc.registry.describe())]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list:
            if selection == "search" and name not in {t["name"] for t in tool_names("search", svc.registry.describe())}:
                raise ValueError(f"{name} is not offered on this endpoint")
            ctx = server.request_context
            who = "unknown"
            try:
                who = ctx.request.scope.get("state", {}).get("principal", "unknown") if ctx.request else "unknown"
            except Exception:
                pass
            result, images = await invoke(f"http:{who}:{selection}", who, name, arguments or {})
            text = result if isinstance(result, str) else json.dumps(result)
            content: list = [types.TextContent(type="text", text=text)]
            content += [types.ImageContent(type="image", data=image["data_b64"], mimeType=image["mime"])
                        for image in images]
            return content
        return server

    managers = [StreamableHTTPSessionManager(app=build("coderunner", "all"), stateless=True, json_response=True),
                StreamableHTTPSessionManager(app=build("coderunner-search", "search"), stateless=True,
                                             json_response=True)]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            for manager in managers:
                await stack.enter_async_context(manager.run())
            yield

    class Endpoint:
        """An ASGI app (not a function, so Starlette's Route passes the connection through untouched)."""

        def __init__(self, manager):
            self.manager = manager

        async def __call__(self, scope, receive, send):
            await self.manager.handle_request(scope, receive, send)
    return Endpoint(managers[0]), Endpoint(managers[1]), lifespan


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
