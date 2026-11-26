"""Owner-only Settings, Models, Benchmark Lab, Connectors, Browser, Voice, Routines, and Privacy APIs.

Every route delegates directly to existing tested modules (`config`, `models/*`, `eval/*`,
`connectors/macos`, `browser/extension`, `agent/scheduler`, `voice`, `server/remote`,
`safety/selfcheck`, `server/service`) behind strict allow-lists—never raw arbitrary config writes.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import shutil
import sys
import threading
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# Strictly allow-listed settings editable from the UI (value -> requires_restart).
LIVE_SETTINGS = {
    "owner.name": str,
    "owner.timezone": str,
    "owner.desktop_notifications": bool,
    "owner.onboarded": bool,
    "files.media": bool,
    "files.media_on_battery": bool,
    "files.pause_in_low_power": bool,
    "files.ocr": bool,
    "files.transcribe": bool,
    "files.code_in_repos": bool,
    "files.cloud_files": bool,
    "files.cloud_file_mb": int,
    "files.yield_during_llm_turn": str,
    "agent.thinking": str,
    "models.discover": bool,
    "models.idle_unload_minutes": int,
}

RESTART_SETTINGS = {
    "models.embedding_runtime": str,
    "sandbox.network": str,
    "sandbox.on_demand": str,
    "sandbox.disk_budget_gb": float,
    "browser.mode": str,
    "voice.stt": str,
    "voice.language": str,
}


class BenchJob:
    """Single-at-a-time background benchmark runner for Tier 0 probe, Tier 1 gate, Tier 2 combo compare, and ANE."""

    def __init__(self, daemon):
        self.daemon = daemon
        self._lock = threading.Lock()
        self.state = "idle"     # idle | running | done | failed
        self.kind = ""
        self.lines: list[str] = []
        self.result: dict | None = None

    def _say(self, line: str) -> None:
        with self._lock:
            self.lines.append(str(line))
            del self.lines[:-120]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"state": self.state, "kind": self.kind, "lines": list(self.lines[-40:]), "result": self.result}

    def start(self, kind: str, payload: dict[str, Any]) -> bool:
        with self._lock:
            if self.state == "running":
                return False
            self.state, self.kind, self.lines, self.result = "running", kind, [], None
        threading.Thread(target=self._run, args=(kind, payload), daemon=True, name="BenchJob").start()
        return True

    def _run(self, kind: str, payload: dict[str, Any]) -> None:
        from .. import eval as eval_pkg
        from ..models import discover, manager

        cfg = self.daemon.config
        try:
            if kind == "probe":
                res = eval_pkg.run_tier0(cfg, force=bool(payload.get("force", True)),
                                         include_cdn=bool(payload.get("cdn", True)), progress=self._say)
            elif kind == "discover":
                hw = manager._hardware(cfg, None)
                found = discover.cached(progress=self._say, refresh=True)
                planned = manager.plan(cfg, hw=hw)
                res = {"ok": True, "found": len(found), "plan": planned}
            elif kind == "tier1":
                res = asyncio.run(eval_pkg.run_tier1(cfg, progress=self._say))
            elif kind == "ane":
                res = asyncio.run(eval_pkg.run_stack(cfg, eval_pkg.stack_from_config(cfg), suite="ane",
                                                     progress=self._say, setup_model=False))
            elif kind == "compare":
                busy_fn = getattr(self.daemon, "_tasks_busy", None)
                if callable(busy_fn) and busy_fn():
                    res = {"ok": False, "error": "Wait for running tasks to finish before comparing models."}
                else:
                    raw_specs = payload.get("stacks") or []
                    specs = [eval_pkg.StackSpec.from_dict(s) for s in raw_specs if isinstance(s, dict)] \
                        or eval_pkg.default_candidates(cfg)[:3]
                    suite = str(payload.get("suite") or "tier1")
                    if suite not in ("tier1", "tier2", "ane"):
                        suite = "tier1"
                    pref = str(payload.get("preference") or "balanced")
                    res = asyncio.run(eval_pkg.compare_stacks(
                        cfg, specs[:4], suite=suite, preference=pref, progress=self._say,
                        reuse_cached=bool(payload.get("reuse_cached", True)),
                    ))
            else:
                res = {"ok": False, "error": f"Unknown benchmark kind {kind!r}"}
        except Exception as exc:
            self._say(f"error: {type(exc).__name__}: {exc}")
            res = {"ok": False, "error": str(exc)[:300]}
        with self._lock:
            self.result = res
            self.state = "done" if res.get("ok") else "failed"


def _installed_weights(config) -> list[dict[str, Any]]:
    """List downloaded local model folders inside models_dir() with disk sizes for safe cleanup."""
    from ..models.runtimes import models_dir
    root = models_dir(config)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in ("cache",):
            continue
        size = 0
        files = 0
        for p in child.rglob("*"):
            if p.is_file() and not p.name.endswith(".part"):
                with contextlib.suppress(OSError):
                    size += p.stat().st_size
                    files += 1
        if size > 0:
            out.append({"id": child.name, "path": str(child), "gb": round(size / 1e9, 2), "files": files})
    return sorted(out, key=lambda r: -r["gb"])


def _suggested_folders(config) -> list[dict[str, Any]]:
    home = pathlib.Path.home()
    shared = {str(pathlib.Path(p).expanduser().resolve()) for p in (config.get("files.roots") or [])}
    out = []
    for name in ("Documents", "Desktop", "Downloads"):
        folder = home / name
        if folder.is_dir():
            out.append({"name": name, "path": str(folder), "added": str(folder.resolve()) in shared})
    return out


def settings_routes(svc, daemon) -> list[Route]:
    """Build owner-only settings, models, benchmark lab, connectors, browser, routines, and privacy routes."""
    bench_job = getattr(daemon, "bench_job", None)
    if bench_job is None:
        bench_job = daemon.bench_job = BenchJob(daemon)

    async def settings_view(request: Request):
        from ..safety import selfcheck
        from . import service as login_service

        cfg = svc.config
        if request.method == "PATCH":
            body = await request.json()
            updates = body.get("settings") if isinstance(body.get("settings"), dict) else body
            restart_needed = False
            applied: list[str] = []
            for key, value in (updates or {}).items():
                if key in LIVE_SETTINGS:
                    caster = LIVE_SETTINGS[key]
                    cfg.set(key, bool(value) if caster is bool else caster(value))
                    applied.append(key)
                elif key in RESTART_SETTINGS:
                    caster = RESTART_SETTINGS[key]
                    cfg.set(key, bool(value) if caster is bool else caster(value))
                    applied.append(key)
                    restart_needed = True
            if "policy.ask_new_site" in (updates or {}):
                rules = [r for r in (cfg.get("policy.rules") or [])
                         if not (r.get("kind") == "browser.navigate" and r.get("target") == "*")]
                if updates["policy.ask_new_site"]:
                    rules.append({"kind": "browser.navigate", "target": "*", "decision": "ask"})
                cfg.set("policy.rules", rules)
                applied.append("policy.ask_new_site")
            if "egress.allow" in (updates or {}) and isinstance(updates["egress.allow"], list):
                cfg.set("egress.allow", [str(h).strip() for h in updates["egress.allow"] if str(h).strip()])
                applied.append("egress.allow")
                restart_needed = True
            if applied:
                cfg.save()
                svc.audit.write("settings.updated", keys=applied, restart=restart_needed)
            return JSONResponse({"ok": True, "applied": applied, "restart_required": restart_needed})

        checks = []
        with contextlib.suppress(Exception):
            checks = [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in selfcheck.run(cfg)]
        svc_status = {"installed": False, "running": False, "supported": sys.platform in ("darwin", "linux")}
        with contextlib.suppress(Exception):
            st = login_service.status()
            if isinstance(st, dict):
                svc_status.update(st)
        ask_new_site = any(r.get("kind") == "browser.navigate" and r.get("target") == "*" and r.get("decision") == "ask"
                           for r in (cfg.get("policy.rules") or []))
        values = {k: cfg.get(k) for k in (*LIVE_SETTINGS, *RESTART_SETTINGS)}
        values["policy.ask_new_site"] = ask_new_site
        values["egress.allow"] = cfg.get("egress.allow") or []
        return JSONResponse({
            "settings": values,
            "security": checks,
            "service": svc_status,
            "suggestions": _suggested_folders(cfg),
            "platform": sys.platform,
        })

    async def restart_daemon(request: Request):
        if daemon is None or not hasattr(daemon, "restart_soon"):
            return JSONResponse({"error": "restart is not available"}, status_code=503)
        daemon.restart_soon()
        return JSONResponse({"ok": True, "restarting": True})

    async def login_service_toggle(request: Request):
        from . import service as login_service
        body = await request.json()
        enable = bool(body.get("enable"))
        try:
            res = await asyncio.to_thread(login_service.install if enable else login_service.uninstall)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "enabled": enable, "detail": res})

    async def models_hub(request: Request):
        from .. import eval as eval_pkg
        from ..models import hardware, manager, probe, runtimes

        cfg = svc.config
        if request.method == "POST":
            body = await request.json()
            action = str(body.get("action") or "")
            if action == "driver":
                off = bool(body.get("off"))
                model_id = str(body.get("model") or "") or None
                try:
                    res = await asyncio.to_thread(manager.setup_driver, cfg, model_id=model_id, off=off)
                except Exception as exc:
                    return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
                if res.get("ok") and daemon is not None:
                    daemon.restart_soon()
                return JSONResponse(res, status_code=200 if res.get("ok") else 400)
            if action == "switch":
                model_id = str(body.get("model") or "")
                if not model_id:
                    return JSONResponse({"error": "model is required"}, status_code=400)
                started = daemon.setup.start(model=model_id, accept_license=bool(body.get("accept_license")))
                return JSONResponse({"ok": started, "started": started}, status_code=200 if started else 409)
            if action == "remove_weights":
                folder_id = str(body.get("id") or "")
                root = runtimes.models_dir(cfg).resolve()
                target = (root / folder_id).resolve()
                if not folder_id or "/" in folder_id or ".." in folder_id or root not in target.parents or not target.exists():
                    return JSONResponse({"error": "unknown model folder"}, status_code=400)
                shutil.rmtree(target, ignore_errors=True)
                svc.audit.write("models.removed_folder", folder=folder_id)
                return JSONResponse({"ok": True, "removed": folder_id, "weights": _installed_weights(cfg)})
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)

        hw = manager._hardware(cfg, None)
        mem = hardware.live_memory(hw.ram_gb)
        probed = probe.load(hw)
        planned = await asyncio.to_thread(manager.plan, cfg, hw=hw)
        candidates = [s.to_dict() for s in eval_pkg.default_candidates(cfg, hw=hw)]
        return JSONResponse({
            "hardware": hw.to_dict(),
            "live_memory": mem,
            "probe": probed,
            "plan": planned,
            "active_stack": eval_pkg.stack_from_config(cfg, hw).to_dict(),
            "candidates": candidates,
            "weights": _installed_weights(cfg),
            "models_dir": str(runtimes.models_dir(cfg)),
            "models_dir_default": not cfg.get("models.dir"),
        })

    async def bench_api(request: Request):
        from .. import eval as eval_pkg
        from ..models import manager, probe

        cfg = svc.config
        if request.method == "POST":
            body = await request.json()
            kind = str(body.get("action") or body.get("kind") or "probe")
            if not bench_job.start(kind, body):
                return JSONResponse({"error": "a benchmark is already running", **bench_job.snapshot()}, status_code=409)
            return JSONResponse(bench_job.snapshot())

        hw = manager._hardware(cfg, None)
        history = eval_pkg.load_history(hw=hw)
        return JSONResponse({
            **bench_job.snapshot(),
            "probe": probe.load(hw),
            "history": history,
            "summary": eval_pkg.plain_summary([r for r in history if r.get("ok")]),
            "candidates": [s.to_dict() for s in eval_pkg.default_candidates(cfg, hw=hw)],
        })

    async def connectors_api(request: Request):
        from ..connectors import macos as mac_conn
        cfg = svc.config
        enabled = list(cfg.get("connectors.enabled") or [])
        send_enabled = list(cfg.get("connectors.send") or [])
        if request.method == "POST":
            body = await request.json()
            action = str(body.get("action") or "")
            name = str(body.get("name") or "")
            if name not in mac_conn.CONNECTORS:
                return JSONResponse({"error": f"unknown connector {name!r}"}, status_code=400)
            if action == "toggle":
                on = bool(body.get("enabled"))
                if on and name not in enabled:
                    enabled.append(name)
                elif not on and name in enabled:
                    enabled.remove(name)
                    if name in send_enabled:
                        send_enabled.remove(name)
                cfg.set("connectors.enabled", enabled)
                cfg.set("connectors.send", send_enabled)
                cfg.save()
                return JSONResponse({"ok": True, "enabled": enabled, "send": send_enabled, "restart_required": True})
            if action == "send":
                on = bool(body.get("enabled"))
                if on and name not in send_enabled:
                    if name not in enabled:
                        enabled.append(name)
                    send_enabled.append(name)
                elif not on and name in send_enabled:
                    send_enabled.remove(name)
                cfg.set("connectors.enabled", enabled)
                cfg.set("connectors.send", send_enabled)
                cfg.save()
                return JSONResponse({"ok": True, "enabled": enabled, "send": send_enabled, "restart_required": True})
            if action == "test":
                try:
                    detail = await asyncio.to_thread(mac_conn.check, name)
                    return JSONResponse({"ok": True, "name": name, "detail": detail})
                except Exception as exc:
                    return JSONResponse({"ok": False, "name": name, "error": str(exc)}, status_code=200)
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)

        rows = []
        descriptions = {
            "calendar": "Read upcoming events and create calendar entries (no invitations sent).",
            "reminders": "Read and create items in Apple Reminders.",
            "notes": "Search, read and create Apple Notes.",
            "mail": "Read recent messages and save drafts (sending stays off unless enabled below).",
            "contacts": "Look up phone numbers and email addresses.",
            "settings": "Read or change appearance (dark/light), volume and mute only.",
            "phone": "Hand a phone number to FaceTime for you to confirm and dial.",
        }
        for name in mac_conn.CONNECTORS:
            rows.append({
                "name": name,
                "label": "Mac settings" if name == "settings" else name.capitalize(),
                "description": descriptions.get(name, ""),
                "enabled": name in enabled,
                "supports_send": name in getattr(mac_conn, "SENDERS", ("mail",)),
                "send": name in send_enabled,
            })
        return JSONResponse({"connectors": rows, "supported": sys.platform == "darwin"})

    async def browser_api(request: Request):
        from ..browser import extension as ext
        cfg = svc.config
        if request.method == "POST":
            body = await request.json()
            action = str(body.get("action") or "")
            if action == "mode":
                mode = str(body.get("mode") or "")
                if mode not in ("computer", "chrome", "dedicated"):
                    return JSONResponse({"error": "mode must be computer, chrome, or dedicated"}, status_code=400)
                cfg.set("browser.mode", mode)
                cfg.save()
                return JSONResponse({"ok": True, "mode": mode, "restart_required": True})
            if action == "connect_chrome":
                try:
                    info = await asyncio.to_thread(ext.install, browser=body.get("browser"))
                    cfg.set("browser.mode", "chrome")
                    cfg.save()
                    return JSONResponse({"ok": True, "installed": info, "extension_dir": str(ext.extension_dir()),
                                         "restart_required": True})
                except Exception as exc:
                    return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            if action == "rules":
                for key in ("allow", "block", "sensitive_hosts"):
                    if key in body and isinstance(body[key], list):
                        cfg.set(f"browser.{key}", [str(h).strip() for h in body[key] if str(h).strip()])
                cfg.save()
                return JSONResponse({"ok": True, "restart_required": True})
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)

        ext_status = {}
        with contextlib.suppress(Exception):
            ext_status = {"extension_dir": str(ext.extension_dir()),
                          "browsers": ext.installed() if hasattr(ext, "installed") else []}
        return JSONResponse({
            "mode": cfg.get("browser.mode") or "dedicated",
            "allow": cfg.get("browser.allow") or [],
            "block": cfg.get("browser.block") or [],
            "sensitive_hosts": cfg.get("browser.sensitive_hosts") or [],
            "extension": ext_status,
        })

    async def update_schedule(request: Request):
        sid = int(request.path_params["id"])
        body = await request.json()
        if "enabled" in body and hasattr(svc.scheduler, "set_enabled"):
            ok = svc.scheduler.set_enabled(sid, bool(body["enabled"]))
            return JSONResponse({"ok": bool(ok)})
        if hasattr(svc.scheduler, "update"):
            ok = svc.scheduler.update(sid, name=body.get("name"), spec=body.get("spec"), prompt=body.get("prompt"))
            return JSONResponse({"ok": bool(ok)})
        return JSONResponse({"error": "schedule update not supported"}, status_code=400)

    async def run_schedule_now(request: Request):
        sid = int(request.path_params["id"])
        row = next((s for s in svc.scheduler.list() if int(s["id"]) == sid), None)
        if row is None:
            return JSONResponse({"error": "no such routine"}, status_code=404)
        if daemon is not None and hasattr(daemon, "tasks"):
            task = daemon.tasks.start(f"app:routine-{sid}", row["prompt"], origin="owner",
                                      routine={"id": sid, "name": row["name"], "spec": row["spec"]})
            return JSONResponse({"ok": True, "task": task})
        return JSONResponse({"error": "task runner unavailable"}, status_code=503)

    async def voice_api(request: Request):
        from ..voice import Voice
        cfg = svc.config
        voice = Voice(cfg)
        if request.method == "POST":
            body = await request.json()
            action = str(body.get("action") or "")
            if action == "config":
                if "stt" in body:
                    cfg.set("voice.stt", str(body["stt"]))
                if "language" in body:
                    cfg.set("voice.language", str(body["language"]))
                cfg.save()
                return JSONResponse({"ok": True, "status": voice.status()})
            if action == "setup":
                try:
                    lines: list[str] = []
                    res = await voice.setup(progress=lines.append)
                    return JSONResponse({"ok": True, "result": res, "lines": lines[-20:], "status": voice.status()})
                except Exception as exc:
                    return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        return JSONResponse({
            "stt": cfg.get("voice.stt") or "auto",
            "language": cfg.get("voice.language") or "",
            "status": voice.status(),
        })

    async def tailscale_api(request: Request):
        from . import remote as ts_remote
        cfg = svc.config
        if request.method == "POST":
            body = await request.json()
            action = str(body.get("action") or "")
            try:
                if action == "enable":
                    owner = str(body.get("owner") or "").strip()
                    res = await asyncio.to_thread(ts_remote.enable, cfg, owner=owner or None)
                    return JSONResponse({"ok": True, **res})
                if action == "disable":
                    res = await asyncio.to_thread(ts_remote.disable, cfg)
                    return JSONResponse({"ok": True, **(res if isinstance(res, dict) else {})})
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        st = {}
        with contextlib.suppress(Exception):
            st = await asyncio.to_thread(ts_remote.status, cfg)
        return JSONResponse({"status": st, "owners": cfg.get("remote.owners") or []})

    async def channels_api(request: Request):
        from ..safety.secrets_store import SecretStore
        cfg = svc.config
        if request.method == "POST":
            body = await request.json()
            name = str(body.get("channel") or "")
            if name != "telegram":
                return JSONResponse({"error": "only Telegram can be configured from Settings right now"}, status_code=400)
            action = str(body.get("action") or "")
            if action == "configure":
                token_val = str(body.get("token") or "").strip()
                if token_val:
                    SecretStore().set("telegram_bot_token", token_val, hosts=["api.telegram.org"])
                ch = dict(cfg.get("channels.telegram") or {})
                ch["enabled"] = bool(body.get("enabled", True))
                ch["voice_replies"] = bool(body.get("voice_replies", False))
                cfg.set("channels.telegram", ch)
                cfg.save()
                return JSONResponse({"ok": True, "restart_required": True})
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        ch = cfg.get("channels") or {}
        tg = ch.get("telegram") or {}
        return JSONResponse({
            "telegram": {"enabled": bool(tg.get("enabled")), "voice_replies": bool(tg.get("voice_replies")),
                         "has_token": bool(SecretStore().get("telegram_bot_token"))},
            "imessage": {"enabled": bool((ch.get("imessage") or {}).get("enabled")), "binary": bool(shutil.which("imsg"))},
            "signal": {"enabled": bool((ch.get("signal") or {}).get("enabled"))},
            "whatsapp": {"enabled": bool((ch.get("whatsapp") or {}).get("enabled")), "binary": bool(shutil.which("wacli"))},
        })

    async def memories_api(request: Request):
        q = str(request.query_params.get("q") or "").strip().lower()
        rows = svc.store.query("SELECT id, text, tags, source, created FROM memories ORDER BY id DESC LIMIT 200")
        items = [{"id": r["id"], "content": r["text"], "text": r["text"], "tags": r["tags"],
                  "source": r["source"], "created": r["created"]}
                 for r in rows if not q or q in str(r["text"]).lower()]
        return JSONResponse({"memories": items})

    async def delete_memory(request: Request):
        mid = int(request.path_params["id"])
        svc.memory.forget(id=mid)
        return JSONResponse({"ok": True, "deleted": mid})

    async def wrap_task(request: Request):
        return JSONResponse({"wrapping": daemon.tasks.wrap(request.path_params["id"])})

    async def mutate_conversation(request: Request):
        name = request.path_params["conversation"]
        if request.method == "DELETE":
            ok = daemon.tasks.delete_conversation(name)
            return JSONResponse({"ok": ok}, status_code=200 if ok else 409)
        body = await request.json()
        ok = daemon.tasks.rename_conversation(name, str(body.get("title") or ""))
        return JSONResponse({"ok": ok}, status_code=200 if ok else 400)

    return [
        Route("/api/settings", settings_view, methods=["GET", "PATCH"]),
        Route("/api/restart", restart_daemon, methods=["POST"]),
        Route("/api/service", login_service_toggle, methods=["POST"]),
        Route("/api/models", models_hub, methods=["GET", "POST"]),
        Route("/api/bench", bench_api, methods=["GET", "POST"]),
        Route("/api/connectors", connectors_api, methods=["GET", "POST"]),
        Route("/api/browser", browser_api, methods=["GET", "POST"]),
        Route("/api/schedules/{id:int}/run", run_schedule_now, methods=["POST"]),
        Route("/api/voice/settings", voice_api, methods=["GET", "POST"]),
        Route("/api/tailscale", tailscale_api, methods=["GET", "POST"]),
        Route("/api/channels", channels_api, methods=["GET", "POST"]),
        Route("/api/memories", memories_api, methods=["GET"]),
        Route("/api/memories/{id:int}", delete_memory, methods=["DELETE"]),
        Route("/api/tasks/{id}/wrap", wrap_task, methods=["POST"]),
    ]
