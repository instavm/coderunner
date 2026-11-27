"""Tests for owner-only Settings, Models, Benchmark Lab, Connectors, Browser, Routines, Memories & Wrap APIs."""

from __future__ import annotations

import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import httpx

from coderunner.agent.memory import Memory
from coderunner.agent.scheduler import Scheduler
from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.core.store import Store
from coderunner.safety.policy import Policy
from coderunner.server.gateway import admin_app
from coderunner.server.tasks import EventBus, Tasks


class _FakeSetup:
    def __init__(self):
        self.started = []

    def public(self):
        return {"configured": True, "state": "idle", "lines": [], "stages": [], "plan": None}

    def start(self, model=None, accept_license=False):
        self.started.append((model, accept_license))
        return True


class _FakeDaemon:
    def __init__(self, svc):
        self.svc = svc
        self.config = svc.config
        self.setup = _FakeSetup()
        self.restarts = 0
        self.bus = EventBus(svc.store)
        self.tasks = Tasks(self, self.bus)

    def restart_soon(self):
        self.restarts += 1

    def status(self):
        return {"ok": True, "models": {"chat": self.config.get("models.chat") or ""}}


class _FakeBrowser:
    async def focus_info(self, scope: str):
        return {"shown": scope == "app:live", "active_conversation": "app:live", "current_url": "https://example.com"}


class SettingsApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

        cfg = Config({"models": {"discover": False, "chat": "coderunner-qwen3.5-9b"}}, path=self.home / "config.json")
        store = Store(self.home / "state.db")
        audit = AuditLog(self.home / "audit.jsonl")
        self.token = "test-admin-token-1234567890"
        self.svc = types.SimpleNamespace(
            config=cfg,
            store=store,
            audit=audit,
            policy=Policy(cfg, store, audit),
            memory=Memory(store, audit),
            scheduler=Scheduler(store, audit, lambda *_: None, lambda *_: None),
            browser=_FakeBrowser(),
            admin_token=lambda: self.token,
            registry=types.SimpleNamespace(describe=lambda: []),
            files=None,
            turns=types.SimpleNamespace(focus=""),
            remote=None,
            llm=None,
            vision=True,
        )
        self.daemon = _FakeDaemon(self.svc)
        self.app = admin_app(self.svc, self.daemon)
        self.headers = {"Authorization": f"Bearer {self.token}", "X-CodeRunner": "1"}

    def test_an_infinite_step_time_is_sent_as_null(self):
        import json
        from coderunner.server.settings_api import json_safe
        plan = {"rejected": [{"id": "big", "step_s": float("inf")}, {"id": "ok", "step_s": 12.5}], "nan": float("nan")}
        self.assertEqual(json.loads(json.dumps(json_safe(plan), allow_nan=False)),
                         {"rejected": [{"id": "big", "step_s": None}, {"id": "ok", "step_s": 12.5}], "nan": None})

    async def test_settings_allowlist_and_restart_flag(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1") as c:
            # Unauthorized fails closed
            r0 = await c.get("/api/settings")
            self.assertEqual(r0.status_code, 401)

            r1 = await c.get("/api/settings", headers=self.headers)
            self.assertEqual(r1.status_code, 200)
            self.assertIn("settings", r1.json())

            # Live setting does not require restart; unknown/dangerous key is ignored
            r2 = await c.patch("/api/settings", headers=self.headers,
                               json={"owner.name": "Abhishek", "owner.onboarded": True, "gateway.admin_bind": "0.0.0.0"})
            self.assertEqual(r2.status_code, 200)
            body = r2.json()
            self.assertFalse(body["restart_required"])
            self.assertIn("owner.name", body["applied"])
            self.assertNotIn("gateway.admin_bind", body["applied"])
            self.assertEqual(self.svc.config.get("gateway.admin_bind"), "127.0.0.1")

            # Structural setting flags restart_required
            r3 = await c.patch("/api/settings", headers=self.headers, json={"sandbox.network": "proxy"})
            self.assertTrue(r3.json()["restart_required"])

            # Restart endpoint triggers daemon.restart_soon()
            r4 = await c.post("/api/restart", headers=self.headers, json={})
            self.assertEqual(r4.status_code, 200)
            self.assertEqual(self.daemon.restarts, 1)

    async def test_connectors_browser_memories_schedules_and_focus(self):
        self.svc.store.execute("INSERT INTO memories(text, tags, source, created) VALUES (?, ?, ?, ?)",
                               ("Prefers non-stop flights", "", "owner", 1.0))
        self.svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               ("t1", "app:old", "check bus", "done", "ok", 1.0, 2.0))
        sid = self.svc.scheduler.add("Morning brief", "weekdays 08:00", "Summarise my day")["id"]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1") as c:
            # Connectors
            rc = await c.get("/api/connectors", headers=self.headers)
            self.assertEqual(rc.status_code, 200)
            names = [x["name"] for x in rc.json()["connectors"]]
            self.assertIn("calendar", names)
            rtog = await c.post("/api/connectors", headers=self.headers, json={"action": "toggle", "name": "calendar", "enabled": True})
            self.assertTrue(rtog.json()["ok"])
            self.assertIn("calendar", self.svc.config.get("connectors.enabled"))

            # Browser mode
            rb = await c.post("/api/browser", headers=self.headers, json={"action": "mode", "mode": "computer"})
            self.assertEqual(rb.status_code, 200)
            self.assertEqual(self.svc.config.get("browser.mode"), "computer")

            # Schedule pause/resume
            rs = await c.patch(f"/api/schedules/{sid}", headers=self.headers, json={"enabled": False})
            self.assertTrue(rs.json()["ok"])

            # Memories list + forget
            rm = await c.get("/api/memories", headers=self.headers)
            self.assertEqual(len(rm.json()["memories"]), 1)
            mid = rm.json()["memories"][0]["id"]
            rdel = await c.delete(f"/api/memories/{mid}", headers=self.headers)
            self.assertTrue(rdel.json()["ok"])
            rm2 = await c.get("/api/memories", headers=self.headers)
            self.assertEqual(len(rm2.json()["memories"]), 0)

            # Focus returns ownership metadata
            rf = await c.post("/api/focus", headers=self.headers, json={"conversation": "app:old"})
            self.assertEqual(rf.status_code, 200)
            self.assertFalse(rf.json()["shown"])
            self.assertEqual(rf.json()["current_url"], "https://example.com")

            # Rename + delete conversation
            rren = await c.patch("/api/conversations/app:old", headers=self.headers, json={"title": "Bus tickets"})
            self.assertTrue(rren.json()["ok"])
            rrm = await c.delete("/api/conversations/app:old", headers=self.headers)
            self.assertTrue(rrm.json()["ok"])

            # Bench GET returns persistent history and candidates
            rbench = await c.get("/api/bench", headers=self.headers)
            self.assertEqual(rbench.status_code, 200)
            self.assertIn("candidates", rbench.json())
            self.assertIn("history", rbench.json())

            # Connector test awaits async macos.check (must return JSON string, not coroutine error)
            from unittest import mock
            async def fake_check(name):
                return f"{name} ok"
            with mock.patch("coderunner.connectors.macos.check", side_effect=fake_check):
                rtest = await c.post("/api/connectors", headers=self.headers, json={"action": "test", "name": "calendar"})
                self.assertEqual(rtest.status_code, 200)
                self.assertEqual(rtest.json(), {"ok": True, "name": "calendar", "detail": "calendar ok"})

            # Tailscale status/enable/disable pass integer admin_port and align remote.provider
            seen_ports = []
            def fake_ts_status(port):
                seen_ports.append(port)
                return {"installed": False, "running": False, "serving": False}
            with mock.patch("coderunner.server.remote.status", side_effect=fake_ts_status):
                rts = await c.get("/api/tailscale", headers=self.headers)
                self.assertEqual(rts.status_code, 200)
                self.assertEqual(seen_ports, [int(self.svc.config.get("gateway.admin_port") or 8224)])
            with mock.patch("coderunner.server.remote.enable", return_value="https://mac.ts.net/"), \
                 mock.patch("coderunner.server.remote.disable", return_value="Remote access disabled."):
                ren = await c.post("/api/tailscale", headers=self.headers, json={"action": "enable", "owner": "me@example.com"})
                self.assertTrue(ren.json()["ok"])
                self.assertEqual(self.svc.config.get("remote.provider"), "tailscale")
                self.assertIn("me@example.com", self.svc.config.get("remote.owners"))
                rdis = await c.post("/api/tailscale", headers=self.headers, json={"action": "disable"})
                self.assertTrue(rdis.json()["ok"])
                self.assertEqual(self.svc.config.get("remote.provider"), "")


if __name__ == "__main__":
    unittest.main()
