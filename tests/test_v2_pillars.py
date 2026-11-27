"""Tests for v2 assistant pillars: QR pairing, owner profile, diff-aware routines, parallel reader,
ANE distiller, Mac browser login/session vault, quick actions, and partial-update auto-continue."""

from __future__ import annotations

import datetime
import unittest
from unittest import mock

import httpx

from coderunner.agent import profile, reader
from coderunner.agent.loop import Agent, RunContext
from coderunner.agent.scheduler import Scheduler
from coderunner.agent.tools import build_registry
from coderunner.browser import walls
from coderunner.core import qr
from coderunner.core.llm import LLM
from coderunner.core.store import Store
from coderunner.models import ane_llm
from coderunner.server.gateway import admin_app
from fakes import FakeOpenAI, Svc, tool_call
from test_agent import FakeBrowser


class QrAndProfileTests(unittest.TestCase):
    def test_qr_svg_generates_valid_matrix_and_svg(self):
        mat = qr.encode_matrix("https://my-mac.ts.net/pair?code=ABCD1234")
        self.assertEqual(len(mat), len(mat[0]))
        self.assertGreaterEqual(len(mat), 25)
        svg = qr.to_svg("https://my-mac.ts.net/pair?code=ABCD1234")
        self.assertTrue(svg.startswith("<svg "))
        self.assertIn("</svg>", svg)
        self.assertIn("<rect ", svg)

    def test_profile_save_render_and_secret_rejection(self):
        store = Store(":memory:")
        saved = profile.save(store, {"city": "Bengaluru", "home_airport": "BLR", "currency": "INR"})
        self.assertEqual(saved["city"], "Bengaluru")
        ctx = profile.render_context(store)
        self.assertIn("Home city: Bengaluru", ctx)
        self.assertIn("Home airport: BLR", ctx)

        with self.assertRaises(ValueError):
            profile.save(store, {"notes": "my key is sk-ant-api03-1234567890abcdef123456"})

    def test_login_wall_detection(self):
        login_html = "<html><head><title>Sign in - Example</title></head><body><h1>Sign in</h1><form><input type='email'><input type='password'></form></body></html>"
        self.assertTrue(walls.detect_login(walls.probe_from_html(login_html)))
        article_html = "<html><head><title>News</title></head><body>" + ("<p>Long article body text. </p>" * 120) + "</body></html>"
        self.assertFalse(walls.detect_login(walls.probe_from_html(article_html)))


class DiffAwareRoutineTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_change_suppresses_unchanged_delivery_and_notifies_on_diff(self):
        store = Store(":memory:")
        audit = Svc().audit
        replies = ["Sony WH-1000XM6 is €329 at 09:00", "Sony WH-1000XM6 is €329 at 10:00", "Sony WH-1000XM6 dropped to €289 at 11:00"]
        delivered = []

        async def run_task(row):
            return replies.pop(0)

        async def deliver(row, text):
            delivered.append(text)

        sched = Scheduler(store, audit, run_task, deliver, tz="UTC")
        base = datetime.datetime(2026, 11, 27, 8, 0, tzinfo=datetime.timezone.utc)
        row = sched.add("Headphone watch", "every 30m", "Check price", on_change=True, now=base)
        self.assertTrue(row["on_change"])

        # Run 1 -> delivered ("changed")
        await sched._run(sched.get(row["id"]), base.timestamp())
        self.assertEqual(len(delivered), 1)
        self.assertEqual(sched.get(row["id"])["last_status"], "changed")

        # Run 2 (same price, different clock) -> suppressed ("unchanged")
        await sched._run(sched.get(row["id"]), base.timestamp() + 1800)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(sched.get(row["id"])["last_status"], "unchanged")

        # Run 3 (price changed to €289) -> delivered!
        await sched._run(sched.get(row["id"]), base.timestamp() + 3600)
        self.assertEqual(len(delivered), 2)
        self.assertIn("€289", delivered[-1])


class ParallelReaderAndAneTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_read_many_extracts_blocks_private_and_flags_walls(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "wiki.example":
                body = (
                    "<html><head><title>Mount Everest</title></head><body>"
                    "<h1>Mount Everest</h1><p>Mount Everest is 8,848.86 metres high in Nepal and China. "
                    "First climbed in 1953 by Edmund Hillary and Tenzing Norgay. Additional geographic details "
                    "about the Mahalangur Himal sub-range of the Himalayas and climbing routes.</p>"
                    "</body></html>"
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/html"})
            if request.url.host == "wall.example":
                return httpx.Response(403, text="<html><head><title>Just a moment...</title></head><body>Checking if the site connection is secure</body></html>",
                                      headers={"content-type": "text/html"})
            return httpx.Response(404, text="Not found", headers={"content-type": "text/html"})

        transport = httpx.MockTransport(handler)
        results = await reader.read_many(
            ["https://wiki.example/everest", "https://wall.example/page", "http://127.0.0.1:8224/secret"],
            focus="height and first climbed year",
            transport=transport,
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["ok"])
        self.assertIn("8,848.86", results[0]["content"])
        self.assertFalse(results[1]["ok"])
        self.assertTrue(results[1]["needs_browser"])
        self.assertFalse(results[2]["ok"])
        self.assertIn("Private or local", results[2]["error"])

    async def test_ane_distiller_probe_and_candidates(self):
        info = ane_llm.probe_foundation()
        self.assertIn("available", info)
        self.assertGreaterEqual(len(ane_llm.ANE_DISTILLER_CANDIDATES), 3)


class AutoContinueAndApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_premature_giveup_posts_partial_update_and_continues(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.browser = FakeBrowser("A page")
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            fake.script = [
                tool_call("browser_open", {"url": "https://a.example/1"}),
                tool_call("browser_open", {"url": "https://a.example/2"}),
                tool_call("browser_open", {"url": "https://a.example/3"}),
                tool_call("browser_open", {"url": "https://a.example/4"}),
                {"role": "assistant", "content": "I couldn't finish Oakland because QuickFacts is down. San Jose is $3,325."},
                {"role": "assistant", "content": "Complete table: San Jose $3,325, Oakland $2,400, SF $3,100 ([a](https://a.example/1))."},
            ]
            emitted = []
            ctx = RunContext(conversation="cont-1", events=emitted.append)
            reply = await Agent(svc, svc.llm, svc.registry).run(ctx, "Find Bay Area rents")
            self.assertIn("Oakland $2,400", reply)
            self.assertTrue(any(e.get("type") == "interim" for e in emitted))
            self.assertTrue(any(e["event"] == "agent.auto_continue" for e in svc.audit.tail()))
            await svc.llm.aclose()

    async def test_profile_quick_actions_and_qr_pair_routes(self):
        import types
        svc = Svc()
        svc.config.set("gateway.admin_port", 8224)
        svc.admin_token = lambda: "tok-1234567890abcdef"
        svc.registry = types.SimpleNamespace(describe=lambda: [])
        daemon = types.SimpleNamespace(config=svc.config, svc=svc)
        app = admin_app(svc, daemon)
        headers = {"Authorization": "Bearer tok-1234567890abcdef", "X-CodeRunner": "1"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as c:
            rp = await c.patch("/api/profile", headers=headers, json={"city": "Bengaluru", "currency": "INR"})
            self.assertEqual(rp.status_code, 200)
            self.assertEqual(rp.json()["profile"]["city"], "Bengaluru")
            rg = await c.get("/api/profile", headers=headers)
            self.assertEqual(rg.json()["profile"]["currency"], "INR")

            with mock.patch("coderunner.connectors.macos.notes_create", return_value="Created note"):
                rn = await c.post("/api/actions/quick", headers=headers, json={"kind": "note", "title": "Trip", "body": "Day 1..."})
                self.assertEqual(rn.status_code, 200)
                self.assertTrue(rn.json()["ok"])

            with mock.patch("coderunner.server.remote.status", return_value={"serving": True, "url": "https://mac.ts.net/"}):
                rpair = await c.post("/api/pair", headers=headers, json={})
                self.assertEqual(rpair.status_code, 200)
                data = rpair.json()
                self.assertIn("<svg ", data["qr_svg"])
                self.assertTrue(data["url"].startswith("https://mac.ts.net/pair?code="))

            rane = await c.get("/api/ane/distiller", headers=headers)
            self.assertEqual(rane.status_code, 200)
            self.assertIn("candidates", rane.json())


class BrowserIdentityTests(unittest.IsolatedAsyncioTestCase):
    def test_chrome_identity_and_headers_match_running_version(self):
        import pathlib
        from coderunner.browser import identity, launcher
        import desktop

        full, major = identity.parse_chrome_version("Chrome/158.0.8127.88")
        self.assertEqual((full, major), ("158.0.8127.88", "158"))
        full_hl, major_hl = identity.parse_chrome_version("HeadlessChrome/142.0.7444.10")
        self.assertEqual((full_hl, major_hl), ("142.0.7444.10", "142"))

        ident = identity.chrome_identity("Chrome/158.0.8127.88")
        self.assertIn("Macintosh; Intel Mac OS X 10_15_7", ident["userAgent"])
        self.assertIn("Chrome/158.0.0.0", ident["userAgent"])
        self.assertNotIn("Linux", ident["userAgent"])
        self.assertNotIn("Headless", ident["userAgent"])
        self.assertEqual(ident["platform"], "MacIntel")
        meta = ident["userAgentMetadata"]
        self.assertEqual(meta["platform"], "macOS")
        self.assertEqual(meta["fullVersion"], "158.0.8127.88")
        self.assertFalse(meta["mobile"])
        self.assertTrue(any(b["brand"] == "Google Chrome" and b["version"] == "158" for b in meta["brands"]))

        hdrs = identity.chrome_headers("Chrome/158.0.8127.88")
        self.assertEqual(hdrs["Sec-CH-UA-Platform"], '"macOS"')
        self.assertEqual(hdrs["Sec-CH-UA-Mobile"], "?0")
        self.assertIn('"Google Chrome";v="158"', hdrs["Sec-CH-UA"])
        self.assertEqual(hdrs["Sec-Fetch-Mode"], "navigate")

        argv = desktop.chrome_argv(pathlib.Path("/home/box/.config/coderunner-chrome"))
        self.assertIn("--lang=en-US", argv)
        self.assertIn("--disable-blink-features=AutomationControlled", argv)
        self.assertIn("--disable-background-timer-throttling", argv)
        self.assertTrue(any(a.startswith("--user-agent=Mozilla/5.0 (Macintosh;") for a in argv))

        local_args = launcher.browser_args(pathlib.Path("/tmp/p"), headless=True)
        self.assertIn("--disable-blink-features=AutomationControlled", local_args)
        self.assertIn("--lang=en-US", local_args)
        self.assertIn("--disable-background-timer-throttling", local_args)

    async def test_broker_applies_chrome_identity_only_for_sandbox_and_headless(self):
        from test_browser import FakeLauncher
        from coderunner.browser.broker import BrowserBroker
        from coderunner.safety.policy import Policy

        class VersionedLauncher(FakeLauncher):
            async def launch(self, executable, profile_dir, headless):
                proc, cdp = await super().launch(executable, profile_dir, headless)
                orig = self.browser.handle

                def handle(method, params, sid):
                    if method == "Browser.getVersion":
                        return {"product": "Chrome/158.0.8127.88", "userAgent": "Mozilla/5.0 (X11; Linux aarch64) Chrome/158.0.0.0"}
                    return orig(method, params, sid)

                self.browser.handle = handle
                return proc, cdp

        def make_broker(mode: str, headless: bool = False):
            svc = Svc()
            svc.config.set("browser.mode", mode)
            svc.config.set("browser.headless", headless)
            launcher = VersionedLauncher()
            return BrowserBroker(svc.config, svc.policy, svc.audit, launcher=launcher, mode=mode), launcher

        # 1. Sandbox VM browser (mode="computer"): gets full Chrome identity + stealth init script
        b_vm, l_vm = make_broker("computer")
        await b_vm.start()
        await b_vm.tabs()
        sid = await b_vm._session("P1")
        self.assertTrue(sid)
        ua_calls = l_vm.browser.sent("Emulation.setUserAgentOverride")
        self.assertEqual(len(ua_calls), 1)
        self.assertIn("Chrome/158.0.0.0", ua_calls[0]["userAgent"])
        self.assertEqual(ua_calls[0]["platform"], "MacIntel")
        self.assertEqual(ua_calls[0]["userAgentMetadata"]["platform"], "macOS")
        self.assertEqual(len(l_vm.browser.sent("Emulation.setLocaleOverride")), 1)
        self.assertEqual(len(l_vm.browser.sent("Page.addScriptToEvaluateOnNewDocument")), 1)
        await b_vm.stop()

        # 2. Built-in Mac browser (mode="mac"): gets version-matched Chrome UA/Client Hints override
        #    (so navigator.userAgentData matches desktop Google Chrome) without VM-only stealth script injection.
        b_mac, l_mac = make_broker("mac")
        await b_mac.start()
        await b_mac.tabs()
        await b_mac._session("P1")
        mac_ua = l_mac.browser.sent("Emulation.setUserAgentOverride")
        self.assertEqual(len(mac_ua), 1)
        self.assertIn("Chrome/158.0.0.0", mac_ua[0]["userAgent"])
        self.assertNotIn("Electron", mac_ua[0]["userAgent"])
        self.assertEqual(l_mac.browser.sent("Emulation.setLocaleOverride"), [])
        self.assertEqual(l_mac.browser.sent("Page.addScriptToEvaluateOnNewDocument"), [])
        await b_mac.stop()

        # 3. Everyday Chrome via extension (mode="chrome"): stays completely untouched.
        b_ext, l_ext = make_broker("chrome")
        await b_ext.start()
        await b_ext.tabs()
        await b_ext._session("P1")
        self.assertEqual(l_ext.browser.sent("Emulation.setUserAgentOverride"), [])
        self.assertEqual(l_ext.browser.sent("Page.addScriptToEvaluateOnNewDocument"), [])
        await b_ext.stop()

    def test_startup_splash_hides_internal_address_and_keeper_detects_local_network_block(self):
        import pathlib
        import sys
        from coderunner.sandbox.keeper import FAILED, LOCAL_NETWORK_BLOCKED, Keeper

        root = pathlib.Path(__file__).resolve().parents[1]
        main_js = (root / "app" / "main.js").read_text()
        self.assertNotIn("Starting the assistant at ${ADDRESS}", main_js)
        self.assertIn('show("CodeRunner is starting", "")', main_js)

        browser_js = (root / "app" / "browser.js").read_text()
        self.assertIn('setUserAgent(identity.userAgent, "en-US,en")', browser_js)
        self.assertIn('"Sec-CH-UA"', browser_js)

        if sys.platform == "darwin":
            from coderunner.core.config import Config
            sb = mock.Mock(base_url="http://coderunner.local:8222", last_error=OSError(65, "No route to host"))
            k = Keeper(Config(), mock.Mock(), sb, mock.Mock())
            self.assertTrue(k._local_network_blocked())


if __name__ == "__main__":
    unittest.main()
