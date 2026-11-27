"""End to end: the app's browser (app/browser.js) in a real Electron, driven through the broker.

Electron runs a test main (tests/js/app_browser_host.js) whose one window is never shown: nothing appears on the
screen, the Dock or the menu bar. It needs the app's Electron: CODERUNNER_APP_E2E=/path/to/Electron (the binary in
app/node_modules/electron/dist). Local pages only; a temporary CODERUNNER_HOME; no real cookies or Keychain.
CODERUNNER_APP_SHOTS=folder keeps the screenshots.
"""

import asyncio
import base64
import contextlib
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from coderunner.browser import BrowserBroker, broker as broker_module
from coderunner.browser.app import AppBridge, AppLauncher
from coderunner.browser.switch import BrowserSwitch
from coderunner.agent.loop import RunContext
from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.core.store import Store
from coderunner.safety.policy import Policy, current_run
from test_browser import StubJudge

ELECTRON = os.environ.get("CODERUNNER_APP_E2E", "")
ROOT = pathlib.Path(__file__).resolve().parents[1]

CHALLENGE = """<!doctype html><html><head><title>Just a moment...</title>
<script>window._cf_chl_opt={cvId:'3',cType:'managed'};</script></head>
<body><div class="main-wrapper"><h1>127.0.0.1</h1><p>Verifying you are human. This may take a few seconds.</p>
<form id="challenge-form" action="/challenge?__cf_chl_f_tk=x" method="POST"></form></div></body></html>"""
PAGES = {
    "/": (200, """<!doctype html><title>Corner shop</title><h1>Corner shop</h1>
<p>Our sign-up form uses a captcha to keep bots out. Are you a robot? We hope not.</p>
<input aria-label="Query" id="q"><button onclick="document.getElementById('out').textContent =
'echo ' + document.getElementById('q').value">Echo</button><p id="out">nothing yet</p>
<a href="/popup" target="_blank">Open offers</a>"""),
    "/challenge": (403, CHALLENGE),
    "/popup": (200, "<!doctype html><title>Offers</title><h1>Offers</h1>"),
}


class _Pages(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/cookie":
            status, body = 200, f"<!doctype html><title>Cookie</title><p>cookie: {self.headers.get('Cookie', 'none')}</p>"
        else:
            status, body = PAGES.get(path, (404, ""))
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


@unittest.skipUnless(ELECTRON and os.path.exists(ELECTRON), "set CODERUNNER_APP_E2E to the app's Electron binary")
class AppBrowserEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
        self.shots = pathlib.Path(os.environ.get("CODERUNNER_APP_SHOTS") or self.tmp / "shots")
        self.shots.mkdir(parents=True, exist_ok=True)
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp / "h")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Pages)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.bridge = AppBridge(audit=self.audit)
        await self.bridge.start()
        from coderunner.core import paths
        env = {**os.environ, "CODERUNNER_APP_DIR": str(ROOT / "app"), "CODERUNNER_ASSISTANT": str(paths.assistant_dir()),
               "CODERUNNER_APP_DATA": str(self.tmp / "electron")}
        self.host = await asyncio.create_subprocess_exec(
            ELECTRON, str(ROOT / "tests" / "js" / "app_browser_host.js"), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.assertEqual(await self.answer(), {"ready": True})
        self.config = Config({
            "browser": {"mode": "computer", "allow": ["127.0.0.1"]},
            "policy": {"rules": [{"kind": "browser.interact", "target": "127.0.0.1", "decision": "allow"}]},
        }, path=self.tmp / "config.json")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.broker = BrowserBroker(self.config, self.policy, self.audit, decider=StubJudge(),
                                    launcher=AppLauncher(self.bridge, wait=30), mode="mac")
        self.assertIsNotNone(await self.bridge.wait(30), "the app's browser did not connect")

    async def asyncTearDown(self):
        await self.broker.stop()
        with contextlib.suppress(Exception):
            await self.tell("quit")
        try:
            await asyncio.wait_for(self.host.wait(), 10)
        except asyncio.TimeoutError:
            self.host.kill()
        await self.bridge.stop()
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def answer(self) -> dict:
        return json.loads(await asyncio.wait_for(self.host.stdout.readline(), 30))

    async def tell(self, line: str) -> dict:
        self.host.stdin.write((line + "\n").encode())
        await self.host.stdin.drain()
        return await self.answer()

    def refs(self):
        return {info["name"]: ref for ref, info in self.broker._refs[self.broker._active].items()}

    async def test_the_broker_drives_the_apps_browser_and_sees_a_wall_for_what_it_is(self):
        opened = await self.broker.open(self.base + "/")
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(opened["title"], "Corner shop")
        self.assertNotIn("wall", opened)                  # a page that talks about captchas is the site's own
        self.assertIn("Corner shop", opened["content"])
        self.assertTrue((await self.broker.type(self.refs()["Query"], "rye loaf"))["ok"])
        clicked = await self.broker.click(self.refs()["Echo"])
        self.assertIn("echo rye loaf", clicked["changes"])
        self.assertTrue(opened["tab"].startswith("m"))     # never taken for one of the sandbox's tabs
        self.assertTrue((await self.tell(f"shot {self.shots / 'app-browser-shop.png'}"))["ok"])
        shot = await self.broker.screenshot()
        self.assertTrue(shot["ok"] and shot["data_b64"], shot.get("error"))
        (self.shots / "app-browser-shop-cdp.jpg").write_bytes(base64.b64decode(shot["data_b64"]))

        # The view put away (the Browser tab not shown): the page is still read, and still pictured.
        await self.tell("hide")
        hidden = await asyncio.wait_for(self.broker.screenshot(), 20)
        self.assertTrue(hidden["ok"] and hidden["data_b64"], hidden.get("error"))
        await self.tell("show")

        with mock.patch.object(broker_module, "WALL_WAIT", 1.5):
            walled = await self.broker.navigate(self.base + "/challenge")
        self.assertEqual(walled.get("wall"), "Cloudflare", walled)
        self.assertTrue((await self.tell(f"shot {self.shots / 'app-browser-challenge.png'}"))["ok"])

        # A popup opens as a tab of the app's browser, with its opener.
        await self.broker.navigate(self.base + "/")
        popped = await self.broker.click(self.refs()["Open offers"])
        self.assertTrue(popped["ok"], popped)
        state = (await self.tell("state"))["state"]
        self.assertIn(self.base + "/popup", [tab["url"] for tab in state["tabs"]])

    async def test_sign_ins_brought_over_reach_the_site(self):
        done = await self.bridge.set_cookies([{"url": self.base + "/", "name": "sid", "value": "fixture-1"}])
        self.assertEqual(done, {"set": 1, "failed": 0})
        page = await self.broker.open(self.base + "/cookie")
        self.assertIn("cookie: sid=fixture-1", page["content"])

    async def act(self, **request) -> object:
        answer = await self.tell("act " + json.dumps(request))
        self.assertTrue(answer["ok"], answer)
        return answer["result"]

    async def test_the_menu_sends_the_address_out_and_clears_what_it_should(self):
        self.assertFalse(await self.act(action="copy"))          # no page yet: nothing leaves
        await self.broker.open(self.base + "/")
        await self.broker.navigate(self.base + "/popup")
        self.assertTrue(await self.act(action="copy"))
        self.assertTrue(await self.act(action="open-outside"))
        self.assertEqual((await self.tell("outside"))["calls"], [["copy", self.base + "/popup"], ["open", self.base + "/popup"]])
        self.assertTrue(str(await self.act(action="still")).startswith("data:image/jpeg;base64,"))

        # Keep cookies: "always" to begin with; "Until quit" is kept, and the next start begins without cookies.
        self.assertEqual((await self.tell("state"))["state"]["keep"], "always")
        self.assertEqual(await self.act(action="keep", url="quit"), "quit")
        self.assertEqual((await self.tell("state"))["state"]["keep"], "quit")
        await self.bridge.set_cookies([{"url": self.base + "/", "name": "sid", "value": "fixture-2"}])
        self.assertIn("sid=fixture-2", (await self.broker.navigate(self.base + "/cookie"))["content"])
        await self.tell("fresh-quit")
        self.assertIn("cookie: none", (await self.broker.navigate(self.base + "/cookie"))["content"])
        self.assertEqual(await self.act(action="keep", url="always"), "always")

        # Clear browsing data: the cookies and each tab's history go.
        await self.bridge.set_cookies([{"url": self.base + "/", "name": "sid", "value": "fixture-3"}])
        self.assertIn("sid=fixture-3", (await self.broker.navigate(self.base + "/cookie"))["content"])
        self.assertEqual(await self.act(action="clear"), {"cleared": True})
        self.assertFalse((await self.tell("state"))["state"]["canBack"])
        self.assertIn("cookie: none", (await self.broker.navigate(self.base + "/cookie"))["content"])

    async def test_a_blocked_site_continues_in_the_apps_browser_once_allowed(self):
        class Walled:                   # the sandbox's browser, turned away by the site
            config, origin = self.config, "agent"

            async def navigate(self, url, tab=None, tainted=False):
                return {"ok": True, "url": url, "title": "Just a moment...", "wall": "Cloudflare"}

        launcher = mock.Mock(available=lambda config: True, app_open=lambda: True)
        switch = BrowserSwitch(Walled(), self.broker, self.policy, self.audit, launcher=launcher, bridge=self.bridge,
                               config=self.config)
        asked = []
        self.policy.add_listener(lambda a: (asked.append(a.summary), self.policy.decide(a.id, True, "once")))
        token = current_run.set(RunContext(conversation="c-e2e"))
        try:
            moved = await switch.navigate(self.base + "/")
        finally:
            current_run.reset(token)
        self.assertEqual(asked, ["127.0.0.1 blocks the sandbox browser. Continue in CodeRunner's built-in browser on your Mac?"])
        self.assertTrue(moved["ok"], moved)
        self.assertEqual(moved["title"], "Corner shop")
        self.assertIn("browser on their Mac", moved["note"])
        self.assertEqual(switch.lane("c-e2e"), "mac")
        self.assertTrue((await self.tell(f"shot {self.shots / 'app-browser-switched.png'}"))["ok"])
