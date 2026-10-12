"""End to end: the real extension in a real browser, driven through the broker.

Branded Chrome no longer loads extensions from the command line, so this needs Chromium or Chrome for
Testing: CODERUNNER_CHROME_E2E=/path/to/chrome. It runs in a temporary profile, never the owner's.
"""

import asyncio
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

from coderunner.core.audit import AuditLog
from coderunner.browser import BrowserBroker, extension, launcher
from coderunner.browser.cdp import CDPError
from coderunner.core.config import Config
from coderunner.safety.policy import Policy
from coderunner.core.store import Store
from test_browser import StubJudge

CHROME = os.environ.get("CODERUNNER_CHROME_E2E", "")

PAGES = {
    "/": """<!doctype html><title>Group shop</title><h1>Group shop</h1>
<input aria-label="Query" id="q"><button onclick="document.getElementById('out').textContent =
'echo ' + document.getElementById('q').value">Echo</button><p id="out">nothing yet</p>
<button onclick="document.getElementById('out').textContent = confirm('Sure?') ? 'agreed' : 'declined'">Ask</button>
<a href="/popup" target="_blank">Open offers</a> <a href="http://10.255.255.1/admin">Router</a>""",
    "/framed": """<!doctype html><title>Framed</title><h1>Framed</h1>
<iframe src="http://localhost:PORT/inner"></iframe>""",
    "/inner": "<!doctype html><title>Inner</title><p>inner page</p>",
    "/popup": "<!doctype html><title>Offers</title><h1>Offers</h1>",
    "/outside": "<!doctype html><title>Owner's own tab</title><h1>Private</h1>",
}


class _Pages(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGES.get(self.path.split("?")[0], "").replace("PORT", str(self.server.server_port)).encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@unittest.skipUnless(CHROME and os.path.exists(CHROME), "set CODERUNNER_CHROME_E2E to Chromium or Chrome for Testing")
class ExtensionEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp / "h")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Pages)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

        profile = self.tmp / "profile"
        # Chromium looks for user-level native hosts in the profile folder (NativeMessagingHosts).
        installed = extension.install(["test"], roots={"test": profile})
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.bridge = extension.ChromeBridge(audit=self.audit)
        await self.bridge.start()
        args = [f"--user-data-dir={profile}", "--remote-debugging-pipe", "--no-first-run",
                "--no-default-browser-check", f"--load-extension={installed['extension_dir']}",
                "--headless=new", "--window-size=1280,900", self.base + "/outside"]
        # The test keeps a protocol pipe of its own, to check what the extension does from outside.
        with mock.patch.object(launcher, "browser_args", return_value=args):
            self.process, self.cdp = await launcher.launch(CHROME, profile, True)

        self.config = Config({
            "browser": {"mode": "chrome", "allow": ["127.0.0.1"]},
            "policy": {"rules": [{"kind": "browser.interact", "target": "127.0.0.1", "decision": "allow"}]},
        }, path=self.tmp / "config.json")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.asked = []
        self.policy.add_listener(lambda a: (self.asked.append(a.summary), self.policy.decide(a.id, allow=False)))
        self.broker = BrowserBroker(self.config, self.policy, self.audit, decider=StubJudge(), chrome=self.bridge)
        self.assertIsNotNone(await self.bridge.wait(30), "the extension did not connect")

    async def asyncTearDown(self):
        await self.broker.stop()
        await self.bridge.stop()
        self.cdp.close()
        await asyncio.to_thread(launcher.terminate, self.process)
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def in_extension(self, expression):
        """Evaluate in the extension's service worker (through the test's own pipe)."""
        infos = (await self.cdp.send("Target.getTargets"))["targetInfos"]
        worker = next(t for t in infos if t["type"] == "service_worker" and extension.EXTENSION_ID in t["url"])
        sid = (await self.cdp.send("Target.attachToTarget", {"targetId": worker["targetId"], "flatten": True}))["sessionId"]
        result = await self.cdp.send("Runtime.evaluate", {"expression": expression, "awaitPromise": True,
                                                          "returnByValue": True}, sid)
        return result["result"].get("value")

    async def group(self):
        return await self.in_extension("""(async () => {
            const groups = await chrome.tabGroups.query({title: "CodeRunner"});
            if (!groups.length) return null;
            const tabs = await chrome.tabs.query({groupId: groups[0].id});
            return tabs.map((t) => ({id: t.id, url: t.url}));
        })()""")

    def refs(self):
        return {info["name"]: ref for ref, info in self.broker._refs[self.broker._active].items()}

    async def test_the_assistant_works_in_its_group_only(self):
        opened = await self.broker.navigate(self.base + "/")
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(opened["title"], "Group shop")
        self.assertEqual([t["url"] for t in await self.group()], [self.base + "/"])

        snap = await self.broker.snapshot()
        self.assertIn("# Group shop", snap["content"])
        refs = self.refs()
        self.assertTrue((await self.broker.type(refs["Query"], "rye loaf"))["ok"])
        clicked = await self.broker.click(refs["Echo"])
        self.assertTrue(clicked["ok"], clicked)
        self.assertIn("echo rye loaf", clicked["changes"])      # what changed comes back with the click
        asked = await self.broker.click(refs["Ask"])             # page dialogs are answered, not left hanging
        self.assertEqual(asked["dialog"]["type"], "confirm")
        self.assertIn("declined", asked["changes"])
        shot = await self.broker.screenshot()
        self.assertTrue(shot["ok"] and shot["data_b64"], shot.get("error"))

        # The document-request guard runs in the group's tabs: a link to a private address is stopped.
        router = await self.broker.click(refs["Router"])
        self.assertIn("10.255.255.1", router.get("blocked", "") + router.get("error", ""))
        self.assertTrue(any(e["event"] == "browser.blocked" and "10.255.255.1" in e["url"] for e in self.audit.tail()))
        after = await self.broker.snapshot()
        self.assertNotIn("10.255.255.1", after.get("url", ""), after)
        # ...and in frames from other sites, which run in processes of their own (localhost is blocked).
        framed = await self.broker.navigate(self.base + "/framed")
        self.assertTrue(framed["ok"], framed)
        for _ in range(30):
            if any(e["event"] == "browser.blocked" and "localhost" in e["url"] for e in self.audit.tail()):
                break
            await asyncio.sleep(0.1)
        self.assertTrue(any(e["event"] == "browser.blocked" and "/inner" in e["url"] for e in self.audit.tail()))
        self.assertTrue((await self.broker.navigate(self.base + "/"))["ok"])

        # A popup the page opens lands in the group, and the broker sees it as a new tab.
        popup = await self.broker.click(self.refs()["Open offers"])
        self.assertTrue(popup["ok"], popup)
        for _ in range(50):
            if len(await self.group()) == 2:
                break
            await asyncio.sleep(0.1)
        self.assertEqual(sorted(t["url"] for t in await self.group()), [self.base + "/", self.base + "/popup"])
        self.assertEqual(len((await self.broker.tabs())["tabs"]), 2)

        # The owner's own tab is invisible and out of reach.
        own = (await self.cdp.send("Target.getTargets"))["targetInfos"]
        outside = next(t["targetId"] for t in own if t["type"] == "page" and t["url"].endswith("/outside"))
        listed = [t["url"] for t in (await self.broker.tabs())["tabs"]]
        self.assertFalse(any(url.endswith("/outside") for url in listed))
        with self.assertRaises(CDPError):
            await self.broker._cdp.send("Target.attachToTarget", {"targetId": outside, "flatten": True})
        with self.assertRaises(CDPError):
            await self.broker._cdp.send("Target.getTargetInfo", {"targetId": outside})
        outside_tab = await self.in_extension(
            f'(async () => (await chrome.tabs.query({{url: "{self.base}/outside"}}))[0].id)()')
        with self.assertRaises(CDPError):
            await self.broker._cdp.send("Runtime.evaluate", {"expression": "document.title"}, f"tab:{outside_tab}")
        with self.assertRaises(CDPError):            # nothing outside the page domains, even in its own tabs
            await self.broker._cdp.send("Storage.getCookies", {}, self.broker._sessions[self.broker._active])

        # Dragging a tab out of the group takes it away from the assistant.
        first = (await self.group())[0]["id"]
        await self.in_extension(f"chrome.tabs.ungroup({first})")
        for _ in range(50):
            if len((await self.broker.tabs())["tabs"]) == 1:
                break
            await asyncio.sleep(0.1)
        self.assertEqual([t["url"] for t in (await self.broker.tabs())["tabs"]], [self.base + "/popup"])
        self.assertEqual(self.asked, [])

        # Tabs left idle are let go (Chrome's "debugging" bar goes away); the next step attaches again.
        self.assertTrue((await self.broker.snapshot())["ok"])
        self.assertTrue(self.broker._sessions)
        await self.in_extension("for (const tabId of used.keys()) used.set(tabId, 0)")
        for _ in range(80):
            if not self.broker._sessions:
                break
            await asyncio.sleep(0.5)
        self.assertEqual(self.broker._sessions, {})
        self.assertIn("# Offers", (await self.broker.snapshot())["content"])

        # Stopping lets go of the tabs and leaves the browser and its tabs open.
        await self.broker.stop()
        self.assertTrue(self.bridge.connected())
        pages = [t["url"] for t in (await self.cdp.send("Target.getTargets"))["targetInfos"] if t["type"] == "page"]
        self.assertIn(self.base + "/popup", pages)
        print(json.dumps({"pages_after_stop": pages}))


class _Yes:
    async def commits(self, state):
        return False


@unittest.skipUnless(CHROME and os.path.exists(CHROME), "set CODERUNNER_CHROME_E2E to Chromium or Chrome for Testing")
@unittest.skipUnless(os.environ.get("CODERUNNER_E2E_HEADFUL") == "1", "opens a browser window: CODERUNNER_E2E_HEADFUL=1")
class EverydayBrowserEndToEnd(unittest.IsolatedAsyncioTestCase):
    """As the owner runs it: a normal window, no debugging switches, only the extension."""

    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp / "h")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Pages)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        profile = self.tmp / "profile"
        installed = extension.install(["test"], roots={"test": profile})
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.bridge = extension.ChromeBridge(audit=self.audit)
        await self.bridge.start()
        self.process = subprocess.Popen(
            [CHROME, f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
             f"--load-extension={installed['extension_dir']}", "--window-size=1280,900", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        config = Config({"browser": {"mode": "chrome", "allow": ["127.0.0.1"]}}, path=self.tmp / "config.json")
        self.policy = Policy(config, Store(":memory:"), self.audit)
        self.asked = []
        self.policy.add_listener(lambda a: (self.asked.append(a.summary), self.policy.decide(a.id, allow=False)))
        self.broker = BrowserBroker(config, self.policy, self.audit, decider=_Yes(), chrome=self.bridge)
        self.assertIsNotNone(await self.bridge.wait(30), "the extension did not connect")

    async def asyncTearDown(self):
        await self.broker.stop()
        await self.bridge.stop()
        await asyncio.to_thread(launcher.terminate, self.process)
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def evaluate(self, expression):
        sid = self.broker._sessions[self.broker._active]
        return (await self.broker._cdp.send("Runtime.evaluate", {"expression": expression,
                                                                 "returnByValue": True}, sid))["result"].get("value")

    async def test_pages_do_not_see_an_automated_browser(self):
        self.assertTrue((await self.broker.navigate(self.base + "/"))["ok"])
        self.assertIs(await self.evaluate("navigator.webdriver"), False)

    @unittest.skipUnless(os.environ.get("CODERUNNER_E2E_LIVE") == "1", "uses booking.com: CODERUNNER_E2E_LIVE=1")
    async def test_booking_destination_suggestions(self):
        # With the dedicated profile (--remote-debugging-pipe) navigator.webdriver is true and booking.com's
        # firewall answers the destination autocomplete with 403; in the owner's browser it works.
        opened = await self.broker.navigate("https://www.booking.com/flights/index.html")
        self.assertTrue(opened["ok"], opened)
        self.assertIs(await self.evaluate("navigator.webdriver"), False)
        going = None
        for _ in range(20):                                   # the search form is drawn by script
            page = (await self.broker.snapshot())["content"]
            going = next((line for line in page.split("\n") if '] button "Going to"' in line), None)
            if going:
                break
            await asyncio.sleep(1)
        self.assertIsNotNone(going, "no Going to button")
        clicked = await self.broker.click(int(going.split("]")[0].strip("[")))
        field = next(line for line in clicked["changes"].split("\n") if "combobox" in line and "Arrival" in line)
        typed = await self.broker.type(int(field.split("[")[1].split("]")[0]), "Delhi")
        self.assertTrue(typed["ok"], typed)
        options = []
        for _ in range(20):                                   # suggestions arrive from the site's API
            page = (await self.broker.snapshot())["content"]
            options = [line for line in page.split("\n") if '] option "' in line and "Delhi" in line]
            if options:
                break
            await asyncio.sleep(0.5)
        self.assertTrue(options, "no destination suggestions")
        self.assertNotIn("Something’s not right", page)
        self.assertNotIn("Something's not right", page)
        self.assertEqual(self.asked, [])
        print(json.dumps({"suggestions": options[:3]}))


if __name__ == "__main__":
    unittest.main()
