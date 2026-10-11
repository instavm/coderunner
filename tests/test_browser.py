import asyncio
import base64
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from coderunner.audit import AuditLog
from coderunner.browser import BrowserBroker, BrowserError
from coderunner.browser import launcher
from coderunner.browser.broker import CONTAINS_JS, DESCRIBE_JS, parse_url, render_ax, sensitive
from coderunner.browser.cdp import CDPError, CDPPipe
from coderunner.config import Config
from coderunner.policy import Policy, current_run
from coderunner.store import Store


# ---- fakes -----------------------------------------------------------------

class FakeBrowser:
    """Just enough of a Chromium CDP endpoint for broker tests. Sessions are "S-<targetId>"."""

    def __init__(self):
        self.calls = []
        self.pages = {"P1": {"url": "about:blank", "title": ""}}
        self.titles = {}
        self.ax = []
        self.nodes = {}       # backendNodeId -> DOM.describeNode node
        self.labels = {}      # backendNodeId -> DESCRIBE_JS result
        self.focused = None
        self.hit = None
        self.image = b"\xff\xd8 first"
        self.on_move = None   # called when the pointer moves, to let tests change the page
        self.closed = False
        self._box = None
        self._loader = 0

    async def start(self, on_message, on_close):
        self.on_message, self.on_close = on_message, on_close

    def send(self, message):
        method, params, sid = message["method"], message.get("params") or {}, message.get("sessionId")
        self.calls.append((method, params, sid))
        try:
            reply = {"id": message["id"], "result": self.handle(method, params, sid)}
        except CDPError as exc:
            reply = {"id": message["id"], "error": {"message": str(exc)}}
        asyncio.get_running_loop().call_soon(self.on_message, reply)

    def emit(self, method, params, sid=None):
        message = {"method": method, "params": params}
        if sid:
            message["sessionId"] = sid
        asyncio.get_running_loop().call_soon(self.on_message, message)

    def close(self):
        if not self.closed:
            self.closed = True
            asyncio.get_running_loop().call_soon(self.on_close)

    def sent(self, method):
        return [params for name, params, _ in self.calls if name == method]

    def handle(self, method, params, sid):
        target = sid[2:] if sid else None
        if method == "Target.getTargets":
            return {"targetInfos": [{"targetId": t, "type": "page", **info} for t, info in self.pages.items()]}
        if method == "Target.getTargetInfo":
            if params["targetId"] not in self.pages:
                raise CDPError("No target with given id found")
            return {"targetInfo": {"targetId": params["targetId"], "type": "page", **self.pages[params["targetId"]]}}
        if method == "Target.createTarget":
            target = f"P{len(self.pages) + 1}"
            self.pages[target] = {"url": params["url"], "title": ""}
            return {"targetId": target}
        if method == "Target.attachToTarget":
            return {"sessionId": "S-" + params["targetId"]}
        if method == "Target.closeTarget":
            self.pages.pop(params["targetId"])
            return {"success": True}
        if method == "Page.navigate":
            self._loader += 1
            loader = f"L{self._loader}"
            self.pages[target] = {"url": params["url"], "title": self.titles.get(params["url"], "")}
            self.emit("Page.frameNavigated", {"frame": {"id": target, "url": params["url"], "loaderId": loader}}, sid)
            self.emit("Page.lifecycleEvent", {"frameId": target, "loaderId": loader, "name": "load"}, sid)
            return {"frameId": target, "loaderId": loader}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": self.ax}
        if method == "DOM.describeNode":
            return {"node": self.nodes.get(params["backendNodeId"], {"nodeName": "BUTTON", "attributes": []})}
        if method == "DOM.getBoxModel":
            self._box = params["backendNodeId"]
            return {"model": {"border": [10, 20, 110, 20, 110, 40, 10, 40]}}
        if method == "Page.getLayoutMetrics":
            return {"cssVisualViewport": {"clientWidth": 1280, "clientHeight": 900},
                    "cssContentSize": {"width": 1280, "height": 3000}}
        if method == "DOM.getNodeForLocation":
            return {"backendNodeId": self.hit or self._box, "frameId": target}
        if method == "Page.createIsolatedWorld":
            return {"executionContextId": 7}
        if method == "DOM.resolveNode":
            return {"object": {"objectId": f"node:{params['backendNodeId']}"}}
        if method == "Runtime.evaluate":
            return {"result": {"objectId": f"node:{self.focused}"} if self.focused else {"type": "object"}}
        if method == "Runtime.callFunctionOn":
            backend = int(params["objectId"].split(":")[1])
            if params["functionDeclaration"] == CONTAINS_JS:
                other = int(params["arguments"][0]["objectId"].split(":")[1])
                return {"result": {"value": other == backend}}
            if params["functionDeclaration"] == DESCRIBE_JS:
                default = {"tag": "button", "type": "", "autocomplete": "", "text": "", "submit": ""}
                return {"result": {"value": self.labels.get(backend, default)}}
            return {"result": {}}
        if method == "DOM.focus":
            self.focused = params["backendNodeId"]
        if method == "Input.dispatchMouseEvent" and params["type"] == "mouseMoved" and self.on_move:
            self.on_move()
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(self.image).decode()}
        return {}


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class FakeLauncher:
    def __init__(self):
        self.browsers = []
        self.terminated = []

    @property
    def browser(self):
        return self.browsers[-1]

    def find_browser(self, config):
        return "/opt/fake/chrome"

    async def launch(self, executable, profile_dir, headless):
        self.profile_dir, self.headless = profile_dir, headless
        self.browsers.append(FakeBrowser())
        return FakeProcess(), await CDPPipe(self.browser).start()

    def terminate(self, process):
        if process.returncode is None:
            process.returncode = -15
        self.terminated.append(process)


def ax(node_id, role, name="", parent=None, children=(), backend=None, value=None, ignored=False, **props):
    node = {"nodeId": str(node_id), "role": {"type": "role", "value": role},
            "childIds": [str(child) for child in children]}
    if name:
        node["name"] = {"type": "computedString", "value": name}
    if parent is not None:
        node["parentId"] = str(parent)
    if backend:
        node["backendDOMNodeId"] = backend
    if value is not None:
        node["value"] = {"type": "string", "value": value}
    if ignored:
        node["ignored"] = True
    node["properties"] = [{"name": key, "value": {"type": "generic", "value": val}} for key, val in props.items()]
    return node


SHOP_AX = [
    ax(1, "RootWebArea", "Shop", children=[2], backend=1),
    ax(2, "generic", parent=1, children=[3, 4, 5, 8, 9, 10, 11, 12], ignored=True, backend=2),
    ax(3, "heading", "Welcome", parent=2, children=[20], backend=3, level=2),
    ax(20, "StaticText", "Welcome", parent=3, children=[30]),
    ax(30, "InlineTextBox", "Welcome", parent=20),
    ax(4, "paragraph", parent=2, children=[21, 22, 23], backend=4),
    ax(21, "StaticText", "Fresh ", parent=4),
    ax(22, "StaticText", "bread", parent=4),
    ax(23, "StaticText", " daily.", parent=4),
    ax(5, "paragraph", parent=2, children=[24], backend=5),
    ax(24, "StaticText", "Fresh  bread daily.", parent=5),
    ax(8, "link", "Read more", parent=2, children=[25], backend=108, url="https://shop.example/about"),
    ax(25, "StaticText", "Read more", parent=8),
    ax(9, "textbox", "Search", parent=2, children=[26], backend=109, value="rye", editable="plaintext"),
    ax(26, "generic", parent=9, children=[27], backend=126),
    ax(27, "StaticText", "rye", parent=26),
    ax(10, "textbox", "Password", parent=2, backend=110),
    ax(11, "button", "Buy now", parent=2, backend=111),
    ax(12, "checkbox", "Gift wrap", parent=2, backend=112, checked="true"),
]
PASSWORD_NODE = {"nodeName": "INPUT", "attributes": ["type", "password", "autocomplete", "current-password"]}
TEXT_NODE = {"nodeName": "INPUT", "attributes": ["type", "search", "name", "q"]}


# ---- unit tests --------------------------------------------------------------

class UrlTests(unittest.TestCase):
    def test_accepts_web_urls(self):
        self.assertEqual(parse_url("example.com"), ("https://example.com/", "example.com"))
        self.assertEqual(parse_url("HTTP://Example.COM/a?b=1#c"), ("http://example.com/a?b=1#c", "example.com"))
        self.assertEqual(parse_url("example.com:8443/x"), ("https://example.com:8443/x", "example.com"))
        self.assertEqual(parse_url("about:blank"), ("about:blank", "about:blank"))
        self.assertEqual(parse_url("https://bücher.de/")[1], "xn--bcher-kva.de")

    def test_rejects_other_schemes_and_tricks(self):
        for url in ("file:///etc/passwd", "chrome://settings", "chrome-extension://abc/x.html", "devtools://devtools",
                    "javascript:alert(1)", "JavaScript://%0aalert(1)", "data:text/html,<b>x</b>", "blob:https://a.com/1",
                    "view-source:https://a.com", "about:config", "ftp://a.com", "java\nscript:alert(1)",
                    "https://user:pw@a.com/", "https://a.com\\@b.com", "https://good.com\\.evil.com", "",
                    "https://a b.com", "https://%65vil.com", "http://1.2.3.4.5/", "http://0x100000000/"):
            with self.subTest(url=url), self.assertRaises(BrowserError):
                parse_url(url)

    def test_numeric_hosts_are_canonical(self):
        # Browsers read these as IPv4 addresses, so policy must too.
        for url, host in (("http://127.1/", "127.0.0.1"), ("http://0x7f.0.0.1/", "127.0.0.1"),
                          ("http://2130706433/", "127.0.0.1"), ("http://0177.0.0.1/", "127.0.0.1"),
                          ("http://[::ffff:127.0.0.1]/", "::ffff:127.0.0.1")):
            with self.subTest(url=url):
                self.assertEqual(parse_url(url)[1], host)


class RenderTests(unittest.TestCase):
    def test_compact_rendering_refs_and_dedupe(self):
        content, refs = render_ax(SHOP_AX, "https://shop.example/", secret={110})
        self.assertEqual(content.splitlines(), [
            "## Welcome",
            "Fresh bread daily.",
            '[1] link "Read more" -> /about',
            '[2] textbox "Search" value="rye"',
            '[3] textbox "Password" (password field: the owner fills it in, use handoff)',
            '[4] button "Buy now"',
            '[5] checkbox "Gift wrap" checked',
        ])
        self.assertEqual({ref: info["backend"] for ref, info in refs.items()},
                         {1: 108, 2: 109, 3: 110, 4: 111, 5: 112})
        self.assertTrue(refs[3]["password"])
        self.assertEqual(refs[4]["name"], "Buy now")


class BrokerTestCase(unittest.IsolatedAsyncioTestCase):
    browser_config: dict = {}
    rules: list = []

    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp)
        self.config = Config({"browser": dict(self.browser_config), "policy": {"rules": list(self.rules)}},
                             path=self.tmp / "config.json")
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.asked, self.answers, self.approvals = [], {}, []
        self.while_asking = None   # runs while an approval is pending, to let tests change the page
        self.policy.add_listener(self._answer)
        self.launcher = FakeLauncher()
        self.broker = BrowserBroker(self.config, self.policy, self.audit, launcher=self.launcher)
        self.dns = {}

        async def resolve(host):
            return self.dns.get(host, ["93.184.216.34"])
        self.broker._resolve = resolve

    def _answer(self, approval):
        self.asked.append((approval.kind, approval.summary))
        self.approvals.append(approval)
        if self.while_asking:
            self.while_asking()
        self.policy.decide(approval.id, allow=self.answers.get(approval.kind, False))

    async def asyncTearDown(self):
        await self.broker.stop()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def kinds(self):
        return [kind for kind, _ in self.asked]

    async def load_shop(self):
        result = await self.broker.navigate("https://shop.example/")
        self.assertTrue(result["ok"], result)
        browser = self.launcher.browser
        browser.ax = SHOP_AX
        browser.nodes = {110: PASSWORD_NODE, 109: TEXT_NODE}
        browser.labels = {
            108: {"tag": "a", "text": "Read more", "submit": ""},
            109: {"tag": "input", "type": "search", "text": "Search", "submit": "Search"},
            111: {"tag": "button", "text": "Buy now", "submit": ""},
        }
        snapshot = await self.broker.snapshot()
        self.assertTrue(snapshot["ok"], snapshot)
        return browser, snapshot


class NavigationPolicyTests(BrokerTestCase):
    """An owner who chose to approve every site (an "ask" rule)."""
    rules = [{"kind": "browser.navigate", "target": "*", "decision": "ask"}]

    async def test_bad_urls_are_rejected_before_asking_or_launching(self):
        self.dns["router.example"] = ["192.168.1.1"]
        for url in ("file:///etc/passwd", "javascript:alert(1)", "chrome://settings", "data:text/html,hi",
                    "http://127.0.0.1:8224/admin", "http://localhost:3000", "localhost:3000", "http://app.localhost/",
                    "http://169.254.169.254/latest", "http://10.0.0.8/", "http://[::1]/", "http://127.1/",
                    "http://0x7f000001/", "http://100.100.1.1/", "https://router.example/"):
            with self.subTest(url=url):
                result = await self.broker.navigate(url)
                self.assertFalse(result["ok"])
                self.assertTrue(result["error"])
        self.assertEqual(self.asked, [])
        self.assertEqual(self.launcher.browsers, [])

    async def test_navigation_asks_and_denial_is_returned(self):
        result = await self.broker.navigate("https://shop.example/")
        self.assertEqual(result["ok"], False)
        self.assertIn("Not permitted", result["error"])
        self.assertEqual(self.kinds(), ["browser.navigate"])
        self.assertEqual(self.launcher.browsers, [])          # nothing launched for a denied action
        self.answers["browser.navigate"] = True
        result = await self.broker.navigate("shop.example/menu")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["url"], "https://shop.example/menu")
        self.assertTrue(result["loaded"])
        self.assertEqual(self.launcher.browser.sent("Page.navigate"), [{"url": "https://shop.example/menu"}])
        events = [entry["event"] for entry in self.audit.tail()]
        self.assertIn("browser.navigate", events)

    async def test_open_and_tabs(self):
        self.answers["browser.navigate"] = True
        opened = await self.broker.open("https://shop.example/")
        self.assertTrue(opened["ok"], opened)
        listed = await self.broker.tabs()
        self.assertEqual([tab["active"] for tab in listed["tabs"]], [False, True])
        self.assertEqual(listed["tabs"][1]["url"], "https://shop.example/")
        closed = await self.broker.close_tab(opened["tab"])        # its own tab: no approval needed
        self.assertTrue(closed["ok"], closed)
        self.assertEqual(len((await self.broker.tabs())["tabs"]), 1)
        self.assertEqual(self.kinds(), ["browser.navigate"])


class InteractionPolicyTests(BrokerTestCase):
    browser_config = {"allow": ["shop.example"]}
    rules = [{"kind": "browser.interact", "target": "shop.example", "decision": "allow"}]

    async def test_snapshot_change_detection(self):
        browser, first = await self.load_shop()
        self.assertTrue(first["changed"])
        self.assertIn('[4] button "Buy now"', first["content"])
        self.assertIn("(password field", first["content"])
        self.assertFalse((await self.broker.snapshot())["changed"])
        browser.ax = SHOP_AX[:-1]
        self.assertTrue((await self.broker.snapshot())["changed"])
        short = await self.broker.snapshot(max_chars=100)
        self.assertIn("truncated", short["content"])

    async def test_click_uses_interact_or_sensitive(self):
        browser, _ = await self.load_shop()
        result = await self.broker.click(1)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.asked, [])
        pressed = [p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"]
        self.assertEqual((pressed[0]["x"], pressed[0]["y"]), (60, 30))

        result = await self.broker.click(4)
        self.assertFalse(result["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        self.assertIn('Click button "Buy now"', self.asked[0][1])
        self.assertEqual(len([p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"]), 1)

        self.answers["browser.sensitive"] = True
        self.assertTrue((await self.broker.click(4))["ok"])
        self.assertEqual(len([p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"]), 2)
        self.assertFalse((await self.broker.click(99))["ok"])

    async def test_click_checks_what_is_actually_under_the_pointer(self):
        browser, _ = await self.load_shop()
        browser.hit = 555
        browser.labels[555] = {"tag": "button", "text": "Confirm order", "submit": ""}
        result = await self.broker.click(1)              # "Read more", but an overlay sits on top
        self.assertFalse(result["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        self.assertIn("covered by button", self.asked[0][1])

    async def test_sensitive_label_in_text_but_not_aria_name(self):
        browser, _ = await self.load_shop()
        browser.labels[108] = {"tag": "a", "text": "Read more / Pay now", "submit": ""}
        self.assertFalse((await self.broker.click(1))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])

    async def test_password_fields_are_never_typed(self):
        browser, _ = await self.load_shop()
        result = await self.broker.type(3, "hunter2")
        self.assertFalse(result["ok"])
        self.assertIn("handoff", result["error"])
        self.assertEqual(browser.sent("Input.insertText"), [])
        policy_events = [e for e in self.audit.tail() if e["event"] == "policy" and e["kind"] == "browser.password"]
        self.assertEqual(policy_events[-1]["decision"], "deny")
        # Same answer when the element only turns out to be a password field when described.
        browser.nodes[109] = PASSWORD_NODE
        self.assertFalse((await self.broker.type(2, "hunter2"))["ok"])
        self.assertEqual(browser.sent("Input.insertText"), [])

    async def test_type_and_submit(self):
        browser, _ = await self.load_shop()
        result = await self.broker.type(2, "sourdough", submit=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(browser.sent("Input.insertText"), [{"text": "sourdough"}])
        keys = [p["key"] for p in browser.sent("Input.dispatchKeyEvent")]
        self.assertEqual(keys, ["Enter", "Enter"])
        self.assertEqual(self.asked, [])
        # Submitting a form whose button places an order needs one-time approval.
        browser.labels[109] = {"tag": "input", "type": "text", "text": "Address", "submit": "Place order"}
        self.assertFalse((await self.broker.type(2, "1 Main St", submit=True))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        self.assertEqual(len(browser.sent("Input.insertText")), 1)

    async def test_press(self):
        browser, _ = await self.load_shop()
        self.assertFalse((await self.broker.press("F5"))["ok"])
        self.assertTrue((await self.broker.press("Tab"))["ok"])
        browser.focused = 111
        self.assertFalse((await self.broker.press("Enter"))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        self.assertTrue((await self.broker.scroll("down", 400))["ok"])
        wheel = browser.sent("Input.dispatchMouseEvent")[-1]
        self.assertEqual((wheel["type"], wheel["deltaY"]), ("mouseWheel", 400))

    async def test_ordinary_clicks_after_untrusted_content_need_no_prompt(self):
        await self.load_shop()
        self.assertTrue((await self.broker.click(1, tainted=True))["ok"])
        self.assertEqual(self.asked, [])

    async def test_screenshot_change_detection(self):
        browser, _ = await self.load_shop()
        first = await self.broker.screenshot()
        self.assertEqual((first["mime"], first["changed"]), ("image/jpeg", True))
        self.assertEqual(base64.b64decode(first["data_b64"]), browser.image)
        self.assertFalse((await self.broker.screenshot())["changed"])
        browser.image = b"\xff\xd8 second"
        self.assertTrue((await self.broker.screenshot(full_page=True))["changed"])
        self.assertTrue(browser.sent("Page.captureScreenshot")[-1]["captureBeyondViewport"])

    async def test_browser_pages_are_not_readable(self):
        browser, _ = await self.load_shop()
        browser.pages["P1"]["url"] = "chrome://settings/passwords"
        for result in (await self.broker.snapshot(), await self.broker.screenshot(), await self.broker.click(1)):
            self.assertFalse(result["ok"])
            self.assertIn("browser page", result["error"])
        self.assertEqual((await self.broker.tabs())["tabs"][0]["url"], "(hidden)")

    async def test_document_requests_to_local_hosts_are_failed(self):
        browser, _ = await self.load_shop()
        self.dns["intranet.example"] = ["10.1.2.3"]
        for request_id, url in enumerate(("http://192.168.1.1/", "https://shop.example/next", "http://printer.local/",
                                          "https://intranet.example/", "http://localhost:8224/")):
            browser.emit("Fetch.requestPaused", {"requestId": str(request_id), "request": {"url": url},
                                                 "resourceType": "Document"}, "S-P1")
        for _ in range(20):
            await asyncio.sleep(0.01)
        self.assertEqual([p["requestId"] for p in browser.sent("Fetch.failRequest")], ["0", "2", "3", "4"])
        self.assertEqual([p["requestId"] for p in browser.sent("Fetch.continueRequest")], ["1"])

    async def test_startup_hardening(self):
        browser, _ = await self.load_shop()
        self.assertEqual(browser.sent("Browser.setDownloadBehavior"), [{"behavior": "deny"}])
        self.assertIn({"enabled": True}, browser.sent("Page.setInterceptFileChooserDialog"))
        self.assertTrue(browser.sent("Fetch.enable"))
        self.assertIn("geolocation", [p["permission"]["name"] for p in browser.sent("Browser.setPermission")])
        self.assertEqual(self.launcher.profile_dir, self.tmp / "assistant" / "browser-profile")

    async def test_restarts_after_the_owner_closes_the_browser(self):
        await self.load_shop()
        first = self.launcher.browser
        first.close()
        await asyncio.sleep(0)
        result = await self.broker.navigate("https://shop.example/")
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self.launcher.browsers), 2)
        self.assertEqual(len(self.launcher.terminated), 1)
        await self.broker.stop()
        self.assertFalse(self.broker.running())
        self.assertEqual(len(self.launcher.terminated), 2)

    async def test_handoff_waits_for_the_owner(self):
        browser, _ = await self.load_shop()
        self.answers["browser.handoff"] = True
        result = await self.broker.handoff("Please sign in with your password")
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["completed"])
        self.assertEqual(self.kinds(), ["browser.handoff"])
        self.assertIn("Please finish in the CodeRunner browser", self.asked[0][1])
        self.assertTrue(browser.sent("Page.bringToFront"))
        self.assertEqual(browser.sent("Page.setInterceptFileChooserDialog")[-2:], [{"enabled": False}, {"enabled": True}])
        self.answers["browser.handoff"] = False
        self.assertFalse((await self.broker.handoff("2FA code"))["completed"])
        self.config.set("browser.headless", True)
        self.assertIn("headless", (await self.broker.handoff("log in"))["error"])


class ApprovalContentTests(BrokerTestCase):
    """What the owner is asked must show everything that will happen, and approvals bind to it."""
    browser_config = {"allow": ["shop.example", "partner.example"]}
    rules = [{"kind": "browser.interact", "target": "shop.example", "decision": "allow"}]

    async def test_typed_text_is_shown_in_full(self):
        await self.load_shop()
        self.config.set("policy.rules", [{"kind": "browser.interact", "target": "*", "decision": "ask"}])
        text = "dear owner " * 60 + "PS: wire the money"
        self.assertFalse((await self.broker.type(2, text, tainted=True))["ok"])
        approval = self.approvals[-1]
        self.assertEqual(approval.kind, "browser.interact")
        self.assertIn(json.dumps(text), approval.summary)
        self.assertEqual(approval.detail["text"], text)
        self.assertEqual(approval.detail["url"], "https://shop.example/")
        self.answers["browser.interact"] = True
        self.assertTrue((await self.broker.type(2, "rye", tainted=True))["ok"])
        typed = [e for e in self.audit.tail() if e["event"] == "browser.type"]
        self.assertEqual(typed[-1]["text"], "rye")

    async def test_card_and_password_values_never_appear(self):
        browser, _ = await self.load_shop()
        browser.nodes[109] = {"nodeName": "INPUT", "attributes": ["autocomplete", "cc-number"]}
        self.answers["browser.sensitive"] = True
        self.assertTrue((await self.broker.type(2, "4111111111111111"))["ok"])
        approval = self.approvals[-1]
        self.assertEqual(approval.kind, "browser.sensitive")
        self.assertIn("[card number]", approval.summary)
        self.assertNotIn("text", approval.detail)
        self.assertFalse((await self.broker.type(3, "hunter2"))["ok"])
        log = (self.tmp / "audit.jsonl").read_text()
        self.assertNotIn("4111111111111111", log)
        self.assertNotIn("hunter2", log)

    async def test_click_labels_are_complete(self):
        browser, _ = await self.load_shop()
        long_text = "Read more " + "about our bakery " * 30 + "and pay now"
        browser.labels[108] = {"tag": "a", "text": long_text, "submit": ""}
        self.assertFalse((await self.broker.click(1))["ok"])
        approval = self.approvals[-1]
        self.assertEqual(approval.kind, "browser.sensitive")      # "pay" is far past the old 200-char cut
        self.assertIn(long_text, approval.summary)
        self.assertIn(long_text, approval.detail["label"])
        browser.labels[108] = {"tag": "a", "text": "Read more", "submit": "", "truncated": True}
        self.assertFalse((await self.broker.click(1))["ok"])
        self.assertEqual(self.approvals[-1].kind, "browser.sensitive")   # cut text is treated as risky

    def mouse_presses(self, browser):
        return [p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"]

    async def test_click_approval_is_bound_to_the_element(self):
        browser, _ = await self.load_shop()
        self.answers["browser.sensitive"] = True
        # Still "sensitive", but no longer what the owner approved.
        self.while_asking = lambda: browser.labels.update({111: {"tag": "button", "text": "Buy 10 now"}})
        result = await self.broker.click(4)
        self.assertFalse(result["ok"])
        self.assertIn("nothing was clicked", result["error"])
        # Something else moved under the pointer during the approval.
        browser.labels[111] = {"tag": "button", "text": "Buy now", "submit": ""}
        self.while_asking = lambda: setattr(browser, "hit", 777)
        browser.labels[777] = {"tag": "button", "text": "Buy now", "submit": ""}
        self.assertFalse((await self.broker.click(4))["ok"])
        self.assertEqual(self.mouse_presses(browser), [])
        # Page script repositions things when the pointer arrives.
        self.while_asking, browser.hit = None, None
        browser.on_move = lambda: setattr(browser, "hit", 777)
        result = await self.broker.click(4)
        self.assertFalse(result["ok"])
        self.assertIn("under the pointer", result["error"])
        self.assertEqual(self.mouse_presses(browser), [])
        browser.on_move, browser.hit = None, None
        self.assertTrue((await self.broker.click(4))["ok"])
        self.assertEqual(len(self.mouse_presses(browser)), 1)

    async def test_key_press_approval_is_bound_to_the_focus(self):
        browser, _ = await self.load_shop()
        self.config.set("policy.rules", [{"kind": "browser.interact", "target": "*", "decision": "ask"}])
        browser.focused = 109
        self.answers["browser.interact"] = True
        self.while_asking = lambda: setattr(browser, "focused", 111)
        result = await self.broker.press("Enter", tainted=True)
        self.assertFalse(result["ok"])
        self.assertIn("Focus changed", result["error"])
        self.assertEqual(browser.sent("Input.dispatchKeyEvent"), [])


class DefaultPermissionTests(BrokerTestCase):
    """Without owner rules: ordinary browsing runs, committing steps and leaks ask."""

    async def test_browsing_runs_without_prompts(self):
        browser, _ = await self.load_shop()
        self.assertTrue((await self.broker.navigate("https://news.example/"))["ok"])
        self.assertTrue((await self.broker.navigate("https://shop.example/"))["ok"])
        browser.ax = SHOP_AX
        await self.broker.snapshot()
        self.assertTrue((await self.broker.click(1, tainted=True))["ok"])
        self.assertTrue((await self.broker.type(2, "sourdough", tainted=True))["ok"])
        self.assertEqual(self.asked, [])
        self.assertFalse((await self.broker.click(4))["ok"])         # "Buy now"
        self.assertEqual(self.kinds(), ["browser.sensitive"])

    async def test_private_data_needs_approval_before_it_reaches_a_site(self):
        browser, _ = await self.load_shop()
        run = SimpleNamespace(content_tainted=True, private_sources={"calendar"}, shared=set())
        token = current_run.set(run)
        try:
            self.assertTrue((await self.broker.click(1, tainted=True))["ok"])      # no data in a click
            self.assertFalse((await self.broker.type(2, "Delhi", tainted=True))["ok"])
            self.assertEqual(self.kinds(), ["data.share"])
            self.assertIn('Type "Delhi"', self.asked[0][1])
            self.assertEqual(browser.sent("Input.insertText"), [])
            self.answers["data.share"] = True
            self.assertTrue((await self.broker.type(2, "Delhi", tainted=True))["ok"])
            self.assertTrue((await self.broker.type(2, "Mumbai", tainted=True))["ok"])   # once per site
            self.assertEqual(self.kinds(), ["data.share", "data.share"])
            self.answers["data.share"] = False
            result = await self.broker.navigate("https://collector.example/?d=Delhi", tainted=True)
            self.assertFalse(result["ok"])
            self.assertEqual(self.approvals[-1].target, "collector.example")
        finally:
            current_run.reset(token)


class NavigationGuardTests(BrokerTestCase):
    browser_config = {"allow": ["shop.example", "partner.example"]}
    rules = [{"kind": "browser.interact", "target": "shop.example", "decision": "allow"}]

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0.01)

    def request(self, browser, request_id, url, frame="P1"):
        browser.emit("Fetch.requestPaused", {"requestId": request_id, "request": {"url": url},
                                             "frameId": frame, "resourceType": "Document"})

    async def test_navigation_approvals_carry_the_full_url(self):
        self.config.set("browser.allow", [])
        self.config.set("policy.rules", [{"kind": "browser.navigate", "target": "*", "decision": "ask"}])
        await self.broker.navigate("https://shop.example/a?b=1")
        await self.broker.open("https://shop.example/c")
        self.assertEqual([a.detail["url"] for a in self.approvals],
                         ["https://shop.example/a?b=1", "https://shop.example/c"])

    async def test_leaving_the_site_during_an_action(self):
        browser, _ = await self.load_shop()
        # By default the page's own links may lead anywhere public: the model did not choose them.
        self.broker._busy["P1"] = True
        self.request(browser, "0", "https://elsewhere.example/")
        await self.settle()
        self.assertEqual(browser.sent("Fetch.failRequest"), [])
        # An owner who approves every site: leaving it mid-action is refused, since nobody can be asked then.
        self.config.set("policy.rules", [{"kind": "browser.navigate", "target": "*", "decision": "ask"}])
        self.request(browser, "1", "https://elsewhere.example/?q=secret")
        self.request(browser, "2", "https://shop.example/next")
        self.request(browser, "3", "https://www.shop.example/")
        self.request(browser, "4", "https://partner.example/")    # in browser.allow: navigate is allowed
        self.request(browser, "5", "https://elsewhere.example/ad", frame="SUBFRAME")
        await self.settle()
        self.assertEqual([p["requestId"] for p in browser.sent("Fetch.failRequest")], ["1"])
        self.assertIn("approval", self.broker._blocked["P1"])
        del self.broker._busy["P1"]                    # the owner's own browsing is not limited
        self.request(browser, "6", "https://elsewhere.example/")
        await self.settle()
        self.assertEqual([p["requestId"] for p in browser.sent("Fetch.failRequest")], ["1"])

    async def test_popups_follow_the_rules_of_the_action_that_opened_them(self):
        browser, _ = await self.load_shop()
        self.config.set("policy.rules", [{"kind": "browser.navigate", "target": "*", "decision": "ask"}])
        self.broker._busy["P1"] = False
        for popup, url in (("P7", "https://shop.example/help"), ("P8", "https://elsewhere.example/"),
                           ("P9", "http://10.0.0.5/admin")):
            browser.pages[popup] = {"url": "", "title": ""}
            browser.emit("Target.targetCreated", {"targetInfo": {"targetId": popup, "type": "page",
                                                                 "openerId": "P1", "url": ""}})
            self.request(browser, popup, url, frame=popup)
        await self.settle()
        self.assertEqual(sorted(p["requestId"] for p in browser.sent("Fetch.failRequest")), ["P8", "P9"])
        self.assertEqual(sorted(p["targetId"] for p in browser.sent("Target.closeTarget")), ["P8", "P9"])
        self.assertEqual([p["requestId"] for p in browser.sent("Fetch.continueRequest")][-1:], ["P7"])
        del self.broker._busy["P1"]
        result = await self.broker._result("P1")
        self.assertIn("blocked", result)
        self.assertIn(self.broker._tab_id("P7"), result["new_tabs"])

    async def test_admin_interface_is_never_opened(self):
        self.config.set("browser.allow", ["127.0.0.1", "localhost"])
        for url in ("http://127.0.0.1:8224/", "http://localhost:8224/approvals", "127.0.0.1:8224"):
            with self.subTest(url=url):
                result = await self.broker.navigate(url)
                self.assertFalse(result["ok"])
                self.assertIn("admin", result["error"])
        self.assertTrue((await self.broker.navigate("http://127.0.0.1:9000/"))["ok"])


class MessagingClientTests(BrokerTestCase):
    browser_config = {"allow": ["web.whatsapp.com", "app.slack.com"]}
    rules = [{"kind": "browser.interact", "target": "*", "decision": "allow"}]

    def test_approval_words_are_sensitive(self):
        for text in ("Allow", "Approve", "approve 1a2b3c4d", "Authorize app", "Grant access", "Deny", "Permit",
                     "approval", "Authorisation"):
            self.assertTrue(sensitive(text), text)
        for text in ("Bookmark", "Posts", "Allowance", "Read more"):
            self.assertFalse(sensitive(text), text)

    async def test_any_interaction_in_a_chat_client_is_sensitive(self):
        for host in ("web.whatsapp.com", "app.slack.com"):
            with self.subTest(host=host):
                self.asked.clear()
                self.assertTrue((await self.broker.navigate(f"https://{host}/"))["ok"])
                browser = self.launcher.browser
                browser.ax = SHOP_AX
                browser.labels = {108: {"tag": "a", "text": "Read more", "submit": ""}}
                await self.broker.snapshot()
                self.assertFalse((await self.broker.click(1))["ok"])
                self.assertFalse((await self.broker.type(2, "hello"))["ok"])
                self.assertFalse((await self.broker.press("Tab"))["ok"])
                self.assertEqual(self.kinds(), ["browser.sensitive"] * 3)


# ---- real browser ----------------------------------------------------------------

FIXTURE = b"""<!doctype html><html><head><title>Fixture shop</title></head><body>
<h1>Fixture shop</h1>
<p>Fresh bread every <b>morning</b>.</p>
<a href="/next">Next page</a>
<a href="/away">Sneaky redirect</a>
<form onsubmit="return false">
  <input type="text" name="q" aria-label="Query" value="old">
  <input type="password" name="pw" aria-label="Password" autocomplete="current-password">
  <button type="button" onclick="document.getElementById('out').textContent = 'bought'">Buy now</button>
</form>
<button type="button" onclick="document.getElementById('out').textContent = 'echo ' + document.querySelector('[name=q]').value">Echo</button>
<button type="button" onclick="document.getElementById('out').textContent = confirm('Sure?') ? 'yes' : 'declined'">Ask</button>
<button type="button" onclick="window.open('http://10.255.255.1/pop')">Popup</button>
<a href="http://127.0.0.2:PORT/elsewhere">Elsewhere</a>
<div id="out"></div>
</body></html>"""


class _Fixture(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/away":
            self.send_response(302)
            self.send_header("Location", "http://10.255.255.1/admin")
            self.end_headers()
            return
        port = str(self.server.server_port).encode()
        body = FIXTURE.replace(b"PORT", port) if self.path == "/" else b"<title>Next</title><h2>Second page</h2>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@unittest.skipIf(os.environ.get("CODERUNNER_SKIP_BROWSER") == "1", "CODERUNNER_SKIP_BROWSER=1")
@unittest.skipIf(launcher.find_browser(Config({})) is None, "no Chromium-family browser installed")
class RealBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fixture)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.config = Config({
            "browser": {"headless": True, "allow": ["127.0.0.1", "127.0.0.2"]},
            "policy": {"rules": [{"kind": "browser.interact", "target": "127.0.0.1", "decision": "allow"},
                                 {"kind": "browser.navigate", "target": "127.0.0.2", "decision": "deny"}]},
        }, path=self.tmp / "config.json")
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.asked, self.answers = [], {}

        def answer(approval):
            self.asked.append(approval.kind)
            self.policy.decide(approval.id, allow=self.answers.get(approval.kind, False))
        self.policy.add_listener(answer)
        self.broker = BrowserBroker(self.config, self.policy, self.audit)

    async def asyncTearDown(self):
        await self.broker.stop()
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def refs(self):
        return {info["name"]: ref for ref, info in self.broker._refs[self.broker._active].items()}

    async def test_end_to_end(self):
        opened = await self.broker.navigate(self.base + "/")
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(opened["title"], "Fixture shop")
        self.assertEqual(self.asked, [])                      # 127.0.0.1 is in browser.allow
        process = self.broker._process
        profile = str(self.tmp / "assistant" / "browser-profile")
        self.assertIn(profile, " ".join(subprocess.run(["ps", "-p", str(process.pid), "-o", "command="],
                                                       capture_output=True, text=True).stdout.split()))

        snap = await self.broker.snapshot()
        self.assertTrue(snap["ok"], snap)
        content = snap["content"]
        self.assertIn("# Fixture shop", content)
        self.assertIn("Fresh bread every morning.", content)
        self.assertRegex(content, r'\[\d+\] link "Next page" -> /next')
        self.assertRegex(content, r'\[\d+\] textbox "Query" value="old"')
        self.assertRegex(content, r'\[\d+\] textbox "Password" \(password field')
        self.assertRegex(content, r'\[\d+\] button "Buy now"')
        self.assertFalse((await self.broker.snapshot())["changed"])
        refs = self.refs()

        typed = await self.broker.type(refs["Query"], "rye loaf")
        self.assertTrue(typed["ok"], typed)
        clicked = await self.broker.click(refs["Echo"])
        self.assertTrue(clicked["ok"], clicked)
        after = await self.broker.snapshot()
        self.assertTrue(after["changed"])
        self.assertIn("echo rye loaf", after["content"])

        asked = await self.broker.click(refs["Ask"])            # a confirm() must not hang the click
        self.assertTrue(asked["ok"], asked)
        self.assertEqual(asked["dialog"]["type"], "confirm")
        self.assertIn("declined", (await self.broker.snapshot())["content"])

        # A popup to a private address opened by a click is failed and closed before it loads.
        popup = await self.broker.click(refs["Popup"])
        self.assertTrue(popup["ok"], popup)
        self.assertIn("10.255.255.1", popup.get("blocked", ""))
        self.assertEqual(len((await self.broker.tabs())["tabs"]), 1)
        self.assertTrue(any(e["event"] == "browser.blocked" and "10.255.255.1/pop" in e["url"]
                            for e in self.audit.tail()))
        # Following a link to another site needs browser.navigate to be allowed outright.
        away = await self.broker.click(refs["Elsewhere"])
        self.assertIn("approval", away.get("blocked", ""), away)
        self.assertTrue((await self.broker.back())["ok"])
        self.assertIn("Unknown element", (await self.broker.click(refs["Echo"]))["error"])
        self.assertTrue((await self.broker.snapshot())["ok"])
        refs = self.refs()

        denied = await self.broker.type(refs["Password"], "hunter2")
        self.assertFalse(denied["ok"])
        self.assertIn("handoff", denied["error"])

        bought = await self.broker.click(refs["Buy now"])
        self.assertFalse(bought["ok"])
        self.assertEqual(self.asked, ["browser.sensitive"])
        self.assertNotIn("bought", (await self.broker.snapshot())["content"])
        self.answers["browser.sensitive"] = True
        self.assertTrue((await self.broker.click(self.refs()["Buy now"]))["ok"])
        self.assertIn("bought", (await self.broker.snapshot())["content"])
        self.assertEqual(self.asked, ["browser.sensitive", "browser.sensitive"])

        shot = await self.broker.screenshot()
        self.assertTrue(shot["ok"], shot)
        self.assertEqual(shot["mime"], "image/jpeg")
        self.assertTrue(base64.b64decode(shot["data_b64"]).startswith(b"\xff\xd8"))

        # A redirect into a private network is failed by the request guard.
        away = await self.broker.navigate(self.base + "/away")
        self.assertFalse(away["ok"])
        self.assertIn("10.255.255.1", away["error"])
        self.assertFalse((await self.broker.snapshot())["ok"])
        back = await self.broker.back()
        self.assertTrue(back["ok"], back)
        self.assertEqual(back["url"], self.base + "/")

        self.assertTrue((await self.broker.snapshot())["ok"])
        nxt = await self.broker.click(self.refs()["Next page"])
        self.assertTrue(nxt["navigated"], nxt)
        self.assertEqual(nxt["title"], "Next")
        tab = (await self.broker.open(self.base + "/next"))["tab"]
        self.assertEqual(len((await self.broker.tabs())["tabs"]), 2)
        self.assertTrue((await self.broker.close_tab(tab))["ok"])

        stopped = await self.broker.stop()
        self.assertTrue(stopped["ok"])
        self.assertIsNotNone(process.poll())
        deadline = time.monotonic() + 5
        leftovers = "?"
        while time.monotonic() < deadline:
            leftovers = subprocess.run(["pgrep", "-f", profile], capture_output=True, text=True).stdout.strip()
            if not leftovers:
                break
            await asyncio.sleep(0.2)
        self.assertEqual(leftovers, "", "browser processes survived stop()")


if __name__ == "__main__":
    unittest.main()
