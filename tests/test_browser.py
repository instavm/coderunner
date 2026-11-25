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
from unittest import mock

from coderunner.core.audit import AuditLog
from coderunner.browser import BrowserBroker, BrowserError
from coderunner.browser import broker as broker_module
from coderunner.browser import launcher
from coderunner.browser.broker import CHALLENGE_JS
from coderunner.browser.page import fit, page_changes, render_ax
from coderunner.browser.scripts import (ANCESTORS_JS, CONTAINS_JS, DESCRIBE_JS, OPTIONS_JS, SCROLL_JS, SELECT_OPTION_JS,
                                        SETTLE_JS)
from coderunner.browser.urls import parse_url
from coderunner.browser.cdp import CDPError, CDPPipe
from coderunner.core.config import Config
from coderunner.safety.policy import Policy, current_run
from coderunner.core.store import Store


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
        self.options = {}     # backendNodeId -> options of a <select>
        self.selected = {}
        self.focused = None
        self.hit = None
        self.hit_at = None    # (x, y) -> backendNodeId, for pages where what a click hits depends on the point
        self.parents = {}     # backendNodeId -> parent's, for hit tests that land inside an element
        self.image = b"\xff\xd8 first"
        self.on_move = None   # called when the pointer moves, to let tests change the page
        self.scroll = (0, 0)  # how far the page is scrolled (page coordinates of the window's corner)
        self.challenge = False  # a password box or CAPTCHA in view (CHALLENGE_JS)
        self.lifecycle = ("load",)    # lifecycle events a navigation sends
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
            for name in self.lifecycle:
                self.emit("Page.lifecycleEvent", {"frameId": target, "loaderId": loader, "name": name}, sid)
            return {"frameId": target, "loaderId": loader}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": self.ax}
        if method == "DOM.describeNode":
            backend = params.get("backendNodeId") or int(params["objectId"].split(":")[1])
            return {"node": {**self.nodes.get(backend, {"nodeName": "BUTTON", "attributes": []}), "backendNodeId": backend}}
        if method == "Runtime.getProperties":
            chain, node = [], int(params["objectId"].split(":")[1])
            while node is not None:
                chain.append(node)
                node = self.parents.get(node)
            return {"result": [{"name": str(i), "value": {"objectId": f"node:{b}"}} for i, b in enumerate(chain)]
                    + [{"name": "length", "value": {"value": len(chain)}}]}
        if method == "DOM.getBoxModel":
            self._box = params["backendNodeId"]
            return {"model": {"border": [10, 20, 110, 20, 110, 40, 10, 40]}}
        if method == "Page.getLayoutMetrics":
            return {"cssVisualViewport": {"clientWidth": 1280, "clientHeight": 900, "pageX": self.scroll[0],
                                          "pageY": self.scroll[1], "scale": 1},
                    "cssContentSize": {"width": 1280, "height": 3000}}
        if method == "DOM.getNodeForLocation":
            hit = self.hit_at and self.hit_at(params["x"] - self.scroll[0], params["y"] - self.scroll[1])
            return {"backendNodeId": hit or self.hit or self._box, "frameId": target}
        if method == "Page.createIsolatedWorld":
            return {"executionContextId": 7}
        if method == "DOM.resolveNode":
            return {"object": {"objectId": f"node:{params['backendNodeId']}"}}
        if method == "Runtime.evaluate" and params.get("expression") == CHALLENGE_JS:
            return {"result": {"value": self.challenge}}
        if method == "Runtime.evaluate":
            return {"result": {"objectId": f"node:{self.focused}"} if self.focused else {"type": "object"}}
        if method == "Runtime.callFunctionOn" and params["functionDeclaration"] == SETTLE_JS:
            return {"result": {"value": False}}
        if method == "Runtime.callFunctionOn" and params["functionDeclaration"] == SCROLL_JS:
            return {"result": {"value": {"moved": True, "percent": 20}}}
        if method == "Runtime.callFunctionOn":
            backend = int(params["objectId"].split(":")[1])
            if params["functionDeclaration"] == ANCESTORS_JS:
                return {"result": {"objectId": f"chain:{backend}"}}
            if params["functionDeclaration"] == CONTAINS_JS:
                other = int(params["arguments"][0]["objectId"].split(":")[1])
                while other is not None and other != backend:
                    other = self.parents.get(other)
                return {"result": {"value": other == backend}}
            if params["functionDeclaration"] == OPTIONS_JS:
                return {"result": {"value": self.options.get(backend)}}
            if params["functionDeclaration"] == SELECT_OPTION_JS:
                choice, options = params["arguments"][0]["value"].lower(), self.options.get(backend)
                if options is None:
                    return {"result": {"value": {"ok": False, "error": "not a dropdown (<select>); click it instead"}}}
                match = next((o for o in options if o.lower() == choice), None) or \
                    next((o for o in options if choice in o.lower()), None)
                if match is None:
                    return {"result": {"value": {"ok": False, "error": "no such option", "options": options}}}
                self.selected[backend] = match
                return {"result": {"value": {"ok": True, "selected": match}}}
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

    def test_result_lists_keep_each_entrys_values(self):
        def card(first, logo, stop, button):
            return [ax(first, "listitem", parent=1, children=[first + 1, first + 2, first + 3], backend=first),
                    ax(first + 1, "image", logo, parent=first, backend=first + 1),
                    ax(first + 2, "StaticText", stop, parent=first),
                    ax(first + 3, "button", "View details", parent=first, backend=button)]
        nodes = [ax(1, "RootWebArea", "Flights", children=[2, 3, 10, 20], backend=1),
                 ax(2, "radio", "One-way", parent=1, backend=200, checked="true"),
                 ax(3, "StaticText", "One-way", parent=1)]
        nodes += card(10, "IndiGo", "Direct", 300) + card(20, "IndiGo", "Direct", 301)
        content, _ = render_ax(nodes)
        self.assertEqual(content.splitlines(), [
            '[1] radio "One-way" checked',                  # its label is not repeated under it
            "IndiGo Direct", '[2] button "View details"',
            "IndiGo Direct", '[3] button "View details"'])   # each result keeps its airline and stops

    def test_footer_is_one_line_and_long_links_lose_their_tracking(self):
        nodes = [ax(1, "RootWebArea", "Shop", children=[2, 3], backend=1),
                 ax(2, "link", "Deals", parent=1, backend=20,
                    url="https://shop.example/deals/index.html?label=gen173nr-10CAEoggI46AdIM1gEaGyIAQGYATO4AQfIAQzYAQPo&aid=304142"),
                 ax(3, "contentinfo", parent=1, children=[4, 5], backend=3),
                 ax(4, "link", "Careers", parent=3, backend=40, url="https://shop.example/careers"),
                 ax(5, "link", "Press", parent=3, backend=50, url="https://shop.example/press")]
        content, _ = render_ax(nodes, "https://shop.example/")
        self.assertEqual(content.splitlines(), ['[1] link "Deals" -> /deals/index.html?…',
                                                "[footer with 2 links and buttons; browser_find searches it]"])
        whole, _ = render_ax(nodes, "https://shop.example/", whole=True)
        self.assertIn('link "Careers" -> /careers', whole)

    def test_a_containers_name_can_be_its_content(self):
        nodes = [ax(1, "RootWebArea", "Flights", children=[2], backend=1),
                 ax(2, "listitem", parent=1, children=[3, 4], backend=2),
                 ax(3, "StaticText", "SpiceJet", parent=2),
                 ax(4, "region", "INR9,995", parent=2, children=[5], backend=4),
                 ax(5, "StaticText", "9,995", parent=4)]           # only repeats the name
        content, _ = render_ax(nodes)
        self.assertEqual(content.splitlines(), ["SpiceJet", "INR9,995"])

    def test_numbers_stay_with_their_elements(self):
        numbers = {"next": 1, "ids": {}}
        render_ax(SHOP_AX, "https://shop.example/", numbers=numbers)
        # A suggestion list opens above the button: old elements keep their numbers, new ones get new numbers.
        opened = SHOP_AX[:13] + [ax(13, "option", "Rye sourdough", parent=2, backend=140)] + SHOP_AX[13:]
        opened[1] = ax(2, "generic", parent=1, children=[3, 4, 5, 8, 9, 13, 10, 11, 12], ignored=True, backend=2)
        content, refs = render_ax(opened, "https://shop.example/", numbers=numbers)
        self.assertIn('[6] option "Rye sourdough"\n[3] textbox "Password"', content)
        self.assertEqual({ref: info["backend"] for ref, info in refs.items()},
                         {1: 108, 2: 109, 6: 140, 3: 110, 4: 111, 5: 112})

    def test_page_changes(self):
        before = ["# Flights", '[1] button "From"', '[2] combobox "To" value=""', '[3] button "Search"']
        after = ["# Flights", '[1] button "From"', '[2] combobox "To" value="Del"', '[7] option "Delhi (DEL)"',
                 '[3] button "Search"']
        self.assertEqual(page_changes(before, after).splitlines(), [
            '  [1] button "From"', '- [2] combobox "To" value=""', '+ [2] combobox "To" value="Del"',
            '+ [7] option "Delhi (DEL)"'])
        self.assertEqual(page_changes(before, before), "")
        results = ["# Results"] + [f"[{n}] link \"Flight {n} from Bengaluru to Delhi\" -> /flight/{n}" for n in range(60)]
        self.assertIsNone(page_changes(before, results))          # a new page is sent whole
        self.assertIn("+ [7] option", page_changes(before + results, after + results))   # small news, big page

    def long_page(self):
        """A page with a long menu, then its main content (a heading and 80 paragraphs), then a button."""
        nodes = [ax(1, "RootWebArea", "News", children=[2, 3, 4], backend=1),
                 ax(2, "navigation", "Sections", parent=1, children=list(range(100, 130)), backend=2),
                 ax(3, "main", parent=1, children=[5] + list(range(200, 280)), backend=3),
                 ax(5, "heading", "Top story", parent=3, children=[6], backend=5, level=1),
                 ax(6, "StaticText", "Top story", parent=5),
                 ax(4, "button", "Load more stories", parent=1, backend=400)]
        nodes += [ax(100 + k, "link", f"Section {k}", parent=2, backend=1000 + k, url=f"https://news.example/s/{k}")
                  for k in range(30)]
        for k in range(80):
            nodes += [ax(200 + k, "paragraph", parent=3, children=[3000 + k], backend=2000 + k),
                      ax(3000 + k, "StaticText", f"Paragraph {k} of the story, with enough words to take room.",
                         parent=200 + k)]
        return nodes

    def test_landmarks_are_marked(self):
        marks = {}
        content, _ = render_ax(self.long_page(), "https://news.example/", marks=marks)
        lines = content.splitlines()
        (start, end, name), = marks["navigation"]
        self.assertEqual((name, lines[start], end - start), ("Sections", '[1] link "Section 0" -> /s/0', 30))
        (start, end, _), = marks["main"]
        self.assertEqual((lines[start], lines[end - 1]), ("# Top story", "Paragraph 79 of the story, with enough "
                                                                         "words to take room."))
        self.assertEqual(render_ax(self.long_page(), "https://news.example/")[0], content)   # marks change nothing

    def test_a_long_page_is_fitted_main_content_first(self):
        marks = {}
        content, _ = render_ax(self.long_page(), "https://news.example/", marks=marks)
        lines = content.splitlines()
        self.assertEqual(fit(lines, 100_000, marks), (content, len(lines)))          # a page that fits is as it is
        text, end = fit(lines, 2000, marks)
        shown = text.splitlines()
        # The menu is one line with its first links; the main content follows it at once.
        self.assertEqual(shown[0], '[menu "Sections" with 30 links and buttons (Section 0, Section 1, Section 2, …); '
                                   'browser_find searches it]')
        self.assertEqual(shown[1], "# Top story")
        self.assertLessEqual(len(text), 2000 + 600)
        self.assertIn("more characters with 1 element: browser_snapshot more=true reads on", text)
        self.assertEqual(shown[-1], 'Controls further down: [31] button "Load more stories"')
        self.assertEqual(lines[end - 1], next(line for line in reversed(shown) if line.startswith("Paragraph")))
        # The element last used is shown with the lines around it, though it is past the cut.
        near, _ = fit(lines, 2000, marks, near=len(lines) - 1)
        self.assertIn('lines left out]\nParagraph 77', near)
        self.assertTrue(near.endswith('Paragraph 79 of the story, with enough words to take room.\n'
                                      '[31] button "Load more stories"'), near)
        self.assertIn("Paragraph 3 of", near)                   # the main content still comes first
        # With no landmarks the page is read from its top, the menu as it is.
        plain, _ = fit(lines, 2000)
        self.assertTrue(plain.startswith('[1] link "Section 0"'))


class StubJudge:
    """Stands in for the decision model: in these test pages, these words mark committing steps."""

    WORDS = ("buy", "pay", "order", "confirm", "send")

    def __init__(self):
        self.states = []
        self.fail = False

    async def commits(self, state):
        self.states.append(state)
        return self.fail or any(word in state.lower() for word in self.WORDS)


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
        self.leave_open = set()    # kinds of approval nobody answers (an owner away from the app)
        self.while_asking = None   # runs while an approval is pending, to let tests change the page
        self.policy.add_listener(self._answer)
        self.launcher = FakeLauncher()
        self.judge = StubJudge()
        self.broker = BrowserBroker(self.config, self.policy, self.audit, launcher=self.launcher, decider=self.judge)
        self.broker.blank_wait = 0          # fake pages are drawn by the test after navigating
        self.dns = {}

        async def resolve(host):
            return self.dns.get(host, ["93.184.216.34"])
        self.broker._resolve = resolve

    def _answer(self, approval):
        self.asked.append((approval.kind, approval.summary))
        self.approvals.append(approval)
        if self.while_asking:
            self.while_asking()
        if approval.kind in self.leave_open:
            return
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

    def ref(self, name):
        """An element's number: numbers continue across documents, so a stale one never hits a new element."""
        return next(n for n, info in self.broker._refs[self.broker._active].items() if info["name"] == name)

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
        # The model words its answer from this: the step is the owner's to take, with no policy or kind names.
        self.assertIn("needs the owner", result["error"])
        self.assertNotIn("browser.navigate", result["error"])
        self.assertNotIn("denied by policy", result["error"])
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

    async def test_a_fixed_viewport_covers_every_tab_and_is_undone(self):
        self.answers["browser.navigate"] = True
        browser, _ = await self.load_shop()
        # A tab is attached at the window's own size: any size a stopped run left on it is cleared.
        self.assertEqual(len(browser.sent("Emulation.clearDeviceMetricsOverride")), 1)
        async with self.broker.fixed_viewport(1440, 900):
            size = {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False}
            self.assertEqual(browser.sent("Emulation.setDeviceMetricsOverride"), [size])       # the open tab
            opened = await self.broker.open("https://shop.example/")
            self.assertTrue(opened["ok"], opened)
            self.assertEqual(browser.sent("Emulation.setDeviceMetricsOverride"), [size, size])  # and a new one
        self.assertEqual(len(browser.sent("Emulation.clearDeviceMetricsOverride")), 1 + 2)    # both undone
        await self.broker.open("https://shop.example/")
        self.assertEqual(len(browser.sent("Emulation.setDeviceMetricsOverride")), 2)          # not after

    async def test_a_link_copied_as_a_path_opens_on_the_same_site(self):
        self.answers["browser.navigate"] = True
        await self.load_shop()
        opened = await self.broker.open("/pdf/fees.pdf")
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(opened["url"], "https://shop.example/pdf/fees.pdf")
        moved = await self.broker.navigate("/help?q=returns")
        self.assertEqual(moved["url"], "https://shop.example/help?q=returns")

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


class RenumberedTests(BrokerTestCase):
    """An element drawn again gets a new number; a step that names the old one still finds it, where that is
    beyond doubt."""

    PAGE = "https://fares.example/search"

    def page(self, *elements, url=PAGE):
        self.broker._renumbered("T", {number: {"backend": 100 + number, "role": role, "name": name}
                                      for number, role, name in elements}, url)

    def test_a_retired_number_finds_the_one_element_with_its_role_and_name_on_the_same_page(self):
        self.page((11, "button", "Next month"), (12, "button", "Search"), (13, "textbox", "From"))
        self.page((21, "button", "Next month"), (22, "button", "Search"), (13, "textbox", "From"))   # the calendar drew again
        self.assertEqual(self.broker._ref("T", 11)["backend"], 121)
        self.assertEqual(self.broker._ref("T", 12)["backend"], 122)
        self.assertEqual(self.broker._ref("T", 13)["backend"], 113)                # a number still current is itself
        said = [e for e in self.audit.tail() if e["event"] == "browser.rebound"]
        self.assertEqual([(e["ref"], e["label"]) for e in said], [(11, "button Next month"), (12, "button Search")])
        self.page((31, "button", "Next month"), (32, "button", "Search"), (13, "textbox", "From"))   # and again
        self.assertEqual(self.broker._ref("T", 11)["backend"], 131)                # still the same button

    def test_it_is_not_guessed_where_two_would_do_none_is_left_or_the_page_is_another(self):
        self.page((11, "button", "Select"), (12, "button", "Search"), (14, "link", ""))
        self.page((21, "button", "Select"), (23, "button", "Select"), (24, "link", ""))
        for number in (11, 12, 14, 99, "x"):            # two candidates; gone; no name to go by; never seen; not a number
            with self.assertRaisesRegex(BrowserError, "Unknown element"):
                self.broker._ref("T", number)
        # The same button's name on another address is another page's button.
        self.page((41, "button", "Search"))
        self.page((51, "button", "Search"), url="https://fares.example/results?from=BLR")
        with self.assertRaisesRegex(BrowserError, "Unknown element"):
            self.broker._ref("T", 41)
        # A new document forgets the old numbers altogether.
        self.page((61, "button", "Pay now"), url=self.PAGE)
        self.page((71, "button", "Pay now"), url=self.PAGE)
        self.broker._targets["S"] = "T"
        self.broker._on_navigated({"frame": {"id": "T", "url": self.PAGE}}, "S")
        self.page((81, "button", "Pay now"), url=self.PAGE)
        with self.assertRaisesRegex(BrowserError, "Unknown element"):
            self.broker._ref("T", 61)

    def test_a_number_missing_a_digit_is_named_not_taken(self):
        self.page((639, "button", "Try now"), (12, "link", "Home"))
        with self.assertRaisesRegex(BrowserError, r"did you mean \[639\]\?"):
            self.broker._ref("T", 39)
        self.page((639, "button", "Try now"), (739, "button", "Later"))
        with self.assertRaisesRegex(BrowserError, "take a new snapshot"):    # two would do: no guess
            self.broker._ref("T", 39)


class OwnTabsTests(BrokerTestCase):
    """Two tasks at once each have their own tabs: neither sees, acts on or closes the other's."""

    def as_task(self, conversation):
        return current_run.set(SimpleNamespace(conversation=conversation, content_tainted=False,
                                               private_sources=set(), shared=set()))

    async def test_each_task_keeps_to_its_own_tabs(self):
        from coderunner.browser import broker as broker_module
        a = self.as_task("app:a")
        await self.broker.open("https://fares.example/")
        browser = self.launcher.browser
        tab_a = self.broker._active
        current_run.reset(a)
        on_its_computer = mock.patch.object(type(self.broker), "mode", new_callable=mock.PropertyMock, return_value="computer")
        on_its_computer.start()
        self.addCleanup(on_its_computer.stop)
        b = self.as_task("app:b")
        await self.broker.open("https://shop.example/")
        tab_b = self.broker._active
        self.assertNotEqual(tab_a, tab_b)
        # The second task's first tab on the assistant's computer opened in a window of its own.
        made = browser.sent("Target.createTarget")
        self.assertEqual([bool(m.get("newWindow")) for m in made], [False, True])
        # It does not see the other's tab, cannot name it, and its "current tab" is its own.
        listed = (await self.broker.tabs())["tabs"]
        self.assertNotIn("https://fares.example/", [t["url"] for t in listed])
        other = self.broker._tab_ids[tab_a]
        self.assertIn("No open tab", (await self.broker.snapshot(tab=other))["error"])
        self.assertEqual((await self.broker.snapshot())["url"], "https://shop.example/")
        # Opening more pages closes its own oldest, never the other task's.
        for n in range(broker_module.MAX_TABS + 2):
            await self.broker.open(f"https://shop.example/p{n}")
        self.assertIn(tab_a, browser.pages)
        current_run.reset(b)
        a = self.as_task("app:a")
        self.assertEqual(self.broker._active, tab_a)                    # the first task is still where it was
        self.assertEqual((await self.broker.snapshot())["url"], "https://fares.example/")
        current_run.reset(a)
        # Finished: its older tabs close, the one it ended on stays for a follow-up, and it is nobody's now.
        await self.broker.tidy(scope="app:b")
        self.assertEqual([t for t, owner in self.broker._tab_owner.items() if owner == "app:b"], [])
        b = self.as_task("app:b")
        ended_on = self.broker._last["app:b"]
        self.assertEqual((await self.broker.snapshot())["url"], browser.pages[ended_on]["url"])
        current_run.reset(b)
        # A page picture is of the task's own tab, whoever asks.
        self.assertEqual((await self.broker.peek(scope="app:a"))[1], "https://fares.example/")

    async def test_two_tasks_on_one_site_do_not_step_at_the_same_moment(self):
        from coderunner.browser import broker as broker_module
        waited = []

        async def nap(seconds):
            waited.append(round(seconds, 1))
        a = self.as_task("app:a")
        await self.broker._courtesy("shop.example")
        current_run.reset(a)
        b = self.as_task("app:b")
        with mock.patch.object(broker_module.asyncio, "sleep", nap):
            await self.broker._courtesy("shop.example")                 # right after the other task's step there
            await self.broker._courtesy("shop.example")                 # its own next step: no wait
            await self.broker._courtesy("elsewhere.example")            # another site: no wait
        current_run.reset(b)
        self.assertEqual(len(waited), 1)
        self.assertTrue(0 < waited[0] <= broker_module.SAME_SITE_GAP)


class TabTidyTests(BrokerTestCase):
    async def test_the_assistants_older_tabs_close_as_new_ones_open_and_after_a_task(self):
        from coderunner.browser import broker as broker_module
        for n in range(8):                                  # eight pages opened one after another, none closed
            await self.broker.open(f"https://site{n}.example/")
        browser = self.launcher.browser
        open_now = set(browser.pages)
        # The oldest of its own were closed as it went; the owner's first tab is not its to close.
        self.assertLessEqual(len(open_now & self.broker._opened), broker_module.MAX_TABS)
        self.assertTrue(browser.sent("Target.closeTarget"))
        self.assertIn("P1", open_now)
        active = self.broker._active
        self.assertGreater(await self.broker.tidy(), 0)
        self.assertLessEqual(len(set(browser.pages) & self.broker._opened), broker_module.TABS_AFTER_TASK)
        self.assertIn(active, browser.pages)                # the page it ended on stays
        self.assertIn("P1", browser.pages)
        self.assertNotIn("P1", [p["targetId"] for p in browser.sent("Target.closeTarget")])
        self.assertTrue([e for e in self.audit.tail(80) if e["event"] == "browser.tabs_closed"])


class RecoveryTests(BrokerTestCase):
    """A browser that fails once is brought back below the model: nothing of it reaches the task's thread."""

    async def test_the_sandbox_computer_is_brought_back_when_its_browser_cannot_be_reached(self):
        from unittest import mock
        from coderunner.browser.cdp import CDPError
        self.config.set("browser.mode", "computer")
        real, tries = self.launcher.launch, []

        async def launch(*args):
            tries.append(1)
            if len(tries) == 1:
                raise CDPError("could not reach the sandbox's browser")
            return await real(*args)
        self.launcher.launch = launch
        self.broker.computer = mock.Mock(ensure=mock.AsyncMock(return_value=True))
        self.assertTrue((await self.broker.start())["ok"])
        self.broker.computer.ensure.assert_awaited_once_with(desktop=True, timeout=600)
        self.assertEqual(len(tries), 2)
        await self.broker.stop()
        # Not brought back: the task is told the browser did not start, after one try.
        tries.clear()
        self.broker.computer.ensure = mock.AsyncMock(return_value=False)
        self.assertFalse((await self.broker.start())["ok"])
        self.assertEqual(len(tries), 1)

    async def test_a_page_read_that_fails_once_is_done_again_and_the_model_sees_only_the_page(self):
        from coderunner.browser.cdp import CDPError
        await self.broker.open("https://shop.example/")
        real, calls = self.broker._snapshot, []
        recovered = []

        async def recover():
            recovered.append(True)
            return True
        self.broker._recover = recover

        async def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise CDPError("Page.enable timed out after 30s")
            return await real(*args, **kwargs)
        self.broker._snapshot = flaky
        result = await self.broker.snapshot()
        self.assertEqual((result["ok"], len(calls), recovered), (True, 2, [True]))
        events = [e["event"] for e in self.audit.tail(20)]
        self.assertIn("browser.fault", events)
        self.assertIn("browser.recovered", events)
        self.assertNotIn("timed out", json.dumps(result))

    async def test_a_click_is_not_repeated_and_what_the_model_is_told_names_no_plumbing(self):
        from coderunner.browser import broker as broker_module
        from coderunner.browser.cdp import CDPError
        await self.broker.open("https://shop.example/")
        tried = []

        async def recover():
            return True
        self.broker._recover = recover

        async def dead(*args, **kwargs):
            tried.append(1)
            raise CDPError("Input.dispatchMouseEvent timed out after 30s")
        self.broker._click = dead
        result = await self.broker.click(1)
        self.assertEqual((result["ok"], len(tried)), (False, 1))                # not clicked twice behind anyone's back
        self.assertEqual(result["error"], "The page stopped responding and was reopened. Look at it again before going on.")
        # Nothing brought it back: the fact, in the owner's terms.
        self.broker._recover = lambda: asyncio.sleep(0, False)
        self.broker._snapshot = dead
        gone = await self.broker.snapshot()
        self.assertEqual(gone["error"], "The page did not respond in time.")
        for text in (result["error"], gone["error"], broker_module.plainly(CDPError("Target closed")),
                     broker_module.plainly(CDPError("Protocol error (Runtime.evaluate)"))):
            self.assertNotRegex(text, r"Page\.|Input\.|Runtime|CDP|timed out after|\d+s\b")

    async def test_a_crashed_tab_is_closed_at_once_and_a_stuck_browser_is_started_again(self):
        await self.broker.open("https://shop.example/")
        browser = self.launcher.browser
        target = self.broker._active
        self.broker._on_crashed(target)                     # Chrome's "Aw, Snap": told by the browser, not waited for
        await asyncio.sleep(0.05)
        self.assertIn(target, [p["targetId"] for p in browser.sent("Target.closeTarget")])
        self.assertIsNone(self.broker._active)
        self.assertIn("browser.tab_crashed", [e["event"] for e in self.audit.tail(20)])
        # The whole browser not answering: its own way of starting again is used, then a new tab is tried.
        restarted = []

        async def restart():
            restarted.append(True)
        self.broker.restart = restart
        launches = len(self.launcher.browsers)
        self.assertTrue(await self.broker._recover())
        self.assertGreater(len(self.launcher.browsers), launches)
        self.assertEqual([e["how"] for e in self.audit.tail(20) if e["event"] == "browser.brought_back"], ["reconnect"])
        self.assertEqual(restarted, [])                     # the gentler step was enough


class InteractionPolicyTests(BrokerTestCase):
    browser_config = {"allow": ["shop.example"]}
    rules = [{"kind": "browser.interact", "target": "shop.example", "decision": "allow"}]

    async def test_a_click_that_changes_nothing_says_what_it_landed_on(self):
        browser, _ = await self.load_shop()
        result = await self.broker.click(self.ref("Read more"))
        self.assertEqual(result["changes"], "No visible change on the page.")
        self.assertEqual(result["note"], 'The click landed on link "Read more". '
                                         'The link did not open its address (/about); browser_open opens it.')
        browser.pages[self.broker._active]["url"] = "https://shop.example/hotels"
        result = await self.broker.click(self.ref("Read more"))         # a late redirect the model has not read
        self.assertIn("address changed since you read it", result["note"])

    async def test_a_blank_page_gets_time_to_show_text(self):
        self.broker.blank_wait = 5

        async def draw_later():
            await asyncio.sleep(0.6)            # a bot check that hands over to the real page
            self.launcher.browser.ax = SHOP_AX
        drawing = asyncio.get_running_loop().create_task(draw_later())
        opened = await self.broker.navigate("https://shop.example/")
        await drawing
        self.assertIn('[4] button "Buy now"', opened["content"])
        self.broker.blank_wait = 0.2
        self.launcher.browser.ax = []
        blank = await self.broker.navigate("https://shop.example/empty")
        self.assertIn("no text yet", blank["note"])

    async def test_snapshot_change_detection(self):
        browser, first = await self.load_shop()
        self.assertTrue(first["changed"])
        self.assertIn('[4] button "Buy now"', first["content"])
        self.assertIn("(password field", first["content"])
        self.assertFalse((await self.broker.snapshot())["changed"])
        browser.ax = SHOP_AX[:-1]
        self.assertTrue((await self.broker.snapshot())["changed"])
        short = await self.broker.snapshot(max_chars=100)
        self.assertIn("more characters with 2 elements: browser_snapshot more=true reads on", short["content"])
        self.assertIn('Controls further down: [3] textbox "Password"', short["content"])   # what is left to act on

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

    async def test_a_click_goes_beside_a_control_laid_over_the_middle(self):
        browser, _ = await self.load_shop()
        browser.hit_at = lambda x, y: 555 if (x, y) == (60, 30) else None    # a play button on a poster link
        browser.labels[555] = {"tag": "button", "text": "Play trailer", "submit": ""}
        result = await self.broker.click(self.ref("Read more"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.asked, [])
        self.assertNotIn("covered", result.get("note", ""))
        press = [p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"][-1]
        self.assertEqual((press["x"], press["y"]), (60, 24))

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

    async def test_a_captcha_box_is_never_typed_into(self):
        browser, _ = await self.load_shop()
        browser.nodes[109] = {"nodeName": "INPUT", "attributes": ["type", "text", "id", "fcaptcha_code",
                                                                   "placeholder", "Enter Captcha"]}
        result = await self.broker.type(2, "x7k2m9")
        self.assertFalse(result["ok"])
        self.assertIn("handoff", result["error"])
        self.assertIn("owner's step", result["error"])
        self.assertEqual(browser.sent("Input.insertText"), [])
        self.assertIn("browser.captcha_left", [e["event"] for e in self.audit.tail()])

    async def test_a_scrolled_page_is_hit_tested_where_the_click_lands(self):
        browser, _ = await self.load_shop()
        browser.scroll = (0, 5550)              # a results page's "Show more", reached by scrolling
        result = await self.broker.click(self.ref("Read more"))
        self.assertTrue(result["ok"], result)
        hit = browser.sent("DOM.getNodeForLocation")[-1]
        self.assertEqual((hit["x"], hit["y"]), (60, 30 + 5550))     # page coordinates for Chrome's hit test
        press = [p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"][-1]
        self.assertEqual((press["x"], press["y"]), (60, 30))        # window coordinates for the mouse

    async def test_typing_picks_the_suggestion_that_matches(self):
        browser, _ = await self.load_shop()
        handle = browser.handle
        suggestions = [ax(40, "listbox", "Going to suggestions", parent=2, children=[41, 42], backend=140),
                       ax(41, "option", "New Delhi Railway Station", parent=40, backend=141),
                       ax(42, "option", "Delhi (DEL) · Indira Gandhi International", parent=40, backend=142)]

        def typed(method, params, sid):
            if method == "Input.insertText":
                browser.ax = SHOP_AX[:1] + [dict(SHOP_AX[1], childIds=SHOP_AX[1]["childIds"] + ["40"])] \
                    + SHOP_AX[2:] + suggestions
            return handle(method, params, sid)
        browser.handle = typed
        result = await self.broker.type(2, "delhi", pick=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["picked"], "Delhi (DEL) · Indira Gandhi International")    # starts with it
        self.assertEqual(self.asked, [])
        clicks = [e for e in self.audit.tail() if e["event"] == "browser.click"]
        self.assertEqual(clicks[-1]["ref"], self.ref("Delhi (DEL) · Indira Gandhi International"))
        # The model's view covers the typing too: the suggestions it never saw are part of the change.
        self.assertIn("Delhi (DEL)", result.get("changes", "") + result.get("content", ""))

    async def test_typing_without_suggestions_or_with_submit_picks_nothing(self):
        browser, _ = await self.load_shop()
        result = await self.broker.type(2, "rye bread", pick=True)
        self.assertTrue(result["ok"], result)
        self.assertNotIn("picked", result)
        result = await self.broker.type(2, "rye", submit=True, pick=True)
        self.assertNotIn("picked", result)
        self.assertEqual([e for e in self.audit.tail() if e["event"] == "browser.click"], [])

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

    async def test_a_point_on_a_screenshot_is_clicked_as_the_element_there(self):
        browser, _ = await self.load_shop()
        browser.hit = 111                                   # "Buy now": the judge's say still applies
        self.assertFalse((await self.broker.click_at(60, 30))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        browser.parents = {126: 109, 109: 2, 4: 2, 2: 1}   # the inner text of the search box
        browser.hit = 126
        clicked = await self.broker.click_at(60, 30)
        self.assertTrue(clicked["ok"], clicked)
        clicks = [e for e in self.audit.tail() if e["event"] == "browser.click_at"]
        self.assertIn("Search", clicks[-1]["clicked"])                   # the search box, found from its text
        pressed = browser.sent("Input.dispatchMouseEvent")[-2]
        self.assertEqual((pressed["type"], pressed["x"], pressed["y"]), ("mousePressed", 60, 30))  # where pointed
        # A paragraph, which nothing numbers: clicked where the model pointed, as a mouse would, through the
        # same checks, and described by what is there.
        browser.hit = 4
        browser.labels[4] = {"tag": "p", "text": "Fresh bread every morning.", "submit": ""}
        clicked = await self.broker.click_at(70, 40)
        self.assertTrue(clicked["ok"], clicked)
        pressed = browser.sent("Input.dispatchMouseEvent")[-2]
        self.assertEqual((pressed["type"], pressed["x"], pressed["y"]), ("mousePressed", 70, 40))
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        clicks = [e for e in self.audit.tail() if e["event"] == "browser.click_at"]
        self.assertIn('p "Fresh bread every morning."', clicks[-1]["clicked"])

    async def test_a_point_on_a_footer_link_finds_it(self):
        browser, _ = await self.load_shop()
        footer = [ax(13, "contentinfo", parent=2, children=[14], backend=13),
                  ax(14, "link", "Careers", parent=13, backend=140, url="https://shop.example/careers")]
        root = dict(SHOP_AX[1], childIds=SHOP_AX[1]["childIds"] + ["13"])
        browser.ax = [SHOP_AX[0], root] + SHOP_AX[2:] + footer      # the footer is one line in page text
        browser.hit = 140
        clicked = await self.broker.click_at(60, 30)
        self.assertTrue(clicked["ok"], clicked)
        clicks = [e for e in self.audit.tail() if e["event"] == "browser.click_at"]
        self.assertIn("Careers", clicks[-1]["clicked"])

    async def test_typing_goes_to_the_focused_field_with_the_usual_checks(self):
        browser, _ = await self.load_shop()
        self.assertIn("No text field has the focus", (await self.broker.type_focused("rye"))["error"])
        browser.focused = 109
        typed = await self.broker.type_focused("sourdough")
        self.assertTrue(typed["ok"], typed)
        self.assertEqual(browser.sent("Input.insertText")[-1]["text"], "sourdough")
        browser.focused = 110                               # a password field: never typed
        self.assertFalse((await self.broker.type_focused("hunter2"))["ok"])

    async def test_press(self):
        browser, _ = await self.load_shop()
        self.assertFalse((await self.broker.press("F5"))["ok"])
        self.assertTrue((await self.broker.press("Tab"))["ok"])
        browser.focused = 111
        self.assertFalse((await self.broker.press("Enter"))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        scrolled = await self.broker.scroll("down", 400)
        self.assertTrue(scrolled["ok"], scrolled)
        self.assertEqual((scrolled["scrolled"], scrolled["position"]), (True, "20% of the way down"))
        script = [c for c in browser.sent("Runtime.callFunctionOn") if c["functionDeclaration"] == SCROLL_JS][-1]
        self.assertEqual([a["value"] for a in script["arguments"]], [0, 400])
        self.assertEqual(browser.sent("Input.dispatchMouseEvent")[-1:], [])   # no wheel events: they can hang

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
        # Pages act as if focused: pop-ups ignored clicks whenever the owner worked in another window.
        self.assertIn({"enabled": True}, browser.sent("Emulation.setFocusEmulationEnabled"))
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


class BrowserToolTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_changes_say_how_many_and_a_snapshot_reads_on(self):
        from coderunner.agent.tools import browser_tools
        asked = []

        class Broker:
            async def click(self, ref, tainted=False):
                return {"ok": True, "url": "https://a.example/", "changes": '  [1] button "Go"\n+ Two\n+ Three\n- One'}

            async def snapshot(self, tainted=False, **options):
                asked.append(options)
                return {"ok": True, "url": "https://a.example/", "content": "Later lines", "changed": True}

            async def items(self, **options):
                return {"ok": True, "total": 0}
        tools = {tool.name: tool for tool in browser_tools(SimpleNamespace(browser=Broker(), vision=False, driver=None))}
        ctx = SimpleNamespace(tainted=False)
        clicked = await tools["browser_click"].handler({"ref": 1}, ctx)
        self.assertIn("Changes on the page (2 added, 1 removed; + added", clicked)
        self.assertIn("Later lines", await tools["browser_snapshot"].handler({"more": True}, ctx))
        await tools["browser_snapshot"].handler({"start": "Price", "more": True}, ctx)
        self.assertEqual(asked, [{"more": True}, {"start": "Price"}])        # start= says where, more is left out


class ReadinessTests(BrokerTestCase):
    browser_config = {"allow": ["shop.example"]}
    rules = [{"kind": "browser.interact", "target": "shop.example", "decision": "allow"}]

    async def test_a_page_is_read_once_parsed(self):
        browser, _ = await self.load_shop()
        browser.lifecycle = ("DOMContentLoaded",)           # its pictures never finish loading
        began = time.monotonic()
        opened = await self.broker.navigate("https://shop.example/slow")
        self.assertTrue(opened["ok"] and opened["loaded"], opened)
        self.assertLess(time.monotonic() - began, 2)
        sid = await self.broker._session(self.broker._active)
        self.assertIn(("Network.enable", sid), [(method, s) for method, _, s in browser.calls])

    async def test_open_requests_are_waited_for_and_long_ones_are_not(self):
        browser, _ = await self.load_shop()
        target = self.broker._active
        sid = await self.broker._session(target)

        async def timed(action=""):
            began = time.monotonic()
            await self.broker._ready(target, sid, action)
            return time.monotonic() - began
        self.assertLess(await timed(), 0.3)                       # nothing open: read at once
        def request(rid, kind, url="https://api.shop.example/search", frame=target):
            browser.emit("Network.requestWillBeSent", {"requestId": rid, "type": kind, "frameId": frame,
                                                       "request": {"url": url},
                                                       "documentURL": "https://shop.example/"}, sid)
        request("r1", "XHR")
        request("r2", "Image")                                    # brings no text
        request("ad", "Fetch", "https://ads.example/auction")     # another site's
        request("frame", "XHR", frame="F2")                       # a frame's in the page
        request("blob", "Script", "blob:https://shop.example/1")
        loop = asyncio.get_running_loop()
        loop.call_later(0.6, browser.emit, "Network.loadingFinished", {"requestId": "r1"}, sid)
        waited = await timed()
        self.assertGreater(waited, 0.6)                           # until it ended
        self.assertLess(waited, broker_module.SETTLE_MOST)
        self.broker._requests[sid].open["poll"] = time.monotonic() - broker_module.LONG_REQUEST - 1   # a long poll
        self.assertLess(await timed(), 0.3)
        request("stream", "Fetch")                                # an answer that keeps coming
        browser.emit("Network.responseReceived", {"requestId": "stream"}, sid)
        await asyncio.sleep(0)
        self.broker._requests[sid].answers["stream"] -= broker_module.STREAMING
        self.assertLess(await timed(), 0.3)
        request("r3", "Fetch")                                    # one that never ends
        await asyncio.sleep(0)
        with mock.patch.object(broker_module, "SETTLE_MOST", 0.5):
            self.assertLess(await timed(), 0.8)                   # the limit still holds

    async def test_a_long_page_is_read_on_with_more(self):
        browser, _ = await self.load_shop()
        lines = [f"Result {n} with a long description of what it is" for n in range(400)]
        browser.ax = [ax(1, "RootWebArea", "Results", children=[100 + k for k in range(len(lines))], backend=1)]
        for k, line in enumerate(lines):
            browser.ax += [ax(100 + k, "paragraph", parent=1, children=[5000 + k], backend=1000 + k),
                           ax(5000 + k, "StaticText", line, parent=100 + k)]
        first = await self.broker.snapshot()
        self.assertLess(len(first["content"]), self.broker.page_chars + 400)
        last = first["content"].split("\n")[-2]
        later = await self.broker.snapshot(more=True)
        self.assertTrue(later["content"].startswith(f"… [{int(last.split()[1]) + 1} lines above left out]\n"
                                                    f"Result {int(last.split()[1]) + 1} with"), later["content"][:200])
        while "more=true reads on" in later["content"]:
            later = await self.broker.snapshot(more=True)
        self.assertIn("Result 399 with", later["content"])
        self.assertIn("end of the page", (await self.broker.snapshot(more=True))["content"])

    async def test_more_changes_than_fit_come_as_the_page(self):
        browser, _ = await self.load_shop()

        def page(extra):
            texts = [f"Line {n} that stays as it was on the page" for n in range(100)] + extra
            nodes = SHOP_AX + [node for n, text in enumerate(texts) for node in (
                ax(600 + n, "paragraph", parent=2, children=[3000 + n], backend=600 + n),
                ax(3000 + n, "StaticText", text, parent=600 + n))]
            nodes[1] = ax(2, "generic", parent=1, children=[3, 4, 5, 8, 9, 10, 11, 12] + [600 + n for n in range(len(texts))],
                          ignored=True, backend=2)
            return nodes
        browser.ax = page([])
        await self.broker.snapshot()
        browser.ax = page([f"New line {n} of a filter panel that opened" for n in range(30)])
        small = await self.broker.click(self.ref("Read more"))
        self.assertIn("+ New line 29", small["changes"])          # changes that fit come as changes
        browser.ax = page([f"Another line {n} of a filter panel that opened" for n in range(30)])
        self.broker.page_chars = 1000
        large = await self.broker.click(self.ref("Read more"))
        self.assertNotIn("changes", large)
        self.assertIn("more=true reads on", large["content"])


class OwnerTurnTests(BrokerTestCase):
    """The owner taking the screen pauses a task's browser steps; hand-offs are rechecked and reported."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        from coderunner.agent.loop import RunContext
        self.said = []
        self.ctx = RunContext(conversation="", events=self.said.append, deadline=time.monotonic() + 100,
                              stop_at=time.monotonic() + 200)
        self.run_token = current_run.set(self.ctx)
        for name, value in (("PAUSE_TICK", 0.05), ("HANDOFF_RECHECK", 0.05)):
            patcher = mock.patch.object(broker_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        current_run.reset(self.run_token)
        await super().asyncTearDown()

    def kinds_said(self):
        return [e["type"] for e in self.said]

    def audited(self, event):
        return [json.loads(line) for line in (self.tmp / "audit.jsonl").read_text().splitlines()
                if json.loads(line).get("event") == event]

    async def test_a_step_waits_while_the_owner_has_control_and_gets_the_page_as_they_left_it(self):
        browser, _ = await self.load_shop()
        buy = self.ref("Buy now")
        deadline, stop_at = self.ctx.deadline, self.ctx.stop_at
        self.broker.set_owner_control(True)
        step = asyncio.create_task(self.broker.click(buy))
        await asyncio.sleep(0.2)
        self.assertFalse(step.done())
        self.assertEqual(self.kinds_said(), ["paused"])
        self.assertEqual(self.said[0]["text"], broker_module.PAUSED)
        # The owner signs in meanwhile: another page, drawn anew.
        browser.pages["P1"] = {"url": "https://shop.example/account", "title": "Your account"}
        browser.ax = [ax(1, "RootWebArea", "Account", children=[2], backend=1),
                      ax(2, "button", "Buy now", parent=1, backend=211)]
        self.broker.set_owner_control(False)
        result = await asyncio.wait_for(step, 2)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["handed_back"])
        self.assertEqual(result["note"], broker_module.HANDED_BACK)
        self.assertEqual(result["url"], "https://shop.example/account")
        self.assertIn('button "Buy now"', result["content"])
        # The step itself was not taken, and nothing was asked about it.
        self.assertFalse([p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"])
        self.assertEqual(self.asked, [])
        # Element numbers read before the owner's turn do not reach the new page.
        self.assertNotIn(buy, self.broker._refs[self.broker._active])
        self.assertEqual(self.kinds_said(), ["paused", "resumed"])
        # The wait is not the task's time: its limit moved on by it.
        self.assertGreaterEqual(self.ctx.deadline - deadline, 0.15)
        self.assertAlmostEqual(self.ctx.deadline - deadline, self.ctx.stop_at - stop_at, places=6)
        self.assertEqual([r["on"] for r in self.audited("browser.owner_control")], [True, False])
        self.assertEqual(self.audited("browser.handed_back")[0]["action"], "click")

    async def test_a_read_is_done_after_the_turn_and_says_so(self):
        await self.load_shop()
        self.broker.set_owner_control(True)
        read = asyncio.create_task(self.broker.snapshot())
        await asyncio.sleep(0.1)
        self.assertFalse(read.done())
        self.broker.set_owner_control(False)
        result = await asyncio.wait_for(read, 2)
        self.assertEqual(result["note"], broker_module.HANDED_BACK_READ)
        self.assertTrue(result["changed"])          # read afresh, not "unchanged since you last saw it"
        self.assertIn("Welcome", result["content"])

    async def test_without_a_task_or_control_nothing_waits(self):
        await self.load_shop()
        self.assertNotIn("note", await self.broker.snapshot())
        self.assertEqual(self.said, [])
        current_run.reset(self.run_token)
        self.run_token = current_run.set(None)
        self.broker.set_owner_control(True)         # the owner's own steps from the app are theirs
        self.assertNotIn("note", await asyncio.wait_for(self.broker.snapshot(), 2))

    async def test_a_turn_nobody_hands_back_lapses(self):
        await self.load_shop()
        self.broker.set_owner_control(True)
        with mock.patch.object(broker_module, "OWNER_TURN_MOST", 0.1):
            result = await asyncio.wait_for(self.broker.snapshot(), 2)
        self.assertFalse(self.broker.owner_control)
        self.assertEqual(result["note"], broker_module.HANDED_BACK_READ)
        self.assertEqual(self.audited("browser.owner_control")[-1]["origin"], "lapsed")

    async def test_a_handoff_is_not_held_by_the_owners_own_turn(self):
        await self.load_shop()
        self.broker.set_owner_control(True)
        self.answers["browser.handoff"] = True
        result = await asyncio.wait_for(self.broker.handoff("Type the CAPTCHA, then click Go."), 2)
        self.assertTrue(result["completed"])
        self.assertNotIn("paused", self.kinds_said())

    async def test_a_handoff_counts_as_done_when_the_captcha_is_gone(self):
        browser, _ = await self.load_shop()
        self.leave_open.add("browser.handoff")
        browser.challenge = True
        deadline = self.ctx.deadline
        waiting = asyncio.create_task(self.broker.handoff("Type the CAPTCHA, then click Go."))
        await asyncio.sleep(0.2)
        self.assertFalse(waiting.done())             # still there: still the owner's
        browser.challenge = False
        result = await asyncio.wait_for(waiting, 2)
        self.assertTrue(result["completed"])
        self.assertEqual(result["note"], broker_module.LOOKS_DONE)
        approval = self.policy.get(self.approvals[0].id)
        self.assertEqual((approval.status, approval.decided_by), ("approved", "page"))
        self.assertGreaterEqual(self.ctx.deadline - deadline, 0.15)
        # A sign-in that leaves for another address is done too, CAPTCHA or not.
        waiting = asyncio.create_task(self.broker.handoff("Sign in, please."))
        await asyncio.sleep(0.15)
        self.assertFalse(waiting.done())
        browser.pages["P1"] = {"url": "https://shop.example/account", "title": "Your account"}
        result = await asyncio.wait_for(waiting, 2)
        self.assertEqual((result["completed"], result["note"]), (True, broker_module.LOOKS_DONE))
        # Skip still ends it at once.
        self.leave_open.clear()
        self.assertFalse((await asyncio.wait_for(self.broker.handoff("2FA code"), 2))["completed"])

    async def test_a_handoff_nobody_answers_is_said_in_the_thread_and_waited_on_longer(self):
        browser, _ = await self.load_shop()
        self.leave_open.add("browser.handoff")
        browser.challenge = True
        self.ctx.found.append("Taj is ₹18,200 a night.")
        with mock.patch.object(broker_module, "HANDOFF_WAIT", 0.1), mock.patch.object(broker_module, "HANDOFF_MOST", 0.4):
            began = time.monotonic()
            result = await asyncio.wait_for(self.broker.handoff("Type the code sent to your phone."), 3)
        self.assertGreaterEqual(time.monotonic() - began, 0.4)
        self.assertFalse(result["completed"])
        interim = [e["text"] for e in self.said if e["type"] == "interim"]
        self.assertEqual(len(interim), 1)
        self.assertTrue(interim[0].startswith("I'm still waiting for you on the computer: Type the code sent to your phone."))
        self.assertIn("Taj is ₹18,200 a night.", interim[0])
        self.assertEqual(self.policy.get(self.approvals[0].id).status, "expired")


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
        self.assertTrue((await self.broker.click(self.ref("Read more"), tainted=True))["ok"])
        self.assertTrue((await self.broker.type(self.ref("Search"), "sourdough", tainted=True))["ok"])
        self.assertEqual(self.asked, [])
        self.assertFalse((await self.broker.click(self.ref("Buy now")))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive"])

    async def test_ordinary_clicks_wait_for_overlays_to_settle(self):
        browser, _ = await self.load_shop()
        browser.labels[777] = {"tag": "div", "text": "Sign in, save money", "submit": ""}
        moves = []

        def overlay_fades():                     # covers the link on the first hover, then fades out
            moves.append(1)
            browser.hit = 777 if len(moves) == 1 else None
        browser.on_move = overlay_fades
        with mock.patch("coderunner.browser.broker.CLICK_SETTLE", 0):
            self.assertTrue((await self.broker.click(1))["ok"])
        self.assertEqual(self.asked, [])
        self.assertEqual(len([p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"]), 1)
        # A page that never settles is not clicked.
        browser.on_move = lambda: setattr(browser, "hit", 777 if browser.hit is None else None)
        with mock.patch("coderunner.browser.broker.CLICK_SETTLE", 0):
            self.assertFalse((await self.broker.click(1))["ok"])

    async def test_the_judge_decides_which_steps_commit(self):
        browser, _ = await self.load_shop()
        self.assertTrue((await self.broker.click(1))["ok"])             # "Read more"
        self.assertIn("Site: shop.example", self.judge.states[-1])
        self.assertIn('click link "Read more"', self.judge.states[-1])
        self.assertTrue((await self.broker.type(2, "rye"))["ok"])
        self.assertIn('Text to type: "rye"', self.judge.states[-1])
        self.assertEqual(self.asked, [])
        self.assertFalse((await self.broker.click(4))["ok"])            # "Buy now"
        self.assertEqual(self.kinds(), ["browser.sensitive"])
        # Keys other than Enter cannot submit anything and are not judged.
        judged = len(self.judge.states)
        self.assertTrue((await self.broker.press("Tab"))["ok"])
        self.assertEqual(len(self.judge.states), judged)

    async def test_a_plain_link_to_another_page_is_not_judged(self):
        browser, _ = await self.load_shop()
        browser.labels[108] = {"tag": "a", "text": "Read more", "submit": "", "opens": "https://shop.example/about"}
        self.assertTrue((await self.broker.click(1))["ok"])
        self.assertEqual(self.judge.states, [])
        skipped = [e for e in self.audit.tail() if e["event"] == "decision.skipped"]
        self.assertEqual([(e["reason"], e["label"]) for e in skipped], [("plain link", 'link "Read more"')])
        # A commit word in its address, a link covered by something else, or a judge that cannot answer: as before.
        browser.labels[108]["opens"] = "https://shop.example/checkout?step=2"
        self.assertTrue((await self.broker.click(1))["ok"])
        self.assertIn("Link opens: /checkout?step=2", self.judge.states[-1])
        browser.labels[108]["opens"] = "https://shop.example/about"
        browser.labels[777] = {"tag": "div", "text": "Newsletter", "submit": ""}
        browser.hit = 777
        self.assertTrue((await self.broker.click(1))["ok"])
        self.assertIn("covered by div", self.judge.states[-1])
        browser.hit = None
        self.judge.fail = True
        self.assertTrue((await self.broker.click(1))["ok"])        # plain: no judge, no approval
        self.assertEqual(self.asked, [])
        self.assertFalse((await self.broker.click(4))["ok"])       # "Buy now", a button: judged, and asked
        self.assertEqual(self.kinds(), ["browser.sensitive"])

    async def test_without_an_answer_from_the_judge_the_owner_is_asked(self):
        await self.load_shop()
        self.judge.fail = True
        self.assertFalse((await self.broker.click(1))["ok"])
        self.broker.decider = None
        self.assertFalse((await self.broker.type(2, "rye"))["ok"])
        self.assertEqual(self.kinds(), ["browser.sensitive", "browser.sensitive"])

    async def test_find_reads_the_whole_page(self):
        browser, _ = await self.load_shop()
        found = await self.broker.find("buy")
        self.assertTrue(found["ok"], found)
        self.assertIn('button "Buy now"', found["content"])
        self.assertGreaterEqual(found["matches"], 1)
        self.assertIn("Nothing on this page", (await self.broker.find("submarine"))["content"])
        self.assertTrue((await self.broker.click(1))["ok"])           # refs from find are usable

    async def test_find_shows_what_follows_a_match_like_a_result_cards_price(self):
        browser, _ = await self.load_shop()
        card = ["Shree Hari Guest House", "Scored 8.3", "Very good", "Economy Double Room", "1 double bed",
                "2 nights, 2 adults", "₹1,450", "+₹208 taxes and charges", "See availability"]
        others = [f"Other text {n}" for n in range(30)]
        lines = others + card
        browser.ax = [ax(1, "RootWebArea", "Results", children=[100 + k for k in range(len(lines))], backend=1)]
        for k, line in enumerate(lines):           # one paragraph a line, as a results page renders
            browser.ax += [ax(100 + k, "paragraph", parent=1, children=[500 + k], backend=1000 + k),
                           ax(500 + k, "StaticText", line, parent=100 + k)]
        found = await self.broker.find("Shree Hari")
        self.assertIn("₹1,450", found["content"])
        self.assertIn("+₹208 taxes", found["content"])
        self.assertNotIn("Other text 10", found["content"])

    async def test_a_long_page_is_read_from_a_line_past_its_cut(self):
        browser, _ = await self.load_shop()
        lines = [f"Filter option number {n} with a long description" for n in range(400)] + [
            "1,240 properties found", "Shree Hari Guest House", "Scored 8.3", "₹1,450"]
        browser.ax = [ax(1, "RootWebArea", "Results", children=[100 + k for k in range(len(lines))], backend=1)]
        for k, line in enumerate(lines):
            browser.ax += [ax(100 + k, "paragraph", parent=1, children=[5000 + k], backend=1000 + k),
                           ax(5000 + k, "StaticText", line, parent=100 + k)]
        whole = await self.broker.snapshot()
        self.assertNotIn("Shree Hari", whole["content"])
        self.assertIn("start=<text> reads from a later line", whole["content"])
        later = await self.broker.snapshot(start="properties found")
        self.assertTrue(later["ok"], later)
        self.assertTrue(later["content"].startswith("… [400 lines above left out]"))
        self.assertIn("₹1,450", later["content"])
        self.assertIn("Nothing on this page mentions", (await self.broker.snapshot(start="submarine"))["error"])

    async def test_select_chooses_an_option_of_a_dropdown(self):
        browser, _ = await self.load_shop()
        browser.options[109] = ["Economy", "Premium economy", "Business"]
        chosen = await self.broker.select(2, "premium")
        self.assertTrue(chosen["ok"], chosen)
        self.assertEqual((chosen["selected"], browser.selected[109]), ("Premium economy", "Premium economy"))
        self.assertIn("choose an option in", self.judge.states[-1])
        missing = await self.broker.select(2, "First")
        self.assertIn('options are "Economy", "Premium economy", "Business"', missing["error"])
        self.assertIn("not a dropdown", (await self.broker.select(1, "x"))["error"])
        self.assertEqual(self.asked, [])

    async def test_a_click_on_a_dropdown_points_to_select(self):
        browser, _ = await self.load_shop()
        browser.options[109] = ["20, Friday", "21, Saturday", "22, Sunday"]
        result = await self.broker.click(2)
        self.assertFalse(result["ok"])
        self.assertIn("is a dropdown list: choose with browser_select", result["error"])
        self.assertIn('"22, Sunday"', result["error"])
        self.assertEqual([p for p in browser.sent("Input.dispatchMouseEvent") if p["type"] == "mousePressed"], [])

    async def test_a_click_that_reaches_a_covering_element_says_so(self):
        browser, _ = await self.load_shop()
        browser.hit = 111                      # "Buy now" lies over the link at the point of the click
        browser.labels[111] = {"tag": "label", "text": "Rooms and occupancy", "submit": ""}
        self.answers["browser.sensitive"] = True
        result = await self.broker.click(self.ref("Read more"))
        self.assertTrue(result["ok"], result)
        self.assertIn("covered by label", result.get("note", ""))

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

    async def test_any_interaction_in_a_chat_client_is_sensitive(self):
        for host in ("web.whatsapp.com", "app.slack.com"):
            with self.subTest(host=host):
                self.asked.clear()
                self.assertTrue((await self.broker.navigate(f"https://{host}/"))["ok"])
                browser = self.launcher.browser
                browser.ax = SHOP_AX
                browser.labels = {108: {"tag": "a", "text": "Read more", "submit": ""}}
                await self.broker.snapshot()
                self.assertFalse((await self.broker.click(self.ref("Read more")))["ok"])
                self.assertFalse((await self.broker.type(self.ref("Search"), "hello"))["ok"])
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
<button type="button" onclick="const b = document.createElement('button'); b.textContent = 'Result ' + (document.querySelectorAll('#more button').length + 1); document.getElementById('more').append(b)">More results</button>
<div id="more"></div>
<input type="text" aria-label="Destination" oninput="const v = this.value; setTimeout(() => { document.getElementById('hints').innerHTML = '<button type=button>' + v + ' International Airport</button>'; }, 800)">
<div id="hints"></div>
<div style="height: 4000px"></div><p>The end</p>
</body></html>"""
# Controls drawn the way most sites draw them: a radio's circle and a button's icon are ::before content, and
# the radio's own input is hidden inside its label (booking.com's flight search).
DRAWN = b"""<!doctype html><html><head><title>Drawn controls</title><style>
body { margin: 0; font: 16px sans-serif; }
label.choice { position: absolute; left: 20px; top: 20px; display: flex; align-items: center; gap: 8px; cursor: pointer; }
label.choice input { position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; }
label.choice .dot { position: relative; width: 24px; height: 24px; }
label.choice .dot::before { content: ""; position: absolute; inset: 0; border: 2px solid #333; border-radius: 50%; }
button.icon { position: absolute; left: 20px; top: 100px; width: 120px; height: 40px; }
button.icon::before { content: ""; position: absolute; left: 40px; top: 5px; width: 40px; height: 30px; background: #08f; }
</style></head><body>
<label class="choice"><input type="radio" name="trip" value="one"><span class="dot"></span>One way</label>
<button class="icon" type="button" onclick="document.getElementById('out').textContent = 'searched'">Search</button>
<div class="route" style="position: absolute; left: 20px; top: 180px; width: 300px; height: 60px">2 hr 20 min, 145 km</div>
<div id="out" style="position: absolute; top: 300px"></div>
<script>document.querySelector('.route').addEventListener('click', () => {
  document.getElementById('out').textContent = 'route chosen'; });</script>
</body></html>"""

# What a model working from screenshots does besides clicking: a wheel over a side list, a hover menu, a double
# click, replacing a field's text, a dropdown.
CONTROLS = (b"""<!doctype html><html><head><title>Controls</title><style>
body { margin: 0; font: 16px sans-serif; } #list { position: absolute; left: 0; top: 0; width: 300px; height: 200px;
overflow-y: scroll; } #list div { height: 40px; } #name { position: absolute; left: 400px; top: 20px; width: 200px; }
#pick { position: absolute; left: 400px; top: 80px; } #word { position: absolute; left: 400px; top: 140px; }
#menu { position: absolute; left: 700px; top: 20px; width: 100px; height: 40px; }
#out { position: absolute; left: 400px; top: 300px; } #tall { height: 3000px; }
</style></head><body><div id="list">""" + b"".join(b"<div>row %d</div>" % i for i in range(40)) + b"""</div>
<input id="name" aria-label="Name" value="old text" onkeydown="window.keys = (window.keys || 0) + 1">
<select id="pick" aria-label="Size"><option>Small</option><option>Medium</option><option>Large</option></select>
<span id="word" ondblclick="document.getElementById('out').textContent = 'double ' + this.textContent">wordy</span>
<div id="menu" onmouseover="document.getElementById('out').textContent = 'menu open'">Menu</div>
<div style="position: absolute; left: 400px; top: 200px; width: 200px"><input id="max" type="range" aria-label="Max."
 min="0" max="20000" step="500" value="20000" oninput="window.inputs = (window.inputs || 0) + 1"
 onchange="document.getElementById('price').textContent = 'max ' + this.value">
<div style="position: absolute; inset: 0"></div></div>
<div id="stars" role="slider" tabindex="0" aria-label="Stars" aria-valuemin="1" aria-valuemax="5" aria-valuenow="1"
 style="position: absolute; left: 400px; top: 240px" onkeydown="const n = +this.getAttribute('aria-valuenow') +
 (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0);
 this.setAttribute('aria-valuenow', Math.max(1, Math.min(5, n)))">stars</div>
<div id="price" style="position: absolute; left: 400px; top: 270px"></div>
<div id="out"></div><div id="tall"></div></body></html>""")

# A shop page: a short menu, five product cards and a footer. The cards are the page's main list.
LISTING = (b"""<!doctype html><html><head><title>Chargers</title></head><body>
<nav><a href="/">Home</a><a href="/deals">Deals</a><a href="/help">Help</a><a href="/cart">Cart</a></nav>
<div class="grid">""" + b"".join(b"""<div class="card s-result-%d"><a href="/p/%d"><h3>USB-C charger %dW</h3></a>
<span class="price">Rs %d,999</span><span class="rating">4.%d of 5 stars, %d ratings</span></div>""" % (n, n, 45 + n * 5, n, n, n * 1000)
                                     for n in range(1, 6)) + b"""</div>
<footer><ul><li>About our company and team</li><li>Careers at the shop today</li><li>Press and media enquiries</li></ul></footer>
</body></html>""")


# Boxes without words, as a screenshot driver clicks them by point: one with nothing around it, one in a search
# form, one beside a pay line.
WORDLESS = b"""<title>Boxes</title><style>div.box{position:absolute;width:40px;height:40px;background:#ccc}</style>
<div class="box" style="left:10px;top:10px" onclick="document.title='empty'"></div>
<section style="position:absolute;left:100px;top:0"><p>FROM Bengaluru TO New Delhi DEPART 15/01/2027</p>
  <div class="box" style="position:static" onclick="document.title='form'"></div></section>
<section style="position:absolute;left:400px;top:0"><p>Pay Rs 460 securely</p>
  <div class="box" style="position:static" onclick="document.title='paid'"></div></section>"""


# Results fetched after the page is parsed, and after a click, from an answer that takes 1.2 s; meanwhile a
# carousel moves for ever without changing how much text there is.
FETCHING = b"""<!doctype html><html><head><title>Fetching</title></head><body>
<p id="spin">Slide 1</p><button id="go">Search</button><div id="out"></div><div id="late"></div>
<script>
let n = 0;
setInterval(() => { n = (n + 1) % 9; document.getElementById('spin').textContent = 'Slide ' + (n + 1); }, 100);
const get = (where) => fetch('/slow', {headers: {'X-Where': where}}).then(r => r.text()).then(t => {
  document.getElementById(where).textContent = t; });
document.addEventListener('DOMContentLoaded', () => get('late'));
document.getElementById('go').onclick = () => get('out');
</script></body></html>"""


class _Fixture(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/slow":
            time.sleep(1.2)
            body = b"12 trains found, the first at 06:10" if self.headers.get("X-Where") == "late" else \
                b"3 seats left on the 06:10"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/away":
            self.send_response(302)
            self.send_header("Location", "http://10.255.255.1/admin")
            self.end_headers()
            return
        port = str(self.server.server_port).encode()
        body = FIXTURE.replace(b"PORT", port) if self.path == "/" else DRAWN if self.path == "/drawn" else \
            CONTROLS if self.path == "/controls" else LISTING if self.path == "/list" else \
            WORDLESS if self.path == "/boxes" else FETCHING if self.path == "/fetching" else \
            b"<title>Next</title><h2>Second page</h2>"
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
        self.broker = BrowserBroker(self.config, self.policy, self.audit, decider=StubJudge())

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

    async def page(self, expression: str):
        sid = await self.broker._session(self.broker._active)
        found = await self.broker._send(sid, "Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return found["result"].get("value")

    async def test_a_pages_main_list_is_read_one_line_per_item(self):
        opened = await self.broker.navigate(self.base + "/list")
        self.assertTrue(opened["ok"], opened)
        listed = await self.broker.items()
        self.assertTrue(listed["ok"], listed)
        self.assertEqual(listed["total"], 5)                                # the cards, not the menu or footer
        first = listed["items"][0]
        self.assertIn("USB-C charger 50W", first["text"])
        self.assertIn("Rs 1,999", first["text"])
        self.assertIn("1000 ratings", first["text"])
        self.assertEqual(first["href"], self.base + "/p/1")
        link = next(ref for ref, info in self.broker._refs[self.broker._active].items()
                    if info["role"] == "link" and info.get("url") == self.base + "/p/1")
        self.assertEqual(first["ref"], link)                    # the item's link, by the number the page text gives it
        later = await self.broker.items(start=3)
        self.assertEqual([i["href"].rsplit("/", 1)[1] for i in later["items"]], ["4", "5"])

    async def test_what_a_screenshot_model_does_besides_clicking(self):
        opened = await self.broker.navigate(self.base + "/controls")
        self.assertTrue(opened["ok"], opened)
        async with self.broker.screens_only():
            shot = await self.broker.screenshot(format="png")
            self.assertEqual(shot["viewport"], await self.page("[innerWidth, innerHeight]"))   # scrollbar included
            # The wheel scrolls what is under the pointer: the side list, not the page.
            wheel = await self.broker.wheel(150, 100, 0, 200)
            self.assertTrue(wheel["ok"], wheel)
            self.assertNotIn("content", wheel)                                # no page text for a screenshot model
            self.assertEqual(await self.page("[document.getElementById('list').scrollTop > 0, scrollY]"), [True, 0])
            self.assertTrue((await self.broker.hover(750, 40))["ok"])
            self.assertEqual(await self.page("document.getElementById('out').textContent"), "menu open")
            self.assertTrue((await self.broker.click_at(420, 150, clicks=2))["ok"])
            self.assertEqual(await self.page("document.getElementById('out').textContent"), "double wordy")
            # Select all, then type key by key: the field's text is replaced, and the page saw the keys.
            self.assertTrue((await self.broker.click_at(450, 30))["ok"])
            self.assertTrue((await self.broker.press_keys(["Control", "a"]))["ok"])
            typed = await self.broker.type_focused("new", keystrokes=True)
            self.assertTrue(typed["ok"], typed)
            self.assertEqual(await self.page("[document.getElementById('name').value, window.keys >= 3]"), ["new", True])
            self.assertIn("not available", (await self.broker.press_keys(["Meta", "v"]))["error"])   # no paste
            # A dropdown clicked by point gets the focus; typing an option's text chooses it.
            self.assertTrue((await self.broker.click_at(420, 90))["ok"])
            chosen = await self.broker.type_focused("Large")
            self.assertTrue(chosen["ok"], chosen)
            self.assertEqual(await self.page("document.getElementById('pick').value"), "Large")

    async def test_a_page_is_read_when_its_results_are_in_not_when_it_stops_moving(self):
        began = time.monotonic()
        opened = await self.broker.navigate(self.base + "/fetching")
        self.assertTrue(opened["ok"], opened)
        self.assertIn("12 trains found", opened["content"])         # fetched after the page was parsed
        self.assertLess(time.monotonic() - began, broker_module.SETTLE_MOST_LOAD)
        began = time.monotonic()
        clicked = await self.broker.click(self.refs()["Search"])
        took = time.monotonic() - began
        self.assertIn("+ 3 seats left", clicked["changes"])        # what the click fetched, though it took 1.2 s
        self.assertLess(took, broker_module.SETTLE_MOST)             # the carousel does not hold it to the limit

    async def test_a_box_without_words_is_told_by_the_words_around_it_and_one_with_none_is_not_asked(self):
        self.assertTrue((await self.broker.navigate(self.base + "/boxes"))["ok"])
        judge = self.broker.decider
        async with self.broker.screens_only():
            # Nothing in or around it: clicked without the judge or the owner.
            self.assertTrue((await self.broker.click_at(30, 30))["ok"])
            self.assertEqual((judge.states, self.asked, await self.page("document.title")), ([], [], "empty"))
            # In a search form: the judge reads the form's words, and it is clicked.
            box = await self.page("(() => { const r = document.querySelectorAll('div.box')[1].getBoundingClientRect();"
                                  " return [r.x + 20, r.y + 20]; })()")
            self.assertTrue((await self.broker.click_at(*box))["ok"])
            self.assertIn('click div without words, in "FROM Bengaluru TO New Delhi', judge.states[-1])
            self.assertEqual((self.asked, await self.page("document.title")), ([], "form"))
            # Beside a pay line: judged committing, asked, refused here.
            box = await self.page("(() => { const r = document.querySelectorAll('div.box')[2].getBoundingClientRect();"
                                  " return [r.x + 20, r.y + 20]; })()")
            self.assertFalse((await self.broker.click_at(*box))["ok"])
            self.assertEqual(self.asked, ["browser.sensitive"])
            self.assertNotEqual(await self.page("document.title"), "paid")

    async def test_a_slider_is_set_by_typing_the_number_it_should_move_to(self):
        self.assertTrue((await self.broker.navigate(self.base + "/controls"))["ok"])
        await self.broker.snapshot()
        refs = self.refs()
        # A range input under a cover, as booking.com's price filter: set, and the page hears it once.
        moved = await self.broker.type(refs["Max."], "₹8,000")
        self.assertIn("stands at 8000 (from 0 to 20000)", moved["note"])
        self.assertEqual(await self.page("[document.getElementById('price').textContent, window.inputs]"),
                         ["max 8000", 1])
        # One drawn by the page's script moves with arrow keys, to the number asked for.
        moved = await self.broker.type(refs["Stars"], "4")
        self.assertIn("stands at 4 (from 1 to 5)", moved["note"])
        self.assertEqual(await self.page("document.getElementById('stars').getAttribute('aria-valuenow')"), "4")
        self.assertIn("typing the number", (await self.broker.type(refs["Stars"], "four"))["error"])

    async def test_points_on_drawn_controls_reach_them(self):
        # A point on ::before content hits the pseudo-element, not an element: booking.com's one-way radio
        # could not be clicked by point, and a button's own icon counted as covering it.
        opened = await self.broker.navigate(self.base + "/drawn")
        self.assertTrue(opened["ok"], opened)
        radio = await self.broker.click_at(32, 32)                      # the radio's drawn circle
        self.assertTrue(radio["ok"], radio)
        self.assertIn('radio "One way" checked', (await self.broker.snapshot())["content"])
        button = await self.broker.click_at(80, 120)                    # the button's drawn icon
        self.assertTrue(button["ok"], button)
        self.assertNotIn("note", button)                                # its own icon, nothing covering it
        self.assertIn("searched", (await self.broker.snapshot())["content"])
        # A card only a script makes clickable (Google Maps' routes): nothing numbers it, a mouse clicks it.
        route = await self.broker.click_at(100, 210)
        self.assertTrue(route["ok"], route)
        self.assertIn("route chosen", (await self.broker.snapshot())["content"])

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
        self.assertIn("+ echo rye loaf", clicked["changes"])      # the click returns what it changed
        after = await self.broker.snapshot()
        self.assertFalse(after["changed"])                       # ...so the page holds no news
        self.assertIn("echo rye loaf", after["content"])

        more = await self.broker.click(refs["More results"])
        self.assertRegex(more["changes"], r'\+ \[\d+\] button "Result 1"')
        more = await self.broker.click(refs["More results"])     # the same number still works
        added = [line for line in more["changes"].splitlines() if line.startswith("+ ")]
        self.assertRegex(" ".join(added), r'^\+ \[\d+\] button "Result 2"$')      # only the new one is added
        # Suggestions that arrive after typing (here 0.8 s later) are part of what typing changed.
        hinted = await self.broker.type(self.refs()["Destination"], "Delhi")
        self.assertRegex(hinted["changes"], r'\+ \[\d+\] button "Delhi International Airport"')
        scrolled = await self.broker.scroll("down", 800)
        self.assertTrue(scrolled["scrolled"], scrolled)
        self.assertEqual(scrolled["changes"], "No visible change on the page.")
        self.assertTrue((await self.broker.press("Home"))["ok"])

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
