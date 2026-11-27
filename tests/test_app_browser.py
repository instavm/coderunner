"""The browser on the owner's Mac for a site that blocks the sandbox's: wall detection, the app's bridge, the
switch with the owner's OK and its per-site memory, the sign-in importer, and the line in the prompt.

No window, no real browser, no Keychain and no real cookies: fakes, fixture pages and a fabricated cookie DB.
(tests/test_app_browser_e2e.py drives the real app browser in a hidden Electron window.)
"""

import asyncio
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import stat
import tempfile
import time
import unittest
from unittest import mock

import httpx

from coderunner.agent.loop import SYSTEM, RunContext
from coderunner.browser import app as app_browser
from coderunner.browser import cookies
from coderunner.browser.app import AppBridge, AppLauncher, MacLauncher
from coderunner.browser.extension import ORIGIN as EXTENSION_ORIGIN, encode, read_frame
from coderunner.browser.switch import BrowserSwitch
from coderunner.browser.walls import classify, probe_from_html
from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.core.store import Store
from coderunner.safety.policy import Policy, current_run

# ---- fixture pages ----------------------------------------------------------------------------------------

CLOUDFLARE = """<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title>
<meta http-equiv="refresh" content="390"><script>(function(){window._cf_chl_opt={cvId: '3',cZone: 'shop.example',
cType: 'managed'};var a=document.createElement('script');a.src='/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1?ray=8';
document.getElementsByTagName('head')[0].appendChild(a);}());</script></head><body><div class="main-wrapper" role="main">
<div class="main-content"><h1 class="zone-name-title h1">shop.example</h1><h2 id="challenge-running" class="h2">Verifying you
are human. This may take a few seconds.</h2><noscript><div class="h2">Enable JavaScript and cookies to continue</div></noscript>
</div></div><div class="footer">Performance &amp; security by Cloudflare</div></body></html>"""
CLOUDFLARE_BLOCKED = """<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head><body>
<div id="cf-wrapper"><h1>Sorry, you have been blocked</h1><h2>You are unable to access shop.example</h2>
<p>This website is using a security service to protect itself from online attacks.</p>
<span>Cloudflare Ray ID: <strong>8a1b2c3d4e5f</strong></span></div></body></html>"""
DATADOME = """<html><head><title>shop.example</title><style>#cmsg{animation: A 1.5s;}</style></head><body
style="margin:0"><p id="cmsg">Please enable JS and disable any ad blocker</p><script data-cfasync="false">var dd={'rt':'c',
'cid':'AHrlqAAAAAMA','hsh':'2211F522B61E269B869FA6EAFFB5E1','t':'fe','s':17434,'host':'geo.captcha-delivery.com'}</script>
<script data-cfasync="false" src="https://ct.captcha-delivery.com/c.js"></script></body></html>"""
PERIMETERX = """<!DOCTYPE html><html lang="en"><head><title>Access to this page has been denied</title></head><body>
<div class="page-title-wrapper"><h1>Press &amp; Hold to confirm you are a human (and not a bot).</h1></div>
<div id="px-captcha"></div><script src="https://captcha.px-cdn.net/PXabc123/captcha.js"></script></body></html>"""
AKAMAI = """<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY><H1>Access Denied</H1>
You don't have permission to access "http&#58;&#47;&#47;www&#46;shop&#46;example&#47;" on this server.<P>
Reference&#32;&#35;18&#46;2f3c1002&#46;1732611234&#46;6a1b2c3d<P>https&#58;&#47;&#47;errors&#46;edgesuite&#46;net&#47;18
</BODY></HTML>"""
IMPERVA = """<html><head><META NAME="robots" CONTENT="noindex,nofollow"><script src="/_Incapsula_Resource?SWJIYLWA=5074a7"></script>
</head><body><iframe id="main-iframe" src="/_Incapsula_Resource?CWUDNSAI=9&xinfo=1" frameborder=0>Request unsuccessful.
Incapsula incident ID: 1234</iframe></body></html>"""
GOOGLE = """<html><head><title>https://www.google.com/search?q=visa</title></head><body><div>Our systems have detected
unusual traffic from your computer network. This page checks to see if it's really you sending the requests, and not a
robot.</div><div id="recaptcha" class="g-recaptcha"></div></body></html>"""
GENERIC = """<html><head><title>Please wait</title></head><body><h1>One moment</h1><p>Please verify you are a human to
continue to the site.</p></body></html>"""

LINKS = "".join(f'<li><a href="/item/{i}">Item {i}</a></li>' for i in range(40))
ARTICLE = f"""<html><head><title>How CAPTCHAs work - Tech Weekly</title>
<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script></head><body><nav><ul>{LINKS}</ul></nav>
<article><h1>How CAPTCHAs work</h1><p>Are you a robot? A captcha asks you to verify you are human. Cloudflare shows
"Just a moment..." and "Verifying you are human" while it checks your browser; DataDome and PerimeterX ("Press &amp; Hold")
do the same. If you see "Access denied", you have been blocked.</p>{'<p>More words about bots and challenges.</p>' * 60}
</article></body></html>"""
SIGN_IN = """<html><head><title>Sign in - Shop</title><script src="https://www.google.com/recaptcha/api.js"></script></head>
<body><form><label>Email <input name="email"></label><label>Password <input type="password" name="password"></label>
<div class="g-recaptcha" data-sitekey="6Lc"></div><button>Sign in</button></form><p>Are you a robot? Tick the box.</p>
</body></html>"""
TURNSTILE_FORM = """<html><head><title>Contact us</title></head><body><h1>Contact us</h1><form><textarea name="m"></textarea>
<div class="cf-turnstile" data-sitekey="0x4AAA"><iframe src="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/if/ov2/av0/rcv0/0/abc/light/normal"></iframe></div>
<button>Send</button></form></body></html>"""
OWN_403 = """<html><head><title>Members only</title></head><body><h1>Access denied</h1><p>Sign in to see this page.</p>
<a href="/login">Sign in</a></body></html>"""
SHORT_SITE = """<html><head><title>Shop</title><script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script></head>
<body><h1>Shop</h1><p>Open 9 to 5. Our checkout uses a captcha.</p><a href="/menu">Menu</a></body></html>"""


class WallTests(unittest.TestCase):
    def test_each_vendors_wall_is_named(self):
        cases = [(CLOUDFLARE, 403, "Cloudflare"), (CLOUDFLARE_BLOCKED, 403, "Cloudflare"), (DATADOME, 403, "DataDome"),
                 (PERIMETERX, 403, "PerimeterX"), (AKAMAI, 403, "Akamai"), (IMPERVA, 200, "Imperva"),
                 (GOOGLE, 429, "Google"), (GENERIC, 429, "a bot check")]
        for markup, status, vendor in cases:
            with self.subTest(vendor=vendor, page=markup[:60]):
                self.assertEqual(classify(probe_from_html(markup, status)), vendor)

    def test_a_cloudflare_wall_is_one_with_a_200_too(self):
        self.assertEqual(classify(probe_from_html(CLOUDFLARE, 200)), "Cloudflare")

    def test_pages_that_only_mention_captchas_or_carry_a_widget_are_not_walls(self):
        for name, markup, status in [("article", ARTICLE, 200), ("sign-in with reCAPTCHA", SIGN_IN, 200),
                                     ("form with Turnstile", TURNSTILE_FORM, 200), ("own 403", OWN_403, 403),
                                     ("site with Cloudflare's detection script", SHORT_SITE, 200),
                                     ("generic words without an error", GENERIC, 200)]:
            with self.subTest(name):
                self.assertIsNone(classify(probe_from_html(markup, status)))

    def test_odd_probes_are_not_walls(self):
        for probe in (None, "x", {}, {"text": "", "status": 0}):
            self.assertIsNone(classify(probe))


# ---- the app's bridge -----------------------------------------------------------------------------------

class HomeCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp / "h")
        self.audit = AuditLog(self.tmp / "audit.jsonl")

    async def asyncTearDown(self):
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)


class FakeApp:
    """The app's end of the socket: says hello, answers protocol commands like browser.js, takes cookies."""

    def __init__(self, path: str, token: str, origin: str = app_browser.ORIGIN):
        self.path, self.token, self.origin = path, token, origin
        self.cookies: list[dict] = []

    async def connect(self) -> dict:
        self.reader, self.writer = await asyncio.open_unix_connection(self.path)
        self.writer.write(encode({"type": "hello", "origin": self.origin, "token": self.token, "pid": 4242}))
        try:
            return await asyncio.wait_for(read_frame(self.reader), 5)
        except (asyncio.IncompleteReadError, ConnectionError):
            self.writer.close()
            await self.writer.wait_closed()
            return {}

    async def serve(self) -> None:
        with contextlib.suppress(asyncio.IncompleteReadError, ConnectionError):
            while True:
                message = await read_frame(self.reader)
                if message.get("type") == "cookies":
                    self.cookies += message["cookies"]
                    self.writer.write(encode({"type": "cookies-set", "ref": message["ref"],
                                              "set": len(message["cookies"]), "failed": 0}))
                elif message.get("method") == "Browser.getVersion":
                    self.writer.write(encode({"id": message["id"], "result": {"product": "CodeRunner app"}}))
                elif "id" in message:
                    self.writer.write(encode({"id": message["id"], "result": {}}))


class BridgeTests(HomeCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bridge = AppBridge(audit=self.audit)
        await self.bridge.start()

    async def asyncTearDown(self):
        await self.bridge.stop()
        await super().asyncTearDown()

    def owner_only(self, path: pathlib.Path) -> bool:
        return stat.S_IMODE(os.stat(path).st_mode) & 0o077 == 0

    async def test_the_app_finds_the_socket_and_token_in_files_only_the_owner_can_read(self):
        folder = app_browser.app_dir()
        self.assertEqual((folder / "socket").read_text().strip(), str(self.bridge.path))
        for path in (folder, folder / "token", folder / "socket", self.bridge.path):
            self.assertTrue(self.owner_only(path), path)

    async def test_a_wrong_token_or_the_extensions_origin_is_refused(self):
        token = (app_browser.app_dir() / "token").read_text().strip()
        self.assertEqual(await FakeApp(str(self.bridge.path), "nope").connect(), {})
        self.assertEqual(await FakeApp(str(self.bridge.path), token, EXTENSION_ORIGIN).connect(), {})
        self.assertFalse(self.bridge.connected())
        refused = [json.loads(line)["event"] for line in (self.tmp / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(refused.count("browser.app_browser_refused"), 2)

    async def test_the_broker_speaks_to_the_app_and_cookies_reach_it(self):
        app = FakeApp(str(self.bridge.path), (app_browser.app_dir() / "token").read_text().strip())
        self.assertEqual(await app.connect(), {"type": "welcome"})
        serving = asyncio.create_task(app.serve())
        try:
            process, cdp = await AppLauncher(self.bridge, wait=2).launch("coderunner-app", self.tmp)
            self.assertEqual((await cdp.send("Browser.getVersion"))["product"], "CodeRunner app")
            self.assertIsNone(process.poll())
            done = await self.bridge.set_cookies([{"url": "https://shop.example/", "name": "sid", "value": "v"}])
            self.assertEqual(done, {"set": 1, "failed": 0})
            self.assertEqual(app.cookies[0]["name"], "sid")
            self.assertEqual(self.bridge.status()["connected"], True)
            cdp.close()
        finally:
            serving.cancel()
            app.writer.close()

    async def test_without_the_app_the_mac_browser_is_a_chrome_window_of_its_own(self):
        fallback = mock.Mock(find_browser=lambda config: "/Applications/Google Chrome.app/x")
        launcher = MacLauncher(self.bridge, fallback=fallback)
        self.assertFalse(launcher.app_open())
        self.assertTrue(launcher.available(Config({})))
        self.assertEqual(launcher.find_browser(Config({})), "/Applications/Google Chrome.app/x")
        nothing = MacLauncher(self.bridge, fallback=mock.Mock(find_browser=lambda config: None))
        self.assertFalse(nothing.available(Config({})))
        with self.assertRaises(Exception):
            await AppLauncher(self.bridge, wait=0.1).launch("coderunner-app", self.tmp)

    async def test_mac_broker_switches_back_to_built_in_app_once_it_reconnects(self):
        from test_browser import FakeLauncher
        from coderunner.browser.broker import BrowserBroker
        fallback = FakeLauncher()
        app_launcher = FakeLauncher()

        class SwitchableMacLauncher(MacLauncher):
            def __init__(self, bridge):
                super().__init__(bridge, fallback=fallback)
                self._connected = False
                self.app = app_launcher

            def app_open(self) -> bool:
                return self._connected

        launcher = SwitchableMacLauncher(self.bridge)
        cfg = Config({})
        store = Store(self.tmp / "switchback.db")
        audit = AuditLog(self.tmp / "switchback.jsonl")
        broker = BrowserBroker(cfg, Policy(cfg, store, audit), audit, launcher=launcher, mode="mac")
        # 1. App closed -> launches fallback Chrome.
        await broker.start()
        self.assertTrue(launcher.using_fallback())
        self.assertEqual(len(fallback.browsers), 1)
        self.assertEqual(len(app_launcher.browsers), 0)
        # 2. App reconnects -> next _ensure() shuts down fallback Chrome and switches to built-in app browser.
        launcher._connected = True
        await broker.start()
        self.assertFalse(launcher.using_fallback())
        self.assertEqual(len(fallback.terminated), 1)
        self.assertEqual(len(app_launcher.browsers), 1)
        await broker.stop()
        store.close()


# ---- the switch -------------------------------------------------------------------------------------------

class FakeBroker:
    def __init__(self, name: str, walls: set[str] | None = None):
        self.name, self.walls = name, walls or set()
        self.calls: list[tuple] = []
        self.config = Config({})
        self.origin = "agent"
        self.alive = False
        self.handoff_result = {"ok": True, "completed": False}

    def result(self, url: str) -> dict:
        found = {"ok": True, "url": url, "title": "page", "content": f"{self.name} page"}
        host = url.split("/")[2] if "://" in url else ""
        if host in self.walls:
            found["wall"] = "Cloudflare"
        return found

    async def navigate(self, url, tab=None, tainted=False):
        self.calls.append(("navigate", url))
        return self.result(url)

    async def open(self, url, tainted=False):
        self.calls.append(("open", url))
        self.alive = True
        return self.result(url)

    async def click(self, ref, tab=None, tainted=False):
        self.calls.append(("click", ref))
        return {"ok": True, "url": "https://shop.example/next", "title": "next"}

    async def snapshot(self, **kwargs):
        self.calls.append(("snapshot",))
        return {"ok": True, "url": "https://shop.example/", "content": "x"}

    async def handoff(self, reason, tab=None, tainted=False):
        self.calls.append(("handoff", reason))
        return self.handoff_result

    def running(self):
        return self.alive

    def release(self, scope):
        self.calls.append(("release", scope))

    async def tidy(self, keep=1, scope=None):
        self.calls.append(("tidy", scope))
        return 0


class SwitchTests(HomeCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config = Config({}, path=self.tmp / "config.json")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.vm = FakeBroker("vm", walls={"shop.example"})
        self.mac = FakeBroker("mac")
        self.launcher = mock.Mock(available=lambda config: True, app_open=lambda: True)
        self.switch = BrowserSwitch(self.vm, self.mac, self.policy, self.audit, launcher=self.launcher, config=self.config)
        self.asked: list = []
        self.answer: tuple | None = (True, "once")
        self.policy.add_listener(self.decide)
        self.token = current_run.set(RunContext(conversation="c1"))

    async def asyncTearDown(self):
        current_run.reset(self.token)
        await super().asyncTearDown()

    def decide(self, approval):
        self.asked.append(approval)
        if self.answer is not None:
            self.policy.decide(approval.id, *self.answer)

    async def test_a_wall_asks_once_and_the_sites_steps_go_on_on_the_mac(self):
        moved = await self.switch.navigate("https://shop.example/item")
        self.assertEqual([a.summary for a in self.asked],
                         ["shop.example blocks the sandbox browser. Continue in CodeRunner's built-in browser on your Mac?"])
        self.assertEqual((self.asked[0].kind, self.asked[0].target), ("browser.mac", "shop.example"))
        self.assertEqual(moved["content"], "mac page")
        self.assertIn(("open", "https://shop.example/item"), self.mac.calls)     # a tab of its own, never the owner's
        self.assertEqual(self.switch.lane(), "mac")
        await self.switch.click(3)
        await self.switch.snapshot()
        self.assertEqual([c[0] for c in self.vm.calls], ["navigate"])
        self.assertIn(("click", 3), self.mac.calls)
        # Back on the site later in this task: straight to the Mac, nothing asked again.
        await self.switch.navigate("https://www.shop.example/cart")
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.mac.calls[-1], ("navigate", "https://www.shop.example/cart"))
        # Another site: the sandbox's browser again.
        await self.switch.navigate("https://other.example/")
        self.assertEqual(self.vm.calls[-1], ("navigate", "https://other.example/"))
        self.assertEqual(self.switch.lane(), "vm")
        self.assertTrue(app_browser.used())

    async def test_allow_once_ends_with_the_task_and_always_is_a_grant_the_owner_can_clear(self):
        await self.switch.navigate("https://shop.example/")
        await self.switch.tidy(scope="c1")
        self.mac.calls.clear()
        await self.switch.navigate("https://shop.example/")
        self.assertEqual(len(self.asked), 2)                    # once means this task only
        self.assertEqual(self.policy.grants(), [])

        self.answer = (True, "always")
        await self.switch.tidy(scope="c1")
        await self.switch.navigate("https://shop.example/")
        grants = [g for g in self.policy.grants() if g["kind"] == "browser.mac"]
        self.assertEqual([g["target"] for g in grants], ["shop.example"])
        # A later task goes straight to the Mac for that site: not asked, the sandbox's browser never tried.
        await self.switch.tidy(scope="c1")
        current_run.set(RunContext(conversation="c2"))
        self.vm.calls.clear()
        await self.switch.navigate("https://shop.example/deals")
        self.assertEqual(len(self.asked), 3)
        self.assertEqual(self.vm.calls, [])
        self.assertEqual(self.mac.calls[-1], ("open", "https://shop.example/deals"))
        # Cleared in Models and privacy: asked again.
        self.policy.revoke(grants[0]["id"])
        await self.switch.tidy(scope="c2")
        await self.switch.navigate("https://shop.example/deals")
        self.assertEqual(len(self.asked), 4)

    async def test_no_hands_the_check_to_the_owner_and_is_not_asked_again_this_task(self):
        self.answer = (False, "once")
        result = await self.switch.navigate("https://shop.example/")
        self.assertEqual(self.vm.calls[-1][0], "handoff")
        self.assertIn("shop.example is checking for a person", self.vm.calls[-1][1])
        self.assertIn("did not clear it", result["note"])
        self.assertEqual(self.switch.lane(), "vm")
        self.assertEqual(self.mac.calls, [])
        again = await self.switch.navigate("https://shop.example/other")
        self.assertEqual(len(self.asked), 1)
        self.assertNotIn("handoff", [c[0] for c in self.vm.calls[-1:]])
        self.assertIn("did not clear it", again["note"])

    async def test_the_owner_clearing_the_check_in_the_sandbox_carries_on_there(self):
        self.answer = (False, "once")
        self.vm.handoff_result = {"ok": True, "completed": True, "url": "https://shop.example/", "title": "Shop"}
        result = await self.switch.navigate("https://shop.example/")
        self.assertEqual(result["note"], "The owner cleared the site's check. Read the page again before going on.")
        self.assertNotIn("wall", result)

    async def test_unanswered_is_no_hand_off_and_nothing_moves(self):
        self.answer = None
        self.config.set("agent.approval_timeout", 0.2)
        result = await self.switch.navigate("https://shop.example/")
        self.assertNotIn("handoff", [c[0] for c in self.vm.calls])
        self.assertEqual(self.mac.calls, [])
        self.assertIn("did not clear it", result["note"])

    async def test_with_no_mac_browser_at_all_the_owner_clears_it_in_the_sandbox_unasked(self):
        self.launcher.available = lambda config: False
        await self.switch.navigate("https://shop.example/")
        self.assertEqual(self.asked, [])
        self.assertEqual(self.vm.calls[-1][0], "handoff")

    async def test_a_page_that_is_not_a_wall_is_left_alone(self):
        result = await self.switch.navigate("https://fine.example/")
        self.assertEqual(result["content"], "vm page")
        self.assertEqual(self.asked, [])

    async def test_their_sign_in_is_offered_on_first_use_only(self):
        bridge = mock.Mock(connected=lambda: True)
        self.switch.bridge = bridge
        with mock.patch.object(cookies, "installed", return_value=["chrome"]):
            await self.switch.navigate("https://shop.example/")
            self.assertEqual(self.asked[0].detail["sign_in"], "chrome")
            await self.switch.tidy(scope="c1")
            await self.switch.navigate("https://shop.example/")
            self.assertNotIn("sign_in", self.asked[1].detail)

    async def test_whatever_else_is_asked_of_the_browser_is_the_sandboxs(self):
        self.vm.mode = "computer"
        self.switch.restart = "restart-hook"                     # set by the daemon after the switch is made
        self.assertEqual(self.vm.restart, "restart-hook")
        self.assertEqual(self.switch.mode, "computer")


# ---- the sign-in importer -------------------------------------------------------------------------------------

PASSWORD = b"fixture-safe-storage-password"


def encrypt(value: str, host: str, version: int) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    plain = (hashlib.sha256(host.encode()).digest() if version >= 24 else b"") + value.encode()
    pad = 16 - len(plain) % 16
    plain += bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(cookies.chrome_key(PASSWORD)), modes.CBC(b" " * 16)).encryptor()
    return b"v10" + encryptor.update(plain) + encryptor.finalize()


def chrome_fixture(home: pathlib.Path, version: int = 24) -> None:
    folder = home / "Library/Application Support/Google/Chrome/Default/Network"
    folder.mkdir(parents=True)
    db = sqlite3.connect(folder / "Cookies")
    db.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('version', ?)", (str(version),))
    db.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, encrypted_value BLOB, path TEXT, "
               "expires_utc INTEGER, is_secure INTEGER, is_httponly INTEGER, samesite INTEGER)")
    expires = int((time.time() + 86400 + cookies.CHROME_EPOCH) * 1_000_000)
    rows = [(".shop.example", "sid", "", encrypt("s3cret-a", ".shop.example", version), "/", expires, 1, 1, 1),
            ("www.shop.example", "cart", "", encrypt("s3cret-b", "www.shop.example", version), "/cart", 0, 1, 0, -1),
            ("www.shop.example", "__Host-id", "", encrypt("s3cret-c", "www.shop.example", version), "/", 0, 1, 1, 2),
            (".notshop.example", "sid", "", encrypt("other", ".notshop.example", version), "/", 0, 1, 1, 0),
            (".shop.example.evil.org", "sid", "", encrypt("evil", ".shop.example.evil.org", version), "/", 0, 1, 1, 0),
            (".bank.example", "session", "", encrypt("never", ".bank.example", version), "/", 0, 1, 1, 0)]
    db.executemany("INSERT INTO cookies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.commit()
    db.close()


class CookieTests(unittest.TestCase):
    def setUp(self):
        self.home = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        # The real Keychain is never asked in these tests.
        patcher = mock.patch.object(cookies.subprocess, "run", side_effect=AssertionError("the Keychain was asked"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_the_ticked_site_comes_and_values_decrypt(self):
        chrome_fixture(self.home)
        found = cookies.read("chrome", ["shop.example"], home=self.home, password=PASSWORD)
        by_name = {c["name"]: c for c in found}
        self.assertEqual(sorted(by_name), ["__Host-id", "cart", "sid"])
        self.assertEqual(by_name["sid"]["value"], "s3cret-a")
        self.assertEqual(by_name["sid"]["domain"], ".shop.example")
        self.assertEqual((by_name["sid"]["sameSite"], by_name["sid"]["httpOnly"]), ("lax", True))
        self.assertAlmostEqual(by_name["sid"]["expirationDate"], time.time() + 86400, delta=60)
        self.assertNotIn("domain", by_name["cart"])                # host-only stays host-only
        self.assertNotIn("expirationDate", by_name["cart"])        # a session cookie stays one
        self.assertEqual(by_name["cart"]["url"], "https://www.shop.example/cart")
        self.assertNotIn("domain", by_name["__Host-id"])
        self.assertEqual(by_name["__Host-id"]["sameSite"], "strict")

    def test_an_older_database_has_no_host_prefix(self):
        chrome_fixture(self.home, version=20)
        found = cookies.read("chrome", ["www.shop.example"], home=self.home, password=PASSWORD)
        self.assertEqual(sorted(c["value"] for c in found), ["s3cret-a", "s3cret-b", "s3cret-c"])

    def test_never_everything(self):
        for bad in ([], [""], ["*"], ["com"], ["*.example"], ["example."]):
            with self.subTest(bad=bad), self.assertRaises(cookies.CookieError):
                cookies.clean_domains(bad)
        self.assertEqual(cookies.clean_domains(["https://www.Shop.example/path", "shop.example"]), ["shop.example"])

    def test_the_keychain_is_asked_only_without_a_password_and_a_refusal_is_said_plainly(self):
        chrome_fixture(self.home)
        refused = mock.Mock(returncode=44, stdout=b"")
        with mock.patch.object(cookies.subprocess, "run", return_value=refused) as run:
            with self.assertRaises(cookies.CookieError) as raised:
                cookies.read("chrome", ["shop.example"], home=self.home)
        self.assertIn("Chrome Safe Storage", run.call_args[0][0])
        self.assertIn("not shared", str(raised.exception))

    def test_firefox_cookies_are_read_unencrypted(self):
        folder = self.home / "Library/Application Support/Firefox/Profiles/abc.default-release"
        folder.mkdir(parents=True)
        db = sqlite3.connect(folder / "cookies.sqlite")
        db.execute("CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT, path TEXT, expiry INTEGER, isSecure INTEGER, "
                   "isHttpOnly INTEGER, sameSite INTEGER)")
        db.executemany("INSERT INTO moz_cookies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [(".shop.example", "sid", "ff-1", "/", 2000000000, 1, 1, 1),
                        (".elsewhere.example", "sid", "no", "/", 2000000000, 1, 1, 1)])
        db.commit()
        db.close()
        self.assertEqual(cookies.installed(self.home), ["firefox"])
        found = cookies.read("firefox", ["shop.example"], home=self.home)
        self.assertEqual([(c["name"], c["value"], c["expirationDate"]) for c in found], [("sid", "ff-1", 2000000000.0)])

    def test_no_browser_no_cookies(self):
        self.assertEqual(cookies.installed(self.home), [])
        with self.assertRaises(cookies.CookieError):
            cookies.read("chrome", ["shop.example"], home=self.home, password=PASSWORD)


class ImportEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_owners_app_imports_and_values_are_never_audited(self):
        from coderunner.server.gateway import admin_app
        from fakes import Svc
        from test_gateway import ADMIN, FakeDaemon
        svc = Svc()
        svc.app_browser = mock.Mock(connected=lambda: True, status=lambda: {"connected": True},
                                    set_cookies=mock.AsyncMock(return_value={"set": 2, "failed": 0}))
        found = [{"url": "https://shop.example/", "name": "sid", "value": "s3cret-value"}] * 2
        app = admin_app(svc, FakeDaemon(svc))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://admin") as client:
            body = {"domains": ["shop.example"], "browser": "chrome"}
            self.assertEqual((await client.post("/api/browser/mac/import", json=body)).status_code, 401)
            with mock.patch.object(cookies, "read", return_value=found) as read:
                done = await client.post("/api/browser/mac/import", json=body, headers={"Authorization": f"Bearer {ADMIN}"})
            self.assertEqual(done.json(), {"imported": 2, "failed": 0, "browser": "Chrome"})
            read.assert_called_once_with("chrome", ["shop.example"])
            listed = await client.get("/api/browser/mac", headers={"Authorization": f"Bearer {ADMIN}"})
            self.assertEqual(listed.json()["available"], True)
        trail = (svc.tmp / "audit.jsonl").read_text()
        self.assertIn("browser.cookies_imported", trail)
        self.assertNotIn("s3cret-value", trail)
        self.assertNotIn('"sid"', trail)


class BrowserTabTests(unittest.TestCase):
    """The Browser tab as the page and the shell have it (the hidden-window checks drive both for real)."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def test_the_menu_is_short_and_clearing_asks_once(self):
        page = (self.ROOT / "coderunner/server/web/app/app.js").read_text()
        menu = page[page.index("async function openBrowserMenu()"):page.index("function closeBrowserMenu()")]
        labels = [label for label in ("Open in default browser", "Copy link", "Import cookies…", "Keep cookies",
                                      "Until quit", "Always", "Clear browsing data") if label in menu]
        self.assertEqual(len(labels), 7)
        self.assertIn('menuItem("Clear browsing data", () => clearForm(), {danger: true})', menu)
        self.assertNotRegex(menu, r"Viewport|Local network|HTML file")
        clear = page[page.index("function clearForm()"):page.index("async function importForm(")]
        self.assertIn('shellBrowser.act({action: "clear"})', clear.split("clear.onclick")[1])   # only on the second press
        self.assertIn('"btn danger-fill", "Clear"', clear)

    def test_cookies_are_kept_unless_the_owner_says_until_quit_and_the_shell_clears_them_then(self):
        shell, main = ((self.ROOT / "app" / name).read_text() for name in ("browser.js", "main.js"))
        self.assertIn('const KEEP_DEFAULT = "always";', shell)
        self.assertIn('if (this.keep() === "quit") this.clearCookies();', shell)        # a start after an unclean quit
        self.assertIn('appBrowser.keep() !== "quit"', main)                              # and the quit itself
        self.assertIn('saveSetting("browserKeepCookies", mode)', main)
        # Only a web page's address leaves the browser, and a test's stand-ins replace the system's.
        self.assertIn("if (!/^https?:\\/\\//i.test(url)) return false;", shell)
        self.assertIn("openExternal = (url) => shell.openExternal(url), writeClipboard = (text) => clipboard.writeText(text)", shell)

    def test_the_browser_session_is_its_own_and_never_reaches_the_daemon(self):
        shell = (self.ROOT / "app" / "browser.js").read_text()
        self.assertIn('const PARTITION = "persist:coderunner-browser";', shell)
        self.assertIn("banned.push(`${this.daemonOrigin}/*`", shell)
        self.assertNotRegex(shell, r"setUserAgent|userAgent\s*=|webdriver|WEBGL|toDataURL\s*=")   # nothing disguised

    def test_closing_a_tab_detaches_without_reattach_spin_loop(self):
        shell = (self.ROOT / "app" / "browser.js").read_text()
        self.assertIn("tab.closing = true;", shell)
        self.assertIn("contents.debugger.removeAllListeners()", shell)
        self.assertIn('reason === "target closed"', shell)
        self.assertIn("tab.reattachTimer = setTimeout(", shell)


class PromptTests(unittest.TestCase):
    def test_the_agent_knows_a_blocked_site_can_continue_on_the_mac(self):
        lines = [line for line in SYSTEM.splitlines() if "blocks the sandbox browser" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("Mac with their OK", lines[0])
        self.assertIn("asked and done for you", lines[0])


if __name__ == "__main__":
    unittest.main()
