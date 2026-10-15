"""The owner's everyday browser through the CodeRunner extension (browser.mode = "chrome")."""

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from coderunner import cli
from coderunner.core.audit import AuditLog
from coderunner.browser import BrowserBroker, extension
from coderunner.browser.chrome_host import FRAME, NATIVE
from coderunner.core.config import Config
from coderunner.safety.policy import Policy
from coderunner.core.store import Store
from test_browser import SHOP_AX, FakeBrowser, StubJudge

HOST = pathlib.Path(extension.__file__).with_name("chrome_host.py")
EXTENSION = pathlib.Path(extension.__file__).with_name("extension")


def short_socket_dir() -> pathlib.Path:
    # Unix socket paths are limited to about 100 bytes.
    return pathlib.Path(tempfile.mkdtemp(dir="/tmp" if os.path.isdir("/tmp") else None))


class ExtensionFilesTests(unittest.TestCase):
    def test_pinned_key_gives_the_extension_id_the_host_allows(self):
        manifest = json.loads((EXTENSION / "manifest.json").read_text())
        digest = hashlib.sha256(base64.b64decode(manifest["key"])).hexdigest()[:32]
        self.assertEqual("".join(chr(ord("a") + int(c, 16)) for c in digest), extension.EXTENSION_ID)
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertIn("debugger", manifest["permissions"])
        self.assertNotIn("host_permissions", manifest)      # no content scripts, no page access of its own

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_background_script_parses(self):
        subprocess.run(["node", "--check", str(EXTENSION / "background.js")], check=True)

    def test_a_restarted_worker_takes_over_its_own_debugger_and_no_one_elses(self):
        # In a real run the extension's worker restarted mid-task and every later step on that tab failed with
        # "Another debugger is already attached": the old worker's attachment outlived its memory of it.
        check = pathlib.Path(__file__).parent / "js" / "attach_check.js"
        done = subprocess.run(["node", str(check), str(EXTENSION / "background.js")], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "us: attached; devtools: refused; version told")


# ---- the native host ---------------------------------------------------------------

class FakeDaemon:
    """A unix socket server speaking the daemon's side of the host protocol, on a thread."""

    def __init__(self, path, token="secret-token"):
        self.path, self.token = str(path), token
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.server.settimeout(10)
        self.conn = None
        self.hello = None

    def accept(self, answer="welcome"):
        self.conn, _ = self.server.accept()
        self.conn.settimeout(10)
        self.hello = self.read()
        self.write({"type": answer})

    def read(self):
        size = FRAME.unpack(self._exact(FRAME.size))[0]
        return json.loads(self._exact(size))

    def _exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self.conn.recv(size - len(data))
            if not chunk:
                raise EOFError
            data += chunk
        return data

    def write(self, message):
        data = json.dumps(message).encode()
        self.conn.sendall(FRAME.pack(len(data)) + data)

    def close(self):
        for sock in (self.conn, self.server):
            if sock is not None:
                sock.close()


class HostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = short_socket_dir()
        self.token_file = self.tmp / "token"
        self.token_file.write_text("secret-token\n")
        self.daemon = FakeDaemon(self.tmp / "d.sock")

    def tearDown(self):
        self.daemon.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def start_host(self, origin=extension.ORIGIN):
        return subprocess.Popen(
            [sys.executable, "-I", str(HOST), "--socket", str(self.tmp / "d.sock"), "--token-file",
             str(self.token_file), "--extension", extension.EXTENSION_ID, origin],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def native_read(stream):
        size = NATIVE.unpack(stream.read(NATIVE.size))[0]
        return json.loads(stream.read(size))

    @staticmethod
    def native_write(stream, message):
        data = json.dumps(message).encode()
        stream.write(NATIVE.pack(len(data)) + data)
        stream.flush()

    def test_relays_between_the_browser_and_the_daemon(self):
        host = self.start_host()
        try:
            self.daemon.accept()
            self.assertEqual(self.daemon.hello["token"], "secret-token")
            self.assertEqual(self.daemon.hello["origin"], extension.ORIGIN)
            self.assertEqual(self.native_read(host.stdout), {"type": "daemon", "connected": True})

            self.daemon.write({"id": 1, "method": "Target.getTargets", "params": {}})
            self.assertEqual(self.native_read(host.stdout)["method"], "Target.getTargets")
            self.native_write(host.stdin, {"id": 1, "result": {"targetInfos": []}})
            self.assertEqual(self.daemon.read(), {"id": 1, "result": {"targetInfos": []}})

            # Commands too large for the browser are answered for it instead of being dropped silently.
            self.daemon.write({"id": 2, "method": "Input.insertText", "params": {"text": "x" * (1024 * 1024)}})
            self.assertEqual(self.daemon.read()["error"]["message"], "message too large for the browser")

            self.daemon.conn.close()
            self.assertEqual(self.native_read(host.stdout), {"type": "daemon", "connected": False})
            host.stdin.close()               # the browser closed the port: the host exits
            self.assertEqual(host.wait(10), 0)
        finally:
            if host.poll() is None:
                host.kill()
            host.wait()
            for stream in (host.stdout, host.stderr):
                stream.close()

    def test_refuses_any_other_extension(self):
        host = self.start_host("chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/")
        _, err = host.communicate(timeout=10)
        self.assertEqual(host.returncode, 1)
        self.assertIn(b"refusing caller", err)


# ---- the daemon's side ---------------------------------------------------------------

class FakeExtension:
    """The extension plus its host, as the bridge sees them: FakeBrowser answers the protocol."""

    def __init__(self, path, token):
        self.path, self.token = path, token
        self.browser = FakeBrowser()
        self.task = None

    async def connect(self, origin=extension.ORIGIN):
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.path))
        self.writer.write(extension.encode({"type": "hello", "token": self.token, "origin": origin, "pid": 77}))
        try:
            reply = await extension.read_frame(self.reader)
        except asyncio.IncompleteReadError:
            return None
        if reply.get("type") == "welcome":
            await self.browser.start(lambda message: self.writer.write(extension.encode(message)), lambda: None)
            self.task = asyncio.create_task(self._pump())
        return reply

    async def _pump(self):
        with contextlib.suppress(asyncio.IncompleteReadError, ConnectionError):
            while True:
                self.browser.send(await extension.read_frame(self.reader))

    async def close(self):
        if getattr(self, "writer", None) is not None:
            self.writer.close()
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = short_socket_dir()
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.bridge = extension.ChromeBridge(self.tmp / "c.sock", "good-token", audit=self.audit)
        await self.bridge.start()

    async def asyncTearDown(self):
        await self.bridge.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_socket_is_private(self):
        self.assertEqual(os.stat(self.tmp / "c.sock").st_mode & 0o777, 0o600)

    async def test_a_daemon_stopping_late_leaves_the_next_ones_socket(self):
        newer = extension.ChromeBridge(self.tmp / "c.sock", "good-token", audit=self.audit)
        await newer.start()                    # started while the first was still stopping
        await self.bridge.stop()
        self.assertTrue((self.tmp / "c.sock").exists())
        await newer.stop()
        self.assertFalse((self.tmp / "c.sock").exists())

    async def test_wrong_token_or_origin_is_refused(self):
        for token, origin in (("bad-token", extension.ORIGIN), ("good-token", "chrome-extension://other/")):
            fake = FakeExtension(self.tmp / "c.sock", token)
            self.assertIsNone(await fake.connect(origin))
            await fake.close()
        self.assertFalse(self.bridge.connected())
        self.assertEqual(sum(e["event"] == "browser.extension_refused" for e in self.audit.tail()), 2)

    async def test_one_browser_at_a_time(self):
        first, second = FakeExtension(self.tmp / "c.sock", "good-token"), FakeExtension(self.tmp / "c.sock", "good-token")
        self.assertEqual(await first.connect(), {"type": "welcome"})
        self.assertEqual(await second.connect(), {"type": "busy"})
        self.assertTrue(self.bridge.connected())
        self.assertEqual(self.bridge.status()["host_pid"], 77)
        await first.close()
        await second.close()
        for _ in range(50):
            if not self.bridge.connected():
                break
            await asyncio.sleep(0.02)
        self.assertEqual(self.bridge.status(), {"connected": False})

    async def test_an_older_extension_is_reported_once(self):
        async def connect(version):
            reader, writer = await asyncio.open_unix_connection(str(self.tmp / "c.sock"))
            writer.write(extension.encode({"type": "hello", "token": "good-token", "origin": extension.ORIGIN,
                                           "pid": 77}))
            self.assertEqual(await extension.read_frame(reader), {"type": "welcome"})
            if version is not None:
                writer.write(extension.encode({"type": "extension", "version": version}))
            for _ in range(50):
                if self.bridge.connection is not None and self.bridge.connection.version == version:
                    break
                await asyncio.sleep(0.02)
            status = self.bridge.status()
            writer.close()
            for _ in range(50):
                if not self.bridge.connected():
                    break
                await asyncio.sleep(0.02)
            return status

        status = await connect("0.9.0")
        self.assertEqual(status["version"], "0.9.0")
        self.assertIn("chrome://extensions", status["outdated"])
        await connect("0.9.0")
        self.assertIn("chrome://extensions", (await connect(None))["outdated"])       # too old to say
        self.assertNotIn("outdated", await connect(extension.current_version()))
        noted = [e for e in self.audit.tail() if e["event"] == "browser.extension_outdated"]
        self.assertEqual([e["had"] for e in noted], ["0.9.0"])                       # once per version

    async def test_waiting_for_the_extension(self):
        self.assertIsNone(await self.bridge.wait(0.05))
        fake = FakeExtension(self.tmp / "c.sock", "good-token")
        waiting = asyncio.create_task(self.bridge.wait(5))
        await fake.connect()
        self.assertIsNotNone(await waiting)
        await fake.close()


class ChromeModeBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = short_socket_dir()
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp)
        self.config = Config({"browser": {"mode": "chrome", "headless": True}}, path=self.tmp / "config.json")
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.policy = Policy(self.config, Store(":memory:"), self.audit)
        self.policy.add_listener(lambda approval: self.policy.decide(approval.id, allow=False))
        self.bridge = extension.ChromeBridge(self.tmp / "c.sock", "good-token", audit=self.audit)
        await self.bridge.start()
        self.broker = BrowserBroker(self.config, self.policy, self.audit, decider=StubJudge(), chrome=self.bridge)

        async def resolve(host):
            return ["93.184.216.34"]
        self.broker._resolve = resolve
        self.fake = FakeExtension(self.tmp / "c.sock", "good-token")

    async def asyncTearDown(self):
        await self.broker.stop()
        await self.fake.close()
        await self.bridge.stop()
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_drives_the_group_through_the_same_broker(self):
        await self.fake.connect()
        self.assertFalse(self.broker.headless)            # the owner's own browser is always visible
        opened = await self.broker.navigate("https://shop.example/")
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(self.fake.browser.sent("Page.navigate"), [{"url": "https://shop.example/"}])
        # The broker's hardening is sent as usual; the extension answers what it can for its tabs.
        self.assertTrue(self.fake.browser.sent("Fetch.enable"))
        self.assertTrue(self.fake.browser.sent("Target.setDiscoverTargets"))

        self.fake.browser.ax = SHOP_AX
        snap = await self.broker.snapshot()
        self.assertIn("# Welcome", snap["content"])
        status = self.broker.running()
        self.assertTrue(status)

        await self.broker.stop()                          # lets go of the tabs; the owner's browser stays open
        self.assertTrue(self.fake.browser.sent("Browser.close"))
        self.assertTrue(self.bridge.connected())
        self.assertTrue((await self.broker.navigate("https://shop.example/menu"))["ok"])   # and comes back

    async def test_without_the_extension_the_assistant_says_how_to_connect(self):
        with mock.patch.object(extension, "CONNECT_WAIT", 0.05):
            result = await self.broker.navigate("https://shop.example/")
        self.assertFalse(result["ok"])
        self.assertIn("coderunner browser connect", result["error"])

    async def test_extension_going_away_stops_the_browser(self):
        await self.fake.connect()
        self.assertTrue((await self.broker.navigate("https://shop.example/"))["ok"])
        await self.fake.close()
        for _ in range(50):
            if not self.broker.running():
                break
            await asyncio.sleep(0.02)
        self.assertFalse(self.broker.running())
        with mock.patch.object(extension, "CONNECT_WAIT", 0.05):
            self.assertIn("not connected", (await self.broker.snapshot())["error"])


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp / "home")
        self.roots = {"chrome": self.tmp / "Chrome", "brave": self.tmp / "Brave"}
        self.roots["chrome"].mkdir()                      # only Chrome has been run on this machine

    def tearDown(self):
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_registers_the_host_for_the_pinned_extension_only(self):
        result = extension.install(roots=self.roots)
        self.assertEqual([m["browser"] for m in result["manifests"]], ["chrome"])
        manifest = json.loads((self.roots["chrome"] / "NativeMessagingHosts" / f"{extension.HOST_NAME}.json")
                              .read_text())
        self.assertEqual(manifest["allowed_origins"], [extension.ORIGIN])
        self.assertEqual(manifest["type"], "stdio")
        wrapper = pathlib.Path(manifest["path"])
        self.assertTrue(os.access(wrapper, os.X_OK))
        script = wrapper.read_text()
        self.assertIn(str(extension.socket_path()), script)
        self.assertIn(extension.EXTENSION_ID, script)
        self.assertTrue((pathlib.Path(result["extension_dir"]) / "manifest.json").exists())
        self.assertEqual(os.stat(extension.token_path()).st_mode & 0o777, 0o600)
        self.assertEqual(extension.installed(self.roots), {"chrome": True})
        self.assertEqual(extension.uninstall(self.roots), [str(self.roots["chrome"] / "NativeMessagingHosts" /
                                                               f"{extension.HOST_NAME}.json")])
        self.assertEqual(extension.installed(self.roots), {"chrome": False})

    def test_named_browsers_and_unknown_names(self):
        result = extension.install(["brave"], roots=self.roots)
        self.assertEqual([m["browser"] for m in result["manifests"]], ["brave"])
        with self.assertRaises(ValueError):
            extension.install(["netscape"], roots=self.roots)

    def test_long_home_falls_back_to_a_short_socket_path(self):
        os.environ["CODERUNNER_HOME"] = str(self.tmp / ("x" * 120))
        self.assertLess(len(str(extension.socket_path())), 100)

    def test_an_installed_copy_is_brought_up_to_date(self):
        self.assertFalse(extension.refresh())                     # never installed: nothing is written
        self.assertFalse(extension.chrome_dir().joinpath("extension").exists())
        extension.install(roots=self.roots)
        self.assertFalse(extension.refresh())                     # already current
        copy = extension.chrome_dir() / "extension" / "background.js"
        copy.write_text("// an older version")
        (extension.chrome_dir() / "extension" / "gone.js").write_text("// removed since")
        self.assertTrue(extension.refresh())
        self.assertEqual(copy.read_bytes(), (extension.EXTENSION_SOURCE / "background.js").read_bytes())
        self.assertFalse((extension.chrome_dir() / "extension" / "gone.js").exists())

    def test_cli_connect_switches_the_mode(self):
        with mock.patch.object(extension, "browser_roots", return_value=self.roots), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["browser", "connect"]), 0)
            self.assertEqual(Config.load().get("browser.mode"), "chrome")
            self.assertEqual(cli.main(["browser", "disconnect"]), 0)
        self.assertEqual(Config.load().get("browser.mode"), "dedicated")
        self.assertEqual(extension.installed(self.roots), {"chrome": False})


if __name__ == "__main__":
    unittest.main()
