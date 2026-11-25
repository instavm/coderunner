"""The owner's terminal on the assistant's computer: the VM's PTY session (terminal.py, run here as the current
user), the assistant's commands echoed into it, the gateway's relay, and the assistant waiting while the owner
types."""

import asyncio
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import shell_session  # noqa: E402
import terminal  # noqa: E402

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import WebSocketRoute  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

BASH = ["bash", "--noprofile", "--norc"]
TOKEN = "terminal-test-credential-0123456789abcdef"


async def until(term_or_queue, needle: bytes, seconds: float = 10.0) -> bytes:
    """What a queue delivers until `needle` appears in it."""
    seen = bytearray()
    deadline = time.monotonic() + seconds
    while needle not in seen:
        left = deadline - time.monotonic()
        if left <= 0:
            raise AssertionError(f"{needle!r} not seen in {bytes(seen)[-400:]!r}")
        chunk = await asyncio.wait_for(term_or_queue.get(), left)
        if chunk is None:
            break
        seen += chunk
    return bytes(seen)


class TerminalSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cwd = tempfile.mkdtemp()
        self.term = terminal.Terminal(argv=BASH, cwd=self.cwd)

    async def asyncTearDown(self):
        await self.term.stop()

    async def test_a_real_pty_echoes_runs_and_resizes(self):
        self.term.start()
        queue = self.term.attach()
        self.term.write(b"echo hel''lo; echo $TERM; pwd\r")
        seen = await until(queue, b"xterm-256color")
        self.assertIn(b"hello", seen)
        self.term.resize(120, 40)
        self.term.write(b"stty size\r")
        await until(queue, b"40 120")
        self.term.write(b"tty\r")
        await until(queue, b"/dev/")             # a terminal of its own, not a pipe
        self.term.write(b"cd /; pwd -P\r")
        self.assertTrue(self.term.running())
        self.assertTrue(self.term.typing())

    async def test_starts_in_its_folder_and_binary_input_passes_through(self):
        self.term.start()
        queue = self.term.attach()
        self.term.write(b"pwd -P; printf '\\xc3\\xa9\\n'\r")
        seen = await until(queue, "é".encode())
        self.assertIn(str(pathlib.Path(self.cwd).resolve()).encode(), seen)

    async def test_a_reconnect_replays_what_was_kept_and_the_shell_lives_on(self):
        self.term.start()
        first = self.term.attach()
        self.term.write(b"export KEPT=yes-still; echo marker-one\r")
        await until(first, b"marker-one\r\n")
        self.term.detach(first)                    # the tab closed; the shell keeps running
        second = self.term.attach()
        replay = await asyncio.wait_for(second.get(), 2)
        self.assertIn(b"marker-one", replay)
        self.term.write(b"echo $KEPT\r")
        await until(second, b"yes-still\r\n")

    async def test_the_scrollback_is_capped_and_starts_on_a_line(self):
        term = terminal.Terminal(argv=BASH, scrollback=1000)
        for n in range(400):
            term._feed(f"line {n:04d}\r\n".encode())
        self.assertLessEqual(len(term.buffer), 1000)
        self.assertTrue(term.buffer.startswith(b"line "))
        self.assertTrue(term.buffer.endswith(b"line 0399\r\n"))

    async def test_a_shell_that_exits_says_so_and_the_next_key_starts_another(self):
        self.term.start()
        queue = self.term.attach()
        self.term.write(b"exit\r")
        await until(queue, b"[shell ended]")
        self.assertFalse(self.term.running())
        self.term.write(b"x")
        self.assertTrue(self.term.running())

    async def test_the_assistants_commands_are_shown_dim_with_their_output_and_a_failing_exit(self):
        queue = self.term.attach()             # the echo needs no shell of the owner's
        echo = self.term.assistant(terminal.shell_label("ls -la /missing\necho second line"))
        echo.output(b"ls: cannot access\n")
        echo.end(2)
        seen = await until(queue, b"exit 2")
        self.assertIn(terminal.DIM + terminal.MARK + "ls -la /missing …".encode() + terminal.RESET, seen)
        self.assertIn(b"ls: cannot access\r\n", seen)
        ok = self.term.assistant(terminal.python_label("print(1)"))
        ok.end(0, output="1\n")
        self.assertIn(b"python  print(1)", bytes(self.term.buffer))
        self.assertFalse(bytes(self.term.buffer).endswith(b"exit 0" + terminal.RESET + b"\r\n"))
        self.assertNotIn(b"\x1b]", terminal.DIM + terminal.MARK + terminal.shell_label("\x1b]0;x").encode())

    async def test_the_assistants_output_streams_from_its_own_shell_without_the_marker(self):
        shells = shell_session.Shells(home=self.cwd, jobs=str(pathlib.Path(self.cwd) / "jobs"))
        chunks = []
        try:
            first = "one " + "x" * 60          # longer than the bytes held back in case they start the marker
            result = await shells.run("s", f"echo {first}; sleep 0.3; echo two; false", on_output=chunks.append)
        finally:
            await shells.reset("s")
        self.assertEqual(b"".join(chunks), f"{first}\ntwo\n".encode())
        self.assertEqual((result.output, result.exit_code), (f"{first}\ntwo", 1))
        self.assertGreater(len(chunks), 1)             # "one" came before the command ended


def vm_app(term):
    from auth import BearerAuthentication
    app = Starlette(routes=[WebSocketRoute("/v1/terminal", terminal.terminal_socket)])
    return BearerAuthentication(app, token_provider=lambda: TOKEN)


def read_until(ws, needle: bytes) -> bytes:
    seen = bytearray()
    while needle not in seen:
        seen += ws.receive_bytes()
    return bytes(seen)


class TerminalRouteTests(unittest.TestCase):
    def test_the_route_needs_the_token_relays_bytes_resizes_and_replays_on_reconnect(self):
        term = terminal.Terminal(argv=BASH, cwd=tempfile.mkdtemp())
        with mock.patch.object(terminal, "TERMINAL", term), TestClient(vm_app(term)) as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/v1/terminal") as ws:
                    ws.receive_bytes()
            auth = {"Authorization": f"Bearer {TOKEN}"}
            with client.websocket_connect("/v1/terminal", headers=auth) as ws:
                ws.send_text('{"type": "resize", "cols": 91, "rows": 27}')
                ws.send_bytes(b"stty size; echo first-visit\r")
                read_until(ws, b"27 91")
            with client.websocket_connect("/v1/terminal", headers=auth) as ws:
                self.assertIn(b"first-visit", read_until(ws, b"first-visit"))
                ws.send_bytes(b"echo again\r")
                read_until(ws, b"again\r\n")
            client.portal.call(term.stop)


class OwnerTypingTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        from coderunner.sandbox.owner_terminal import OwnerTerminal
        now = [1000.0]

        async def sleep(seconds):
            now[0] += seconds

        return OwnerTerminal(clock=lambda: now[0], sleep=sleep), now

    async def test_no_wait_without_typing_and_a_wait_until_a_pause(self):
        use, now = self.make()
        said = []
        self.assertEqual(await use.wait_for_pause(lambda: said.append(1)), 0)
        self.assertEqual(said, [])
        use.typed()
        now[0] += 5
        waited = await use.wait_for_pause(lambda: said.append(1))
        self.assertAlmostEqual(waited, 10, delta=0.2)          # 15 s after the last key
        self.assertEqual(said, [1])

    async def test_a_minute_at_most_while_the_owner_keeps_typing(self):
        use, now = self.make()
        real = use.sleep

        async def typing_on(seconds):
            await real(seconds)
            use.typed()

        use.sleep = typing_on
        use.typed()
        self.assertAlmostEqual(await use.wait_for_pause(), 60, delta=1.1)

    async def test_run_shell_and_run_python_wait_while_the_owner_types_and_say_so(self):
        from fakes import Svc
        from coderunner.agent.tools import sandbox_tools
        from coderunner.sandbox.owner_terminal import WAITING
        use, now = self.make()
        order = []

        class Sandbox:
            async def shell(self, command, session, timeout, background):
                order.append(("shell", now[0]))
                return {"output": "ok", "exit_code": 0, "cwd": "/workspace"}

            async def start_session(self, session):
                return {}

            async def execute(self, code, session_id=None):
                order.append(("python", now[0]))
                return {"stdout": "2", "stderr": ""}

        svc = Svc({"policy": {"sandbox.exec": "allow"}})
        svc.sandbox, svc.terminal = Sandbox(), use
        tools = {t.name: t for t in sandbox_tools(svc)}
        emitted = []
        ctx = mock.Mock(content_tainted=False, private_sources=False, origin="test", events=None,
                        emit=emitted.append)
        with mock.patch.object(svc.policy, "check", mock.AsyncMock(return_value=True)):
            use.typed()
            started = now[0]
            await tools["run_shell"].handler({"command": "ls"}, ctx)
            self.assertGreaterEqual(order[-1][1] - started, 15 - 0.2)
            self.assertEqual(emitted, [{"type": "note", "text": WAITING}])
            emitted.clear()
            await tools["run_python"].handler({"code": "print(1+1)"}, ctx)       # nobody typing: no wait
            self.assertEqual(order[-1], ("python", order[-1][1]))
            self.assertEqual(emitted, [])
            use.typed()
            await tools["run_python"].handler({"code": "print(1+1)"}, ctx)
            self.assertEqual(emitted, [{"type": "note", "text": WAITING}])


class GatewayRelayTests(unittest.TestCase):
    """/api/computer/terminal: the owner's credential, relayed to the VM's /v1/terminal with the sandbox's."""

    def setUp(self):
        from websockets.sync.server import serve
        self.seen = {"headers": [], "frames": []}

        def handler(connection):
            self.seen["headers"].append(connection.request.headers.get("Authorization"))
            connection.send(b"welcome\r\n")
            for message in connection:
                self.seen["frames"].append(message)
                if isinstance(message, bytes):
                    connection.send(b"got:" + message)

        self.server = serve(handler, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)

    def test_owner_auth_then_bytes_and_resizes_pass_and_keys_mark_typing(self):
        from fakes import Svc
        from test_gateway import ADMIN, SANDBOX, FakeDaemon
        from coderunner.agent import build_registry
        from coderunner.sandbox.client import SandboxClient
        from coderunner.sandbox.owner_terminal import OwnerTerminal
        from coderunner.server.gateway import admin_app
        svc = Svc()
        svc.registry = build_registry(svc)
        svc.sandbox = SandboxClient(f"http://127.0.0.1:{self.port}", SANDBOX)
        svc.terminal = OwnerTerminal()
        app = admin_app(svc, FakeDaemon(svc))
        with TestClient(app, base_url="http://127.0.0.1:8224") as client:
            for headers in ({}, {"Authorization": "Bearer " + SANDBOX}):
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect("/api/computer/terminal", headers=headers) as ws:
                        ws.receive_bytes()
            self.assertEqual(self.seen["headers"], [])                 # nothing reached the VM
            self.assertFalse(svc.terminal.typing())
            with client.websocket_connect("/api/computer/terminal", headers={"Authorization": "Bearer " + ADMIN}) as ws:
                self.assertEqual(ws.receive_bytes(), b"welcome\r\n")
                ws.send_text('{"type":"resize","cols":80,"rows":24}')
                self.assertFalse(svc.terminal.typing())                  # a resize is not typing
                ws.send_bytes(b"ls\r")
                self.assertEqual(ws.receive_bytes(), b"got:ls\r")
            self.assertTrue(svc.terminal.typing())
            self.assertEqual(self.seen["headers"], ["Bearer " + SANDBOX])
            self.assertEqual(self.seen["frames"], ['{"type":"resize","cols":80,"rows":24}', b"ls\r"])
            events = [line for line in svc.audit.path.read_text().splitlines() if "terminal_opened" in line]
            self.assertEqual(len(events), 1)

    def test_no_vm_closes_quietly_and_asks_the_keeper(self):
        from fakes import Svc
        from test_gateway import ADMIN, FakeDaemon
        from coderunner.agent import build_registry
        from coderunner.sandbox.client import SandboxClient
        from coderunner.server.gateway import admin_app
        svc = Svc()
        svc.registry = build_registry(svc)
        svc.sandbox = SandboxClient("http://127.0.0.1:9", "x" * 40)          # nothing listens
        svc.computer = mock.Mock()
        app = admin_app(svc, FakeDaemon(svc))
        with TestClient(app, base_url="http://127.0.0.1:8224") as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/api/computer/terminal", headers={"Authorization": "Bearer " + ADMIN}) as ws:
                    ws.receive_bytes()
        svc.computer.kick.assert_called_once_with(desktop=True)


if __name__ == "__main__":
    unittest.main()
