"""Fixes on the owner's Mac through macOS's own password prompt (core/macfix.py): told by the files helper's OCR
fallback, run as a constant command with administrator privileges, checked after, and only from the owner's app.
osascript never runs here: every run gets a stand-in for the subprocess."""

import asyncio
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.core import macfix
from coderunner.files import extract

ADMIN_ARGV = ["osascript", "-e", 'do shell script "killall ANECompilerService aned" with administrator privileges']


def fake_helper(folder: pathlib.Path) -> pathlib.Path:
    """A files-helper stand-in: reads picture paths, says "Lighthouse", with "fallback" while a flag file exists."""
    script = folder / "files-helper"
    script.write_text(f"#!{sys.executable}\n"
                      "import json, os, sys\n"
                      "flag = os.path.join(os.path.dirname(sys.argv[0]), 'fallback')\n"
                      "for line in sys.stdin:\n"
                      "    reply = {'path': line.strip(), 'text': 'Lighthouse'}\n"
                      "    if os.path.exists(flag): reply['fallback'] = True\n"
                      "    print(json.dumps(reply), flush=True)\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class Proc:
    def __init__(self, code=0, stderr=b""):
        self.returncode, self.stderr = code, stderr

    async def communicate(self, data=None):
        return b"", self.stderr

    def kill(self):
        pass


class Spawn:
    def __init__(self, proc):
        self.proc, self.calls = proc, []

    async def __call__(self, *argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return self.proc


class Audit:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class DetectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.helper = fake_helper(self.tmp)
        extract.ocr_reset()
        self.addCleanup(extract.ocr_reset)
        self.picture = self.tmp / "receipt.png"
        self.picture.write_bytes(b"png")

    async def test_the_helpers_fallback_is_noticed_and_offered_as_a_fix_on_a_mac(self):
        media = extract.Media()
        with mock.patch.object(extract.Media, "helper", mock.AsyncMock(return_value=str(self.helper))):
            self.assertEqual(await media.read_text_in(self.picture), "Lighthouse")
            self.assertFalse(extract.ocr_fell_back())
            (self.tmp / "fallback").write_text("")
            self.assertEqual(await media.read_text_in(self.picture), "Lighthouse")
            self.assertTrue(extract.ocr_fell_back())
            with mock.patch.object(macfix.sys, "platform", "darwin"):
                self.assertEqual([f.id for f in macfix.detected()], ["neural-engine"])
                self.assertEqual(macfix.detected()[0].title, "Text in pictures is being read at lower quality.")
            with mock.patch.object(macfix.sys, "platform", "linux"):
                self.assertEqual(macfix.detected(), [])            # not a Mac: no fixes at all
                self.assertIsNone(macfix.find("neural-engine"))
            # Fixed: forgotten, and the running helper (stuck on the fast model) is started afresh.
            first = media._ocr
            extract.ocr_reset()
            (self.tmp / "fallback").unlink()
            self.assertFalse(extract.ocr_fell_back())
            await media.read_text_in(self.picture)
            self.assertIsNot(media._ocr, first)
            self.assertFalse(extract.ocr_fell_back())
            await media.close()

    async def test_a_pdf_says_fallback_only_when_a_scan_needed_it(self):
        media = extract.Media()
        replies = [json.dumps({"pages": ["typed text"], "total": 1}).encode(),
                   json.dumps({"pages": ["scan"], "total": 1, "fallback": True}).encode()]
        with mock.patch.object(extract.Media, "helper", mock.AsyncMock(return_value="helper")), \
                mock.patch.object(extract.Media, "_run", mock.AsyncMock(side_effect=replies)):
            await media.pages(self.picture)
            self.assertFalse(extract.ocr_fell_back())
            await media.pages(self.picture)
            self.assertTrue(extract.ocr_fell_back())

    async def test_the_check_reads_the_built_in_picture_on_a_fresh_helper(self):
        self.assertTrue(macfix.CHECK_PICTURE.is_file())
        with mock.patch.object(extract.Media, "helper", mock.AsyncMock(return_value=str(self.helper))):
            self.assertTrue(await macfix._accurate_ocr_works())
            (self.tmp / "fallback").write_text("")
            self.assertFalse(await macfix._accurate_ocr_works())       # read, but only by the fast model
        with mock.patch.object(extract.Media, "helper", mock.AsyncMock(return_value=None)):
            self.assertFalse(await macfix._accurate_ocr_works())       # no helper: cannot say it is fixed


class RunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(macfix, "SETTLE_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)
        extract.ocr_reset()
        self.addCleanup(extract.ocr_reset)

    def fix(self, ok=True):
        self.checked = []

        async def check():
            self.checked.append(1)
            return ok

        return macfix.Fix(id="neural-engine", title="t", detect=lambda: True, command=macfix.FIXES[0].command,
                          check=check, not_fixed=macfix.FIXES[0].not_fixed)

    async def test_the_command_is_constant_and_asks_for_an_administrator(self):
        self.assertEqual(macfix.FIXES[0].command, "killall ANECompilerService aned")
        self.assertEqual(macfix.admin_argv(macfix.FIXES[0].command), ADMIN_ARGV)
        spawn, audit = Spawn(Proc(0)), Audit()
        extract._OCR["fallback"] = True
        result = await macfix.run(self.fix(True), audit=audit, spawn=spawn)
        argv, kwargs = spawn.calls[0]
        self.assertEqual(argv, ADMIN_ARGV)                 # an argument list: no shell anywhere
        self.assertNotIn("shell", kwargs)
        self.assertEqual(result, {"id": "neural-engine", "result": "fixed", "message": "Fixed."})
        self.assertEqual(self.checked, [1])
        self.assertFalse(extract.ocr_fell_back())          # fixed: the line goes, the helpers start afresh
        self.assertEqual(audit.rows, [("macfix.run", {"id": "neural-engine", "result": "fixed"})])

    async def test_cancel_in_the_dialog_is_not_done_and_nothing_is_checked(self):
        spawn, audit = Spawn(Proc(1, b"0:39: execution error: User canceled. (-128)\n")), Audit()
        result = await macfix.run(self.fix(True), audit=audit, spawn=spawn)
        self.assertEqual(result["result"], "cancelled")
        self.assertEqual(result["message"], "Not done.")
        self.assertEqual(self.checked, [])
        self.assertEqual(audit.rows, [("macfix.run", {"id": "neural-engine", "result": "cancelled"})])

    async def test_the_check_decides_fixed_or_not(self):
        # killall found one of the two not running: it says so and exits 1, and the check still decides.
        result = await macfix.run(self.fix(False), spawn=Spawn(Proc(1, b"No matching processes were found\n")))
        self.assertEqual(result, {"id": "neural-engine", "result": "not_fixed",
                                  "message": "Not fixed. Restarting your Mac also fixes this."})
        self.assertEqual(self.checked, [1])
        result = await macfix.run(self.fix(True), spawn=Spawn(Proc(1, b"No matching processes were found\n")))
        self.assertEqual(result["result"], "fixed")

    async def test_one_dialog_at_a_time(self):
        gate = asyncio.Event()

        class Slow(Proc):
            async def communicate(self, data=None):
                await gate.wait()
                return b"", b""

        first = asyncio.create_task(macfix.run(self.fix(True), spawn=Spawn(Slow(0))))
        await asyncio.sleep(0)
        self.assertEqual((await macfix.run(self.fix(True), spawn=Spawn(Proc(0))))["result"], "busy")
        gate.set()
        self.assertEqual((await first)["result"], "fixed")

    def test_no_command_could_leave_its_applescript_string(self):
        for fix in macfix.FIXES:
            self.assertNotRegex(fix.command, r"[\"\\'$`;|&<>\n]")


class RouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from fakes import Svc
        from test_gateway import ADMIN, SANDBOX, FakeDaemon
        from coderunner.agent import build_registry
        from coderunner.server.gateway import admin_app
        self.admin, self.sandbox = {"Authorization": "Bearer " + ADMIN}, {"Authorization": "Bearer " + SANDBOX}
        self.svc = Svc()
        self.svc.registry = build_registry(self.svc)
        self.app = admin_app(self.svc, FakeDaemon(self.svc))
        extract.ocr_reset()
        self.addCleanup(extract.ocr_reset)

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1:8224")

    async def test_owner_only_known_ids_and_mac_only(self):
        ran = mock.AsyncMock(return_value={"id": "neural-engine", "result": "fixed", "message": "Fixed."})
        with mock.patch.object(macfix, "run", ran), mock.patch.object(macfix.sys, "platform", "darwin"):
            async with self.client() as client:
                for headers in ({}, self.sandbox):
                    self.assertEqual((await client.get("/api/fixes", headers=headers)).status_code, 401)
                    self.assertEqual((await client.post("/api/fixes/neural-engine", headers=headers)).status_code, 401)
                self.assertEqual((await client.get("/api/fixes", headers=self.admin)).json(), {"fixes": []})
                extract._OCR["fallback"] = True
                self.assertEqual((await client.get("/api/fixes", headers=self.admin)).json(), {"fixes": [
                    {"id": "neural-engine", "title": "Text in pictures is being read at lower quality."}]})
                refused = await client.post("/api/fixes/rm-rf", headers=self.admin)
                self.assertEqual(refused.status_code, 404)
                response = await client.post("/api/fixes/neural-engine", headers=self.admin)
                self.assertEqual(response.json()["result"], "fixed")
            self.assertEqual(ran.await_count, 1)
            self.assertIs(ran.await_args.args[0], macfix.FIXES[0])
        with mock.patch.object(macfix, "run", ran), mock.patch.object(macfix.sys, "platform", "linux"):
            async with self.client() as client:
                self.assertEqual((await client.get("/api/fixes", headers=self.admin)).json(), {"fixes": []})
                self.assertEqual((await client.post("/api/fixes/neural-engine", headers=self.admin)).status_code, 404)
        self.assertEqual(ran.await_count, 1)

    def test_the_assistant_has_no_tool_that_runs_a_fix(self):
        names = " ".join(self.svc.registry.tools)
        self.assertNotIn("fix", names)
        tools_source = (pathlib.Path(macfix.__file__).parent.parent / "agent" / "tools.py").read_text()
        self.assertNotIn("macfix", tools_source)


class AppTests(unittest.TestCase):
    def test_one_line_and_one_button(self):
        web = pathlib.Path(macfix.__file__).parent.parent / "server" / "web" / "app"
        page, markup = (web / "app.js").read_text(), (web / "index.html").read_text()
        self.assertIn('<div id="fixes" class="panel-section fixes" hidden></div>', markup)
        self.assertIn('el("button", "btn primary fix-go", "Fix")', page)
        self.assertIn('api(`/api/fixes/${encodeURIComponent(shown.fix.id)}`, {method: "POST", body: "{}"})', page)


if __name__ == "__main__":
    unittest.main()
