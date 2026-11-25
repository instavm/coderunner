"""A thread's trace for the developers (server/trace.py): what it holds, in what order, what it removes, who may have it,
and that the app offers it only in developer mode."""

import asyncio
import datetime
import io
import json
import pathlib
import re
import types
import unittest
import zipfile

import httpx

from coderunner import appshell
from coderunner.safety.policy import current_run
from coderunner.server import tasks as tasks_module
from coderunner.server import trace
from coderunner.server.gateway import admin_app
from fakes import Svc
from test_gateway import ADMIN, FakeDaemon

WEB = pathlib.Path(__file__).resolve().parent.parent / "coderunner" / "server" / "web" / "app"
AUTH = {"Authorization": "Bearer " + ADMIN}
SERVER = "http://10.20.30.40:8081/v1"


class TraceRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc({"models": {"chat": "qwen-test", "base_url": SERVER, "decision": "judge-small"},
                        "owner": {"name": "Priya Raman"}, "agent": {"step_tokens": 2048}})
        self.svc.llm = types.SimpleNamespace(model="qwen-test", base_url=SERVER)
        self.daemon = FakeDaemon(self.svc)
        self.gate = asyncio.Event()

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            # What the agent loop does, in the order it does it, with the audit written inside the run.
            token = current_run.set(types.SimpleNamespace(events=events))
            try:
                audit = self.svc.audit
                audit.write("agent.start", conversation=f"{channel}:{chat_id}", origin=origin)
                audit.write("agent.turn", step=0, think=None, read=7200, cached=0, wrote=310, read_ms=13400, wrote_ms=9500)
                events({"type": "note", "text": "Checking Skyscanner for the 12th, write to priya@example.com"})
                events({"type": "plan", "steps": [{"step": "Search fares", "status": "doing"}, {"step": "Book", "status": "todo"}]})
                events({"type": "tool", "id": "c1", "name": "browser_open",
                        "arguments": json.dumps({"url": "https://www.skyscanner.net/?token=abcdef123456", "say": "Opening"})})
                await asyncio.sleep(0.03)
                events({"type": "activity", "event": "browser.navigate", "url": "https://www.skyscanner.net/", "ok": True})
                events({"type": "tool_done", "id": "c1", "name": "browser_open", "ok": True, "preview": "page " * 60,
                        "result": "Flights from ₹4,812. Call +44 20 7946 0958. " + "x" * 3000})
                audit.write("decision.check", model="judge-small", p_commit=0.92, commits=True, seconds=0.4)
                audit.write("approval.requested", id="ap1", kind="browser.click", target="Book now",
                            summary="Book the 07:10 flight for ₹4,812")
                events({"type": "activity", "event": "approval.requested", "id": "ap1", "kind": "browser.click",
                        "target": "Book now", "summary": "Book the 07:10 flight for ₹4,812"})
                audit.write("agent.turn", step=1, think=False, read=400, cached=7200, wrote=60, read_ms=1300, wrote_ms=1900)
                events({"type": "plan", "steps": [{"step": "Search fares", "status": "done"}, {"step": "Book", "status": "doing"}]})
                events({"type": "tool", "id": "c2", "name": "run_shell", "arguments": '{"command": "cat /Users/priya/notes.txt"}'})
                events({"type": "tool_done", "id": "c2", "name": "run_shell", "ok": False, "preview": "Error: no such file"})
            finally:
                current_run.reset(token)
            # Decided from the app: outside the run, so by its approval id.
            self.svc.audit.write("approval.decided", id="ap1", status="approved", scope="once", decided_by="device:ab12cd34")
            events({"type": "activity", "event": "approval.decided", "id": "ap1", "status": "approved"})
            events({"type": "interim", "text": "So far: ₹4,812 on the 07:10."})
            await self.gate.wait()
            return "Booked the 07:10 flight for ₹4,812. Card 4111 1111 1111 1111 was not used."
        self.daemon.handle_message = handle
        self.bus = self.daemon.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = self.daemon.tasks = tasks_module.Tasks(self.daemon, self.bus)

    async def run_thread(self):
        first = self.tasks.start("app:fares", "Find the cheapest flight to Goa on the 12th", "owner", time_limit=600)
        await asyncio.sleep(0.15)
        self.gate.set()
        await self.tasks.wait(first["id"])
        self.bus.publish({"task": "", "conversation": "app:fares", "type": "title", "title": "Goa fares"})
        return first

    async def get(self, path, headers=AUTH):
        app = admin_app(self.svc, self.daemon)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            return await client.get(path, headers=headers)

    async def test_the_route_needs_the_owner(self):
        await self.run_thread()
        response = await self.get("/api/conversations/app:fares/trace", headers={})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("Goa", response.text)
        self.assertEqual((await self.get("/api/conversations/app:none/trace")).status_code, 404)
        self.assertEqual((await self.get("/api/conversations/app:fares/trace?format=exe")).status_code, 400)

    async def test_a_thread_comes_back_as_one_markdown_file_in_time_order(self):
        task = await self.run_thread()
        response = await self.get("/api/conversations/app:fares/trace")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/markdown"))
        disposition = response.headers["content-disposition"]
        today = datetime.date.today().isoformat()
        self.assertRegex(disposition, rf'^attachment; filename="coderunner-trace-find-the-cheapest-flight-to-goa[a-z0-9-]*-{today}\.md"$')
        md = response.text
        self.assertTrue(md.startswith("> CodeRunner trace: what the assistant saw and did"))
        self.assertIn("were removed", md.splitlines()[0])
        for part in ("## Task 1 · done", "**Request**", "Find the cheapest flight to Goa", "**Plan v1**", "**Plan v2**",
                     "  - [x] Search fares", "**Step 1 · browser\\_open** ok", "**Step 2 · run\\_shell** failed",
                     "Turn (step 0): read 7,200 new tokens", "Judge `decision.check`", "**Approval asked**",
                     "**Approval decided** status approved, scope once, decided_by device:ab12cd34",
                     "**Interim answer**", "**Ended: done**", "### Answer", "Booked the 07:10 flight",
                     "model qwen-test, slot 0", "time limit 600 s", "| Models | chat: qwen-test"):
            self.assertIn(part, md)
        # The timeline's entries are in the order they happened.
        order = [md.index(s) for s in ("+0.0 s **Request**", "Turn (step 0)", "**Plan v1**",
                                       "**Step 1", "Judge `decision.check`", "**Approval asked**", "Turn (step 1)",
                                       "**Plan v2**", "**Step 2", "**Approval decided**", "**Interim answer**", "**Ended: done**")]
        self.assertEqual(order, sorted(order))
        self.assertNotIn(task["id"] + "x", md)

    async def test_json_has_each_part_with_times_and_the_zip_has_both(self):
        await self.run_thread()
        response = await self.get("/api/conversations/app:fares/trace?format=json")
        self.assertEqual(response.status_code, 200)
        self.assertRegex(response.headers["content-disposition"], r'filename="coderunner-trace-[a-z0-9-]+-\d{4}-\d{2}-\d{2}\.json"')
        data = response.json()
        self.assertEqual(list(data)[:3], ["about", "format", "exported"])
        self.assertEqual(data["format"], "coderunner-trace/1")
        for key in ("app", "models", "browser", "limits", "settings", "thread", "tasks"):
            self.assertIn(key, data)
        self.assertEqual(data["models"]["chat"], "qwen-test")
        self.assertEqual(data["limits"]["step_tokens"], 2048)           # a count, not a secret
        task = data["tasks"][0]
        self.assertEqual((task["status"], task["ended"]["how"], task["model"], task["slot"]), ("done", "done", "qwen-test", 0))
        self.assertEqual(task["time_limit_s"], 600)
        timeline = task["timeline"]
        times = [e["t"] for e in timeline]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], 0)
        for entry in timeline:
            datetime.datetime.fromisoformat(entry["at"])                 # ISO, with the offset
        kinds = [e["kind"] for e in timeline]
        for kind in ("request", "status", "plan", "step", "turn", "judge", "approval", "interim", "end", "activity"):
            self.assertIn(kind, kinds)
        steps = [e for e in timeline if e["kind"] == "step"]
        self.assertEqual([(s["tool"], s["ok"]) for s in steps], [("browser_open", True), ("run_shell", False)])
        self.assertGreaterEqual(steps[0]["seconds"], 0.02)
        self.assertEqual(steps[0]["arguments"]["say"], "Opening")
        self.assertLess(len(steps[0]["result"]), trace.RESULT_CHARS + 60)    # the longer result, cut at 2 KB
        self.assertTrue(steps[0]["result"].startswith("Flights from"))
        plans = [e for e in timeline if e["kind"] == "plan"]
        self.assertEqual([p["version"] for p in plans], [1, 2])
        self.assertEqual(plans[1]["steps"][0], {"step": "Search fares", "status": "done"})
        approvals = [e for e in timeline if e["kind"] == "approval"]
        self.assertEqual([a["what"] for a in approvals], ["asked", "decided"])
        self.assertEqual(approvals[1]["decided_by"], "device:ab12cd34")
        self.assertEqual(task["totals"]["turns"], 2)
        self.assertEqual((task["totals"]["read"], task["totals"]["cached"], task["totals"]["wrote"]), (7600, 7200, 370))
        self.assertEqual((task["totals"]["steps"], task["totals"]["failed_steps"], task["totals"]["approvals"]), (2, 1, 1))
        response = await self.get("/api/conversations/app:fares/trace?format=zip")
        self.assertEqual(response.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(sorted(archive.namelist()), ["trace.json", "trace.md"])
            self.assertEqual(json.loads(archive.read("trace.json"))["tasks"][0]["id"], task["id"])

    async def test_what_leaves_the_computer_is_scrubbed(self):
        await self.run_thread()
        md = (await self.get("/api/conversations/app:fares/trace")).text
        data = (await self.get("/api/conversations/app:fares/trace?format=json")).text
        self.assertIn("\\[email\\]", md)                           # a status line: escaped as Markdown
        self.assertIn("[email]", data)
        for text in (md, data):
            for secret in ("priya@example.com", "7946 0958", "4111 1111 1111 1111", "10.20.30.40", "abcdef123456",
                           "/Users/priya", "Priya", ADMIN):
                self.assertNotIn(secret, text)
            self.assertIn("[number]", text)
            self.assertIn("[card number]", text)
            self.assertIn("~/notes.txt", text)
            self.assertIn("₹4,812", text)                          # prices and times stay
            self.assertIn("07:10", text)
        self.assertNotIn("10.20.30.40", json.dumps(json.loads(data)["models"]))


class ScrubberTests(unittest.TestCase):
    def setUp(self):
        self.scrub = trace.Scrubber([("admin-token-0123456789abcdefghij", "[redacted token]"), ("Priya Raman", "[owner]")],
                                    ["http://gpu-box.lan:8080/v1", "gpu-box.lan:8080"])

    def test_each_kind_of_identifier(self):
        # Key-shaped values are written in pieces, so the file itself holds none whole for a secret scanner.
        cases = {
            "key sk-" + "ant-api03-AbCdEfGhIjKlMnOpQrStUvWx": "key [redacted",
            "gh gh" + "p_0123456789abcdefABCDEF0123456789ab": "gh [redacted",
            "aws AKIA" + "IOSFODNN7EXAMPLE and hf" + "_AbCdEfGhIjKlMnOpQrStUv": "aws [redacted key] and [redacted key]",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop": "[redacted",
            "password=hunter2secret": "password=[redacted]",
            "Set-Cookie: session_id=Zx81kQ7Lm2Np": "[redacted]",
            "Your verification code is 482913.": "[redacted code]",
            "write to someone.name+tag@mail.example.co.uk": "[email]",
            "call +1 (415) 555-2671 now": "[number]",
            "card 4242-4242-4242-4242": "[card number]",
            "IBAN GB82 WEST 1234 5698 7654 32": "[account number]",
            "account 0012345678901": "[number]",
            "from 192.168.1.20 and fe80::1ff:fe23:4567:890a": "[ip]",
            "server http://gpu-box.lan:8080/v1/chat": "[model server]",
            "local http://127.0.0.1:8224/api": "[local server]",
            "open /Users/someone/Documents/lease.pdf": "~/Documents/lease.pdf",
            "token admin-token-0123456789abcdefghij": "[redacted token]",
            "for Priya Raman": "[owner]",
            "https://example.com/reset?token=abc123def456&x=1": "[redacted",
        }
        for text, mask in cases.items():
            out = self.scrub.text(text)
            self.assertIn(mask, out, text)
        self.assertNotIn("fe80::1ff", self.scrub.text("from 192.168.1.20 and fe80::1ff:fe23:4567:890a"))
        self.assertNotIn("hunter2", self.scrub.text("password=hunter2secret"))

    def test_ordinary_text_stays(self):
        for text in ("₹4,812 for 2 adults", "on 2026-11-25 at 19:05", "flight AI 2845, gate 12", "https://www.gov.uk/passport-fees",
                     "room 1204, 3 nights", "version 0.2.0", "read 7,221 tokens"):
            self.assertEqual(self.scrub.text(text), text)

    def test_fields_named_for_secrets_go_whole_and_counts_stay(self):
        out = self.scrub.value({"api_key": "plain", "cookie": "a=b", "access_token": "x", "step_tokens": 2048,
                                "nested": [{"password": "p"}]})
        self.assertEqual(out, {"api_key": "[redacted]", "cookie": "[redacted]", "access_token": "[redacted]",
                               "step_tokens": 2048, "nested": [{"password": "[redacted]"}]})

    def test_markdown_keeps_page_text_from_becoming_markup(self):
        data = {"about": trace.ABOUT, "format": trace.FORMAT, "exported": "2026-11-25T10:00:00.000+00:00",
                "app": {"version": "0", "commit": "", "os": "x", "machine": "y", "python": "3"}, "models": {},
                "browser": {}, "limits": {}, "settings": {},
                "thread": {"title": "<script>alert(1)</script>", "tasks": 1, "started": None},
                "tasks": [{"number": 1, "id": "t", "status": "done", "started": None, "seconds": 1.0, "request": "hi",
                           "ended": {"how": "done"}, "answer": "```\n<img src=x onerror=alert(1)>",
                           "totals": {"steps": 0, "failed_steps": 0, "turns": 0, "read": 0, "cached": 0, "wrote": 0,
                                      "read_s": 0, "wrote_s": 0, "approvals": 0},
                           "timeline": [{"at": "2026-11-25T10:00:00.000+00:00", "t": 0, "kind": "status",
                                         "text": "<script>bad()</script> **bold**"}]}]}
        md = trace.to_markdown(data)
        self.assertNotIn("<script>", md)
        self.assertIn("\\<script\\>", md)
        self.assertIn("````\n```\n<img", md)            # inside a longer fence than any it holds


class AuditByTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_written_in_a_task_carry_its_id(self):
        svc = Svc({"models": {"chat": "m"}})
        daemon = FakeDaemon(svc)

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            token = current_run.set(types.SimpleNamespace(events=events))
            try:
                svc.audit.write("agent.turn", step=0, read=1)
            finally:
                current_run.reset(token)
            return "ok"
        daemon.handle_message = handle
        daemon.bus = tasks_module.EventBus(svc.store)
        daemon.tasks = tasks_module.Tasks(daemon, daemon.bus)
        task = daemon.tasks.start("app:a", "hi", "owner")
        await daemon.tasks.wait(task["id"])
        svc.audit.write("daemon.started")
        records = [json.loads(line) for line in svc.audit.path.read_text().splitlines()]
        turn = next(r for r in records if r["event"] == "agent.turn")
        self.assertEqual(turn["task"], task["id"])
        running = next(r for r in records if r["event"] == "task.running")
        self.assertEqual((running["task"], running["slot"], running["model"]), (task["id"], 0, "m"))
        self.assertNotIn("task", records[-1])                    # outside any task: no id

    def test_the_audit_file_is_read_between_two_times(self):
        svc = Svc()
        lines = [json.dumps({"ts": 1000.0 + i, "event": "e", "n": i}) for i in range(5000)]
        svc.audit.path.write_text("\n".join(lines) + "\n")
        found = trace._audit_between(svc.audit.path, 3000.5, 3010)
        self.assertEqual([r["n"] for r in found], list(range(2001, 2011)))
        self.assertEqual(trace._audit_between(svc.audit.path / "missing", 0, 1), [])


class DeveloperModeTests(unittest.TestCase):
    def test_off_by_default_and_the_button_only_shows_when_on(self):
        markup, page = (WEB / "index.html").read_text(), (WEB / "app.js").read_text()
        button = re.search(r'<button id="exportTrace"[^>]*>', markup).group(0)
        self.assertIn(" hidden", button)                        # drawn hidden: nothing shows until it is turned on
        self.assertIn('title="Export trace"', button)
        self.assertRegex(re.search(r'<div id="traceAsk"[^>]*>', markup).group(0), r"\bhidden\b")
        self.assertIn("This file has what the assistant saw and did in this thread, with keys and contact details "
                      "removed. Check it before sending.", markup)
        self.assertIn("developer: false,", page)
        self.assertIn('const on = state.developer && typeof state.conversation === "string" && '
                      'state.conversation.startsWith("app:");', page)
        self.assertIn('$("exportTrace").hidden = !on;', page)
        self.assertIn("function setDeveloper(on) { state.developer = on === true;", page)
        # Only one place shows it, and the save itself checks the mode again.
        self.assertEqual(page.count('$("exportTrace").hidden ='), 1)
        self.assertIn("if (!state.developer ||", page)
        self.assertIn("#exportTrace[hidden], .trace-ask[hidden] { display: none; }", (WEB / "app.css").read_text())

    def test_the_shell_keeps_it_off_until_the_menu_turns_it_on(self):
        main = (appshell.app_dir() / "main.js").read_text()
        preload = (appshell.app_dir() / "preload.js").read_text()
        self.assertIn('{label: "Developer Mode", type: "checkbox", checked: developerMode(), '
                      'click: (item) => setDeveloperMode(item.checked)}', main)
        self.assertIn("return developerThisRun ?? settings().developerMode === true;", main)
        self.assertIn('win.webContents.send("developer-mode", Boolean(on))', main)
        self.assertIn('developerMode: () => ipcRenderer.invoke("developer-mode")', preload)


if __name__ == "__main__":
    unittest.main()
