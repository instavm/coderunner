"""Files the owner attaches to a task in the app: kept, served back, and handed to the model."""

import asyncio
import base64
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.agent import build_registry
from coderunner.files import attached
from coderunner.files.discover import FileAccessError, Folders
from coderunner.server import tasks as tasks_module
from coderunner.server.gateway import admin_app
from fakes import Svc
from test_gateway import ADMIN, FakeDaemon

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


async def chunks(*parts):
    for part in parts:
        yield part


class Home(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = pathlib.Path(self._tmp.name).resolve()
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)


class KeepingTests(Home):
    async def test_a_file_is_kept_under_its_own_id_with_its_own_name(self):
        kept = await attached.save(chunks(PNG[:20], PNG[20:]), "Holiday in Goa.png", 1 << 20)
        self.assertRegex(kept["id"], r"^[0-9a-f]{12}$")
        self.assertEqual((kept["name"], kept["mime"], kept["kind"], kept["size"], kept["inline"]),
                         ("Holiday in Goa.png", "image/png", "image", len(PNG), True))
        self.assertEqual(kept["url"], f"/api/uploads/{kept['id']}")
        path = attached.path_of(kept["id"])
        self.assertEqual((path.read_bytes(), path.parent.parent), (PNG, self.home / "assets" / "outputs" / "attached"))
        self.assertEqual(attached.describe(kept["id"]), kept)

    async def test_a_name_cannot_leave_its_folder_or_hide(self):
        for given, kept in (("../../etc/passwd", "passwd"), ("C:\\Users\\me\\tax 2025.pdf", "tax 2025.pdf"),
                            (".env", "env"), ("", "file"), ("a\x00b\n.txt", "ab.txt"), ("x" * 300 + ".jpeg", "x" * 100 + ".jpeg")):
            self.assertEqual(attached.safe_name(given), kept)

    async def test_one_that_is_too_large_or_empty_leaves_nothing_behind(self):
        with self.assertRaises(attached.TooLarge):
            await attached.save(chunks(b"x" * 600, b"x" * 600), "big.bin", 1000)
        with self.assertRaises(ValueError):
            await attached.save(chunks(), "empty.txt", 1000)
        self.assertEqual(list(attached.attached_dir().iterdir()), [])

    async def test_only_a_plain_file_of_ours_is_found_by_an_id(self):
        kept = await attached.save(chunks(b"notes"), "notes.txt", 1000)
        self.assertIsNone(attached.path_of("../" + kept["id"]))
        self.assertIsNone(attached.path_of("0123456789ab"))
        # Code in the sandbox can write into this folder: a link it leaves there is not followed.
        secret = self.home / "assistant" / "admin-token"
        secret.parent.mkdir(parents=True)
        secret.write_text("the admin token")
        planted = attached.attached_dir() / "aaaaaaaaaaaa"
        planted.mkdir()
        (planted / "token.txt").symlink_to(secret)
        self.assertIsNone(attached.path_of("aaaaaaaaaaaa"))
        self.assertIsNone(attached.describe("aaaaaaaaaaaa"))
        both = attached.path_of(kept["id"]).parent / "second.txt"       # or a second file beside the owner's
        both.write_text("added")
        self.assertIsNone(attached.path_of(kept["id"]))

    async def test_the_file_tools_may_read_an_attached_file_and_nothing_else_under_the_home(self):
        kept = await attached.save(chunks(b"notes"), "notes.txt", 1000)
        folders = Folders(Svc().config)                                  # no folder is shared at all
        path = attached.path_of(kept["id"])
        self.assertEqual(folders.resolve(str(path)), path.resolve())
        # The model is told the sandbox's path for it too, and gave that one to file_view.
        self.assertEqual(folders.resolve(f"/app/uploads/outputs/attached/{kept['id']}/notes.txt"), path.resolve())
        for elsewhere in (f"/app/uploads/outputs/attached/{kept['id']}/../../x", "/app/uploads/outputs/attached/zz/notes.txt",
                          "/app/uploads/outputs/made-by-code.txt"):
            with self.assertRaises(FileAccessError):
                folders.resolve(elsewhere)
        secret = self.home / "assistant" / "admin-token"
        secret.parent.mkdir(parents=True)
        secret.write_text("the admin token")
        planted = attached.attached_dir() / "bbbbbbbbbbbb"
        planted.mkdir()
        (planted / "token.txt").symlink_to(secret)
        for refused in (secret, planted / "token.txt", self.home / "assets" / "outputs" / "made-by-code.txt",
                        attached.attached_dir() / kept["id"]):
            with self.assertRaises(FileAccessError):
                folders.resolve(str(refused))


class ModelTests(Home):
    class Media:
        def __init__(self):
            self.asked = []

        async def image(self, path, at=None, size=1024):
            self.asked.append((path.name, at))
            if path.name == "broken.png":
                raise RuntimeError("not a picture")
            return b"jpeg of " + path.name.encode()

        async def duration(self, path):
            return 7.6

    async def test_the_model_is_told_where_each_file_is_and_shown_the_pictures(self):
        ids = [(await attached.save(chunks(b"x" * size), name, 1 << 22))["id"]
               for name, size in (("goa.png", 2_100_000), ("walk.mov", 300_000), ("lease.pdf", 4000), ("broken.png", 10))]
        media = self.Media()
        note, images, shown = await attached.for_the_model(ids + ["0123456789ab"], media, vision=True, sandbox=True)
        self.assertEqual([s["name"] for s in shown], ["goa.png", "walk.mov", "lease.pdf", "broken.png"])
        self.assertEqual(media.asked, [("goa.png", None), ("walk.mov", 1.0), ("broken.png", None)])
        self.assertEqual(images, [("image/jpeg", base64.b64encode(b"jpeg of goa.png").decode()),
                                  ("image/jpeg", base64.b64encode(b"jpeg of walk.mov").decode())])
        lines = note.splitlines()
        self.assertEqual(lines[0], "Files the owner attached to this message:")
        self.assertEqual(lines[1], f"- goa.png (image, 2.1 MB) at {attached.path_of(ids[0])}. You are shown it with this message.")
        # A video's line says its sound is somewhere else: shown one frame, a small model never listened.
        self.assertEqual(lines[2], f"- walk.mov (video, 0.3 MB, 8 s long) at {attached.path_of(ids[1])}. You are shown one "
                                   "frame of it (at 1 s) with this message; file_view shows other moments. "
                                   "Its sound is not in the frame: file_listen gives what is said in it.")
        self.assertEqual(lines[3], f"- lease.pdf (document, 4 KB) at {attached.path_of(ids[2])}. file_read reads it.")
        self.assertTrue(lines[4].endswith("broken.png. file_view shows it to you."))      # still there to try
        self.assertEqual(lines[5], "Code in the sandbox finds each one under /app/uploads/outputs/attached/<its id>/<its name>.")

    async def test_a_model_that_cannot_see_is_shown_nothing_and_no_sandbox_is_promised_without_one(self):
        kept = await attached.save(chunks(PNG), "goa.png", 1 << 20)
        media = self.Media()
        note, images, _ = await attached.for_the_model([kept["id"]], media, vision=False, sandbox=False)
        self.assertEqual((images, media.asked), ([], []))
        self.assertTrue(note.endswith("goa.png. This model cannot see pictures."))
        self.assertNotIn("sandbox", note)
        self.assertEqual(await attached.for_the_model(["0123456789ab"], media, True, True), ("", [], []))


class TaskTests(Home):
    async def test_a_task_carries_its_files_to_the_model_and_to_the_thread(self):
        svc = Svc()
        svc.vision = True
        svc.files = mock.Mock(media=ModelTests.Media())
        daemon = FakeDaemon(svc)
        got = {}

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            got.update(text=text, images=images)
            return "It is a beach."
        daemon.handle_message = handle
        bus = tasks_module.EventBus(svc.store)
        tasks = tasks_module.Tasks(daemon, bus)
        kept = await attached.save(chunks(PNG), "goa.png", 1 << 20)
        task = tasks.start("app:one", "Where was this taken?", "owner", attachments=[kept["id"], "0123456789ab"])
        await tasks.running[task["id"]]
        started = next(e for e in bus.since(0) if e["type"] == "started")
        self.assertEqual((started["prompt"], started["attachments"]), ("Where was this taken?", [kept]))
        self.assertEqual(tasks.get(task["id"])["prompt"], "Where was this taken?")     # the owner's own words
        self.assertTrue(got["text"].startswith("Where was this taken?\n\nFiles the owner attached to this message:\n- goa.png"))
        self.assertEqual(got["images"], [("image/jpeg", base64.b64encode(b"jpeg of goa.png").decode())])
        # A task without files is as it was: no note, no pictures.
        plain = tasks.start("app:one", "And the weather there?", "owner")
        await tasks.running[plain["id"]]
        self.assertEqual((got["text"], got["images"]), ("And the weather there?", None))
        self.assertNotIn("attachments", [e for e in bus.since(0) if e["type"] == "started"][-1])


class ProducedTests(Home):
    async def test_what_a_tasks_code_wrote_is_shown_under_its_answer(self):
        outputs = attached.outputs_dir()
        outputs.mkdir(parents=True)
        (outputs / "old.csv").write_text("a,b\n")
        svc = Svc()
        daemon = FakeDaemon(svc)

        async def handle(channel, chat_id, text, origin="", events=None, task=""):
            (outputs / "sales.png").write_bytes(PNG)                     # as code in the sandbox would
            (outputs / "report.html").write_text("<script>alert(1)</script>")
            (outputs / "link.png").symlink_to(self.home / "assistant" / "admin-token")
            (outputs / "attached").mkdir()
            return "Here is the chart."
        daemon.handle_message = handle
        bus = tasks_module.EventBus(svc.store)
        tasks = tasks_module.Tasks(daemon, bus)
        task = tasks.start("app:one", "Chart the sales", "owner")
        await tasks.running[task["id"]]
        kinds = [e["type"] for e in bus.since(0)]
        self.assertEqual(kinds[-2:], ["done", "files"])
        made = [e for e in bus.since(0) if e["type"] == "files"][0]["files"]
        self.assertEqual([(f["name"], f["kind"], f["inline"], f["url"]) for f in made],
                         [("report.html", "text", False, "/api/outputs/report.html"), ("sales.png", "image", True, "/api/outputs/sales.png")])
        svc.registry = build_registry(svc)
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            self.assertEqual((await client.get("/api/outputs/sales.png")).status_code, 401)
            shown = await client.get("/api/outputs/sales.png", headers=admin)
            self.assertEqual((shown.status_code, shown.content, shown.headers["content-type"]), (200, PNG, "image/png"))
            page = await client.get("/api/outputs/report.html", headers=admin)       # never run as the app's own page
            self.assertEqual((page.headers["content-type"], page.headers["content-disposition"][:10]),
                             ("application/octet-stream", "attachment"))
            for name in ("link.png", "attached", "nope.png", "..%2Fapi-token"):
                self.assertEqual((await client.get(f"/api/outputs/{name}", headers=admin)).status_code, 404, name)
        # A task that wrote nothing says nothing.
        daemon.handle_message = lambda *a, **k: asyncio.sleep(0, "Done.")
        plain = tasks.start("app:one", "And?", "owner")
        await tasks.running[plain["id"]]
        self.assertEqual([e["type"] for e in bus.since(0)][-1], "done")


class EndingTests(Home):
    async def test_a_task_that_breaks_ends_with_a_human_explanation_and_the_cause_goes_to_the_audit_trail(self):
        svc = Svc()
        daemon = FakeDaemon(svc)

        async def handle(channel, chat_id, text, origin="", events=None, task=""):
            raise KeyError("targetId")
        daemon.handle_message = handle
        bus = tasks_module.EventBus(svc.store)
        tasks = tasks_module.Tasks(daemon, bus)
        task = tasks.start("app:one", "Plan the trip", "owner")
        await tasks.running[task["id"]]
        ended = tasks.get(task["id"])
        self.assertEqual((ended["status"], ended["reply"]),
                         ("failed", "I couldn't finish this: something went wrong on my side partway through. "
                                    "Anything said above still stands. You can ask again."))
        self.assertNotIn("KeyError", json.dumps(bus.since(0)))
        self.assertIn("KeyError", (svc.tmp / "audit.jsonl").read_text())


class ApprovalThreadTests(Home):
    async def test_a_thread_says_how_each_approval_ended_also_when_it_was_answered_from_the_app(self):
        from coderunner.safety.policy import Action, current_run
        from types import SimpleNamespace
        svc = Svc()
        daemon = FakeDaemon(svc)

        async def handle(channel, chat_id, text, origin="", events=None, task=""):
            token = current_run.set(SimpleNamespace(events=events, content_tainted=False, private_sources=set(), shared=set()))
            try:
                return str(await svc.policy.check(Action("channel.send", "x", "Send the note"), origin=origin, wait=5))
            finally:
                current_run.reset(token)
        daemon.handle_message = handle
        bus = tasks_module.EventBus(svc.store)
        tasks = tasks_module.Tasks(daemon, bus)
        task = tasks.start("app:one", "Send it", "owner")
        await asyncio.sleep(0.05)
        [approval] = svc.policy.pending()
        svc.policy.decide(approval.id, True)             # as the app's Allow does: outside the task
        await tasks.running[task["id"]]
        # What is asked about reaches the thread whole (a message to send is read in full before it is allowed).
        long = "Send this to Asha:\n" + "Dear Asha, " + "the rent is due. " * 60
        again = tasks.start("app:two", "Send it", "owner")
        daemon.handle_message = lambda *a, events=None, **k: asyncio.sleep(0, "ok")
        await tasks.running[again["id"]]

        async def asks(channel, chat_id, text, origin="", events=None, task=""):
            token = current_run.set(SimpleNamespace(events=events, content_tainted=False, private_sources=set(), shared=set()))
            try:
                return str(await svc.policy.check(Action("channel.send", "x", long), origin=origin, wait=0.05))
            finally:
                current_run.reset(token)
        daemon.handle_message = asks
        third = tasks.start("app:three", "Send it", "owner")
        await tasks.running[third["id"]]
        asked = [e for e in bus.since(0, "app:three") if e.get("event") == "approval.requested"][0]
        self.assertEqual((asked["summary"], asked["risk"]), (long, "high"))
        seen = [(e["event"], e.get("id"), e.get("status")) for e in bus.since(0, "app:one") if e["type"] == "activity"]
        self.assertEqual(seen, [("approval.requested", approval.id, "pending"), ("approval.decided", approval.id, "approved")])


class RestartTests(Home):
    async def test_a_task_cut_short_by_a_restart_says_so_in_its_thread(self):
        svc = Svc()
        svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                          "VALUES ('abc123abc123', 'app:one', 'Plan the trip', 'running', '', 1.0, NULL)")
        bus = tasks_module.EventBus(svc.store)
        tasks = tasks_module.Tasks(FakeDaemon(svc), bus)
        ended = tasks.get("abc123abc123")
        self.assertEqual((ended["status"], ended["reply"]),
                         ("cancelled", "Stopped: the assistant was restarted while this was running."))
        self.assertEqual([(e["type"], e["conversation"], e["reply"]) for e in bus.since(0)],
                         [("cancelled", "app:one", "Stopped: the assistant was restarted while this was running.")])


class SandboxPathTests(Home):
    async def test_code_that_names_an_attached_file_by_its_host_path_is_given_the_sandboxs(self):
        from coderunner.agent.loop import RunContext
        from test_agent import FakeSandbox
        svc = Svc()
        svc.sandbox = FakeSandbox()
        registry = build_registry(svc)
        kept = await attached.save(chunks(b"month,sales\nJan,4\n"), "sales.csv", 1000)
        host = attached.path_of(kept["id"])
        await registry.invoke("run_python", {"code": f"print(open('{host}').read())"}, RunContext(conversation="t"))
        self.assertEqual(svc.sandbox.calls[-1][0], f"print(open('/app/uploads/outputs/attached/{kept['id']}/sales.csv').read())")


class RouteTests(Home):
    def setUp(self):
        super().setUp()
        self.svc = Svc({"models": {"chat": "coderunner-test"}})
        self.svc.registry = build_registry(self.svc)
        self.daemon = FakeDaemon(self.svc)
        self.daemon.tasks = mock.Mock()
        self.daemon.tasks.start.return_value = {"id": "t1", "conversation": "app:one"}
        self.admin = {"Authorization": "Bearer " + ADMIN}

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(self.svc, self.daemon)),
                                 base_url="http://127.0.0.1:8224")

    async def test_the_owner_attaches_a_file_and_the_app_shows_it(self):
        async with self.client() as client:
            self.assertEqual((await client.post("/api/uploads", content=PNG)).status_code, 401)
            sent = await client.post("/api/uploads", content=PNG, headers={**self.admin, "X-File-Name": "Goa%20%E2%9B%B1.png"})
            self.assertEqual(sent.status_code, 200)
            kept = sent.json()
            self.assertEqual((kept["name"], kept["kind"], kept["size"]), ("Goa ⛱.png", "image", len(PNG)))
            self.assertEqual((await client.get(kept["url"])).status_code, 401)
            shown = await client.get(kept["url"], headers=self.admin)
            self.assertEqual((shown.status_code, shown.content, shown.headers["content-type"]), (200, PNG, "image/png"))
            self.assertTrue(shown.headers["content-disposition"].startswith("inline"))
            part = await client.get(kept["url"], headers={**self.admin, "Range": "bytes=0-7"})      # a video is read in parts
            self.assertEqual((part.status_code, part.content), (206, PNG[:8]))
            self.assertEqual((await client.get("/api/uploads/0123456789ab", headers=self.admin)).status_code, 404)
            self.assertEqual((await client.post("/api/uploads", content=b"", headers=self.admin)).status_code, 400)
        self.assertTrue(any('"files.attached"' in line for line in (self.svc.tmp / "audit.jsonl").read_text().splitlines()))

    async def test_a_page_someone_attaches_is_never_run_as_the_apps_own(self):
        async with self.client() as client:
            for name in ("report.html", "drawing.svg"):
                kept = (await client.post("/api/uploads", content=b"<script>alert(1)</script>",
                                          headers={**self.admin, "X-File-Name": name})).json()
                self.assertFalse(kept["inline"])
                got = await client.get(kept["url"], headers=self.admin)
                self.assertEqual(got.headers["content-type"], "application/octet-stream")
                self.assertTrue(got.headers["content-disposition"].startswith("attachment"))
                self.assertEqual(got.headers["x-content-type-options"], "nosniff")
                self.assertIn("sandbox", got.headers["content-security-policy"])

    async def test_a_file_too_large_is_refused_and_a_task_names_its_files(self):
        self.svc.config.set("files.max_file_mb", 0.001)
        async with self.client() as client:
            big = await client.post("/api/uploads", content=b"x" * 5000, headers={**self.admin, "X-File-Name": "big.bin"})
            self.assertEqual((big.status_code, big.json()["error"]), (413, "That file is larger than 0 MB."))
            self.svc.config.set("files.max_file_mb", 1)
            kept = (await client.post("/api/uploads", content=PNG, headers={**self.admin, "X-File-Name": "a.png"})).json()
            body = {"message": "", "conversation": "app:one", "attachments": [kept["id"], "nope", "0123456789ab"]}
            self.assertEqual((await client.post("/api/tasks", json=body, headers=self.admin)).status_code, 200)
            self.daemon.tasks.start.assert_called_once_with("app:one", "", "admin-token", attachments=[kept["id"]])
            # Words or a file: a message with neither is not a task.
            empty = await client.post("/api/tasks", json={"message": " ", "attachments": ["0123456789ab"]}, headers=self.admin)
            self.assertEqual(empty.status_code, 400)


if __name__ == "__main__":
    unittest.main()
