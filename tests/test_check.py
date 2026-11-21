"""`coderunner check`: the core tasks, run through the daemon's own task runner and checked."""

import asyncio
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.agent import build_registry
from coderunner.files import attached
from coderunner.models import sites
from coderunner.models import tasks as sample
from coderunner.safety.policy import Action
from coderunner.server import check as check_module
from coderunner.server import tasks as tasks_module
from coderunner.server.gateway import admin_app
from fakes import Svc
from test_gateway import ADMIN, FakeDaemon


class Files:
    """File search as the check uses it: a folder shared for the run, found once it is indexed."""

    def __init__(self, searchable=True):
        self.embedder, self.media = object(), mock.Mock(ffmpeg="/usr/bin/ffmpeg")
        self.wake, self.roots, self.removed, self.searchable = asyncio.Event(), [], [], searchable

    def add_root(self, folder):
        self.roots.append(folder)
        return folder

    def remove_root(self, folder):
        self.removed.append(folder)
        return 0

    async def search(self, query, kind, limit):
        return [{"path": str(pathlib.Path(self.roots[-1]).resolve() / "Warranty.txt")}] if self.searchable and self.roots else []


class CheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)
        (pathlib.Path(self._tmp.name) / "assistant").mkdir()
        self.svc = Svc({"models": {"chat": "coderunner-test"}, "browser": {"allow": ["intranet.example"]}})
        self.svc.vision, self.svc.browser, self.svc.files = True, object(), Files()
        self.svc.voice = mock.Mock(available=lambda: True)
        self.svc.sandbox = mock.Mock(health=mock.AsyncMock(return_value=True))
        self.daemon = FakeDaemon(self.svc)
        self.daemon.driver = mock.Mock(gateway_address=lambda: "127.0.0.1")
        self.daemon.tasks = tasks_module.Tasks(self.daemon, tasks_module.EventBus(self.svc.store))
        self.seen = []                  # (prompt, attachment names, browser.allow at the time)
        self.wrong: set[str] = set()    # checks answered wrongly
        self.daemon.handle_message = self.handle
        self.job = check_module.CheckJob(self.daemon)

    def answer(self, name: str, site: sites.Sites) -> str:
        day = sample.target_day(site.today).isoformat()
        cheap, evening = sites.cheapest("BLR", "DEL", day), sites.evening("BLR", "DEL", day)
        fares = "; ".join(f"{f['airline']} {f['price']}" for f in cheap)
        papers = "Covered until 14 March 2030. Service: 1800 266 1880."
        library = "It closes at 14:00 on Sundays and a DVD is ₹15 a day late."
        late = f"{evening['airline']} {evening['number']} at ₹{evening['price']}"
        return {"answer": "391", "picture": "The total is Rs 1,280.", "document": "It ends on 28 February 2027; two months' notice.",
                "video": "He comes on Thursday at 4:30; the visit costs 600 rupees.", "code": "May was highest; 299,600 in all.",
                "files": papers, "website": library, "flights": fares, "booking": f"{cheap[0]['number']} at ₹{cheap[0]['price']}",
                "appliance": "Descale it every 3 months.", "errands": f"{late}. {library} {papers}"}[name]

    async def handle(self, channel, chat_id, text, origin="", events=None, task="", images=None):
        name = chat_id.split("-")[1]
        self.seen.append((name, origin, [line.split(" (")[0][2:] for line in text.splitlines() if line.startswith("- ")],
                          list(self.svc.config.get("browser.allow"))))
        site = self.site
        if name in ("code", "video"):
            events({"type": "tool", "name": {"code": "run_python", "video": "file_listen"}[name], "arguments": {}})
        if name == "booking":
            best = sites.cheapest("BLR", "DEL", sample.target_day(site.today).isoformat())[0]
            site.visits.append(f"/flights/checkout?flight={best['number'].replace(' ', '')}")
        if name == "appliance":
            site.visits.append("/kelvin/guide/dw-412")
        return "I could not tell." if name in self.wrong else self.answer(name, site)

    async def run_check(self, names=None, long=False):
        made = []
        real = sites.Sites

        def make(*args, **kwargs):
            made.append(real(*args, **kwargs))
            self.site = made[-1]
            return made[-1]
        with mock.patch.object(check_module.fixtures, "Sites", side_effect=make):
            self.assertEqual(self.job.start(names, long=long), "")
            await self.job._job
        return self.job.public()

    async def test_the_quick_checks_run_as_tasks_of_the_app_and_each_answer_is_checked(self):
        shown = await self.run_check()
        self.assertEqual([(t["name"], t["state"]) for t in shown["tasks"]],
                         [(n, "passed") for n in ("answer", "picture", "document", "video", "code", "files", "website")])
        self.assertEqual((shown["state"], shown["result"]["passed"], shown["result"]["ran"], shown["result"]["ok"]), ("done", 7, 7, True))
        # Each went through the task runner, as a message from the app would, with its test file attached.
        self.assertEqual([(name, origin, files) for name, origin, files, _ in self.seen], [
            ("answer", "check", []), ("picture", "check", ["receipt.png"]), ("document", "check", ["lease.pdf"]),
            ("video", "check", ["plumber.mp4"]), ("code", "check", ["sales.csv"]), ("files", "check", []), ("website", "check", [])])
        listed = self.daemon.tasks.conversations()
        self.assertEqual(len(listed), 7)                                    # and stays in the app's list, to be opened
        self.assertTrue(all(c["conversation"].startswith("app:check-") for c in listed))
        self.assertEqual(shown["tasks"][4]["steps"], 1)
        # For the run the test sites could be browsed and the test documents were shared; afterwards neither is.
        self.assertEqual(self.seen[-1][3], ["intranet.example", "127.0.0.1"])
        self.assertEqual(self.svc.config.get("browser.allow"), ["intranet.example"])
        self.assertEqual(self.svc.files.removed, self.svc.files.roots)
        self.assertFalse(pathlib.Path(self.svc.files.roots[0]).exists())
        self.assertEqual(self.svc.policy._listeners, [])
        self.assertEqual(check_module.last()["passed"], 7)
        self.assertEqual(shown["lines"][-1][:14], "7 of 7 passed,")

    async def test_a_wrong_answer_fails_with_what_was_missing_and_a_tool_not_used_fails_too(self):
        self.wrong = {"document", "website"}
        original = self.handle

        async def no_python(channel, chat_id, text, origin="", events=None, task="", images=None):
            if "-code-" in chat_id:
                return "May was highest; 299,600 in all."                   # said without running anything
            return await original(channel, chat_id, text, origin, events, task, images)
        self.daemon.handle_message = no_python
        shown = await self.run_check()
        states = {t["name"]: (t["state"], t["why"]) for t in shown["tasks"]}
        self.assertEqual(states["document"], ("failed", "missing the end date and the notice period"))
        self.assertEqual(states["website"], ("failed", "missing Sunday closing time 14:00, DVD fee ₹15"))
        self.assertEqual(states["code"], ("failed", "answered without using run_python"))
        self.assertEqual((shown["state"], shown["result"]["passed"], shown["result"]["ran"]), ("failed", 4, 7))

    async def test_what_this_install_cannot_do_is_skipped_with_the_reason(self):
        self.svc.vision, self.svc.browser = False, None
        self.svc.voice = mock.Mock(available=lambda: False)
        self.svc.sandbox.health = mock.AsyncMock(side_effect=httpx.ConnectError("down"))
        self.svc.files.searchable = False
        with mock.patch.object(check_module, "INDEX_S", 0):
            shown = await self.run_check()
        self.assertEqual({t["name"]: (t["state"], t["why"]) for t in shown["tasks"]}, {
            "answer": ("passed", "17 × 23 = 391"), "document": ("passed", "the lease's end (28 February 2027) and its notice (two months)"),
            "picture": ("skipped", "the model here reads text only"), "video": ("skipped", "speech to text is not set up"),
            "code": ("skipped", "the sandbox is not running"), "website": ("skipped", "no browser is set up"),
            "files": ("skipped", "the test documents were not searchable after 0 minutes (the index is busy)")})
        self.assertEqual((shown["result"]["passed"], shown["result"]["ran"], shown["result"]["skipped"], shown["result"]["ok"]),
                         (2, 2, 5, True))

    async def test_the_long_ones_are_asked_for_and_a_booking_must_stop_before_payment(self):
        shown = await self.run_check(long=True)
        self.assertEqual([t["name"] for t in shown["tasks"]][7:], ["flights", "booking", "appliance", "errands"])
        self.assertTrue(all(t["state"] == "passed" for t in shown["tasks"]), shown["tasks"])
        self.assertEqual(shown["tasks"][-1]["why"], "3 of 3 errands")
        original = self.handle

        async def pays(channel, chat_id, text, origin="", events=None, task="", images=None):
            reply = await original(channel, chat_id, text, origin, events, task, images)
            self.site.visits.append("/flights/payment")
            return reply
        self.daemon.handle_message = pays
        shown = await self.run_check(["booking", "errands"])
        self.assertEqual([(t["name"], t["state"], t["why"]) for t in shown["tasks"]],
                         [("booking", "failed", "went on to the payment page"),
                          ("errands", "failed", "2 of 3 errands: missing the evening flight")])

    async def test_approvals_of_its_own_tasks_are_answered_and_a_step_that_pays_is_refused(self):
        asked = {}

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            policy = self.svc.policy
            asked["sandbox"] = await policy.check(Action("channel.send", "x", "Run in the sandbox:\nprint(1)"), origin=origin, wait=2)
            asked["pay"] = await policy.check(Action("channel.send", "x", "Click Pay now on skyfare"), origin=origin, wait=2)
            asked["other"] = await policy.check(Action("channel.send", "x", "Send a message"), origin="telegram:1", wait=0.2)
            return "391"
        self.daemon.handle_message = handle
        await self.run_check(["answer"])
        self.assertEqual(asked, {"sandbox": True, "pay": False, "other": False})    # another task's is left for the owner

    async def test_a_check_is_not_started_twice_or_without_a_model_or_for_a_name_it_does_not_have(self):
        self.assertEqual(self.job.start(["nope"]), "No such check: nope. There are: answer, picture, document, video, code, "
                                                   "files, website, flights, booking, appliance, errands.")
        self.daemon.tasks.running["x"] = object()
        self.assertEqual(self.job.start(), "A task is running; the check waits for the model to be free.")
        self.daemon.tasks.running.clear()
        self.svc.config.set("models.chat", "")
        self.assertEqual(self.job.start(), "No model is set up yet.")
        self.svc.config.set("models.chat", "coderunner-test")
        self.assertEqual(self.job.start(["answer"]), "")
        self.assertEqual(self.job.start(["answer"]), "A check is already running.")
        await self.job._job

    async def test_the_owner_starts_and_follows_a_check_over_the_api(self):
        self.svc.registry = build_registry(self.svc)
        self.daemon.check = self.job
        admin = {"Authorization": "Bearer " + ADMIN}
        real = sites.Sites

        def make(*args, **kwargs):
            self.site = real(*args, **kwargs)
            return self.site
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(self.svc, self.daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            self.assertEqual((await client.get("/api/check")).status_code, 401)
            first = (await client.get("/api/check", headers=admin)).json()
            self.assertEqual((first["state"], first["result"], len(first["names"])), ("idle", None, 11))
            bad = await client.post("/api/check", json={"names": ["nope"]}, headers=admin)
            self.assertEqual((bad.status_code, bad.json()["error"][:19]), (409, "No such check: nope"))
            with mock.patch.object(check_module.fixtures, "Sites", side_effect=make):
                started = await client.post("/api/check", json={"names": ["answer", "document"]}, headers=admin)
                self.assertEqual((started.status_code, started.json()["state"]), (200, "running"))
                await self.job._job
            done = (await client.get("/api/check", headers=admin)).json()
            self.assertEqual((done["state"], [t["state"] for t in done["tasks"]]), ("done", ["passed", "passed"]))


class FixtureTests(unittest.TestCase):
    def test_the_test_files_ship_with_the_package(self):
        for name in {name for check in check_module.CHECKS for name in check.files}:
            self.assertTrue((check_module.FIXTURES / name).is_file(), name)
        self.assertEqual(attached.safe_name("plumber.mp4"), "plumber.mp4")


if __name__ == "__main__":
    unittest.main()
