"""Setup from the app: the job that runs `coderunner setup`'s plan and setup in the daemon, and its route."""

import sys
import threading
import time
import unittest
from unittest import mock

import httpx

from coderunner.agent import build_registry
from coderunner.models import manager
from coderunner.server.gateway import admin_app
from coderunner.server.setup import SetupJob
from fakes import Svc
from test_gateway import ADMIN, FakeDaemon

PLAN = {"hardware": {"chip": "Apple M3 Pro", "ram_gb": 18.0, "gpu": "apple", "measured": ""},
        "model": {"id": "qwen3.5-9b", "runtime": "llama.cpp", "ctx": 32768, "step_s": 12.0, "vision": True,
                  "reasons": ["fits", "12 s a step", "vision", "apache-2.0", "benchmark ability +0.21"], "need_gb": 8.2},
        "alternatives": [{"id": "qwen3.5-4b", "runtime": "llama.cpp", "step_s": 7.1, "reasons": []}],
        "notes": ["No model runtime is installed here: setup fetches its own copy of Ollama 0.40.3."]}
DOWNLOADS = [{"what": "Ollama 0.40.3", "gb": 0.17}, {"what": "assistant model: qwen3.5-9b", "gb": 6.6},
             {"what": "assistant model: hf-lab-agent", "gb": None}]


def until(done, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return done()


class SetupJobTests(unittest.TestCase):
    def setUp(self):
        self.svc = Svc()
        for name, value in (("downloads", DOWNLOADS), ("_room_gb", (6.4, 11.4, 5.0))):
            patch = mock.patch.object(manager, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_briefing_is_what_the_app_shows_before_anything_is_fetched(self):
        plan = {**PLAN, "hardware": {**PLAN["hardware"], "budget_gb": 11.72, "bandwidth_gbs": 150, "unified": True},
                "model": {**PLAN["model"], "tok_s": 14.5, "read_tok_s": 225.0}, "decision": None}
        job = SetupJob(self.svc.config, plan=lambda config: plan, voice=False)
        with mock.patch("coderunner.models.coreml.supported", return_value=False):
            shown = job.plan()
        self.assertEqual(shown["hardware"], {"chip": "Apple M3 Pro", "ram_gb": 18, "gpu": "apple", "budget_gb": 11.7,
                                             "unified": True, "vram_gb": 0, "measured": "",
                                             "os": "mac" if sys.platform == "darwin" else "linux"})
        self.assertEqual(shown["speed"], {"measured": False, "read_tok_s": 225.0, "tok_s": 14.5, "step_s": 12.0,
                                          "bandwidth_gbs": 150, "when": "", "tasks": None})
        self.assertEqual((shown["model"]["id"], shown["model"]["name"], len(shown["model"]["reasons"])),
                         ("qwen3.5-9b", "Qwen3.5 9B", 4))
        # A model for each job: the planner, what reads pictures (the same model), and what checks each step.
        self.assertEqual([(r["key"], r["short"], r["model"], bool(r.get("shared"))) for r in shown["roles"]],
                         [("planner", "Planning", "Qwen3.5 9B", False), ("eyes", "Vision", "The planning model", True),
                          ("checker", "Safety check", "The planning model", True)])
        self.assertEqual((shown["roles"][0]["on"], shown["roles"][0]["download_gb"]), ("llama.cpp, on the GPU", 6.6))
        self.assertEqual([a["name"] for a in shown["alternatives"]], ["Qwen3.5 4B"])
        self.assertEqual((shown["download_gb"], shown["free_gb"], shown["room_gb"], shown["after_gb"]), (6.77, 11.4, 6.4, 4.6))
        # The list of downloads says what each is for; the names are in the list of jobs.
        self.assertEqual([d["label"] for d in shown["downloads"]],
                         ["Model engine (Ollama)", "Planning model", "Planning model"])
        self.assertTrue(shown["models_dir"].endswith("assistant/models") and shown["models_dir_default"])
        self.assertTrue(shown["models_dir_short"].endswith("assistant/models") and shown["disk_gb"] > 0)
        self.assertEqual(job.public()["plan"], shown)
        self.assertEqual((job.public()["configured"], job.public()["state"]), (False, "idle"))

    def test_the_owner_can_choose_another_model_that_fits_and_the_plan_is_made_for_it(self):
        sizes = {"qwen3.5-9b": 6.6, "qwen3.5-4b": 0}                    # the smaller one is already here
        downloads = lambda config, planned, hw=None: (
            [{"what": f"assistant model: {planned['model']['id']}", "gb": sizes[planned["model"]["id"]]}]
            if sizes[planned["model"]["id"]] else [])
        job = SetupJob(self.svc.config, plan=lambda config: PLAN, voice=False)
        with mock.patch.object(manager, "downloads", side_effect=downloads), \
                mock.patch("coderunner.models.coreml.supported", return_value=False):
            first, other, unknown = job.plan(), job.plan("qwen3.5-4b"), job.plan("not-a-model")
        self.assertEqual((first["model"]["id"], first["recommended"], first["download_gb"]), ("qwen3.5-9b", "qwen3.5-9b", 6.6))
        self.assertEqual([(c["id"], c["name"], c["download_gb"], c["step_s"]) for c in first["choices"]],
                         [("qwen3.5-9b", "Qwen3.5 9B", 6.6, 12.0), ("qwen3.5-4b", "Qwen3.5 4B", 0, 7.1)])
        # The chosen one takes the planner's line, and the downloads and the speed are its own.
        self.assertEqual((other["model"]["id"], other["recommended"], other["download_gb"], other["speed"]["step_s"]),
                         ("qwen3.5-4b", "qwen3.5-9b", 0, 7.1))
        self.assertEqual((other["roles"][0]["model"], other["roles"][0]["download_gb"]), ("Qwen3.5 4B", 0))
        self.assertEqual([a["id"] for a in other["alternatives"]], ["qwen3.5-9b"])
        self.assertEqual(other["choices"], first["choices"])
        self.assertEqual(unknown["model"]["id"], "qwen3.5-9b")          # a name the plan does not have changes nothing
        # Setup is then asked for that model, and told not to load it into memory that is not free.
        asked = {}
        job = SetupJob(self.svc.config, plan=lambda config: PLAN, voice=False,
                       setup=lambda config, **kw: (asked.update(kw), {"ok": True})[1])
        job.start(model="qwen3.5-4b")
        self.assertTrue(until(lambda: job.public()["state"] == "done"))
        self.assertEqual((asked["model_id"], asked["insist"], asked["other_downloads"]), ("qwen3.5-4b", False, False))

    def test_the_first_screen_says_when_memory_is_short_right_now_and_setup_fetches_no_other_model_unasked(self):
        from coderunner.models import hardware
        seen = {}

        def setup(config, **kw):
            seen.update(kw)
            return {"ok": False, "error": "Bonsai 2 27B could not be set up: the server exited",
                    "next": {"model": "qwen3.5-9b", "download_gb": 6.6}}
        job = SetupJob(self.svc.config, plan=lambda config: PLAN, setup=setup, voice=False)
        with mock.patch("coderunner.models.coreml.supported", return_value=False):
            with mock.patch.object(hardware, "available_gb", return_value=6.1):
                self.assertEqual(job.plan()["memory"], {"need_gb": 8.2, "free_gb": 6.1, "short": True})
            with mock.patch.object(hardware, "available_gb", return_value=12.0):
                self.assertFalse(job.plan()["memory"]["short"])
            with mock.patch.object(hardware, "available_gb", return_value=None):
                self.assertIsNone(job.plan()["memory"])
        self.assertTrue(job.start())
        self.assertTrue(until(lambda: job.public()["state"] == "failed"))
        self.assertIs(seen["other_downloads"], False)
        self.assertEqual(job.public()["result"]["next"], {"model": "qwen3.5-9b", "download_gb": 6.6, "name": "Qwen3.5 9B"})

    def test_a_model_that_reads_text_only_is_said_so_and_a_long_folder_is_shown_by_its_end(self):
        import pathlib
        from coderunner.server.setup import _short_path
        plan = {**PLAN, "model": {**PLAN["model"], "vision": False}}
        job = SetupJob(self.svc.config, plan=lambda config: plan, voice=False)
        with mock.patch("coderunner.models.coreml.supported", return_value=False):
            eyes = job.plan()["roles"][1]
        self.assertEqual((eyes["key"], eyes["model"], eyes["shared"], eyes["missing"]),
                         ("eyes", "Not on this computer", False, True))
        home = pathlib.Path.home()
        self.assertEqual(_short_path(home / ".coderunner" / "assistant" / "models"), "~/.coderunner/assistant/models")
        self.assertEqual(_short_path(pathlib.Path("/Volumes/External Disk/Projects/Local AI/CodeRunner/models")),
                         "…/CodeRunner/models")
        self.assertEqual(_short_path(pathlib.Path("/Volumes/Disk/CodeRunner")), "/Volumes/Disk/CodeRunner")

    def test_the_owner_chooses_where_the_models_are_kept(self):
        import pathlib
        import tempfile
        job = SetupJob(self.svc.config, plan=lambda config: PLAN, voice=False)
        with tempfile.TemporaryDirectory() as disk, mock.patch("coderunner.models.coreml.supported", return_value=False):
            chosen = pathlib.Path(disk) / "External" / "CodeRunner"
            shown = job.set_location(str(chosen))
            self.assertTrue(chosen.is_dir())                                   # made for the owner
            self.assertEqual((shown["models_dir"], shown["models_dir_default"]), (str(chosen), False))
            self.assertEqual(self.svc.config.get("models.dir"), str(chosen))
            from coderunner.models.runtimes import LlamaCppRuntime, models_dir
            self.assertEqual((models_dir(self.svc.config), LlamaCppRuntime(self.svc.config).root), (chosen, chosen))
            for bad, why in (("models", "full path"), ("/dev/null/x", "cannot be made")):
                with self.assertRaisesRegex(ValueError, why):
                    job.set_location(bad)
            self.assertEqual(self.svc.config.get("models.dir"), str(chosen))   # a refused folder changes nothing
            self.assertTrue(job.set_location("")["models_dir_default"])        # back to the usual folder
        job.state = "running"
        with self.assertRaisesRegex(ValueError, "Setup is running"):
            job.set_location("/tmp/elsewhere")

    def test_setup_runs_in_the_background_and_its_progress_is_kept(self):
        release, finished = threading.Event(), []

        def setup(config, model_id=None, accept_license=False, progress=print, **more):
            progress(f"qwen3.5-9b via llama.cpp (asked for {model_id}, licence accepted: {accept_license})")
            release.wait(5)
            progress("smoke testing coderunner-qwen3.5-9b")
            config.set("models.chat", "coderunner-qwen3.5-9b")
            return {"ok": True, "entry": "qwen3.5-9b", "model": "coderunner-qwen3.5-9b", "runtime": "llama.cpp",
                    "smoke": {"checks": []}, "attempts": [{"model": "qwen3.5-9b"}]}

        job = SetupJob(self.svc.config, self.svc.audit, on_done=finished.append, setup=setup, voice=False)
        self.assertTrue(job.start(model="qwen3.5-9b", accept_license=True))
        self.assertTrue(until(lambda: job.public()["line_count"] == 1))
        self.assertEqual(job.public()["state"], "running")
        self.assertFalse(job.start())                                   # one setup at a time
        release.set()
        self.assertTrue(until(lambda: job.public()["state"] == "done"))
        seen = job.public()
        self.assertIn("asked for qwen3.5-9b, licence accepted: True", seen["lines"][0])
        self.assertEqual(seen["result"], {"ok": True, "entry": "qwen3.5-9b", "model": "coderunner-qwen3.5-9b",
                                          "runtime": "llama.cpp"})        # no attempts or smoke details
        self.assertTrue(seen["configured"])
        self.assertTrue(until(lambda: len(finished) == 1))              # the daemon starts again once

    def test_a_failed_setup_says_why_and_does_not_restart_the_daemon(self):
        finished = []
        job = SetupJob(self.svc.config, on_done=finished.append, voice=False,
                       setup=lambda config, **kw: {"ok": False, "error": "Not enough free disk"})
        job.start()
        self.assertTrue(until(lambda: job.public()["state"] == "failed"))
        self.assertEqual(job.public()["result"], {"ok": False, "error": "Not enough free disk"})

        def broken(config, **kw):
            raise ValueError("unforeseen")
        job = SetupJob(self.svc.config, on_done=finished.append, setup=broken, voice=False)
        job.start()
        self.assertTrue(until(lambda: job.public()["state"] == "failed"))
        # The app shows a sentence; the exception's own words are under Details.
        self.assertEqual(job.public()["result"]["error"], "Setup stopped on an error it did not expect. Details has the last lines.")
        self.assertIn("Error: ValueError: unforeseen", job.public()["lines"])
        self.assertEqual(finished, [])


class StageTests(unittest.TestCase):
    """The app shows setup as a few stages that say what is being fetched or done; setup's own lines stay behind
    them."""

    LINES = ["downloading llama-b11564-bin-macos-arm64.tar.gz",
             "bonsai-2-27b via llama.cpp: fits: 9.7 of 11.7 GB at 32K context", "bonsai-2-27b: 8.3 GB to download, 18.1 GB free",
             "downloading bonsai-2-27b-Q2_0.gguf: 61% of 7.63 GB",
             "downloading Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf: 30% of 0.63 GB", "loading bonsai-2-27b into memory",
             "smoke testing coderunner-bonsai-2-27b with vision", "  pass tool_arguments: add_numbers(a=3, b=7) (5.95s)",
             "timing coderunner-bonsai-2-27b on this machine",
             "measured 80 tok/s reading, 12.0 tok/s writing: 18 s per agent step (estimated 19 s)",
             "installing Core ML packages (coremltools, tokenizers)", "downloading tokenizer.json: 78% of 0.03 GB",
             "preparing the embedding model for the Neural Engine (once, about a minute)",
             "embeddings: no GPU build to compare with", "setting up speech to text",
             "downloading Decoder.mlmodelc/model.mil: 40% of 0.20 GB", "building the Parakeet helper with SwiftPM (a minute or two)"]

    def job(self, upto):
        job = SetupJob(Svc().config, plan=lambda config: PLAN, setup=lambda *a, **k: {"ok": True}, voice=False)
        job.state = "running"
        for line in self.LINES[:upto]:
            job._say(line)
        return job

    def latest(self, upto):
        stage = self.job(upto).public()["stages"][-1]
        return stage["key"], stage["label"], stage["percent"], stage["of_gb"]

    def test_each_stage_says_what_is_being_fetched_or_done_and_names_no_file(self):
        self.assertEqual(self.latest(1), ("engine", "Downloading the model engine", None, None))
        self.assertEqual(self.latest(4), ("planner", "Downloading the planning model", 61, 7.63))
        self.assertEqual(self.latest(5), ("planner", "Downloading the vision part", 30, 0.63))
        self.assertEqual(self.latest(6)[:2], ("check", "Loading the planning model"))
        self.assertEqual(self.latest(8)[:2], ("check", "Testing tools and vision"))
        self.assertEqual(self.latest(9)[:2], ("speed", "Measuring speed"))
        # A download belongs to what is being set up when it happens: the search model's, then the voice model's.
        self.assertEqual(self.latest(12), ("search", "Downloading the file search model", 78, 0.03))
        self.assertEqual(self.latest(13)[:2], ("search", "Preparing file search"))
        self.assertEqual(self.latest(16), ("voice", "Downloading the voice model", 40, 0.2))
        self.assertEqual(self.latest(17)[:2], ("voice", "Preparing voice"))
        for upto in range(1, len(self.LINES) + 1):
            for stage in self.job(upto).public()["stages"]:
                self.assertNotRegex(stage["label"], r"(?i)gguf|bonsai|mmproj|\.mil|getting the model")

    def test_a_model_that_failed_the_picture_check_is_not_said_to_read_pictures(self):
        job = self.job(8)
        job._say("  vision check failed: the model could not read the test picture")
        job._say("timing coderunner-bonsai-2-27b on this machine")
        self.assertEqual(job.public()["stages"][2]["label"], "Tools work (text only on this computer)")

    def test_a_finished_stage_says_what_came_of_it(self):
        shown = self.job(len(self.LINES)).public()["stages"]
        self.assertEqual([s["key"] for s in shown], ["engine", "planner", "check", "speed", "search", "voice"])
        self.assertEqual([s["state"] for s in shown], ["done"] * 5 + ["running"])
        self.assertEqual([s["label"] for s in shown[:5]], [
            "Model engine ready", "Planning model downloaded", "Tools and vision work",
            "Speed measured: 18 s per step", "File search ready"])
        # A planning model that was already on the disk is not said to have been downloaded.
        job = SetupJob(Svc().config, plan=lambda config: PLAN, setup=lambda *a, **k: {"ok": True}, voice=False)
        job.state = "running"
        for line in ("qwen3.5-9b via llama.cpp: fits: 8.2 of 12.7 GB at 32K context", "loading qwen3.5-9b into memory"):
            job._say(line)
        self.assertEqual(job.public()["stages"][0]["label"], "Planning model ready")

    def test_the_last_stage_shows_how_setup_ended_and_a_new_run_starts_clean(self):
        job = self.job(8)
        job._finish({"ok": False, "error": "the smoke test failed"})
        self.assertEqual([(s["key"], s["state"]) for s in job.public()["stages"]],
                         [("engine", "done"), ("planner", "done"), ("check", "failed")])
        self.assertEqual(job.public()["stages"][-1]["label"], "Testing tools and vision")
        job = self.job(len(self.LINES))
        job._finish({"ok": True})
        self.assertEqual({s["state"] for s in job.public()["stages"]}, {"done"})
        self.assertEqual(job.public()["stages"][-1]["label"], "Voice ready")
        self.assertTrue(job.start())
        self.assertTrue(until(lambda: job.public()["state"] == "done"))
        self.assertEqual(job.public()["stages"], [])


class PictureTests(unittest.IsolatedAsyncioTestCase):
    """A task's thread carries pictures of the page where they say more than words."""

    async def asyncSetUp(self):
        import os
        import tempfile

        from coderunner.server import tasks as tasks_module
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)
        for name, value in (("SHOT_SETTLE_S", 0.01),):
            patch = mock.patch.object(tasks_module, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.tasks_module = tasks_module
        self.svc = Svc()
        self.page = "https://news.example/front"

        class Browser:
            async def peek(inner):
                return (b"\xff\xd8jpeg", self.page) if self.page else None

        self.svc.browser = Browser()
        self.daemon = FakeDaemon(self.svc)
        self.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = tasks_module.Tasks(self.daemon, self.bus)

    def events(self, kind):
        return [e for e in self.bus.since(0) if e["type"] == kind]

    async def run_task(self, steps):
        from coderunner.safety.policy import current_run

        async def handle(channel, chat_id, text, origin="", events=None, task=""):
            from types import SimpleNamespace
            token = current_run.set(SimpleNamespace(events=events))
            try:
                await steps()
            finally:
                current_run.reset(token)
            return "the answer"
        self.daemon.handle_message = handle
        task = self.tasks.start("app:one", "find the top story", "owner")
        await self.tasks.running[task["id"]]
        return task["id"]

    async def test_a_page_just_opened_is_pictured_once_and_the_last_page_at_the_end(self):
        import asyncio

        async def steps():
            self.svc.audit.write("browser.open", url=self.page, ok=True)
            await asyncio.sleep(0.05)
            self.svc.audit.write("browser.navigate", url=self.page, ok=True)      # the same page again: no second picture
            await asyncio.sleep(0.05)
            self.svc.audit.write("browser.click", label="Next", ok=True)
            self.page = "https://news.example/page-2"
        task_id = await self.run_task(steps)
        shots = self.events("shot")
        self.assertEqual([(s["file"], s["page"]) for s in shots],
                         [(f"{task_id}-1.jpg", "https://news.example/front"), (f"{task_id}-2.jpg", "https://news.example/page-2")])
        self.assertEqual((self.tasks_module.shots_dir() / shots[0]["file"]).read_bytes(), b"\xff\xd8jpeg")
        kinds = [e["type"] for e in self.bus.since(0)]
        self.assertLess(kinds.index("shot"), kinds.index("done"))               # the last picture comes before the answer

    async def test_a_task_that_never_opened_a_page_has_no_pictures_and_a_failed_capture_costs_nothing(self):
        async def steps():
            self.svc.audit.write("files.search", query="lease")
        await self.run_task(steps)
        self.assertEqual(self.events("shot"), [])

        async def opened():
            import asyncio
            self.page = ""                              # no web page to show
            self.svc.audit.write("browser.open", url="https://news.example/front", ok=True)
            await asyncio.sleep(0.05)
        await self.run_task(opened)
        self.assertEqual(self.events("shot"), [])
        self.assertEqual(self.events("done")[-1]["reply"], "the answer")

    async def test_the_owner_fetches_a_picture_by_its_name_and_nothing_else(self):
        self.svc.registry = build_registry(self.svc)
        folder = self.tasks_module.shots_dir()
        folder.mkdir(parents=True)
        (folder / "0123456789ab-1.jpg").write_bytes(b"\xff\xd8jpeg")
        (folder.parent / "admin-token").write_text("secret")
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(self.svc, self.daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            self.assertEqual((await client.get("/api/shots/0123456789ab-1.jpg")).status_code, 401)
            found = await client.get("/api/shots/0123456789ab-1.jpg", headers=admin)
            self.assertEqual((found.status_code, found.content, found.headers["content-type"]), (200, b"\xff\xd8jpeg", "image/jpeg"))
            for name in ("0123456789ab-2.jpg", "..%2Fadmin-token", "admin-token", "0123456789ab-1.png"):
                self.assertEqual((await client.get(f"/api/shots/{name}", headers=admin)).status_code, 404, name)


class SetupRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc()
        self.svc.registry = build_registry(self.svc)
        self.daemon = FakeDaemon(self.svc)
        for name, value in (("downloads", DOWNLOADS), ("_room_gb", (6.4, 11.4, 5.0))):
            patch = mock.patch.object(manager, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(self.svc, self.daemon)),
                                 base_url="http://127.0.0.1:8224")

    async def test_the_owner_plans_and_starts_setup_over_the_api(self):
        admin = {"Authorization": "Bearer " + ADMIN}
        release = threading.Event()

        def setup(config, model_id=None, accept_license=False, progress=print, **more):
            progress("downloading")
            release.wait(5)
            return {"ok": False, "error": "stopped by the test"}

        async with self.client() as client:
            self.assertEqual((await client.get("/api/setup", headers=admin)).status_code, 503)   # no job here
            self.daemon.setup = SetupJob(self.svc.config, plan=lambda config: PLAN, setup=setup, voice=False)
            unsupported = mock.patch("coderunner.models.coreml.supported", return_value=False)
            unsupported.start()
            self.addCleanup(unsupported.stop)
            self.assertEqual((await client.get("/api/setup")).status_code, 401)
            self.assertEqual((await client.post("/api/setup", json={"action": "start"})).status_code, 401)
            state = (await client.get("/api/setup", headers=admin)).json()
            self.assertEqual((state["configured"], state["state"], state["plan"]), (False, "idle", None))
            planned = await client.post("/api/setup", json={"action": "plan"}, headers=admin)
            self.assertEqual((planned.status_code, planned.json()["model"]["id"]), (200, "qwen3.5-9b"))
            self.assertEqual((await client.post("/api/setup", json={"action": "x"}, headers=admin)).status_code, 400)
            refused = await client.post("/api/setup", json={"action": "location", "path": "models"}, headers=admin)
            self.assertEqual((refused.status_code, "full path" in refused.json()["error"]), (400, True))
            usual = await client.post("/api/setup", json={"action": "location", "path": ""}, headers=admin)
            self.assertEqual((usual.status_code, usual.json()["models_dir_default"]), (200, True))
            started = await client.post("/api/setup", json={"action": "start"}, headers=admin)
            self.assertEqual((started.status_code, started.json()["started"]), (200, True))
            again = await client.post("/api/setup", json={"action": "start"}, headers=admin)
            self.assertEqual((again.status_code, again.json()["started"]), (409, False))
            release.set()
            self.assertTrue(until(lambda: self.daemon.setup.public()["state"] == "failed"))
            state = (await client.get("/api/setup", headers=admin)).json()
            self.assertEqual((state["lines"], state["result"]["error"]), (["downloading"], "stopped by the test"))


class FakeDriver:
    """A sandbox that is this machine: commands meant for the VM run here, in a folder standing in for it."""

    def __init__(self, vm, state="running", starts=True):
        self.vm, self.state, self.starts, self.calls = vm, state, starts, []

    def status(self):
        return self.state

    def start(self):
        self.calls.append("start")
        if self.starts:
            self.state = "running"
        return self.starts

    def stop(self):
        self.calls.append("stop")
        self.state = "stopped"
        return True

    def exec_argv(self, command, tty):
        if command[:2] == ["sh", "-c"]:              # cat > /app/desktop.py -> the stand-in folder
            target = self.vm / command[2].split("> ", 1)[1].lstrip("/").replace("/", "_")
            return ["sh", "-c", f"cat > '{target}'"]
        return command                                # bash -s: the script on stdin


class ComputerSetupTests(unittest.TestCase):
    def setUp(self):
        import pathlib
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.checkout, self.vm = self.root / "checkout", self.root / "vm"
        self.checkout.mkdir()
        self.vm.mkdir()
        for name in ("desktop.py", "server.py", "auth.py", "shell_session.py", "entrypoint.sh"):
            (self.checkout / name).write_text(f"# {name}\n")
        (self.checkout / "desktop-setup.sh").write_text("echo installing xfce\necho\necho Desktop installed\n")
        self.svc = Svc()
        self.svc.config.path = self.root / "config.json"

    def test_the_desktop_is_installed_in_the_sandbox_and_the_browser_moves_there(self):
        from coderunner.sandbox import computer
        driver, said = FakeDriver(self.vm), []
        self.assertTrue(computer.setup(self.svc.config, said.append, driver=driver, checkout=self.checkout))
        self.assertEqual(sorted(p.name for p in self.vm.iterdir()),
                         ["app_auth.py", "app_desktop.py", "app_server.py", "app_shell_session.py", "entrypoint.sh"])
        self.assertEqual((self.vm / "app_server.py").read_text(), "# server.py\n")
        self.assertEqual(said[1:3], ["installing xfce", "Desktop installed"])          # the script's own lines
        self.assertEqual(driver.calls, ["stop", "start"])                               # restarted to pick it up
        self.assertEqual(self.svc.config.get("browser.mode"), "computer")

    def test_a_failed_install_changes_nothing_and_says_why(self):
        from coderunner.sandbox import computer
        (self.checkout / "desktop-setup.sh").write_text("echo E: Unable to locate package xfce4-session\nexit 100\n")
        driver, said = FakeDriver(self.vm), []
        self.assertFalse(computer.setup(self.svc.config, said.append, driver=driver, checkout=self.checkout))
        self.assertEqual(said[-2:], ["E: Unable to locate package xfce4-session",
                                     "Error: the desktop did not install; see the lines above."])
        self.assertEqual((driver.calls, self.svc.config.get("browser.mode")), ([], "dedicated"))
        stopped, said = FakeDriver(self.vm, state="stopped", starts=False), []
        self.assertFalse(computer.setup(self.svc.config, said.append, driver=stopped, checkout=self.checkout))
        self.assertIn("the sandbox is not running and did not start", said[0])
        (self.checkout / "desktop-setup.sh").unlink()
        said = []
        self.assertFalse(computer.setup(self.svc.config, said.append, driver=FakeDriver(self.vm), checkout=self.checkout))
        self.assertIn("files are missing", said[0])


class ComputerRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_computer_panel_is_told_how_getting_the_computer_ready_stands_and_nothing_to_press(self):
        from coderunner.sandbox.keeper import Keeper
        svc = Svc()
        svc.registry = build_registry(svc)
        svc.browser = mock.Mock(mode="computer", running=lambda: False)
        svc.sandbox = mock.Mock(desktop=mock.AsyncMock(side_effect=RuntimeError("connection refused")),
                                base_url="http://coderunner.local:8222")
        svc.computer = mock.Mock(spec=Keeper)
        svc.computer.public.return_value = {"state": "working", "message": "Getting the computer ready…"}
        daemon = FakeDaemon(svc)
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            async def shown():
                got = (await client.post("/api/computer", json={}, headers=admin)).json()
                return got["available"], got["state"], got["message"]
            self.assertEqual(await shown(), (False, "working", "Getting the computer ready…"))
            svc.computer.kick.assert_called_with(desktop=True)            # asking for the screen gets it ready
            svc.computer.public.return_value = {"state": "failed", "message":
                                                "The assistant's computer could not be started. It will try again shortly."}
            self.assertEqual((await shown())[1], "failed")
            # The sandbox's own words (409) never reach the app, nor a command to type.
            svc.sandbox.desktop = mock.AsyncMock(return_value={
                "available": False, "running": False, "error": "This sandbox has no desktop. Run `coderunner computer setup`."})
            text = (await client.post("/api/computer", json={}, headers=admin)).text
            self.assertNotRegex(text, r"RuntimeError|coderunner|connection refused|Run `")
            # A keeper that thought it ready a moment ago still makes the panel say it is getting ready.
            svc.computer.public.return_value = {"state": "ready", "message": ""}
            self.assertEqual((await shown())[1:], ("working", "Getting the computer ready…"))
            svc.sandbox.desktop = mock.AsyncMock(return_value={"available": True, "running": True, "port": 5900,
                                                               "password": "pw"})
            got = (await client.post("/api/computer", json={}, headers=admin)).json()
            self.assertEqual((got["state"], got["password"], got["running"]), ("ready", "pw", True))
            svc.computer.saw_ready.assert_called()
            # There is no setup to ask for any more.
            self.assertEqual((await client.post("/api/computer/setup", json={}, headers=admin)).status_code, 404)
            self.assertEqual((await client.post("/api/computer", json={})).status_code, 401)


class RestartTests(unittest.IsolatedAsyncioTestCase):
    """After setup from the app the daemon starts again, in place, with the configuration setup wrote."""

    async def test_a_finished_setup_stops_the_daemon_shortly_after(self):
        import asyncio
        from types import SimpleNamespace

        from coderunner.server import daemon as daemon_module
        running = SimpleNamespace(restart_requested=False, _loop=asyncio.get_running_loop(), _stop=asyncio.Event())
        with mock.patch.object(daemon_module, "RESTART_AFTER_S", 0.05):
            threading.Thread(target=daemon_module.Daemon._after_setup, args=(running, {"ok": True})).start()
            await asyncio.wait_for(running._stop.wait(), 2)
        self.assertTrue(running.restart_requested)

    def test_the_daemon_starts_again_in_place(self):
        import sys

        from coderunner.server import daemon as daemon_module

        class Stopped:
            restart_requested = True

            async def run(self):
                return None

        with mock.patch.object(daemon_module, "Daemon", Stopped), \
                mock.patch.object(daemon_module.os, "execv") as execv:
            daemon_module.main()
        execv.assert_called_once_with(sys.executable, [sys.executable, "-m", "coderunner", "daemon"])
        Stopped.restart_requested = False
        with mock.patch.object(daemon_module, "Daemon", Stopped), \
                mock.patch.object(daemon_module.os, "execv") as execv:
            daemon_module.main()
        execv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
