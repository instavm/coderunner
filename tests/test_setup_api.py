"""Setup from the app: the job that runs `coderunner setup`'s plan and setup in the daemon, and its route."""

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

    def test_the_plan_is_what_the_app_shows_before_anything_is_fetched(self):
        job = SetupJob(self.svc.config, plan=lambda config: PLAN)
        shown = job.plan()
        self.assertEqual(shown["hardware"], {"chip": "Apple M3 Pro", "ram_gb": 18, "gpu": "apple", "measured": ""})
        self.assertEqual((shown["model"]["id"], len(shown["model"]["reasons"])), ("qwen3.5-9b", 4))
        self.assertEqual([a["id"] for a in shown["alternatives"]], ["qwen3.5-4b"])
        self.assertEqual((shown["download_gb"], shown["free_gb"], shown["room_gb"]), (6.77, 11.4, 6.4))
        self.assertEqual(job.public()["plan"], shown)
        self.assertEqual((job.public()["configured"], job.public()["state"]), (False, "idle"))

    def test_setup_runs_in_the_background_and_its_progress_is_kept(self):
        release, finished = threading.Event(), []

        def setup(config, model_id=None, accept_license=False, progress=print):
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
        self.assertEqual(len(finished), 1)                              # the daemon starts again once

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
        self.assertEqual(job.public()["result"]["error"], "ValueError: unforeseen")
        self.assertEqual(finished, [])


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

        def setup(config, model_id=None, accept_license=False, progress=print):
            progress("downloading")
            release.wait(5)
            return {"ok": False, "error": "stopped by the test"}

        async with self.client() as client:
            self.assertEqual((await client.get("/api/setup", headers=admin)).status_code, 503)   # no job here
            self.daemon.setup = SetupJob(self.svc.config, plan=lambda config: PLAN, setup=setup, voice=False)
            self.assertEqual((await client.get("/api/setup")).status_code, 401)
            self.assertEqual((await client.post("/api/setup", json={"action": "start"})).status_code, 401)
            state = (await client.get("/api/setup", headers=admin)).json()
            self.assertEqual((state["configured"], state["state"], state["plan"]), (False, "idle", None))
            planned = await client.post("/api/setup", json={"action": "plan"}, headers=admin)
            self.assertEqual((planned.status_code, planned.json()["model"]["id"]), (200, "qwen3.5-9b"))
            self.assertEqual((await client.post("/api/setup", json={"action": "x"}, headers=admin)).status_code, 400)
            started = await client.post("/api/setup", json={"action": "start"}, headers=admin)
            self.assertEqual((started.status_code, started.json()["started"]), (200, True))
            again = await client.post("/api/setup", json={"action": "start"}, headers=admin)
            self.assertEqual((again.status_code, again.json()["started"]), (409, False))
            release.set()
            self.assertTrue(until(lambda: self.daemon.setup.public()["state"] == "failed"))
            state = (await client.get("/api/setup", headers=admin)).json()
            self.assertEqual((state["lines"], state["result"]["error"]), (["downloading"], "stopped by the test"))


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
