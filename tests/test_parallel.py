"""Two tasks at once: a place each, turns at the model, and the rest waiting in a queue."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from coderunner.agent import Agent, RunContext, build_registry
from coderunner.core.llm import LLM
from coderunner.server import tasks as tasks_module
from coderunner.server.gateway import admin_app
from fakes import FakeOpenAI, Svc, tool_call
from test_agent import FakeSandbox
from test_gateway import ADMIN, FakeDaemon


class TurnsTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_turn_at_a_time_the_open_task_first_then_whoever_asked_first(self):
        turns, order, inside = tasks_module.Turns(), [], 0

        async def step(name, hold):
            nonlocal inside
            async with turns.turn(name):
                inside += 1
                self.assertEqual(inside, 1)                     # never two at the model
                order.append(name)
                await hold.wait()
                inside -= 1

        first = asyncio.Event()
        rest = asyncio.Event()
        rest.set()
        running = [asyncio.create_task(step("a", first))]
        await asyncio.sleep(0)
        running += [asyncio.create_task(step(name, rest)) for name in ("b", "c", "d")]
        await asyncio.sleep(0)
        turns.focus = "d"                                       # the owner opens d's thread while it waits
        turns._waiting[:] = [(0 if name == "d" else 1, n, w) for (_, n, w), name in zip(turns._waiting, "bcd")]
        first.set()
        await asyncio.gather(*running)
        self.assertEqual(order, ["a", "d", "b", "c"])
        self.assertFalse(turns._busy)

    async def test_a_task_stopped_while_waiting_does_not_hold_the_turn(self):
        turns, order = tasks_module.Turns(), []
        release = asyncio.Event()

        async def step(name):
            async with turns.turn(name):
                order.append(name)
                await release.wait()

        a = asyncio.create_task(step("a"))
        await asyncio.sleep(0)
        b, c = asyncio.create_task(step("b")), asyncio.create_task(step("c"))
        await asyncio.sleep(0)
        b.cancel()                                              # stopped by the owner while it waited
        release.set()
        await asyncio.gather(a, c)
        self.assertEqual(order, ["a", "c"])
        self.assertTrue(b.cancelled())
        self.assertFalse(turns._busy)


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc()
        self.daemon = FakeDaemon(self.svc)
        self.gates: dict[str, asyncio.Event] = {}
        self.seen: list[tuple[str, int]] = []

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            self.seen.append((chat_id, events.slot))
            await self.gates.setdefault(chat_id, asyncio.Event()).wait()
            return f"done {chat_id}"
        self.daemon.handle_message = handle
        self.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = tasks_module.Tasks(self.daemon, self.bus)

    def kinds(self, task):
        return [e["type"] for e in self.bus.since(0) if e["task"] == task["id"]]

    async def test_two_run_the_third_waits_and_starts_when_a_place_is_free(self):
        self.assertEqual(self.tasks.limit(), 2)                 # a local model: two take turns
        one = self.tasks.start("app:one", "first", "owner")
        two = self.tasks.start("app:two", "second", "owner")
        three = self.tasks.start("app:three", "third", "owner")
        four = self.tasks.start("app:four", "fourth", "owner")
        await asyncio.sleep(0.01)
        self.assertEqual([t["status"] for t in (one, two, three, four)], ["running", "running", "queued", "queued"])
        self.assertEqual(self.seen, [("one", 0), ("two", 1)])   # each has a place of its own
        self.assertEqual(self.kinds(three), ["started", "queued"])
        self.assertEqual([e["ahead"] for e in self.bus.since(0) if e["type"] == "queued"], [0, 1])
        # Stopped before it started: it never runs.
        self.assertTrue(self.tasks.cancel(four["id"]))
        self.assertEqual((self.tasks.get(four["id"])["status"], self.kinds(four)[-1]), ("cancelled", "cancelled"))
        # The first ends: the third takes its place, the same place.
        self.gates["one"].set()
        await asyncio.sleep(0.05)
        self.assertEqual(self.tasks.get(one["id"])["status"], "done")
        self.assertEqual(self.tasks.get(three["id"])["status"], "running")
        self.assertEqual(self.seen[-1], ("three", 0))
        self.assertEqual(self.kinds(three), ["started", "queued", "resumed"])
        for name in ("two", "three"):
            self.gates.setdefault(name, asyncio.Event()).set()
        await asyncio.sleep(0.05)
        self.assertEqual((self.tasks.running, self.tasks.waiting, self.tasks.slots), ({}, [], {}))

    async def test_more_run_at_once_where_the_model_is_remote_and_the_owner_can_set_it(self):
        self.svc.remote = {"provider": "anthropic"}
        self.assertEqual(self.tasks.limit(), 4)
        self.svc.config.set("agent.parallel_tasks", 1)
        self.assertEqual(self.tasks.limit(), 1)

    async def test_a_task_left_waiting_by_a_restart_is_closed_like_a_running_one(self):
        self.svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                               "VALUES ('q1', 'app:old', 'x', 'queued', '', 1, NULL)")
        tasks_module.Tasks(self.daemon, self.bus)
        self.assertEqual(self.svc.store.one("SELECT status FROM tasks WHERE id = 'q1'")["status"], "cancelled")


class PlaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_task_names_its_slot_and_has_its_own_code_session(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.sandbox = FakeSandbox()
            svc.turns = tasks_module.Turns()
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            agent = Agent(svc, svc.llm, svc.registry)

            def events(event):
                pass
            events.slot = 1
            fake.script = [tool_call("run_python", {"code": "x = 1"}), {"role": "assistant", "content": "ok"}]
            await agent.run(RunContext(conversation="app:two", events=events), "set x")
            self.assertEqual([r.get("id_slot") for r in fake.requests], [1, 1])
            self.assertEqual(svc.sandbox.calls, [("x = 1", "task2")])
            # The first place is the usual session and slot 0; outside the app's tasks nothing is named.
            events.slot = 0
            fake.script = [tool_call("run_python", {"code": "y = 2"}), {"role": "assistant", "content": "ok"}]
            await agent.run(RunContext(conversation="app:one", events=events), "set y")
            self.assertEqual((fake.requests[-1].get("id_slot"), svc.sandbox.calls[-1]), (0, ("y = 2", "main")))
            fake.script = [{"role": "assistant", "content": "hello"}]
            await agent.run(RunContext(conversation="cli"), "hi")
            self.assertNotIn("id_slot", fake.requests[-1])
            self.assertFalse(svc.turns._busy)
            await svc.llm.aclose()

    async def test_the_open_task_is_told_to_the_daemon(self):
        svc = Svc()
        svc.registry = build_registry(svc)
        svc.turns = tasks_module.Turns()
        svc.browser = mock.Mock(focus=mock.AsyncMock(return_value=True))
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, FakeDaemon(svc))),
                                     base_url="http://127.0.0.1:8224") as client:
            self.assertEqual((await client.post("/api/focus", json={"conversation": "app:two"})).status_code, 401)
            reply = await client.post("/api/focus", json={"conversation": "app:two"}, headers=admin)
            self.assertEqual(reply.json(), {"conversation": "app:two", "shown": True})
        self.assertEqual(svc.turns.focus, "app:two")
        svc.browser.focus.assert_awaited_once_with("app:two")


if __name__ == "__main__":
    unittest.main()
