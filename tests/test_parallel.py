"""Two tasks at once: a place each, turns at the model, and the rest waiting in a queue."""

import asyncio
import json
import unittest
from unittest import mock

import httpx

from coderunner.agent import Agent, RunContext, build_registry
from coderunner.agent import loop as loop_module
from coderunner.agent.tools import Tool
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
        turns._waiting[:] = [(0 if name == "d" else 1, n, w) for (_, n, w), name in zip(turns._waiting, "bcd", strict=True)]
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
        # The list names each conversation's latest task: a window opened now knows what is running.
        listed = {c["conversation"]: (c["task"], c["status"]) for c in self.tasks.conversations()}
        self.assertEqual(listed["app:two"], (two["id"], "running"))
        self.assertEqual(listed["app:three"], (three["id"], "queued"))
        # A task asking for the owner from a thread that is not open is listed as waiting until answered.
        def asked(event, approval):
            self.bus.publish({"type": "activity", "event": event, "id": approval, "task": two["id"],
                              "conversation": "app:two"})
        waiting = lambda: {c["conversation"]: c["waiting"] for c in self.tasks.conversations()}  # noqa: E731
        asked("approval.requested", "a1")
        self.assertEqual((waiting()["app:two"], waiting()["app:one"]), (True, False))
        asked("approval.decided", "a1")
        self.assertFalse(waiting()["app:two"])
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

    async def test_the_checks_threads_do_not_push_the_owners_out_of_the_list(self):
        add = lambda i, conversation, created: self.svc.store.execute(  # noqa: E731
            "INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
            "VALUES (?, ?, 'x', 'done', '', ?, 1)", (f"t{i}", conversation, created))
        add(0, "app:mine", 1)
        add(1, "app:check_like", 2)                             # an underscore is not a wildcard
        for i in range(5):
            add(10 + i, f"app:check-world-{i}", 10 + i)
        listed = [c["conversation"] for c in self.tasks.conversations(limit=3)]
        self.assertEqual(listed, ["app:check-world-4", "app:check-world-3", "app:check-world-2",
                                  "app:check_like", "app:mine"])

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

    async def test_the_warm_up_reads_the_instructions_once_and_copies_them_to_the_other_task_slot(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.turns = tasks_module.Turns()
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            await Agent(svc, svc.llm, svc.registry).warm()
            self.assertEqual([w.get("id_slot") for w in fake.warmed], [0])
            self.assertEqual(fake.slot_actions, [(0, "save", "instructions-test.bin"), (1, "restore", "instructions-test.bin")])
            # Three at once: the instructions are copied to both other slots.
            svc.config.set("agent.parallel_tasks", 3)
            fake.slot_actions.clear()
            await Agent(svc, svc.llm, svc.registry).warm()
            self.assertEqual([a[:2] for a in fake.slot_actions], [(0, "save"), (1, "restore"), (2, "restore")])
            await svc.llm.aclose()

    async def test_a_task_starts_and_ends_with_its_slot_holding_only_the_instructions(self):
        svc = Svc()
        svc.turns = tasks_module.Turns()
        svc.llm = mock.Mock(remote=False, reset_slot=mock.AsyncMock(return_value=True))
        daemon = FakeDaemon(svc)

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            svc.turns.sizes["app:" + chat_id] = 18000
            return "done"
        daemon.handle_message = handle
        tasks = tasks_module.Tasks(daemon, tasks_module.EventBus(svc.store))
        tasks.start("app:one", "first", "owner")
        tasks.start("app:two", "second", "owner")
        await asyncio.sleep(0.05)
        self.assertEqual(sorted(c.args for c in svc.llm.reset_slot.await_args_list), [(0,), (0,), (1,), (1,)])
        self.assertEqual(svc.turns.sizes, {})               # what a finished task held is no longer counted

    async def test_a_runtime_without_slots_is_warmed_as_before(self):
        llm = LLM("http://127.0.0.1:9/v1", "test")
        self.assertEqual(await llm.copy_slot(0, [1]), 0)    # nothing there: no copy, no error
        self.assertEqual(await llm.copy_slot(0, []), 0)
        await llm.aclose()

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


def page(n: int, chars: int) -> str:
    return f'<untrusted source="browser_open">\nurl: https://example.com/{n}\n' + "w" * chars + "\n</untrusted>"


class RoomTests(unittest.IsolatedAsyncioTestCase):
    """The runtime's context is shared by every slot: a task keeps its conversation within what is left."""

    def agent(self, svc):
        svc.registry = build_registry(svc)
        return Agent(svc, svc.llm, svc.registry)

    def test_the_room_is_the_context_less_the_judge_and_the_other_slots(self):
        svc = Svc()
        svc.turns, svc.llm = tasks_module.Turns(), mock.Mock(remote=False)
        agent = self.agent(svc)
        agent.warm_tokens = 4800
        self.assertEqual(agent._room(32768, "app:one"), 32768 - loop_module.JUDGE_ROOM - 4800)
        svc.turns.sizes.update({"app:one": 9000, "app:two": 15000})        # the other task holds 15K
        self.assertEqual(agent._room(32768, "app:one"), 32768 - loop_module.JUDGE_ROOM - 15000)
        svc.llm.remote = True                                               # a remote model has its own
        self.assertEqual(agent._room(32768, "app:one"), 32768)

    def test_old_pages_go_first_then_other_long_results_and_one_step_is_cut_to_fit(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "tool", "content": page(1, 9000)}, {"role": "tool", "content": "r" * 5000},
                    {"role": "tool", "content": page(2, 9000)}, {"role": "tool", "content": "short"}]
        removed = Agent._drop_old(messages, 3, pages_only=True)
        self.assertEqual([i for i, _ in removed], [1])
        self.assertTrue(loop_module.shrunk(messages[1]["content"]))
        self.assertEqual(len(messages[2]["content"]), 5000)                  # not a page: kept this time
        self.assertEqual([i for i, _ in Agent._drop_old(messages, 3, pages_only=False)], [2])
        self.assertEqual(messages[4]["content"], "short")
        step = [{"role": "tool", "content": page(n, 12000)} for n in range(6)]
        messages += step
        Agent._cut_step(messages, 5, 6 * 9000)
        for m in messages[5:]:
            self.assertLess(len(m["content"]), 3200)
            self.assertTrue(m["content"].endswith(loop_module.CUT + "\n</untrusted>"))

    @staticmethod
    def steps(results: list[str]) -> list[dict]:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        for n, result in enumerate(results):
            messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}"}]},
                         {"role": "tool", "tool_call_id": f"c{n}", "content": result}]
        return messages

    def test_one_trim_goes_down_to_the_low_watermark_and_keeps_the_latest_two_steps(self):
        agent = self.agent(Svc())
        shell = "x" * 7000                                      # a long result that is not a page
        messages = self.steps([page(n, 7000) if n % 2 == 0 else shell for n in range(10)])
        known = [0, 0]
        before = agent._size(messages, known)
        done = agent._trim(messages, known, 32768, 32768)
        dropped = [n for n in range(10) if loop_module.shrunk(messages[3 + 2 * n]["content"])]
        # Every old page first (oldest first), then other long results only as far as the low watermark.
        self.assertEqual(dropped, [0, 1, 2, 3, 4, 6])
        self.assertEqual(done["results"], 6)
        self.assertFalse(done["must"])
        self.assertEqual(done["before"], before)
        self.assertLessEqual(done["after"], loop_module.LOW_WATER * 32768)
        self.assertEqual(done["after"], agent._size(messages, known))
        self.assertEqual(done["freed"], before - done["after"])
        # Right after, nothing is due: the next trim is far away.
        self.assertIsNone(agent._trim(messages, known, 32768, 32768))

    def test_a_trim_that_would_free_little_waits_unless_the_room_is_nearly_full(self):
        agent = self.agent(Svc())
        # Over 60% of the context, but nearly all of it in the latest two steps: one old page would go.
        messages = self.steps([page(0, 4000), page(1, 30000), page(2, 30000)])
        snapshot = [dict(m) for m in messages]
        self.assertIsNone(agent._trim(messages, [0, 0], 32768, 32768))
        self.assertEqual(messages, snapshot)                    # untouched: the runtime reads only what is new
        # Past FIT of a smaller room it must: the old page, then the long results before the latest step.
        done = agent._trim(messages, [0, 0], 32768, 22000)
        self.assertTrue(done["must"])
        self.assertEqual(done["results"], 2)
        self.assertTrue(loop_module.shrunk(messages[3]["content"]))
        self.assertTrue(loop_module.shrunk(messages[5]["content"]))
        self.assertIn("w" * 30000, messages[7]["content"])     # the latest step's page stays whole

    def test_old_medium_results_go_too_when_the_long_ones_do_not_reach_the_low_watermark(self):
        agent = self.agent(Svc())
        # Two old pages and many old click results under BIG_RESULT, then two steps kept whole.
        results = [page(0, 9000), page(1, 9000)] + ["c" * 1200] * 40 + ["k" * 800] * 2
        messages = self.steps(results)
        done = agent._trim(messages, [0, 0], 32768, 32768)
        self.assertIsNotNone(done)
        self.assertLessEqual(done["after"], loop_module.LOW_WATER * 32768)
        tools = [m for m in messages if m["role"] == "tool"]
        self.assertTrue(all(loop_module.shrunk(m["content"]) for m in tools[:2]))
        self.assertTrue(any(loop_module.shrunk(m["content"]) for m in tools[2:42]))
        self.assertEqual([m["content"] for m in tools[-2:]], ["k" * 800] * 2)     # the latest two steps stay
        # Results too short to shrink are never touched.
        short = self.steps(["s" * 500] * 200)
        snapshot = [dict(m) for m in short]
        self.assertIsNone(agent._trim(short, [0, 0], 32768, 32768))
        self.assertEqual(short, snapshot)

    def test_a_trim_that_must_but_would_free_almost_nothing_is_not_made(self):
        agent = self.agent(Svc())
        # Past FIT of the room with one old long result: dropping it frees ~400 tokens of ~28,000, at the price
        # of reading everything after it again.
        messages = self.steps(["z" * 2000] + ["s" * 500] * 165)
        snapshot = [dict(m) for m in messages]
        self.assertGreater(agent._size(messages, [0, 0]), loop_module.FIT * 32768)
        self.assertIsNone(agent._trim(messages, [0, 0], 32768, 32768))
        self.assertEqual(messages, snapshot)

    async def test_six_pages_at_once_are_cut_and_a_task_that_cannot_fit_stops_in_a_sentence(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.turns = tasks_module.Turns()
            svc.llm = LLM(fake.base_url, "test")
            svc.llm.context = 16000
            svc.registry = build_registry(svc)
            async def opened(args, ctx):
                return page(1, 11000)
            svc.registry.tools["browser_open"] = Tool("browser_open", "Open a page.", {
                "type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}, opened)
            agent = Agent(svc, svc.llm, svc.registry)
            agent.warm_tokens = 1000
            calls = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{n}", "type": "function",
                 "function": {"name": "browser_open", "arguments": f'{{"url": "https://example.com/{n}"}}'}}
                for n in range(6)]}
            fake.script = [calls, {"role": "assistant", "content": "done"}]
            await agent.run(RunContext(conversation="app:one"), "compare six pages")
            sent = sum(len(m["content"]) for m in fake.requests[-1]["messages"] if m["role"] == "tool")
            self.assertLess(sent / loop_module.NEW_CHARS, loop_module.FIT * agent._room(16000, "app:one"))
            self.assertIn(loop_module.CUT, fake.requests[-1]["messages"][-1]["content"])
            # Instructions alone larger than the room: no request that would fail, a plain sentence instead.
            svc.llm.context = 3000
            before = len(fake.requests)
            reply = await agent.run(RunContext(conversation="app:two"), "hello")
            self.assertEqual(len(fake.requests), before)
            self.assertTrue(reply.startswith(loop_module.TOO_LONG))
            await svc.llm.aclose()


    async def test_a_task_that_fills_its_room_answers_from_what_it_found(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.llm = LLM(fake.base_url, "test")
            svc.llm.context = 100000
            svc.registry = build_registry(svc)

            async def looked(args, ctx):
                return "x" * 590                   # a short result each time: nothing the trims let go
            svc.registry.tools["lookup"] = Tool("lookup", "Look up a review.", {
                "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}, looked)
            agent = Agent(svc, svc.llm, svc.registry)
            step = lambda n: {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function",
                              "function": {"name": "lookup", "arguments": f'{{"q": "review {n}", "say": "Read review {n}."}}'}}]}
            fake.script = [step(n) for n in range(24)] + [{"role": "assistant", "content": "The Sony WH-CH720N, from two sites."}]
            rooms = iter([100000] * 24 + [6000] * 5)
            with mock.patch.object(agent, "_room", side_effect=lambda context, conversation: next(rooms)):
                ctx = RunContext(conversation="app:full")
                reply = await agent.run(ctx, "Best headphones?")
            self.assertEqual(reply, "The Sony WH-CH720N, from two sites.")
            sent = fake.requests[-1]["messages"]
            self.assertEqual(sent[-1]["content"], loop_module.OUT_OF_ROOM)
            self.assertIn("- Read review 23.", json.dumps(sent))            # what it said along the way
            self.assertEqual([m["content"] for m in sent if m["role"] == "user"][1], "Best headphones?")
            self.assertLess(len(sent), 12)                                  # the middle steps let go
            self.assertTrue(ctx.incomplete)
            await svc.llm.aclose()


if __name__ == "__main__":
    unittest.main()
