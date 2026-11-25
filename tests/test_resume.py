"""A task goes on after the daemon restarts under it: from its journal, under the same id, with what it had."""

import asyncio
import copy
import json
import os
import tempfile
import time
import types
import unittest
from unittest import mock

from coderunner.agent import Agent
from coderunner.agent.journal import BROWSER_MOVED, INTERRUPTED, Journal
from coderunner.agent.loop import SKIPPED
from coderunner.agent.tools import Tool, ToolRegistry
from coderunner.safety.policy import Action, Policy
from coderunner.server import tasks as tasks_module
from coderunner.server.daemon import Daemon
from fakes import Svc


def call(*calls):
    """A reply with these tool calls: (name, arguments) each."""
    return {"content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}
                                          for i, (name, args) in enumerate(calls)]}


def answer(text):
    return {"content": text, "tool_calls": []}


NEVER = object()            # in a script: a reply the model is still writing when the daemon goes under it


class FakeLLM:
    context, remote, provider, model = 32768, False, "", "fake"

    def __init__(self):
        self.script, self.requests, self.contexts = [], [], []
        self.waiting = asyncio.Event()

    async def chat(self, messages, tools=None, think=None, **extra):
        self.requests.append(copy.deepcopy(messages))
        item = self.script.pop(0) if self.script else answer("done")
        if item is NEVER:
            self.waiting.set()
            await asyncio.Event().wait()
        reply = await item(messages) if callable(item) else item
        return {"usage": {}, "finish_reason": "stop", **reply}


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        home = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"CODERUNNER_HOME": home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.svc = Svc({"models": {"chat": "test"}, "owner": {"name": "Asha"}})
        self.svc.llm = self.llm = FakeLLM()
        self.ran: list[str] = []            # each tool call as it is run
        self.hold: set[str] = set()         # tools that hang (the daemon goes while they run) the next time
        self.inside = asyncio.Event()
        self.contexts = []

        def tool(name, result, replay_safe=False, effect=None):
            async def handler(arguments, ctx):
                self.ran.append(name)
                self.contexts.append(ctx)
                if effect is not None:
                    out = await effect(arguments, ctx)
                    if out is not None:
                        return out
                if name in self.hold:
                    self.hold.discard(name)
                    self.inside.set()
                    await asyncio.Event().wait()        # the daemon goes while it runs
                return result
            return Tool(name, name, {"type": "object", "properties": {}}, handler, replay_safe=replay_safe)

        async def planned(arguments, ctx):
            ctx.plan = [{"step": "Read the menu", "status": "done"}, {"step": "Book", "status": "doing"}]
            ctx.taint()

        async def send(arguments, ctx):
            allowed = await self.svc.policy.check(Action("channel.send", "telegram:ravi", "Send to Ravi: hi"),
                                                  origin=ctx.origin)
            return "Sent." if allowed else "Error: the owner did not approve sending this message."
        self.tools = [tool("read_page", "<untrusted>Menu: dosa ₹90</untrusted>", replay_safe=True, effect=planned),
                      tool("click", "Clicked: the table is held."), tool("look", "Two tables free.", replay_safe=True),
                      tool("send", "", effect=send), tool("browser_click", "Clicked.")]
        self.svc.registry = ToolRegistry(self.svc, self.tools)
        self.svc.agent = Agent(self.svc, self.llm, self.svc.registry)
        self.tasks = self.boot()

    def boot(self):
        """A daemon starting on the same home: its own task runner, approvals and conversation locks."""
        daemon = types.SimpleNamespace(svc=self.svc, config=self.svc.config, _locks={})
        daemon.handle_message = types.MethodType(Daemon.handle_message, daemon)
        self.remote_runs = []

        async def claude_agent():
            self.remote_runs.append(True)
            return self.svc.agent
        daemon.claude_agent = claude_agent
        self.bus = tasks_module.EventBus(self.svc.store)
        return tasks_module.Tasks(daemon, self.bus)

    async def restart(self):
        """The daemon goes (as at a crash: nothing more is written for its tasks) and starts again."""
        old = self.tasks
        old.shutdown()
        for task in old.running.values():   # those without a journal end as at the loop's close
            task.cancel()
        await asyncio.gather(*old.running.values(), return_exceptions=True)
        self.svc.audit.listeners.remove(old._on_audit)
        self.svc.policy = Policy(self.svc.config, self.svc.store, self.svc.audit)
        self.llm.waiting.clear()
        self.inside.clear()
        self.tasks = self.boot()

    def events(self, kind, conversation):
        return [e for e in self.bus.since(0, conversation=conversation) if e["type"] == kind]

    async def test_a_restart_while_the_model_writes_sends_the_same_request_again_with_the_run_restored(self):
        self.llm.script = [call(("read_page", {})), NEVER, answer("Dosa is ₹90; booked.")]
        task = self.tasks.start("app:a", "Book dinner", "admin-token")
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        in_flight = self.llm.requests[-1]
        await self.restart()
        self.assertEqual(self.tasks.get(task["id"])["status"], "running")      # not ended by the restart
        self.assertEqual(self.events("cancelled", "app:a"), [])
        self.assertEqual(self.svc.store.query("SELECT * FROM messages WHERE conversation = 'app:a'"), [])
        self.assertEqual(self.tasks.resume(), [task["id"]])
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], row["reply"]), ("done", "Dosa is ₹90; booked."))
        self.assertEqual(self.llm.requests[-1], in_flight)      # the request in flight, sent again as it was
        self.assertEqual(self.ran, ["read_page"])               # nothing done twice
        ctx = self.tasks_ctx(task["id"])
        self.assertEqual(ctx.plan[1], {"step": "Book", "status": "doing"})     # the run's state is back
        self.assertTrue(ctx.tainted)                            # and what it read is still untrusted
        self.assertEqual([(e["text"], e["restarted"]) for e in self.events("note", "app:a")],
                         [("Resumed after a restart", 1)])
        self.assertEqual(len(self.tasks.tasks("app:a")), 1)
        self.assertIsNone(Journal(self.svc.store, task["id"]).row())          # let go once it ended
        kept = [json.loads(r["content"]) for r in self.svc.store.query(
            "SELECT content FROM messages WHERE conversation = 'app:a' ORDER BY id")]
        self.assertEqual(kept, ["Book dinner", "Dosa is ₹90; booked."])

    def tasks_ctx(self, task_id):
        """The run's context (the event sender keeps it), from the events the app was sent."""
        return self._ctx[task_id]

    async def asyncSetUp(self):
        self._ctx = {}
        run = Agent.run

        async def keep(agent, ctx, text, images=None):
            self._ctx[getattr(ctx.events, "task", "")] = ctx
            return await run(agent, ctx, text, images)
        patcher = mock.patch.object(Agent, "run", keep)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_a_call_that_only_reads_runs_again_and_one_that_may_have_changed_something_does_not(self):
        self.hold = {"click"}
        self.llm.script = [call(("read_page", {}), ("click", {}), ("look", {})), answer("Held a table.")]
        task = self.tasks.start("app:b", "Hold a table", "admin-token")
        await asyncio.wait_for(self.inside.wait(), 5)
        await self.restart()
        self.tasks.resume()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual(row["status"], "done")
        self.assertEqual(self.ran, ["read_page", "click"])      # the page read is kept, the click not run again
        results = [m["content"] for m in self.llm.requests[-1] if m["role"] == "tool"]
        self.assertEqual(results, ["<untrusted>Menu: dosa ₹90</untrusted>", INTERRUPTED, SKIPPED])
        self.assertTrue(self.tasks_ctx(task["id"]).tainted)     # from the kept result's state

        # A read the daemon went under is simply read again.
        self.ran.clear()
        self.hold = {"look"}
        self.llm.script = [call(("look", {})), answer("Two are free.")]
        task = self.tasks.start("app:c", "Any tables?", "admin-token")
        await asyncio.wait_for(self.inside.wait(), 5)
        await self.restart()
        self.tasks.resume()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], self.ran), ("done", ["look", "look"]))
        self.assertEqual([m["content"] for m in self.llm.requests[-1] if m["role"] == "tool"], ["Two tables free."])

    async def test_after_a_restart_a_run_that_used_the_browser_is_told_to_look_again(self):
        self.llm.script = [call(("browser_click", {})), NEVER, answer("ok")]
        task = self.tasks.start("app:w", "Click it", "admin-token")
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        await self.restart()
        self.tasks.resume()
        await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual(self.llm.requests[-1][-1], {"role": "user", "content": BROWSER_MOVED})

    async def test_a_task_that_keeps_restarting_ends_as_failed(self):
        self.llm.script = [NEVER] * 4
        task = self.tasks.start("app:d", "Plan the trip", "admin-token")
        for times in range(3):
            await asyncio.wait_for(self.llm.waiting.wait(), 5)
            await self.restart()
            self.assertEqual(self.tasks.resume(), [task["id"]])
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        await self.restart()
        self.assertEqual(self.tasks.resume(), [])
        row = self.tasks.get(task["id"])
        self.assertEqual((row["status"], row["reply"]), ("failed", tasks_module.GAVE_UP))
        self.assertEqual(self.events("failed", "app:d")[-1]["incomplete"], True)
        self.assertEqual([e["restarted"] for e in self.events("note", "app:d")], [1, 2, 3])
        self.assertIsNone(Journal(self.svc.store, task["id"]).row())

    async def test_the_time_limit_counts_the_time_used_before_the_restart(self):
        async def slow(arguments, ctx):
            await asyncio.sleep(0.3)
        self.svc.registry.tools["look"] = Tool("look", "look", {"type": "object", "properties": {}}, slow,
                                               replay_safe=True)
        self.llm.script = [call(("look", {})), NEVER, answer("ok")]
        task = self.tasks.start("app:t", "Check prices", "check-runner", time_limit=100)
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        clock = Journal(self.svc.store, task["id"]).state()["clock"]
        self.assertLess(clock["deadline"], 99.75)
        self.assertLess(clock["stop"], 199.75)
        await self.restart()
        self.tasks.resume()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual(row["status"], "done")
        ctx = self.tasks_ctx(task["id"])
        # Not the full 100 s again: what was left when it was kept, measured from the restart.
        self.assertLess(ctx.deadline - time.monotonic(), clock["deadline"] + 0.01)
        self.assertLess(ctx.stop_at - time.monotonic(), clock["stop"] + 0.01)

    async def test_a_restart_past_the_limit_still_leaves_a_minute_for_the_answer(self):
        self.llm.script = [call(("look", {})), NEVER, answer("Answer from what I had.")]
        task = self.tasks.start("app:u", "Check prices", "check-runner", time_limit=100)
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        journal = Journal(self.svc.store, task["id"])
        state = journal.state()
        journal.save({**state, "clock": {"deadline": 0.0, "stop": 5.0}})
        await self.restart()
        self.tasks.resume()
        resumed = time.monotonic()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], row["reply"]), ("done", "Answer from what I had."))
        left = self.tasks_ctx(task["id"]).stop_at - resumed
        self.assertTrue(tasks_module.LAST_MINUTE_S - 1 < left <= tasks_module.LAST_MINUTE_S + 0.5)

    async def test_stop_holds_across_a_restart(self):
        self.hold = {"click"}
        self.llm.script = [call(("click", {}))]
        task = self.tasks.start("app:s", "Hold a table", "admin-token")
        await asyncio.wait_for(self.inside.wait(), 5)
        Journal(self.svc.store, task["id"]).stop()      # the owner's Stop, the daemon gone before the task ended
        await self.restart()
        self.assertEqual(self.tasks.resume(), [])
        row = self.tasks.get(task["id"])
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(self.ran, ["click"])
        # A Stop that ends as usual leaves nothing to resume.
        self.hold = {"click"}
        self.llm.script = [call(("click", {}))]
        task = self.tasks.start("app:s2", "Hold a table", "admin-token")
        await asyncio.wait_for(self.inside.wait(), 5)
        self.tasks.cancel(task["id"])
        await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertIsNone(Journal(self.svc.store, task["id"]).row())
        await self.restart()
        self.assertEqual(self.tasks.resume(), [])

    async def test_an_approval_the_restart_left_unanswered_is_asked_again(self):
        self.llm.script = [call(("send", {})), answer("Sent it.")]
        task = self.tasks.start("app:p", "Tell Ravi hi", "admin-token")
        for _ in range(100):
            if self.svc.policy.pending():
                break
            await asyncio.sleep(0.01)
        [first] = self.svc.policy.pending()
        await self.restart()
        self.tasks.resume()
        for _ in range(100):
            if self.svc.policy.pending():
                break
            await asyncio.sleep(0.01)
        [again] = self.svc.policy.pending()
        self.assertNotEqual(again.id, first.id)
        self.assertEqual(self.svc.policy.get(first.id).status, "expired")
        self.svc.policy.decide(again.id, True)
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], row["reply"]), ("done", "Sent it."))
        self.assertEqual(self.ran, ["send", "send"])    # asked twice, sent once (the first never got its yes)
        asked = [(e["event"], e["id"], e.get("status")) for e in self.bus.since(0, "app:p")
                 if e["type"] == "activity" and e["event"].startswith("approval.")]
        self.assertEqual(asked, [("approval.requested", first.id, "pending"), ("approval.decided", first.id, "expired"),
                                 ("approval.requested", again.id, "pending"), ("approval.decided", again.id, "approved")])

    async def test_queued_tasks_start_in_order_and_resuming_twice_makes_no_second_task(self):
        self.svc.config.set("agent.parallel_tasks", 1)
        self.llm.script = [NEVER, answer("First done."), answer("Second done.")]
        first = self.tasks.start("app:q1", "First", "admin-token")
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        second = self.tasks.start("app:q2", "Second", "admin-token")
        self.assertEqual(second["status"], "queued")
        steer = self.tasks.start("app:q2", "In French", "admin-token")       # read by it once it runs
        self.assertEqual(steer["id"], second["id"])
        await self.restart()
        self.assertEqual(self.tasks.resume(), [first["id"], second["id"]])
        self.assertEqual(self.tasks.resume(), [])
        self.assertEqual((self.tasks.get(first["id"])["status"], self.tasks.get(second["id"])["status"]),
                         ("running", "queued"))
        for task, reply in ((first, "First done."), (second, "Second done.")):
            row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
            self.assertEqual(row["reply"], reply)
        self.assertEqual(len(self.svc.store.query("SELECT id FROM tasks")), 2)
        self.assertEqual([m["content"] for m in self.llm.requests[-1][-2:]],
                         ["Second", "Asha adds while you work: In French"])
        self.assertEqual(self.events("note", "app:q2"), [])         # it had not started: nothing to say

    async def test_a_claude_task_resumes_on_claude(self):
        self.llm.script = [NEVER, answer("Done on Claude.")]
        task = self.tasks.start("app:r", "Try again", "admin-token", remote="claude-test")
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        self.remote_runs.clear()
        await self.restart()
        self.tasks.resume()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], row["reply"]), ("done", "Done on Claude."))
        self.assertEqual(self.remote_runs, [True])
        self.assertEqual(self.events("done", "app:r")[-1]["remote"], "claude-test")

    async def test_a_task_from_before_journals_or_the_checks_is_closed_as_before(self):
        self.svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                               "VALUES ('abc123abc123', 'app:old', 'Plan the trip', 'running', '', 1.0, NULL)")
        self.llm.script = [NEVER]
        check = self.tasks.start("app:check-1", "A check", "check")
        await asyncio.wait_for(self.llm.waiting.wait(), 5)
        await self.restart()
        self.assertEqual(self.tasks.resume(), [])
        self.assertEqual(self.tasks.get("abc123abc123")["status"], "cancelled")
        # The check's task had no journal: the shutdown left it to end as it always did.
        self.assertNotEqual(self.tasks.get(check["id"])["status"], "running")


if __name__ == "__main__":
    unittest.main()
