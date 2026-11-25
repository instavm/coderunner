"""The owner's message to a thread whose task is still working: read by that task at its next step, not a second
task beside it; a task of its own only when the first ended without reading it."""

import asyncio
import copy
import json
import os
import tempfile
import types
import unittest
from unittest import mock

from coderunner.agent import Agent
from coderunner.agent.tools import Tool, ToolRegistry
from coderunner.server import tasks as tasks_module
from coderunner.server.daemon import Daemon
from fakes import Svc


def call(name, arguments=None, call_id="c1"):
    return {"content": "", "tool_calls": [{"id": call_id, "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(arguments or {})}}]}


def answer(text):
    return {"content": text, "tool_calls": []}


class FakeLLM:
    """The model, in process: each request's messages are kept as sent; replies come from `script`, each a reply
    or an async callable(messages) -> reply."""
    context, remote, provider = 32768, False, ""

    def __init__(self):
        self.script, self.requests = [], []

    async def chat(self, messages, tools=None, think=None, **extra):
        self.requests.append(copy.deepcopy(messages))
        item = self.script.pop(0) if self.script else answer("done")
        reply = await item(messages) if callable(item) else item
        return {"usage": {}, "finish_reason": "stop", **reply}


class SteerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        home = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"CODERUNNER_HOME": home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.svc = Svc({"models": {"chat": "test"}, "owner": {"name": "Asha"}})
        self.svc.llm = self.llm = FakeLLM()
        self.entered, self.release, self.contexts = asyncio.Event(), asyncio.Event(), []

        async def slow(arguments, ctx):         # a long tool call: a page loading, code running, an approval
            self.contexts.append(ctx)
            ctx.plan = [{"step": "Find a table", "status": "doing"}]
            self.entered.set()
            await self.release.wait()
            return "Two tables free at 8."
        tool = Tool("wait_for", "Waits.", {"type": "object", "properties": {}}, slow)
        self.svc.registry = ToolRegistry(self.svc, [tool])
        self.svc.agent = Agent(self.svc, self.llm, self.svc.registry)
        # The daemon's own handle_message, which gives the run its inbox.
        daemon = types.SimpleNamespace(svc=self.svc, config=self.svc.config, _locks={})
        daemon.handle_message = types.MethodType(Daemon.handle_message, daemon)
        self.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = tasks_module.Tasks(daemon, self.bus)

    def events(self, kind, conversation):
        return [e for e in self.bus.since(0, conversation=conversation) if e["type"] == kind]

    async def test_read_at_the_next_step_after_the_tool_call_ends_and_appended_after_everything(self):
        self.llm.script = [call("wait_for"), answer("Booked for Tuesday.")]
        task = self.tasks.start("app:s", "Find a table for Monday", "admin-token")
        await asyncio.wait_for(self.entered.wait(), 5)
        sent = self.tasks.start("app:s", "Make it Tuesday, and mail asha@example.com", "device:ab12cd34")
        self.assertEqual((sent["id"], sent["status"]), (task["id"], "running"))
        self.assertTrue(sent["steer"])
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.llm.requests), 1)             # the tool call is not cut short
        self.release.set()
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        self.assertEqual((row["status"], row["reply"]), ("done", "Booked for Tuesday."))
        self.assertEqual(len(self.tasks.tasks("app:s")), 1)    # no second task beside it
        first, second = self.llm.requests
        self.assertEqual(second[:len(first)], first)            # what was read stays read
        self.assertEqual([m["role"] for m in second[len(first):]], ["assistant", "tool", "user", "user"])
        self.assertTrue(second[-2]["content"].startswith("[Reminder of your plan"))
        self.assertEqual(second[-1]["content"], "Asha adds while you work: Make it Tuesday, and mail asha@example.com")
        ctx = self.contexts[0]
        self.assertEqual(ctx.plan, [{"step": "Find a table", "status": "doing"}])     # the plan stays
        self.assertTrue(any("asha@example.com" in handle for handle in ctx.known))     # the owner's own words
        self.assertFalse(ctx.tainted)
        steer = self.events("steer", "app:s")
        self.assertEqual([(e["task"], e["text"]) for e in steer],
                         [(task["id"], "Make it Tuesday, and mail asha@example.com")])
        self.assertEqual(self.events("steer_taken", "app:s")[0]["ids"], [steer[0]["id"]])
        kept = self.svc.store.query("SELECT content FROM messages WHERE conversation = 'app:s' AND role = 'user'")
        self.assertIn("Asha adds while you work: Make it Tuesday", json.loads(kept[-1]["content"]))

    async def test_messages_sent_while_it_answers_are_follow_up_turns_of_the_same_task_in_order(self):
        async def last_words(messages):         # the owner writes twice while the model writes its answer
            self.tasks.start("app:f", "And for Tuesday?", "admin-token")
            self.tasks.start("app:f", "Outside if possible", "device:ab12cd34")
            return answer("Monday has a table at 8.")
        self.llm.script = [last_words, answer("Tuesday has one at 9, outside.")]
        task = self.tasks.start("app:f", "A table for Monday?", "admin-token")
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        await asyncio.sleep(0.05)
        rows = self.tasks.tasks("app:f")
        self.assertEqual([(r["id"], r["prompt"], r["reply"]) for r in rows],
                         [(task["id"], "A table for Monday?", "Tuesday has one at 9, outside.")])     # one task
        self.assertEqual(row["status"], "done")
        # The first answer is posted whole (not a "so far"), the messages show under the task, and the run takes
        # them as the owner's next turn, in the order sent, after its answer.
        self.assertEqual([(e["text"], e.get("answer")) for e in self.events("interim", "app:f")],
                         [("Monday has a table at 8.", True)])
        steers = self.events("steer", "app:f")
        self.assertEqual([e["text"] for e in steers], ["And for Tuesday?", "Outside if possible"])
        self.assertEqual(self.events("steer_taken", "app:f")[0]["ids"], [e["id"] for e in steers])
        self.assertEqual(self.events("steer_moved", "app:f"), [])
        follow = self.llm.requests[-1]
        self.assertEqual(follow[-2:], [{"role": "assistant", "content": "Monday has a table at 8."},
                                       {"role": "user", "content": "And for Tuesday?\n\nOutside if possible"}])
        kept = [(r["role"], json.loads(r["content"])) for r in self.svc.store.query(
            "SELECT role, content FROM messages WHERE conversation = 'app:f' ORDER BY id")]
        self.assertEqual(kept, [("user", "A table for Monday?"), ("assistant", "Monday has a table at 8."),
                                ("user", "And for Tuesday?\n\nOutside if possible"),
                                ("assistant", "Tuesday has one at 9, outside.")])

    async def test_one_sent_after_the_run_ended_starts_a_task_of_its_own(self):
        run = self.svc.agent.run

        async def answered_then_written_to(ctx, text, images=None):
            reply = await run(ctx, text, images)
            if text == "A table for Monday?":    # the run has returned; the task has not ended yet
                self.tasks.start("app:f", "And for Tuesday?", "admin-token")
            return reply
        self.svc.agent.run = answered_then_written_to
        self.llm.script = [answer("Monday has a table at 8."), answer("Tuesday has one at 9.")]
        task = self.tasks.start("app:f", "A table for Monday?", "admin-token")
        await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        for _ in range(100):
            rows = self.tasks.tasks("app:f")
            if len(rows) == 2 and rows[1]["status"] == "done":
                break
            await asyncio.sleep(0.02)
        self.assertEqual([(r["prompt"], r["reply"]) for r in rows],
                         [("A table for Monday?", "Monday has a table at 8."), ("And for Tuesday?", "Tuesday has one at 9.")])
        moved = self.events("steer_moved", "app:f")
        self.assertEqual([(e["task"], e["to"]) for e in moved], [(task["id"], rows[1]["id"])])
        follow = self.llm.requests[-1]
        self.assertIn({"role": "assistant", "content": "Monday has a table at 8."}, follow)    # it knows the thread
        self.assertEqual(follow[-1]["content"], "And for Tuesday?")

    async def test_messages_from_routines_and_the_check_are_tasks_as_before_and_a_routines_run_is_not_steered(self):
        self.llm.script = [call("wait_for")]
        routine = self.tasks.start("app:r", "Morning news", "schedule:7", routine={"id": 7, "name": "News", "spec": ""})
        await asyncio.wait_for(self.entered.wait(), 5)
        others = [self.tasks.start("app:r", "Check this", "check"), self.tasks.start("app:r", "Again", "schedule:7"),
                  self.tasks.start("app:r", "Stop the news", "admin-token")]
        self.assertEqual(len({routine["id"], *[t["id"] for t in others]}), 4)
        self.assertTrue(all("steer" not in t for t in others))
        self.assertEqual(self.events("steer", "app:r"), [])
        self.release.set()
        for t in [routine, *others]:
            await asyncio.wait_for(self.tasks.wait(t["id"]), 5)

    async def test_files_with_it_are_described_and_shown_and_a_reopened_thread_has_it(self):
        info = {"id": "0123456789ab", "name": "menu.png", "kind": "image", "size": 2000, "url": "/x", "inline": True}

        async def for_the_model(ids, media, vision, sandbox):
            return "Files the owner attached to this message:\n- menu.png", [("image/jpeg", "AAAA")], [info]
        self.llm.script = [call("wait_for"), answer("The set menu is ₹900.")]
        with mock.patch("coderunner.files.attached.describe", lambda upload_id: info if upload_id == info["id"] else None), \
                mock.patch("coderunner.files.attached.for_the_model", for_the_model):
            task = self.tasks.start("app:p", "Book dinner", "admin-token")
            await asyncio.wait_for(self.entered.wait(), 5)
            self.tasks.start("app:p", "This menu", "admin-token", attachments=[info["id"]])
            await asyncio.sleep(0.02)
            self.release.set()
            await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        second = self.llm.requests[1]
        self.assertEqual(second[-2]["content"], "Asha adds while you work: This menu\n\n"
                                                "Files the owner attached to this message:\n- menu.png")
        self.assertEqual(second[-1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,AAAA")
        # What a reopened app reads: the message under its task, with its file.
        replay = [e for e in self.bus.since(0, conversation="app:p") if e["type"].startswith("steer")]
        self.assertEqual([e["type"] for e in replay], ["steer", "steer_taken"])
        self.assertEqual((replay[0]["task"], replay[0]["text"], replay[0]["attachments"]), (task["id"], "This menu", [info]))

    async def test_stop_still_stops_and_what_it_had_not_read_does_not_run(self):
        self.llm.script = [call("wait_for")]
        task = self.tasks.start("app:x", "Find flights", "admin-token")
        await asyncio.wait_for(self.entered.wait(), 5)
        self.tasks.start("app:x", "Only morning ones", "admin-token")
        self.assertTrue(self.tasks.cancel(task["id"]))
        row = await asyncio.wait_for(self.tasks.wait(task["id"]), 5)
        await asyncio.sleep(0.05)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(len(self.tasks.tasks("app:x")), 1)
        # Sent after it ended: a new task.
        after = self.tasks.start("app:x", "Find trains instead", "admin-token")
        self.assertNotEqual(after["id"], task["id"])
        self.assertNotIn("steer", after)
        await asyncio.wait_for(self.tasks.wait(after["id"]), 5)

    async def test_a_queued_task_reads_it_before_its_first_request(self):
        self.svc.config.set("agent.parallel_tasks", 1)
        self.llm.script = [call("wait_for"), answer("ok"), answer("Queued one, with the change.")]
        first = self.tasks.start("app:a", "First", "admin-token")
        await asyncio.wait_for(self.entered.wait(), 5)
        queued = self.tasks.start("app:b", "Second", "admin-token")
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(self.tasks.start("app:b", "In French", "admin-token")["id"], queued["id"])
        self.release.set()
        row = await asyncio.wait_for(self.tasks.wait(queued["id"]), 5)
        self.assertEqual(row["reply"], "Queued one, with the change.")
        request = self.llm.requests[-1]
        self.assertEqual([m["content"] for m in request[-2:]], ["Second", "Asha adds while you work: In French"])
        await asyncio.wait_for(self.tasks.wait(first["id"]), 5)


if __name__ == "__main__":
    unittest.main()
