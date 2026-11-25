"""Remote mode: Claude on Anthropic's API as the assistant's model, with the owner's key. Not private, and said so."""

import json
import os
import pathlib
import stat
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.core.claude import Claude, convert, tools_of
from coderunner.core.llm import LLMError
from coderunner.files.discover import FileAccessError, Folders
from coderunner.files.kinds import holds_key
from coderunner.models import remote
from coderunner.safety.decision import ActionCheck, DecisionModel
from fakes import Svc

try:
    import anthropic
    import httpx2
except ImportError:                     # the client library is installed when remote mode is first turned on
    anthropic = httpx2 = None

KEY = "sk-ant-" + "test0123456789abcdefghij" * 2
TOOLS = [{"type": "function", "function": {"name": "browser_open", "description": "Open a page.",
                                           "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]


def message(content, stop="end_turn", usage=None):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 7, "cache_read_input_tokens": 900,
                      "cache_creation_input_tokens": 30, **(usage or {})}}


class Home(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = pathlib.Path(self._tmp.name).resolve()
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)


class ConversionTests(unittest.TestCase):
    def test_a_conversation_of_the_agent_loop_becomes_messages_api_turns(self):
        picture = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
        system, turns = convert([
            {"role": "system", "content": "You are the owner's assistant."},
            {"role": "user", "content": "Today is Sunday."},
            {"role": "user", "content": [{"type": "text", "text": "What is on this receipt?"}, picture]},
            {"role": "assistant", "content": "Let me look.", "tool_calls": [
                {"id": "toolu_1", "type": "function", "function": {"name": "browser_open", "arguments": '{"url": "https://a.example"}'}},
                {"id": "toolu_2", "type": "function", "function": {"name": "browser_open", "arguments": "not json"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "Page: A"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": "Screenshot (untrusted page content):"}, picture]},
            {"role": "assistant", "content": "", "tool_calls": []},
        ])
        self.assertEqual(system, [{"type": "text", "text": "You are the owner's assistant."}])
        self.assertEqual([t["role"] for t in turns], ["user", "assistant", "user"])
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}
        self.assertEqual(turns[0]["content"], [{"type": "text", "text": "Today is Sunday."},
                                               {"type": "text", "text": "What is on this receipt?"}, image])
        self.assertEqual(turns[1]["content"], [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "browser_open", "input": {"url": "https://a.example"}},
            {"type": "tool_use", "id": "toolu_2", "name": "browser_open", "input": {}}])
        # Every result of the turn's calls is in the one user turn after it, ahead of what else is said there.
        self.assertEqual(turns[2]["content"], [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Page: A"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "(no output)"},
            {"type": "text", "text": "Screenshot (untrusted page content):"}, image])
        self.assertEqual(tools_of(TOOLS), [{"name": "browser_open", "description": "Open a page.",
                                            "input_schema": TOOLS[0]["function"]["parameters"]}])
        # A picture is only ever sent as data: an address is not handed over to be fetched.
        self.assertEqual(convert([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}])[1], [])


@unittest.skipIf(anthropic is None, "the anthropic client library is not installed")
class ClaudeTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, **options):
        self.requests = []

        def record(request):
            self.requests.append((dict(request.headers), json.loads(request.content)))
            return handler(request)
        return Claude(KEY, http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(record)), **options)

    async def test_a_step_with_tools_is_asked_and_read_as_the_agent_loop_expects(self):
        reply = message([{"type": "text", "text": "Opening both."},
                         {"type": "tool_use", "id": "toolu_a", "name": "browser_open", "input": {"url": "https://a.example"}},
                         {"type": "tool_use", "id": "toolu_b", "name": "browser_open", "input": {"url": "https://b.example"}}],
                        stop="tool_use")
        llm = self.client(lambda r: httpx2.Response(200, json=reply, headers={"request-id": "req_123"}))
        got = await llm.chat([{"role": "system", "content": "Instructions."}, {"role": "user", "content": "Compare a and b."}],
                             tools=TOOLS, think=False)
        self.assertEqual(got["content"], "Opening both.")
        self.assertEqual([(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in got["tool_calls"]],
                         [("toolu_a", "browser_open", {"url": "https://a.example"}),
                          ("toolu_b", "browser_open", {"url": "https://b.example"})])
        self.assertEqual(got["usage"], {"prompt_tokens": 942, "completion_tokens": 7, "cache_read_input_tokens": 900,
                                        "cache_creation_input_tokens": 30})
        headers, sent = self.requests[0]
        self.assertEqual((headers["x-api-key"], sent["model"]), (KEY, "claude-haiku-5-5"))
        # The instructions and the tools are kept for the next step, and so is the conversation so far.
        self.assertEqual(sent["system"], [{"type": "text", "text": "Instructions.", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(sent["cache_control"], {"type": "ephemeral"})
        # No thinking blocks to carry (the loop edits earlier turns), no sampling settings, no forced tool.
        self.assertEqual((sent["thinking"], sent["output_config"]), ({"type": "disabled"}, {"effort": "medium"}))
        self.assertFalse({"temperature", "top_p", "top_k", "tool_choice"} & set(sent))
        self.assertEqual(llm.last, {"model": "claude-haiku-5-5", "request_id": "req_123"})
        self.assertEqual((llm.usage["requests"], llm.usage["cache_read_input_tokens"]), (1, 900))
        self.assertIsNotNone(llm.step_seconds())
        await llm.chat([{"role": "user", "content": "Sum up."}], tools=TOOLS, tool_choice="none")
        self.assertEqual(self.requests[1][1]["tool_choice"], {"type": "none"})
        await llm.aclose()

    async def test_a_refusal_a_cut_off_step_and_api_failures_are_said_plainly(self):
        refused = message([], stop="refusal")
        refused["stop_details"] = {"type": "refusal", "category": "cyber", "explanation": "x"}
        llm = self.client(lambda r: httpx2.Response(200, json=refused))
        with self.assertRaisesRegex(LLMError, r"Claude declined this request \(cyber\)"):
            await llm.chat([{"role": "user", "content": "x"}])
        await llm.aclose()
        cut = message([{"type": "text", "text": "Opening"},
                       {"type": "tool_use", "id": "toolu_a", "name": "browser_open", "input": {"url": "https://a.example"}},
                       {"type": "tool_use", "id": "toolu_b", "name": "browser_open", "input": {}}], stop="max_tokens")
        llm = self.client(lambda r: httpx2.Response(200, json=cut))
        got = await llm.chat([{"role": "user", "content": "x"}], tools=TOOLS)
        self.assertEqual([c["id"] for c in got["tool_calls"]], ["toolu_a"])       # the one cut off is not run
        await llm.aclose()
        for status, body, said in ((401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
                                    "did not accept the key"),
                                   (404, {"type": "error", "error": {"type": "not_found_error", "message": "model"}}, "no model named"),
                                   (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad shape"}},
                                    "refused the request"),
                                   (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}, "limiting requests"),
                                   (529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}, "returned 529")):
            llm = self.client(lambda r, status=status, body=body: httpx2.Response(status, json=body))
            llm._client = llm._client.with_options(max_retries=0)
            with self.assertRaises(LLMError) as raised:
                await llm.chat([{"role": "user", "content": "x"}])
            self.assertIn(said, str(raised.exception))
            self.assertNotIn(KEY, str(raised.exception))
            await llm.aclose()

        def down(request):
            raise httpx2.ConnectError("no route")
        llm = self.client(down)
        llm._client = llm._client.with_options(max_retries=0)
        with self.assertRaisesRegex(LLMError, "could not be reached"):
            await llm.chat([{"role": "user", "content": "x"}])
        await llm.aclose()

    async def test_the_step_judge_asks_claude_for_its_letter_with_the_worked_examples(self):
        answers = iter(["A", "B", "I cannot say"])
        llm = self.client(lambda r: httpx2.Response(200, json=message([{"type": "text", "text": next(answers)}])))
        check = ActionCheck(DecisionModel(llm.base_url, llm.model, api="chat", claude=llm))
        self.assertFalse(await check.commits('Site: x\nAction: click button "Next month"'))
        self.assertTrue(await check.commits('Site: x\nAction: click button "Pay now"'))
        self.assertTrue(await check.commits("Site: x"))                       # no letter: the owner is asked
        sent = self.requests[0][1]
        self.assertIn("commit", sent["system"][0]["text"])
        self.assertEqual(len(sent["messages"]), 19)                             # nine examples, then the step
        self.assertEqual(json.loads(sent["messages"][-1]["content"][0]["text"])["state"],
                         'Site: x\nAction: click button "Next month"')
        self.assertFalse({"temperature", "tools"} & set(sent))
        await llm.aclose()


class KeyTests(Home):
    async def test_the_key_is_kept_where_only_the_owner_reads_it_and_never_repeated(self):
        given = self.home / "Downloads" / "antkey"
        given.parent.mkdir()
        given.write_text(f"ANTHROPIC_API_KEY={KEY}\n")
        self.assertEqual(remote.key(), "")
        remote.keep_key_file(str(given))
        kept = remote.key_path()
        self.assertEqual((kept, kept.read_text().strip(), stat.S_IMODE(kept.stat().st_mode)),
                         (self.home / "assistant" / "anthropic-key", KEY, 0o600))
        self.assertTrue(given.exists())                                         # the owner's own file is left alone
        for bad in ("hello", "", "sk-ant-short"):
            with self.assertRaises(ValueError) as refused:
                remote.keep_key(bad)
            self.assertNotIn(bad or "x-none", str(refused.exception))
        with self.assertRaisesRegex(ValueError, "could not be read"):
            remote.keep_key_file(str(self.home / "nowhere"))
        config = Svc().config
        self.assertIsNone(remote.active(config))                                # a key alone turns nothing on
        with mock.patch.object(remote, "install_sdk"):
            now = remote.turn_on(config, "claude-haiku-5-5")
        self.assertEqual((now["model"], now["name"], remote.ready(config)), ("claude-haiku-5-5", "Claude Haiku 5.5", True))
        self.assertNotIn(KEY, json.dumps(config.data))                          # never in the config
        with self.assertRaises(ValueError):
            remote.turn_on(config, "gpt-something")
        remote.turn_off(config)
        self.assertEqual((remote.active(config), remote.key()), (None, KEY))    # kept for next time
        remote.turn_off(config, forget=True)
        self.assertEqual(remote.key(), "")

    async def test_a_file_that_holds_a_key_is_never_read_to_a_model_or_indexed(self):
        shared = self.home.parent / (self.home.name + "-docs")
        shared.mkdir()
        self.addCleanup(lambda: __import__("shutil").rmtree(shared, ignore_errors=True))
        (shared / "antkey").write_text(KEY + "\n")
        (shared / "notes.txt").write_text("the staging token is " + KEY + " for now\n")
        (shared / "deploy.txt").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
        (shared / "plain.txt").write_text("Dentist on Friday. sk-ant is how their keys start.\n")
        (shared / "long.txt").write_text("x" * 9000 + KEY)                      # a long document that quotes one
        self.assertEqual([holds_key(shared / n) for n in ("antkey", "notes.txt", "deploy.txt", "plain.txt", "long.txt")],
                         [True, True, True, False, False])
        folders = Folders(Svc({"files": {"roots": [str(shared)]}}).config)
        for name in ("antkey", "notes.txt", "deploy.txt"):
            with self.assertRaisesRegex(FileAccessError, "may hold secrets"):
                folders.resolve(str(shared / name))
        self.assertEqual(folders.resolve(str(shared / "plain.txt")), (shared / "plain.txt").resolve())
        from coderunner.files.extract import Extractor, Media
        pieces = await Extractor(Media()).pieces(shared / "notes.txt", "text")
        self.assertFalse(any(KEY in (piece.text or "") or "staging" in (piece.text or "") for piece in pieces))


class DaemonTests(Home):
    async def test_the_status_names_the_models_really_in_use_in_each_mode(self):
        from coderunner.browser import page
        from coderunner.core import paths
        from coderunner.core.config import Config
        from coderunner.server.daemon import Daemon
        self.addCleanup(setattr, page, "LINK_CHARS", page.LINK_CHARS)       # remote mode gives links whole
        paths.ensure_dirs()
        data = {"models": {"runtime": "openai", "base_url": "http://127.0.0.1:9/v1", "chat": "coderunner-qwen3.5-9b",
                           "embedding": "coderunner-embeddinggemma-2", "driver": {"model": "fara1.5-4b", "exclusive": True}},
                "sandbox": {"driver": "none"}, "browser": {"mode": "dedicated"}}
        local = Daemon(Config(json.loads(json.dumps(data)), path=paths.config_path()))
        self.assertIsNone(local.svc.remote)
        shown = await local.status()
        self.assertEqual((shown["remote"], shown["models"]["chat"], shown["models"]["driver"]),
                         (None, "coderunner-qwen3.5-9b", "fara1.5-4b"))
        self.assertIn("browser_task", local.svc.registry.tools)
        if anthropic is None:
            self.skipTest("the anthropic client library is not installed")
        remote.keep_key(KEY)
        data["models"]["remote"] = {"provider": "anthropic", "model": "claude-haiku-5-5"}
        away = Daemon(Config(json.loads(json.dumps(data)), path=paths.config_path()))
        shown = await away.status()
        # Claude plans, works the web pages and judges the steps; file search stays with the local model, and the
        # screenshot driver is not offered to a model that is not waiting on this computer's memory.
        self.assertEqual((shown["models"]["chat"], shown["models"]["driver"], shown["models"]["decision"],
                          shown["models"]["embedding"], shown["models"]["local_chat"]),
                         ("claude-haiku-5-5",) * 3 + ("coderunner-embeddinggemma-2", "coderunner-qwen3.5-9b"))
        self.assertEqual((shown["remote"]["name"], shown["model"]["model"], shown["model"]["vision"]),
                         ("Claude Haiku 5.5", "claude-haiku-5-5", True))
        self.assertNotIn("browser_task", away.svc.registry.tools)
        self.assertIs(away.svc.browser.decider.model.claude, away.svc.llm)
        self.assertIsNone(away.svc.guard.model)                               # the guard is off unless asked for
        self.assertTrue(away.svc.agent._system().startswith("You are the owner's assistant. You run on Anthropic's API;"))
        self.assertTrue(local.svc.agent._system().startswith("You are a private assistant running entirely on"))
        # One task sent to Claude from the local setup: its own agent, without the screenshot driver, told it runs
        # on Anthropic's API; the local agent and remote mode are untouched.
        with mock.patch.object(remote, "install_sdk"):
            once = await local.claude_agent()
        self.assertEqual((once.llm.model, once is local.svc.agent, local.svc.remote), ("claude-haiku-5-5", False, None))
        self.assertNotIn("browser_task", once.registry.tools)
        self.assertIn("browser_open", once.registry.tools)
        self.assertTrue(once._system().startswith("You are the owner's assistant. You run on Anthropic's API;"))
        self.assertIs(await local.claude_agent(), once)                      # built once
        self.assertIs(await away.claude_agent(), away.svc.agent)             # in remote mode, the one already there
        await once.llm.aclose()
        self.assertEqual((away.svc.llm.context, away.svc.llm.tool_output_chars), (150_000, 40_000))
        self.assertEqual((away.svc.browser.page_chars, page.LINK_CHARS, local.svc.browser.page_chars), (36_000, 200, page.PAGE_CHARS))
        self.assertNotIn(KEY, json.dumps(shown))
        from coderunner.models import manager
        started = mock.Mock()
        fake = mock.Mock(name="llama.cpp", base_url="http://127.0.0.1:9/v1", running=lambda: False, start=started)
        fake.name = "llama.cpp"
        with mock.patch.object(manager, "choose_runtime", return_value=fake):
            manager.ensure_running(away.config)                                 # the local model is not loaded
            started.assert_not_called()
            fake.running = lambda: True                                         # and one still loaded is let go
            manager.ensure_running(away.config)
            fake.stop.assert_called_once()
            fake.running = lambda: False
            manager.ensure_running(local.config)
            started.assert_called_once()
        for daemon in (local, away):
            await daemon.svc.llm.aclose()


class ServerDaemonTests(Home):
    async def test_on_the_owners_server_it_plans_judges_and_drives_and_tasks_take_its_slots(self):
        from coderunner.core import paths
        from coderunner.core.config import Config
        from coderunner.server.daemon import Daemon
        from coderunner.server.tasks import limit
        from fakes import FakeOpenAI
        from coderunner.browser import page
        self.addCleanup(setattr, page, "LINK_CHARS", page.LINK_CHARS)
        paths.ensure_dirs()
        with FakeOpenAI() as server:
            config = Config({"models": {"runtime": "openai", "base_url": "http://127.0.0.1:9/v1",
                                        "chat": "coderunner-qwen3.5-9b", "driver": {"model": "fara1.5-4b"}},
                             "sandbox": {"driver": "none"}, "browser": {"mode": "dedicated"}}, path=paths.config_path())
            remote.use_server(config, server.base_url, "bonsai2-27b", "local-key-1")
            daemon = Daemon(config)
            svc = daemon.svc
            self.assertEqual((svc.remote["provider"], svc.llm.model, svc.llm.base_url), ("openai", "bonsai2-27b", server.base_url))
            judge = svc.browser.decider.model
            self.assertEqual((judge.base_url, judge.model, judge.api), (server.base_url, "bonsai2-27b", "chat"))
            self.assertNotIn("browser_task", svc.registry.tools)          # it sees pages itself: no driver loaded here
            self.assertEqual((limit(svc), judge.slot), (1, 1))             # two slots: one task, and the judge's
            # The judge tells the planner's client before each judgement, which keeps the task's slot through it.
            self.assertIs(judge.planner, svc.llm)
            self.assertIsNotNone(svc.llm.keeper)
            self.assertTrue(svc.agent._system().startswith("You are the owner's assistant. You run on the owner's own "
                                                           "model server;"))
            # A task's steps go to its own slot, under the cap a served model has.
            from coderunner.agent.loop import RunContext

            def events(event):
                pass
            events.slot = 0
            server.script = [{"role": "assistant", "content": "Done."}]
            await svc.agent.run(RunContext(conversation="t-slot", events=events), "hi")
            self.assertEqual((server.requests[-1]["id_slot"], server.requests[-1]["max_tokens"]), (0, 4096))
            shown = await daemon.status()
            self.assertEqual((shown["models"]["chat"], shown["remote"]["host"]), ("bonsai2-27b", f"127.0.0.1:{server.port}"))
            # The server renamed its model in an upgrade: the judge and the settings follow the planner.
            svc.llm.renamed("bonsai2.5-27b")
            self.assertEqual((judge.model, svc.remote["model"], config.get("models.remote")["model"]),
                             ("bonsai2.5-27b",) * 3)
            await svc.llm.aclose()
            # Asked for, the judge runs on the driver's model on this Mac, without the driver working the pages,
            # and falls back to the server's judge when that model cannot answer.
            config.set("models.remote", {**config.get("models.remote"), "local_judge": True})
            daemon = Daemon(config)
            local = daemon.svc.browser.decider.model
            self.assertNotEqual(local.base_url, server.base_url)
            self.assertIsNotNone(local.prepare)
            self.assertEqual((local.fallback.base_url, local.fallback.slot), (server.base_url, 1))
            self.assertNotIn("browser_task", daemon.svc.registry.tools)
            await daemon.svc.llm.aclose()


class SlotServer:
    """A llama.cpp server as seen through its API: a request in one slot empties the other slots (what the owner's
    server did after each step judgement), and a slot restored from a saved file resumes."""

    def __init__(self, save_path: bool = True, resumes: bool = True):
        self.save_path, self.resumes = save_path, resumes
        self.held: dict[int, int] = {}          # slot -> tokens it holds
        self.files: dict[str, int] = {}
        self.log: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if "/slots/" in request.url.path:
            slot, action = int(request.url.path.rsplit("/", 1)[1]), request.url.params["action"]
            name = json.loads(request.content)["filename"]
            self.log.append(f"{action} {slot}")
            if not self.save_path:
                return httpx.Response(400, json={"error": {"message": "This server does not support slots action."}})
            if action == "save":
                self.files[name] = self.held.get(slot, 0)
                return httpx.Response(200, json={"id_slot": slot, "n_saved": self.files[name]})
            self.held[slot] = self.files[name] if self.resumes else 0
            return httpx.Response(200, json={"id_slot": slot, "n_restored": self.files[name]})
        body = json.loads(request.content)
        slot = body.get("id_slot")
        tokens = sum(len(str(m.get("content", ""))) for m in body["messages"])
        cached = min(self.held.get(slot, 0), tokens)
        self.held = {slot: tokens}              # the others are emptied
        self.log.append(f"chat {slot} cached={cached}")
        if str(body["messages"][0].get("content", "")).startswith("Evaluate the supplied decision task"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": tokens},
                                         "timings": {"prompt_n": tokens - cached, "cache_n": cached}})


class SlotKeepingTests(unittest.IsolatedAsyncioTestCase):
    def client(self, server: SlotServer):
        with mock.patch.object(httpx, "get", side_effect=httpx.ConnectError("no props")):
            llm = remote.server_client({"base_url": "http://server/v1", "model": "bonsai2.5-27b"})
        llm.slots = 2
        llm.keep_slots()
        llm._client = httpx.AsyncClient(base_url="http://server/v1", transport=httpx.MockTransport(server))
        judge = DecisionModel("http://server/v1", "bonsai2.5-27b", api="chat", transport=httpx.MockTransport(server))
        judge.slot, judge.planner = 1, llm
        return llm, ActionCheck(judge)

    async def test_a_tasks_slot_is_kept_through_a_judgement_and_put_back_before_its_next_step(self):
        server = SlotServer()
        llm, check = self.client(server)
        messages = [{"role": "system", "content": "s" * 100}, {"role": "user", "content": "go" * 50}]
        await llm.chat(messages, id_slot=0)
        messages += [{"role": "assistant", "content": "a"}, {"role": "tool", "content": "page" * 20}]
        self.assertFalse(await check.commits("Site: a.example\nAction: click button \"Search\""))
        self.assertFalse(await check.commits("Site: a.example\nAction: click button \"Next\""))
        reply = await llm.chat(messages, id_slot=0)
        # Kept once before the first judgement (the second finds it kept already), put back once before the step.
        self.assertEqual([line for line in server.log if not line.startswith("chat 1")],
                         ["chat 0 cached=0", "save 0", "restore 0", "chat 0 cached=200"])
        self.assertEqual(reply["restored"], 200)
        # A step without a judgement since is sent as it is.
        messages += [{"role": "assistant", "content": "b"}, {"role": "tool", "content": "more"}]
        server.log.clear()
        await llm.chat(messages, id_slot=0)
        self.assertEqual(server.log, [f"chat 0 cached={sum(len(m['content']) for m in messages[:-2])}"])

    async def test_an_edited_conversation_is_not_restored_and_a_server_that_cannot_save_is_left_alone(self):
        server = SlotServer()
        llm, check = self.client(server)
        messages = [{"role": "system", "content": "s"}, {"role": "tool", "content": "page" * 500}]
        await llm.chat(messages, id_slot=0)
        await check.commits("Site: a.example\nAction: click button \"Search\"")
        trimmed = [messages[0], {"role": "tool", "content": "[dropped]"}, {"role": "assistant", "content": "a"}]
        await llm.chat(trimmed, id_slot=0)
        self.assertNotIn("restore 0", server.log)       # read from the change anyway: no file read for nothing

        plain = SlotServer(save_path=False)             # no --slot-save-path
        llm, check = self.client(plain)
        await llm.chat(messages, id_slot=0)
        for n in range(3):
            self.assertFalse(await check.commits(f"Site: a.example\nAction: click button \"{n}\""))
            await llm.chat(messages + [{"role": "assistant", "content": str(n)}], id_slot=0)
        self.assertEqual(plain.log.count("save 0"), 1)  # asked once, then never again
        self.assertFalse(llm.keeper.on)

    async def test_keeping_stops_when_restored_slots_are_not_resumed_from(self):
        server = SlotServer(resumes=False)
        llm, check = self.client(server)
        messages = [{"role": "system", "content": "s" * 100}]
        await llm.chat(messages, id_slot=0)
        for n in range(5):
            await check.commits(f"Site: a.example\nAction: click button \"{n}\"")
            messages = messages + [{"role": "assistant", "content": str(n)}]
            await llm.chat(messages, id_slot=0)
        from coderunner.core.llm import KEEP_MISSES
        self.assertEqual(server.log.count("restore 0"), KEEP_MISSES)
        self.assertFalse(llm.keeper.on)

    async def test_a_judge_elsewhere_or_a_failing_keeper_never_changes_a_judgement(self):
        server = SlotServer()
        llm, check = self.client(server)
        await llm.chat([{"role": "user", "content": "go"}], id_slot=0)
        check.model.base_url = "http://driver.local/v1"     # the driver's own server judges: nothing to keep
        check.model._client = httpx.AsyncClient(transport=httpx.MockTransport(server))
        await check.commits("Site: a.example\nAction: click button \"Search\"")
        self.assertNotIn("save 0", server.log)
        check.model.base_url = "http://server/v1"

        async def broken(slot):
            raise RuntimeError("disk full")
        llm.side_request = broken
        self.assertFalse(await check.commits("Site: a.example\nAction: click button \"Next\""))
        # Not on the owner's server (no slots of its own), or turned off: no keeper at all.
        with mock.patch.object(httpx, "get", side_effect=httpx.ConnectError("no props")):
            self.assertIsNone(remote.server_client({"base_url": "http://server/v1", "model": "m"}).keeper)
        props = mock.Mock(json=lambda: {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 2})
        with mock.patch.object(httpx, "get", return_value=props):
            self.assertIsNotNone(remote.server_client({"base_url": "http://server/v1", "model": "m"}).keeper)
            self.assertIsNone(remote.server_client({"base_url": "http://server/v1", "model": "m",
                                                    "keep_slots": False}).keeper)


class SharedServerTests(unittest.IsolatedAsyncioTestCase):
    """Two installs on one model server over a tailnet each keep to their own slots (models.remote.slots)."""

    def test_a_slot_range_is_read_within_the_servers_slots(self):
        self.assertEqual(remote.slot_share("2-3", 4), (2, 2))
        self.assertEqual(remote.slot_share([0, 1], 4), (0, 2))
        self.assertEqual(remote.slot_share("3", 4), (3, 1))
        self.assertIsNone(remote.slot_share("", 4))
        self.assertIsNone(remote.slot_share("2-5", 4))             # past the server's slots: all of them, as unset
        self.assertIsNone(remote.slot_share("two", 4))

    async def test_requests_go_to_this_installs_slots(self):
        from coderunner.core.llm import LLM
        seen = []

        def answer(request):
            seen.append(json.loads(request.content).get("id_slot"))
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        llm = LLM("http://server/v1", "bonsai")
        llm._client = httpx.AsyncClient(base_url="http://server/v1", transport=httpx.MockTransport(answer))
        llm.slot_base = 2
        await llm.chat([{"role": "user", "content": "hi"}], id_slot=0)
        await llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(seen, [2, None])                          # a task's slot 0 is the server's slot 2
        await llm.aclose()


class RenameTests(unittest.IsolatedAsyncioTestCase):
    def test_the_newest_bonsai_of_the_same_size_replaces_a_name_the_server_dropped(self):
        self.assertIsNone(remote.successor(["bonsai2-27b", "bonsai2.5-27b"], "bonsai2-27b"))
        self.assertEqual(remote.successor(["Bonsai2.5-8B", "Bonsai2.5-27B-Q4_K_M"], "bonsai2-27b"), "Bonsai2.5-27B-Q4_K_M")
        self.assertEqual(remote.successor(["bonsai2-27b-old", "bonsai-2_5-27b"], "bonsai2-27b"), "bonsai-2_5-27b")
        self.assertEqual(remote.successor(["qwen4-32b"], "bonsai2-27b"), "qwen4-32b")
        self.assertIsNone(remote.successor(["qwen4-32b", "gemma5-27b"], "bonsai2-27b"))
        self.assertIsNone(remote.successor([], "bonsai2-27b"))

    async def test_a_step_refused_for_its_model_name_is_sent_again_under_the_name_served(self):
        sent = []

        def answer(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "Bonsai2.5-27B"}]})
            body = json.loads(request.content)
            sent.append(body["model"])
            if body["model"] != "Bonsai2.5-27B":
                return httpx.Response(404, json={"error": {"message": f"model '{body['model']}' not found"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        with mock.patch.object(httpx, "get", side_effect=httpx.ConnectError("no props")):
            llm = remote.server_client({"base_url": "http://server/v1", "model": "bonsai2-27b"})
        llm._client = httpx.AsyncClient(base_url="http://server/v1", transport=httpx.MockTransport(answer))
        told = []
        llm.renamed = told.append
        reply = await llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual((reply["content"], sent, told, llm.model),
                         ("hi", ["bonsai2-27b", "Bonsai2.5-27B"], ["Bonsai2.5-27B"], "Bonsai2.5-27B"))
        # Another failure is not taken for a renamed model.
        llm.model = "other"
        sent.clear()
        llm._client = httpx.AsyncClient(base_url="http://server/v1", transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": {"message": "context too long"}})))
        with self.assertRaises(LLMError):
            await llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(llm.model, "other")


class RouteTests(Home):
    async def test_the_owner_turns_remote_mode_on_and_off_and_is_told_it_is_not_private(self):
        from coderunner.agent import build_registry
        from coderunner.server.gateway import admin_app
        from test_gateway import ADMIN, FakeDaemon
        svc = Svc({"models": {"chat": "coderunner-qwen3.5-9b"}})
        svc.registry = build_registry(svc)
        daemon = FakeDaemon(svc)
        daemon.restart_soon = mock.Mock()
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            self.assertEqual((await client.get("/api/remote")).status_code, 401)
            first = (await client.get("/api/remote", headers=admin)).json()
            self.assertEqual((first["on"], first["has_key"], first["local"]), (False, False, "coderunner-qwen3.5-9b"))
            self.assertEqual(first["warning"], "Your requests, the pages it reads and the files it opens are sent to "
                                               "Anthropic. This is not private.")
            bad = await client.post("/api/remote", json={"action": "on", "key": "nope"}, headers=admin)
            self.assertEqual((bad.status_code, bad.json()["error"]), (400, "That is not an Anthropic API key (they start with sk-ant-)."))
            daemon.restart_soon.assert_not_called()
            with mock.patch.object(remote, "install_sdk"):
                on = await client.post("/api/remote", json={"action": "on", "key": KEY}, headers=admin)
            self.assertEqual((on.status_code, on.json()["on"], on.json()["name"], on.json()["restarting"]),
                             (200, True, "Claude Haiku 5.5", True))
            self.assertNotIn(KEY, on.text)
            daemon.restart_soon.assert_called_once()
            off = await client.post("/api/remote", json={"action": "off"}, headers=admin)
            self.assertEqual((off.json()["on"], off.json()["has_key"]), (False, True))
        audit = (svc.tmp / "audit.jsonl").read_text()
        self.assertIn('"remote.changed"', audit)
        self.assertNotIn(KEY, audit)

    async def test_the_owners_own_model_server_is_checked_then_used_with_its_key_kept_apart(self):
        from coderunner.agent import build_registry
        from coderunner.server.gateway import admin_app
        from fakes import FakeOpenAI
        from test_gateway import ADMIN, FakeDaemon
        svc = Svc({"models": {"chat": "coderunner-qwen3.5-9b"}})
        svc.registry = build_registry(svc)
        daemon = FakeDaemon(svc)
        daemon.restart_soon = mock.Mock()
        admin = {"Authorization": "Bearer " + ADMIN}
        with FakeOpenAI() as server:
            server.served = ["bonsai2-27b"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, daemon)),
                                         base_url="http://127.0.0.1:8224") as client:
                wrong = await client.post("/api/remote", headers=admin, json={
                    "action": "server", "base_url": server.base_url, "model": "other", "key": "local-key-1"})
                self.assertEqual((wrong.status_code, wrong.json()["error"]),
                                 (400, "That server serves bonsai2-27b, not other."))
                gone = await client.post("/api/remote", headers=admin, json={
                    "action": "server", "base_url": "http://127.0.0.1:9/v1", "model": "bonsai2-27b"})
                self.assertEqual(gone.json()["error"], "No model server answered at http://127.0.0.1:9/v1.")
                daemon.restart_soon.assert_not_called()
                on = await client.post("/api/remote", headers=admin, json={
                    "action": "server", "base_url": server.base_url + "/", "model": "bonsai2-27b", "key": "local-key-1"})
                shown = on.json()
                self.assertEqual((on.status_code, shown["on"], shown["provider"], shown["model"], shown["base_url"]),
                                 (200, True, "openai", "bonsai2-27b", server.base_url))
                self.assertEqual(shown["warning"], "Your requests, the pages it reads and the files it opens are sent "
                                                   f"to your model server at 127.0.0.1:{server.port}.")
                self.assertNotIn("local-key-1", on.text)
                # After an upgrade the server serves Bonsai2.5 only: the old name switches to it.
                server.served = ["bonsai2.5-27b"]
                again = await client.post("/api/remote", headers=admin, json={
                    "action": "server", "base_url": server.base_url, "model": "bonsai2-27b", "key": "local-key-1"})
                self.assertEqual((again.status_code, again.json()["model"]), (200, "bonsai2.5-27b"))
                server.served = ["bonsai2-27b"]
                remote.use_server(svc.config, server.base_url, "bonsai2-27b", "local-key-1")
                self.assertEqual(daemon.restart_soon.call_count, 2)
            self.assertEqual(server.authorizations[-1], "Bearer local-key-1")
            # The key is kept apart, readable by the owner only, and never in the config.
            self.assertEqual(remote.server_key(), "local-key-1")
            self.assertEqual(remote.server_key_path().stat().st_mode & 0o777, 0o600)
            self.assertNotIn("local-key-1", json.dumps(svc.config.data))
            # The client: the server's context shared by its two slots, no slot pinning or restarts of ours.
            llm = remote.client(svc.config)
            self.assertEqual((llm.base_url, llm.model, llm.remote, llm.provider, llm.context, llm.slots),
                             (server.base_url, "bonsai2-27b", True, "openai", 32768, 2))
            self.assertTrue(await llm.recover())            # a failed request is tried again once it is healthy
            await llm.aclose()
        self.assertEqual(remote.claude_model(svc.config), remote.DEFAULT)    # a Claude retry is not sent to bonsai

    async def test_one_task_is_sent_to_claude_only_with_a_kept_key_and_remote_mode_stays_off(self):
        from coderunner.agent import build_registry
        from coderunner.server.gateway import admin_app
        from test_gateway import ADMIN, FakeDaemon
        svc = Svc({"models": {"chat": "coderunner-qwen3.5-9b"}})
        svc.registry = build_registry(svc)
        daemon = FakeDaemon(svc)
        daemon.restart_soon = mock.Mock()
        daemon.tasks = mock.Mock(start=mock.Mock(return_value={"id": "t1"}))
        admin = {"Authorization": "Bearer " + ADMIN}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app(svc, daemon)),
                                     base_url="http://127.0.0.1:8224") as client:
            retry = {"message": "Find three hotels", "conversation": "app:abc", "remote": True}
            first = await client.post("/api/tasks", json=retry, headers=admin)
            self.assertEqual((first.status_code, first.json()["needs_key"]), (409, True))
            daemon.tasks.start.assert_not_called()
            kept = await client.post("/api/remote", json={"action": "key", "key": KEY}, headers=admin)
            self.assertEqual((kept.status_code, kept.json()["has_key"], kept.json()["on"]), (200, True, False))
            self.assertNotIn(KEY, kept.text)
            daemon.restart_soon.assert_not_called()                 # only kept: remote mode is as it was
            sent = await client.post("/api/tasks", json=retry, headers=admin)
            self.assertEqual(sent.status_code, 200)
            args, options = daemon.tasks.start.call_args
            self.assertEqual((args[:2], options["remote"]), (("app:abc", "Find three hotels"), "Claude Haiku 5.5"))
            await client.post("/api/tasks", json={**retry, "remote": False}, headers=admin)
            self.assertNotIn("remote", daemon.tasks.start.call_args[1])
        self.assertNotIn(KEY, (svc.tmp / "audit.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
