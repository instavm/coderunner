"""Short names for the app's threads: written by the model once the first task has ended, else made from the request."""

import asyncio
import unittest

from coderunner.core.llm import LLM, LLMError
from coderunner.server import tasks as tasks_module
from coderunner.server import titles
from fakes import FakeOpenAI, Svc
from test_gateway import FakeDaemon

MAURITIUS = "Open https://en.wikipedia.org/wiki/Mauritius and tell me its capital."
HEADPHONES = "Which are the three best-reviewed noise-cancelling headphones under 300 dollars, and where can I buy them?"


class FakeLLM:
    """The model as the naming sees it: ask() answers `answer`, or raises it when it is an exception."""

    def __init__(self, answer="Mauritius capital"):
        self.answer = answer
        self.asked: list[dict] = []

    async def ask(self, system, turns, max_tokens=16, **options):
        self.asked.append({"system": system, "turns": turns, "max_tokens": max_tokens, **options})
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


class Heuristic(unittest.TestCase):
    def test_names_made_from_the_request(self):
        cases = {
            MAURITIUS: "Mauritius and tell me its capital",
            HEADPHONES: "Which are the three best-reviewed",
            "Use Python to plot the sales in sales.csv by month": "Plot the sales in sales.csv by month",
            "Please can you check the weather in Bengaluru tomorrow?": "The weather in Bengaluru tomorrow",
            "go to github.com/instavm/coderunner/issues/45 and summarise it": "Coderunner and summarise it",
            "https://www.example.com/": "Example.com",
            "Bengaluru weather tomorrow": "Bengaluru weather tomorrow",
            "": "",
        }
        for prompt, name in cases.items():
            self.assertEqual(titles.heuristic(prompt), name, prompt)

    def test_kept_in_the_requests_language(self):
        self.assertEqual(titles.heuristic("कृपया कल बेंगलुरु का मौसम बताइए।"), "कल बेंगलुरु का मौसम बताइए")
        self.assertEqual(titles.heuristic("Bitte öffne https://www.spiegel.de und fasse die Schlagzeilen zusammen."),
                         "Spiegel.de und fasse die Schlagzeilen")
        self.assertEqual(titles.heuristic("Kannst du mir das Wetter in Berlin für morgen sagen?"),
                         "Das Wetter in Berlin für morgen sagen")

    def test_cut_on_a_word_without_a_url_or_a_closing_full_stop(self):
        for prompt in (MAURITIUS, HEADPHONES, "Open https://shop.example.com/p/123456 and buy it.", "x " * 80,
                       "A" * 90):
            name = titles.heuristic(prompt)
            self.assertLessEqual(len(name), titles.MAX_CHARS)
            self.assertNotIn("http", name)
            self.assertFalse(name.endswith("."))


class Clean(unittest.TestCase):
    def test_the_models_answer_is_a_name_or_nothing(self):
        self.assertEqual(titles.clean("Mauritius capital."), "Mauritius capital")
        self.assertEqual(titles.clean('"Title: Noise-cancelling headphones"\nmore'), "Noise-cancelling headphones")
        self.assertEqual(titles.clean("बेंगलुरु मौसम कल"), "बेंगलुरु मौसम कल")
        for junk in ("", "   ", "x" * 61, "https://en.wikipedia.org/wiki/Mauritius", "See wikipedia.org",
                     "This is a whole sentence and not a name for a thread at all"):
            self.assertEqual(titles.clean(junk), "", junk)


class Naming(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc()
        self.svc.llm = FakeLLM()
        self.daemon = FakeDaemon(self.svc)
        self.gates: dict[str, asyncio.Event] = {}

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            await self.gates.setdefault(chat_id, asyncio.Event()).wait()
            return f"done {chat_id}"
        self.daemon.handle_message = handle
        self.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = tasks_module.Tasks(self.daemon, self.bus)

    async def finish(self, chat_id):
        self.gates.setdefault(chat_id, asyncio.Event()).set()
        for _ in range(20):
            await asyncio.sleep(0.01)

    def listed(self):
        return {c["conversation"]: c for c in self.tasks.conversations()}

    def named(self):
        return [e for e in self.bus.since(0) if e["type"] == "title"]

    async def test_the_models_name_once_the_first_task_has_ended(self):
        long = MAURITIUS + " " + "and more " * 60
        self.tasks.start("app:one", long, "owner")
        await asyncio.sleep(0.01)
        self.assertEqual(self.svc.llm.asked, [])                # not before: the start is not slowed
        self.assertEqual(self.listed()["app:one"]["title"], "Mauritius and tell me its capital")
        await self.finish("one")
        asked = self.svc.llm.asked
        self.assertEqual(len(asked), 1)
        self.assertEqual((asked[0]["system"], asked[0]["max_tokens"]), (titles.SYSTEM, 16))
        self.assertLessEqual(len(asked[0]["turns"][0]["content"]), titles.PROMPT_CHARS)
        self.assertNotIn("slot", asked[0])                      # no slot of a task's to keep clear here
        row = self.listed()["app:one"]
        self.assertEqual((row["title"], row["prompt"]), ("Mauritius capital", long))
        self.assertEqual([(e["conversation"], e["title"]) for e in self.named()], [("app:one", "Mauritius capital")])
        # A second message in the same thread: named once only.
        self.tasks.start("app:one", "And its population?", "owner")
        self.gates["one"] = asyncio.Event()
        await self.finish("one")
        self.assertEqual(len(self.svc.llm.asked), 1)
        self.assertEqual(self.listed()["app:one"]["title"], "Mauritius capital")

    async def test_junk_from_the_model_leaves_the_name_made_from_the_request(self):
        for answer in ("", "https://en.wikipedia.org/wiki/Mauritius", "y" * 70):
            self.svc.llm.answer = answer
            conversation = f"junk-{len(answer)}"
            self.tasks.start(f"app:{conversation}", MAURITIUS, "owner")
            await self.finish(conversation)
            self.assertEqual(self.listed()[f"app:{conversation}"]["title"], "Mauritius and tell me its capital")
        self.assertEqual(len(self.svc.llm.asked), 3)
        self.assertEqual(self.named(), [])
        sources = {r["source"] for r in self.svc.store.query("SELECT source FROM conversation_titles")}
        self.assertEqual(sources, {"heuristic"})

    async def test_a_model_that_is_away_or_too_slow_leaves_the_name_made_from_the_request(self):
        self.svc.llm.answer = LLMError("Model runtime at http://127.0.0.1:1 is unreachable")
        self.tasks.start("app:away", HEADPHONES, "owner")
        await self.finish("away")
        self.assertEqual(self.listed()["app:away"]["title"], "Which are the three best-reviewed")
        self.assertEqual(self.named(), [])
        self.svc.llm.answer = asyncio.TimeoutError()
        self.tasks.start("app:slow", HEADPHONES, "owner")
        await self.finish("slow")
        self.assertEqual(self.listed()["app:slow"]["title"], "Which are the three best-reviewed")
        self.assertEqual(len(self.svc.llm.asked), 2)            # asked once each, not again later
        self.tasks.start("app:away", "again", "owner")
        self.gates["away"] = asyncio.Event()
        await self.finish("away")
        self.assertEqual(len(self.svc.llm.asked), 2)

    async def test_not_while_another_task_has_the_model_then_once_it_is_free(self):
        self.tasks.start("app:one", MAURITIUS, "owner")
        self.tasks.start("app:two", HEADPHONES, "owner")
        await self.finish("one")
        self.assertEqual(self.svc.llm.asked, [])                # two still running: the model is not asked
        self.assertEqual(self.listed()["app:one"]["title"], "Mauritius and tell me its capital")
        self.svc.llm.answer = "Noise-cancelling headphones"
        await self.finish("two")
        self.assertEqual([a["turns"][0]["content"][:5] for a in self.svc.llm.asked], ["Which", "Open "])
        listed = self.listed()
        self.assertEqual((listed["app:one"]["title"], listed["app:two"]["title"]),
                         ("Noise-cancelling headphones", "Noise-cancelling headphones"))
        # A local model taking a turn for a message from elsewhere is busy too.
        self.svc.turns = tasks_module.Turns()
        self.svc.turns._busy = True
        self.tasks.start("app:three", "Bengaluru weather tomorrow?", "owner")
        await self.finish("three")
        self.assertEqual(len(self.svc.llm.asked), 2)

    async def test_on_the_owners_server_in_the_last_slot_and_on_anthropic_beside_running_tasks(self):
        self.svc.llm.slots = 2
        self.tasks.start("app:one", MAURITIUS, "owner")
        await self.finish("one")
        self.assertEqual(self.svc.llm.asked[-1]["slot"], 1)     # the judge's slot, not the task's
        claude = FakeLLM("Noise-cancelling headphones")
        claude.concurrent = True
        self.svc.llm = claude
        self.svc.remote = {"provider": "anthropic"}
        self.tasks.start("app:two", HEADPHONES, "owner")
        self.tasks.start("app:three", "Bengaluru weather tomorrow?", "owner")
        await self.finish("two")
        self.assertEqual(len(claude.asked), 1)                  # three still runs: Anthropic's API takes it anyway
        self.assertNotIn("slot", claude.asked[0])
        await self.finish("three")

    async def test_the_checks_threads_are_named_from_the_request_and_recent_old_ones_by_the_model_once(self):
        self.svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                               "VALUES ('old', 'app:old', ?, 'done', '', 1, 2)", (HEADPHONES,))
        self.tasks.start("app:check-answer-1", MAURITIUS, "check")
        await self.finish("check-answer-1")
        self.assertEqual(self.svc.llm.asked, [])
        listed = self.listed()
        self.assertEqual(listed["app:check-answer-1"]["title"], "Mauritius and tell me its capital")
        # A thread from before names: the request's words at first, then the model's name, asked once a run.
        self.assertEqual(listed["app:old"]["title"], "Which are the three best-reviewed")
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.svc.llm.asked), 1)
        self.assertIn("best-reviewed", self.svc.llm.asked[0]["turns"][-1]["content"])
        self.assertNotEqual(self.listed()["app:old"]["title"], "Which are the three best-reviewed")
        self.listed()
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.svc.llm.asked), 1)            # the check's thread is never sent


class Ask(unittest.IsolatedAsyncioTestCase):
    async def test_one_short_request_with_reasoning_off_kept_apart_from_the_tasks_steps(self):
        with FakeOpenAI() as fake:
            fake.title = lambda prompt: "Mauritius capital."
            llm = LLM(fake.base_url, "test")
            recovered = []

            async def recover():
                recovered.append(1)
                return True
            llm.recover = recover
            self.assertEqual(await titles.from_model(llm, MAURITIUS, slot=2), "Mauritius capital")
            body = fake.titled[-1]
            self.assertEqual((body["max_tokens"], body["id_slot"], body["chat_template_kwargs"]),
                             (16, 2, {"enable_thinking": False}))
            self.assertEqual(fake.requests, [])
            await llm.aclose()
        down = LLM("http://127.0.0.1:9/v1", "test")
        down.recover = recover
        with self.assertRaises(LLMError):
            await down.ask(titles.SYSTEM, [{"role": "user", "content": "x"}])
        self.assertEqual(recovered, [])                         # a side question never restarts the runtime
        await down.aclose()


if __name__ == "__main__":
    unittest.main()
