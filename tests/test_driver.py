"""The browser driver: Fara1.5's format, the step loop over the broker's checked actions, taking turns with
the assistant's model, and picking a driver per machine."""

import contextlib
import dataclasses
import json
import unittest
from datetime import datetime

from fakes import FakeOpenAI, Svc

from coderunner.agent import RunContext, build_registry, fara
from coderunner.agent.driver import Driver, Swap
from coderunner.core.config import Config
from coderunner.core.llm import LLM
from coderunner.models import hardware, manager


def step(action: dict, thought: str = "Next I will act.") -> dict:
    call = json.dumps({"name": "computer_use", "arguments": action})
    return {"role": "assistant", "content": f"{thought}\n<tool_call>\n{call}\n</tool_call>"}


class FakeBroker:
    """Records what the driver asks of the browser; every action succeeds unless told otherwise."""

    def __init__(self):
        self.calls = []
        self.kw = []             # the keyword arguments of each call
        self.fail = {}
        self.extra = {}          # more for an action's result (a tab it opened)
        self.viewports = []      # the page size each screenshot was taken at
        self.shot_tabs = []      # the tab each screenshot was taken in
        self.quiet = []          # whether page text was off for each screenshot

    async def screenshot(self, **kw):
        self.viewports.append(self.viewport)
        self.shot_tabs.append(kw.get("tab"))
        self.quiet.append(self.screens)
        return {"ok": True, "mime": "image/png", "data_b64": "iVBORw0KGgo=", "url": self.url,
                "viewport": [1440, 900]}

    viewport = None
    screens = False
    url = "https://shop.example/"

    @contextlib.asynccontextmanager
    async def screens_only(self):
        self.screens = True
        try:
            yield
        finally:
            self.screens = False

    @contextlib.asynccontextmanager
    async def fixed_viewport(self, width, height):
        self.viewport = (width, height)
        try:
            yield
        finally:
            self.viewport = None

    def __getattr__(self, name):
        async def action(*args, **kw):
            self.calls.append((name, args))
            self.kw.append(kw)
            if name in self.fail:
                return {"ok": False, "error": self.fail[name]}
            result = {"ok": True, "content": "Price: INR 9,995"} if name == "snapshot" else {"ok": True}
            return {**result, **self.extra.pop(name, {})}
        return action


class FormatTests(unittest.TestCase):
    def test_prompt_and_replies(self):
        prompt = fara.system_prompt()
        self.assertTrue(prompt.startswith("You are Fara, a computer use agent"))
        self.assertIn("The screen's resolution is 1000x1000.", prompt)
        self.assertIn('"name": "computer_use"', prompt)
        thoughts, args = fara.parse('I will click.\n<tool_call>\n{"name": "computer_use", "arguments": '
                                    '{"action": "left_click", "coordinate": [232, 411]}}\n</tool_call>')
        self.assertEqual((thoughts, args), ("I will click.", {"action": "left_click", "coordinate": [232, 411]}))
        self.assertIsNone(fara.parse("No call here.")[1])
        self.assertEqual(fara.parse("<tool_call>{'name': 'computer_use', 'arguments': {'action': 'wait'}}")[1],
                         {"action": "wait"})                          # a Python-style dict still reads
        self.assertEqual(fara.to_pixels([500, 1000], 1440, 900), (720.0, 900.0))
        self.assertEqual(fara.to_pixels([-5, 2000], 1440, 900), (0.0, 900.0))
        self.assertIsNone(fara.to_pixels("500,500", 1440, 900))


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_driver(self, script, steps=10, fail=None, format="fara1.5", extra=None, url=None):
        with FakeOpenAI() as fake:
            fake.script = list(script)
            svc = Svc()
            svc.browser = FakeBroker()
            svc.browser.fail = fail or {}
            svc.browser.extra = extra or {}
            if url:
                svc.browser.url = url
            driver = Driver(svc, LLM(fake.base_url, "fara"), steps=steps, format=format)
            ctx = RunContext(conversation="d1")
            report = await driver.run("Find the cheapest flight.", ctx)
            await driver.llm.aclose()
            self.viewports = svc.browser.viewports + [svc.browser.viewport]
            self.broker = svc.browser
            return report, svc.browser.calls, fake.requests, ctx

    async def test_steps_become_checked_broker_actions(self):
        report, calls, requests, ctx = await self.run_driver([
            step({"action": "left_click", "coordinate": [500, 500]}),
            step({"action": "type", "text": "Delhi"}),
            step({"action": "key", "keys": ["Return"]}),
            step({"action": "key", "keys": ["Control", "a"]}),
            step({"action": "scroll", "pixels": -500}),
            step({"action": "web_search", "query": "flights to Delhi"}),
            step({"action": "read_page_answer_question", "question": "What is the price?"}),
            {"role": "assistant", "content": "INR 9,995"},             # the text-only answer to that question
            step({"action": "pause_and_memorize_fact", "fact": "SpiceJet is cheapest"}),
            step({"action": "terminate", "answer": "SpiceJet at INR 9,995"}),
        ])
        self.assertTrue(report.startswith("SpiceJet at INR 9,995"))
        self.assertIn("Noted on the way: SpiceJet is cheapest", report)
        # Every screenshot at the page size of its training, and the window's own size again after.
        self.assertEqual(set(self.viewports[:-1]), {fara.VIEWPORT})
        self.assertIsNone(self.viewports[-1])
        kw = self.broker.kw
        self.assertEqual(calls[0], ("click_at", (720.0, 450.0)))        # 1000-space point on a 1440x900 page
        self.assertEqual(kw[0]["clicks"], 1)
        self.assertEqual((calls[1], kw[1]["keystrokes"]), (("type_focused", ("Delhi",)), True))   # key by key
        self.assertEqual(calls[2], ("press", ("Enter",)))
        self.assertEqual(calls[3], ("press_keys", (["Control", "a"],)))  # combinations, as Microsoft's harness
        self.assertEqual(calls[4], ("wheel", (720.0, 450.0, 0, 450.0)))  # the wheel where the mouse is, 0.9 x 500
        self.assertEqual(calls[5][0], "navigate")
        self.assertIn("q=flights+to+Delhi&FORM=QBLH", calls[5][1][0])
        self.assertEqual(calls[6][0], "snapshot")
        self.assertTrue(ctx.tainted)                                    # web pages were read
        # After an action that worked the model hears only the address and the usual request, as in training.
        self.assertEqual(requests[1]["messages"][-1]["content"][1]["text"],
                         f"Current URL: https://shop.example/\n{fara.NEXT}")
        steps = [r["messages"][-1]["content"] for r in requests if isinstance(r["messages"][-1]["content"], list)]
        self.assertTrue(any(part.get("text", "").startswith("Current URL: https://shop.example/\nI read the page to "
                                                            "answer: What is the price?\nAnswer: INR 9,995")
                            for content in steps for part in content))    # the step after read_page_answer_question
        extract = next(r for r in requests if r["messages"][0]["content"].startswith("You are given the markdown"))
        self.assertIn("Question: What is the price?", extract["messages"][1]["content"])
        self.assertTrue(all(self.broker.quiet))                         # no page text for a screenshot model
        # The model's own format, and at most three screenshots in any request.
        self.assertEqual(requests[0]["messages"][0]["content"], fara.system_prompt())
        task = requests[0]["messages"][1]["content"][1]["text"]
        self.assertIn(f"Today is {datetime.now().strftime('%A')} ", task)       # the date it cannot know
        for request in requests:
            images = sum(1 for m in request["messages"] if isinstance(m["content"], list)
                         for part in m["content"] if part.get("type") == "image_url")
            self.assertLessEqual(images, fara.KEEP_SCREENSHOTS)

    async def test_fara_7b_speaks_its_own_format(self):
        report, calls, requests, _ = await self.run_driver([
            step({"action": "left_click", "coordinate": [714, 448]}),
            step({"action": "type", "text": "Delhi", "coordinate": [100, 50], "delete_existing_text": True}),
            step({"action": "scroll", "pixels": -1}),
            step({"action": "terminate", "status": "success"}, thought="The cheapest is SpiceJet at INR 9,995."),
        ], format="fara7b")
        system = requests[0]["messages"][0]["content"]
        self.assertTrue(system.startswith("You are a helpful assistant.\n\nYou are a web automation agent"))
        self.assertIn("The screen's resolution is 1428x896.", system)          # 1440 x 900 after Qwen2.5-VL's resize
        self.assertEqual(calls[0], ("click_at", (720.0, 450.0)))                 # resized pixels, not a 1000 grid
        self.assertEqual(calls[1][0], "click_at")                                # it types at a point: click first
        self.assertEqual(calls[2][0], "type_focused")
        self.assertEqual((calls[3][0], calls[3][1][3]), ("wheel", 400))         # 400 px down, as its harness does
        self.assertEqual(report, "The cheapest is SpiceJet at INR 9,995.")      # the answer is its text

    async def test_refusals_reach_the_driver_and_questions_reach_the_planner(self):
        report, calls, requests, _ = await self.run_driver([
            step({"action": "left_click", "coordinate": [100, 100]}),
            step({"action": "ask_user_question", "question": "Which card should I use?"}),
        ], fail={"click_at": "Not permitted: Click button \"Pay now\" (not approved by the owner)"})
        self.assertIn("needs an answer before it can go on: Which card should I use?", report)
        last = requests[-1]["messages"][-1]["content"]
        self.assertIn("ERROR: Not permitted", json.dumps(last))

    def test_old_screenshots_go_together_so_most_steps_only_add_one(self):
        messages = [{"role": "system", "content": "s"}]
        shown = []
        for n in range(8):
            messages.append({"role": "user", "content": [{"type": "image_url"}, {"type": "text", "text": f"step {n}"}]})
            Driver._keep_recent_screenshots(messages)
            shown.append(sum(isinstance(m["content"], list) for m in messages))
            messages.append({"role": "assistant", "content": "next"})
        self.assertEqual(shown, [1, 2, 3, 1, 2, 3, 1, 2])            # never more than three, as in training
        self.assertEqual(messages[1]["content"], "step 0")            # the task keeps its text
        texts = [m for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
        self.assertEqual(len(texts), 1)                               # other dropped steps go, as in training

    async def test_addresses_words_and_popups_as_microsofts_harness_has_them(self):
        long = "https://shop.example/" + "p" * 120 + "?session=1"
        report, calls, requests, _ = await self.run_driver([
            step({"action": "visit_url", "url": "shop.example/deals"}),
            step({"action": "visit_url", "url": "cheap flights to Delhi"}),
            step({"action": "left_click", "coordinate": [500, 500]}),
            step({"action": "terminate", "answer": "done"}),
        ], extra={"click_at": {"new_tabs": ["tab-2"]}}, url=long)
        self.assertEqual(calls[0], ("navigate", ("https://shop.example/deals",)))
        self.assertIn("bing.com/search?q=cheap+flights+to+Delhi&FORM=QBLH", calls[1][1][0])
        said = requests[1]["messages"][-1]["content"][1]["text"]
        self.assertEqual(said.split("\n")[0], "Current URL: " + long.split("?")[0][:100] + " ...")
        self.assertEqual(self.broker.shot_tabs[-1], "tab-2")          # the popup the click opened is followed

    async def test_a_driver_going_round_in_circles_is_told_and_then_stopped(self):
        # A Google Maps run clicked the same button 18 times in a row. Points a little apart are the same step.
        script = [step({"action": "left_click", "coordinate": [610 + i % 2, 300]}) for i in range(7)] + [
            step({"action": "terminate", "status": "success", "answer": "2 h 17 min"})]
        report, calls, requests, _ = await self.run_driver(script, steps=10)
        self.assertEqual(report, "2 h 17 min")
        self.assertEqual(sum(1 for name, _ in calls if name == "click_at"), 5)    # the sixth and seventh not taken
        said = [json.dumps(r["messages"][-1]["content"]) for r in requests[1:]]
        self.assertNotIn("same step", said[1])                                   # after the second click
        self.assertIn("same step 3 times in a row", said[2])
        self.assertIn("That step was not taken", said[5])

    async def test_a_driver_that_does_not_finish_says_so(self):
        report, _, _, _ = await self.run_driver([step({"action": "wait", "time": 0})] * 3, steps=3)
        self.assertIn("used all its steps", report)


class FakeRuntime:
    def __init__(self, name, log, base_url="http://127.0.0.1:8083/v1"):
        self.name, self.log, self.base_url = name, log, base_url

    def start(self):
        self.log.append(f"start {self.name}")

    def stop(self):
        self.log.append(f"stop {self.name}")

    def unload(self, model):
        self.log.append(f"unload {model}")


class SwapTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_models_take_turns_and_the_judge_follows(self):
        log = []
        svc = Svc()
        svc.llm = LLM("http://127.0.0.1:8082/v1", "coderunner-bonsai-2-27b")
        judge = type("Judge", (), {"base_url": "http://127.0.0.1:8082/v1", "model": "coderunner-bonsai-2-27b",
                                   "api": "chat"})()
        svc.browser = type("Browser", (), {"decider": type("Check", (), {"model": judge})()})()
        svc.driver = type("D", (), {"llm": LLM("http://127.0.0.1:8083/v1", "coderunner-fara1.5-4b")})()
        async with Swap(svc, FakeRuntime("prism", log), FakeRuntime("driver", log)):
            self.assertEqual(log, ["stop prism", "start driver"])
            self.assertEqual((judge.base_url, judge.model), ("http://127.0.0.1:8083/v1", "coderunner-fara1.5-4b"))
        self.assertEqual(log, ["stop prism", "start driver", "stop driver", "start prism"])
        self.assertEqual((judge.base_url, judge.model), ("http://127.0.0.1:8082/v1", "coderunner-bonsai-2-27b"))
        log.clear()
        async with Swap(svc, FakeRuntime("ollama", log), FakeRuntime("driver", log)):
            pass                            # Ollama keeps serving embeddings: only the chat model is unloaded
        self.assertEqual(log, ["unload coderunner-bonsai-2-27b", "start driver", "stop driver", "start ollama"])
        await svc.llm.aclose()
        await svc.driver.llm.aclose()


class PickTests(unittest.TestCase):
    def tier(self, ram, chat, runtime):
        limit = ram * (0.75 if ram >= 32 else 2 / 3)
        hw = hardware.Hardware(os="macos", arch="arm64", ram_gb=ram, gpu="apple", gpu_limit_gb=limit,
                               budget_gb=hardware.apple_budget(ram, limit, 1.0), bandwidth_gbs=150.0)
        config = Config({"models": {"chat": f"coderunner-{chat}", "runtime": runtime, "context": 32768,
                                    "embedding": "embeddinggemma-2"}})
        used = manager._chat_need(config, hw) + manager._embedding_need(config, hw)
        return manager.driver_pick(hw, config, used)

    def test_small_machines_take_turns_and_large_ones_keep_both(self):
        pick = self.tier(18, "bonsai-2-27b", "prism")
        self.assertEqual((pick["id"], pick["exclusive"]), ("fara1.5-4b", True))      # measured: both OOM here
        pick = self.tier(64, "qwen3.8-27b", "llama.cpp")
        self.assertEqual((pick["id"], pick["exclusive"]), ("fara1.5-9b", False))
        self.assertTrue(dataclasses.is_dataclass(hardware.Hardware))


class ToolTests(unittest.TestCase):
    def test_browser_task_exists_only_with_a_driver(self):
        svc = Svc()
        svc.browser = FakeBroker()
        self.assertNotIn("browser_task", build_registry(svc).tools)
        svc.driver = object()
        self.assertIn("browser_task", build_registry(svc).tools)
        self.assertTrue(build_registry(svc).tools["browser_task"].untrusted)


if __name__ == "__main__":
    unittest.main()
