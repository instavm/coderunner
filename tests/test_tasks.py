import contextlib
import datetime
import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock

from starlette.testclient import TestClient

from coderunner.models import sites, tasks

TODAY = datetime.date(2026, 10, 13)


class Flights(unittest.TestCase):
    def test_answer_is_fixed_by_route_and_date_and_hidden_from_the_default_order(self):
        found = sites.flights("BLR", "DEL", "2026-11-20")
        self.assertEqual(found, sites.flights("BLR", "DEL", "2026-11-20"))
        self.assertEqual(len({f["price"] for f in found}), len(found))
        cheapest = {f["number"] for f in sites.cheapest("BLR", "DEL", "2026-11-20")}
        self.assertFalse(cheapest & {f["number"] for f in found[:4]})
        self.assertNotEqual(found, sites.flights("BLR", "DEL", "2026-11-27"))

    def test_target_day_is_a_friday_in_about_five_weeks(self):
        day = tasks.target_day(TODAY)
        self.assertEqual(day, datetime.date(2026, 11, 20))
        for offset in range(7):
            day = tasks.target_day(TODAY + datetime.timedelta(days=offset))
            self.assertEqual(day.weekday(), 4)
            self.assertTrue(35 <= (day - TODAY - datetime.timedelta(days=offset)).days < 42)

    def test_indian_grouping(self):
        self.assertEqual(sites.rupees(5774), "₹5,774")
        self.assertEqual(sites.rupees(150000), "₹1,50,000")
        self.assertEqual(sites.rupees(999), "₹999")

    def test_results_page_prices_follow_the_search(self):
        site = sites.Sites(TODAY)
        client = TestClient(site.app)
        query = "from=BLR&to=DEL&depart=2026-11-20&adults=1&cabin=economy"
        cheapest = sites.cheapest("BLR", "DEL", "2026-11-20")
        page = client.get(f"/flights/results?trip=oneway&{query}&sort=cheapest").text
        prices = re.findall(r"class='price'>(₹[\d,]+)<", page)
        self.assertEqual(prices[:3], [sites.rupees(f["price"]) for f in cheapest])
        self.assertIn("one way per person", page)
        round_trip = client.get(f"/flights/results?trip=round&return=2026-11-24&{query}").text
        self.assertNotIn(sites.rupees(cheapest[0]["price"]) + "<", round_trip)
        self.assertIn("couldn't run that search", client.get(f"/flights/results?trip=round&{query}").text)
        self.assertEqual(site.visits[0], f"/flights/results?trip=oneway&{query}&sort=cheapest")

    def test_home_page_and_library(self):
        client = TestClient(sites.Sites(TODAY).app)
        home = client.get("/flights/").text
        self.assertIn('"2026-10-13"', home)
        self.assertIn("Kempegowda", home)
        self.assertIn("10:00–14:00", client.get("/library/visit").text)
        self.assertIn("₹15", client.get("/library/borrowing").text)
        self.assertIn("DW-412 · Series 4", client.get("/kelvin/search?q=dw412").text)
        self.assertIn("every 3 months", client.get("/kelvin/guide/dw-412").text)


class Checks(unittest.TestCase):
    def setUp(self):
        self.site = sites.Sites(TODAY)
        self.cheapest = sites.cheapest("BLR", "DEL", "2026-11-20")

    def test_flights(self):
        a, b, c = self.cheapest
        answer = (f"1. {a['airline']} 13:45, ₹{a['price']:,}\n2. {b['airline']} — INR {b['price']}\n"
                  f"3. {c['airline']} Rs. {c['price'] // 1000} {c['price'] % 1000:03d}")
        self.assertEqual(tasks.check_flights(answer, self.site)[0], True)
        self.assertEqual(tasks.check_flights(answer.split("\n3.")[0], self.site)[0], False)
        self.site.visits.append("/flights/checkout?flight=AI2327")       # who's flying: not booked yet
        self.assertTrue(tasks.check_flights(answer, self.site)[0])
        self.site.visits.append("/flights/payment")
        self.assertEqual(tasks.check_flights(answer, self.site), (False, "went on to payment"))

    def test_evening(self):
        want = sites.evening("BLR", "DEL", "2026-11-20")
        self.assertFalse(want["stops"])
        self.assertGreater(want["depart"], 18 * 60)
        self.assertNotIn(want["number"], [f["number"] for f in self.cheapest])
        answer = f"{want['airline']} {want['number'].replace(' ', '-')} at {sites.clock(want['depart'])}, {sites.rupees(want['price'])}"
        self.assertTrue(tasks.check_evening(answer, self.site)[0])
        self.assertFalse(tasks.check_evening(answer.replace(want["number"].split()[1], "0000"), self.site)[0])

    def test_appliance(self):
        self.assertFalse(tasks.check_appliance("Descale every 3 months.", self.site)[0])     # guessed, not looked up
        self.site.visits.append("/kelvin/guide/DW-412")
        self.assertTrue(tasks.check_appliance("Descale every three months (two in hard water).", self.site)[0])
        self.assertFalse(tasks.check_appliance("Descale every 6 months.", self.site)[0])

    def test_library(self):
        self.assertTrue(tasks.check_library("It closes at 2 pm on Sundays; DVDs cost ₹15 a day.", self.site)[0])
        self.assertTrue(tasks.check_library("Sunday: 10:00–14:00. Late fee: Rs 15/day", self.site)[0])
        self.assertFalse(tasks.check_library("Sunday: 10:00–14:00. Late fee: ₹5 a day", self.site)[0])
        self.assertFalse(tasks.check_library("Closes 15:00; fee 15", self.site)[0])

    def test_documents(self):
        self.assertTrue(tasks.check_documents("Covered until 14 March 2030. Call 1800-555-0199.", self.site)[0])
        self.assertTrue(tasks.check_documents("Until 2030-03-14; Kelvin Care 1800 555 0199", self.site)[0])
        self.assertFalse(tasks.check_documents("Until 14 March 2027; call 1800 555 0199", self.site)[0])
        self.assertFalse(tasks.check_documents("Until March 2030; call 1800 555 0177", self.site)[0])


class ShoppingAndSupport(unittest.TestCase):
    def setUp(self):
        self.site = sites.Sites(TODAY)
        self.client = TestClient(self.site.app)
        self.message = f"Why is my {self.site.points:,}-point AirMiles transfer still pending?"

    def test_compare_requires_all_three_local_inventories_and_delivered_price(self):
        offers = sites.drink_offers(TODAY)
        total, winner = min((p + fee, name) for name, (p, fee, available) in offers.items() if available)
        answer = f"{winner} — ₹{total}"
        for i in range(len(sites.STORES)):
            url = f"/grocers/{i}"
            self.assertNotIn("Volt", self.client.get(url, params={"pin": sites.DELIVERY_PIN}).text)
            self.assertNotIn("Volt", self.client.post(url, data={"pin": sites.DELIVERY_PIN}).text)
            self.assertNotIn("Volt", self.client.post(url, data={"pin": "999999", "action": "stock"}).text)
        self.assertFalse(self.site.stock_checked)
        self.assertFalse(tasks.check_groceries(answer, self.site)[0])
        for i, name in enumerate(sites.STORES):
            page = self.client.post(f"/grocers/{i}", data={"pin": sites.DELIVERY_PIN, "action": "stock"}).text
            item, fee, available = offers[name]
            self.assertIn(f"350 ml — ₹{item} — {'in stock' if available else 'sold out'}", page)
            self.assertIn(f"fee per order: ₹{fee}", page)
            self.assertIn("Volt Zero Sugar", page)
            self.assertIn("250 ml", page)
            self.assertEqual(tasks.check_groceries(answer, self.site)[0], i == len(sites.STORES) - 1)
        for wrong in (f"{winner} ₹{offers[winner][0]}", f"{winner} ₹{total}1",
                      f"{winner} ₹{total}.50", f"{winner} ₹49", f"{winner} ₹54"):
            self.assertFalse(tasks.check_groceries(wrong, self.site)[0], wrong)
        for name, (item, fee, available) in offers.items():
            if name != winner:
                self.assertFalse(tasks.check_groceries(f"{name} ₹{item + fee}", self.site)[0])
        for formatted in (answer + ".", f"{winner}, {total} rupees", f"{winner} Rs. {total}.00"):
            self.assertTrue(tasks.check_groceries(formatted, self.site)[0], formatted)
        self.site.reset()
        self.assertFalse(self.site.stock_checked)

    def test_case_requires_one_correct_delivery_and_matching_receipt(self):
        self.client.get("/rewards/activity")
        self.client.get("/rewards/chat")
        receipt = self.site.ticket
        self.assertFalse(tasks.check_rewards_chat(receipt, self.site)[0])
        self.assertEqual(self.client.post("/rewards/chat", data={"message": "  "}).status_code, 400)
        sent = self.client.post("/rewards/chat", data={"message": self.message})
        self.assertIn(receipt, sent.text)
        self.assertFalse(tasks.check_rewards_chat("I wrote a complaint.", self.site)[0])
        self.assertFalse(tasks.check_rewards_chat(receipt + "1", self.site)[0])
        self.assertTrue(tasks.check_rewards_chat(receipt, self.site)[0])
        self.client.post("/rewards/chat", data={"message": self.message})
        self.assertFalse(tasks.check_rewards_chat(receipt, self.site)[0])

    def test_draft_and_send_have_opposite_side_effect_requirements(self):
        self.assertFalse(tasks.check_rewards_draft(self.message, self.site)[0])
        self.client.get("/rewards/activity")
        self.assertTrue(tasks.check_rewards_draft(self.message, self.site)[0])
        wrong = self.message.replace(f"{self.site.points:,}", "800")
        self.assertFalse(tasks.check_rewards_draft(wrong, self.site)[0])
        self.client.post("/rewards/chat", data={"message": wrong})
        self.assertFalse(tasks.check_rewards_draft(self.message, self.site)[0])
        self.assertFalse(tasks.check_rewards_chat(self.site.ticket, self.site)[0])
        self.site.reset()
        self.assertFalse(self.site.visits)
        self.assertFalse(self.site.complaints)

    def test_data_varies_with_the_date_and_has_both_price_traps(self):
        winners = set()
        for day in range(12):
            offers = sites.drink_offers(TODAY + datetime.timedelta(days=day))
            available = [k for k, (_, _, stock) in offers.items() if stock]
            winner = min(available, key=lambda k: sum(offers[k][:2]))
            winners.add(winner)
            self.assertNotEqual(winner, min(available, key=lambda k: offers[k][0]))
            self.assertFalse(offers[min(offers, key=lambda k: offers[k][0])][2])
        self.assertEqual(winners, set(sites.STORES))


@unittest.skipIf(os.environ.get("CODERUNNER_SKIP_BROWSER") == "1", "CODERUNNER_SKIP_BROWSER=1")
class RenderedButtons(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from coderunner.browser import BrowserBroker, launcher
        from coderunner.core.audit import AuditLog
        from coderunner.core.config import Config
        from coderunner.safety.policy import Policy
        from coderunner.core.store import Store

        cfg = Config({"browser": {"headless": True, "allow": ["127.0.0.1"]}})
        if not launcher.find_browser(cfg):
            self.skipTest("no Chromium-family browser installed")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(tasks._home(pathlib.Path(tmp)))
        self.site = sites.Sites(TODAY)
        server, serving = await self.site.start()

        async def stop_site():
            server.should_exit = True
            await serving

        self.addAsyncCleanup(stop_site)
        audit = AuditLog()
        policy = Policy(cfg, Store(":memory:"), audit)
        policy.add_listener(lambda a: policy.decide(a.id, True, decided_by="fixture test"))
        self.browser = BrowserBroker(cfg, policy, audit)
        self.addAsyncCleanup(self.browser.stop)
        self.clicks = []

    async def open(self, path):
        result = await self.browser.navigate(self.site.base + path)
        self.assertTrue(result["ok"], result)

    async def ref(self, role, name):
        content = (await self.browser.snapshot())["content"]
        match = re.search(r'\[(\d+)\] ' + re.escape(f'{role} "{name}"'), content)
        self.assertIsNotNone(match, content)
        return int(match[1])

    async def click(self, name):
        ref = await self.ref("button", name)
        result = await self.browser.click(ref)
        self.assertTrue(result["ok"], result)
        self.clicks.append({"name": name, "ref": ref})

    def assert_dynamic_source(self, path, label):
        with TestClient(sites.Sites(TODAY).app) as client:
            source = client.get(path).text
        self.assertNotIn("<button", source)
        self.assertNotIn(label, source)
        self.assertIn('src="/demo-controls.js"', source)

    def assert_grade(self, name, answer):
        task = next(t for t in tasks.TASKS if t.name == name)
        passed, reason = task.grade(answer, self.site, self.clicks)
        self.assertTrue(passed, reason)
        self.assertFalse(task.grade(answer, self.site, self.clicks[:-1])[0])

    async def test_stock_buttons_appear_after_pin_and_open_each_stores_inventory(self):
        offers = sites.drink_offers(TODAY)
        for i, store in enumerate(sites.STORES):
            label = f"Check stock at {store}"
            self.assert_dynamic_source(f"/grocers/{i}", label)
            await self.open(f"/grocers/{i}")
            self.assertNotIn(label, (await self.browser.snapshot())["content"])
            typed = await self.browser.type(await self.ref("textbox", "Delivery PIN"), sites.DELIVERY_PIN, submit=True)
            self.assertTrue(typed["ok"], typed)
            self.assertNotIn(store, self.site.stock_checked)
            self.assertNotIn("Volt", (await self.browser.snapshot())["content"])
            await self.click(label)
            price, fee, available = offers[store]
            content = (await self.browser.snapshot())["content"]
            self.assertIn(f"350 ml — ₹{price} — {'in stock' if available else 'sold out'}", content)
            self.assertIn(f"Delivery fee per order: ₹{fee}", content)
        total, winner = min((price + fee, store) for store, (price, fee, stock) in offers.items() if stock)
        self.assert_grade("groceries", f"{winner} ₹{total}")

    async def test_rewards_chat_reveals_send_after_typing_and_delivers_once(self):
        self.assert_dynamic_source("/rewards/chat", "Send message")
        await self.open("/rewards/")
        await self.click("Points activity")
        await self.click("Chat with us")
        self.assertNotIn('button "Send message"', (await self.browser.snapshot())["content"])
        message = f"Why is my {self.site.points:,}-point AirMiles transfer still pending?"
        typed = await self.browser.type(await self.ref("textbox", "Write a message"), message)
        self.assertTrue(typed["ok"], typed)
        await self.click("Send message")
        self.assertEqual(self.site.complaints, [message])
        self.assertIn(self.site.ticket, (await self.browser.snapshot())["content"])
        self.assert_grade("rewards-chat", self.site.ticket)

    async def test_draft_reads_activity_through_its_rendered_button_and_leaves_inbox_empty(self):
        self.assert_dynamic_source("/rewards/", "Points activity")
        await self.open("/rewards/")
        await self.click("Points activity")
        content = (await self.browser.snapshot())["content"]
        self.assertIn(f"{self.site.points:,} points", content)
        self.assertIn("Pending", content)
        self.assertEqual(self.site.complaints, [])
        self.assert_grade("rewards-draft", f"Why is my {self.site.points:,}-point AirMiles transfer pending?")


class MeterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_speed_from_the_runtime_timings(self):
        class LLM:
            async def chat(self, messages, tools=None, **kwargs):
                return {"content": "", "tool_calls": [], "usage": {"prompt_tokens": 900},
                        "timings": {"prompt_n": 300, "predicted_n": 44, "predicted_ms": 2000.0}}
        meter = tasks.Meter(LLM())
        await meter.llm.chat([])
        await meter.llm.chat([])
        summary = meter.summary()
        self.assertEqual((summary["gen_tok_s"], summary["read_tokens"], summary["largest_prompt"]), (22.0, 600, 900))


class Results(unittest.TestCase):
    def test_latest_result_per_machine_and_build(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}):
            base = {"suite": tasks.SUITE, "machine": {"chip": "Apple M3 Pro", "ram_gb": 18}, "runtime": "prism",
                    "runtime_version": "b1", "passed": 1, "total": 3, "seconds": 600.0, "median_request_s": 20.0,
                    "tasks": [{"name": "flights", "passed": True, "seconds": 200.0, "tool_calls": 9}]}
            tasks.save({**base, "build": "a.gguf"})
            tasks.save({**base, "build": "b.gguf"})
            tasks.save({**base, "build": "a.gguf", "passed": 3})
            saved = tasks.load()
            self.assertEqual([(r["build"], r["passed"]) for r in saved], [("b.gguf", 1), ("a.gguf", 3)])
            self.assertTrue(pathlib.Path(home, "assistant", "models", "tasks.json").exists())
            table = tasks.report(saved)
            self.assertLess(table.index("a.gguf"), table.index("b.gguf"))


if __name__ == "__main__":
    unittest.main()


class Harness(unittest.IsolatedAsyncioTestCase):
    """The whole run against a scripted model: no browser, files or real model needed."""

    async def test_a_claude_benchmark_with_no_key_cannot_fall_back_to_a_local_model(self):
        from coderunner.core.config import Config
        from coderunner.models import manager
        with tempfile.TemporaryDirectory() as home, tasks._home(pathlib.Path(home)), \
                mock.patch.object(manager, "ensure_running") as local:
            config = Config({"models": {"chat": "local", "remote": {"provider": "anthropic"}}})
            result = await tasks.run(config, names=["library"])
            self.assertFalse(result["ok"])
            local.assert_not_called()

    async def test_native_claude_uses_the_private_benchmark_home_and_records_its_model(self):
        from coderunner.core.claude import Claude
        from coderunner.core.config import Config
        from coderunner.models import manager, remote
        if not remote.sdk_installed():
            self.skipTest("Claude extra is optional")
        test_key = "sk-ant-" + "fixture" * 5
        with tempfile.TemporaryDirectory() as home, tasks._home(pathlib.Path(home)):
            remote.keep_key(test_key)
            owner_path = remote.key_path()
            used = []

            async def reply(client, *args, **kwargs):
                used.append(remote.key_path())
                self.assertNotEqual(used[-1], owner_path)
                self.assertEqual(remote.key(), test_key)
                return {"content": "Sunday closing time 14:00, DVD fee ₹15 per day.", "tool_calls": []}

            config = Config({"models": {"remote": {"provider": "anthropic", "model": remote.DEFAULT}}})
            with mock.patch.object(Claude, "chat", reply), mock.patch.object(manager, "ensure_running") as local:
                result = await tasks.run(config, names=["library"], progress=lambda s: None)
            self.assertEqual((result["model"], result["runtime"], result["passed"]), (remote.DEFAULT, "anthropic", 1))
            self.assertTrue(used and all(not p.exists() for p in used))
            self.assertNotIn(test_key, str(result))
            self.assertTrue(owner_path.exists())
            local.assert_not_called()

    async def test_answers_are_checked_and_a_slow_task_is_stopped(self):
        from coderunner.core.config import Config
        from fakes import FakeOpenAI

        slow = {"role": "assistant", "content": "still thinking", "_delay": 2.5}

        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}), \
                FakeOpenAI() as fake, mock.patch.object(tasks, "TASK_SECONDS", 1):
            fake.script = [slow, {"role": "assistant", "content": "Sundays until 14:00; DVDs ₹15 a day."}]
            config = Config({"models": {"runtime": "openai", "base_url": fake.base_url, "chat": "fake"}},
                            path=pathlib.Path(home) / "config.json")
            result = await tasks.run(config, names=["library", "flights"], progress=lambda m: None)
            by_name = {t["name"]: t for t in result["tasks"]}
            self.assertTrue(by_name["flights"]["passed"] is False and "still running" in by_name["flights"]["why"],
                            by_name)
            self.assertLess(by_name["flights"]["seconds"], 3)
            self.assertTrue(by_name["library"]["passed"], by_name["library"])
            self.assertEqual(tasks.load()[-1]["passed"], result["passed"])


class TimeLimitTests(unittest.IsolatedAsyncioTestCase):
    """A task with a time limit (the check's) posts an interim answer, works on past the limit, and is stopped at
    twice it."""

    async def test_an_interim_answer_is_kept_with_the_task_and_the_hard_stop_is_at_twice_the_limit(self):
        import asyncio
        import time
        from coderunner.server import tasks as server_tasks
        from fakes import Svc
        from test_gateway import FakeDaemon
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}):
            (pathlib.Path(home) / "assistant").mkdir()
            svc = Svc({"models": {"chat": "coderunner-test"}})
            daemon = FakeDaemon(svc)
            runner = server_tasks.Tasks(daemon, server_tasks.EventBus(svc.store))
            seen = {}

            async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
                seen["hard"] = events.stop_at - events.deadline
                events({"type": "interim", "text": "Taj at ₹18,200 so far."})
                await asyncio.sleep(0.3)            # past the 0.2 s limit: not stopped there
                seen["past"] = True
                await asyncio.sleep(60)
            daemon.handle_message = handle
            began = time.monotonic()
            task = runner.start("app:t", "Hotels?", "check", time_limit=0.2)
            await asyncio.sleep(0.05)
            self.assertEqual(runner.so_far(task["id"]), ("Taj at ₹18,200 so far.", True))
            row = await runner.wait(task["id"])
            took = time.monotonic() - began
            self.assertTrue(seen["past"])
            self.assertTrue(0.38 <= took < 1.5, took)
            self.assertAlmostEqual(seen["hard"], 0.2, places=2)
            # Stopped for good: as a task the owner stops, from what it had found.
            self.assertEqual((row["status"], row["reply"]), ("cancelled", "Stopped before I found anything."))
            kinds = [(e["type"], e.get("text")) for e in runner.bus.since(0, "app:t")]
            # Kept with the task's events, so a reopened thread shows it, before how the task ended.
            self.assertLess(kinds.index(("interim", "Taj at ₹18,200 so far.")), [k for k, _ in kinds].index("cancelled"))
            # A task with no time limit is never stopped by one.
            async def quick(channel, chat_id, text, origin="", events=None, task="", images=None):
                seen["limit"] = getattr(events, "stop_at", 0)
                return "ok"
            daemon.handle_message = quick
            row = await runner.wait(runner.start("app:u", "Hi", "owner")["id"])
            self.assertEqual((row["status"], row["reply"], seen["limit"]), ("done", "ok", 0))

    async def test_time_spent_waiting_for_the_owner_moves_the_limit_and_the_hard_stop_on(self):
        import asyncio
        import time
        from coderunner.agent.loop import RunContext
        from coderunner.server import tasks as server_tasks
        from fakes import Svc
        from test_gateway import FakeDaemon
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}):
            (pathlib.Path(home) / "assistant").mkdir()
            svc = Svc({"models": {"chat": "coderunner-test"}})
            daemon = FakeDaemon(svc)
            runner = server_tasks.Tasks(daemon, server_tasks.EventBus(svc.store))
            seen = {}

            async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
                # As the daemon builds the run, and as the browser counts a wait for the owner.
                ctx = RunContext(conversation="app:t", events=events, deadline=events.deadline, stop_at=events.stop_at)
                before = (events.deadline, events.stop_at, ctx.deadline, ctx.stop_at)
                ctx.pause(0.5)
                seen["moved"] = [after - was for after, was in
                                 zip((events.deadline, events.stop_at, ctx.deadline, ctx.stop_at), before)]
                await asyncio.sleep(60)
            daemon.handle_message = handle
            began = time.monotonic()
            row = await runner.wait(runner.start("app:t", "Hotels?", "check", time_limit=0.2)["id"])
            took = time.monotonic() - began
            for moved in seen["moved"]:
                self.assertAlmostEqual(moved, 0.5, places=6)
            # Stopped at twice the limit plus the wait (0.9 s), not at twice the limit (0.4 s).
            self.assertTrue(0.85 <= took < 1.6, took)
            self.assertEqual(row["status"], "cancelled")
