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
        self.assertTrue(tasks.check_documents("Covered until 14 March 2030. Call 1800-266-1880.", self.site)[0])
        self.assertTrue(tasks.check_documents("Until 2030-03-14; Kelvin Care 1800 266 1880", self.site)[0])
        self.assertFalse(tasks.check_documents("Until 14 March 2027; call 1800 266 1880", self.site)[0])
        self.assertFalse(tasks.check_documents("Until March 2030; call 1800 103 4455", self.site)[0])


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
