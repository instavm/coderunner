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
        self.site.visits.append("/flights/checkout?flight=AI2327")
        self.assertEqual(tasks.check_flights(answer, self.site), (False, "opened a booking page"))

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
