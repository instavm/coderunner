import json
import unittest

from coderunner.agent import Agent, RunContext, build_registry
from coderunner.agent.recipes import MAX_CHARS, MAX_SITES, MAX_STEPS, Recipes, build, matches, save_from_run
from coderunner.core.llm import LLM
from fakes import FakeOpenAI, Svc, tool_call

HOME = "https://www.trains.example/"
RESULTS = ("https://www.trains.example/search/results?from=SBC&to=NDLS&date=2026-12-01&class=sleeper&adults=1"
           "&utm_source=newsletter&sid=8f2a9c1e")
REQUEST = "Find a sleeper train from Bengaluru to New Delhi on 1 December 2026 on trains.example"


class SiteBrowser:
    """A site with a search form: [4] From, [5] To, [6] Date, [7] the Search button, which opens the results."""

    LABELS = {4: "combobox From", 5: "combobox To", 6: "textbox Date", 7: "button Search"}

    def __init__(self, fail=()):
        self.url, self.fail, self.opened = HOME, set(fail), []

    def _label(self, ref, tab=None):
        return self.LABELS.get(int(ref), "")

    async def navigate(self, url, tab=None, tainted=False):
        self.opened.append(url)
        if url in self.fail:
            return {"ok": False, "error": "This page could not be reached"}
        self.url = url
        return {"ok": True, "url": url, "title": "Trains", "content": "[4] combobox From\n[5] combobox To\n"
                "[6] textbox Date\n[7] button Search"}

    async def type(self, ref, text, submit=False, tab=None, tainted=False, clear=True, pick=False):
        picked = {4: "Bengaluru (SBC)", 5: "New Delhi (NDLS)"}.get(int(ref))
        return {"ok": True, "url": self.url, "changes": f"+ {text}", **({"picked": picked} if picked else {})}

    async def click(self, ref, tab=None, tainted=False):
        if int(ref) == 7:
            self.url = RESULTS
            return {"ok": True, "navigated": True, "url": RESULTS, "title": "Results", "content": "Rajdhani 22691"}
        return {"ok": True, "url": self.url, "changes": "+ clicked"}

    async def snapshot(self, tab=None, tainted=False, start=""):
        return {"ok": True, "url": self.url, "title": "Trains", "content": "page", "changed": True}


FORM = {"steps": [{"action": "type", "ref": 4, "text": "Bengaluru"}, {"action": "type", "ref": 5, "text": "New Delhi"},
                  {"action": "type", "ref": 6, "text": "2026-12-01"}, {"action": "click", "ref": 7}]}


def trail() -> list[dict]:
    return [{"action": "search", "url": "https://www.bing.com/search?q=trains", "ok": True},
            {"action": "open", "url": HOME, "ok": True},
            {"action": "type", "url": HOME, "ok": True, "label": "combobox From", "text": "Bengaluru",
             "picked": "Bengaluru (SBC)"},
            {"action": "type", "url": HOME, "ok": True, "label": "combobox To", "text": "New Delhi",
             "picked": "New Delhi (NDLS)"},
            {"action": "type", "url": HOME, "ok": True, "label": "textbox Date", "text": "2026-12-01"},
            {"action": "click", "url": HOME, "ok": False, "label": "button Search"},
            {"action": "click", "url": RESULTS, "ok": True, "label": "button Search"},
            {"action": "click", "url": "https://www.trains.example/train/22691", "ok": True, "label": "link Rajdhani"}]


class RecipeTests(unittest.TestCase):
    def test_a_recipe_names_what_varies_and_keeps_the_shortest_way_that_worked(self):
        recipe = build(trail(), "trains.example", REQUEST)
        self.assertEqual(recipe["url"], "https://www.trains.example/search/results?from={from}&to={to}&date={date}"
                                        "&class=sleeper&adults={adults}")
        self.assertEqual(recipe["start"], HOME)
        self.assertEqual(recipe["steps"], ["type {from} in combobox From", "type {to} in combobox To",
                                           "type {date} in textbox Date", "click button Search"])
        text = json.dumps(recipe)
        for kept_out in ("Bengaluru", "SBC", "NDLS", "Delhi", "2026", "utm_source", "newsletter", "sid", "8f2a9c1e"):
            self.assertNotIn(kept_out, text)

    def test_an_address_opened_directly_is_the_whole_recipe(self):
        direct = [{"action": "open", "url": "https://shop.example/s?k=usb+c+hub&sort=price", "ok": True}]
        recipe = build(direct, "shop.example", "Cheapest \"usb c hub\" on shop.example")
        self.assertEqual(recipe["url"], "https://shop.example/s?k={k}&sort=price")
        self.assertEqual(recipe["steps"], [])

    def test_personal_values_credentials_and_sign_in_links_are_never_kept(self):
        steps = [{"action": "open", "url": "https://mail.example/inbox?user=me@home.example&q=invoice&page=2"
                  "&auth=abc&tracking=x" + "y" * 60, "ok": True},
                 {"action": "type", "url": "https://mail.example/inbox", "ok": True, "label": "textbox Password",
                  "text": "hunter2!"},
                 {"action": "click", "url": "https://mail.example/inbox/v/9f8e7d6c5b4a39281706f5e4d3c2b1a0Zx",
                  "ok": True, "label": "link Acct 4111 1111"}]
        recipe = build(steps, "mail.example", "find my invoice")
        text = json.dumps(recipe)
        for kept_out in ("me@home", "hunter2", "auth", "tracking", "4111", "9f8e7d6c5b4a"):
            self.assertNotIn(kept_out, text)
        self.assertIn("q=invoice", recipe["url"])     # a plain value of the site's own stays
        for link in ("https://site.example/login?token=abc", "https://site.example/verify-email?code=123456",
                     "https://site.example/reset/password"):
            self.assertIsNone(build([{"action": "open", "url": link, "ok": True}], "site.example", "x"))

    def test_recipes_are_capped_per_site_and_in_number_newest_winning(self):
        long_way = [{"action": "open", "url": HOME, "ok": True}] + [
            {"action": "click", "url": HOME, "ok": True, "label": f"button Step {chr(65 + n)} " + "x" * 25}
            for n in range(20)] + [{"action": "click", "url": RESULTS, "ok": True, "label": "button Search"}]
        recipe = build(long_way, "trains.example", REQUEST)
        self.assertLessEqual(len(recipe["steps"]), MAX_STEPS)
        from coderunner.agent.recipes import render
        self.assertLessEqual(len(render(recipe)), MAX_CHARS)
        svc = Svc()
        recipes = Recipes(svc.store)
        for n in range(MAX_SITES + 5):
            recipes.put({"host": f"site{n}.example", "url": f"https://site{n}.example/", "start": "", "steps": [],
                         "saved": n})
        self.assertEqual(len(recipes.all()), MAX_SITES)
        self.assertNotIn("site0.example", recipes.all())
        recipes.put({"host": "site9.example", "url": "https://site9.example/new", "start": "", "steps": [],
                     "saved": 1000})
        self.assertEqual(recipes.get("site9.example")["url"], "https://site9.example/new")

    def test_only_sites_the_request_names_are_recalled(self):
        svc = Svc()
        save_from_run(svc.store, trail(), REQUEST, f"The [Rajdhani]({RESULTS}) leaves at 20:00.")
        recipes = Recipes(svc.store)
        self.assertEqual(list(recipes.all()), ["trains.example"])     # never the search site
        self.assertEqual(recipes.named_in("Any sleeper on Trains from Pune?"), ["trains.example"])
        self.assertEqual(recipes.named_in("check www.trains.example for Goa"), ["trains.example"])
        self.assertEqual(recipes.named_in("What is the weather in Pune?"), [])
        self.assertEqual(recipes.block([]), "")

    def test_an_address_matches_its_recipe_with_new_values(self):
        template = "https://www.trains.example/route/{from}/{to}?date={date}"
        self.assertTrue(matches(template, "https://trains.example/route/MAS/HWH?date=2027-01-02"))
        self.assertFalse(matches(template, "https://trains.example/route/MAS"))
        self.assertFalse(matches(template, "https://other.example/route/MAS/HWH"))


class RecipeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, svc, fake, script, request, origin_is_owner=True):
        fake.script = script
        ctx = RunContext(conversation=f"c{len(fake.requests)}", origin_is_owner=origin_is_owner)
        reply = await Agent(svc, svc.llm, svc.registry).run(ctx, request)
        return reply, ctx

    def _svc(self, fake, browser):
        svc = Svc()
        svc.browser = browser
        svc.llm = LLM(fake.base_url, "test")
        svc.registry = build_registry(svc)
        return svc

    async def test_a_recipe_is_saved_after_an_answer_and_offered_when_the_site_comes_up_again(self):
        with FakeOpenAI() as fake:
            svc = self._svc(fake, SiteBrowser())
            answer = {"role": "assistant", "content": f"The [Rajdhani]({RESULTS}) at 20:00 is the only sleeper."}
            await self._run(svc, fake, [tool_call("browser_open", {"url": HOME}), tool_call("browser_batch", FORM),
                                        answer], REQUEST)
            recipe = Recipes(svc.store).get("trains.example")
            self.assertIn("from={from}&to={to}&date={date}", recipe["url"])
            self.assertNotIn("Bengaluru", json.dumps(recipe))
            # Named in the next request: offered with its background, as untrusted data.
            start = len(fake.requests)
            await self._run(svc, fake, [{"role": "assistant", "content": "ok"}],
                            "A sleeper from Chennai to Kolkata on trains.example tomorrow?")
            background = fake.requests[start]["messages"][-2]["content"]
            self.assertIn('<untrusted source="site_notes">', background)
            self.assertIn("results at https://www.trains.example/search/results?from={from}", background)
            # Not named: nothing in the background, but the note comes once when the task reaches the site.
            start = len(fake.requests)
            await self._run(svc, fake, [tool_call("browser_open", {"url": HOME}),
                                        tool_call("browser_open", {"url": HOME + "help"}),
                                        {"role": "assistant", "content": "ok"}], "Cheapest way to Goa by rail?")
            self.assertNotIn("site_notes", fake.requests[start]["messages"][-2]["content"])
            results = [m["content"] for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
            self.assertIn("site_notes", results[0])
            self.assertNotIn("site_notes", results[1])
            await svc.llm.aclose()

    async def test_no_recipe_after_a_task_that_gave_up_or_was_not_the_owners(self):
        with FakeOpenAI() as fake:
            svc = self._svc(fake, SiteBrowser())
            steps = [tool_call("browser_open", {"url": HOME}), tool_call("browser_batch", FORM)]
            await self._run(svc, fake, steps + [{"role": "assistant", "content":
                                                 "I couldn't find a sleeper train on that date."}], REQUEST)
            self.assertEqual(Recipes(svc.store).all(), {})
            await self._run(svc, fake, steps + [{"role": "assistant", "content": f"[Rajdhani]({RESULTS})"}],
                            REQUEST, origin_is_owner=False)
            self.assertEqual(Recipes(svc.store).all(), {})
            svc.config.set("agent.max_steps", 1)        # out of steps: answered only when told to
            await self._run(svc, fake, steps * 4 + [{"role": "assistant", "content": f"[Rajdhani]({RESULTS})"}] * 4,
                            REQUEST)
            self.assertEqual(Recipes(svc.store).all(), {})
            await svc.llm.aclose()

    async def test_a_stale_recipe_is_dropped_when_its_address_fails_and_the_task_goes_on(self):
        stale = "https://www.trains.example/old/route?from=MAS"
        with FakeOpenAI() as fake:
            svc = self._svc(fake, SiteBrowser(fail={stale}))
            Recipes(svc.store).put({"host": "trains.example", "url": "https://www.trains.example/old/route?from={from}",
                                    "start": "", "steps": [], "saved": 1})
            reply, ctx = await self._run(svc, fake, [
                tool_call("browser_open", {"url": stale}), tool_call("browser_open", {"url": HOME}),
                tool_call("browser_batch", FORM),
                {"role": "assistant", "content": f"The [Rajdhani]({RESULTS}) at 20:00."}], REQUEST)
            self.assertIn("Rajdhani", reply)
            recipe = Recipes(svc.store).get("trains.example")        # replaced by the way that worked
            self.assertIn("/search/results?from={from}", recipe["url"])
            events = [json.loads(line)["event"] for line in svc.audit.path.read_text().splitlines()]
            self.assertIn("agent.recipe_dropped", events)
            await svc.llm.aclose()

    async def test_recipes_can_be_turned_off(self):
        with FakeOpenAI() as fake:
            svc = self._svc(fake, SiteBrowser())
            svc.config.set("agent.site_recipes", False)
            await self._run(svc, fake, [tool_call("browser_open", {"url": HOME}), tool_call("browser_batch", FORM),
                                        {"role": "assistant", "content": f"[Rajdhani]({RESULTS})"}], REQUEST)
            self.assertEqual(Recipes(svc.store).all(), {})
            await svc.llm.aclose()


class PlanDoneTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_plan_all_done_is_a_nudge_to_answer_said_once(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            done = {"steps": [{"step": "Find the fare", "status": "done"}, {"step": "Find the hotel", "status": "done"}]}
            fake.script = [tool_call("plan_update", {"steps": [{"step": "Find the fare", "status": "doing"},
                                                               {"step": "Find the hotel"}]}),
                           tool_call("plan_update", done), tool_call("tell_owner", {"text": "Fare ₹4,000"}),
                           tool_call("plan_update", {"steps": done["steps"] + [{"step": "Check the bus", "status": "done"}]}),
                           {"role": "assistant", "content": "done"}]
            await Agent(svc, svc.llm, svc.registry).run(RunContext(conversation="plan"), "fare and hotel")
            said = [m["content"] for m in fake.requests[-1]["messages"]
                    if m["role"] == "user" and "Reminder of your plan" in str(m["content"])]
            self.assertNotIn("Every step is done", said[0])
            self.assertIn("Every step is done", said[1])
            self.assertIn("add it as a step", said[1])
            self.assertNotIn("Every step is done", said[2])      # still all done: not said again
            await svc.llm.aclose()


if __name__ == "__main__":
    unittest.main()
