"""What a task keeps of what it read: key lines in place of a cut result, a page read before answered from
them, and trims that scale with the context."""

import json
import unittest

from coderunner.agent import Agent, RunContext, build_registry
from coderunner.agent import loop as loop_module
from coderunner.agent.digest import (PAGE_CHARS, OUTPUT_CHARS, Readings, cut_level, digest, is_cut, normal_url,
                                     request_words)
from coderunner.core.llm import LLM
from fakes import FakeOpenAI, Svc, tool_call

REQUEST = "Compare the best noise cancelling headphones under 40000 rupees: price, rating and weight"
FILLER = ['[{r}] link "Visit the store"', '[{r}] button "Add to Wish List"', "Frequently bought together",
          "Industry-leading noise cancellation with two processors controlling eight microphones.",
          "Ultra-comfortable, lightweight design with soft fit leather.", '[{r}] link "Report an issue"',
          "Sound is amazing and calls are clear even outdoors.", '[{r}] button "Helpful"', "Technical Details"]
PRODUCTS = [("Sony WH-1000XM5", "₹26,990", "4.4", "250 g"), ("Bose QuietComfort Ultra", "₹35,900", "4.3", "254 g"),
            ("Sennheiser Momentum 4", "₹24,990", "4.2", "293 g"), ("JBL Tour One M2", "₹17,999", "4.1", "268 g"),
            ("Sony WH-CH720N", "₹8,990", "4.2", "192 g")]


def product(n: int, name: str, price: str, rating: str, weight: str, chars: int = 6800,
            source: str = "browser_open", url: str = "") -> str:
    lines = ['[menu "Main" with 84 links and buttons (Deals, Service, …); browser_find searches it]',
             f"# {name} Wireless Noise Cancelling Headphones", f"{rating} out of 5 stars", '[12] link "12,408 ratings"',
             "-21%", price, "M.R.P.: ₹34,990", f"FREE delivery Friday, 28 November", f"Item Weight {weight}",
             '[51] button "Add to Cart"']
    r = 60
    while sum(len(line) + 1 for line in lines) < chars:
        lines.append(FILLER[r % len(FILLER)].format(r=r))
        r += 1
    url = url or f"https://www.shop.example/p{n}/dp/B0{n:04d}"
    return (f'<untrusted source="{source}">\ntab: t1\nurl: {url}\ntitle: {name} : Shop\n\n' + "\n".join(lines)
            + "\n</untrusted>")


def conversation(steps: int) -> tuple[list[dict], list[str]]:
    messages = [{"role": "system", "content": "s" * 11000}, {"role": "user", "content": REQUEST}]
    for n in range(steps):
        name = "run_python" if n % 5 == 4 else "browser_open"
        content = ('<untrusted source="run_python">\n' + "\n".join(f"row {i}: {i * 37 % 101}" for i in range(260))
                   + "\ntotal 4242\n</untrusted>") if name == "run_python" else product(n, *PRODUCTS[(n - n // 5) % 5])
        messages += [{"role": "assistant", "content": "", "tool_calls": [
                         {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": f"c{n}", "content": content}]
    return messages, sorted({p[1] for p in PRODUCTS})


class DigestTests(unittest.TestCase):
    def test_a_page_keeps_its_address_title_and_the_lines_that_matter_within_its_cap(self):
        page = product(3, *PRODUCTS[0])
        short = digest(page, 7, request_words(REQUEST))
        self.assertTrue(short.startswith('<untrusted source="browser_open">\n[page read at step 7, cut to its key lines]\n'))
        self.assertTrue(short.endswith("\n</untrusted>"))
        for fact in ("url: https://www.shop.example/p3/dp/B00003", "title: Sony WH-1000XM5 : Shop",
                     "# Sony WH-1000XM5", "₹26,990", "M.R.P.: ₹34,990", "4.4 out of 5 stars", "12,408 ratings",
                     "28 November", "Item Weight 250 g"):
            self.assertIn(fact, short)
        self.assertNotIn("[12]", short)                        # element numbers are stale once the page moved on
        self.assertNotIn("[menu", short)
        self.assertNotIn("Report an issue", short)
        body = short.split("\n")[4:-1]
        self.assertLessEqual(sum(len(line) + 1 for line in body), PAGE_CHARS)
        self.assertLess(len(short), len(page) / 8)
        self.assertEqual(len(body), len(set(body)))            # no repeats
        self.assertTrue(is_cut(short) and cut_level(short) == 1)
        self.assertFalse(is_cut(page))
        # A digest of a digest, cut short, keeps the price still; a page's text cannot pose as a digest.
        shorter = digest(short, 7, request_words(REQUEST), short=True)
        self.assertIn("₹26,990", shorter)
        self.assertEqual(cut_level(shorter), 2)
        self.assertFalse(is_cut(page.replace("tab: t1", "[page read at step 1, cut to its key lines]")))

    def test_other_output_keeps_its_first_and_last_lines_and_lines_with_numbers(self):
        out = '<untrusted source="run_shell">\n' + "\n".join(
            ["Collecting requests"] + ["Downloading chunk ......"] * 300 + ["Saved 3 files, 41.5 MB", "exit code 0"]
        ) + "\n</untrusted>"
        short = digest(out, 4)
        self.assertTrue(short.startswith('<untrusted source="run_shell">\n[output of step 4, cut to its key lines]'))
        for line in ("Collecting requests", "Saved 3 files, 41.5 MB", "exit code 0"):
            self.assertIn(line, short)
        self.assertLessEqual(len(short), OUTPUT_CHARS + 100)
        self.assertEqual(short.count("Downloading chunk"), 1)

    def test_addresses_are_one_key_without_tracking_or_fragment(self):
        self.assertEqual(normal_url("https://www.Shop.example/dp/B01/?utm_source=x&th=1&gclid=abc#reviews"),
                         normal_url("https://shop.example/dp/B01?th=1"))
        self.assertNotEqual(normal_url("https://shop.example/dp/B01?th=1"), normal_url("https://shop.example/dp/B02"))
        # The page header cuts long values (tools._short_address): both sides compare alike.
        long = "https://shop.example/s?k=" + "x" * 90
        self.assertEqual(normal_url(long), normal_url("https://shop.example/s?k=" + "x" * 40 + "…"))


class TrimTests(unittest.TestCase):
    def agent(self):
        svc = Svc()
        svc.registry = build_registry(svc)
        return Agent(svc, svc.llm, svc.registry)

    def play(self, steps: int, context: int) -> tuple[list[dict], list[dict]]:
        """A task played step by step against _trim, as the loop does."""
        full, _ = conversation(steps)
        agent, messages, trims = self.agent(), full[:2], []
        for i in range(2, len(full), 2):
            messages += full[i:i + 2]
            known = [len(messages), agent._size(messages, [0, 0])]
            done = agent._trim(messages, known, context, context, request_words(REQUEST))
            if done:
                trims.append(done)
        return messages, trims

    def test_prices_survive_trims_of_a_thirty_step_task_of_long_pages(self):
        messages, trims = self.play(30, 32768)
        self.assertTrue(trims)
        text = "\n".join(m["content"] for m in messages)
        for price in sorted({p[1] for p in PRODUCTS}):
            self.assertIn(price, text)
        cut = [m["content"] for m in messages if m["role"] == "tool" and is_cut(m["content"])]
        self.assertGreater(len(cut), 20)
        self.assertTrue(all("[page read at step" in c or "[output of step" in c for c in cut))
        # Each trim goes down in one go and the conversation stays well inside the context.
        for done in trims:
            self.assertGreaterEqual(done["freed"], loop_module.MIN_FREED * 32768)
        self.assertLess(self.agent()._size(messages, [0, 0]), loop_module.TRIM_AT * 32768)

    def test_digests_keep_to_their_share_the_oldest_cut_short_first(self):
        messages, _ = self.play(150, 32768)
        levels = [cut_level(m["content"]) for m in messages if m["role"] == "tool"]
        kept = sum(len(m["content"]) for m in messages if m["role"] == "tool" and cut_level(m["content"]) in (1, 2))
        budget = loop_module.DIGESTS * loop_module.low_water(32768) * loop_module.CHARS_PER_TOKEN
        self.assertLessEqual(kept, budget * 1.2)
        first_whole = levels.index(1)
        self.assertTrue(all(level >= 2 for level in levels[:first_whole]))       # older ones shorter
        self.assertIn(3, levels)
        self.assertTrue(all(level == 0 for level in levels[-2:]))                # the latest two steps whole
        text = "\n".join(m["content"] for m in messages)
        for price in sorted({p[1] for p in PRODUCTS}):                           # the latest pages' prices
            self.assertIn(price, text)

    def test_the_models_own_steps_are_never_cut(self):
        agent = self.agent()
        messages, _ = conversation(12)
        finding = "Sony WH-1000XM5 is ₹26,990 at the shop; Bose ₹35,900. " * 20      # a long tell_owner result
        messages[4:4] = [{"role": "assistant", "content": "", "tool_calls": [
                             {"id": "t1", "type": "function", "function": {
                                 "name": "tell_owner", "arguments": json.dumps({"text": "XM5 ₹26,990",
                                                                                "say": "Found the XM5 price"})}}]},
                         {"role": "tool", "tool_call_id": "t1", "content": finding},
                         {"role": "user", "content": loop_module.PLAN_REMINDER + "1. [done] Find prices"}]
        snapshot = [dict(m) for m in messages[4:7]]
        done = agent._trim(messages, [0, 0], 32768, 22000, request_words(REQUEST))
        self.assertTrue(done and done["must"])
        self.assertEqual(messages[4:7], snapshot)                # its call (with the say line), result and plan
        self.assertTrue(is_cut(messages[3]["content"]))


class ContextTests(unittest.TestCase):
    def test_a_larger_context_trims_later_and_still_in_one_go(self):
        self.assertEqual(loop_module.low_water(32768), loop_module.LOW_WATER * 32768)
        self.assertEqual(loop_module.low_water(16384), loop_module.LOW_WATER * 16384)
        for context in (131072, 262144):
            low = loop_module.low_water(context)
            self.assertGreater(low, loop_module.low_water(32768))
            self.assertLess(low, loop_module.LOW_WATER * context)           # less left to read again after a trim
            self.assertGreater(loop_module.TRIM_AT * context - low, loop_module.MIN_FREED * context)
        svc = Svc()
        svc.registry = build_registry(svc)
        agent = Agent(svc, svc.llm, svc.registry)
        full, _ = conversation(60)
        # At 128K, 40K tokens of pages is no reason to trim; the first trim comes past 60% and goes down at once.
        messages = full[:2 + 2 * 14]
        self.assertGreater(agent._size(messages, [0, 0]), 0.6 * 32768)
        self.assertIsNone(agent._trim(messages, [0, 0], 131072, 131072))
        messages, trims = full[:2], []
        for i in range(2, len(full), 2):
            messages += full[i:i + 2]
            done = agent._trim(messages, [0, 0], 131072, 131072, request_words(REQUEST))
            if done:
                trims.append(done)
        self.assertEqual(len(trims), 1)
        self.assertGreater(trims[0]["before"], loop_module.TRIM_AT * 131072)
        self.assertLessEqual(trims[0]["after"], loop_module.low_water(131072))


class ReadingsTests(unittest.TestCase):
    def test_a_page_read_before_comes_back_as_its_key_lines_unless_fresh(self):
        seen = Readings(request_words(REQUEST))
        page = product(1, *PRODUCTS[0], url="https://www.shop.example/xm5?utm_source=feed")
        seen.record("browser_open", {"url": "https://www.shop.example/xm5?utm_source=feed"}, page, 3)
        again = seen.kept("browser_open", {"url": "https://shop.example/xm5#reviews"}, 9)
        self.assertTrue(again.startswith("You read this page at step 3;"))
        self.assertIn("fresh=true", again)
        self.assertIn("₹26,990", again)
        self.assertIn('<untrusted source="browser_open">', again)       # page text stays marked as such
        self.assertEqual(seen.kept("browser_open", {"url": "https://shop.example/xm5", "fresh": True}, 9), "")
        self.assertEqual(seen.kept("browser_open", {"url": "https://shop.example/other"}, 9), "")
        # A snapshot of it later is answered too; one right after (a page still loading) is not, nor a part.
        self.assertTrue(seen.kept("browser_snapshot", {}, 9).startswith("You read this page at step 3;"))
        self.assertEqual(seen.kept("browser_snapshot", {}, 4), "")
        self.assertEqual(seen.kept("browser_snapshot", {"start": "Reviews"}, 9), "")

    def test_a_page_acted_on_is_read_again_and_a_change_in_place_lets_go_of_the_site(self):
        seen = Readings()
        cart = product(2, "Cart", "₹26,990", "4", "1 g", url="https://shop.example/cart")
        seen.record("browser_open", {"url": "https://shop.example/cart"}, cart, 1)
        seen.record("browser_open", {"url": "https://shop.example/a"}, product(3, *PRODUCTS[0], url="https://shop.example/a"), 2)
        seen.record("browser_open", {"url": "https://news.example/x"}, product(4, *PRODUCTS[1], url="https://news.example/x"), 3)
        seen.record("browser_open", {"url": "https://shop.example/b"}, product(5, *PRODUCTS[2], url="https://shop.example/b"), 4)
        # A link followed from b: b was acted on; what else was read stays.
        moved = '<untrusted source="browser_click">\ntab: t1\nurl: https://shop.example/c\nnavigated: True\n\n# C\n</untrusted>'
        seen.record("browser_click", {"ref": 4}, moved, 5)
        self.assertEqual(seen.kept("browser_open", {"url": "https://shop.example/b"}, 9), "")
        self.assertTrue(seen.kept("browser_open", {"url": "https://shop.example/a"}, 9))
        self.assertTrue(seen.kept("browser_open", {"url": "https://shop.example/c"}, 9))   # read by that click
        # Scrolling or typing on c changes c only.
        seen.record("browser_scroll", {"direction": "down"}, '<untrusted source="browser_scroll">\nscrolled: down\n</untrusted>', 5)
        self.assertEqual(seen.kept("browser_open", {"url": "https://shop.example/c"}, 9), "")
        self.assertTrue(seen.kept("browser_open", {"url": "https://shop.example/a"}, 9))
        # A change made in place on c (an add-to-cart button): the shop's other pages may show it.
        seen.record("browser_click", {"ref": 7}, '<untrusted source="browser_click">\ntab: t1\nChanges on the page:\n+ Added\n</untrusted>', 6)
        for url in ("https://shop.example/cart", "https://shop.example/a", "https://shop.example/c"):
            self.assertEqual(seen.kept("browser_open", {"url": url}, 9), "")
        self.assertTrue(seen.kept("browser_open", {"url": "https://news.example/x"}, 9))
        # Read again after: answered from the new reading, but a page acted on never is.
        seen.record("browser_open", {"url": "https://shop.example/a"}, product(3, *PRODUCTS[0], url="https://shop.example/a"), 7)
        self.assertIn("at step 7", seen.kept("browser_open", {"url": "https://shop.example/a"}, 9))
        seen.record("browser_open", {"url": "https://shop.example/c"}, product(6, *PRODUCTS[3], url="https://shop.example/c"), 8)
        self.assertEqual(seen.kept("browser_open", {"url": "https://shop.example/c"}, 9), "")
        # The owner or the driver moved the browser on: nothing is answered from before.
        seen.record("browser_handoff", {"reason": "Sign in"}, "The owner finished the step.", 9)
        self.assertEqual(seen.kept("browser_open", {"url": "https://news.example/x"}, 10), "")

    def test_the_second_return_to_a_page_is_nudged(self):
        repeats = loop_module.Repeats()
        self.assertEqual(repeats.check("browser_open", {"url": "https://a.example/x"}, 0), "")
        self.assertEqual(repeats.check("browser_open", {"url": "https://a.example/x?utm_medium=y"}, 3, kept=True), "")
        self.assertIn("visit 3 to this page (first at step 1)", repeats.check("browser_open", {"url": "https://a.example/x"}, 5))


class Shop:
    """A browser whose pages are product pages, counting what it loaded."""

    def __init__(self):
        self.loaded: list[str] = []
        self.clicks = 0

    async def navigate(self, url, tainted=False):
        self.loaded.append(url)
        n = len(self.loaded)
        page = product(n, *PRODUCTS[n % 5], chars=1200, url=url)
        content = page.split("\n\n", 1)[1].rsplit("\n</untrusted>", 1)[0]
        return {"ok": True, "url": url, "title": "Shop", "content": content}

    async def click(self, ref, tainted=False):
        self.clicks += 1
        return {"ok": True, "navigated": False, "changes": "+ Added to cart"}


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_page_opened_again_is_answered_from_its_reading_and_fresh_loads_it(self):
        with FakeOpenAI() as fake:
            svc = Svc()
            svc.browser = Shop()
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            a, b = "https://shop.example/a", "https://shop.example/b"
            fake.script = [tool_call("browser_open", {"url": a}, "c1"), tool_call("browser_open", {"url": b}, "c2"),
                           tool_call("browser_open", {"url": a + "?utm_source=x", "say": "Back to the XM5"}, "c3"),
                           tool_call("browser_open", {"url": a, "fresh": True}, "c4"),
                           tool_call("browser_click", {"ref": 51}, "c5"),
                           tool_call("browser_open", {"url": a}, "c6"),
                           {"role": "assistant", "content": "done"}]
            ctx = RunContext(conversation="kept")
            await Agent(svc, svc.llm, svc.registry).run(ctx, REQUEST)
            tools = [m["content"] for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
            self.assertEqual(svc.browser.loaded, [a, b, a, a])     # c3 not loaded; c4 fresh; c6 after acting on a
            self.assertTrue(tools[2].startswith("You read this page at step 1;"))
            self.assertIn("visit 3 to this page", tools[3])         # fresh, yet the second return: nudged
            self.assertFalse(tools[5].startswith("You read this page"))
            self.assertIn("Back to the XM5", ctx.said)
            self.assertTrue(any(e["event"] == "agent.page_kept" for e in svc.audit.tail()))
            await svc.llm.aclose()


class Pages:
    def __init__(self, text):
        self.text = text

    async def navigate(self, url, tainted=False):
        return {"ok": True, "url": url, "title": "Page", "content": self.text}


class NearTrimTests(unittest.IsolatedAsyncioTestCase):
    async def test_once_before_each_trim_the_model_is_asked_to_say_its_findings_appended(self):
        with FakeOpenAI() as fake:
            svc = Svc({"models": {"context": 8192}})        # each page is about 600 tokens
            svc.browser = Pages("Price ₹1,999\n" + "[1] link \"More\"\n" * 110)
            svc.llm = LLM(fake.base_url, "test")
            svc.registry = build_registry(svc)
            fake.script = [tool_call("browser_open", {"url": f"https://shop.example/{i}"}, f"c{i}")
                           for i in range(14)] + [{"role": "assistant", "content": "done"}]
            await Agent(svc, svc.llm, svc.registry).run(RunContext(conversation="near"), REQUEST)
            asked = [sum(m["content"] == loop_module.SAY_FINDINGS for m in r["messages"]) for r in fake.requests]
            events = [e["event"] for e in svc.audit.tail(500) if e["event"] in ("agent.trim_near", "agent.trimmed")]
            # Said once, then not again until a trim has been made: one per trim cycle.
            self.assertEqual(events[:3], ["agent.trim_near", "agent.trimmed", "agent.trim_near"])
            self.assertTrue(all(a != b for a, b in zip(events, events[1:])))
            first = asked.index(1)
            self.assertEqual(asked[first - 1], 0)
            self.assertEqual(asked[first + 1], 1)                                # not said twice
            # Appended at the end: the request before it is read as it was (the cache prefix holds).
            self.assertEqual(fake.requests[first]["messages"][-1]["content"], loop_module.SAY_FINDINGS)
            before = fake.requests[first - 1]["messages"]
            self.assertEqual(fake.requests[first]["messages"][:len(before)], before)
            await svc.llm.aclose()


if __name__ == "__main__":
    unittest.main()
