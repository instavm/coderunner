"""A long task's conversation compacted by a summary the model writes (agent.compaction): when, where it is cut, what
is always kept as it was, the summary carried forward, the extractive trim when the summary fails, one retry when
the runtime says it is full, and what the audit and the trace show."""

import asyncio
import copy
import json
import re
import unittest

from coderunner.agent import Agent, RunContext, build_registry
from coderunner.agent import compaction
from coderunner.agent import loop as loop_module
from coderunner.agent.digest import is_cut
from coderunner.core.llm import LLMError
from coderunner.server import trace
from fakes import Svc

REQUEST = "Find the price of each of these thirty kettles and tell me the cheapest"
FILLER = ("Brushed steel body with a soft-close lid and a hidden element.", '[{r}] link "Customer questions"',
          "Free returns within thirty days of delivery.", '[{r}] button "Add to Wish List"',
          "Customers who viewed this also viewed similar items.")


def price(n: int) -> str:
    return f"₹{1000 + 137 * n:,}"


def page_text(n: int, chars: int = 3500) -> str:
    lines = [f"# Kettle model K{n}", f"Price {price(n)}", "4.2 out of 5 stars"]
    r = 40
    while sum(len(line) + 1 for line in lines) < chars:
        lines.append(FILLER[r % len(FILLER)].format(r=r))
        r += 1
    return "\n".join(lines)


class Pages:
    async def navigate(self, url, tainted=False):
        n = int(url.rsplit("/", 1)[1])
        return {"ok": True, "url": url, "title": f"Kettle K{n}", "content": page_text(n)}


def opened(n: int) -> dict:
    return {"content": "", "tool_calls": [{"id": f"c{n}", "type": "function", "function": {
        "name": "browser_open", "arguments": json.dumps({"url": f"https://shop.example/k/{n}"})}}]}


def faithful(messages: list[dict], number: int) -> str:
    """A summary as a model that does its job writes it: the prices it can see, the earlier summary's included."""
    seen = []
    for m in messages:
        for found in re.findall(r"₹[\d,]+", str(m.get("content") or "")):
            if found not in seen:
                seen.append(found)
    return (f"## Goal\nThe cheapest kettle (summary {number}).\n## Constraints & Preferences\n(none)\n## Progress\n"
            f"### Done\n- Opened kettle pages.\n### In progress\n- More pages.\n### Blocked\n(none)\n"
            f"## Key Decisions\n(none)\n## Key facts found\n" + "\n".join(f"- {p}" for p in seen)
            + "\n## Next Steps\n- Open the rest.")


class Model:
    """The model, in process. Step requests take their reply from `script` (a reply, or a callable(messages) ->
    reply, which may raise); summary requests are answered by `summary(messages, number)` (text, or raise)."""
    remote, provider = False, ""

    def __init__(self, context: int, script: list, summary=faithful):
        self.context, self.script, self.summary = context, script, summary
        self.requests: list[dict] = []          # every request: {"messages", "summary": bool, **options}

    async def chat(self, messages, tools=None, think=None, **options):
        tokens = int(sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str)) / 3.5)
        is_summary = messages[-1].get("content") == compaction.SUMMARIZE
        self.requests.append({"messages": copy.deepcopy(messages), "summary": is_summary, "think": think,
                              "tools": bool(tools), **options})
        if is_summary:
            number = sum(r["summary"] for r in self.requests)
            text = self.summary(messages, number)
            if asyncio.iscoroutine(text):
                text = await text
            return {"content": text, "tool_calls": [], "finish_reason": "stop",
                    "usage": {"prompt_tokens": tokens, "completion_tokens": len(text) // 4}}
        item = self.script.pop(0) if self.script else {"content": "K0 is the cheapest.", "tool_calls": []}
        reply = item(messages) if callable(item) else item
        return {"finish_reason": "stop", "usage": {"prompt_tokens": tokens, "completion_tokens": 30}, **reply}

    def steps(self) -> list[list[dict]]:
        return [r["messages"] for r in self.requests if not r["summary"]]

    def summaries(self) -> list[dict]:
        return [r for r in self.requests if r["summary"]]


def agent_for(model: Model, config: dict | None = None) -> tuple[Agent, Svc]:
    svc = Svc(config or {})
    svc.browser = Pages()
    svc.llm = model
    svc.registry = build_registry(svc)
    return Agent(svc, model, svc.registry), svc


def compacted(svc) -> list[dict]:
    return [e for e in svc.audit.tail(1000) if e["event"] == "agent.compacted"]


def paired(messages: list[dict]) -> bool:
    """Every tool result follows the call that asked for it."""
    asked: set[str] = set()
    for m in messages:
        for call in m.get("tool_calls") or []:
            asked.add(call["id"])
        if m.get("role") == "tool" and m.get("tool_call_id") not in asked:
            return False
    return True


class CutPointTests(unittest.TestCase):
    def conversation(self) -> list[dict]:
        messages = [{"role": "system", "content": "s" * 3000}, {"role": "user", "content": REQUEST}]
        for n in range(10):
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"a{n}", "type": "function", "function": {"name": "browser_open", "arguments": "{}"}},
                {"id": f"b{n}", "type": "function", "function": {"name": "browser_open", "arguments": "{}"}}]})
            messages += [{"role": "tool", "tool_call_id": f"a{n}", "content": "x" * 3500},
                         {"role": "tool", "tool_call_id": f"b{n}", "content": "y" * 3500}]
            if n == 4:
                messages.append({"role": "user", "content": loop_module.PLAN_REMINDER + "1. [doing] Prices"})
        return messages

    def test_the_tail_starts_at_a_message_the_model_or_owner_wrote_and_holds_about_keep_tokens(self):
        messages = self.conversation()
        for keep in range(100, 12000, 250):
            cut = compaction.cut_point(messages, 2, keep)
            self.assertIsNotNone(cut)
            self.assertIn(messages[cut]["role"], ("assistant", "user"))         # never a tool's result
            self.assertTrue(paired(messages[cut:]))
            self.assertGreaterEqual(compaction.tokens_of(messages[cut:]), min(keep, compaction.tokens_of(messages[2:])) - 1)

    def test_the_latest_step_is_always_kept_and_nothing_to_summarize_is_none(self):
        messages = self.conversation()
        last = max(i for i, m in enumerate(messages) if m.get("tool_calls"))
        self.assertEqual(compaction.cut_point(messages, 2, 1), last)          # a tiny keep still holds the step whole
        self.assertIsNone(compaction.cut_point(messages, 2, 10 ** 6))          # all of it kept: nothing to summarize
        self.assertIsNone(compaction.cut_point(messages[:5], 2, 1))            # only one step

    def test_threshold_and_tail_scale_with_the_room(self):
        self.assertEqual(compaction.threshold(32768), compaction.COMPACT_AT * 32768)
        self.assertEqual(compaction.keep_tokens(32768), int(compaction.KEEP_TAIL * 32768))
        self.assertEqual(compaction.keep_tokens(65536), int(compaction.KEEP_TAIL * 65536))
        self.assertEqual(compaction.keep_tokens(200000), compaction.KEEP_MOST)  # a large context re-reads little

    def test_a_reply_stopped_for_length_well_before_its_limit_means_the_context_is_full(self):
        full = {"finish_reason": "length", "usage": {"prompt_tokens": 16000, "completion_tokens": 40}}
        self.assertTrue(compaction.ran_out_of_room(full, 2048))
        self.assertFalse(compaction.ran_out_of_room({**full, "usage": {"prompt_tokens": 9000, "completion_tokens": 2048}}, 2048))
        self.assertFalse(compaction.ran_out_of_room({**full, "finish_reason": "stop"}, 2048))
        self.assertFalse(compaction.ran_out_of_room(full, 0))
        for said in ("Model runtime returned 400: {\"error\":{\"message\":\"the request exceeds the available context "
                     "size, try increasing it\",\"type\":\"exceed_context_size_error\"}}",
                     "This model's maximum context length is 32768 tokens", "prompt is too long: 210000 tokens"):
            self.assertTrue(compaction.OVERFLOW.search(said), said)
        self.assertFalse(compaction.OVERFLOW.search("Model runtime returned 500: Compute error."))

    def test_a_summary_must_have_its_sections_and_some_length(self):
        self.assertEqual(compaction.usable({"content": faithful([], 1)})[1], "")
        self.assertEqual(compaction.usable({"content": "Done."}), ("", "too_short"))
        self.assertEqual(compaction.usable({"content": "I looked at many pages today. " * 20}), ("", "junk"))
        self.assertEqual(compaction.usable({"content": faithful([], 1), "tool_calls": [{"id": "x"}]}), ("", "junk"))


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_past_its_threshold_the_conversation_becomes_request_summary_and_latest_steps(self):
        model = Model(16384, [opened(n) for n in range(30)])
        agent, svc = agent_for(model)
        await agent.run(RunContext(conversation="c1"), REQUEST)
        steps, summaries, events = model.steps(), model.summaries(), compacted(svc)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(len(summaries), len(events))
        head = steps[0]
        for event in events:
            self.assertEqual(event["reason"], "threshold")
            self.assertGreater(event["before"], compaction.threshold(16384))
            self.assertLess(event["after"], 0.6 * 16384)
            self.assertLessEqual(event["tail"], 0.5 * 16384)
            self.assertGreater(event["summary"], 0)
            self.assertNotIn("fallback", event)
        rewrites = 0
        for i, request in enumerate(model.requests):
            if request["summary"]:
                # Appended to the conversation as the last request had it: only the instruction is new.
                before = model.requests[i - 1]["messages"]
                self.assertEqual(request["messages"][:-1][:len(before)], before)
                self.assertEqual((request["max_tokens"], request["tool_choice"], request["think"], request["tools"]),
                                 (compaction.SUMMARY_TOKENS, "none", False, True))
                after = model.requests[i + 1]["messages"]
                self.assertEqual(after[:len(head)], head)                    # instructions and request as they were
                self.assertTrue(compaction.is_summary(after[len(head)]))
                self.assertTrue(paired(after))
                continue
            if i and not model.requests[i - 1]["summary"]:
                previous = model.requests[i - 1]["messages"]
                # Between compactions nothing written before is changed: each request extends the one before.
                self.assertEqual(request["messages"][:len(previous)], previous)
            elif i:
                rewrites += 1
        self.assertEqual(rewrites, len(events))
        last = steps[-1]
        self.assertEqual(sum(compaction.is_summary(m) for m in last), 1)        # the latest summary only
        text = "\n".join(str(m.get("content")) for m in last)
        for n in range(30):
            self.assertIn(price(n), text)                                       # in the summary or the tail
        tail = [m for m in last[len(head):] if m.get("role") == "tool"]
        self.assertFalse(any(is_cut(m["content"]) for m in tail))               # the kept steps are verbatim

    async def test_the_summary_is_carried_forward_and_refined(self):
        model = Model(16384, [opened(n) for n in range(30)])
        agent, svc = agent_for(model)
        await agent.run(RunContext(conversation="c2"), REQUEST)
        summaries = model.summaries()
        self.assertGreaterEqual(len(summaries), 2)
        for number, request in enumerate(summaries[1:], 1):
            earlier = [m for m in request["messages"] if compaction.is_summary(m)]
            self.assertEqual(len(earlier), 1)
            self.assertIn(f"(summary {number})", earlier[0]["content"])
            self.assertIn("If an earlier summary is above, start from it", request["messages"][-1]["content"])

    async def test_the_request_plan_and_the_owners_words_are_kept_verbatim(self):
        plan = {"content": "", "tool_calls": [{"id": "p", "type": "function", "function": {
            "name": "plan_update", "arguments": json.dumps({"steps": [
                {"step": "Open each kettle page", "status": "doing"}, {"step": "Pick the cheapest"}]})}}]}
        model = Model(16384, [plan] + [opened(n) for n in range(24)])
        agent, svc = agent_for(model, {"owner": {"name": "Asha"}})
        ctx = RunContext(conversation="c3")
        ctx.inbox.append({"id": "s1", "text": "Only kettles under 2 litres, please"})
        await agent.run(ctx, REQUEST)
        self.assertTrue(compacted(svc))
        after = [model.requests[i + 1]["messages"] for i, r in enumerate(model.requests) if r["summary"]]
        steer = loop_module.STEER.format(owner="Asha", text="Only kettles under 2 litres, please")
        for messages in after:
            contents = [m.get("content") for m in messages]
            self.assertIn(REQUEST, contents)
            self.assertIn(steer, contents)
            self.assertIn(loop_module.PLAN_REMINDER + loop_module.render_plan(ctx.plan), contents)
            summary = next(i for i, m in enumerate(messages) if compaction.is_summary(m))
            self.assertLess(contents.index(REQUEST), summary)

    async def test_a_failed_summary_falls_back_to_the_trim_in_one_go_and_stops_asking_after_two(self):
        def broken(messages, number):
            raise LLMError("Model runtime returned 500: Compute error.")
        model = Model(16384, [opened(n) for n in range(30)], summary=broken)
        agent, svc = agent_for(model)
        reply = await agent.run(RunContext(conversation="c4"), REQUEST)
        self.assertEqual(reply, "K0 is the cheapest.")
        events = compacted(svc)
        self.assertEqual(len(model.summaries()), compaction.SUMMARY_FAILS)
        self.assertEqual([e["fallback"] for e in events[:2]], ["failed", "failed"])
        self.assertTrue(all(e.get("fallback") for e in events))
        trimmed = [e for e in events if e.get("results")]
        self.assertTrue(trimmed and all(e["after"] < e["before"] for e in trimmed))
        last = model.steps()[-1]
        self.assertFalse(any(compaction.is_summary(m) for m in last))
        self.assertTrue(any(m.get("role") == "tool" and is_cut(m["content"]) for m in last))     # key lines kept

    async def test_a_summary_that_is_junk_too_short_or_too_slow_falls_back(self):
        async def slow(messages, number):
            await asyncio.sleep(1)
            return faithful(messages, number)
        for summary, why, config in ((lambda m, n: "Done.", "too_short", {}),
                                     (lambda m, n: "I read the kettle pages. " * 30, "junk", {}),
                                     (slow, "timeout", {"agent": {"summary_timeout": 0.1}})):
            model = Model(16384, [opened(n) for n in range(14)], summary=summary)
            agent, svc = agent_for(model, config)
            await agent.run(RunContext(conversation="c5"), REQUEST)
            self.assertEqual(compacted(svc)[0]["fallback"], why)
            self.assertFalse(any(compaction.is_summary(m) for m in model.steps()[-1]))

    async def test_the_runtime_saying_its_context_is_full_compacts_and_retries_once(self):
        def full(messages):
            raise LLMError('Model runtime returned 400: {"error":{"code":400,"message":"the request exceeds the '
                           'available context size, try increasing it","type":"exceed_context_size_error"}}')
        # Four pages are well under the threshold, so only the runtime's word makes it compact.
        model = Model(65536, [opened(n) for n in range(8)] + [full, opened(8)])
        agent, svc = agent_for(model)
        reply = await agent.run(RunContext(conversation="c6"), REQUEST)
        self.assertEqual(reply, "K0 is the cheapest.")
        events = compacted(svc)
        self.assertEqual([e["reason"] for e in events], ["overflow"])
        refused = model.steps()[8]
        summary = model.summaries()[0]["messages"]
        # Asked of the part before the cut (the whole would not fit either), then the step sent again, shorter.
        self.assertLess(len(summary) - 1, len(refused))
        self.assertEqual(summary[:-1], refused[:len(summary) - 1])
        retried = model.steps()[9]
        self.assertTrue(any(compaction.is_summary(m) for m in retried))
        self.assertLess(len(retried), len(refused))
        # Full again on the retry: the task stops in plain words, not in a loop.
        model = Model(65536, [opened(n) for n in range(8)] + [full, full, full])
        agent, svc = agent_for(model)
        reply = await agent.run(RunContext(conversation="c7"), REQUEST)
        self.assertIn("stopped answering", reply)
        self.assertEqual(len(compacted(svc)), 1)

    async def test_a_reply_cut_by_a_full_context_compacts_and_retries(self):
        cut = {"content": "", "tool_calls": [], "finish_reason": "length",
               "usage": {"prompt_tokens": 65000, "completion_tokens": 12}}
        model = Model(65536, [opened(n) for n in range(8)] + [cut, opened(8)])
        agent, svc = agent_for(model)
        reply = await agent.run(RunContext(conversation="c8"), REQUEST)
        self.assertEqual(reply, "K0 is the cheapest.")
        self.assertEqual([e["reason"] for e in compacted(svc)], ["length"])
        self.assertFalse(any(e["event"] == "agent.reply_cut" for e in svc.audit.tail(500)))


class TraceTests(unittest.TestCase):
    def test_the_trace_shows_each_compaction(self):
        task = {"id": "t1", "status": "done", "created": 1000.0, "finished": 1100.0, "prompt": REQUEST, "reply": "K0"}
        records = [{"ts": 1010.0, "event": "agent.compacted", "task": "t1", "step": 12, "reason": "threshold",
                    "room": 32768, "before": 24800, "after": 13100, "tail": 10200, "summary": 540, "summarized": 18,
                    "seconds": 21.4},
                   {"ts": 1050.0, "event": "agent.compacted", "task": "t1", "step": 30, "reason": "overflow",
                    "room": 32768, "before": 25000, "after": 14000, "fallback": "timeout", "results": 9,
                    "seconds": 60.1}]
        info = trace._task(1, task, [], records)
        entries = [e for e in info["timeline"] if e["kind"] == "compaction"]
        self.assertEqual([e["reason"] for e in entries], ["threshold", "overflow"])
        self.assertEqual(entries[1]["fallback"], "timeout")
        self.assertEqual(info["totals"]["compactions"], 2)
        md = "\n".join(trace._md_task(info))
        self.assertIn("**Compacted** (step 12, threshold): 24,800 → 13,100 tokens · summary 540 tokens, last steps "
                      "kept verbatim 10,200 · 21.4 s", md)
        self.assertIn("**Compacted** (step 30, overflow): 25,000 → 14,000 tokens · fallback trim of 9 results (timeout)", md)
        self.assertIn("· 2 compactions", md)


if __name__ == "__main__":
    unittest.main()
