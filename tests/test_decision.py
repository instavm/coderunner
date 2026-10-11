import json
import pathlib
import tempfile
import unittest

import httpx

from coderunner.audit import AuditLog
from coderunner.config import Config
from coderunner.decision import ActionCheck, describe


class ActionCheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.audit = AuditLog(pathlib.Path(tempfile.mkdtemp()) / "audit.jsonl")
        self.requests = []

    def check(self, handler, **kwargs):
        def record(request):
            self.requests.append(json.loads(request.content))
            return handler(request)
        return ActionCheck("http://ollama/v1", "tev1:4b", audit=self.audit, transport=httpx.MockTransport(record),
                           **kwargs)

    async def test_systemone_probabilities_and_threshold(self):
        answers = iter([{"safe": 0.2, "commit": 0.7, "unclear": 0.1}, {"safe": 0.4, "commit": 0.5, "unclear": 0.1}])
        check = self.check(lambda r: httpx.Response(200, json={"answers": {"risk": {"probabilities": next(answers)}}}))
        state = describe("shop.example", "/checkout", "click", "button", "Continue", nearby="Total $899")
        self.assertTrue(await check.commits(state))
        self.assertFalse(await check.commits(state))            # 0.6 is under the default 0.7
        self.assertEqual(self.requests[0]["model"], "tev1:4b")
        self.assertIn("Total $899", self.requests[0]["state"])
        self.assertEqual(set(self.requests[0]["questions"]["risk"]["criteria"]), {"safe", "commit", "unclear"})
        events = [e for e in self.audit.tail() if e["event"] == "decision.check"]
        self.assertEqual([e["commits"] for e in events], [True, False])
        await check.aclose()

    async def test_failures_ask_the_owner(self):
        check = self.check(lambda r: httpx.Response(500, text="model not loaded"))
        self.assertTrue(await check.commits("Site: x"))
        self.assertIn("error", [e for e in self.audit.tail() if e["event"] == "decision.check"][-1])
        await check.aclose()

    async def test_chat_models_answer_with_one_letter(self):
        replies = iter(["B", '"A"'])
        check = self.check(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]}),
                           api="chat")
        self.assertTrue(await check.commits("Site: x"))
        self.assertFalse(await check.commits("Site: x"))
        self.assertEqual(self.requests[0]["temperature"], 0)
        await check.aclose()

    def test_from_config(self):
        self.assertIsNone(ActionCheck.from_config(Config({}, path=None)))
        # Without a decision model, the assistant's own model judges.
        chat_only = Config({"models": {"base_url": "http://x/v1", "chat": "qwen"}}, path=None)
        self.assertEqual((ActionCheck.from_config(chat_only).model, ActionCheck.from_config(chat_only).api),
                         ("qwen", "chat"))
        own = Config({"models": {"base_url": "http://x/v1", "chat": "qwen", "decision": "tev1:4b",
                                 "decision_threshold": 0.9}}, path=None)
        check = ActionCheck.from_config(own)
        self.assertEqual((check.api, check.threshold), ("systemone", 0.9))


if __name__ == "__main__":
    unittest.main()
