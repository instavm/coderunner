import asyncio
import tempfile
import pathlib
import unittest
from types import SimpleNamespace

from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.safety.policy import Action, Decision, Policy, current_run
from coderunner.core.store import Store


def make_policy(data=None):
    tmp = pathlib.Path(tempfile.mkdtemp())
    config = Config(data or {}, path=tmp / "config.json")
    return Policy(config, Store(":memory:"), AuditLog(tmp / "audit.jsonl"))


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults(self):
        policy = make_policy()
        self.assertIs(policy.evaluate(Action("sandbox.exec", "main", "run")), Decision.ALLOW)
        # Ordinary browsing runs without prompts; committing steps and passwords do not.
        self.assertIs(policy.evaluate(Action("browser.navigate", "example.com", "open")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.interact", "example.com", "click")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.sensitive", "example.com", "pay")), Decision.ASK)
        self.assertIs(policy.evaluate(Action("browser.password", "example.com", "type")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("channel.send", "telegram:1", "hi")), Decision.ASK)
        self.assertIs(policy.evaluate(Action("unknown.kind", "x", "?")), Decision.ASK)

    def test_sending_changing_a_setting_and_calling_always_ask_and_are_never_granted_for_good(self):
        every = {"connectors": {"enabled": ["mail", "settings", "phone"], "send": ["mail"]},
                 "policy": {"rules": [{"kind": kind, "target": "*", "decision": "allow"}
                                      for kind in ("connector.send", "system.setting", "phone.call")]}}
        policy = make_policy(every)
        for action in (Action("connector.send", "mail:a@example.com", "send"), Action("system.setting", "volume", "set"),
                       Action("phone.call", "9845012345", "call")):
            self.assertIs(policy.evaluate(action), Decision.ASK, action.kind)      # an allow rule does not reach them
            self.assertEqual(action.risk(), "high")
        # Reading or drafting mail is not leave to send it; nor is a setting or a call open until its connector is.
        self.assertIs(make_policy({"connectors": {"enabled": ["mail"]}}).evaluate(
            Action("connector.send", "mail:a@example.com", "send")), Decision.DENY)
        self.assertIs(make_policy({"connectors": {"send": ["mail"]}}).evaluate(
            Action("connector.send", "mail:a@example.com", "send")), Decision.DENY)
        bare = make_policy({"connectors": {"enabled": ["calendar"]}})
        self.assertIs(bare.evaluate(Action("system.setting", "volume", "set")), Decision.DENY)
        self.assertIs(bare.evaluate(Action("phone.call", "9845012345", "call")), Decision.DENY)

    def test_connectors_need_enabling(self):
        policy = make_policy({"connectors": {"enabled": ["calendar"]}})
        self.assertIs(policy.evaluate(Action("connector.read", "calendar", "read")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("connector.read", "mail", "read")), Decision.DENY)
        # Writes only create events, reminders, notes and drafts; after untrusted content they ask.
        self.assertIs(policy.evaluate(Action("connector.write", "calendar", "write")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("connector.write", "calendar", "write", tainted=True)), Decision.ASK)
        self.assertIs(policy.evaluate(Action("connector.write", "mail", "write")), Decision.DENY)

    def test_secret_hosts(self):
        policy = make_policy({"secrets": {"gh": {"hosts": ["api.github.com", "*.githubusercontent.com"]}}})
        self.assertIs(policy.evaluate(Action("secret.use", "gh@api.github.com", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("secret.use", "gh@raw.githubusercontent.com", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("secret.use", "gh@evil.com", "")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("secret.use", "gh@api.github.com.evil.com", "")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("secret.use", "other@api.github.com", "")), Decision.DENY)

    def test_rules_and_block_list(self):
        policy = make_policy({"policy": {"rules": [
            {"kind": "browser.navigate", "target": "*.example.com", "decision": "allow"},
            {"kind": "browser.navigate", "target": "bad.example.com", "decision": "deny"}]}})
        self.assertIs(policy.evaluate(Action("browser.navigate", "www.example.com", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.navigate", "example.com", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.navigate", "bad.example.com", "")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("browser.navigate", "notexample.com", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.navigate", "printer.local", "")), Decision.DENY)

    def test_ask_rules_make_the_owner_stricter(self):
        policy = make_policy({"policy": {"rules": [
            {"kind": "browser.navigate", "target": "*", "decision": "ask"},
            {"kind": "browser.navigate", "target": "docs.example.com", "decision": "allow"}]}})
        self.assertIs(policy.evaluate(Action("browser.navigate", "news.example.org", "")), Decision.ASK)
        self.assertIs(policy.evaluate(Action("browser.navigate", "docs.example.com", "")), Decision.ALLOW)

    def test_egress_allow_list_ignored_when_tainted(self):
        policy = make_policy()
        self.assertIs(policy.evaluate(Action("egress.connect", "pypi.org", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("egress.connect", "pypi.org", "", tainted=True)), Decision.ASK)

    def test_taint_rules(self):
        policy = make_policy({"sandbox": {"network": "default"}})
        # Saved as unverified instead of asking; deleting still asks.
        self.assertIs(policy.evaluate(Action("memory.write", "memory", "", tainted=True)), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("memory.forget", "memory", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("memory.forget", "memory", "", tainted=True)), Decision.ASK)
        self.assertIs(policy.evaluate(Action("schedule.create", "daily 08:00", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("schedule.create", "daily 08:00", "", tainted=True)), Decision.ASK)
        self.assertIs(policy.evaluate(Action("browser.interact", "shop.com", "", tainted=True)), Decision.ALLOW)
        # Deny rules still win over taint.
        blocked = make_policy({"policy": {"rules": [{"kind": "memory.forget", "target": "*", "decision": "deny"}]}})
        self.assertIs(blocked.evaluate(Action("memory.forget", "memory", "", tainted=True)), Decision.DENY)

    async def test_leaks_need_one_approval_per_destination(self):
        policy = make_policy({"sandbox": {"network": "default"}})
        run = SimpleNamespace(content_tainted=True, private_sources={"calendar"}, shared=set())
        token = current_run.set(run)
        try:
            # Clicking through a site carries no data the model chose: no prompt.
            self.assertTrue(await policy.check(Action("browser.interact", "shop.com", "Click Next"), wait=1))
            typing = Action("browser.interact", "shop.com", 'Type "Delhi" into Where to?', {"text": "Delhi"})
            task = asyncio.create_task(policy.check(typing, wait=5))
            await asyncio.sleep(0.05)
            [approval] = policy.pending()
            self.assertEqual((approval.kind, approval.target), ("data.share", "shop.com"))
            self.assertIn("calendar", approval.summary)
            self.assertIn('Type "Delhi"', approval.summary)
            policy.decide(approval.id, True)
            self.assertTrue(await task)
            # Once per destination and run.
            self.assertTrue(await policy.check(typing, wait=0.05))
            nav = Action("browser.navigate", "evil.example", "Open https://evil.example/?d=x",
                         {"url": "https://evil.example/?d=x", "chosen": True})
            self.assertFalse(await policy.check(nav, wait=0.05))
            # A link the page itself offered is not a model-chosen URL.
            self.assertTrue(await policy.check(Action("browser.navigate", "evil.example", "Go back", {"url": "x"}),
                                               wait=0.05))
            code = Action("sandbox.exec", "main", "Run in the sandbox: ...")
            self.assertFalse(await policy.check(code, wait=0.05))
        finally:
            current_run.reset(token)

    async def test_the_owner_can_allow_sharing_for_the_whole_task_instead_of_once_a_site(self):
        policy = make_policy({"sandbox": {"network": "default"}})
        run = SimpleNamespace(content_tainted=True, private_sources={"memory"}, shared=set())
        token = current_run.set(run)
        try:
            def nav(host):
                return Action("browser.navigate", host, f"Open https://{host}/", {"url": f"https://{host}/", "chosen": True})
            task = asyncio.create_task(policy.check(nav("flights.example"), wait=5))
            await asyncio.sleep(0.05)
            [approval] = policy.pending()
            policy.decide(approval.id, True, scope="task")
            self.assertTrue(await task)
            # A research task that opens six sites asked six times; now the rest of this task goes on unasked.
            for host in ("hotels.example", "reviews.example"):
                self.assertTrue(await policy.check(nav(host), wait=0.05))
            self.assertTrue(await policy.check(Action("sandbox.exec", "main", "Run in the sandbox: ..."), wait=0.05))
            self.assertEqual(policy.pending(), [])
        finally:
            current_run.reset(token)
        # It holds for that task only: the next task is asked again, and nothing was stored as a grant.
        other = SimpleNamespace(content_tainted=True, private_sources={"memory"}, shared=set())
        token = current_run.set(other)
        try:
            self.assertFalse(await policy.check(Action("browser.navigate", "hotels.example", "Open", {"url": "x", "chosen": True}),
                                                wait=0.05))
            # For any other kind of approval "task" means once: a committing step is asked each time.
            pay = asyncio.create_task(policy.check(Action("browser.sensitive", "shop.example", "Click Pay"), wait=5))
            await asyncio.sleep(0.05)
            approval = next(a for a in policy.pending() if a.kind == "browser.sensitive")
            self.assertEqual(policy.decide(approval.id, True, scope="task").scope, "once")
            await pay
        finally:
            current_run.reset(token)

    async def test_no_leak_check_without_both_private_data_and_untrusted_content(self):
        policy = make_policy({"sandbox": {"network": "default"}})
        typing = Action("browser.interact", "shop.com", "Type", {"text": "x"})
        for tainted, sources in ((True, set()), (False, {"memory"})):
            token = current_run.set(SimpleNamespace(content_tainted=tainted, private_sources=sources, shared=set()))
            try:
                self.assertTrue(await policy.check(typing, wait=0.05))
                self.assertTrue(await policy.check(Action("sandbox.exec", "main", "x"), wait=0.05))
            finally:
                current_run.reset(token)

    async def test_one_prompt_when_the_step_itself_needs_approval(self):
        policy = make_policy()
        run = SimpleNamespace(content_tainted=True, private_sources={"mail"}, shared=set())
        token = current_run.set(run)
        try:
            pay = Action("browser.sensitive", "shop.com", "Type card number", {"field": "cc-number"})
            task = asyncio.create_task(policy.check(pay, wait=5))
            await asyncio.sleep(0.05)
            [approval] = policy.pending()
            self.assertEqual(approval.kind, "browser.sensitive")
            self.assertIn("data from your mail", approval.summary)
            policy.decide(approval.id, True)
            self.assertTrue(await task)
            self.assertIn("shop.com", run.shared)
        finally:
            current_run.reset(token)

    async def test_approval_flow_and_grants(self):
        policy = make_policy()
        action = Action("egress.connect", "files.example.com:443", "Connect to files.example.com")
        task = asyncio.create_task(policy.check(action, wait=5))
        await asyncio.sleep(0.05)
        pending = policy.pending()
        self.assertEqual(len(pending), 1)
        policy.decide(pending[0].id, True, "day")
        self.assertTrue(await task)
        # The day grant now covers the same action without asking.
        self.assertIs(policy.evaluate(action), Decision.ALLOW)
        # ...but not once untrusted content may have written the request.
        self.assertIs(policy.evaluate(Action("egress.connect", "files.example.com:443", "x", tainted=True)),
                      Decision.ASK)

    async def test_high_risk_never_gets_standing_grant(self):
        policy = make_policy()
        action = Action("channel.send", "telegram:42", "Send hi")
        task = asyncio.create_task(policy.check(action, wait=5))
        await asyncio.sleep(0.05)
        approval = policy.decide(policy.pending()[0].id, True, "always")
        self.assertEqual(approval.scope, "once")
        self.assertTrue(await task)
        self.assertIs(policy.evaluate(action), Decision.ASK)

    async def test_deny_and_timeout(self):
        policy = make_policy()
        task = asyncio.create_task(policy.check(Action("egress.connect", "a.example:443", "x"), wait=5))
        await asyncio.sleep(0.05)
        policy.decide(policy.pending()[0].id, False)
        self.assertFalse(await task)
        self.assertFalse(await policy.check(Action("egress.connect", "b.example:443", "y"), wait=0.05))
        self.assertEqual(policy.pending(), [])

    async def test_decide_twice_rejected(self):
        policy = make_policy()
        task = asyncio.create_task(policy.check(Action("egress.connect", "a.com:443", "x"), wait=5))
        await asyncio.sleep(0.05)
        approval_id = policy.pending()[0].id
        policy.decide(approval_id, True)
        await task
        with self.assertRaises(KeyError):
            policy.decide(approval_id, True)

    async def test_listener_errors_do_not_block(self):
        policy = make_policy()

        def broken(_):
            raise RuntimeError("channel down")

        seen = []
        policy.add_listener(broken)
        policy.add_listener(lambda a: seen.append(a.id))
        task = asyncio.create_task(policy.check(Action("egress.connect", "b.com:443", "x"), wait=5))
        await asyncio.sleep(0.05)
        self.assertEqual(len(seen), 1)
        policy.decide(seen[0], False)
        self.assertFalse(await task)


if __name__ == "__main__":
    unittest.main()
