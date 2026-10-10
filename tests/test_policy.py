import asyncio
import tempfile
import pathlib
import unittest

from coderunner.audit import AuditLog
from coderunner.config import Config
from coderunner.policy import Action, Decision, Policy
from coderunner.store import Store


def make_policy(data=None):
    tmp = pathlib.Path(tempfile.mkdtemp())
    config = Config(data or {}, path=tmp / "config.json")
    return Policy(config, Store(":memory:"), AuditLog(tmp / "audit.jsonl"))


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults(self):
        policy = make_policy()
        self.assertIs(policy.evaluate(Action("sandbox.exec", "main", "run")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("browser.navigate", "example.com", "open")), Decision.ASK)
        self.assertIs(policy.evaluate(Action("browser.password", "example.com", "type")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("unknown.kind", "x", "?")), Decision.ASK)

    def test_connectors_need_enabling(self):
        policy = make_policy({"connectors": {"enabled": ["calendar"]}})
        self.assertIs(policy.evaluate(Action("connector.read", "calendar", "read")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("connector.read", "mail", "read")), Decision.DENY)
        self.assertIs(policy.evaluate(Action("connector.write", "calendar", "write")), Decision.ASK)
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
        self.assertIs(policy.evaluate(Action("browser.navigate", "notexample.com", "")), Decision.ASK)
        self.assertIs(policy.evaluate(Action("browser.navigate", "printer.local", "")), Decision.DENY)

    def test_egress_allow_list_ignored_when_tainted(self):
        policy = make_policy()
        self.assertIs(policy.evaluate(Action("egress.connect", "pypi.org", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("egress.connect", "pypi.org", "", tainted=True)), Decision.ASK)

    def test_taint_rules(self):
        policy = make_policy({"sandbox": {"network": "default"}})
        self.assertIs(policy.evaluate(Action("memory.write", "memory", "")), Decision.ALLOW)
        self.assertIs(policy.evaluate(Action("memory.write", "memory", "", tainted=True)), Decision.ASK)
        self.assertIs(policy.evaluate(Action("memory.forget", "memory", "", tainted=True)), Decision.ASK)
        self.assertIs(policy.evaluate(Action("sandbox.exec", "main", "")), Decision.ALLOW)
        # Open-network sandbox code after untrusted input is an exfiltration path.
        self.assertIs(policy.evaluate(Action("sandbox.exec", "main", "", tainted=True)), Decision.ASK)
        guarded = make_policy({"sandbox": {"network": "proxy"}})
        self.assertIs(guarded.evaluate(Action("sandbox.exec", "main", "", tainted=True)), Decision.ALLOW)
        # Deny rules still win over taint.
        blocked = make_policy({"policy": {"rules": [{"kind": "memory.write", "target": "*", "decision": "deny"}]}})
        self.assertIs(blocked.evaluate(Action("memory.write", "memory", "", tainted=True)), Decision.DENY)

    def test_tainted_navigation(self):
        policy = make_policy({"browser": {"allow": ["docs.example.com"]}})
        nav = lambda host, url, tainted: Action("browser.navigate", host, "", {"url": url}, tainted=tainted)  # noqa: E731
        self.assertIs(policy.evaluate(nav("docs.example.com", "https://docs.example.com/a", False)), Decision.ALLOW)
        self.assertIs(policy.evaluate(nav("docs.example.com", "https://docs.example.com/a", True)), Decision.ALLOW)
        # A query string can carry data out after untrusted content was read.
        self.assertIs(policy.evaluate(nav("docs.example.com", "https://docs.example.com/f?entry=x", True)),
                      Decision.ASK)
        self.assertIs(policy.evaluate(nav("other.com", "https://other.com/", True)), Decision.ASK)

    async def test_approval_flow_and_grants(self):
        policy = make_policy()
        action = Action("browser.interact", "shop.com", "Click Add to cart")
        task = asyncio.create_task(policy.check(action, wait=5))
        await asyncio.sleep(0.05)
        pending = policy.pending()
        self.assertEqual(len(pending), 1)
        policy.decide(pending[0].id, True, "day")
        self.assertTrue(await task)
        # The day grant now covers the same action without asking.
        self.assertIs(policy.evaluate(action), Decision.ALLOW)
        # ...but not once the run has read untrusted content.
        self.assertIs(policy.evaluate(Action("browser.interact", "shop.com", "x", tainted=True)), Decision.ASK)

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
        task = asyncio.create_task(policy.check(Action("schedule.create", "daily 08:00", "x"), wait=5))
        await asyncio.sleep(0.05)
        policy.decide(policy.pending()[0].id, False)
        self.assertFalse(await task)
        self.assertFalse(await policy.check(Action("schedule.create", "daily 09:00", "y"), wait=0.05))
        self.assertEqual(policy.pending(), [])

    async def test_decide_twice_rejected(self):
        policy = make_policy()
        task = asyncio.create_task(policy.check(Action("browser.navigate", "a.com", "x"), wait=5))
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
        task = asyncio.create_task(policy.check(Action("browser.navigate", "b.com", "x"), wait=5))
        await asyncio.sleep(0.05)
        self.assertEqual(len(seen), 1)
        policy.decide(seen[0], False)
        self.assertFalse(await task)


if __name__ == "__main__":
    unittest.main()
