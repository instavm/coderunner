import asyncio
import unittest

import httpx

from coderunner.agent import build_registry
from coderunner.gateway import admin_app, sandbox_app
from coderunner.llm import LLM
from fakes import FakeOpenAI, Svc

SANDBOX = "sandbox-token-0123456789abcdefghijklmnop"
ADMIN = "admin-token-0123456789abcdefghijklmnopq"


class FakeDaemon:
    def __init__(self, svc):
        self.svc = svc
        self.messages = []

    async def status(self):
        return {"ok": True}

    async def handle_message(self, channel, chat_id, text, origin=""):
        self.messages.append((channel, chat_id, text, origin))
        return f"echo: {text}"


class SandboxGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_and_model_proxy(self):
        with FakeOpenAI() as fake:
            fake.script = [{"role": "assistant", "content": "hi there"}]
            svc = Svc()
            svc.llm = LLM(fake.base_url, "qwen-test")
            app = sandbox_app(svc)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
                self.assertEqual((await client.get("/health")).status_code, 200)
                self.assertEqual((await client.post("/v1/chat/completions", json={})).status_code, 401)
                wrong = {"Authorization": "Bearer " + ADMIN}
                self.assertEqual((await client.post("/v1/chat/completions", json={}, headers=wrong)).status_code, 401)
                ok = {"Authorization": "Bearer " + SANDBOX}
                response = await client.post("/v1/chat/completions", headers=ok, json={
                    "model": "coderunner", "messages": [{"role": "user", "content": "hello"}]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "hi there")
                self.assertEqual(fake.requests[-1]["model"], "qwen-test")
                fake.script = [{"role": "assistant", "content": "streamed"}]
                async with client.stream("POST", "/v1/chat/completions", headers=ok, json={
                        "stream": True, "messages": [{"role": "user", "content": "x"}]}) as stream:
                    body = b"".join([chunk async for chunk in stream.aiter_raw()])
                self.assertIn(b"streamed", body)
                # The sandbox surface has no owner endpoints.
                self.assertEqual((await client.get("/api/approvals", headers=ok)).status_code, 404)
                await svc.llm.aclose()


    async def test_no_fetch_in_none_mode(self):
        svc = Svc({"sandbox": {"network": "none"}})
        svc.llm = LLM("http://127.0.0.1:9/v1", "m")
        app = sandbox_app(svc)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            response = await client.post("/v1/egress/fetch", headers={"Authorization": "Bearer " + SANDBOX},
                                         json={"url": "https://pypi.org/"})
            self.assertEqual(response.status_code, 403)
        await svc.llm.aclose()


class AdminGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc()
        self.svc.registry = build_registry(self.svc)
        self.daemon = FakeDaemon(self.svc)
        self.app = admin_app(self.svc, self.daemon)

    def client(self, base="http://127.0.0.1:8224"):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=base)

    async def test_requires_admin_token(self):
        async with self.client() as client:
            self.assertEqual((await client.get("/api/approvals")).status_code, 401)
            self.assertEqual((await client.get("/api/approvals",
                                               headers={"Authorization": "Bearer " + SANDBOX})).status_code, 401)
            response = await client.get("/api/approvals", headers={"Authorization": "Bearer " + ADMIN})
            self.assertEqual(response.status_code, 200)
            self.assertEqual((await client.get("/")).status_code, 200)

    async def test_pairing_cookie_and_csrf(self):
        admin = {"Authorization": "Bearer " + ADMIN}
        async with self.client() as client:
            code = (await client.post("/api/pair", headers=admin)).json()["code"]
            # Missing CSRF header and wrong codes are refused.
            self.assertEqual((await client.post("/pair", json={"code": code})).status_code, 400)
            self.assertEqual((await client.post("/pair", json={"code": "WRONG"},
                                                headers={"X-CodeRunner": "1"})).status_code, 403)
            paired = await client.post("/pair", json={"code": code, "name": "phone"}, headers={"X-CodeRunner": "1"})
            self.assertEqual(paired.status_code, 200)
            self.assertIn("cr_device", client.cookies)
            # Single use.
            self.assertEqual((await client.post("/pair", json={"code": code},
                                                headers={"X-CodeRunner": "1"})).status_code, 403)
            self.assertEqual((await client.get("/api/approvals")).status_code, 200)
            # Unsafe methods need the custom header and a same-origin Origin.
            self.assertEqual((await client.post("/api/chat", json={"message": "hi"})).status_code, 401)
            response = await client.post("/api/chat", json={"message": "hi"}, headers={"X-CodeRunner": "1"})
            self.assertEqual(response.json()["reply"], "echo: hi")
            cross = {"X-CodeRunner": "1", "Origin": "https://evil.example"}
            self.assertEqual((await client.post("/api/chat", json={"message": "x"}, headers=cross)).status_code, 401)

    async def test_pair_code_burns_after_failures(self):
        admin = {"Authorization": "Bearer " + ADMIN}
        async with self.client() as client:
            code = (await client.post("/api/pair", headers=admin)).json()["code"]
            for _ in range(5):
                await client.post("/pair", json={"code": "NOPE"}, headers={"X-CodeRunner": "1"})
            response = await client.post("/pair", json={"code": code}, headers={"X-CodeRunner": "1"})
            self.assertEqual(response.status_code, 403)

    async def test_tailscale_owner_restriction(self):
        self.svc.config.set("remote.owners", ["owner@example.com"])
        admin = {"Authorization": "Bearer " + ADMIN}
        async with self.client() as client:
            code = (await client.post("/api/pair", headers=admin)).json()["code"]
            await client.post("/pair", json={"code": code}, headers={"X-CodeRunner": "1"})
            self.assertEqual((await client.get("/api/approvals", headers={
                "Tailscale-User-Login": "owner@example.com"})).status_code, 200)
            self.assertEqual((await client.get("/api/approvals", headers={
                "Tailscale-User-Login": "friend@example.com"})).status_code, 401)

    async def test_mcp_clients_keep_taint(self):
        from coderunner.agent.tools import Tool
        seen = []

        async def read_page(args, ctx):
            return "page text"

        async def act(args, ctx):
            seen.append(ctx.tainted)
            return "done"

        self.svc.registry.tools["read_page"] = Tool("read_page", "", {"type": "object", "properties": {}},
                                                    read_page, untrusted=True)
        self.svc.registry.tools["act"] = Tool("act", "", {"type": "object", "properties": {}}, act)
        admin = {"Authorization": "Bearer " + ADMIN}
        async with self.client() as client:
            await client.post("/api/tools/act", headers=admin, json={"client": "c1"})
            await client.post("/api/tools/read_page", headers=admin, json={"client": "c1"})
            await client.post("/api/tools/act", headers=admin, json={"client": "c1"})
            await client.post("/api/tools/act", headers=admin, json={"client": "c2"})
        self.assertEqual(seen, [False, True, False])

    async def test_decide_approval_over_api(self):
        from coderunner.policy import Action
        admin = {"Authorization": "Bearer " + ADMIN}
        task = asyncio.create_task(self.svc.policy.check(Action("channel.send", "telegram:42", "Send hi"), wait=5))
        await asyncio.sleep(0.05)
        async with self.client() as client:
            pending = (await client.get("/api/approvals", headers=admin)).json()
            self.assertEqual(pending[0]["summary"], "Send hi")
            decided = await client.post(f"/api/approvals/{pending[0]['id']}", headers=admin,
                                        json={"allow": True, "scope": "once"})
            self.assertEqual(decided.json()["status"], "approved")
            self.assertEqual(decided.json()["decided_by"], "admin-token")
        self.assertTrue(await task)


if __name__ == "__main__":
    unittest.main()
