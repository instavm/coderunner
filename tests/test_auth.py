import unittest

import httpx
from starlette.responses import JSONResponse

from auth import BearerAuthentication


class AuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.token = "fixture-credential-not-for-use-123456789"

        async def downstream(scope, receive, send):
            self.calls.append(scope["path"])
            await JSONResponse({"ok": True})(scope, receive, send)

        self.app = BearerAuthentication(downstream, lambda: self.token)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://localhost")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_commands_require_exact_bearer_credential(self):
        for path in ("/mcp", "/execute", "/v1/sessions/session", "/v1/browser/interactions/navigate"):
            for auth in (None, "Basic " + self.token, "Bearer wrong", "Bearer " + self.token + " "):
                result = await self.client.post(path, headers={"Authorization": auth} if auth else {})
                self.assertEqual(result.status_code, 401)
        self.assertEqual(self.calls, [])
        response = await self.client.post("/execute", headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.calls, ["/execute"])

    async def test_missing_or_weak_configuration_fails_closed(self):
        for value in ("", "short", " " + "x" * 40):
            self.token = value
            result = await self.client.post("/execute", headers={"Authorization": "Bearer " + value})
            self.assertEqual(result.status_code, 503)
        self.assertEqual(self.calls, [])
        self.assertEqual((await self.client.get("/health")).status_code, 200)
        self.assertEqual((await self.client.post("/health")).status_code, 503)

    async def test_duplicate_auth_and_rotated_token_are_rejected(self):
        previous = self.token
        headers = [("Authorization", "Bearer " + previous)] * 2
        self.assertEqual((await self.client.post("/execute", headers=headers)).status_code, 401)
        self.token = "replacement-fixture-credential-123456789"
        self.assertEqual((await self.client.post("/execute", headers={"Authorization": "Bearer " + previous})).status_code, 401)

    async def test_websocket_cannot_bypass_auth(self):
        sent = []

        async def send(value):
            sent.append(value)

        await self.app({"type": "websocket", "path": "/mcp", "headers": []}, None, send)
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])
