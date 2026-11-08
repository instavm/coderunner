"""Credential boundary for the MCP and REST interfaces.

Host/Origin validation is complementary; a chosen Host header is not identity.
The credential is supplied by the operator. This module never generates one.
"""

import hmac
import os

from starlette.responses import JSONResponse


class BearerAuthentication:
    def __init__(self, app, token_provider=None):
        self.app = app
        self.token_provider = token_provider or (lambda: os.getenv("CODERUNNER_API_TOKEN", ""))

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        if scope["type"] == "http" and scope["path"] == "/health" and scope["method"] in {"GET", "HEAD"}:
            return await self.app(scope, receive, send)
        expected = self.token_provider()
        configured = len(expected.encode()) >= 32 and not any(c.isspace() for c in expected)
        headers = [v for k, v in scope["headers"] if k.lower() == b"authorization"]
        supplied = headers[0].split(b" ", 1) if len(headers) == 1 else []
        authorized = (
            configured and len(supplied) == 2 and supplied[0].lower() == b"bearer"
            and hmac.compare_digest(supplied[1], expected.encode())
        )
        if not authorized:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            response = JSONResponse(
                {"error": "Authentication required" if configured else "Server authentication is not configured"},
                status_code=401 if configured else 503,
                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
            )
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)
