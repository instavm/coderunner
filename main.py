"""Unified application combining MCP and REST API servers"""

from fastapi import FastAPI
from server import app as mcp_app  # FastMCP streamable_http_app
from api.rest_server import app as rest_app

# Use the MCP app as the main app since it's already a FastAPI instance
main_app = mcp_app

# Mount the REST API on a subpath to avoid conflicts
main_app.mount("/api", rest_app)

# Get the existing startup/shutdown events from REST API
for handler in rest_app.router.on_startup:
    main_app.add_event_handler("startup", handler)

for handler in rest_app.router.on_shutdown:
    main_app.add_event_handler("shutdown", handler)

# Export the unified app
app = main_app