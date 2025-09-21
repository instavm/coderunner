"""Unified application combining MCP and REST API servers"""

from fastapi import FastAPI
from server import app as mcp_app  # FastMCP streamable_http_app
from api.rest_server import app as rest_app

# Use the MCP app as the main app since it's already a FastAPI instance
main_app = mcp_app

# Mount the REST API on a subpath to avoid conflicts
main_app.mount("/api", rest_app)

# Get the existing startup/shutdown events from REST API
rest_startup = None
rest_shutdown = None
for event_handler in rest_app.router.on_startup:
    rest_startup = event_handler

for event_handler in rest_app.router.on_shutdown:
    rest_shutdown = event_handler

# Add REST API startup/shutdown to main app
if rest_startup:
    @main_app.on_event("startup")
    async def combined_startup():
        await rest_startup()

if rest_shutdown:
    @main_app.on_event("shutdown") 
    async def combined_shutdown():
        await rest_shutdown()

# Export the unified app
app = main_app