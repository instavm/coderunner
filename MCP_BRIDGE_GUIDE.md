# Universal NPX MCP Bridge Guide

## Overview

The CodeRunner MCP server now includes a **Universal NPX MCP Bridge** that allows you to expose any NPX-based MCP server through your existing HTTP endpoint. This means you can easily add filesystem operations, GitHub integration, web scraping, and more without writing custom code.

## How It Works

The bridge:
1. Spawns NPX-based MCP servers as child processes
2. Communicates with them via stdin/stdout using JSON-RPC
3. Discovers their capabilities (tools, resources, prompts)
4. Dynamically registers all tools with FastMCP
5. Exposes them via HTTP at `http://coderunner.local:8222/mcp`

## Configuration

### Quick Start

Edit the `MCP_BRIDGE_SERVERS` list in `server.py` (around line 325):

```python
MCP_BRIDGE_SERVERS = [
    {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(SHARED_DIR)],
        "description": "File system operations (read, write, list, etc.)",
        "enabled": True
    },
    # Add more servers here...
]
```

### Available Official MCP Servers

Here are some popular NPX-based MCP servers you can add:

#### 1. **Filesystem** (Already enabled by default)
```python
{
    "name": "filesystem",
    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(SHARED_DIR)],
    "description": "File system operations (read, write, list, etc.)",
    "enabled": True
}
```
**Tools**: read_file, write_file, edit_file, create_directory, list_directory, move_file, search_files, get_file_info

#### 2. **Chrome DevTools** (Already enabled by default with headless mode)
```python
{
    "name": "chrome-devtools",
    "command": ["npx", "-y", "chrome-devtools-mcp@latest", "--headless=true"],
    "description": "Chrome DevTools Protocol for browser automation and debugging (headless mode)",
    "enabled": True
}
```
**Tools**: chrome_navigate, chrome_screenshot, chrome_click, chrome_evaluate, chrome_console_logs, chrome_network_logs, and more
**Features**: Advanced browser automation using Chrome DevTools Protocol (CDP), debugging capabilities, network interception, JavaScript execution
**Package**: Official Chrome DevTools MCP server from Google
**Headless Mode**: Enabled by default (change `--headless=true` to `--headless=false` for visible browser)
**Max Viewport (Headless)**: 3840x2160px

**Additional Options**:
- `--channel=canary` - Use Chrome Canary instead of stable
- `--isolated=true` - Use isolated user data directory
- Remove `--headless=true` for visible browser debugging

#### 3. **GitHub**
```python
{
    "name": "github",
    "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
    "description": "GitHub repository operations",
    "enabled": bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
}
```
**Tools**: create_repository, create_or_update_file, push_files, search_repositories, create_issue, etc.
**Setup**: Set `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable

#### 3. **Puppeteer**
```python
{
    "name": "puppeteer",
    "command": ["npx", "-y", "@modelcontextprotocol/server-puppeteer"],
    "description": "Web scraping and browser automation",
    "enabled": True
}
```
**Tools**: puppeteer_navigate, puppeteer_screenshot, puppeteer_click, puppeteer_fill, puppeteer_evaluate

#### 4. **Memory**
```python
{
    "name": "memory",
    "command": ["npx", "-y", "@modelcontextprotocol/server-memory"],
    "description": "Persistent memory across conversations",
    "enabled": True
}
```
**Tools**: create_entities, create_relations, search_nodes, open_nodes, etc.

#### 5. **Brave Search**
```python
{
    "name": "brave-search",
    "command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
    "description": "Web search via Brave Search API",
    "enabled": bool(os.getenv("BRAVE_API_KEY"))
}
```
**Tools**: brave_web_search, brave_local_search
**Setup**: Set `BRAVE_API_KEY` environment variable

#### 6. **Google Maps**
```python
{
    "name": "google-maps",
    "command": ["npx", "-y", "@modelcontextprotocol/server-google-maps"],
    "description": "Google Maps geocoding and places",
    "enabled": bool(os.getenv("GOOGLE_MAPS_API_KEY"))
}
```
**Setup**: Set `GOOGLE_MAPS_API_KEY` environment variable

#### 7. **Fetch (Web Content)**
```python
{
    "name": "fetch",
    "command": ["npx", "-y", "@modelcontextprotocol/server-fetch"],
    "description": "Fetch and process web content",
    "enabled": True
}
```
**Tools**: fetch

#### 8. **Postgres**
```python
{
    "name": "postgres",
    "command": ["npx", "-y", "@modelcontextprotocol/server-postgres", "postgresql://connection-string"],
    "description": "PostgreSQL database operations",
    "enabled": bool(os.getenv("DATABASE_URL"))
}
```
**Setup**: Replace connection string with your database URL

#### 9. **SQLite**
```python
{
    "name": "sqlite",
    "command": ["npx", "-y", "@modelcontextprotocol/server-sqlite", str(SHARED_DIR / "database.db")],
    "description": "SQLite database operations",
    "enabled": True
}
```

#### 10. **Sequential Thinking**
```python
{
    "name": "sequential-thinking",
    "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
    "description": "Step-by-step thinking and reasoning",
    "enabled": True
}
```

## Usage

### 1. List Available Bridged Tools

Call the `list_bridged_tools` tool to see all available tools from configured servers:

```python
# This will show all servers and their tools
result = await client.call_tool("list_bridged_tools", {})
```

### 2. Using Bridged Tools

All bridged tools are prefixed with their server name. For example:

```python
# Read a file using the filesystem server
result = await client.call_tool("filesystem_read_file", {
    "path": "/app/uploads/example.txt"
})

# Search GitHub repositories
result = await client.call_tool("github_search_repositories", {
    "query": "mcp server",
    "page": 1,
    "per_page": 10
})

# Take a screenshot with Puppeteer
result = await client.call_tool("puppeteer_screenshot", {
    "url": "https://example.com",
    "width": 1920,
    "height": 1080
})
```

### 3. Manual Initialization

By default, bridges initialize automatically on server startup. You can also manually trigger initialization:

```python
# Initialize or check initialization status
result = await client.call_tool("initialize_bridges", {})
```

## Environment Variables

Some MCP servers require API keys or configuration via environment variables:

```bash
# GitHub
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token_here"

# Brave Search
export BRAVE_API_KEY="your_brave_api_key"

# Google Maps
export GOOGLE_MAPS_API_KEY="your_google_maps_key"

# Database
export DATABASE_URL="postgresql://user:pass@host:5432/db"
```

## Adding Custom NPX Servers

You can add any NPX-based MCP server:

```python
{
    "name": "my-custom-server",
    "command": ["npx", "-y", "@your-org/your-mcp-server", "arg1", "arg2"],
    "description": "Your custom MCP server",
    "enabled": True
}
```

## Troubleshooting

### Check Server Logs

The server logs will show initialization status:

```
INFO - Starting MCP server 'filesystem' with command: npx -y @modelcontextprotocol/server-filesystem /app/uploads
INFO - Successfully initialized MCP server 'filesystem' with 8 tools
INFO - Discovered 8 tools from 'filesystem'
INFO - Registered: filesystem_read_file
INFO - Registered: filesystem_write_file
...
```

### Common Issues

1. **Server fails to initialize**
   - Check that `npx` is installed and available
   - Verify environment variables are set (for servers that require them)
   - Check server logs for error messages

2. **Tools not appearing**
   - Ensure `enabled: True` is set for the server
   - Check that the server initialized successfully
   - Call `list_bridged_tools` to see what's available

3. **Tool calls failing**
   - Verify you're using the correct tool name format: `{server_name}_{tool_name}`
   - Check the tool's required parameters
   - Review server logs for error details

## Architecture

```
┌─────────────────────────────────────────────────┐
│         LLM Client (OpenAI, Gemini, etc.)       │
└─────────────────────┬───────────────────────────┘
                      │ HTTP MCP Protocol
                      ▼
┌─────────────────────────────────────────────────┐
│         CodeRunner FastMCP Server               │
│  ┌─────────────────────────────────────────┐   │
│  │   Universal NPX MCP Bridge              │   │
│  │  ┌────────────┐  ┌────────────┐         │   │
│  │  │ Filesystem │  │   GitHub   │   ...   │   │
│  │  │  Server    │  │   Server   │         │   │
│  │  └────────────┘  └────────────┘         │   │
│  │       ▲              ▲                   │   │
│  │       │ stdio        │ stdio             │   │
│  │       ▼              ▼                   │   │
│  │  [npx process]  [npx process]           │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Performance Notes

- Bridge initialization adds ~1-3 seconds per server on startup (due to NPX downloads)
- Once initialized, tool calls have minimal overhead (<10ms)
- NPX servers are cached after first download
- Each server runs as a separate process with ~20-50MB memory overhead

## Security Considerations

1. **File Access**: The filesystem server is restricted to SHARED_DIR (`/app/uploads` by default)
2. **API Keys**: Store sensitive keys in environment variables, not in code
3. **Command Injection**: Never use user input directly in server commands
4. **Network Access**: Some servers (GitHub, Brave) require internet access

## Next Steps

1. Edit `MCP_BRIDGE_SERVERS` in `server.py`
2. Set any required environment variables
3. Restart the CodeRunner server
4. Call `list_bridged_tools` to verify setup
5. Start using the bridged tools in your AI workflows!

## Example Configuration

Here's a complete example with multiple servers:

```python
MCP_BRIDGE_SERVERS = [
    {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(SHARED_DIR)],
        "description": "File system operations",
        "enabled": True
    },
    {
        "name": "github",
        "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
        "description": "GitHub repository operations",
        "enabled": bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
    },
    {
        "name": "memory",
        "command": ["npx", "-y", "@modelcontextprotocol/server-memory"],
        "description": "Persistent memory across conversations",
        "enabled": True
    },
    {
        "name": "fetch",
        "command": ["npx", "-y", "@modelcontextprotocol/server-fetch"],
        "description": "Fetch and process web content",
        "enabled": True
    },
]
```

This configuration gives you:
- Local file operations
- GitHub integration (when token is set)
- Persistent memory for conversations
- Web content fetching

All tools from these servers will be automatically available via your HTTP MCP endpoint!
