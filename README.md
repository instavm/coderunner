<div align="center">

[![Stars](https://img.shields.io/github/stars/instavm/coderunner?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/instavm/coderunner/stargazers)
[![License](http://img.shields.io/:license-Apache%202.0-green.svg?style=flat)](https://github.com/instavm/coderunner/blob/master/LICENSE)
</div>

# CodeRunner

Sandboxed code execution for Mac using [Apple containers](https://github.com/apple/container).

- Python execution in persistent Jupyter kernels
- AI coding agents: Claude Code, OpenAI Codex, Cursor, Gemini CLI
- Cloud CLIs: AWS, GCP, Azure, GitHub
- Browser automation via Playwright
- Data science stack: pandas, numpy, scipy, matplotlib

**Requirements:** macOS 15+, Apple Silicon

## Install

```bash
brew install instavm/cli/coderunner
```

Server starts automatically at `http://coderunner.local:8222`

### Manual Install

```bash
git clone https://github.com/instavm/coderunner.git
cd coderunner
./install.sh
```

### Commands

```bash
coderunner status              # Check if running
coderunner stop                # Stop server
coderunner start               # Start server
coderunner run claude          # Run Claude Code
coderunner run codex           # Run OpenAI Codex
coderunner run cursor          # Run Cursor
coderunner exec bash           # Shell into container
coderunner logs                # View logs
```

### Start Options

```bash
coderunner start --with-ssh-agent      # Forward SSH agent for git
coderunner start --with-credentials    # Mount ~/.claude, ~/.aws, ~/.config/gh
coderunner start --env-file ~/.env     # Load environment from file
```

## Usage

### Python Library

Install the client (server must be running locally):
```bash
pip install git+https://github.com/instavm/coderunner.git
```

```python
from coderunner import CodeRunner

# Connect to local server
cr = CodeRunner()  # defaults to http://coderunner.local:8222

# Execute Python code
result = cr.execute("""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.describe())
""")
print(result.stdout)

# One-liner for quick scripts
from coderunner import execute
print(execute("2 + 2"))  # 4
```

### REST API

```bash
# Execute Python code
curl -X POST http://coderunner.local:8222/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import pandas as pd; print(pd.__version__)"}'
```

Response:
```json
{"stdout": "2.0.3\n", "stderr": "", "execution_time": 0.12}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/execute` | POST | Execute Python code |
| `/v1/sessions` | POST | Create session |
| `/v1/sessions/{id}` | GET/DELETE | Session management |
| `/v1/browser/navigate` | POST | Browser navigation |
| `/v1/browser/content` | POST | Extract page content |
| `/health` | GET | Health check |

<details>
<summary>Deprecated endpoints (still work, will be removed in v1.0)</summary>

| Old | New |
|-----|-----|
| `/execute` | `/v1/execute` |
| `/v1/sessions/session` | `/v1/sessions` |
| `/v1/browser/interactions/navigate` | `/v1/browser/navigate` |
| `/v1/browser/interactions/content` | `/v1/browser/content` |

</details>

## MCP Server

For AI tools that support the Model Context Protocol, connect to:
```
http://coderunner.local:8222/mcp
```

### Available Tools
- `execute_python_code(command)` - Run Python code
- `navigate_and_get_all_visible_text(url)` - Web scraping
- `list_skills()` - List available skills
- `get_skill_info(skill_name)` - Get skill documentation
- `get_skill_file(skill_name, filename)` - Read skill files

### Integration Examples

<details>
<summary>Claude Desktop</summary>

Edit your Claude Desktop config:
```json
{
  "mcpServers": {
    "coderunner": {
      "command": "/path/to/python3",
      "args": ["/path/to/coderunner/examples/claude_desktop/mcpproxy.py"]
    }
  }
}
```

See `examples/claude_desktop/claude_desktop_config.example.json` for a complete example.
</details>

<details>
<summary>Claude Code CLI</summary>

```bash
claude plugin marketplace add https://github.com/instavm/coderunner-plugin
claude plugin install instavm-coderunner
```
</details>

<details>
<summary>OpenCode</summary>

Edit `~/.config/opencode/opencode.json`:
```json
{
  "mcp": {
    "coderunner": {
      "type": "remote",
      "url": "http://coderunner.local:8222/mcp",
      "enabled": true
    }
  }
}
```
</details>

<details>
<summary>Gemini CLI</summary>

Edit `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "coderunner": {
      "httpUrl": "http://coderunner.local:8222/mcp"
    }
  }
}
```
</details>

<details>
<summary>Amazon Kiro</summary>

Edit `~/.kiro/settings/mcp.json`:
```json
{
  "mcpServers": {
    "coderunner": {
      "command": "/path/to/python",
      "args": ["/path/to/coderunner/examples/claude_desktop/mcpproxy.py"]
    }
  }
}
```
</details>

<details>
<summary>OpenAI Agents</summary>

```bash
export OPENAI_API_KEY="your-key"
python examples/openai_agents/openai_client.py
```
</details>

## Skills

CodeRunner includes a skills system for common tasks.

**Built-in skills:**
- `pdf-text-replace` - Replace text in PDF forms
- `image-crop-rotate` - Image manipulation

**Add custom skills:** Place them in `~/.coderunner/assets/skills/user/`

See [SKILLS-README.md](SKILLS-README.md) for details.

## Configuration

Create `~/.coderunner.config` with your API keys:

```
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
GITHUB_TOKEN=ghp_xxx
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxx
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Keys are loaded automatically on `coderunner start`. CLI params override config file values.

### Credential Locations

| Tool | Config Directory | Environment Variable |
|------|-----------------|---------------------|
| Claude Code | `~/.claude/` | `ANTHROPIC_API_KEY` |
| OpenAI Codex | `~/.codex/` | `OPENAI_API_KEY` |
| Cursor | `~/.cursor/` | `CURSOR_API_KEY` |
| GitHub CLI | `~/.config/gh/` | `GITHUB_TOKEN` |
| AWS CLI | `~/.aws/` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| GCP CLI | `~/.config/gcloud/` | `GOOGLE_APPLICATION_CREDENTIALS` |
| Azure CLI | `~/.azure/` | `AZURE_CREDENTIALS` |

Use `--with-credentials` to mount these directories into the container (read-only).

## Pre-installed Tools

**Python:** pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, pillow, pypdf, python-docx, openpyxl, beautifulsoup4, requests, httpx, playwright

**AI Agents:** Claude Code, OpenAI Codex, Cursor, Gemini CLI

**Cloud CLIs:** aws, gcloud, az, gh

## Security

Code runs in VM-level isolation using Apple containers.

From [apple/container docs](https://github.com/apple/container/blob/main/docs/technical-overview.md):
> Each container has the isolation properties of a full VM, using a minimal set of core utilities and dynamic libraries to reduce resource utilization and attack surface.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0
