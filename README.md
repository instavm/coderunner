<div align="center">

[![Stars](https://img.shields.io/github/stars/instavm/coderunner?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/instavm/coderunner/stargazers)
[![License](http://img.shields.io/:license-Apache%202.0-green.svg?style=flat)](https://github.com/instavm/coderunner/blob/master/LICENSE)
</div>

# CodeRunner

Sandboxed Python execution for Mac. Local files, local processing.

CodeRunner runs Python code in an isolated container on your Mac using [Apple's native container technology](https://github.com/apple/container). Process your local files without uploading them anywhere.

- Execute Python in a persistent Jupyter kernel
- Pre-installed data science stack (pandas, numpy, matplotlib, etc.)
- Files stay on your machine
- Optional browser automation via Playwright

**Requirements:** macOS, Apple Silicon (M1+), Python 3.10+

## Install

### Homebrew (coming soon)
```bash
brew install instavm/cli/coderunner
```

### Manual
```bash
git clone https://github.com/instavm/coderunner.git
cd coderunner
chmod +x install.sh
sudo ./install.sh
```

Server runs at: `http://coderunner.local:8222`

## Usage

### Claude Code CLI (Apple container)

Build the minimal Claude image and run it via the `coderunner` CLI:

```bash
container build --tag coderunner-claude --file Dockerfile.claude .
coderunner claude --branch <your-branch>
```

Notes:
- The command passes extra args through to `claude` (so `--branch` goes to Claude Code).
- Set `ANTHROPIC_API_KEY` in your environment for non-interactive usage.
- Override the image with `CODERUNNER_IMAGE` or `--image` if you tag it differently.

### LiteBox Claude (Linux)

Run Claude using Microsoft LiteBox (Linux userland runner) inside a container:

```bash
docker compose -f docker-compose.litebox-claude.yml run --rm litebox-claude
```

Notes:
- Requires a prebuilt LiteBox runner binary at `third_party/litebox-bin/litebox_runner_linux_userland` (build on x86_64 Linux).
- Example build on x86_64: `git clone https://github.com/microsoft/litebox.git && cd litebox && cargo build -p litebox_runner_linux_userland --release` then copy `target/release/litebox_runner_linux_userland` into that path.
- The container mounts the current repo into `/workspace` and injects it into LiteBox by default.
- Docker seccomp is set to `unconfined` for LiteBox syscall interception.
- Uses `linux/amd64` by default (LiteBox currently requires x86_64). The runner itself should be built on native x86_64 to avoid QEMU rustc crashes.
- Set `ANTHROPIC_API_KEY` in your environment for non-interactive usage.

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

## Pre-installed Libraries

The sandbox includes: pandas, numpy, scipy, matplotlib, seaborn, pillow, pypdf, python-docx, openpyxl, beautifulsoup4, requests, httpx, and more.

## Security

Code runs in VM-level isolation using Apple containers.

From [apple/container docs](https://github.com/apple/container/blob/main/docs/technical-overview.md):
> Each container has the isolation properties of a full VM, using a minimal set of core utilities and dynamic libraries to reduce resource utilization and attack surface.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0
