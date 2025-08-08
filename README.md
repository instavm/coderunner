
<div align="center">

[![Start](https://img.shields.io/github/stars/instavm/coderunner?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/instavm/coderunner/stargazers)
[![License](http://img.shields.io/:license-Apache%202.0-green.svg?style=flat)](https://github.com/instavm/coderunner/blob/master/LICENSE)
</div>

# CodeRunner: Run AI Generated Code Locally

CodeRunner is a cross-platform MCP (Model Context Protocol) server that executes AI-generated code in a secure, sandboxed environment. It supports macOS via Apple's native [containers](https://github.com/apple/container) and Linux/Windows via Docker.

**Key use case:** Process your local files (videos, images, documents, data) with remote LLMs like Claude or ChatGPT without uploading your files to the cloud. The LLM generates code that runs locally on your machine to analyze, transform, or process your files.

## What CodeRunner Enables

| Without CodeRunner | With CodeRunner |
| :--- | :--- |
| LLM writes code, you run it manually | LLM writes and executes code, returns results |
| Upload files to cloud for AI processing | Files stay on your machine, processed locally |
| Install tools and dependencies yourself | Tools available in sandbox, auto-installs others |
| Copy/paste scripts to run elsewhere | Code runs immediately, shows output/files |
| LLM analyzes text descriptions of files | LLM directly processes your actual files |
| Manage Python environments and packages | Pre-configured environment ready to use |

## Quick Start

**Prerequisites:**
*   **For macOS:** Apple Silicon (M1/M2/M3/M4) and the [Apple Container](https://github.com/apple/container/releases) tool installed.
*   **For Linux/Windows:** [Docker](https://docs.docker.com/get-docker/) installed and running.

```bash
git clone https://github.com/instavm/coderunner.git
cd coderunner
chmod +x install.sh
./install.sh
```

The script will detect your operating system and set up CodeRunner accordingly.

*   **On macOS:** The MCP server will be available at `http://coderunner.local:8222/mcp`
*   **On Linux/Windows:** The MCP server will be available at `http://localhost:8222/mcp`

**Install required packages for examples:**
```bash
pip install -r examples/requirements.txt
```

## Integration Options

### Option 1: Claude Desktop Integration


<details>
<summary>Configure Claude Desktop to use CodeRunner as an MCP server:</summary>

![demo1](images/demo.png)

![demo2](images/demo2.png)

![demo4](images/demo4.png)

1. **Copy the example configuration:**
   ```bash
   cd examples
   cp claude_desktop/claude_desktop_config.example.json claude_desktop/claude_desktop_config.json
   ```

2. **Edit the configuration file** and replace the placeholder paths:
   - Replace `/path/to/your/python` with your actual Python path (e.g., `/usr/bin/python3` or `/opt/homebrew/bin/python3`)
   - Replace `/path/to/coderunner` with the actual path to your cloned repository

   Example after editing:
   ```json
   {
     "mcpServers": {
       "coderunner": {
         "command": "/opt/homebrew/bin/python3",
         "args": ["/Users/yourname/coderunner/examples/claude_desktop/mcpproxy.py"]
       }
     }
   }
   ```

3. **Update Claude Desktop configuration:**
   - Open Claude Desktop
   - Go to Settings → Developer
   - Add the MCP server configuration
   - Restart Claude Desktop

4. **Start using CodeRunner in Claude:**
   You can now ask Claude to execute code, and it will run safely in the sandbox!
</details>

### Option 2: Python OpenAI Agents
<details>
<summary>Use CodeRunner with OpenAI's Python agents library:</summary>

![demo3](images/demo3.png)

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Run the client:**
   ```bash
   python examples/openai_agents/openai_client.py
   ```

3. **Start coding:**
   Enter prompts like "write python code to generate 100 prime numbers" and watch it execute safely in the sandbox!
</details>

### Option 3: Gemini-CLI
[Gemini CLI](https://github.com/google-gemini/gemini-cli) is recently launched by Google.

<details>
<summary>~/.gemini/settings.json</summary>

```json
{
  "theme": "Default",
  "selectedAuthType": "oauth-personal",
  "mcpServers": {
    "coderunner": {
      "httpUrl": "http://coderunner.local:8222/mcp"
    }
  }
}
```


</details>

![gemini1](images/gemini1.png)

![gemini2](images/gemini2.png)


### Option 4: Kiro by Amazon
[Kiro](https://kiro.dev/blog/introducing-kiro/) is recently launched by Amazon.

<details>
<summary>~/.kiro/settings/mcp.json</summary>

```json
{
  "mcpServers": {
    "coderunner": {
      "command": "/path/to/venv/bin/python",
      "args": [
        "/path/to/coderunner/examples/claude_desktop/mcpproxy.py"
      ],
      "disabled": false,
      "autoApprove": [
        "execute_python_code"
      ]
    }
  }
}
```


![kiro](images/kiro.png)

</details>



## Security

Code runs in an isolated container with VM-level isolation. Your host system and files outside the sandbox remain protected.

From [@apple/container](https://github.com/apple/container/blob/main/docs/technical-overview.md):
>Each container has the isolation properties of a full VM, using a minimal set of core utilities and dynamic libraries to reduce resource utilization and attack surface.

On Linux and Windows, CodeRunner uses Docker for similar containerization and security benefits.

## Architecture

CodeRunner consists of:
- **Sandbox Container:** Isolated execution environment (Apple Container or Docker) with a Jupyter kernel.
- **MCP Server:** Handles communication between AI models and the sandbox.

## Examples

The `examples/` directory contains:
- `openai-agents` - Example OpenAI agents integration
- `claude-desktop` - Example Claude Desktop integration

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
