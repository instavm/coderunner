# Contributing to CodeRunner

Thank you for your interest in contributing to CodeRunner! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/instavm/coderunner.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

1. Install dependencies: `pip install -r examples/requirements.txt`
2. Copy the example config: `cp examples/claude_desktop/claude_desktop_config.example.json examples/claude_desktop/claude_desktop_config.json`
3. Update the config file with your local paths
4. Follow the setup instructions in the README

## Build
<details>
<summary>Build Instructions</summary>

To start building the container, you might need to perform the following commands:

```bash
# Stop any running container services
sudo pkill -f container

# Start the container system
container system start

# Remove existing buildkit if necessary
container rm buildkit

# Build the container with the specified Dockerfile and tag
container build --tag cr --file Dockerfile .

# Tag the newly built container
container images tag cr instavm/coderunner

# Push the image to the registry
container images push instavm/coderunner
```

</details>

## Code layout

The assistant (`coderunner/`) runs on the host; the sandbox server (`server.py`, `auth.py`, `Dockerfile`)
runs inside the VM. Each package's `__init__.py` lists its modules.

| Package | What it holds |
| --- | --- |
| `core` | settings, paths, database, audit log, model client, local tool runner, Swift helper builder |
| `safety` | policy and approvals, the System One step judge, injection guard, egress proxy, secrets |
| `server` | the daemon, HTTP APIs and web page, run-at-login, Tailscale access, MCP bridge |
| `agent` | the tool-calling loop, its tools, memory and scheduled tasks |
| `models` | hardware detection, published-score ranking, discovery, runtimes, setup and timing |
| `browser` | the browser broker, page text and changes, URL rules, the tab-group extension |
| `files` | finding, extracting, embedding and searching the owner's files |
| `voice` | speech to text (Apple, Qwen3-ASR) and spoken replies |
| `sandbox` | the VM drivers (Apple container, Docker) and the client for the server inside |
| `channels`, `connectors` | messaging apps and macOS apps |

Tests run with `PYTHONPATH=.:tests python -m unittest discover -s tests`.

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Write tests for new features

## Testing

- Ensure all existing tests pass
- Add tests for new functionality
- Test with different Python versions if possible

## Submitting Changes

- Keep commits focused and atomic
- Write clear commit messages
- Update documentation as needed
- Ensure no sensitive information is included

## Questions?

Feel free to open an issue for questions or discussions about contributing.
