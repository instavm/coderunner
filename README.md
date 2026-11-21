<div align="center">

[![Start](https://img.shields.io/github/stars/instavm/coderunner?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/instavm/coderunner/stargazers)
[![License](http://img.shields.io/:license-Apache%202.0-green.svg?style=flat)](https://github.com/instavm/coderunner/blob/master/LICENSE)
</div>

# CodeRunner

A private index of your Mac for the agent you already use, and a sandbox for the code it writes.

You use Claude Code, Codex or Claude Desktop (Cowork). You want it to find and read your own files, mail,
messages, notes and calendar without uploading your disk, and to run code on what it finds somewhere it
cannot do damage. CodeRunner indexes the folders and apps you choose on your Mac's Neural Engine, serves the
search to your agent over MCP under an approval and leak rule, and runs the agent's code in a VM.

## Search for your agent

```bash
git clone https://github.com/instavm/coderunner.git && cd coderunner
./install.sh --search                                  # embedding model and index only; about 1.5 GB, a few minutes
coderunner files add ~/Documents                       # and any other folder
coderunner files sources add mail                      # also messages, notes, calendar
claude mcp add coderunner-search -- coderunner mcp --tools search
```

Needs an Apple silicon Mac on macOS 15 or later (the embedding model runs on the Neural Engine) or Ollama on
Linux. Codex: `codex mcp add coderunner-search -- coderunner mcp --tools search`. Claude Desktop, OpenClaw, LM
Studio and other clients: [ASSISTANT.md](ASSISTANT.md#from-claude-code-codex-cowork-openclaw-and-lm-studio).
`coderunner find "the lease for the apartment"` searches from the terminal.

What your agent gets, through `files_search` and `file_read`:

* Documents, notes, code, spreadsheets, slides, photos, screenshots, video and recordings, found by what they
  are about or by exact codes and names, with the page, line or moment that matched. Folders by name and by
  what they hold. Measured on a reproducible corpus of personal files: the right file first for 94% of 108
  questions, in the top ten for 98%, in 19 ms ([VALIDATION.md](VALIDATION.md)).
* Mail, Messages, Notes and Calendar entries from the apps you added, read from the index.
* A memory of things you tell it (`memory_save`, `memory_search`).

What stays on your side:

* Nothing leaves the Mac unless your agent sends it. Search results carry your private data and third-party
  text, so the bridge marks the client as holding them, and sensitive actions it takes afterwards need a fresh
  approval; sharing that data with a new destination asks once (the leak rule in ASSISTANT.md).
* Secrets, keys, `.env` files, hidden folders and `~/Library` are never indexed. Files only in iCloud are
  downloaded to be read when small and given back.
* One-time codes and sign-in links are redacted from mail before indexing.
* The stdio bridge holds the admin token: configure it only for clients you run yourself. The HTTP endpoint
  (`http://127.0.0.1:8224/mcp/search`) answers nothing without it.

## The full assistant

`./install.sh` without `--search` also builds the sandbox VM, picks the strongest local model your hardware can
run (benchmarked on the machine, not guessed), and starts an assistant you can talk to: your own Chrome behind
approvals, Calendar, Reminders, Notes, Mail drafts and Contacts, voice, scheduled tasks, and Telegram, iMessage
or Signal to reach it from your phone. Needs macOS 26 on Apple silicon, or Linux with Docker. No model runtime
needs installing first: on a Mac setup fetches llama.cpp's server and runs embeddings on Core ML (no Ollama), and
it checks the disk before it downloads a model.
[ASSISTANT.md](ASSISTANT.md) has the architecture, the safety model and every command.

For managed cloud VMs for agents, see [InstaVM](https://instavm.io).

## The sandbox

The sandbox is an isolated VM (Apple `container` on macOS 26, Docker elsewhere) exposed over MCP with a bearer
token, with explicitly chosen host mounts and network modes. Isolation reduces risk; it does not prevent all
data loss or exfiltration.

```bash
./install.sh            # add --build to build the sandbox image from this checkout
```

### Credentials

The installer generates the sandbox API token in `~/.coderunner/api-token`
(or saves the `CODERUNNER_API_TOKEN` you exported). Clients send it as
`Authorization: Bearer <token>`; print it with `coderunner token`. Only
`GET /health` is public, and a missing or weak token fails closed. Images
published before authentication ignore the token, so the installer refuses them;
use `./install.sh --build` until an authenticated image is published. See
[authentication and network boundaries](AUTHENTICATION.md).

### Stop and resume

```bash
coderunner sandbox stop        # stop; the filesystem is kept (Python kernel state is not)
coderunner sandbox start       # resume (the daemon also starts it at login)
coderunner sandbox reset       # delete it and its filesystem, then run the installer again
coderunner sandbox logs -f     # follow its logs; `status` shows whether it is running
```

### Resource limits and workspace mounts

```bash
./install.sh --cpus 4 --memory 8g
./install.sh \
  --mount "$HOME/projects/api:/workspace/api" \
  --mount "$HOME/projects/output:/workspace/output:rw"
```

`CODERUNNER_CPUS` and `CODERUNNER_MEMORY` work as alternatives. Extra mounts go under `/workspace`, read-only
unless `:rw`, and default to none. Resource and mount settings are fixed when the sandbox is created; change
them with `coderunner sandbox reset --cpus 4 --memory 8g --mount ...`, which takes the same options.

### Network modes

```bash
./install.sh --network proxy   # only hosts you allow or approve, through the host's egress proxy
./install.sh --network none    # no network at all
```

By default code in the sandbox has unrestricted network access. In `proxy` mode package registries (PyPI, npm,
GitHub, Hugging Face) are allowed and every other host becomes an approval request; LAN, loopback and tailnet
addresses are refused. In both modes the MCP server is at `http://127.0.0.1:8222/mcp`. The setting is fixed
when the sandbox is created (`coderunner sandbox reset --network ...` changes it). These modes need Apple
`container` on macOS. `CODERUNNER_NETWORK` also works.

### Claude Code inside the sandbox

```bash
coderunner sandbox shell
root@coderunner:/app# npm install -g @anthropic-ai/claude-code
```

### MCP clients for the sandbox

The sandbox's MCP server is at `http://coderunner.local:8222/mcp` (`http://127.0.0.1:8222/mcp` in `none` and
`proxy` mode and on Linux). Every client sends `Authorization: Bearer <token>`; `coderunner token` prints it.
The tools: `execute_python_code` (a persistent Jupyter kernel), `start_python_session`, `list_python_sessions`,
`stop_python_session` (up to five named kernels, for agents that must not share state), `run_shell` and
`shell_job`, `navigate_and_get_all_visible_text` (Playwright), and `list_skills`, `get_skill_info`,
`get_skill_file` (see Skills).

```bash
# Claude Code: the server directly, or the plugin (it sends the same header)
claude mcp add --transport http coderunner http://coderunner.local:8222/mcp \
  --header "Authorization: Bearer $(coderunner token)"
claude plugin marketplace add https://github.com/instavm/coderunner-plugin && claude plugin install instavm-coderunner
```

```json
// Gemini CLI (~/.gemini/settings.json) and OpenCode (~/.config/opencode/opencode.json, under "mcp" with
// "type": "remote" and "url") take the same URL and header:
{"mcpServers": {"coderunner": {"httpUrl": "http://coderunner.local:8222/mcp",
                               "headers": {"Authorization": "Bearer <output of coderunner token>"}}}}
```

Claude Desktop and Kiro start the stdio proxy in `examples/claude_desktop/mcpproxy.py` with
`~/.coderunner/venv/bin/python`; it reads the token from `~/.coderunner/api-token`. Copy
`examples/claude_desktop/claude_desktop_config.example.json`, set the two paths, and add it under Settings,
Developer in Claude Desktop (`~/.kiro/settings/mcp.json` for Kiro). `examples/openai_agents/openai_client.py`
shows the sandbox from the OpenAI Agents library (`pip install -r examples/requirements.txt`).
[Coderunner-UI](https://github.com/instavm/coderunner-ui) is an offline chat workspace on top of the sandbox.

### Isolation

Code runs in a container with VM-level isolation; the host and files outside the mounts you chose are out of its reach, and the network modes above decide where it can connect.

From [@apple/container](https://github.com/apple/container/blob/main/docs/technical-overview.md):
>Each container has the isolation properties of a full VM, using a minimal set of core utilities and dynamic libraries to reduce resource utilization and attack surface.

### Skills

Skills are folders with a `SKILL.md` and scripts that run inside the sandbox; the MCP tools `list_skills`,
`get_skill_info` and `get_skill_file` describe them to the agent, which runs their scripts with
`execute_python_code`. Two ship with every install: `pdf-text-replace` (fillable PDF forms) and
`image-crop-rotate`. Your own go under `~/.coderunner/assets/skills/user/<name>/` with a `SKILL.md` and a
`scripts/` folder, and are listed on the next `list_skills`. [SKILLS-README.md](SKILLS-README.md) covers
running Anthropic's published skills locally.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
