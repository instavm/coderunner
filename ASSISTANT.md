# CodeRunner Assistant: a private assistant on your own machine

CodeRunner runs a capable assistant entirely on hardware you own: a local
language/vision model, an isolated sandbox VM for code, your own browser under
guard rails, opt-in connectors to your apps, and messaging channels so you can
reach it from your phone. Nothing leaves your machine unless you approve it.

```bash
git clone https://github.com/instavm/coderunner.git && cd coderunner
./install.sh          # sandbox VM + host runtime + best local model for this machine
coderunner chat       # talk to it; approvals are asked right in the terminal
```

## What runs where

```
 your devices ──(Tailscale, optional)──▶ owner UI / API  127.0.0.1:8224   admin token or paired device
                                              │
 ┌──────────────────────── host (trusted) ────┴──────────────────────────────┐
 │ coderunner daemon: agent loop · policy + approvals · audit log · secrets   │
 │   ├─ model runtime (Ollama / llama.cpp)       127.0.0.1 only              │
 │   ├─ browser broker ── your Chrome/Brave/Edge, dedicated profile,        │
 │   │                     controlled over a private pipe (no debug port)    │
 │   ├─ connectors: Calendar, Reminders, Notes, Mail drafts, Contacts (opt-in)│
 │   └─ channels: Telegram, iMessage, Signal, WhatsApp (experimental)         │
 │ sandbox gateway  <VM bridge>:8223  model API + secret-injecting fetch      │
 │ egress proxy     <VM bridge>:8225  (proxy network mode)                    │
 └──────────────────────────────────────┬────────────────────────────────────┘
                                         │ sandbox token only
 ┌──────────── sandbox VM (untrusted code) ┴─────────────────────────────────┐
 │ CodeRunner server :8222 · Jupyter kernels · headless Playwright · MCP     │
 └────────────────────────────────────────────────────────────────────────────┘
```

* **Host daemon** (`coderunner daemon`, installed as a login service) owns
  everything trusted. It is the only process holding the admin token, owner
  secrets, the browser and connector access.
* **Sandbox VM** (Apple `container` on macOS, Docker elsewhere) runs code the
  assistant writes. It can call the local model and an allow-listed fetch
  through the gateway with the sandbox token; it cannot approve anything,
  drive the browser, read memories or reach connectors.
* **Models** run on the host so they get the GPU (Apple's VM framework does not
  pass the GPU through to Linux guests). They listen on loopback only.

## Models: picked for your hardware

`coderunner setup` (run by the installer) detects the machine (Apple Silicon
unified memory and GPU working-set limit, NVIDIA/AMD VRAM, CPU-only RAM,
WSL2), subtracts what the sandbox VM needs, and picks the strongest
tool-calling model that fits, preferring one that can also see images. It then
pulls it through Ollama, or straight from Hugging Face as GGUF for llama.cpp
when you prefer, starts the runtime and runs a smoke test (exact tool
arguments, tool choice, plain answers, and image understanding) before it
declares success. If a model fails the test, the next candidate is tried.

* Licences are checked from the model's base repository. Permissive licences
  are used automatically; licences with extra terms need `--accept-license`;
  non-commercial and unknown licences are never picked automatically.
* Never below ~4B parameters for the assistant: smaller models get tool
  arguments wrong.
* `coderunner models plan` shows what would be chosen and why;
  `coderunner models setup --model <id>` overrides; `coderunner models test`
  re-runs the smoke test.
* Optional fast "System 1" models share the runtime: an embedding model for
  memory search and a small classifier that scores untrusted text for prompt
  injection. They add signal; they are not the security boundary.

## Safety model

| Risk | Control |
| --- | --- |
| Prompt injection from web pages, mail, notes, other people, sandbox output | Untrusted output is wrapped as `<untrusted>`, screened, and **taints** the run. In a tainted run, sensitive actions (browser clicks and typing, writes to connectors, messages, schedules, memory changes, sandbox egress) ignore standing grants and need a fresh approval; navigation is limited to your `browser.allow` list without query strings. After third-party content, sandbox code needs approval in the open network mode, and in `proxy` mode the sandbox's egress allow list is suspended for 30 minutes. Schedules created by a tainted run keep running tainted; MCP clients keep their taint for 30 minutes. |
| The model approves its own actions | Approvals are decided only through the owner API (CLI, web UI, paired channel). The model never sees an approval endpoint. |
| Purchases, messages to others, deletions | High-risk kinds are always one-time approvals showing the full content; "always allow" is not offered. |
| Passwords and one-time codes | The model may not type into password or OTP fields. It hands off to you in the visible browser window. |
| Using your browser | A dedicated profile in your installed Chrome/Brave/Edge, visible so you can watch and take over, controlled over `--remote-debugging-pipe` (no TCP port exists for anything else to reach). No JavaScript evaluation, cookie or storage access is exposed; downloads and file pickers are blocked; new sites, clicks and typing need approval or a grant; `file:`, `chrome:` and LAN addresses are refused. |
| Sandbox exfiltration | Network mode `proxy`: the VM is on a host-only network whose only way out is the egress proxy. Destinations are allow-listed or approved per host; LAN, loopback, link-local and tailnet (CGNAT) addresses are refused; connections go to the address that was checked; denials are remembered for five minutes and at most five network approvals can wait at once. Mode `none` has no egress at all (the secret-injecting fetch is disabled too). |
| API keys | Stored on the host. Sandbox code writes `{{secret:name}}` in a header; the gateway substitutes it only over https, only for hosts you named (and the header you named, if any), and strips it from responses. |
| Connectors | Off until enabled one by one. Reads are allowed once enabled; writes ask. Mail can only create drafts. One-time codes and reset links are redacted from mail. Scripts get data as arguments, never as code. |
| Remote access | Tailscale Serve on your tailnet only (never Funnel). A paired-device cookie is always required; Tailscale identity headers only narrow who may use it, because local processes can forge them. |
| Accountability | Every tool call, decision, approval and egress connection is in `~/.coderunner/assistant/audit.jsonl` (`coderunner audit`). Secrets are redacted. |

Limits worth knowing:

* The sandbox credential is readable by code inside the sandbox (that code is
  meant to use the gateway). It cannot approve actions or reach owner APIs.
* In the default network mode the sandbox has normal internet access. Use
  `--network proxy` for enforced egress, or `none` for no network.
* The sandbox can reach services on the host that listen on all interfaces
  (for example a model runtime started with `OLLAMA_HOST=0.0.0.0`, or a
  development server). Keep host services on 127.0.0.1.
* Docker Desktop lets containers reach host loopback via
  `host.docker.internal`, which would expose loopback-only services such as
  the model runtime and signal-cli. The installer refuses Docker Desktop unless
  `CODERUNNER_ALLOW_DOCKER_DESKTOP=1` is set; prefer Apple `container` on macOS
  and Docker Engine on Linux.
* `proxy` mode tunnels pin the checked IP address but cannot pin the TLS
  server name, so a shared CDN address could serve another site on it.
* `--mount` refuses sources that contain or sit inside `~/.coderunner`, which
  holds tokens, secrets and assistant state.
* macOS privacy permissions (Automation, Full Disk Access) attach to the
  process that asks. Run `coderunner connectors test <name>` from Terminal to
  grant them; a background service may not be able to show the prompt.

## Channels

Only paired owners can instruct the assistant; everything else is ignored and
audited without content. Approvals arrive in the same chat (`approve <id>
[once|session|day|always]` / `deny <id>`, or buttons on Telegram). No channel
needs a public webhook.

| Channel | Setup | Notes |
| --- | --- | --- |
| Telegram | Create a bot with @BotFather, `coderunner channels enable telegram` (stores the token as a secret), restart, `coderunner channels pair telegram`, send `/pair <code>` | Private chats only by default; voice notes are transcribed locally when a speech engine is installed |
| iMessage | `brew install steipete/tap/imsg`; sign Messages on this Mac into a dedicated Apple ID; grant the daemon Full Disk Access and Automation → Messages; enable and pair | Messages.app may fetch link previews for URLs in replies, so a URL written by the model can be fetched from your Mac |
| Signal | Link signal-cli as a secondary device and run `signal-cli -a +NUM daemon --http=127.0.0.1:8080`; `coderunner channels enable signal --set url=http://127.0.0.1:8080 --set account=+NUM` | CodeRunner does not launch signal-cli |
| WhatsApp (experimental) | `brew install openclaw/tap/wacli`, `wacli auth` with a secondary number; enable with `--set accept_tos_risk=true` | Uses the unofficial linked-device protocol; the account can be banned. The official Cloud API does not allow general-purpose assistants |

Voice: install `whisper-cli` (whisper.cpp) with a `ggml-*.bin` model in
`~/.coderunner/assistant/models/`, or `parakeet-mlx` / `mlx_whisper`. Spoken
replies use macOS `say`, `piper` or `kokoro` with `ffmpeg`.

## Connectors (macOS)

`coderunner connectors enable <calendar|reminders|notes|mail|contacts>`, then
`coderunner connectors test <name>` from Terminal to grant the Automation
permission. Reads: calendar events, reminders, note search and read, recent
mail and message bodies, contact lookup. Writes (always approved): calendar
events, reminders, notes, and mail drafts saved to Drafts. There is no send
tool. Calendar queries miss recurring series that started before the range.

## Remote access

`coderunner remote enable --owner you@example.com` runs `tailscale serve` for
the owner UI on this machine's tailnet name (never Funnel) and prints the URL.
`coderunner remote pair` prints a single-use code to pair a phone or laptop
browser. Restrict tcp:443 on this device to your own login in your tailnet
policy.

## Everyday commands

```bash
coderunner status | doctor | audit
coderunner chat ["one-off question"]
coderunner approvals [approve|deny <id> --scope once|session|day|always]
coderunner grants [revoke <id>]
coderunner secrets set github --host api.github.com
coderunner connectors enable calendar && coderunner connectors test calendar
coderunner channels enable telegram && coderunner channels pair telegram
coderunner remote enable --owner you@example.com && coderunner remote pair
coderunner schedule add --name brief --spec "weekdays 08:00" --prompt "Summarise my day"
coderunner service install            # start at login
coderunner mcp                        # use the guarded tools from Claude Code / Claude Desktop
```

Configuration lives in `~/.coderunner/assistant/config.json`; state, logs and
the browser profile live next to it.
