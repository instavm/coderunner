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
 │   ├─ browser broker ── your Chrome/Brave/Edge: its own profile over a     │
 │   │     private pipe, or a CodeRunner tab group (extension)               │
 │   ├─ connectors: Calendar, Reminders, Notes, Mail drafts, Contacts (opt-in)│
 │   └─ channels: Telegram, iMessage, Signal, WhatsApp (experimental)         │
 │ sandbox gateway  <VM bridge>:8223  model API + secret-injecting fetch      │
 │ egress proxy     <VM bridge>:8225  (proxy network mode)                    │
 └──────────────────────────────────────┬────────────────────────────────────┘
                                         │ sandbox token only
 ┌──────────── sandbox VM (untrusted code) ┴─────────────────────────────────┐
 │ CodeRunner server :8222 · runs code as plain processes (lite, ~1 GB)     │
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
unified memory and GPU working-set limit; NVIDIA VRAM pooled across cards,
DGX Spark / GB10 and Jetson unified memory, and the GPU generation; AMD;
CPU-only RAM; WSL2), subtracts what the sandbox VM needs, and picks the
strongest tool-calling model that fits, preferring one that can also see
images. The same process runs on every machine, from an 8 GB MacBook Air to a
512 GB Mac Studio or a multi-GPU server:

1. Candidates come from the curated catalog (checked models with measured
   speed factors) and from recent releases discovered on Hugging Face from
   trusted publishers (`coderunner models discover`).
2. Each model's own config gives its context-cache cost; its real GGUF and
   MLX builds give the file sizes; the build chosen is the most precise one
   that fits this machine's budget at 4 bits or more with at least 16K of
   context, up to 6-bit when it still runs at 15 tok/s or more (beyond that,
   precision costs more speed than it is worth in long agent loops).
3. Candidates that fit well are ranked by published benchmark scores, not
   size. Hugging Face's structured evaluation results (GPQA, MMLU-Pro,
   Terminal-Bench, Toolathlon, SWE-bench, MMMU-Pro, AIME) give each model one
   ability score, fitted across whichever benchmarks it reports, less what
   its quantized build loses. Speed counts next, up to about 32 tok/s.
   Models with no published scores come last. The scores ship with the
   catalog and are refreshed with discovery each day; `models discover` and
   `models plan` show them. The top candidates are downloaded and tested on
   the machine itself until one passes. It then
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
* Models are ranked across runtimes. Prism ML's Bonsai builds of 27B models
  (1-bit, 3.8 GB; ternary Bonsai 2, 6-7 GB) run on Prism's llama.cpp build,
  which setup downloads pinned to a release and checks against its published
  checksum. They fit 16-24 GB Macs, but quantization costs them: Prism's own
  scores for the 1-bit 27B are 60.5 on MMMU-Pro and 87.1 on AIME 2026,
  against 75.8 and 94.1 for Qwen3.6-27B, which puts it below Qwen3.5-9B. They
  stay available as alternatives (`coderunner models setup --model
  bonsai-27b-1bit`). Speeds measured on an M3 Pro: the 1-bit 27B generates
  14.6 tok/s and reads prompts at 81 tok/s; Bonsai 2 27B 9 and 70. Reading
  the prompt is the slow part on this class of chip, so the agent keeps its
  instructions and conversation stable for the runtime's prompt cache.

* Runtimes per model: Ollama serves families whose tool calls it parses;
  others (GLM-5.x, DeepSeek V4, MiMo, Ling...) run on llama.cpp's own server
  next to it. When llama.cpp is missing, the plan says which better model it
  would allow and how to install it.
* NVIDIA cards with plenty of system RAM can run mixture-of-experts models
  with the experts in RAM (llama.cpp `--cpu-moe`): a 12 GB card with 64 GB of
  RAM runs Qwen3.6-35B-A3B instead of Qwen3.5-9B.
* Combinations known to be broken are skipped with the reason, e.g. DeepSeek
  V4 on GB10 (copied strings, tool arguments included, come out corrupted)
  and DeepSeek V4.1, which llama.cpp does not support yet.
* An 8 GB Mac gets a compact 4B build, text only: the vision part does not
  fit next to the sandbox and macOS.
* On Macs with 48 GB or more, the plan notes when raising the GPU memory
  limit (`sudo sysctl iogpu.wired_limit_mb=...`, until restart) would allow
  a clearly better model.
* Several machines (e.g. two or four DGX Sparks) can pool memory through
  llama.cpp's RPC mode: list the others under `models.cluster` as
  `{"host", "port", "memory_gb"}`, each running `rpc-server`. This is
  experimental and untested on real hardware. The RPC link has no
  authentication or encryption, so use it only on a direct cable or a
  private network you control.
* With discovery and published scores today (October 2026) the planner
  picks, by its estimates:

  | Machine | Assistant model |
  | --- | --- |
  | 8 GB Mac | Qwen3.5-4B compact, text only, 8K context |
  | 16-24 GB Mac | Qwen3.5-9B |
  | 64-128 GB Mac (M4 Max), 24-32 GB NVIDIA | Qwen3.8-27B |
  | 64 GB M1 Max | Qwen3.6-35B-A3B (about 9x faster than the 27B there) |
  | DGX Spark | Qwen3.6-35B-A3B (a dense 27B is slow on its memory) |
  | 256-512 GB Mac Studio | MiMo-V2.6-Flash (llama.cpp) |
  | 12 GB NVIDIA, 64 GB RAM | Qwen3.6-35B-A3B, experts in RAM (llama.cpp) |

  A larger model with weaker published scores is not picked: Granite 4.2 30B
  scores below Qwen3.5-9B on MMLU-Pro (77.6 against 82.5) and far below
  Qwen3.8-27B on Terminal-Bench 2.1 (29 against 73).

* The sandbox runs the lite profile with the assistant: code runs as plain
  processes in a work folder per session (files persist, variables do not),
  with no Jupyter server and no browser inside the VM, which is capped at 1 GB
  and used about 170 MB in testing. The assistant browses with your own
  browser on the host. `./install.sh --profile full` keeps Jupyter kernels and
  the headless browser for MCP clients that use them.

* Embeddings for memory and files: EmbeddingGemma 2 (text, images and audio
  in one space) through Ollama, also when the assistant model runs elsewhere
  (`models.embedding_url`).
* Optional fast "System 1" models share the runtime: an embedding model for
  memory search, and a decision model (`models.decision`, served through
  Ollama's decision API) that judges whether a browser step commits something
  and screens untrusted text for instructions aimed at the assistant. Without
  a decision model, the assistant model judges browser steps itself and
  untrusted text is not screened (it is still wrapped and taints the task).
  There are no keyword lists: these judgements come from models.

## Safety model

You are asked only about steps that commit something or could leak your data.
Ordinary work (opening sites, clicking through them, filling in a search,
picking dates and filters, running code, saving notes) runs without prompts.

| Risk | Control |
| --- | --- |
| Committing actions: paying, booking, placing orders or bids, sending or posting, inviting, deleting, cancelling, changing account settings, granting access | Always a one-time approval showing exactly what will happen; "always allow" is not offered. In the browser a model judges every click, typed entry and Enter in its page context: a small decision model when one is installed (`models.decision`, e.g. tev1 or clef-flash via Ollama's decision API), otherwise the assistant's own model. If the judge cannot answer, you are asked. Payment and password fields, web chat clients and labels too long to show are always treated as committing, whatever the judge says. |
| Leaking your data | Injected instructions can only leak your data when one task has both read your private data (memories, mail, calendar, notes, contacts) and read third-party content. In such a task, the first time data could leave towards a destination (typing into a site, opening a URL the model wrote, running code with open network access) you approve sharing with that destination once for the task. Links the page itself offers are not model-written URLs. |
| Prompt injection from web pages, mail, notes, other people, sandbox output | Untrusted output is wrapped as `<untrusted>`, screened by the decision model when one is installed, and **taints** the run. A tainted run cannot reuse your standing grants for committing or lasting actions: messages, connector writes, schedules, deleting memories and sandbox egress ask again. Memories it saves are kept as *unverified* and are recalled later as untrusted data. Schedules created by a tainted run keep running tainted; MCP clients keep what they read for 30 minutes. |
| The model approves its own actions | Approvals are decided only through the owner API (CLI, web UI, paired channel). The model never sees an approval endpoint. |
| Passwords and one-time codes | The model may not type into password or OTP fields. It hands off to you in the visible browser window. |
| Using your browser | A dedicated profile in your installed Chrome/Brave/Edge, visible so you can watch and take over, controlled over `--remote-debugging-pipe` (no TCP port exists for anything else to reach). No JavaScript evaluation, cookie or storage access is exposed; downloads and file pickers are blocked; `file:`, `chrome:` and LAN addresses are refused; web chat clients count every click and keystroke as committing. Prefer stricter? Add `{"kind": "browser.navigate", "target": "*", "decision": "ask"}` to `policy.rules` to approve every new site. |
| Using your everyday browser (optional, `coderunner browser connect`) | The CodeRunner extension lets the assistant use only the tabs in its "CodeRunner" tab group (and popups those tabs open); your other tabs are never listed, read or controlled, and dragging a tab out of the group takes it back. The extension reaches the daemon through a native messaging host registered for its ID alone, over a unix socket only your account can open, with a token. The same approvals, step judge, password and payment rules and request guard apply. What differs from the dedicated profile: the group's tabs have your sign-ins and cookies; permission prompts (camera, location...) are shown to you instead of being denied; the protocol is limited to the page domains the assistant needs (no cookie, storage or browser-wide access); a popup's very first request can leave before the guard attaches to it. Chrome shows that CodeRunner is debugging the browser while the assistant works. |
| Sandbox exfiltration | Network mode `proxy`: the VM is on a host-only network whose only way out is the egress proxy. Destinations are allow-listed or approved per host; LAN, loopback, link-local and tailnet (CGNAT) addresses are refused; connections go to the address that was checked; denials are remembered for five minutes and at most five network approvals can wait at once. Mode `none` has no egress at all (the secret-injecting fetch is disabled too). |
| API keys | Stored on the host. Sandbox code writes `{{secret:name}}` in a header; the gateway substitutes it only over https, only for hosts you named (and the header you named, if any), and strips it from responses. |
| Connectors | Off until enabled one by one. Writes only create calendar events (no invitations), reminders, notes and mail drafts; they run without a prompt unless the task has read untrusted content. Mail can only create drafts. One-time codes and reset links are redacted from mail. Scripts get data as arguments, never as code. |
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

## Your browser: its own profile, or a tab group in your Chrome

By default the assistant starts your Chrome (or Brave, Edge, Chromium) with a profile of its own: no
sign-ins, nothing shared with your everyday browsing. Some sites treat that browser as automated
(it is started with a debugging pipe, so `navigator.webdriver` is true) and refuse it: booking.com's
flight search answers its destination suggestions with an error there.

`coderunner browser connect` switches to your everyday browser instead, the way browser assistant
extensions do it: the assistant opens its tabs in a "CodeRunner" tab group, in a window of its own,
and can use only the tabs in that group. To set it up:

1. `coderunner browser connect` registers the extension's native messaging host with each installed
   Chrome-family browser (or `--browser chrome|brave|edge|chromium`) and switches `browser.mode` to
   `chrome`.
2. In Chrome, open `chrome://extensions`, turn on Developer mode and choose "Load unpacked" with the
   folder it prints (`~/.coderunner/assistant/chrome/extension`).
3. Restart the assistant (`coderunner down && coderunner up`). `coderunner browser status` shows
   whether the extension is connected; the extension's badge reads "on".

`coderunner browser disconnect` goes back to the dedicated profile. Chrome needs to be open for the
assistant to browse; if it is not, the assistant says so. Windows is not supported yet.

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
mail and message bodies, contact lookup. Writes: calendar events (without
invitations), reminders, notes, and mail drafts saved to Drafts; they ask first
only after the task has read untrusted content. There is no send tool. Calendar queries miss recurring series that started before the range.

## Your files

`coderunner files add ~/Documents` shares a folder with the assistant. The
daemon indexes it in the background and keeps it current:

* Text, code and documents are indexed by meaning and by exact words. PDFs use
  poppler's `pdftotext`; Word, RTF and Pages use macOS `textutil`.
* Images, audio and video are indexed by meaning with a multimodal embedding
  model (`files.embedding`, e.g. EmbeddingGemma 2 in Ollama). Video is indexed
  as sampled frames plus its sound.

The assistant finds files with `files_search`, then uses them with `file_read`,
`file_view` (images and video frames, for vision models) and `file_listen`
(speech, with a speech-to-text engine installed). Rules:

* Only shared folders are read. Hidden folders, `~/Library`, key, certificate
  and credential files, `.env` files and CodeRunner's own state are never
  indexed or read, and symlinks out of a shared folder are not followed.
* File contents are your private data and can contain third-party text, so a
  task that reads files counts as holding private data and as having read
  untrusted content (see the leak rule above).
* `coderunner files status | index | search "query" | remove <folder>`.

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
coderunner files add ~/Documents     # let it find and use your files
coderunner browser connect           # browse in a CodeRunner tab group in your own Chrome
coderunner channels enable telegram && coderunner channels pair telegram
coderunner remote enable --owner you@example.com && coderunner remote pair
coderunner schedule add --name brief --spec "weekdays 08:00" --prompt "Summarise my day"
coderunner service install            # start at login
coderunner mcp                        # use the guarded tools from Claude Code / Claude Desktop
```

Configuration lives in `~/.coderunner/assistant/config.json`; state, logs and
the browser profile live next to it.
