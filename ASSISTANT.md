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
   its quantized build loses. Speed counts next, as the time of one agent
   step: reading what changed (about 600 new tokens: the runtime keeps the
   earlier turns), writing a tool call (about 100 tokens) and carrying it out
   (about 2.5 s for the browser and the commit judge), as measured in the
   sample tasks below. Prompt reading is bound by compute and often takes
   longer than writing, so generation speed alone misjudges models. The
   balance rule is the same on every machine: a step within 15 s
   (`models.step_budget_s`) costs nothing; each doubling beyond it costs 0.25
   of ability, about what separates Qwen3.5-9B from a 27B model at 98% of
   its scores; steps over 90 s rank last; utilities within a tenth are a tie
   that the faster model wins. Measured sample-task results replace the
   estimates: models that passed more tasks on this kind of machine rank
   first, and a run's seconds over the steps its tasks need replace the
   estimated step. Models with
   no published scores come last. The scores ship with the catalog and are
   refreshed with discovery each day; `models discover` and `models plan`
   show them. The top candidates are downloaded and tested on the machine
   itself until one passes. It then
pulls it through Ollama, or straight from Hugging Face as GGUF for llama.cpp
when you prefer, starts the runtime and runs a smoke test (exact tool
arguments, tool choice, plain answers, and image understanding) before it
declares success. If a model fails the test, the next candidate is tried.
4. Setup then times the model on this machine: a fresh page of about 3,900
   tokens and a 128-token answer. The measurement replaces the spec-sheet
   numbers (memory bandwidth per chip, a rough compute factor per
   accelerator) for every candidate on this machine. If the model is more
   than 1.5x slower than estimated and another candidate now ranks clearly
   higher, setup moves on to it and deletes the weights it downloaded for
   the first. `coderunner models bench` repeats the measurement;
   `coderunner models bench --report` prints a summary to share (hardware,
   runtime, measured and estimated speeds, the plan), with no names, paths
   or tokens.
5. `coderunner models bench --tasks` measures the served model end to end:
   the real assistant (instructions, tools, browser, file search, the commit
   judge) does sample tasks against local stand-ins for what owners ask, in
   a headless browser and a throwaway home folder, so the owner's browser,
   memories and files are untouched. The tasks: a flight search with
   booking.com's hurdles and page sizes (round trip preselected, airports
   that need a pick from their suggestions, a two-month calendar, results in
   "best" order behind filters, 15 at a time); the cheapest direct evening
   flight; two facts from two pages of a library website; a warranty date
   that needs two of the owner's documents; and a care instruction that
   needs the appliance's model from the documents, then the maker's support
   site. Answers are generated from the date and route, so the checker knows
   them and no model can have learned them. Each result records whether each
   task passed, its time, model requests and tokens, keyed by machine profile
   (chip, memory, GPU), model build (file or tag and quantization) and
   runtime version, in `assistant/models/tasks.json`. Speed per step alone
   misjudges models: on an M3 Pro, Qwen3.5-9B answers in 5 s against 15 s for
   Bonsai 2 27B, but needed twice the steps on the flight search and took
   longer overall. `coderunner models bench --tasks --candidates 2` measures
   the two best candidates not measured on this kind of machine yet, one
   model at a time, sets up the best by the measurements and deletes what it
   downloaded for the others. Setup unloads the previous model before it
   loads another (two at once froze an 18 GB Mac). A run in which the model
   wrote at under half the speed setup measured for it (with the owner's
   apps open it was waiting for the disk: 0.06 tokens/s against 22) is
   marked as slowed; it counts until a run that was not slowed replaces it.

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
  checksum. Where a maker publishes how much of the full model's scores a
  build keeps, that figure sets its loss instead of the generic one for its
  bit width: Bonsai 2 27B (1.72 bits per weight) keeps 98.2% of
  Qwen3.8-27B's scores across 14 benchmarks in Prism's measurement (84.78
  against 86.32; a conventional 2-bit IQ2_XXS build keeps 84.1%), so it
  leads on 18 GB Macs and up, ahead of Qwen3.5-9B. The 1-bit Bonsai 27B is
  scored on its own published results (60.5 on MMMU-Pro and 87.1 on AIME
  2026, against 75.8 and 94.1 for Qwen3.6-27B), which keep it below
  Qwen3.5-9B; it stays available (`coderunner models setup --model
  bonsai-27b-1bit`). Speeds measured on an M3 Pro: the 1-bit 27B generates
  14.6 tok/s and reads prompts at 81 tok/s; Bonsai 2 27B 11-13 and 65-75,
  so its agent steps take about 20 s against 8-12 s for Qwen3.5-9B. Reading the
  prompt is the slow part on this class of chip, so the agent keeps its
  prompts append-only for the runtime's prompt cache (Qwen3.5 and the
  Bonsai models are hybrid models, whose cache can only resume where the
  previous prompt ended).
* Backends on Apple silicon, measured with Qwen3.5-9B (Q4_K_M, or MLX
  4-bit) on an M3 Pro: prompt reading is a tie between Ollama (254 tok/s at
  4K tokens, 236 at 15K), llama.cpp b11564 (262 and 242) and mlx-lm 0.32
  (247 and 253); MLX generates about 15% faster (25.6 against 21-22 tok/s),
  which is about 6% of an agent step. CodeRunner stays on llama.cpp (Ollama
  runs llama.cpp's server for GGUF models) and adds no MLX runtime. Ollama's
  own MLX-engine build (`qwen3.5:9b-nvfp4`) ran out of memory on a
  15K-token prompt on 18 GB.

* Models do not sit in memory unused: after `models.idle_unload_minutes`
  (10) without requests, llama.cpp's servers sleep (`--sleep-idle-seconds`)
  and CodeRunner's Ollama unloads them (`OLLAMA_KEEP_ALIVE`; the Ollama app
  does so after 5 minutes). Bonsai 2 27B held 10 GB of an 18 GB Mac and 0.5
  GB asleep; the next request wakes it in about 4 s and reads the
  instructions again (37 s for a 27B on an M3 Pro, once per wake). Setup
  unloads the previous model before it loads another.

* On Apple silicon chat models run on CodeRunner's own llama.cpp server
  (llama.cpp's official build, pinned and checksum-checked, installed by
  setup), Ollama keeping the embedding and decision models: Ollama's server
  keeps up to 8 GB of earlier prompts in memory with no setting for it (2.4
  GB in one sample-task run on an 18 GB Mac, which then wrote 0.06 tokens/s),
  and its plain tags became MLX builds that took 16.3 GB for Qwen3.5-9B. On
  NVIDIA, llama.cpp's official build is CPU-only, so Ollama serves them.
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
  `{"host", "port", "memory_gb"}`, each running llama.cpp's
  `ggml-rpc-server` (`rpc-server` in older releases). This is
  experimental and untested on real hardware. The RPC link has no
  authentication or encryption, so use it only on a direct cable or a
  private network you control.
* With discovery and published scores today (October 2026) the planner
  picks, by its estimates:

  | Machine | Assistant model |
  | --- | --- |
  | 8 GB Mac | Qwen3.5-4B compact, text only, 8K context |
  | 16 GB M1 | Qwen3.5-9B (Bonsai 2 steps would take about 46 s, against 23 s) |
  | 16 GB M2 and newer | Qwen3.5-9B (32 s against 17 s a step: a tie the faster model wins) |
  | 18-128 GB Macs | Bonsai 2 27B (Prism's llama.cpp build); on 32 GB and up it ties with Qwen3.8-27B, whose scores it keeps 98.2% of, and its steps are faster |
  | 12-24 GB NVIDIA | Bonsai 2 27B (Prism's CUDA build), 16K context on 12 GB |
  | 32 GB NVIDIA | Qwen3.8-27B (steps as fast as Bonsai 2's: its scores break the tie) |
  | DGX Spark | Qwen3.8-27B (Prism has no CUDA build for ARM) |
  | 256-512 GB Mac Studio | MiMo-V2.6-Flash (llama.cpp) |

  `coderunner models bench --report` prints what a given machine measures
  and picks; reports from other machines replace these estimates.

  A larger model with weaker published scores is not picked: Granite 4.2 30B
  scores below Qwen3.5-9B on MMLU-Pro (77.6 against 82.5) and far below
  Qwen3.8-27B on Terminal-Bench 2.1 (29 against 73).

* The sandbox runs the lite profile with the assistant: code runs as plain
  processes in a work folder per session (files persist, variables do not),
  with no Jupyter server and no browser inside the VM, which is capped at 1 GB
  and used about 170 MB in testing. The assistant browses with your own
  browser on the host. `./install.sh --profile full` keeps Jupyter kernels and
  the headless browser for MCP clients that use them.

* The browser is read as text from Chrome's accessibility tree over CDP, with
  numbered elements that keep their numbers while the page changes (new pages
  continue the numbering, so a stale number fails instead of hitting another
  element). Each click, typed entry, key press, choice or scroll returns only
  what it changed on the page, or the whole page after it navigates, so a
  local model does not spend a step and a page read after every action.
  Vision models can also take screenshots. With `agent.thinking` set to
  `auto` (the default) the model reasons before the first step and after a
  failed one, and answers routine steps directly; `on` and `off` force it.

* Embeddings for memory and files: EmbeddingGemma 2 (text, images and audio
  in one space) through Ollama, also when the assistant model runs elsewhere
  (`models.embedding_url`). On Apple silicon (macOS 15+) it runs on the
  Neural Engine through Core ML instead (FluidInference's build, served by
  CodeRunner on 127.0.0.1:8084 in an environment of its own): text, and
  pictures and sound through the build's own vision and audio encoders
  (a picture at the model's default 280 tokens, about 0.5 s on an M3 Pro;
  sound 20 s per embedding). Without those encoders pictures and sound go
  to the build in Ollama, whose vectors share the space. Against Google's
  full-precision model the Core ML text vectors agree to a cosine of 1.000.
  Measured on an M3 Pro over 821 passages of CodeRunner's own files:

  | | Ollama (GPU) | Core ML (Neural Engine) |
  | --- | --- | --- |
  | Time | 53.3 s | 40.3 s |
  | GPU busy | 76% | 1% |
  | Power above idle | 8.8 W | 3.6 W |
  | Energy per passage | 570 mJ | 178 mJ |
  | Right file first, of 20 questions | 12 | 12 |

  A bigger Mac's GPU may be the faster one, so setup times both on the
  machine and keeps Core ML unless it is more than 10% slower or its vectors
  disagree (mean cosine under 0.97); then the GPU build does everything and
  the Core ML download is removed. `coderunner models bench --embedding`
  measures again; `models.embedding_runtime` (`auto`, `coreml`, `gpu`)
  overrides it. The Core ML encoder reads at most 512 tokens, so longer
  text is read in windows whose vectors are averaged. Setup compiles the
  three lengths it uses for the Neural Engine (about 85 s on an M3 Pro, once;
  Core ML keeps the result); after a restart, or after
  `models.idle_unload_minutes` releases the model, it is back in 1-1.5 s.
  On 64 questions over a real Downloads folder (documents, notes, code,
  screenshots, photos, videos) search found the right file first as often
  with Core ML pictures and sound as with Ollama's (mean reciprocal rank
  0.76 both), with the GPU idle.
* Fast "System 1" models share the runtime: an embedding model for memory
  search, and a decision model (`models.decision`, served through Ollama's
  decision API) that judges whether a browser step commits something and
  screens untrusted text for instructions aimed at the assistant. Setup
  installs one when it fits well beside the chat and embedding models (24
  GB Macs and up), chosen by accuracy on 66 labelled browser steps: tev1 4B
  missed no committing step and asked on 7 of 39 safe ones (0.5 s each);
  laya (421M) missed 7 and asked on 21; tev1 0.8B asked on 35 of 39.
  Otherwise the assistant model judges browser steps itself (Qwen3.5-9B
  missed 1 of 27, an injected label, with 3 false prompts; Bonsai 2 missed
  none with 6) and untrusted text is not screened (it is still wrapped and
  taints the task). A judge call on the assistant's own model keeps the
  agent's cached conversation: Ollama keeps llama.cpp's prompt cache, and
  CodeRunner's llama.cpp servers give the judge its own slot (with one slot,
  every step after a judge call re-read the whole conversation). The judge's
  question sits with its instructions and the step comes in a message of its
  own, so the runtime reads only the step: hybrid models such as Bonsai 2
  resume only where the last message starts, and a 27B on an M3 Pro took
  4 s a click re-reading the question, 1.3 s now. `tests/judge_eval.py`
  measures a judge on 30 labelled browser steps (Bonsai 2 27B: 29 right,
  the miss a flight result's "Select", which asks the owner). There are no
  keyword lists: these judgements come from models.
* EmbeddingGemma 2 needs Ollama 0.36.0 and decision models 0.35.0. When the
  installed Ollama is older, the installer upgrades a Homebrew install, and
  on macOS setup otherwise downloads Ollama 0.40.3 into
  `~/.coderunner/runtime` (checked against its published checksum) and
  leaves the Ollama app alone; if the app is serving an older version, setup
  asks you to update or quit it.

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

### A browser driver for website tasks

`coderunner models driver` adds a second model trained to use websites from screenshots: Microsoft's
Fara1.5 (MIT; 80.8% on WebVoyager for the 4B, 86.6% for the 9B, as Microsoft reports). The assistant
hands it whole website tasks with the `browser_task` tool and gets its report back. On 11 captured
states of a booking.com flight search, Fara1.5-4B chose the right next step every time and read the
three cheapest flights with their prices from the results; a step took 7.5 s (median) on an M3 Pro,
against about 43 s for Bonsai 2 27B reading the page as text.

The driver acts through the same broker as the assistant: a point on the screenshot is turned into the
element under it, which is then clicked with the usual checks, so policy, the step judge, approvals and
the password and card rules all apply; a point with nothing actionable under it is refused. The driver
also stops before irreversible steps on its own and asks.

Where the driver and the assistant's model do not both fit in memory, they take turns: the assistant's
model is unloaded while the driver works (the step judge asks the driver meanwhile) and loaded again
after. Each turn costs a model load (Fara1.5-4B 10-20 s, Bonsai 2 27B 4-19 s on an M3 Pro) and the
assistant's model reading its conversation again. Estimated from memory needs:

| Mac | Assistant's model | Driver |
| --- | --- | --- |
| 16-36 GB | Qwen3.5-9B, Bonsai 2 27B or Qwen3.8-27B | Fara1.5-4B, taking turns |
| 48 GB and up | Qwen3.8-27B | Fara1.5-9B, loaded alongside |

On an 18 GB M3 Pro, Bonsai 2 27B and Fara1.5-4B loaded together ran the GPU out of memory, which is why
they take turns there. 24 GB is just over the estimate (16.5 of 16 GB) and may fit both; not yet tried.
`coderunner models driver --model fara1.5-9b` picks a model by hand, `--off` switches the driver off.

## Channels

Only paired owners can instruct the assistant; everything else is ignored and
audited without content. Approvals arrive in the same chat (`approve <id>
[once|session|day|always]` / `deny <id>`, or buttons on Telegram). No channel
needs a public webhook.

| Channel | Setup | Notes |
| --- | --- | --- |
| Telegram | Create a bot with @BotFather, `coderunner channels enable telegram` (stores the token as a secret), restart, `coderunner channels pair telegram`, send `/pair <code>` | Private chats only by default; voice notes are transcribed locally once voice is set up (see Voice) |
| iMessage | `brew install steipete/tap/imsg`; sign Messages on this Mac into a dedicated Apple ID; grant the daemon Full Disk Access and Automation → Messages; enable and pair | Messages.app may fetch link previews for URLs in replies, so a URL written by the model can be fetched from your Mac |
| Signal | Link signal-cli as a secondary device and run `signal-cli -a +NUM daemon --http=127.0.0.1:8080`; `coderunner channels enable signal --set url=http://127.0.0.1:8080 --set account=+NUM` | CodeRunner does not launch signal-cli |
| WhatsApp (experimental) | `brew install openclaw/tap/wacli`, `wacli auth` with a secondary number; enable with `--set accept_tos_risk=true` | Uses the unofficial linked-device protocol; the account can be banned. The official Cloud API does not allow general-purpose assistants |

## Voice

Speech is turned into text on this computer. `coderunner setup` (or `coderunner
voice setup` later) prepares it:

* Macs with Apple silicon and macOS 15 or later, for 25 European languages
  (English, German, French, Spanish, Italian, Portuguese, Dutch, Polish,
  Russian, Ukrainian and others): Parakeet Ultra on the Neural Engine, in
  FluidInference's Core ML build (Moondream's post-training of NVIDIA's
  Parakeet TDT 0.6B v3, CC-BY-4.0). Of the Core ML speech models it scores
  best on the Open ASR Leaderboard's English sets (5.80% average WER;
  Parakeet TDT v3 6.26%, Parakeet Unified 6.27%, the latter under NVIDIA's
  Open Model License). Setup downloads FluidAudio's pinned source release
  (Apache-2.0), checks it, builds a 10 MB helper with the Xcode command line
  tools (about a minute and a half) and keeps only the helper; then the model
  (631 MB, each file checksummed). Core ML compiles the model for the Neural
  Engine once during setup (about 20 s; again after a macOS update). After
  that each recording is one short process that loads the model in about
  0.1 s, so nothing stays in memory between messages and the GPU stays free
  for the chat model.
* macOS 26 and later: Apple's on-device speech model (SpeechAnalyzer), through
  a small helper built with the Xcode command line tools. Nothing to download
  for languages macOS already has; it transcribes one set language
  (`--language en-IN`, otherwise the Mac's own) from 30 that Apple supports.
* Everywhere else, and for languages neither has (Hindi, for example):
  Qwen3-ASR on llama.cpp's official build, both downloaded and checksummed into
  the CodeRunner folder (1.7B, about 2.5 GB; 0.6B, about 1 GB, on 8 GB
  machines). It detects the language itself. Of the open speech models
  llama.cpp runs, it scores best on the Open ASR Leaderboard (5.75% average
  English WER; Whisper large v3 7.84%). Setup downloads it only when neither
  engine above serves the configured language.

Engines are tried in that order (`voice.stt: auto`; `voice.stt:
parakeet-coreml`, `apple` or `qwen3-asr` picks one); one that lacks the
language or fails hands over to the next. Voice setup also times Parakeet and
Apple speech on one sentence spoken by `say` and saves the result with the
machine's profile in `models/speech.json`: on a machine where Parakeet took
more than 0.5 s longer, Apple speech goes first. A bigger GPU speeds up only
Qwen3-ASR, which comes last for the languages Parakeet knows (see below), so
it does not change the order.

Measured on an M3 Pro (18 GB, macOS 26.5), one process per recording as
CodeRunner runs it: 50 LibriSpeech test-clean and 50 test-other utterances,
40 segments of an Earnings-22 call (accented, conversational), the same call
as one 4-minute recording, 15 FLEURS German and 10 FLEURS Hindi sentences.
WER after lowercasing and removing punctuation and fillers (numbers as
written, so "1855" against "eighteen fifty five" counts as three errors).
Energy is CPU, GPU and Neural Engine power above idle (macmon), per minute of
audio.

| | Parakeet Ultra, Core ML | Apple speech | Qwen3-ASR 1.7B, llama.cpp |
| --- | --- | --- | --- |
| WER test-clean / test-other | 1.87% / 2.84% | 1.95% / 4.07% | 1.06% / 2.96% |
| WER Earnings-22 segments / 4-minute recording | 8.33% / 7.44% | 11.16% / 11.61% | 8.48% / 5.51% |
| WER German / Hindi | 2.98% / no | 3.31% / no | 3.64% / 12.84% |
| WER, all short recordings (3,012 words) | 3.69% | 4.71% | 3.49% |
| time per second of audio | 0.026 s | 0.033 s | 0.22 s |
| 4-minute recording | 1.1 s | 3.8 s | 29.8 s |
| text of a 3-10 s voice message after | 0.18 s | 0.19 s | 1.37 s |
| energy per minute of audio | 9 J | 10 J | 121 J |
| runs on | Neural Engine | Neural Engine and CPU | GPU (5 W) |

Parakeet is more accurate than Apple speech (1.0 points fewer errors over all
short recordings; 95% bootstrap interval 0.2 to 1.9) at the same speed.
Against Qwen3-ASR the difference over all short recordings is within noise
(+0.2 points, interval -0.5 to +1.0): Qwen3-ASR was better on clean read
English (+0.8 points, interval +0.2 to +1.6) and on the one long recording,
the two were level on the rest, and Parakeet takes an eighth of the time and
a thirteenth of the energy, without loading a second model on the GPU next to
the chat model.

Talk to the assistant with the 🎤 button in the web page (it can read replies
aloud with the browser's on-device voices) or `coderunner chat --voice`
(replies spoken with `say` on macOS). Voice notes from paired chat apps are
transcribed the same way, and `file_listen` transcribes audio and video in
shared folders. `coderunner voice transcribe <file>` checks the setup. Spoken
replies to chat apps use macOS `say`, `piper` or `kokoro`; ffmpeg converts
formats (setup installs it with Homebrew when missing).

## Connectors (macOS)

`coderunner connectors enable <calendar|reminders|notes|mail|contacts>`, then
`coderunner connectors test <name>` from Terminal to grant the Automation
permission. Reads: calendar events, reminders, note search and read, recent
mail and message bodies, contact lookup. Writes: calendar events (without
invitations), reminders, notes, and mail drafts saved to Drafts; they ask first
only after the task has read untrusted content. There is no send tool. Calendar queries miss recurring series that started before the range.

## Your files

`coderunner files add ~/Documents` shares a folder with the assistant. The
daemon indexes it in the background and keeps it current, and
`coderunner find "boarding pass for the Delhi trip"` searches it from the
terminal by describing what you want.

* One multimodal embedding model (`files.embedding`, EmbeddingGemma 2 on
  the Neural Engine or in Ollama) places names, text, pictures and sound in
  one space. Every file is findable by its name; text by passage (with its
  line); PDFs page by page, scanned pages read with on-device OCR; pictures
  by what they show and by the text in them (screenshots, photos of
  documents); audio in overlapping 30-second windows (silence skipped);
  video by a frame a second (up to 32, spread evenly over a longer video),
  by the whole video (the mean of its frames) and by its sound. Results
  point at the page, line or moment that matched. Source code is searched
  with the model's code-retrieval prompt.
* Words are matched too (full-text index), so names, codes and numbers
  ("HX-55821") are found exactly. Text made mostly of figures (data exports)
  is matched by words only.
* Search reads each similarity against its kind for that query (names,
  passages, pictures, sound), so a photo that matches well is not buried under
  text that matches a little; asking for a kind ("photo of...", "...in a
  video") favours it; weak results are cut rather than padded out.
* Fast: the daemon keeps a compact code of every piece in memory (about 100
  bytes each) and the model loads on demand (it is freed after
  `models.idle_unload_minutes`); a search compares the query with all of them,
  then scores the closest few hundred exactly. Vectors are stored at half
  precision. Without the daemon, `coderunner find` searches directly.
* Kept current: on macOS new and changed files are picked up within seconds
  (file events); elsewhere the folders are walked every `files.scan_interval`.
  Indexing waits on a battery below `files.min_battery` (30%) and in Low
  Power Mode; on battery, pictures, sound and video wait for the charger
  while text goes on (`files.media_on_battery`; `files.media_waits` holds
  them whatever the power). A different embedding model means indexing
  again; a new picture or sound encoder for the same text model re-embeds
  only pictures, sound and video.
* Files kept only in iCloud are indexed by name and never downloaded (set
  `files.cloud_files` to read them); folders kept only in iCloud are not
  listed. Source code in git checkouts is left to code tools
  (`files.code_in_repos`). Spoken words in recordings become searchable text
  with `files.transcribe` (slow; uses the speech-to-text engine).
* On macOS, PDF pages, OCR and file events come from a small Swift helper
  built on first use (Xcode Command Line Tools); without it PDFs need
  poppler's `pdftotext`, and there is no OCR. Word, RTF and Pages use macOS
  `textutil`; media use ffmpeg.

The assistant finds files with `files_search`, then uses them with `file_read`,
`file_view` (images and video frames, for vision models) and `file_listen`
(speech, once voice is set up). Rules:

* Only shared folders are read. Hidden folders, `~/Library`, key, certificate
  and credential files, `.env` files and CodeRunner's own state are never
  indexed or read, and symlinks out of a shared folder are not followed.
* File contents are your private data and can contain third-party text, so a
  task that reads files counts as holding private data and as having read
  untrusted content (see the leak rule above).
* `coderunner files status | index | search "query" | remove <folder>`;
  `coderunner find "query" [--kind image] [--json]`.

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
