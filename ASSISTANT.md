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

## Search mode

`./install.sh --search` (or `coderunner setup --search`) installs the index and the search tools only: the
embedding model on the Neural Engine (Ollama on Linux), the daemon as a login service, no sandbox VM and no
chat model. It is for an agent you already use, over MCP (below); `coderunner chat` says so and stops. Add a
local model later with `coderunner setup`, or the sandbox with `./install.sh`. `sandbox.driver` is `none` and
`models.chat` empty in this mode.

## What runs where

```
 your devices ──(Tailscale, optional)──▶ owner UI / API  127.0.0.1:8224   admin token or paired device
                                              │
 ┌──────────────────────── host (trusted) ────┴──────────────────────────────┐
 │ coderunner daemon: agent loop · policy + approvals · audit log · secrets   │
 │   ├─ model runtime (llama.cpp; Ollama on Linux) 127.0.0.1 only            │
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
* **Sandbox VM** (Apple `container` on macOS, Docker elsewhere) is the
  assistant's own Linux computer (Debian, as root): `run_shell` is a bash that
  keeps its directory, variables and functions between commands, anything it
  installs stays, and servers or long builds run as background jobs whose
  output it reads with `shell_job`. It can call the local model and an allow-listed fetch
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
* Models are ranked across runtimes. Ternary Bonsai 2 27B runs on llama.cpp
  itself on Apple silicon: the packing llama.cpp reads (Q2_0, 7.63 GB) is
  published only in Ollama's registry, which serves it as a plain download
  that setup checks against its SHA-256 (Ori's finding; neither Ollama nor
  Prism's fork is involved), with the maker's vision projector from Hugging
  Face. The registry is slow by the connection (0.15 MB/s each on a line
  that carries 3 MB/s), so that file comes in twelve ranges at once. On an
  18 GB M3 Pro it passed the tool and vision checks and was timed at 69
  tokens a second reading and 11.6 writing, 20 s for an agent step (0.87
  and 1.07 of the estimate from memory bandwidth; the catalog carries the
  two factors). It needs about 11 GB of memory free to load there. Prism
  ML's other builds (the 1-bit 27B, 3.8 GB, and Bonsai 2's denser PTQ1_0
  and PQ2_0 packings) run on Prism's llama.cpp build, which setup downloads
  pinned to a release and checks against its published checksum; it is used
  only for what the main runtime cannot serve. Where a maker publishes how much of the full model's scores a
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

* On Apple silicon there is one runtime: CodeRunner's own llama.cpp server
  (llama.cpp's official build, pinned and checksum-checked, fetched by setup
  when no `llama-server` is installed) serves every model, and embeddings
  and speech run on Core ML. Ollama is neither installed nor used there
  unless you name it (`coderunner setup --runtime ollama`); the same shape
  as Ori, which bundles llama.cpp's server and embeds on Core ML. Ollama's
  server keeps up to 8 GB of earlier prompts in memory with no setting for
  it (2.4 GB in one sample-task run on an 18 GB Mac, which then wrote 0.06
  tokens/s), its plain tags became MLX builds that took 16.3 GB for
  Qwen3.5-9B, and a first run fetched it (0.17 GB) and a second copy of the
  embedding model through it (1.33 GB) that Core ML does not need. The
  decision model, which only Ollama serves, is therefore set up on a Mac
  only when Ollama is the runtime; otherwise the assistant's model judges
  browser steps. Elsewhere llama.cpp's official build is CPU-only, so
  Ollama serves the models (its Linux build carries CUDA).
* A second CodeRunner home (a test run, a bench) can serve its own models
  beside the first: `models.port_base` (8081) is llama.cpp's port, and
  Prism's build, the browser driver and the Core ML embedding server take
  the three after it.
* Model files are kept in `assistant/models` under the CodeRunner home, or
  in the folder `models.dir` names (another disk, say); the app's setup
  screen offers the choice before anything is downloaded. Files already
  downloaded are not moved when the folder changes.
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
  | 18-128 GB Macs | Bonsai 2 27B (llama.cpp); on 32 GB and up it ties with Qwen3.8-27B, whose scores it keeps 98.2% of, and its steps are faster |
  | 12-24 GB NVIDIA | Bonsai 2 27B (Prism's CUDA build), 16K context on 12 GB |
  | 32 GB NVIDIA | Qwen3.8-27B (steps as fast as Bonsai 2's: its scores break the tie) |
  | DGX Spark | Qwen3.8-27B (Prism has no CUDA build for ARM) |
  | 256-512 GB Mac Studio | MiMo-V2.6-Flash (llama.cpp) |

  `coderunner models bench --report` prints what a given machine measures
  and picks; reports from other machines replace these estimates.

  A larger model with weaker published scores is not picked: Granite 4.2 30B
  scores below Qwen3.5-9B on MMLU-Pro (77.6 against 82.5) and far below
  Qwen3.8-27B on Terminal-Bench 2.1 (29 against 73).

* The sandbox runs the lite profile with the assistant: a persistent bash per
  session (`run_shell`; a command past its timeout is stopped, not the shell),
  detached jobs for servers and long work, and `run_python` for quick scripts,
  with no Jupyter server and no browser inside the VM, which gets 2 GB and 4
  CPUs (`./install.sh --memory 4g --cpus 8` for more; `sandbox.memory_gb`
  tells model selection what it takes). The image keeps its build tools,
  Java, ffmpeg, Node and Playwright browsers, so most work needs no install.
  Apple's container service is started with the sandbox when it is stopped
  (as after a restart). The assistant browses with your own browser on the
  host. `./install.sh --profile full` adds Jupyter kernels and the headless
  browser for MCP clients that use them.

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
  in one space). On Apple silicon (macOS 15+) it runs on the Neural Engine
  through Core ML and nothing else is fetched (FluidInference's build,
  served by CodeRunner on 127.0.0.1:8084 in an environment of its own);
  elsewhere, or with Ollama named as the runtime, through Ollama
  (`models.embedding_url`). On Core ML: text, and
  pictures and sound through the build's own vision and audio encoders
  (a picture at the model's default 280 tokens, sound 20 s per
  embedding). Without those encoders pictures and sound go to the build in
  Ollama, whose vectors share the space. Against Google's full-precision
  model the Core ML text vectors agree to a cosine of 1.000. Pictures are
  a little slower on the Neural Engine than on this Mac's GPU, but take a
  third of the energy and leave the GPU to the assistant model (48
  pictures, M3 Pro):

  | | Ollama (GPU) | Core ML (Neural Engine) |
  | --- | --- | --- |
  | Time a picture | 0.41 s | 0.53 s |
  | Chip power | 16.3 W | 5.4 W |
  | Energy a picture, above idle | 5.6 J | 1.7 J |

  A search does not wait for a picture being indexed: the encoders run
  beside the text model (5 ms a query on an M3 Pro, instead of up to 0.5 s).
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
  screens untrusted text for instructions aimed at the assistant. Where
  Ollama is the runtime, setup
  installs one when it fits well beside the chat and embedding models (24
  GB and up), chosen by accuracy on 66 labelled browser steps: tev1 4B
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
* The installer needs Python 3.10 or later for the host assistant and macOS
  ships 3.9: where no newer one is found it downloads Python 3.12.16
  (python-build-standalone's build, the one uv installs; pinned and checked
  against the release's checksums) into `~/.coderunner/runtime`, 67 MB.
* Nothing to run models with needs installing first. On a Mac setup
  fetches llama.cpp's pinned build (30 MB) when there is no `llama-server`.
  Elsewhere, where neither Ollama
  nor llama.cpp is installed, setup downloads its own copy of Ollama 0.40.3
  into `~/.coderunner/runtime`, checked against its published checksum (167
  MB on macOS; 1.4 GB on Linux, where the archive carries NVIDIA's CUDA
  libraries; it is a zstd archive, unpacked by the zstandard package that
  comes with CodeRunner on Linux, so no `zstd` command is needed); llama.cpp's and Prism's builds
  come the same way when a model needs them. EmbeddingGemma 2 needs Ollama
  0.36.0 and decision models 0.35.0: when the installed Ollama is older, the
  installer upgrades a Homebrew install and setup otherwise fetches the same
  copy, leaving the Ollama app alone; if the app is serving an older
  version, setup asks you to update or quit it. AMD's ROCm libraries (a
  separate archive) are not fetched, and Windows has no build.
* A machine with no GPU CodeRunner can use (no Apple silicon, no NVIDIA or
  AMD card) is told so before anything is downloaded, with the speed to
  expect: in `coderunner models plan`, on the app's first-run screen and as
  setup's first line. Setup then goes on, on the processor.
* Downloads never take the disk under `files.min_free_gb` (5 GB, the floor
  file indexing also pauses at). A first run on an 18 GB Mac fetches 9.9 GB:
  the assistant's model (8.3 GB with its vision projector), the embedding
  model for Core ML (1.5 GB) and llama.cpp's build. Setup
  says what a model takes before it starts, skips one that would not leave
  the floor free and takes the best that fits, and says how much to free
  when none does; `coderunner models plan` notes when its pick does not fit
  the disk.
* Discovery reads each model's builds from Hugging Face's own list of its
  quantized builds, one request a model, and guesses the usual repository
  names only for a model that list does not know. Hugging Face allows 500
  API requests per five minutes to callers without an account, and setup
  needs some after discovery: when the limit is reached discovery stops and
  the catalog alone is used, and a request setup itself needs says when to
  try again (or set `HF_TOKEN`). `tests/setup_eval.py` measures a first run:
  what it fetches, how long each step takes, how many requests it makes and
  where it stops.

## Safety model

You are asked only about steps that commit something or could leak your data.
Ordinary work (opening sites, clicking through them, filling in a search,
picking dates and filters, running code, saving notes) runs without prompts.

| Risk | Control |
| --- | --- |
| Committing actions: paying, booking, placing orders or bids, sending or posting, inviting, deleting, cancelling, changing account settings, granting access | Always a one-time approval showing exactly what will happen; "always allow" is not offered. In the browser a model judges every click, typed entry and Enter in its page context: a small decision model when one is installed (`models.decision`, e.g. tev1 or clef-flash via Ollama's decision API), otherwise the assistant's own model. A chat model asked as the judge is first shown nine steps already judged (a search, a calendar's next month, an order, a message sent...), because small models asked cold call harmless steps committing. If the judge cannot answer, you are asked. Payment and password fields, web chat clients and labels too long to show are always treated as committing, whatever the judge says. |
| Leaking your data | Injected instructions can only leak your data when one task has both read your private data (memories, mail, calendar, notes, contacts) and read third-party content. In such a task, the first time data could leave towards a destination (typing into a site, opening a URL the model wrote, running code with open network access) you approve sharing with that destination once for the task. Links the page itself offers are not model-written URLs. |
| Prompt injection from web pages, mail, notes, other people, sandbox output | Untrusted output is wrapped as `<untrusted>`, screened by the decision model when one is installed, and **taints** the run. A tainted run cannot reuse your standing grants for committing or lasting actions: messages, connector writes, schedules, deleting memories and sandbox egress ask again. Memories it saves are kept as *unverified* and are recalled later as untrusted data. Schedules created by a tainted run keep running tainted; MCP clients keep what they read for 30 minutes. |
| The model approves its own actions | Approvals are decided only through the owner API (CLI, web UI, paired channel). The model never sees an approval endpoint. |
| Passwords and one-time codes | The model may not type into password or OTP fields. It hands off to you in the visible browser window. |
| Using your browser | A dedicated profile in your installed Chrome/Brave/Edge, visible so you can watch and take over, controlled over `--remote-debugging-pipe` (no TCP port exists for anything else to reach). No JavaScript evaluation, cookie or storage access is exposed; downloads and file pickers are blocked; `file:`, `chrome:` and LAN addresses are refused; web chat clients count every click and keystroke as committing. Prefer stricter? Add `{"kind": "browser.navigate", "target": "*", "decision": "ask"}` to `policy.rules` to approve every new site. |
| Using your everyday browser (optional, `coderunner browser connect`) | The CodeRunner extension lets the assistant use only the tabs in its "CodeRunner" tab group (and popups those tabs open); your other tabs are never listed, read or controlled, and dragging a tab out of the group takes it back. The extension reaches the daemon through a native messaging host registered for its ID alone, over a unix socket only your account can open, with a token. The same approvals, step judge, password and payment rules and request guard apply. What differs from the dedicated profile: the group's tabs have your sign-ins and cookies; permission prompts (camera, location...) are shown to you instead of being denied; the protocol is limited to the page domains the assistant needs (no cookie, storage or browser-wide access); a popup's very first request can leave before the guard attaches to it. Chrome shows that CodeRunner is debugging the browser while the assistant works. |
| Sandbox exfiltration | Network mode `proxy`: the VM is on a host-only network whose only way out is the egress proxy. Destinations are allow-listed or approved per host; LAN, loopback, link-local and tailnet (CGNAT) addresses are refused; connections go to the address that was checked; denials are remembered for five minutes and at most five network approvals can wait at once. Mode `none` has no egress at all (the secret-injecting fetch is disabled too). |
| API keys | Stored on the host. Sandbox code writes `{{secret:name}}` in a header; the gateway substitutes it only over https, only for hosts you named (and the header you named, if any), and strips it from responses. |
| Connectors | Off until enabled one by one. Writes only create calendar events (no invitations), reminders, notes and mail drafts; they run without a prompt unless the task has read untrusted content. Mail only creates drafts unless you switch sending on for it, and then every message is approved with its recipients and whole text; changing one of three Mac settings and dialling a call are likewise approved one at a time. One-time codes and reset links are redacted from mail. Scripts get data as arguments, never as code. |
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

After CodeRunner is updated, the daemon brings the extension's folder up to date, but Chrome keeps
running the old extension (also after a restart) until you press its reload arrow in
`chrome://extensions`; `coderunner browser status` says when the extension is older than CodeRunner.

Two things keep a small local model's browser steps from going to waste. An element a page draws again
(a calendar after "Next month") gets a new number; a step that still names the old one is taken to the element
with the same role and name when the page is the same and exactly one matches, and is refused as before
otherwise ("Unknown element", which was 8 of 13 failed browser steps in a day of local runs). And when the
model sends two single clicks or entries in a row on a page it has already read, the second result carries a
note to send what it already knows together in one `browser_batch`.

A local model's reply also has a length limit, `agent.step_tokens` (2,048 tokens; twice that where it may think
first; 0 for none). A model can fall into writing without end: one step of a flight search ran to 10,400
tokens, eight minutes, on Qwen3.5 9B. A reply that hits the limit is dropped whole and asked for again, shorter,
and after two of those the task ends in plain words.

### More than one task at a time

Tasks started while another is running do not share a page. Each has its own tabs: the tab it is on, the tabs
it opened (and the ones those opened), its own limit of three, and on the assistant's own computer a window of
its own. It cannot see, name or close another task's tabs, and the picture in its thread is of its own page.
When it ends, its older tabs close and the one it ended on is nobody's: a follow-up in the same conversation
goes on from there, and the next task may use or close it. Opening a task's thread in the app brings its window
to the front of the assistant's computer (`POST /api/focus`). Code each task runs in the sandbox has its own
session ("main" for the first place, "task2" for the second), so their variables do not meet.

On a local model two tasks run at once and the rest wait ("Queued" in the list, "Waiting its turn" in the
thread; `agent.parallel_tasks` changes the two, and a remote model runs four). They take turns at the model, one
reply at a time: a step is a reply and then a wait that needs no model (a page loading, code running, your
answer to an approval), so while one waits the other has the model. The task open in the app goes first. Each
names its own slot in llama.cpp's server (started with one slot per place and one for the step judge), so its
conversation stays read while the other has the model. Two tasks working on the same site leave two seconds
between one's step and the other's. A task that asks for you while another thread is open shows "Needs you"
in the list, and the computer panel says "Another task needs you"; what is running is read from the list as
well as from live events, so a window opened mid-task does not show the computer as idle.

The slots share the model's context (32K tokens), so they are kept small. The warm-up at start reads the
instructions once in the first slot and keeps that reading in a file (`run/slots/instructions-<model>.bin`, about
370 MB for Qwen3.5 9B); every task's slot is put back to it when the task starts and when it ends, about 0.1 s each,
including after the server has slept. The step judge has the slot after the tasks'. Each task keeps its
conversation within the room the others leave (the context less the judge's 3K and what the other slots hold,
counted from the server's own token counts): past 85% of it, old pages go, then other long results, and a step
whose results would not fit (six pages opened at once) has them cut. A task that still cannot fit says so in a
sentence instead of failing.

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
also stops before irreversible steps on its own and asks. It sees pages as it was trained to: laid out
at 1440 x 900 (one screenshot pixel per CSS pixel, also on a Retina screen) while it works, with only its
three most recent screenshots kept, under its own system prompt word for word.

Where the driver and the assistant's model do not both fit in memory, they take turns: the assistant's
model is unloaded while the driver works (the step judge asks the driver meanwhile, with the same worked examples: without them Fara1.5 4B asked the owner about 6 harmless steps in 10) and loaded again
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
only after the task has read untrusted content. Calendar queries miss recurring series that started before the range.

Three things act for you outside the browser, each off until you turn it on and each asked about every time
(one-time approval only; no rule or standing grant covers them):

* **Sending mail.** `coderunner connectors enable mail --send` adds `mail_send` beside the draft tool
  (`connectors.send` in the config; enabling Mail alone never allows sending, and `disable mail --send` takes
  it away again). The approval shows the recipients and the whole message. A recipient that is neither in what
  you wrote in the conversation nor in a result from your Contacts is pointed out in the approval ("Check ...:
  not in what you asked, nor in your contacts"), since text on a page or in a mail can ask for a message to be
  sent somewhere.
* **A few settings of this Mac.** `coderunner connectors enable settings` adds `settings_get` and
  `settings_set` for appearance (dark or light), sound volume and mute, and nothing else: no security, privacy,
  account, sharing or network setting can be reached. One setting per approval; the reply says what the Mac
  reports afterwards.
* **Dialling a call.** `coderunner connectors enable phone` adds `phone_call`: it hands a full phone number to
  FaceTime, which asks you to confirm, and you do the talking. Short numbers (emergency and service codes) are
  refused. A number that is not in your request or your contacts is pointed out like a mail recipient.

Texting someone else goes through the iMessage channel's `send_message`, approved per message, as before.
These three were written and tested against stand-ins for the apps and their scripts compile; they have not
been run against Mail, System Events or FaceTime on a real Mac.

Where a file tool is given a path that does not exist, it answers with the paths this task's own searches
returned that have a similar name, to choose from: a model that remembered a file's name but not its folder
otherwise guessed again.

## Your files, mail, messages, notes and calendar: one index

`coderunner files add ~/Documents` shares a folder with the assistant. The
daemon indexes it in the background and keeps it current, and
`coderunner find "boarding pass for the Delhi trip"` searches it from the
terminal by describing what you want. `coderunner files sources add mail`
(and `messages`, `notes`, `calendar`) adds the owner's apps to the same index.

* One multimodal embedding model (`files.embedding`, EmbeddingGemma 2 on
  the Neural Engine or in Ollama) places names, text, pictures and sound in
  one space. Every file is findable by its name, read with its kind ("HEIF
  image", "QuickTime movie"); text by passage (with its line); PDFs page by
  page, scanned pages read with on-device OCR; CSV and TSV files and Excel
  sheets as tables, their rows with their columns named and their columns of
  numbers totalled; PowerPoint by slide; Numbers, Keynote and Pages through
  the preview they carry; pictures by what they show and by the text in them
  (screenshots, photos of documents); audio in overlapping 30-second windows
  (silence skipped); video by a frame a second (up to 32, spread evenly over
  a longer video), by the whole video (the mean of its frames) and by its
  sound. Results point at the page, line or moment that matched. A passage
  with nothing to say (a note that reads "Untitled", the residue of markup,
  base64) keeps its words for keyword search but gets no vector: in a large
  index such passages sat near every question. Generated and minified code,
  lock files and source files over 350 KB are skipped, as code search engines
  do.
* Folders are results too: each is found by its name and the kinds of file
  in it ("Garden, a folder of images"), by the mean similarity of every file
  under it ("budget and tax stuff" finds Finance), and the folder the best
  results gather in is shown after the first of them, or first when the
  question asks for a group of things ("flower photos"). A name search that
  finds nothing tries the folder's parent ("audio downloads"), drops words
  for kinds of file ("costa rica pics"), then the nearest name word for a
  misspelt one ("personel").
* Search is hybrid, fused by reciprocal rank with nothing to tune: meaning
  (plain cosine across media; EmbeddingGemma 2's space is aligned, so a photo
  that matches well is not buried under text that matches a little, and a
  file ranks by its best piece), words (BM25 over the pieces holding *every*
  word of the question, which finds codes and numbers such as "HX-55821" and
  stays quiet for plain language), names (the same over each file's or
  folder's own name and kind), and the kind the question names ("photo
  of...", "the slides about...", "emails about..."), as one more ranking over
  files of that kind. Exact copies count once ("same file at ..."); a
  document saved twice and changed since, found by the same passage, is shown
  once with the other under "also at".
* Each file's contents are indexed once: a byte-identical copy (a second
  checkout, "name (1).pdf", an export unpacked twice) takes the pieces and
  vectors of the file already read, and only its name is embedded. A file
  that moved or was renamed keeps its pieces; only its name is embedded
  again when that changed.
* Fast: the daemon keeps a compact code of every piece in memory (about 100
  bytes each) and the model loads on demand (it is freed after
  `models.idle_unload_minutes`); a search compares the query with all of them,
  then scores the closest few hundred exactly. Vectors are stored at half
  precision. Without the daemon, `coderunner find` searches directly.
* Kept current: on macOS new and changed files are picked up within seconds
  (file events; a folder that appears is walked on its own); elsewhere the
  folders are walked every `files.scan_interval`. Indexing waits on a
  battery below `files.min_battery` (30%) and in Low Power Mode; on battery,
  pictures, sound, video and iCloud downloads wait for the charger while text
  and screenshots go on (`files.media_on_battery`; `files.media_waits` holds
  them whatever the power). A different embedding model means indexing
  again; a new picture or sound encoder for the same text model re-embeds
  only pictures, sound and video.
* Files kept only in iCloud are downloaded to be read when they are under
  `files.cloud_file_mb` (20 MB; `files.cloud_files` reads them all, 0 reads
  none) and given back to iCloud about a minute later, so the Mac's storage
  stays as it was; larger ones are indexed by name. Source code in git
  checkouts is left to code tools (`files.code_in_repos`); code that is
  indexed is titled with its path inside the checkout. Spoken words in
  recordings become searchable text with `files.transcribe` (slow; uses the
  speech-to-text engine).
* On macOS, PDF pages, OCR and file events come from a small Swift helper
  built on first use (Xcode Command Line Tools); without it PDFs need
  poppler's `pdftotext`, and there is no OCR. Word, RTF and Pages use macOS
  `textutil`; Excel and PowerPoint are read from their zipped parts; media
  use ffmpeg.

### Mail, Messages, Notes and Calendar

`coderunner files sources add mail|messages|notes|calendar` indexes the
owner's apps as entries beside the files, each with an address instead of a
path: `mail://inbox/4120`, `message://12/2026-11-02` (one conversation, one
day), `note://...`, `event://...`. Mail, Notes and Calendar are read through
the connectors (JavaScript for Automation; the connector must be enabled),
Messages through its database (`~/Library/Messages/chat.db`, which needs
Full Disk Access for the process that runs the daemon). Mail is read from
the inbox and sent mailboxes (`files.mail_mailboxes`), Calendar a year back
and ahead (`files.calendar_days`); the apps are listed again every
`files.sources_interval` seconds (900). One-time codes and sign-in links are
redacted before mail is indexed, as the mail tools do. The entries are found
by `files_search` (kinds `mail`, `message`, `note`, `event`) and read with
`file_read` by their address, from the index, without going back to the app.

What these entries hold is the owner's private data and third-party text:
a task that searches or reads them counts as holding that source (mail,
messages, notes, calendar) and as having read untrusted content, so the
leak rule above applies before anything leaves the Mac, and a read is
checked as the connector's own reads are (disabled connector: refused).
`coderunner files sources remove mail` forgets the entries.

The assistant finds files with `files_search`, then uses them with `file_read`,
`file_view` (images and video frames, for vision models) and `file_listen`
(speech, once voice is set up). Rules:

* Only shared folders are read. Hidden folders, `~/Library`, key, certificate
  and credential files, `.env` files, `*.noindex` folders and CodeRunner's own
  state are never indexed or read, and symlinks out of a shared folder are not
  followed.
* File contents are your private data and can contain third-party text, so a
  task that reads files counts as holding private data and as having read
  untrusted content (see the leak rule above).
* `coderunner files status | index | search "query" | remove <folder> | sources`;
  `coderunner find "query" [--kind image|folder|mail|...] [--json]`.

### From Claude Code, Codex, Cowork, OpenClaw and LM Studio

The same index, memory and guarded tools are an MCP server, so an agent you
already use can find and read local things, under the same approvals and
leak rule (what a client reads taints its later calls for 30 minutes):

* `coderunner mcp` is a stdio server with every tool; `coderunner mcp --tools
  search` offers only `files_search`, `file_read`, `file_view`,
  `file_listen`, `memory_search`, `memory_save` and `memory_forget`, for an
  agent that should find local things and nothing more.
* The daemon serves the same two sets over Streamable HTTP at
  `http://127.0.0.1:8224/mcp` and `/mcp/search` with the admin token as a
  bearer (`coderunner token --admin`), for clients that only speak HTTP and,
  over the tailnet, from another machine.

```bash
# Claude Code (and Claude Desktop / Cowork through claude_desktop_config.json)
claude mcp add coderunner-search -- coderunner mcp --tools search
# Codex CLI
codex mcp add coderunner-search -- coderunner mcp --tools search
# OpenClaw, LM Studio or anything with a remote MCP client: ~/.config/.../mcp.json
{"mcpServers": {"coderunner-search": {"url": "http://127.0.0.1:8224/mcp/search",
                                      "headers": {"Authorization": "Bearer <admin token>"}}}}
```

The stdio bridge holds the admin token, so configure it only for clients
you run yourself; the HTTP endpoint answers nothing without the token.

## Remote access

`coderunner remote enable --owner you@example.com` runs `tailscale serve` for
the owner UI on this machine's tailnet name (never Funnel) and prints the URL.
`coderunner remote pair` prints a single-use code to pair a phone or laptop
browser. Restrict tcp:443 on this device to your own login in your tailnet
policy.

## The app

`app/` is a desktop shell (Electron, macOS and Linux) around the page the daemon serves at `/app/`: tasks,
approvals, and the sandbox computer's screen to watch or take over (`coderunner browser computer` gives the
sandbox that screen).

The answer is a chat reply, not a report. The instructions ask for the finding first in a sentence or two,
plain sentences without headings or bold labels, a source linked by its name where it is mentioned, one
sentence for a caveat, no recap and no offer at the end, and give two short examples. The app (`render.js`)
shows it as messages: two or three short paragraphs are a message each, a longer text stays one; a table or a
list is a block of its own, and a comparison of four columns or more becomes a card a row, since such a table
does not fit a thread's width. A link is the page's name with a small mark for its site, the site's first
letter on a neutral tile: no icon is fetched, as a favicon service would be told every site an answer mentions.
Products or places come as a row of cards with picture, name, price and seller under the words, from the
structure the assistant gives with `show_items` (never cut from its prose, every field shown as text); a
picture is cut from the page as the sandbox browser already shows it, so nothing is requested for it.

A task reads as a conversation with a person, not as a list of tool calls. The thread holds what the owner
asked, what the assistant tells them on purpose while it works (a finding or a partial result, sent with its
`tell_owner` tool), its plan as a checklist that fills in, its questions and approval requests, pictures of
pages, and the answer with how long it took. One line says what it is doing now, in the owner's words ("Opening
booking.com", "Reading the results on the page"), without the names of tools or kinds of control. What the
model writes beside its tool calls is its own working and is not shown: "The browser didn't respond on the
first try, so I'll retry" used to be put in the thread. A task that cannot be finished ends with what was done,
what could not be and what to do next; one that breaks ends with a sentence a person would say, with the cause
in the audit trail (`task.failed`); one stopped by a restart says so.

Faults are met below the model where doing so is harmless. A page whose process died (Chrome's "Aw, Snap") is
closed as soon as the browser reports it. A browser call that fails is followed by a new connection and, if a
new tab still does not answer, by starting Chrome again (the sandbox's own through the daemon); a step that
only reads (open, read, find, list, look) is then done again and the model sees only its result, while a click
or typed text is not repeated: the model is told the page was reopened and to look again. No message to the
model names a protocol method or a timeout. The assistant's own tabs are capped at four, the oldest closed as
new ones open, and a finished task keeps only the page it ended on: after forty tasks the sandbox's Chrome had
fourteen pages in a 2 GB machine, the kernel killed one, and no page would open until Chrome was restarted by
hand. While the daemon itself is away (a restart, sleep) the app keeps what is on screen, says "Reconnecting…",
tries again quietly and picks everything up when it is back; the desktop app starts a daemon that died, unless
the owner stopped it with `coderunner down`, and then offers "Start it". The Computer panel says "Starting"
while the sandbox comes up and offers a button for one that is stopped. No raw error, exception name or
terminal command is shown anywhere in the app. Where it works on a web page, a picture of the page is put in the thread
after it opens one, when it asks for approval on one, and at the end: at most 8 a task and 20 seconds apart,
kept 14 days in `assistant/shots` and served on the owner surface at `/api/shots/<name>`.

Files can be attached to a task: the paperclip in the box where a task is typed, a file dropped anywhere on
the window, or a picture pasted into the box. Each is sent to the daemon as soon as it is chosen and waits
above the box as a small tile (a picture or a video's first frame, or a file's name and size) with a bar for
how far it has got; at most 8 to a message, each up to `files.max_file_mb`. In the thread a picture shows as
itself and opens large on a click, a video and a recording play in place, and anything else is a card with its
name and size. A message can be files alone.

The assistant is told each file's name, kind, size and place, and what reads it: it is shown pictures (and one
frame of a video, at 1 s) with the message itself, up to 4, where the model can see; `file_read` reads
documents and text, `file_view` other moments of a video, `file_listen` what is said in a recording or a
video. The files are kept in `assets/outputs/attached/<id>/` under the home, which the sandbox mounts, so code
run there works on them at `/app/uploads/outputs/attached/<id>/<name>`; a script that names the host's path is
given the sandbox's. The file tools read an attached file though it is in no shared folder, and nothing else
under the home. Code in the sandbox can write into that folder, so only a regular file that is the one thing
in its id's folder counts: a link left there is not followed. `POST /api/uploads` takes the file as the body
and its name in `X-File-Name`; `GET /api/uploads/<id>` serves pictures, video, sound and PDFs in place and
everything else only as a download, so an attached HTML page or SVG never runs as the daemon's own page;
`POST /api/tasks` takes their ids in `attachments`.

A file the task's code wrote to the sandbox's outputs folder (a chart, a sheet, a document) is shown under its
answer in the same way: a picture as itself, anything else as a card to download (`/api/outputs/<name>`, only
for plain files directly in that folder).

A web address in an answer is shown as where it leads (the site and the start of its path); the whole address
is the link.

An approval in the thread is one sentence with Allow and Deny; the step it is about (a script, a form) is two
small lines under it that open on a click, and once decided the card says how: Allowed, Declined, or not
answered in time. A data-sharing approval (the task has read your files or memories and wants to use what it
read on a site) also offers "Allow for this task", which covers every site that task opens: a research task
over six sites asked six times before. Nothing is stored for it; the next task is asked again.

The window opens at 1152 by 720. `coderunner app` opens it: with the checkout's
own Electron where `npm install` was run in `app/`, and otherwise with Electron 45.2.3 downloaded once into
`~/.coderunner/runtime` (121 MB, checked against the release's checksums), so no Node is needed. The app finds
the daemon of `CODERUNNER_HOME` and starts it with `coderunner up` when it is not running; the daemon keeps
running after the app quits. It is not a packaged application yet: it runs from the checkout.

Where the sandbox has no desktop yet, the Computer panel offers "Set up the computer": the same installation
as `coderunner browser computer` (Xfce, a VNC screen and Chrome inside the VM, a few minutes), with its latest
line shown; when it is done the daemon starts again with the browser on that computer and the screen connects
(`GET` and `POST /api/computer/setup`). The app's files are served with `Cache-Control: no-cache`, so a
browser does not pair an old page with a new script.

Where no model is set up yet, the app opens on a setup screen instead of tasks, with the window to itself (no
task list, no computer panel). Behind it, small hairline drawings of the owner's own things (a document with one line marked, a web page,
a day in the calendar, a voice memo, a conversation, a sheet, a photo, a letter) lie at three depths, the far
ones small and out of focus; they drift slowly and lean a little with the pointer, keep still for whoever
asks for reduced motion, and are left out of a narrow window. Before anything is fetched it shows, on one screen:

* this computer in four figures: chip, memory, graphics, and the seconds a step of a task will take
  (estimated from the chip until setup measures it);
* a model for each job, one line each: planning, vision, file search, voice and the safety check, with what
  each downloads ("Same model" where the planning model does the job, "Installed" where it is here already).
  The planning model's name opens the list of the models that fit, best first, each with the seconds a step
  takes and its download ("Installed" for one already here); choosing one makes the plan again for it, and
  the label above the list changes from "Recommended for this Mac" to "Your choice for this Mac";
* the download as a share of the free disk, what is left afterwards, and the folder the models go to, with
  "Change" (the system's folder dialog in the app, a path box in a browser). The folder is `models.dir`; a
  download that would take the disk under `files.min_free_gb` is refused here, with what to free;
* where the planning model needs more memory than is free at that moment, a line saying so (what it needs,
  what is free) before anything is started;
* one button, "Download and set up". "How these were chosen" opens the reasons, the speed estimate, the
  memory budget, the other models that fit and the list of downloads.

Setup looks at memory again before it loads a model (the kernel's own figure of what is free: the percentage
`memory_pressure` prints on a Mac, `MemAvailable` on Linux) and wants 1 GB left beside the model. A model that
suits the computer but not that moment is not loaded and not replaced for good by a smaller one: the app says
"Not enough memory free right now" with what it needs and what is free, "Try again", and the next model that
fits now as a second choice with its download size. Asked for by name (`--model`), a model is loaded anyway,
with a warning (one chosen on the app's first screen is not: the app says the memory is short). A setup started from the app also never goes on to download a model the first screen did not
show: when the chosen one does not work out it stops, says why in a sentence, and offers the next one.
`coderunner setup` in a terminal, which prints each step, still moves down its list.

Setup then shows the stages it has reached, the running one with a bar (a download with its percentage and
the time it has left, from how fast it has been going), in the words of that first screen
(downloading the planning model, testing tools and vision, measuring speed, file search, voice) and never a
file's name; setup's own lines are under "Details". Below them the app plays itself in small: a request is
typed and sent, the assistant says what it is doing, and what it found and its answer come in, for five kinds
of request (files, the web, a message it asks before sending, the day ahead, voice), one sliding in after the
other. It is the same setup as `coderunner setup`, run in the daemon, and it carries on if the
window is closed. When it is done the daemon starts again in place with the new model and the page reloads.
The API behind it, on the owner surface: `GET /api/setup` (state, stages, progress lines, the last plan),
`POST /api/setup` with `{"action": "plan", "model": ...}` (a model the owner chose among `choices`; without it
the recommended one), `{"action": "location", "path": ..., "model": ...}` (an empty path: the usual folder) or
`{"action": "start", "model": ..., "accept_license": ...}`.

Until setup has chosen a model the daemon starts no model runtime: an Ollama installed for something else is
left alone on a Mac that will use llama.cpp.

## Remote mode: Claude with your own Anthropic key

The assistant is built to run on local models. As a setting, its model can instead be Claude on Anthropic's
API, with a key of your own. **It is not private**: each request sends the conversation, the text of the pages
the assistant reads and the contents of files it opens to Anthropic. The app says so where it is turned on,
and for as long as it is on: the sidebar reads the model's name, "Anthropic" and "Not private" where it reads
"Local · <model>" otherwise, and the Models panel carries the warning with the way back.

```bash
coderunner claude on --key-file ~/anthropic-key    # or without --key-file: the key is asked for, unseen
coderunner claude status
coderunner claude off                              # back to the model on this computer (--forget deletes the key)
```

In the app: "Use an Anthropic key" under Models in the Computer panel, or "Use an Anthropic key instead" on
the first-run screen (nothing is downloaded then; file search and voice wait until a local setup is run).
"Use the local model" in the same place switches back. The daemon starts again in place after either.

What changes while it is on:

* Claude Haiku 5.5 (`claude-haiku-5-5`; `--model claude-sonnet-5-5` or `claude-opus-5-5` for the others)
  plans, reads pictures and screenshots, judges every browser step (the same question and worked examples as
  a local judge) and screens untrusted text. The Models panel names it for planning, web pages and the
  safety check, and `/api/status` reports it as `models.chat`, `models.driver` and `models.decision`.
* File search, memory recall and voice keep their local models and say "on this computer".
* The local chat model stays set up but is unloaded, and the browser driver is not used: `browser_task` is not
  among the tools.
* Approvals, the leak rule, the sandbox and the egress rules are unchanged.

The key is kept in `assistant/anthropic-key`, readable only by you; it is never written to the config, the
logs or the audit trail, and no message that reports a failure repeats it. Requests go through Anthropic's
own client library (installed into CodeRunner's environment the first time remote mode is turned on), with
thinking off and effort `medium`: the agent loop drops old screenshots and pages from earlier turns, and a
thinking block is valid only while everything before it is unchanged. The instructions and tools are cached,
and so is the conversation so far, so each step re-reads only what is new (the daemon's log has one line a
request: the model that answered, the request id and the token counts, never the text). A refusal, a reply cut
off at its limit, a rejected key, a rate limit or a network failure is said in the task's thread in a sentence.
`GET /api/remote` says how it stands; `POST /api/remote` takes `{"action": "on", "key": ...}` (or `"key_file"`)
or `{"action": "off"}`.

A small file that holds a credential (an API key, a token, a private key: told by its contents, whatever it
is named) is never read by the file tools and its contents are never indexed.

## Checking an install

`coderunner check` does each kind of task once on this install and checks the answer, so "does it work here"
has an answer that nobody has to read a transcript for. Every task goes the way one typed in the app goes: the
daemon's task runner, the model that is set up, the sandbox, the browser the assistant really uses (the
sandbox's own computer, or Chrome), with test files and test websites made for the run.

| Check | What it does | Passes when |
|---|---|---|
| `answer` | a plain question | the number is right |
| `picture` | a photo of a receipt, attached | the total is read off it |
| `document` | a PDF, attached | the lease's end date and notice period are given |
| `video` | a video with speech, attached | what was said is given, and `file_listen` was used |
| `code` | a sheet worked on in the sandbox, attached | the best month and the total are right, and `run_python` was used |
| `files` | two test documents shared and indexed for the run | both facts are found by search |
| `website` | facts from two pages of a test website | both facts are given |
| `flights` (long) | a flight search through forms, a calendar and sorting | the three cheapest fares are named |
| `booking` (long) | a booking taken to the page before payment | the cheapest flight's passenger page was reached, the payment page never |
| `appliance` (long) | a document, then the maker's website | the answer is the guide's, and the guide was opened |
| `errands` (long) | three errands in one request: a flight search, a website, the documents | all three are answered |
| `driver` (long) | the flight search again, handed to the browser driver, which works from screenshots | the three cheapest fares are named, and `browser_task` was used |

`coderunner check` runs the first seven (about five minutes with Qwen3.5 9B on an 18 GB M3 Pro), `--long`
adds the rest, names run only those, and `--last` prints the last result (kept in `assistant/check.json`). A
task whose need is not met here is skipped with the reason (a model that reads text only, no speech to text,
the sandbox down, no browser, no file search, no browser driver). Each task's line also says how many
approvals it asked for: the check answers them, an owner would have been asked. The exit code is 0 when everything that ran passed. In the app
the same thing is "Check" at the foot of the task list: one line per task with a tick or a cross and the time
it took, and a click on a line opens that task's thread.

For the run only, and put back at the end: a folder of test documents is shared and indexed; the address the
test sites are served on (the sandbox's gateway address when the browser is the sandbox's computer) is added to
`browser.allow`; and approvals asked by the check's own tasks are answered for the owner, allowed unless the
step would pay. Approvals of any other task are left alone. A check does not start while a task is running.
The API: `GET /api/check`, `POST /api/check` with `{"names": [...], "long": ...}` or `{"action": "stop"}`.

### Twenty tasks and five forms on real websites

`coderunner check --real` (the "Real websites" tab of Check in the app) hands the assistant twenty things people
give an assistant, on the real web: a week in Mauritius planned with flights, hotels and a day-by-day plan with
costs; a phone priced at three retailers; laptops under a budget; headphones according to two review sites; the
cheapest non-stop flight on a date; morning trains from a timetable; hotels under a nightly budget; restaurants
and repair shops in a neighbourhood; a forecast; a currency conversion; three countries compared; visa rules
from official sources; a price against a threshold; the day's headlines; a cart and a hotel reservation taken
to the step before sign-in or payment; a recipe turned into a priced shopping list; and two that work from
attached documents (a letter drafted from a lease and a receipt, a chart drawn from a sheet).

Real sites have no fixed answers, so each is graded on what can be told without reading the answer through:
the parts asked for are there, it links to pages the browser really opened (a link to a site it never visited
does not count), prices carry a currency, a task that must stop before committing says where it stopped, and no
committing step went through. A site that shows a CAPTCHA or turns automated visits away is recorded as
"blocked" beside passed and failed, and is never worked around. Every task carries the same rules in its own
text: no sign-in, no account, no payment, no personal details, a few pages a site. The run answers its tasks'
approvals as the check does, except that every committing step and every hand-over of the screen is refused.
Each line gives the time, the tool calls and the approvals asked; the result is kept in
`assistant/check-real.json`, and each task opens from its row on the Check screen (finished check tasks are not
listed among the owner's own).

Five more are public forms, listed apart: an eCourts case search for the district courts of Gautam Buddha Nagar
(state, district and court complex lists that depend on one another, a party name and a year), a
trains-between-stations form, a PIN code and a bank's IFSC code found through dependent lists, and a loan
calculator checked by the instalment it returns. A form is graded on what was chosen and typed in the page,
read from the browser's own record of the task and from the form's values at the end, not from the answer. The
eCourts search ends in a CAPTCHA: the assistant fills in the rest, hands the screen to the owner and stops; the
run declines the hand-over and records "needs you", which is neither a pass nor a failure.

To find a page the assistant searches the web (`browser_search`: the results one line each with the link, from
a second search site when the first answers with a challenge) instead of opening an address from memory. A
CAPTCHA is the owner's step, as passwords and one-time codes are: the browser refuses to type into a CAPTCHA
box, the assistant calls `browser_handoff`, the thread shows "Your turn on the computer" with the one thing to
do, and the large screen carries the same sentence with "I'm done"; the task goes on from the result.

`coderunner models bench --tasks` is a different measurement: it times a model on the same test sites in a
throwaway home with a headless browser, to compare models. `check` asks whether this install works.

## Everyday commands

```bash
coderunner app                       # the desktop app: tasks, the computer's screen, first-run setup
coderunner status | doctor | audit
coderunner check [--long]            # do each kind of task once here and say which worked
coderunner chat ["one-off question"]
coderunner approvals [approve|deny <id> --scope once|session|day|always]
coderunner grants [revoke <id>]
coderunner secrets set github --host api.github.com
coderunner connectors enable calendar && coderunner connectors test calendar
coderunner setup --search            # index and search tools only (no model, no sandbox)
coderunner files add ~/Documents     # let it find and use your files
coderunner files sources add mail    # index Mail (also messages, notes, calendar)
coderunner browser connect           # browse in a CodeRunner tab group in your own Chrome
coderunner channels enable telegram && coderunner channels pair telegram
coderunner remote enable --owner you@example.com && coderunner remote pair
coderunner schedule add --name brief --spec "weekdays 08:00" --prompt "Summarise my day"
coderunner service install            # start at login
coderunner mcp [--tools search]       # the guarded tools (or only search and memory) from Claude Code, Codex...
```

Configuration lives in `~/.coderunner/assistant/config.json`; state, logs and
the browser profile live next to it.
