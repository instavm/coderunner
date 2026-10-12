# Branch validation and limits

## Automated tests

Run everything with:

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -v
```

CI (`.github/workflows/tests.yml`) runs the suite on Ubuntu and macOS with
Python 3.10 and 3.12, plus shell syntax checks. The real-browser tests are
skipped there (`CODERUNNER_SKIP_BROWSER=1`).

| Area | What is covered |
| --- | --- |
| Sandbox auth (`tests/test_auth.py`) | Anonymous, wrong, duplicated and rotated credentials; weak or missing configuration fails closed; WebSocket cannot bypass |
| Policy (`test_policy.py`) | Defaults, deny rules, block lists, secret host binding, grant scopes, taint overriding grants, high-risk kinds never getting standing grants, timeouts, listener failures |
| Egress (`test_egress.py`) | Proxy credential, CONNECT tunnel, absolute-form HTTP without proxy headers, unapproved hosts, LAN/loopback/link-local/CGNAT refusal, secret substitution only for allowed hosts and never echoed back |
| Gateways (`test_gateway.py`) | Sandbox surface exposes only model and fetch; owner surface needs the admin token or a paired cookie; pairing codes are single use and burn after failures; CSRF header and Origin checks; Tailscale owner restriction |
| Agent (`test_agent.py`) | Tool calls, text-form tool calls from local models, invalid arguments fed back, step limit, untrusted wrapping, injection flagging, taint |
| Browser (`test_browser.py`) | URL and host rules, policy kinds per action, password and card fields, accessibility snapshots and change detection, plus an end-to-end run against the installed Chrome (headless) including a redirect to a private address being blocked and no leftover processes |
| Everyday browser (`test_chrome.py`, `test_chrome_e2e.py`) | Extension ID pinned by its key; the native host relays both ways, answers commands too large for the browser and refuses other extensions; the daemon socket is 0600 and refuses a wrong token or origin and a second browser; the broker drives a fake extension through the bridge, says how to connect when it is missing, and stops when it goes away; `browser connect` registers the host for installed browsers only. With Chromium or Chrome for Testing (`CODERUNNER_CHROME_E2E=/path/to/chrome`): the real extension opens the assistant's tab in a CodeRunner group, types, clicks, answers a confirm(), takes a screenshot, puts a page's popup into the group, blocks a link to a private address and a cross-site frame from a blocked host, hides and refuses the owner's other tab (and the Storage domain), lets go of a tab dragged out of the group and of idle tabs (attaching again on the next step), and leaves the browser open on stop. `CODERUNNER_E2E_HEADFUL=1` adds a normal window with no debugging switches (navigator.webdriver is false); `CODERUNNER_E2E_LIVE=1` adds booking.com's destination suggestions |
| Daemon (`test_daemon.py`) | Real HTTP servers: owner chat, memory recall across conversations, sandbox model proxy, credential separation, scheduled delivery, and the agent driving the real browser where a "Buy now" click waits for an owner approval over the API |
| Models (`test_models.py`) | Hardware parsing (Apple, NVIDIA, AMD, CPU, WSL2), budgets, KV-cache sizing, picks per memory tier from 8 GB Macs to 512 GB, expert offload on NVIDIA, runtime routing to llama.cpp, cluster flags, GPU-limit notes, licence resolution through base models, resumable verified downloads, smoke-test pass/fail including the wrong-argument case below; timing a served model (streamed and llama.cpp timings), calibration per machine, setup moving on from a model far slower than estimated and deleting its weights, the shareable report; the decision model only where it fits; published retention for Bonsai 2; Ollama version checks and its own checksum-verified copy |
| Discovery (`test_discover.py`) | Hugging Face listings, configs and file trees from a fake hub: filters, sharded builds, KV and MoE sizes, build choice per machine, known breakages, stale caches; published scores read from eval results, ranking by them (a model scored like Granite 4.2 30B ranks below Qwen3.5-9B), quantization losses, GPU-limit notes |
| Files (`test_files.py`) | Only shared, non-secret files; iCloud-only files named but not read (read once on the Mac or with `files.cloud_files`); bundles, libraries and code in source checkouts skipped; half-precision vectors and sign codes; embedding batched across files (153 files in at most 12 requests); a new embedding model means indexing again; file events bring changes in first; pausing on a low battery or Low Power Mode; figures matched by words only; passages with lines; ranking on a calibrated corpus (a picture ranks by its own scale, a kind hint, weak results cut, codes lean on words); the first pass finds the closest codes; `coderunner find` output and direct fallback. With the Swift compiler: PDFs by page, OCR of a screenshot, FSEvents within seconds |
| Voice (`test_voice.py`) | Engine order (Apple speech, Qwen3-ASR, installed engines), languages Apple lacks and failing engines handing over, 16 kHz mono WAV for every engine, the Apple helper built once per source, llama.cpp checked against its checksum before unpacking, Qwen3-ASR downloaded only when Apple cannot serve the language; the web page's voice upload (size limit, temporary files removed) |
| Memory, scheduler, connectors, channels, service, remote | Unit tests with fakes; see each test file |

## Manual checks on this branch

* `install.sh` passes `bash -n` and `shellcheck -S warning`; option parsing,
  mount validation and token generation/reuse were exercised in isolation.
* `coderunner models plan` on an 18 GB M3 Pro chose `qwen3.5-9b` (Q4, 32K
  context, vision, ~8.1 GB of a 9.7 GB budget) with `qwen3-embedding-0.6b` for
  memory.
* The model runtime was exercised against a real Ollama server (start, alias
  with context length, stop). No model was pulled: the test machine had less
  than 20 GB free.

### Measured on an 18 GB M3 Pro (October 2026)

* Sandbox: `./install.sh --build` with Apple `container` 1.5.0; `test-e2e.sh` 12/12; the sandbox
  can call the local model through the gateway with its token, and cannot reach the owner API,
  the admin port, Ollama or the egress proxy port.
* Qwen3.5-9B via Ollama: smoke test 4/4 at 12.6 tok/s; memory recall across conversations; a
  page read through the real browser; code run in the sandbox.
* Bonsai 2 27B via Prism's llama.cpp build: smoke test 4/4 at 9 tok/s.
* Step judge on 66 labelled browser steps (27 committing, 39 not, 5 with injected text):
  keywords missed 6 commits; Qwen3.5-9B missed 1 (an injected label) with 3 false prompts;
  tev1-4B with keywords missed none with 10 false prompts; Bonsai missed none with 6.
* Files: EmbeddingGemma 2 ranked the right image, voice note, video (frames and sound), Word
  document and text file first for plain-language queries; the assistant answered "what is the
  total on my invoice" from an image with no prompts.
* booking.com flight search (BLR to DEL), Qwen3.5-9B: approval prompts fell from 11 to 0-2 with
  the new rules, but the task was not completed within 30 steps; Bonsai on 18 GB swapped with
  the browser and embeddings loaded (4 steps in 18 minutes).

### Later runs on the same Mac

* Bonsai builds measured: 1-bit 27B 14.6 tok/s generation and 81 tok/s prompt reading; Bonsai 2
  27B PQ2_0 9.0 and 70; PTQ1_0 6.9 and 62. All passed the smoke test with vision. As step judge,
  Bonsai 2 missed no committing step with 6 false prompts; the 1-bit model missed one (an injected
  "Send") with 12 false prompts.
* Lite sandbox in a 1 GB VM: code, files across runs, sessions and token checks work; about 170 MB
  in use.
* booking.com flight search with Bonsai 2 (21 steps, 32 minutes) and the 1-bit 27B (30 steps,
  18.5 minutes, reached the Bengaluru to New Delhi route page): no unwanted prompts and correct
  judge decisions, but neither returned the three cheapest flights. Prompt reading at 60-80 tok/s
  is the bottleneck on this chip.
* Live model discovery found 31 recent models from trusted publishers; on 18 GB it proposed
  MiMo-V2.6-Distill-Qwen-9B, Granite 4.2 8B and Mage-VL besides the catalog's picks. With the
  specialist filter, 22 remain; published scores were read for 25 repos, and every discovered model
  that fits 18 GB scores below Qwen3.5-9B. (Superseded: with Prism's published 98.2% retention,
  Ternary Bonsai 2 27B ranks above Qwen3.5-9B on 16-24 GB.)
* Everyday-browser mode with Chrome for Testing 155 in a temporary profile: all of the above pass
  (three runs). In a normal window with only the extension, booking.com's flight form opened, "Going
  to" expanded, and typing "Delhi" listed DEL Delhi International Airport, Hindon and Noida, with no
  403 and no "Something's not right"; `navigator.webdriver` was false. The suggestions arrive about
  a second after the typing; the broker now waits for the page to settle (up to 3 s), and a fixture
  whose suggestions appear 0.8 s after typing shows them in the typing step's result.
* Dedicated profile on booking.com: Chrome started with `--remote-debugging-pipe` reports
  `navigator.webdriver` as true, and the site's bot filter (AWS WAF) answers the destination
  autocomplete with 403, so no model can finish a flight search there. Qwen3.5-9B with the new
  browser loop took 30 steps in 9 min 47 s (32 minutes before) with no approval prompts, but kept
  retrying the destination field for that reason.
* booking.com flight search completed end to end by a local model: Ternary Bonsai 2 27B (Prism
  llama.cpp, 18 GB M3 Pro) in the everyday-browser mode (Chrome for Testing with the extension)
  set one-way BLR to DEL on 20 November 2026, sorted by price, and read the three cheapest flights
  (SpiceJet 6:05-9:00 at INR 9,994, Akasa Air 5:00-7:55 at INR 10,363, SpiceJet 21:15-00:10 at
  INR 10,442) from screenshots: the result cards kept the price in a region's name and the old
  renderer dropped repeated airline names. Both now read as text (the whole results page is 7,671
  characters). 29 steps in 30.5 minutes, one needless approval prompt (opening a price
  breakdown), nothing booked.
* File search eval (`python tests/files_eval.py`, EmbeddingGemma 2 in Ollama): 16 natural-language
  queries over notes, data, PDFs, screenshots, a picture and a spoken memo among 70 distractors,
  all answered first (R@1 1.00): a PDF page ("how much is the rent"), a policy number, two
  screenshots by their OCR text, the memo by its sound, the red picture by its look.
* Real folders (Documents, Desktop, Movies; 169,051 files, 163,806 of them only in iCloud and
  indexed by name without downloading): walk plus the first 2,000 files 83 s; 54,000 files in 30
  minutes, 30 files/s, limited by embedding (32 inputs/s while a chat model shared the GPU and
  Spotlight was re-indexing). Search over 55,925 vectors in memory: 20 ms median (26 ms p95) once
  the query is embedded. Embedding a new query takes 27 ms median (44 ms p95) on a quiet machine,
  so a new search answers in about 50 ms; with a chat model generating it took 241 ms. Database
  2.7 KB per piece; 445 MB peak memory for the indexing process. iCloud-only names now enter the
  keyword index at once and are embedded after the files on the Mac.
* Ollama embeds short texts at 39/s one at a time and 80/s in batches of 64-256 (quiet M3 Pro);
  more sequence slots (OLLAMA_NUM_PARALLEL 8) were slower (46-63/s). Indexing a large folder faster
  needs a runtime that embeds many sequences in one pass (llama.cpp's server), not yet tried.
* Speech to text on 20 LibriSpeech clips (188 s of read English): word error rate 2.0% for Apple's
  on-device model, 2.5% for Qwen3-ASR 1.7B and 4.0% for 0.6B (Q8_0, llama.cpp b11564). Apple needed
  no download (English was installed) and builds its helper in about 6 s. An 84 s recording came
  back as timed phrases from Apple and as three pieces cut at pauses from Qwen3-ASR, with no words
  lost at the cuts; a WebM/Opus recording sent to the web page's voice endpoint came back word for
  word. Qwen3-ASR twice returned no transcript while other heavy jobs ran; that now counts as a
  failure and the next engine is tried. Speed was not measured cleanly (Spotlight was re-indexing
  after a restart, load 5-15): under that load a clip of about 9 s took about 0.4 s with Apple
  (real-time factor 0.04), 1.7 s with Qwen3-ASR 1.7B (0.18) and 1.1 s with 0.6B (0.12), each call
  starting its engine.

### Backends, prompt cache and decision models (October 12, same Mac)

Qwen3.5-9B at about 4 bits, no reasoning, nothing else loaded (tokens per second; a fresh page each
time, so no prompt cache helps; the other track's test suite was using one CPU core):

| Backend | Read 3.7K tokens | Read 15.3K | Write (after 3.7K / 15.3K) | Tool call |
| --- | --- | --- | --- | --- |
| Ollama 0.40.3 (llama.cpp runner, Q4_K_M) | 254 | 236 | 22.2 / 19.9 | ok |
| llama.cpp b11564, CodeRunner's flags | 262 | 242 | 21.3 / 19.3 | ok |
| llama.cpp b11564, `-b 2048 -ub 2048` | 219 | 202 | 20.8 / 19.4 | ok |
| mlx-lm 0.32, mlx-community 4-bit | 247 | 253 | 25.6 / 22.9 | ok |
| Ollama 0.40.3 MLX engine (`qwen3.5:9b-nvfp4`) | 22.1 s for the 3.8K request (llama.cpp: 23.4-24) | out of memory after 6m51s | | not reached |

Prompt reading is a tie; MLX writes about 15% faster, about 6% of an agent step. CodeRunner stays on
llama.cpp and adds no MLX runtime.

Prompt cache in an agent loop (6 append-only steps of about 800 new tokens; tokens the server
processed per step, from its log):

| Setup | Without judge calls | With a same-model judge call between steps |
| --- | --- | --- |
| llama.cpp, one slot, `--cache-ram 0` | ~810 (3.7 s) | whole prompt, 1.5K to 4.8K (6 to 19 s) |
| llama.cpp, `-np 2 --kv-unified` | | ~820 (3.6 s) |
| Ollama 0.40.3, one slot (its runner keeps llama.cpp's RAM prompt cache) | | ~815 (3.7 s) |

Qwen3.5 is a hybrid model: llama.cpp can only resume its cache where an earlier prompt ended, so a
plan reminder appended to each request and dropped from the next forced a full re-read every step
(seen in the Ollama log as "forcing full prompt re-processing"). CodeRunner's llama.cpp servers now
run two slots.

Decision models on the 66 labelled browser steps (judged as CodeRunner does: ask when P(commit) +
P(unclear) reaches the threshold):

| Model | Threshold | Missed commits (of 27) | False prompts (of 39) | Median |
| --- | --- | --- | --- | --- |
| tev1:4b (mxfp8, 4.4 GB) | 0.7 | 0 | 7 | 0.5 s |
| tev1:4b | 0.9 | 2 | 1 | 0.5 s |
| laya (421M) | 0.7 | 7 | 21 | 0.05 s |
| laya | 0.5 | 1 | 38 | 0.05 s |

`coderunner models bench --report` against Prism's server with Bonsai 2 27B (PQ2_0, 32K context; load
average about 18 from other work during the run): prompt reading 68 tok/s (estimate 69), generation
11.2 tok/s (estimate 9.0), 42.7 s per agent step (estimate 45.5), first token 58.6 s after a fresh
3,984-token page. The plan on this Mac picks Bonsai 2 27B on Prism with EmbeddingGemma 2 from Ollama;
Qwen3.5-9B (about 19 s per step) is the first Ollama alternative.

Setup installs tev1 4B where it fits beside the chat and embedding models (24 GB Macs and up). It
does not fit on 18 GB, where the chat model judges. nimble (9B) and clef-flash (9B, 11 GB) were not
measured: they cannot sit beside a 27B or 9B chat model on this Mac.

## Not yet validated

* Voice: Qwen3-ASR on Linux and NVIDIA machines; the 🎤 button in a real browser and
  `coderunner chat --voice` with a real microphone (both ask for microphone permission); setup's
  Homebrew install of ffmpeg; Apple speech for languages other than English.

* The everyday-browser extension in branded Chrome with the owner's real profile (loaded by hand),
  Brave and Edge, and on Linux; Windows is not supported (no native host registration in the
  registry).
* Discovered models on large machines (192-512 GB Macs, multi-GPU, DGX Spark clusters) and the
  on-device ranking among several downloaded candidates.
* Microsoft Fara1.5-9B (screenshot-based browser specialist) as a second browser driver.
* Starting the sandbox VM on demand on 16-18 GB machines, to leave room for a bigger model.
* Model picks for machines other than the 18 GB M3 Pro are the planner's estimates from file sizes
  and memory bandwidth (prompt reading: a factor per accelerator, measured only on this M3 Pro):
  llama.cpp next to Ollama, experts in RAM (`--cpu-moe`) on NVIDIA cards, the GPU-limit note on big
  Macs, and the per-tier picks in ASSISTANT.md have not been run on that hardware.
  `coderunner models bench --report` on those machines would replace the estimates.
* `coderunner models bench` ran end to end against a real server (above); setup's own timing and its
  fallback to the next model were tested with fakes only.
* tev1 4B as the decision model beside a chat model on a 24 GB or larger Mac.
* Ollama's own copy (setup downloading 0.40.3 when the installed one is older) was tested with
  fakes; the download and checksum were not run.
* Pooling several machines (`models.cluster`, llama.cpp RPC), e.g. DGX Sparks, has not been run.
* Sandbox images: the installer now pulls `instavm/coderunner:0.2.0` and builds from the checkout
  when that tag is not published yet (the published `latest` images ignore the token). Publishing
  is prepared but not done: `gh workflow run docker-build.yml -f version=0.2.0` once the workflow is
  on the default branch.
* Proxy network mode assumes the host is reachable from an `--internal`
  Apple `container` network at the network's gateway address. Verify that the
  sandbox can reach `:8223`/`:8225` and nothing else.
* Linux/Docker and WSL2 installation paths have not been run.
* macOS connectors were compiled and argument handling tested, but not run
  against Calendar, Reminders, Notes, Mail or Contacts.
* Telegram, iMessage, Signal and WhatsApp adapters were tested against fakes
  of their documented interfaces only.
* Tailscale Serve was not enabled for real.

## Earlier local-inference experiment

A standalone run of llama.cpp b11146 with Qwen3-1.7B Q8_0 (reasoning disabled,
one slot, 4096 context, four threads) reached ~66 generated tokens per second
but failed correctness: a synthetic `report_sum` tool call returned 13 instead
of 10. The model manager therefore never selects chat models below ~4B
parameters and refuses to finish setup until a model passes the tool-call smoke
test.
