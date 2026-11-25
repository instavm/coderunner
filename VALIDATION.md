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
| Models (`test_models.py`) | Hardware parsing (Apple, NVIDIA, AMD, CPU, WSL2), budgets, KV-cache sizing, picks per memory tier from 8 GB Macs to 512 GB, expert offload on NVIDIA, runtime routing to llama.cpp, cluster flags, GPU-limit notes, licence resolution through base models, resumable verified downloads, smoke-test pass/fail including the wrong-argument case below; timing a served model (streamed and llama.cpp timings), calibration per machine, setup moving on from a model far slower than estimated and deleting its weights, the shareable report; the decision model only where it fits; published retention for Bonsai 2; Ollama version checks and its own checksum-verified copy, fetched where no runtime is installed (a build for Macs and Linux, none for Windows; the Linux archive's layout and zstd unpacking); downloads skipped or refused when they would take the disk under its floor, and the plan saying so; a rate limit from Hugging Face named with when to try again, yesterday's cached answer served in its place, and a refused licence lookup failing that model instead of setup |
| Discovery (`test_discover.py`) | Hugging Face listings, configs and file trees from a fake hub: filters, sharded builds, KV and MoE sizes, build choice per machine, known breakages, stale caches; builds read from the Hub's list of a model's quantized builds with no guessed names, a rate limit ending discovery so the catalog alone is used; published scores read from eval results, ranking by them (a model scored like Granite 4.2 30B ranks below Qwen3.5-9B), quantization losses, GPU-limit notes |
| Files (`test_files.py`) | Only shared, non-secret files; iCloud-only files named but not read (read once on the Mac or with `files.cloud_files`); bundles, libraries and code in source checkouts skipped; half-precision vectors and sign codes; embedding batched across files (153 files in at most 12 requests); a new embedding model means indexing again; file events bring changes in first; pausing on a low battery or Low Power Mode; figures matched by words only; passages with lines; ranking on a calibrated corpus (a picture ranks by its own scale, a kind hint, weak results cut, codes lean on words); the first pass finds the closest codes; `coderunner find` output and direct fallback. With the Swift compiler: PDFs by page, OCR of a screenshot, FSEvents within seconds |
| Voice (`test_voice.py`) | Engine order (Parakeet on Core ML where setup did not measure it noticeably slower than Apple speech on this machine, Apple speech, Qwen3-ASR, installed engines), Parakeet only on Apple silicon with macOS 15+ and for its 25 languages, languages an engine lacks and failing engines handing over, 16 kHz mono WAV for every engine, the Apple helper built once per source, FluidAudio and llama.cpp checked against their checksums before use, Parakeet's licence checked before its pinned files are downloaded and only its helper kept after the build, Qwen3-ASR downloaded only when neither Parakeet nor Apple can serve the language; the web page's voice upload (size limit, temporary files removed) |
| The app from the command line (`test_appshell.py`) | The pinned Electron is the one `app/package.json` names, with a build for Macs and Linux and none for Windows; the checkout's own Electron is used before CodeRunner's copy; a download that does not match its checksum is refused; a zip is unpacked with its links and executable bits and cannot write outside its folder; the app is started on this home in a process of its own |
| Setup from the app (`test_setup_api.py`) | The plan as the app shows it (downloads, free disk, notes); setup as a background job with its progress kept, one at a time, a failure reported without restarting; the owner-only route; the daemon stopping shortly after a finished setup and starting again in place. The sandbox's computer: files copied and the script run through the driver with its lines passed on, the sandbox restarted and the browser moved there; a failed install, a sandbox that does not start and a missing script change nothing; its job and owner-only route |
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
* Sixteen everyday read-only tasks on real sites, one attempt each, 12 minutes at most (M3 Pro,
  18 GB, while files indexed and a Docker VM held ~4.7 GB): flights, trains, hotels, two shops, an
  encyclopedia, a code host, a news site, weather, a currency rate, a recipe, a film, a government
  fee, a returns policy, a map route and the Python docs. Qwen3.5-9B answered 12 right and 2 in
  part (a pre-release taken for the latest release; a route's time without its distance, which it
  said it could not see) and gave up on 2 (a two-shop price comparison, a fee calculator), in 52
  minutes; numbers in the answers were in the pages it read, and the answers checkable against an
  API (top news story, exchange rate) matched. Bonsai 2 27B ran out of time on the first two tasks
  (~60 s a step: 50 tokens/s reading, 9 writing) and its runtime then crashed under memory
  pressure, so the run was stopped. The run found and fixed: the chat command's daemon check (3 s
  sandbox probe), a daemon frozen 7.5 s while loading the file index, the extension's worker
  restart leaving every later step on a tab failing, a stopping daemon removing the next one's
  socket, and links copied from a page as paths being refused.
* Retrieval on a real Downloads folder (`python tests/retrieval_eval.py [BENCH]`, benches kept outside the
  repo: plain-language questions naming no person, number or company, answers stored as hashes of their
  paths). 64 questions, and 25 more written later on other files and held out from every choice; ~28,000
  of the owner's files, 292,000 pieces; EmbeddingGemma 2 on the Neural Engine for text, pictures and sound.
  Ranking methods compared on one frozen index (MRR, held-out / benchmark):

  | Ranking | Held-out | Benchmark |
  | --- | --- | --- |
  | Meaning, standardised per modality, plus the named kind (shipped) | 0.77 | 0.70 |
  | Same, plus word matches at weight 0.5 with gates and a cut (before) | 0.76 | 0.70 |
  | Same without the kind bonus | 0.72 | 0.69 |
  | Reciprocal rank fusion of meaning and BM25 | 0.50 | 0.57 |
  | Distribution-based score fusion of meaning and BM25 | 0.47 | 0.57 |

  Shipped, per kind: documents 0.91 / 0.85, videos 0.65 / 0.93, screenshots 0.48 / 0.59, photos — / 0.62,
  notes 0.72 / 0.67, pictures 1.00 / 0.80, sound 1.00 / —; right file first 0.68 / 0.61, in the top 5
  0.92 / 0.83. Misses are mostly near-twins (dozens of screenshots of one app, notes on one subject), a
  document about something outranking the picture of it, and pictures whose point is a shape (a giant
  digit). A search takes 155 ms (median). The two rank fusions failed on pictures, video and sound: words
  say nothing about them, yet each fusion gives the word ranking an equal say.
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
* Parakeet Ultra on Core ML against Apple speech and Qwen3-ASR 1.7B (October 14, same Mac, under
  the machine-wide measurement lock, power from macmon): 155 short recordings (LibriSpeech
  test-clean and test-other, an Earnings-22 call, FLEURS German), the call as one 4-minute
  recording and 10 FLEURS Hindi sentences; table in ASSISTANT.md (Voice). Parakeet 3.69% WER,
  Apple 4.71%, Qwen3-ASR 3.49%; text of a 3-10 s voice message after 0.18 s, 0.19 s and 1.37 s;
  9, 10 and 121 J per minute of audio. Voice setup built the helper and prepared the
  model in 105 s, timed Parakeet at 0.19 s and Apple speech at 0.17 s on a 4 s sentence and put
  Parakeet first; Hindi went on to Qwen3-ASR. Once, while other jobs ran, the
  4-minute recording took 32 s instead of 1 s with the Neural Engine idle; three repeats right
  after took 0.9 s.

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

### Browser drivers on twelve real-site tasks (October 19-20, same Mac)

`browser_task` run alone (the assistant's own model unloaded), one model at a time, each from a browser with
no cookies or site storage, on twelve read-only tasks: booking.com flights and hotels, Google Finance (two
quotes), Wikipedia, GitHub's latest release, Hacker News, xe.com, timeanddate, amazon.in, IMDb and Google
Maps. Answers were checked against APIs at task time where one exists (quotes, rates, Hacker News, GitHub)
and by hand otherwise. Each model in its own trained format: Fara1.5 in Microsoft's harness, Holo3.1 in H
Company's (tools, observations, thinking on).

| Driver | Right (of 12) | Resident memory | Failed |
| --- | --- | --- | --- |
| Fara1.5-4B Q4_K_M | 11 | 4.4 GB (peak 4.8) | Google Maps |
| Holo3.1-4B Q4_K_M | 8 | 4.1 GB (peak 5.2) | both booking.com tasks, Google Maps; GitHub gave a build, not the latest release |
| Holo3.1-0.8B Q8_0 | 2 | 2.0 GB (peak 2.4) | most tasks; answered Reliance with NVIDIA's price, xe.com in euros |

Times are not compared here: the Mac was left idle during this run and ran about three times slower (pages
and model alike). With someone at it, Fara1.5-4B took 30 minutes for the twelve.

What the runs changed in CodeRunner: a click by point now reaches controls drawn with ::before and elements
only scripts make clickable (19 of 25 clicks on booking.com were refused before); the driver's server keeps
two slots, so the step judge does not evict its conversation (steps re-read 10,000 tokens, 40 s); old
screenshots are dropped together (30 minutes and 11 right, against 70 minutes and 10, dropping one per
step); the driver is told the date; a request lost to a closing idle connection is sent again; a driver
going round in circles is told, then stopped; and actions and messages follow Microsoft's harness (clicks
where pointed, double and triple clicks, key combinations, the wheel where the mouse is, no note after an
action that worked, popups followed).

### File search measured on ori's bench corpus (November 4, same Mac)

The search and indexing changes made in Ori (the Swift menu-bar search app, github.com/instavm/ori) after the
two diverged on October 16 were brought over and measured on Ori's reproducible corpus (`bench/make-corpus.sh`:
389 files with noise, 85 folders; 108 file questions, 42 folder questions used to choose and 45 held out),
indexed by CodeRunner with EmbeddingGemma 2 on the Neural Engine (FluidInference build, 280-token pictures, sound
windows, no transcription). MRR@10:

| Ranking | files (108) | R@1 | folders v2 (42) | held-out v3 (45) | index time |
| --- | --- | --- | --- | --- | --- |
| before (meaning standardised per modality, kind bonus, words for codes) | 0.927 | 0.898 | 0.000 | 0.000 | 224 s |
| hybrid: plain cosine, every-word BM25, names, named kinds, folders, clusters; code prompt on | 0.945 | 0.935 | 0.649 | 0.603 | 116 s |
| the same without the code-retrieval prompt (shipped) | **0.957** | **0.944** | **0.695** | **0.742** | 116 s |
| shipped, but similarities standardised per modality | 0.939 | 0.907 | 0.685 | 0.750 | |
| the old ranker without the code prompt | 0.931 | 0.917 | 0.256 | 0.272 | |

Per kind, shipped: documents 0.97, notes 1.00, code 0.89, photos 1.00, screenshots 1.00, video 0.93, audio 0.86,
lookups 1.00 (0.58 before: "invoice INV-2024-0117" needed every word, not the meaning), names 1.00; folders by
name 0.93 / 0.96, by what they hold 0.49 / 0.54. Ori's own engine scores 0.978 / 0.715 / 0.703 on the same sets.
Indexing halved because the noise's copies are read once (a copy takes its twin's pieces). The code-retrieval
prompt is off: under plain cosine every code passage scored high against any question, which put scripts first
for "when should I plant garlic" and "frog croaking sound". A search takes 19 ms (median). The owner's real
Downloads folder (the 64 + 25 private questions above) was not measured again: a fresh index of its 64,207
files was started (`--cloud-mb 0`, code in checkouts off) and reached 2,486 text files, 103 documents, 18
pictures and 1,085 folders before it was stopped, slowed by the Messages indexing sharing the embedding server;
indexed alone it needs a few hours with pictures and video.

The Mail, Messages, Notes and Calendar sources are covered by `tests/test_sources.py` with fake apps, and were
run once against the owner's real apps from a search-mode daemon on this Mac (November 4): Messages indexed
5,592 conversation-days in 14.7 minutes (200 an index pass, from 24,147 of 145,125 rows) while a Downloads
index shared the Neural Engine; the listings that start the other sources took 0.4 s for 1,357 notes, 19.4 s
for the 112,435 ids and dates of the inbox, 0.4 s for the sent mailbox, and 131 s for 249 events over two years
of ten calendars (Calendar's scripting interface is slow; the range is listed once per `files.sources_interval`).
The run was stopped before Mail, Notes and Calendar entries were written, so their reading rate is not measured;
at 40 mails a script run, a 112,000-message inbox will take hours on its first pass and is read newest first.

### Search mode from Claude Code (November 4, same Mac)

A daemon in search mode (`sandbox.driver` none, no chat model, EmbeddingGemma 2 on the Neural Engine) over
Ori's corpus, started on a dev home, answered the search tools over the stdio bridge in 63-100 ms a query
including the bridge hop (`files_search` for "what time is our pena palace entry", "zen garden photo",
"freelance invoices folder"; the itinerary, the Japanese Garden folder and the Invoices folder first). A real
Claude Code session (`claude -p --model sonnet --strict-mcp-config --mcp-config ... --allowedTools
mcp__coderunner-search__files_search,mcp__coderunner-search__file_read`) asked for the Pena Palace entry time
called `files_search`, answered from the snippet of "Lisbon trip itinerary.pdf" and named the file, in 6.4 s
wall time including the model. The bridge must be able to import the package: the `coderunner` command the
installer links does that; a bare `python -m coderunner` from another directory does not.

### A first run of setup, measured (November 21, same Mac)

`python tests/setup_eval.py` runs the real setup in a throwaway home and reports what a first run fetches, how
long each step takes, how many requests it makes to Hugging Face and where it stops; `--bare` hides Ollama,
llama.cpp and Homebrew, as on a machine nobody developed on. The Mac had 11.4 GB of disk free and the owner's
assistant running from another home, so no run downloaded a model: runs stop at the first model download, and
the steps after it were timed on the model that home already serves.

| | Before | After |
| --- | --- | --- |
| Bare machine | failed after 0.0 s: "No local model runtime is installed" | setup fetched its own Ollama 0.40.3 (167 MB in 179 s and 289 s on this line, checksum checked), which started and answered as 0.40.3; then llama.cpp b11564 (14 s) or Prism's build (21 s) for the model it chose |
| Before anything is downloaded | 171 s of model discovery | 21 s |
| Requests to Hugging Face in discovery | 636, of them 500 to its API: the whole allowance of a caller without an account (500 per five minutes), 240 spent on guessed repository names that do not exist; setup's next request was refused (HTTP 429) and setup ended in a traceback | 427, about 300 of them to the API; a refusal ends discovery (the catalog is used) or fails that model with when to try again |
| 10.95 GB of downloads with 11.4 GB free | started, with no check | Bonsai 2 27B, its compact build, Qwen3.5-9B and the 1-bit Bonsai skipped (7.2 to 10.7 GB with the embedding models, 6.3 GB usable above the 5 GB floor); Qwen3.5-4B chosen (6.2 GB), with a note on what to free for the better one. With the floor at 0, Bonsai 2 27B as before |

A first run's downloads here: Bonsai 2 27B 7.84 GB, EmbeddingGemma 2 in Ollama 1.33 GB and for Core ML 1.48 GB,
Ollama 0.17 GB, Core ML packages 0.12 GB, Prism's llama.cpp 0.01 GB; about 57 minutes at the 3 MB/s four
streams got from Hugging Face. On the model already served (Qwen3.5-9B Q4_K_M, llama.cpp, woken from sleep,
while files were being indexed): smoke test 17 s with all four checks passing, timing 23 s, 253 tokens/s read
and 21 written, 10 s a step.

The installer's own Python, on a PATH like a stock Mac's (`python3` is 3.9.6 from the command line tools):
Python 3.12.16 downloaded in 90 s, checksum checked, 67 MB; a virtualenv made from it took the assistant's
packages in 46 s and ran `coderunner --version` and `coderunner doctor`. The functions were run on their own,
not the whole installer.

The first-run screen, in a browser paired with a daemon that had no model (a throwaway home on other ports,
its disk floor set to 500 GB so nothing could be downloaded): the plan appeared with the model, its reasons,
five downloads with sizes and the note that the pick does not fit the disk; "Download and set up" showed
setup's lines as they came and ended on "Setup did not finish" with the disk message and a way to look again.
A setup that succeeds, and the daemon starting again after it, were not run live (tests only). The Computer
panel against a daemon with no sandbox: the reason on the screen, "Set up the computer", then its refusal
("the sandbox is not running and did not start") with "Try again". Installing the desktop from the app in a
real sandbox was not run: the only sandbox on this Mac already has one and is in use. That check found the
panel dead when the browser kept an older page beside the new script; the script now makes its own button and
the app's files are served to be revalidated.

`coderunner app`: Electron 45.2.3 downloaded into a throwaway home in 81 s (121 MB, checksum checked, 259 MB
unpacked) and answered `--version`; the app's window was not opened from it.

The whole suite on Linux, in a throwaway container made from the sandbox image on this Mac (aarch64, Python
3.13.3, no zstd command): 571 tests pass, 18 skipped. The first run there failed three, each a fault outside
the tests: the pinned Ollama and Electron builds were not found when a Mac's chip was named the Linux way
(aarch64); a minimal Debian has no zstd, which unpacking Ollama's Linux archive needed, so the zstandard
package now comes with CodeRunner on Linux and does it; and Mail, Messages, Notes and Calendar were not
listed for the first quarter of an hour after any start of the machine (the clock counts from boot, and
"never listed" was kept as time zero). No GPU, no Ollama download and no sandbox were part of that run.

Not run: a model download and everything after it in a clean home (disk space, and every home uses the same
ports, so a second home would talk to the first one's servers); the installer end to end; voice setup; the
sandbox image build.

### The app's first run, driven in the Electron app (November 21-22, same Mac)

The setup screen, its folder choice and the failure screens were run in the desktop app on a throwaway home and
photographed over its debugging port. The planning model was downloaded (7.63 GB, checksum matched; the
connection fell to 2 Mbps and dropped once, and the download resumed from its ranges).

Loading it did not succeed, and that is the finding. With about 6 GB of memory free (34%), reading Bonsai 2 27B
into memory took swap from 2.2 to 5.9 GB in 25 seconds and free memory to 1%; a watchdog of the test stopped
the server. A second try with 8.8 GB free added 1.6 GB of swap within seconds. The plan had said "fits: 9.7 of
11.7 GB": true of the machine, not of the moment (other apps and another CodeRunner home's VM were open). After
the stop, setup went on by itself to download a different 6 GB model, and the app's headline error was about
disk space for a third.

What changed because of it: setup reads the memory free right now before loading and wants 1 GB beside the
model; the first screen says when it is short; a setup started from the app does not fetch a model that screen
did not show; errors are one or two short sentences. With 7.2 GB free the app then refused the load at once
("Not enough memory free right now", nothing started) and offered the next model that fits.

With 71% of memory counted free (12.8 GB) the same button then went through, in the app: the model loaded on
the pinned llama.cpp build, passed all four checks (a tool call with arguments in 14.7 s, choosing a tool 6.2 s,
answering without one 4.8 s, naming the colour of a picture 9.2 s) and was timed at 69.4 tokens a second reading
a 3,999-token prompt and 11.6 writing: 19.8 s for an agent step against 19.3 s estimated. Swap grew by about
2.6 GB while it was loaded and free memory stayed between 8% and 17%: it runs on this Mac, at the cost of other
apps' idle memory. The catalog now carries the two factors (0.87 reading, 1.07 writing); with them a 16 GB M2
goes back to Qwen3.5-9B (Bonsai's steps there would take 28 s).

Setup then finished (file search on Core ML, the voice model), the daemon started again in place and the app
moved to its task screen. One task was given there, a rent increase to work out: the status line read
"Thinking" with a running clock and "Stop", and the answer (three correct figures) came after 2 minutes 15
seconds, most of it the first read of the system prompt at 69 tokens a second.

Still open from this run: tasks that use the browser or files have not been run in the app with Bonsai 2 27B
(the next section runs them with Qwen3.5 9B), and the measurement was taken with swap in use. The second
daemon on one home is refused since November 22.

### A default install in the app, its core tasks and `coderunner check` (November 22, same Mac)

The home the assistant normally runs from on this Mac (sandbox VM on, the browser on the sandbox's own
computer, 107,000 files indexed) had its model settings set aside and was opened in the desktop app, which
started the daemon itself. Each step was photographed over the app's debugging port.

First run. The setup screen came up with Bonsai 2 27B recommended (8.3 GB to download) and three faults:

* the daemon had started Ollama (installed on this Mac for other things) although nothing was set up and this
  Mac uses llama.cpp. It now starts no runtime before setup;
* the screen named the other models that fit but gave no way to take one, though Qwen3.5 9B was already on the
  disk. The planning model's name now opens the list (seconds a step, download or "Installed");
* a planning model that was already there was reported as "downloaded".

With Qwen3.5 9B chosen ("Nothing to download" beside the 0.6 GB voice model), setup loaded it, passed the four
checks (6.0, 3.6, 2.2 and 4.2 s), measured 241 tokens a second reading and 19.9 writing (10 s a step, as
estimated), prepared file search (31 passages a second on the Neural Engine against 18 on the GPU), fetched and
prepared the voice model, started the daemon again in place and showed the task screen with the computer's
screen live in its panel. The voice download took about seven minutes on this connection.

Tasks given in the app, with Qwen3.5 9B:

| Task | Result |
|---|---|
| A receipt (PNG), a video and a PDF attached; the receipt's total and what the video says | Total right in 47 s. The video's on-screen words were read, its speech was not listened to. |
| The same video alone, after the fix below | The speech transcribed and the cost given, 14 s |
| A PDF lease attached: end date and notice | Right, 13 s |
| A CSV attached, "use Python": best month and total | Right, but 9 min 24 s: an approval nobody answered (5 minutes), then a script that named the host's path. After the fixes: 32 s with one approval. |
| Wikipedia in the sandbox's browser: a river's length and states | Right, 1 min 56 s, two page pictures in the thread |
| "Search my files for a rental or lease agreement": file names and notice periods | Four agreements read and tabled, 2 min 13 s; one path the model made up was refused as not a file |
| Take control of the computer, then back to watching | The panel said "You're in control" and went back |

What those runs changed: each attached file's line now says which tool reads it and that a video's sound is
not in its frame; the file tools take the sandbox's path for an attached file as well as the host's, and sandbox
code that names the host's path is given the sandbox's; an approval is one sentence with the step small under
it, says how it ended (it said "Answered" for one that had timed out), and the status line stops saying
"Waiting for you" once it is answered.

`coderunner check`, on that install:

| Check | Result | Time | Tool calls |
|---|---|---|---|
| answer | passed | 6 s | 0 |
| picture | passed | 26 s | 0 |
| document | passed | 11 s | 1 |
| video | passed | 14 s | 1 |
| code | passed | 32 s | 1 |
| files | passed | 86 s | 9 |
| website | passed | 31 s | 4 |
| flights | passed | 146 s | 6 |
| booking | passed (the cheapest flight's passenger page reached, payment never opened) | 222 s | 9 |
| appliance | passed | 104 s | 12 |
| errands (a flight search, a website and two documents in one request) | passed, 3 of 3 | 394 s | 22 |

The seven quick ones ran in one go (5 min 18 s with the wait for the test documents to be indexed). Of the
four long ones, two were run twice: in the first run `appliance` took 396 s and `errands` did not finish in its
15 minutes, because the check answered only approvals asked under its own origin and the browser asks under
another, so each of those waited five minutes. That was the check's fault and is fixed; the figures above are
the second run. After the run the shared folders and `browser.allow` were as before and the test folder was
gone.

The check also showed the daemon answering nothing for tens of seconds at a time on this index (3.1 GB,
518,000 pieces). Three reads ran on its event loop: every scan read all indexed files (8 s); the first search
after a start read every vector (63 s); every indexing pass asked for names without a vector with a read of
the whole table (13 s). Polled every 0.4 s while the daemon started, a folder was shared and scanned, the
slowest request took 40 s before and 2.95 s after (1 of 357 over a second).

A second daemon started by hand on the same home now stops at once with "Another CodeRunner daemon is already
running".

All eleven were then run in one go (`coderunner check --long`) with a browser driver set (Fara1.5 4B, taking
the GPU in turn with the planner): 11 of 11 in 26 min 18 s. `errands` passed in 867 s of its 900: the planner
spent seven minutes on the calendar before handing the flight search to the driver, which took four and a half
more. While the driver worked it was also the judge of its own steps, and it called five harmless clicks
committing ("Previous month" three times, the date, "Search"), so each asked for approval; the check answered
them, an owner would have been asked five times.

That judgement was then measured and fixed. On the approval benchmark (`tests/approval_eval.py`: 100 synthetic
and 30 real labelled steps, plus 28 held-out ones written after the fix was chosen and before any judge was run
on them) Fara1.5 4B, asked cold, missed no committing step and called 50 of 87 harmless ones committing. Shown
nine steps already judged before each question, it calls 3 of 87 so and still misses none:

| Judge | Synthetic (100) | Real (30) | Held-out (28) | Commits missed (71) | False alarms (87) |
|---|---|---|---|---|---|
| Fara1.5 4B, cold | 67 | 20 | 21 | 0 | 50 |
| Fara1.5 4B, with examples | 98 | 30 | 27 | 0 | 3 |
| Qwen3.5 9B, cold | 97 | 30 | 27 | 1 | 3 |
| Qwen3.5 9B, with examples | 95 | 30 | 28 | 0 | 5 |

The examples were chosen against the first two sets, so the held-out column is the fair one for Fara (21 to
27 of 28). The planner model judging itself is about as accurate either way and no longer lets LinkedIn's
"Connect" through. Every chat judge is shown the examples now; Bonsai 2 27B has not been measured with them.

In the product: a twelfth check, `driver`, hands the flight search to the browser driver on purpose. With Fara
driving and judging its own 14 steps, none was called committing and no approval was asked (five of the first
five were, before); the three cheapest fares came back in 244 s. Each check's line now also says how many
approvals the task asked for. `errands`, run again, passed in 287 s without the planner handing off at all.

A picture pasted into the task box and a file dropped on the thread were attached in the app, by events
dispatched in the page (not a drag from Finder).

Not covered by these runs: a real drag from Finder; the system's file dialog itself; files near the 200 MB
limit; attaching from a paired phone; the check on a text-only model, without the sandbox or on Linux (those paths are in the unit
tests only); any model but Qwen3.5 9B on these tasks.

### Remote mode on Claude Haiku 5.5, and twenty tasks and five forms on real websites (November 22, same Mac)

The same home was switched to remote mode with `coderunner claude on` (the key read from a file into
`assistant/anthropic-key`, mode 600) and driven in the desktop app.

That the calls reach `claude-haiku-5-5`: the daemon's log has one line a request with the model the API named
and its request id, e.g. a first planner step with tools `req_011ChHjENr2ycyLqnNWcHN1p` (4,929 tokens written
to the cache), the judge of a step `req_011ChHjEVHUnDPHdQcGnU1r9`, and the task's second step
`req_011ChHjF3Uz72kyaz6evGiVW` with `cache_read=4929`. A step takes 1.1 to 1.3 s (the Models panel shows the
median measured). The first task in the app, a sum and a Wikipedia lookup, was answered in 10 s.

As the step judge on the approval benchmark (100 synthetic, 30 real, 28 held-out steps): 96 / 30 / 28 right with
the worked examples, no committing step missed of 71, 4 false alarms of 87, 0.78 s a step; asked cold 94 / 29 /
27 with 8 false alarms. The injection guard was tried with it and turned off again for remote mode: it flagged
7 ordinary pages in about 80 reads, taking the page snapshot's own reading hints for instructions.

`coderunner check --long` on Haiku: 11 of 11 that apply passed in about three and a half minutes (answer 1 s,
picture 1 s, document 3 s, video 5 s, code 5 s, files 7 s, website 11 s, flights 40 s, booking 49 s, appliance
17 s, errands 57 s); `driver` is skipped, as the browser driver is not used with Claude.

`coderunner check --real`, the twenty tasks on real websites. The first full run passed all twenty by its
criteria in 10 min 24 s, and reading the answers showed the criteria were too easy: a price comparison of a
phone no shop listed and a cart nothing had been put in had passed. The grading was tightened (an answer that
opens by saying it could not do the task fails; a cart or reservation task must have clicked what it turns on,
or been stopped by the approval for that very step), the phone was changed to one on sale and the SSD budget to
one that exists. Problems those runs exposed and that were fixed:

* with a local model's limits the loop dropped pages once the conversation passed 20,000 tokens and gave 12,000
  characters of each page, so a three-country comparison came back "not captured"; Claude is given 36,000
  characters of a page, links whole, and a 150,000-token working context;
* a research task that held a recalled memory asked for approval once a site (five times for a flight search);
  "Allow for this task" now covers the rest of the task;
* every task's pages stayed open: after about forty tasks the sandbox's Chrome held fourteen page processes in
  its 2 GB, the kernel killed one (`memory.events`: `oom_kill 1`), and every later page open failed three
  times over thirty seconds each until Chrome was restarted by hand. Tabs are capped and closed after a task,
  a crashed tab is closed when the browser reports it, and a fault is met by reconnecting and restarting
  Chrome without the model being told. With the page processes killed in the middle of a task, the task went
  on and reached its next approval;
* answers opened with the model's own narration, and its interim remarks were shown in the thread; only
  deliberate updates are shown now.

The last full run, on the code as committed (2026-11-22 17:19:44, `claude-haiku-5-5`, 16 min 31 s in all): 19 of 20 everyday tasks passed; of the five forms 4 passed and 1 stopped at the CAPTCHA that is the owner's.

| Task | Result | Time | Steps | Approvals asked (refused) |
|---|---|---|---|---|
| mauritius: a week in Mauritius planned: flights, hotels, a day-by-day plan with costs | passed | 54 s | 10 | 0 (0) |
| phone: one phone priced at three retailers | passed | 33 s | 10 | 2 (2) |
| laptop: three laptops under a budget from two shops | passed | 36 s | 9 | 0 (0) |
| headphones: the best-reviewed headphones according to two review sites | passed | 1 min 37 s | 13 | 0 (0) |
| flight: the cheapest non-stop flight on a date, from two travel sites | passed | 2 min 50 s | 19 | 1 (0) |
| train: morning trains between two cities from a timetable | passed | 15 s | 5 | 0 (0) |
| hotel: three hotels under a nightly budget for given dates | passed | 30 s | 6 | 0 (0) |
| restaurant: three restaurants in a neighbourhood, open at a given time | passed | 22 s | 5 | 0 (0) |
| repair: three repair shops nearby with hours and phone numbers | failed (the answer says it could not be done) | 25 s | 5 | 1 (0) |
| weather: a three-day forecast and the best day for a picnic | passed | 15 s | 4 | 0 (0) |
| currency: an amount converted into two currencies at today's rates | passed | 17 s | 5 | 0 (0) |
| islands: three countries compared in a table with a source for each | passed | 46 s | 11 | 0 (0) |
| visa: visa rules for three countries, each from an official source | passed | 1 min 7 s | 14 | 0 (0) |
| price: one product's price at the maker and at a retailer, against a threshold | passed | 16 s | 4 | 1 (0) |
| news: today's technology headlines from two news sites | passed | 39 s | 4 | 0 (0) |
| cart: a product put in a cart and taken to the step before sign-in or payment | passed | 1 min 11 s | 4 | 2 (1) |
| reserve: a hotel room taken to the page that asks for the guest's details | passed | 25 s | 6 | 2 (1) |
| biryani: a recipe turned into a shopping list, with five items priced at a grocer | passed | 1 min 5 s | 14 | 0 (0) |
| papers: two attached documents read, and a letter drafted from them | passed | 5 s | 2 | 1 (0) |
| chart: an attached sheet charted in the sandbox, with the figures read off it | passed | 7 s | 3 | 1 (0) |

The five forms:

| Task | Result | Time | Steps | Approvals asked (refused) |
|---|---|---|---|---|
| ecourts: a district court case search filled in, up to its CAPTCHA | needs you: CAPTCHA | 28 s | 11 | 1 (1) |
| stations: a trains-between-stations search form filled in and read | passed | 17 s | 5 | 0 (0) |
| pincode: a PIN code found through state, district and post office lists | passed | 41 s | 8 | 0 (0) |
| ifsc: a bank branch's IFSC code found through bank, state, district and branch lists | passed | 21 s | 6 | 0 (0) |
| emi: a loan calculator given four figures, checked by the instalment it returns | passed | 30 s | 5 | 1 (0) |

The one failure of that run, `repair`, was an answer that opened by saying the shops' hours could not be
confirmed. The run before it (16:19, before the web search step) passed 18 of 20: `train` and `visa` had opened
addresses written from memory, none of which existed; with `browser_search` both pass. Pass rates on these
tasks were not pursued further on Haiku: how well tasks succeed is to be measured on the local model.

The forms were run four times while the grading was settled. What they showed about form handling: lists that
depend on one another worked as they are, both the kind that fills the next list in the page (eCourts) and the
kind where each choice opens a new page (the PIN and IFSC finders); a station field with suggestions and a
calculator's number fields took typing. What was wrong and is fixed: the assistant ended the task to ask for
the CAPTCHA instead of handing the screen over, and nothing stopped it from typing into a CAPTCHA box. Not
exercised by these five: select widgets drawn by a page's own script, date pickers (the flight and hotel tasks
use booking.com's), results in a frame.

Reading every thread of the run (7 updates, 25 answers) for internal workings: no tool names, element numbers,
retries or error texts. One answer (`phone`) opened with "I'm stopping on Croma here" after a step was not
allowed; the wording given to the model for such a step was changed again after this run and not re-measured.

The sandbox VM's 2 GB is the limit that remains: `memory.events` in the VM showed `oom_kill 3` by the end of
the day (the first before the tab cap, two more at 16:02 to 16:04 while heavy shopping pages loaded), each met
by the browser being brought back without the task being told. No kill in the runs after 16:04 (some sixty
tasks). A 4 GB VM (`coderunner sandbox reset --memory 4g`, which recreates the VM and deletes its filesystem)
is the remedy and was not applied here.

Tokens for the afternoon in remote mode, from the daemon's log up to 17:03 (718 requests): 227,073 input,
9,894,858 read from the cache, 2,822,607 written to it, 118,084 output.

Not covered: the "Start it" button and the shell bringing a dead daemon back were not exercised in a window
(the open window's main process predates them; unit tests cover the logic); the approval cards for sending
mail, changing a setting and dialling a call were looked at with stand-in approvals written into the page's
data, not raised by the connectors, which were not enabled or run; the first run's "Use Claude instead" path
in a window; remote mode on Linux; Sonnet or Opus as the remote model.

### Two tasks at once, and the shared context, on the local model (November 23, same Mac)

Qwen3.5 9B in llama.cpp at 23 tok/s (checked first, against 20 measured before, so the Mac was not in its slow
sleep-wake state). Two tasks that need the model and the sandbox but no browser (a Fibonacci sum, a bill split),
each alone and then both at once, with the server's slots read every 0.3 s:

| | alone, one after the other | both at once |
|---|---|---|
| as first built | 15.1 + 19.2 = 33.3 s | 47.4 s |
| first slot copied to the second | 17.1 + 13.1 = 30.2 s | 38.4 s |
| slots reset from the kept file at each task's start and end | 21.1 + 16.1 = 37.2 s | 35.2 s |

The tasks used slots 0 and 1 and were never at the model together. As first built, the second task's slot was
empty and its first request read the 4,834-token instructions for 18.8 s; copying the first slot's reading
(llama.cpp's slot save and restore) takes about 0.1 s. In the second row the joint run's first Fibonacci step
thought for 24 s against 12.7 s alone; the time between one request and the next was about 15 ms. After the
change no request but the warm-up at start read more than 239 tokens. Run-to-run variation in thinking length
is as large as the differences here, so the rows say the extra cost is gone, not that one is faster.

The day before, 4 of 22 tasks in the local run of the real-website bench (`laptop`, `hotel`, `restaurant`,
`islands`) ended with "the model on this computer stopped answering". The server's log gave the reason for all
eight such errors: the slots share one 32K context (`--kv-unified`), and the sum of what they held passed it
(18,011 + 15,874; 17,730 + 15,627; 28,375 + 4,749) or one conversation passed it alone (28,547 tokens before the
request that failed). Before tasks named their slots, a conversation's earlier reading stayed in the slot it had
left. The trim then in place dropped only old pages, at 60% of the context by an estimate of 3.5 characters a
token that left out the tool list. `islands` opened six pages in one step and went from about 6K tokens to over
32K before the trim could run. `restaurant` had started while `hotel`, which the bench had stopped at its time
limit, was still running: the stop had arrived while the driver had the model, the planning model failed to
start again, and that failure took the stop's place, so `hotel` went on for six more minutes in the same slot.
Each of these is fixed as ASSISTANT.md's "More than one task at a time" describes, and the stop is kept.

### Four long real tasks again, typed into the Electron app (November 23, same Mac)

The four tasks that had run out of time the day before (`mauritius`, `headphones`, `flight`, `visa`), each
typed into the app's own task box in a new thread, one at a time, and judged afterwards by the bench's own
grader (`server/realworld.py`). In the app a task ends after 30 steps, not at the bench's ten minutes:

| task | time | outcome |
|---|---|---|
| mauritius | 643 s | failed: eight minutes on Skyscanner's date picker, whose search never loaded; no flights, hotels or plan |
| headphones | 751 s | failed: a fair table of three, one review site each and no links (asked for) |
| flight | 1,330 s | answered: Skyscanner's dated result (SpiceJet 06:10, ₹8,877, after the driver set the search in 195 s); MakeMyTrip's price came from its route page, not the date, after the driver used all its steps in 644 s |
| visa | 692 s | failed: one official link of two; a passport office "Click Here" link was judged committing and waited 300 s |
| visa, after the changes below | 2,118 s | failed: the driver, judging its own clicks while the planner was unloaded, flagged 4 of 7 ("Share", "Non-citizens…", "Additional Information"), each waiting 300 s; Mauritius's rule came out wrong |

The time goes to steps of about 20 s on the 9B model (reading a page's changes at about 230 tokens a second, then
writing), to the driver's handovers (195 to 644 s each), and to approvals nobody answers. The first request after
the server's idle sleep generated at 2.5 tokens a second, against 20, while about 3 GB of swap was in use.

Changed after these runs: a reply asked for links and given none is written again once, with links to the pages
it read; the reply after the step limit is held to the same rules as any other and goes through the task's own
slot and turn; a link's address is shown to the step judge (with it a "Click Here" to a visa page is safe, and
links that close an account, unsubscribe or delete a payee still commit; the judge benchmark is unchanged, 0 of
57 commits missed, since its steps carry no address); and once an approval of a task expires unanswered, its next
ones wait 30 s, not 300. Two worked examples of vague links added to the judge made it miss Airbnb's "Request to
book" in the benchmark and were taken out again.

The driver's own judging came next. `tests/approval_eval.py --sets driver` holds 34 clicks of the kind a screenshot
driver makes on page text no control names (19 harmless, 15 committing; six of them on boxes with no words of their
own). Fara1.5 4B, which judges while it drives where the planner is unloaded, flipped on these: the same
"Additional Information" was committing on one site and safe on the next, and a box was put to it as 'div ""'. Now
such a click is put with the words it sits in, is shown four worked examples of clicks on page text (only before
such a click: shown before every step they made it flag 6 more of 55 harmless buttons), and a box with no words in
or around it is not asked about at all, since nobody could answer 'Click div ""'. Fara went from 24 of 28 (a missed
"Complete booking", three false alarms) to 34 of 34, Qwen3.5 9B to 34 of 34, the other sets unchanged. The check's
driver task (a flight search done from screenshots) had failed at 15 minutes with 30 approvals asked; afterwards it
passed in 475 s with none, the judge clearing all 37 of its clicks.

The app showed a task started from a new thread twice, one copy stuck at "0s" (the thread was read while the
task's start event arrived); a stopped task's dot was the brand blue of a working one; and a three-column table
hid its third column behind a sideways scroll that macOS does not show. All three are fixed and seen in the app.

### The owner's own model server, and a Claude retry (November 23-24)

A Linux machine with an NVIDIA GPU served Ternary Bonsai 2 27B (PQ2_0) through llama.cpp's OpenAI-compatible
API on the tailnet: 64K context over two slots, images, tool calls, 28.6 tokens a second written and about 510
read. With it as the model (`/api/remote {"action": "server"}`) the Mac loaded no model for planning, pages or
judging; free memory went from 15% to 56-63%.

| run | time | result |
|---|---|---|
| mauritius and headphones at once, before the changes below | 1,550 and 1,732 s | both failed (no plan; no table), 34 and 42 steps, both at the 30-step budget |
| mauritius, slots pinned | 1,028 s | flights, a hotel, costs and links; no day-by-day plan: still stopped at 30 steps, saved in the dev home's config |
| headphones, no step budget | 1,867 s | 160 steps, then "I had to stop here": the conversation was full and nothing was answered |
| headphones, answering from what it found | 1,526 s | one request dropped at step 75 ended it |
| headphones, the request tried again | 2,521 s | a table of three with what two sites say and an Amazon price and link each; full once, answered from its findings |

What changed: each task keeps to a slot of the server and the judge to the last (unpinned, a 20,000-token
conversation was read again at 510 tokens a second, about 40 s a step); a step's writing is capped as on a
local model (one step wrote 4,700 tokens of code in 166 s); the 30-step budget became a 200-step stop, with a
reminder every 25 steps after the 30th, and an old 30 saved in a config gives way; a full conversation answers
from the request, what it said along the way, its plan and its latest steps; a request that fails is sent again
once the server's /health answers. Bonsai wrote nothing beside its tool calls (34 of 34 steps), so every tool
gained an optional `say` line, shown in the thread as a chat message: 74 in one run. It still searches long
after it has enough: 139 steps for the headphones table.

Try with Claude, on the local model's Mauritius plan (which said it could not write the day-by-day plan): the
same request ran in the same thread on Claude Haiku 5.5 in 45 s and stopped short as well, with dated
Skyscanner fares and no hotels, noting that the local answer's hotel prices had not been checked against any
site (none of its three hotel links was a page it had opened).

### Research and real checks, Bonsai against Claude Haiku 5.5 (November 24)

Both suites (`research`, 12 questions; `real`, 25 tasks) ran on the owner's server with Bonsai 2 27B, then on
Claude Haiku 5.5, with a 30-minute limit per task. Graders found wrong in this run were fixed and the saved
answers graded again (a choice said as "I'd go with"; three headlines written as linked paragraphs; one research
key).

| suite | Bonsai 2 27B, own server | Claude Haiku 5.5 |
|---|---|---|
| research | 11 of 12, 85 min, 281 steps; missed the moon landing's date | 11 of 12, 3 min 16 s, 44 steps; one answer without the links asked for |
| real, tasks Bonsai reached | 13 of 16; hotel and restaurant at the 30-minute limit (57 and 45 steps); the cart task cut off when the server went offline | 15 of 16; train gave no train names, restaurant said Sunday hours could not be checked |
| real, all 25 | 9 not run: the server's machine left the tailnet at about 10:20 and every later task ended "your model server stopped answering" at once | 21 of 25 and a correct stop at the court site's CAPTCHA; emi left the calculator unfilled |

Where Bonsai ran, it answers nearly as well as Haiku, and passed the train task Haiku missed; the gap is time: 61 s to 1,786 s a task against 6 to 120 s, and 3 to 71 steps
against 2 to 15. Its time goes mostly to reading, not writing: restaurant spent 622 s reading prompts and 129 s
writing. Each step's `agent.turn` audit event (new) gives tokens read, found cached and written, and showed
three causes of whole-conversation re-reads:

* The chat template writes "Reasoning effort is set to xhigh" at the top of the system prompt with reasoning on
  and nothing with it off, so every switch between the two read everything again. The client now renders the
  template both ways once (`/apply-template`) and sends reasoning steps with a setting that renders like
  reasoning off (`reasoning_effort: medium` on this server), or keeps a task to one setting.
* Any request on one slot discarded the other slot's cache: a 132-token judge call on slot 1 made the task on
  slot 0 read 15,372 tokens again. Every judged click did this, 117 in the restaurant run, 29 of them a step
  already judged. Verdicts are now remembered per step (up to 512; a failed judgement is asked again).
* Trimming old pages at 60% of the room edits earlier messages and reads from there.

Two faults in the check itself were fixed: a task stopped at its limit still held the slot, so the next was
queued and read as ended (18 tasks "ended as queued" in the first Bonsai run); the check now waits for a task to
start and for a stopped one to let go. A rerun of Bonsai with these fixes waits on the server.

For the server's owner: the cross-slot eviction is the server's (this build, or its flags); two slots that keep
their own cache, `--slot-save-path` (so the instructions are restored rather than read), or the judge on a small
separate server would each remove most of the remaining re-reads. Until then the room per task stays at half
the context, since a larger conversation makes every re-read slower.

### World and everyday tasks: Bonsai 2, Bonsai 2.5 and Claude Haiku 5.5 (November 24-25)

Two suites of real-site tasks outside India: `world` (13: homes and rents in San Francisco, US tax, streaming
prices, London, German, French, Spanish, Japanese and Brazilian sites in their own languages) and `everyday` (14:
bookings taken to the details page, cancellations, plans, prices, routines). One run each, 30 minutes a task, on
the owner's server for Bonsai and through the API for Haiku. Times and steps below are at the 30-minute mark.

| run | world | everyday | together |
|---|---|---|---|
| Claude Haiku 5.5 | 6 of 13, 107 steps, 13 min | 6 of 14, 90 steps, 8 min | 12 of 27 |
| Bonsai 2 27B, before the fixes | 4 of 13, 665 steps, 182 min | not run | |
| Bonsai 2 27B, with the fixes | 7 of 13, 591 steps, 172 min | 7 of 14, 352 steps, 122 min | 14 of 27 |
| Bonsai 2.5 27B, same code | 9 of 13, 527 steps, 149 min | 9 of 14, 344 steps, 140 min | 18 of 27 |

A site turning the browser away counts as a fail here: one task in each Bonsai 2.5 suite (sflistings, jobs),
one in Bonsai 2's world (sfmarket). Leaving those out, together with passportfees (Bonsai 2's run started it
during a server switch and took no step), Bonsai 2.5 passed 16 of 23 and Bonsai 2 12 of 23. Bonsai 2.5 gained
ustax, ave, brasil, dentist, table and konzert, and lost kaffee (no German in the answer) and gymcancel (its one
link was not a page it had opened). With one run each, a task or two either way is within the noise.

Haiku takes 4.5 to 5.5 s a step and Bonsai 16 to 18; Haiku also needs about five times fewer steps. On
Bonsai 2 a step was about 11 s of model, 5 s of browser and under a second of judging, so fewer steps is the
larger saving. Bonsai 2.5 writes at the same 32 tokens a second as Bonsai 2; its gain is fewer steps per task
in `world` (41 against 45). Haiku's misses were links to pages it had not opened (6 tasks) and answers not in
the site's language (5). It ran just before those two became standing rules in the system prompt, which every
Bonsai run with the fixes had, so this table understates it.

What changed between the two Bonsai 2 `world` runs (4 to 7 of 13): the step judge is not asked about plain
links, a batch of browser steps stops at the first that changes the page, old results are trimmed in one go
down to a low mark, a task tries another site and gives the part it found before saying it cannot be done, and
answers follow the request's language and link each source. The language check was also loosened (common
words in any order), so part of the gain is the check's.

Still failing on both Bonsai versions: bayrent (the shares were not worked out with `run_python`), bahn and
onibus (the answer not in the site's language), dinners (prices without a currency) and preiswecker (no routine
set up). A run of `world` with Fara 1.5 4B driving the browser beside Bonsai 2.5 stopped after its first task
when the disk filled; it is not counted.

## Not yet validated

* CodeRunner's own Ollama on Linux: the archives' checksums are the release's own, their layout (`bin/ollama`)
  was read from the first megabytes of each and unpacking a zstd archive is tested (with the zstd command on
  macOS, with the zstandard package on Linux), but the 1.4 GB download was not run and nothing was started on
  Linux. AMD's ROCm libraries are not fetched.
* The app starting the daemon: its logic was run with Electron replaced by a stand-in (daemon down and
  `coderunner up` working, failing, missing; daemon already up), not in the app's own window. `coderunner app`
  on Linux, where Electron's sandbox helper may need setting up, has not been run.
* `tests/test_files.py`'s OCR test failed twice while the owner's assistant was indexing PDFs on the same Mac
  (the picture's text was not read in time) and passed once that load had dropped.

* Voice: Qwen3-ASR on Linux and NVIDIA machines; the 🎤 button in a real browser and
  `coderunner chat --voice` with a real microphone (both ask for microphone permission); setup's
  Homebrew install of ffmpeg; Apple speech for languages other than English and German; Parakeet
  on other chips (M1, M4, Max and Ultra), on macOS 15, and in languages other than English and
  German.

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
* Ollama's own copy replacing an older installed one: the download, checksum and start ran on macOS (above, on
  a machine made to look bare); the case of an older Ollama already serving was tested with fakes only.
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
