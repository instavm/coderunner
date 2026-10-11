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
| Daemon (`test_daemon.py`) | Real HTTP servers: owner chat, memory recall across conversations, sandbox model proxy, credential separation, scheduled delivery, and the agent driving the real browser where a "Buy now" click waits for an owner approval over the API |
| Models (`test_models.py`) | Hardware parsing (Apple, NVIDIA, AMD, CPU, WSL2), budgets, KV-cache sizing, picks per memory tier, licence resolution through base models, resumable verified downloads, smoke-test pass/fail including the wrong-argument case below |
| Memory, scheduler, connectors, channels, voice, service, remote | Unit tests with fakes; see each test file |

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
  MiMo-V2.6-Distill-Qwen-9B, Granite 4.2 8B and Mage-VL besides the catalog's picks.

## Not yet validated

* A full booking.com flight search completed end to end by a local model.
* Discovered models on large machines (192-512 GB Macs, multi-GPU, DGX Spark clusters) and the
  on-device ranking among several downloaded candidates.
* Microsoft Fara1.5-9B (screenshot-based browser specialist) as a second browser driver.
* Starting the sandbox VM on demand on 16-18 GB machines, to leave room for a bigger model.

* No sandbox image was built or launched from this branch, and the Apple
  `container` 1.5.0 upgrade path was not run end to end. Build with
  `./install.sh --build` before release; previously published images ignore
  the token and are refused by the installer.
* Proxy network mode assumes the host is reachable from an `--internal`
  Apple `container` network at the network's gateway address. Verify that the
  sandbox can reach `:8223`/`:8225` and nothing else.
* Linux/Docker and WSL2 installation paths have not been run.
* No real model has completed the tool-calling smoke test on this branch.
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
