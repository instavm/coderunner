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

## Not yet validated

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
