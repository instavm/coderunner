# Authentication and network boundaries

## Credentials

| Credential | Where | Who holds it | Grants |
| --- | --- | --- | --- |
| Sandbox token (`CODERUNNER_API_TOKEN`) | `~/.coderunner/api-token` (0600) | MCP/REST clients, the sandbox VM's environment | The sandbox's MCP and REST API (run code, sessions, headless browser) and the host gateway's model API and secret-injecting fetch |
| Admin token | `~/.coderunner/assistant/admin-token` (0600) | The CLI and the local MCP bridge | The owner API: chat, approvals, grants, schedules, audit, the guarded tools |
| Paired-device cookie | Browser cookie, stored hashed on the host | Devices paired with `coderunner remote pair` | Same as the admin token, from the web UI |

The installer creates the sandbox token automatically (`secrets.token_urlsafe(32)`)
unless `CODERUNNER_API_TOKEN` is already exported, in which case your value is
used. Print it for client configuration with `coderunner token`. The server
inside the sandbox never generates a credential: it fails closed (503) when the
token is missing or shorter than 32 bytes, and returns 401 for missing,
duplicated or incorrect `Authorization: Bearer` headers. `GET/HEAD /health` is
the only public route. Host and Origin validation still applies.

## Network exposure

* The sandbox API listens on `127.0.0.1:8222` (Docker Compose, Apple `container`
  in `none`/`proxy` mode) or on the VM's host-only address
  (`coderunner.local:8222`, default Apple `container` mode). Never forward it to
  the internet.
* The host gateway for the sandbox listens only on the VM bridge address
  (`:8223`, plus the egress proxy on `:8225` in `proxy` mode).
* The owner API and web UI listen on `127.0.0.1:8224`. Remote access is
  through Tailscale Serve on your tailnet only, and still requires a paired
  device; Tailscale identity headers are an extra restriction, never the
  credential, because any local process can forge them.

When running the Docker image by hand, publish it the same way:
`-p 127.0.0.1:8222:8222 -e CODERUNNER_API_TOKEN`. The process inside the guest
still listens on `0.0.0.0:8222`. The installer passes the credential by
environment variable name, checks that anonymous requests are rejected and that
an authenticated session list works before it reports readiness. It does not
change an existing container's environment: to rotate the token, delete the
container and run the installer again.

## What the credentials do not protect against

Anyone with the sandbox token can execute arbitrary code in the sandbox, use
its network and browser, and call the local model. Code inside the sandbox can
read the token from its environment; that is expected, and it is why the
sandbox token cannot approve actions, read memories, drive your local browser
or reach connectors. Jupyter sessions share one guest, filesystem and network;
they are not separate VMs. HTTP between the host and the VM is unencrypted and
stays on host-only interfaces. Malware already running as your user is outside
this boundary.

The named-session execution path never retries a command after a kernel
failure; clients should not retry uncertain commands either. The unnamed legacy
execution API keeps its retry behaviour and should not be used for actions that
must run at most once.

Images published before authentication was added ignore the token. The
installer verifies that anonymous requests are rejected and refuses such an
image; build from this checkout with `./install.sh --build` until a new image
is published.

`tests/test_auth.py` covers HTTP, WebSocket, weak or missing configuration,
duplicated credentials and rotation; `test-e2e.sh` checks a running sandbox.
