# Authentication and network boundaries

Both the MCP and the REST routes sit behind a bearer credential. The installer
generates it once into `~/.coderunner/api-token` (mode 0600), or saves the
`CODERUNNER_API_TOKEN` you export; it must be at least 32 bytes with no
whitespace. Every request except `GET/HEAD /health` requires the exact bearer
credential. Missing or invalid server configuration returns 503; missing,
duplicated, or incorrect client credentials return 401. Host and Origin
validation still applies. Authentication does not replace it.

Docker Compose publishes only `127.0.0.1:8222`. When invoking Docker directly,
use `-p 127.0.0.1:8222:8222 -e CODERUNNER_API_TOKEN`. The process inside the guest
still listens on `0.0.0.0:8222`; do not forward this endpoint to the internet.
The Mac installer passes the credential to the container by environment
variable name, checks anonymous rejection, and verifies an authenticated
session-list request before reporting readiness. It does not rotate an existing
container's environment: to change the token, delete the container and run the
installer again.

This is an API credential, not a multi-user authorization system. Anyone with
it can execute arbitrary code, access the guest's files and network, and control
the browser. A task that can read the server process environment inside the
same guest may recover the credential. Keep the guest, mounted directories,
and approved code inside the same trust boundary. Jupyter sessions have separate
interpreter state but share a guest, filesystem and network; they are not
separate VMs or security sandboxes. HTTP is unencrypted; the loopback publication
and host/guest trust assumptions matter. Local malware with the same OS access
is outside this boundary.

The named-session execution path does not automatically retry a command after
a kernel failure. Clients must also avoid retrying uncertain commands. The
unnamed legacy execution API retains its retry behavior and should not be used
for consequential actions that require at-most-once dispatch.

Images published before this change ignore the token; the installer refuses
them. `tests/test_auth.py` covers HTTP, WebSocket, weak or missing
configuration, duplicated credentials and rotation; `test-e2e.sh` checks a
running sandbox.
