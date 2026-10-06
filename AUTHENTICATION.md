# Authentication and network boundaries

This unreleased branch adds a bearer-credential boundary around both MCP and
REST routes. Set `CODERUNNER_API_TOKEN` to an existing operator-managed secret of
at least 32 bytes with no whitespace. Every request except `GET/HEAD /health`
requires an exact bearer credential. Missing or invalid server configuration
returns 503; missing, duplicated, or incorrect client credentials return 401.
Host and Origin validation still applies. Authentication does not replace it.

Docker Compose publishes only `127.0.0.1:8222`. When invoking Docker directly,
use `-p 127.0.0.1:8222:8222 -e CODERUNNER_API_TOKEN`. The process inside the guest
still listens on `0.0.0.0:8222`; do not forward this endpoint to the internet.
The Mac installer retains its existing network choices and now passes the
credential by environment-variable name, checks anonymous rejection, and
verifies an authenticated session-list request before reporting readiness.
It does not rotate an existing container's environment. Recreating a container
with an authenticated build is an explicit operator action.

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

No image was built or launched for this change. Previously published images
may be unauthenticated. Source-level middleware tests cover HTTP, WebSocket,
weak/missing configuration, duplicated credentials and rotation; a real image,
installer, network and client compatibility check is still required before
release. The installer can perform downloads, system restarts and DNS changes;
do not use it as a read-only verifier.
