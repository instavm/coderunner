
import os
import pathlib
import socket
import sys

from fastmcp import FastMCP


def api_token() -> str:
    """CODERUNNER_API_TOKEN, else the token the installer saved in ~/.coderunner/api-token."""
    token = os.environ.get("CODERUNNER_API_TOKEN")
    if token:
        return token
    home = pathlib.Path(os.environ.get("CODERUNNER_HOME", pathlib.Path.home() / ".coderunner"))
    return (home / "api-token").read_text().strip()


def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}", file=sys.stderr)
        return None

hostname = "coderunner.local"
address = resolve_with_system_dns(hostname)
if address is None:
    print(
        f"Could not resolve {hostname}. Make sure the coderunner container is "
        "running (container list) and local DNS is configured "
        "(sudo container system dns create local).",
        file=sys.stderr,
    )
    sys.exit(1)

# Create a proxy directly from a config dictionary
config = {
    "mcpServers": {
        "default": {  # For single server configs, 'default' is commonly used
            "url": f"http://{address}:8222/mcp",
            "headers": {"Authorization": "Bearer " + api_token()},
            "transport": "http"
        }
    }
}


proxy = FastMCP.as_proxy(config, name="SSE to Stdio Proxy")
# Run the proxy with stdio transport for local access
if __name__ == "__main__":
    proxy.run()
