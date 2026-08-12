
from fastmcp import FastMCP
import socket
import sys


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
            "transport": "http"
        }
    }
}


proxy = FastMCP.as_proxy(config, name="SSE to Stdio Proxy")
# Run the proxy with stdio transport for local access
if __name__ == "__main__":
    proxy.run()
