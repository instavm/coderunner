
from fastmcp import FastMCP
import socket


def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}")
        return None

hostname = "coderunner.local"
address = resolve_with_system_dns(hostname)
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
