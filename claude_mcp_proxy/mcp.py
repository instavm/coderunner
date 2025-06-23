
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
proxy = FastMCP.as_proxy(f"http://{address}:8222/sse", name="SSE to Stdio Proxy")
# Run the proxy with stdio transport for local access
if __name__ == "__main__":
    proxy.run()