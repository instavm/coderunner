"""The assistant as a running service.

  daemon      starts and wires everything: model runtime, sandbox, browser, files, voice, channels
  gateway     the HTTP APIs: the owner's (chat, approvals, the web page in web/) and the sandbox's narrow one
  service     running the daemon at login (launchd, systemd)
  remote      private access from the owner's devices over Tailscale
  mcp_bridge  the assistant's guarded tools for MCP clients
"""
