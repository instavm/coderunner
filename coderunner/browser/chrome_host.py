"""Native messaging host for the CodeRunner browser extension.

The browser starts this program when the extension connects (`coderunner browser connect` registers
it for the extension's ID only) and talks to it on stdin/stdout: JSON messages, each preceded by its
length in 4 bytes of native byte order. The host relays them to the assistant daemon over a unix
socket that only the owner's account can open, after presenting the token stored next to it, and
reconnects whenever the daemon restarts. Standard library only: the browser runs it outside the
assistant's environment.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

TO_BROWSER_LIMIT = 1024 * 1024      # the browser refuses longer messages from a host
FRAME = struct.Struct("!I")          # daemon socket: 4-byte big-endian length, then JSON
NATIVE = struct.Struct("=I")         # browser pipe: 4-byte native-order length, then JSON
RETRY = 2.0


def read_native(stream) -> dict | None:
    header = stream.read(NATIVE.size)
    if len(header) < NATIVE.size:
        return None
    data = stream.read(NATIVE.unpack(header)[0])
    try:
        return json.loads(data)
    except ValueError:
        return {}


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def read_frame(sock: socket.socket) -> dict | None:
    header = _recv_exact(sock, FRAME.size)
    if header is None:
        return None
    data = _recv_exact(sock, FRAME.unpack(header)[0])
    return None if data is None else json.loads(data)


def write_frame(sock: socket.socket, message: dict) -> None:
    data = json.dumps(message, separators=(",", ":")).encode()
    sock.sendall(FRAME.pack(len(data)) + data)


class Relay:
    def __init__(self, socket_path: str, token: str, origin: str, stdout=None):
        self.socket_path, self.token, self.origin = socket_path, token, origin
        self.stdout = stdout or sys.stdout.buffer
        self.sock: socket.socket | None = None
        self.out_lock = threading.Lock()
        self.sock_lock = threading.Lock()

    def to_browser(self, message: dict) -> None:
        data = json.dumps(message, separators=(",", ":")).encode()
        if len(data) > TO_BROWSER_LIMIT:
            if "id" in message:
                self.to_daemon({"id": message["id"], "error": {"message": "message too large for the browser"}})
            return
        with self.out_lock:
            self.stdout.write(NATIVE.pack(len(data)) + data)
            self.stdout.flush()

    def to_daemon(self, message: dict) -> None:
        with self.sock_lock:
            if self.sock is None:
                return                       # replies and events with nobody to read them
            try:
                write_frame(self.sock, message)
            except OSError:
                pass

    def from_browser(self, stdin) -> None:
        while True:
            message = read_native(stdin)
            if message is None:              # the browser closed the port
                os._exit(0)
            if message:
                self.to_daemon(message)

    def serve_daemon(self) -> None:
        """Connect to the daemon, relay its messages to the browser, and reconnect when it goes away."""
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.socket_path)
                write_frame(sock, {"type": "hello", "token": self.token, "origin": self.origin,
                                   "pid": os.getpid()})
                reply = read_frame(sock)
            except (OSError, ValueError):
                sock.close()
                time.sleep(RETRY)
                continue
            if not reply or reply.get("type") != "welcome":
                sock.close()                 # another browser is connected, or the token is stale
                time.sleep(RETRY * 3)
                continue
            with self.sock_lock:
                self.sock = sock
            self.to_browser({"type": "daemon", "connected": True})
            try:
                while True:
                    message = read_frame(sock)
                    if message is None:
                        break
                    self.to_browser(message)
            except (OSError, ValueError):
                pass
            finally:
                with self.sock_lock:
                    self.sock = None
                sock.close()
                self.to_browser({"type": "daemon", "connected": False})
            time.sleep(RETRY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeRunner native messaging host")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--extension", required=True, help="the only extension ID allowed to start this host")
    parser.add_argument("origin", nargs="?", default="")
    args, _ = parser.parse_known_args(argv)  # Windows adds --parent-window
    if args.origin != f"chrome-extension://{args.extension}/":
        print(f"refusing caller {args.origin!r}", file=sys.stderr)
        return 1
    with open(args.token_file) as handle:
        token = handle.read().strip()
    relay = Relay(args.socket, token, args.origin)
    threading.Thread(target=relay.from_browser, args=(sys.stdin.buffer,), daemon=True).start()
    relay.serve_daemon()
    return 0


if __name__ == "__main__":
    sys.exit(main())
