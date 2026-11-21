"""Give the sandbox VM its own desktop and Chrome: `coderunner browser computer`, and the app's Computer panel.

Xfce on a VNC screen and Google Chrome are installed inside the VM that already exists (desktop-setup.sh, the
script the image's DESKTOP=1 build runs), the sandbox server's desktop and browser routes are copied in beside
them, the sandbox restarts and browser.mode becomes "computer". What the installation prints is passed on line
by line, so a terminal and the app show the same progress.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Callable

FILES = (("desktop.py", "/app/desktop.py"), ("server.py", "/app/server.py"), ("entrypoint.sh", "/entrypoint.sh"))


def _run(argv: list[str], stdin, say: Callable[[str], None]) -> int:
    """A command in the sandbox, its output passed on as it comes; returns its exit status."""
    process = subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               errors="replace")
    for line in process.stdout:
        if line.strip():
            say(line.rstrip())
    return process.wait()


def setup(config, say: Callable[[str], None] = print, driver=None, checkout: pathlib.Path | None = None) -> bool:
    """Install the desktop in the running sandbox and switch the assistant's browser to it. False, with the
    reason said, when it could not."""
    from . import driver_for
    driver = driver or driver_for(config)
    if driver.status() != "running" and not driver.start():
        say("Error: the sandbox is not running and did not start. Check: coderunner sandbox status")
        return False
    checkout = checkout or pathlib.Path(__file__).resolve().parent.parent.parent
    script = checkout / "desktop-setup.sh"
    if not script.exists():
        say("Error: desktop-setup.sh was not found next to this copy of CodeRunner; run this from a checkout.")
        return False
    say("Installing the desktop (Xfce, a VNC screen and Google Chrome) in the sandbox; this takes a few minutes...")
    for name, target in FILES:      # the sandbox server's desktop and browser routes
        argv = driver.exec_argv(["sh", "-c", f"cat > {target}"], tty=False)
        with open(checkout / name, "rb") as source:
            if argv is None or _run(argv, source, say):
                say(f"Error: could not copy {name} into the sandbox.")
                return False
    with open(script, "rb") as source:
        if _run(driver.exec_argv(["bash", "-s"], tty=False), source, say):
            say("Error: the desktop did not install; see the lines above.")
            return False
    say("Restarting the sandbox so its server picks up the desktop...")
    driver.stop()
    driver.start()
    config.set("browser.mode", "computer")
    config.save()
    return True
