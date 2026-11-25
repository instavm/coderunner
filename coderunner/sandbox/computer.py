"""Give the sandbox VM its own desktop and Chrome: done by itself by the daemon (sandbox/keeper.py) when the app
shows the assistant's computer, and by `coderunner browser computer` from a terminal.

Xfce on a VNC screen and Google Chrome are installed inside the VM that already exists (desktop-setup.sh, the
script the image's DESKTOP=1 build runs), the sandbox server's desktop and browser routes are copied in beside
them, the sandbox restarts and browser.mode becomes "computer". The files come with the package (vm/), so an
installed copy works without a source checkout.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Callable

# The sandbox server and its desktop and browser routes, with what they import, as the VM keeps them.
FILES = (("desktop.py", "/app/desktop.py"), ("server.py", "/app/server.py"), ("auth.py", "/app/auth.py"),
         ("shell_session.py", "/app/shell_session.py"), ("entrypoint.sh", "/entrypoint.sh"))
SCRIPT = "desktop-setup.sh"


def files_dir() -> pathlib.Path | None:
    """Where the files for the VM are: beside this module in an installed package, else the checkout's root."""
    here = pathlib.Path(__file__).resolve().parent
    for folder in (here / "vm", here.parent.parent):
        if all((folder / name).is_file() for name, _ in FILES) and (folder / SCRIPT).is_file():
            return folder
    return None


def _run(argv: list[str], stdin, say: Callable[[str], None]) -> int:
    """A command in the sandbox, its output passed on as it comes; returns its exit status."""
    try:
        process = subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   errors="replace")
    except OSError as exc:
        say(str(exc))
        return 127
    for line in process.stdout:
        if line.strip():
            say(line.rstrip())
    return process.wait()


def install(driver, say: Callable[[str], None] = print, folder: pathlib.Path | None = None) -> bool:
    """Install the desktop in the running sandbox, without restarting it. False, with the reason said, when it
    could not."""
    folder = folder or files_dir()
    if folder is None or not (folder / SCRIPT).is_file():
        say("Error: the desktop's files are missing from this copy of CodeRunner.")
        return False
    say("Installing the desktop (Xfce, a VNC screen and Google Chrome) in the sandbox; this takes a few minutes...")
    for name, target in FILES:      # the sandbox server's desktop and browser routes
        argv = driver.exec_argv(["sh", "-c", f"cat > {target}"], tty=False)
        with open(folder / name, "rb") as source:
            if argv is None or _run(argv, source, say):
                say(f"Error: could not copy {name} into the sandbox.")
                return False
    with open(folder / SCRIPT, "rb") as source:
        if _run(driver.exec_argv(["bash", "-s"], tty=False), source, say):
            say("Error: the desktop did not install; see the lines above.")
            return False
    return True


def setup(config, say: Callable[[str], None] = print, driver=None, checkout: pathlib.Path | None = None) -> bool:
    """`coderunner browser computer`: install the desktop, restart the sandbox so its server picks it up, and
    switch the assistant's browser to it."""
    from . import driver_for
    driver = driver or driver_for(config)
    if driver.status() != "running" and not driver.start():
        say("Error: the sandbox is not running and did not start. Check: coderunner sandbox status")
        return False
    if not install(driver, say, checkout):
        return False
    say("Restarting the sandbox so its server picks up the desktop...")
    driver.stop()
    driver.start()
    config.set("browser.mode", "computer")
    config.save()
    return True
