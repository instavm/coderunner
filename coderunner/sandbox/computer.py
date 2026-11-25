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


# What a VM created before the data volume keeps on its own disk, carried across `coderunner sandbox reset`: the
# desktop Chrome's profile (without its caches), Downloads, and /workspace when nothing from the host is mounted
# there. Paths that already link into /data live on the volume and need no carrying.
CARRY_OUT = r"""cd / || exit 1
home="$(getent passwd "${CODERUNNER_DESKTOP_USER:-box}" | cut -d: -f6)"; home="${home#/}"
set --
for p in "$home/.config/coderunner-chrome" "$home/Downloads"; do
    [ -n "$home" ] && [ -d "$p" ] && [ ! -L "$p" ] && set -- "$@" "$p"
done
if [ -d workspace ] && [ ! -L workspace ] && ! awk '$2 == "/workspace" || index($2, "/workspace/") == 1 {f=1} END {exit !f}' /proc/mounts; then
    set -- "$@" workspace
fi
[ $# -eq 0 ] && exit 0
exec tar -cf - --exclude='*/Cache' --exclude='*/Code Cache' --exclude='*/GPUCache' --exclude='*/Singleton*' "$@"
"""
CARRY_IN = "tar -C / --keep-directory-symlink -xpf -"


def carry_out(driver, target: pathlib.Path, say: Callable[[str], None] = print) -> bool | None:
    """Copy what the owner would miss out of the VM into `target` (a tar file) before the VM is deleted. None when
    there was nothing to carry, False when it could not be copied."""
    argv = driver.exec_argv(["sh", "-c", CARRY_OUT], tty=False)
    if argv is None:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "wb") as out:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        say(f"Could not copy the browser profile, downloads and workspace out of the sandbox: {exc}")
        target.unlink(missing_ok=True)
        return False
    if done.returncode != 0:
        say("Could not copy the browser profile, downloads and workspace out of the sandbox: "
            + done.stderr.decode(errors="replace").strip()[-300:])
        target.unlink(missing_ok=True)
        return False
    if target.stat().st_size == 0:
        target.unlink()
        return None
    return True


def carry_in(driver, source: pathlib.Path, say: Callable[[str], None] = print) -> bool:
    """Put what carry_out kept into the new VM (through its links into the data volume), then drop the copy."""
    argv = driver.exec_argv(["sh", "-c", CARRY_IN], tty=False)
    if argv is None:
        return False
    try:
        with open(source, "rb") as given:
            done = subprocess.run(argv, stdin=given, capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        say(f"Could not put the browser profile, downloads and workspace back: {exc}. They are kept in {source}.")
        return False
    if done.returncode != 0:
        say(f"Could not put the browser profile, downloads and workspace back. They are kept in {source}.")
        return False
    source.unlink(missing_ok=True)
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
