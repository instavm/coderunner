"""Find and start the user's Chromium-family browser under a dedicated profile.

The browser is controlled only through --remote-debugging-pipe, so no
debugging port is ever opened. It always runs with the assistant's own
profile directory, never the user's everyday profile.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import subprocess
import sys

from .cdp import CDPError, CDPPipe, PipeTransport

MAC_APPS = [
    "Google Chrome.app/Contents/MacOS/Google Chrome",
    "Brave Browser.app/Contents/MacOS/Brave Browser",
    "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "Chromium.app/Contents/MacOS/Chromium",
]
LINUX_NAMES = [
    "google-chrome", "google-chrome-stable",
    "brave-browser", "brave",
    "microsoft-edge", "microsoft-edge-stable",
    "chromium", "chromium-browser",
]

# Places the pipe fds at 3 and 4, where Chrome expects them, then execs the browser.
TRAMPOLINE = "import os,sys; os.dup2(int(sys.argv[1]),3); os.dup2(int(sys.argv[2]),4); os.execv(sys.argv[3], sys.argv[3:])"


def _executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_browser(config) -> str | None:
    configured = (config.get("browser.executable") or "").strip()
    if configured:
        path = os.path.expanduser(configured)
        if _executable(path):
            return path
        return shutil.which(configured)
    if sys.platform == "darwin":
        for root in ("/Applications", os.path.expanduser("~/Applications")):
            for app in MAC_APPS:
                path = os.path.join(root, app)
                if _executable(path):
                    return path
    for name in LINUX_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _above_four(fd: int) -> int:
    # The trampoline overwrites fds 3 and 4, so its inputs must live elsewhere.
    import fcntl

    if fd > 4:
        return fd
    moved = fcntl.fcntl(fd, fcntl.F_DUPFD, 5)
    os.close(fd)
    os.set_inheritable(moved, False)
    return moved


def browser_args(profile_dir: pathlib.Path, headless: bool) -> list[str]:
    args = [
        f"--user-data-dir={profile_dir}",
        "--remote-debugging-pipe",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-crash-restore-bubble",
        "--lang=en-US",
        "--disable-blink-features=AutomationControlled",
    ]
    if headless:
        args += [
            "--headless=new",
            "--window-size=1280,900",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
        ]
    return args + ["about:blank"]


async def launch(executable: str, profile_dir, headless: bool = False) -> tuple[subprocess.Popen, CDPPipe]:
    if os.name != "posix":
        raise OSError("the local browser needs macOS or Linux (pipe-based DevTools control)")
    profile_dir = pathlib.Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(profile_dir, 0o700)
    commands_r, commands_w = os.pipe()     # we write, browser reads on fd 3
    replies_r, replies_w = os.pipe()       # browser writes on fd 4, we read
    child_r, child_w = _above_four(commands_r), _above_four(replies_w)
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", TRAMPOLINE, str(child_r), str(child_w), executable,
             *browser_args(profile_dir, headless)],
            pass_fds=(child_r, child_w), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        for fd in (commands_w, replies_r):
            os.close(fd)
        raise
    finally:
        os.close(child_r)
        os.close(child_w)
    cdp = CDPPipe(PipeTransport(replies_r, commands_w))
    await cdp.start()
    try:
        await cdp.send("Browser.getVersion", timeout=20)
    except CDPError as exc:
        cdp.close()
        exited = process.poll() is not None
        await asyncio.to_thread(terminate, process)
        if exited:   # typically another browser process already owns this profile
            raise CDPError(f"the browser exited at startup ({exc}); "
                           f"is {profile_dir} in use by another browser?") from None
        raise
    return process, cdp


def terminate(process: subprocess.Popen, grace: float = 5.0) -> None:
    """Stop the browser and every helper in its process group."""
    if process.poll() is None:
        _signal_group(process.pid, 15)
        try:
            process.wait(grace)
        except subprocess.TimeoutExpired:
            _signal_group(process.pid, 9)
            process.wait(5)
    # Helpers (renderers, GPU, crash handler) share the group; make sure none linger.
    _signal_group(process.pid, 9)


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
