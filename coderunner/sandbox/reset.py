"""Creating the sandbox VM again: `coderunner sandbox reset`, and the keeper when the VM's disk has grown past its
budget (sandbox/keeper.py).

The VM and its filesystem are deleted and the installer creates it again (--no-assistant: only the sandbox, not
the models). Settings not given are kept from the old VM: a reset must not reopen the network of a `--network
none` sandbox, drop its mounts, or move it to another CodeRunner home. What the owner would miss lives on the data
volume, which outlives the VM; a VM from before the volume has it copied out first and put back after.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Callable

from ..core import paths
from . import computer


def installer() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent / "install.sh"


def carry_file() -> pathlib.Path:
    return paths.home() / "sandbox-carry.tar"


def options_for(driver, state: str, *, cpus: str | None = None, memory: str | None = None,
                network: str | None = None, mounts: list[str] | None = None,
                build: bool = False, desktop: bool = False) -> tuple[list[str], dict]:
    """The installer's arguments and environment for the VM made again, from what is given and the old VM."""
    kept = driver.settings() if state != "missing" else {}
    chosen = {"cpus": cpus or kept.get("cpus"), "memory": memory or kept.get("memory"),
              "network": network or kept.get("network"), "profile": kept.get("profile")}
    options = [part for key, value in chosen.items() if value for part in (f"--{key}", value)]
    for mount in mounts or kept.get("mounts") or []:     # given mounts replace the old ones
        options += ["--mount", mount]
    # The same image again unless a build asks for a fresh one: published images can lag behind the checkout.
    # --no-assistant alone builds an image without the desktop, and the daemon then installs it into every VM made
    # from it (minutes of downloads, and the VM's disk back near its budget): an assistant's build keeps it baked in.
    options += ["--no-assistant", *(["--build", *(["--desktop"] if desktop else [])] if build else
                                    ["--image", kept["image"]] if kept.get("image") else [])]
    env = {**os.environ, **({"CODERUNNER_HOME": kept["home"]} if kept.get("home") else {})}
    return options, env


def recreate(driver, state: str, options: list[str], env: dict, say: Callable[[str], None] = print,
             needs_installer: bool = False, log: pathlib.Path | None = None) -> int:
    """Delete the VM and run the installer to create it again; the installer's exit status. needs_installer: with
    no installer here, nothing is deleted (the daemon cannot tell anyone to run it by hand). log: where the
    installer's output goes, instead of the terminal."""
    script, carried = installer(), carry_file()
    if needs_installer and not script.exists():
        say("No installer beside this copy of CodeRunner; the sandbox was left as it is.")
        return 1
    if state != "missing":
        # A sandbox from before the data volume keeps the desktop Chrome's sign-ins, its downloads and the
        # workspace on its own disk: they are copied out first and put back into the new one.
        if state == "running":
            say("Copying the browser profile, downloads and workspace out of the sandbox...")
            if computer.carry_out(driver, carried, say) is False:
                say("Could not preserve the browser profile, downloads and workspace; the sandbox was left as it is.")
                return 1
        if not driver.delete():
            say("❌ The sandbox could not be deleted.")
            return 1
        say("Deleted the sandbox.")
    if not script.exists():
        say(f"Create it again from the CodeRunner checkout: ./install.sh {' '.join(options)}")
        if carried.exists():
            say(f"The browser profile, downloads and workspace are kept in {carried}.")
        return 0
    say(f"Recreating it: ./install.sh {' '.join(options)}")
    if log is None:
        code = subprocess.call([str(script), *options], cwd=script.parent, env=env)
    else:                           # nobody at a terminal: the installer's output goes to the log
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as out:
            code = subprocess.call([str(script), *options], cwd=script.parent, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT)
    if carried.exists():
        if code == 0 and driver.status() == "running" and computer.carry_in(driver, carried, say):
            say("Put the browser profile, downloads and workspace back.")
        else:
            say(f"The browser profile, downloads and workspace are kept in {carried}.")
    return code


def recreate_quietly(driver, say: Callable[[str], None] = lambda line: None) -> bool:
    """What the keeper does: the VM made again with its own settings and image, nothing asked."""
    state = driver.status()
    options, env = options_for(driver, state)
    log = paths.assistant_dir() / "logs" / "sandbox-recreate.log"
    return recreate(driver, state, options, env, say, needs_installer=True, log=log) == 0
