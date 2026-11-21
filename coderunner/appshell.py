"""The desktop app from the command line: `coderunner app`.

app/ in the checkout is the app's source, a thin Electron shell around the page the daemon serves. Electron
itself comes from the checkout's node_modules where the owner ran `npm install` there, and is otherwise
downloaded: the release app/package.json pins, checked against that release's checksums, into
CODERUNNER_HOME/runtime. Opening the app therefore needs no Node.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import zipfile
from typing import Callable

import httpx

from .core import paths

ELECTRON = "45.2.3"         # as app/package.json
BUILDS = {  # (os, arch) -> (asset platform, sha256 of electron-v<version>-<platform>.zip)
    ("darwin", "arm64"): ("darwin-arm64", "4d7c429064087a47b855d9bd06c76ead4819b7f02b385de232d4a5c37ecef55f"),
    ("darwin", "x86_64"): ("darwin-x64", "d87323533cc8750ef177e5b48058f7a8b381f677d4e103cc63360006c08829c2"),
    ("linux", "x86_64"): ("linux-x64", "a87fe741f6d121285117e574da6c9a2e65210d410123bbc04713b099add0263b"),
    ("linux", "aarch64"): ("linux-arm64", "397c060fd9da99ab72b78787d65545b432fcadf13ef6c96658e4f525692aae0f"),
}


def app_dir() -> pathlib.Path | None:
    """The app's source beside this package (a checkout, as the installer leaves it)."""
    found = pathlib.Path(__file__).resolve().parent.parent / "app"
    return found if (found / "main.js").is_file() else None


def build(system: str | None = None, machine: str | None = None) -> tuple[str, str] | None:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
    if system == "linux" and machine == "arm64":
        machine = "aarch64"
    return BUILDS.get((system, machine))


def _binary(folder: pathlib.Path, system: str | None = None) -> pathlib.Path:
    if (system or platform.system()).lower() == "darwin":
        return folder / "Electron.app" / "Contents" / "MacOS" / "Electron"
    return folder / "electron"


def managed_dir() -> pathlib.Path:
    return paths.home() / "runtime" / f"electron-{ELECTRON}"


def installed(app: pathlib.Path | None = None) -> str | None:
    """Electron already here: the checkout's own (npm install), else CodeRunner's copy."""
    places = ([app / "node_modules" / "electron" / "dist"] if app else []) + [managed_dir()]
    for folder in places:
        found = _binary(folder)
        if found.is_file() and os.access(found, os.X_OK):
            return str(found)
    return None


def _unzip(archive: pathlib.Path, into: pathlib.Path) -> None:
    """Unpack keeping what a plain zip reader drops and an app bundle needs: links and executable bits."""
    if platform.system() == "Darwin" and os.path.exists("/usr/bin/ditto"):
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(into)], check=True, capture_output=True)
        return
    with zipfile.ZipFile(archive) as bundle:
        root = into.resolve()
        for item in bundle.infolist():
            target = (into / item.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"{archive.name} holds a path outside its folder: {item.filename}")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(bundle.read(item).decode(), target)
            else:
                bundle.extract(item, into)
                if mode and not item.is_dir():
                    os.chmod(target, stat.S_IMODE(mode))


def install(progress: Callable[[str], None] | None = None) -> str:
    """Download the pinned Electron into CODERUNNER_HOME/runtime and return its binary."""
    found = build()
    if found is None:
        raise RuntimeError("Electron publishes no build for this platform; open the owner page in a browser instead "
                           "(`coderunner remote pair`)")
    asset, sha256 = found
    name = f"electron-v{ELECTRON}-{asset}.zip"
    folder = managed_dir()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / name
    if progress:
        progress(f"downloading Electron {ELECTRON} for the app (about 120 MB, once)")
    digest = hashlib.sha256()
    with httpx.stream("GET", f"https://github.com/electron/electron/releases/download/v{ELECTRON}/{name}",
                      follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as response:
        response.raise_for_status()
        with open(archive, "wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != sha256:
        archive.unlink()
        raise RuntimeError(f"{name} does not match its published checksum")
    try:
        _unzip(archive, folder)
    finally:
        archive.unlink(missing_ok=True)
    binary = _binary(folder)
    if not binary.is_file():
        shutil.rmtree(folder, ignore_errors=True)
        raise RuntimeError(f"{name} has no Electron binary")
    return str(binary)


def launch(progress: Callable[[str], None] | None = None) -> subprocess.Popen:
    """Open the app in its own process; it finds this home's daemon and starts it when it is down."""
    app = app_dir()
    if app is None:
        raise RuntimeError("The app's source (app/) is not beside this copy of CodeRunner: run this from a checkout "
                           "installed with ./install.sh")
    binary = installed(app) or install(progress)
    paths.ensure_dirs()
    with open(paths.assistant_dir() / "logs" / "app.log", "ab") as log:
        return subprocess.Popen([binary, str(app)], env={**os.environ, "CODERUNNER_HOME": str(paths.home())},
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
