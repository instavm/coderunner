"""Which files the assistant may index and read: the shared folders, and the rules applied inside them.

Only regular files inside the folders the owner shared are read, after resolving symlinks. Secrets (key
files, .env files, ~/.ssh, keychains, password databases), app data under ~/Library and CodeRunner's own
state are never indexed or read. Files kept only in iCloud (macOS "Optimize Storage") are indexed by name
unless files.cloud_files is on, since reading one downloads it, and folders kept only in iCloud are not
listed at all: listing one waits for a download. Source code inside git checkouts is left to code tools
unless files.code_in_repos is on (a few checkouts hold hundreds of thousands of files). App bundles and
libraries (Photos, Music...) are single documents to macOS and are skipped.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
from dataclasses import dataclass
from typing import Iterator

from ..core import paths
from .kinds import PROSE, kind_of

SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store", "Keychains", ".coderunner"}
SECRET_FILES = (".env", ".env.*", "*.env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials",
                "credentials.*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "*.pem", "*.key", "*.p12",
                "*.pfx", "*.jks", "*.keystore", "*.kdbx", "*.keychain", "*.keychain-db", "*.ovpn", "*.gpg", "*.asc")
PACKAGES = {".app", ".bundle", ".framework", ".photoslibrary", ".musiclibrary", ".tvlibrary", ".imovielibrary",
            ".fcpbundle", ".logicx", ".band", ".lrlibrary", ".xcodeproj", ".xcworkspace"}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".tox", "target", "build", "dist", ".next", ".cache", ".Trash", "DerivedData", "Pods"}
DATALESS = 0x40000000      # st_flags: the content is only in iCloud, and reading the file downloads it


class FileAccessError(PermissionError):
    pass


def is_secret(path: pathlib.Path) -> bool:
    if any(part in SECRET_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILES)


def in_cloud_only(stat: os.stat_result) -> bool:
    return bool(getattr(stat, "st_flags", 0) & DATALESS)


@dataclass
class Found:
    path: pathlib.Path
    stat: os.stat_result
    remote: bool            # only iCloud holds the content: index the name, never download it
    in_repo: bool = False   # inside a git checkout: mostly other people's code and docs


class Folders:
    """The shared folders and the rules for what inside them may be indexed and read."""

    def __init__(self, config):
        self.config = config
        self.cloud_only = 0     # files and folders left alone (or named only) because only iCloud has them

    def roots(self) -> list[pathlib.Path]:
        return [pathlib.Path(os.path.expanduser(r)).resolve() for r in self.config.get("files.roots", []) or []]

    @staticmethod
    def private_dirs() -> set[pathlib.Path]:
        # App data under the home folder (mail, messages, keychains, browser profiles) is never indexed.
        return {pathlib.Path.home().resolve() / "Library"}

    # ---- sharing ---------------------------------------------------------------

    def resolve(self, path: str) -> pathlib.Path:
        """The real path of a shared-folder file the assistant may read, or FileAccessError."""
        real = pathlib.Path(os.path.expanduser(str(path))).resolve()
        root = next((r for r in self.roots() if r in real.parents), None)
        if root is None:
            raise FileAccessError(f"{path} is not in a folder the owner shared (coderunner files add <folder>)")
        state = paths.home().resolve()
        hidden = any(part.startswith(".") for part in real.relative_to(root).parts)
        private = any(d == real or d in real.parents for d in self.private_dirs())
        if real == state or state in real.parents or is_secret(real) or hidden or private:
            raise FileAccessError(f"{path} may hold secrets or private app data and is never read")
        if not real.is_file():
            raise FileAccessError(f"{path} is not a file")
        return real

    def add(self, folder: str) -> str:
        real = pathlib.Path(os.path.expanduser(folder)).resolve()
        if not real.is_dir():
            raise ValueError(f"{folder} is not a folder")
        state = paths.home().resolve()
        if real == state or state in real.parents:
            raise ValueError("that folder holds CodeRunner's own state")
        if len(real.parts) <= 2 or real in self.private_dirs():
            raise ValueError("choose specific folders, not the whole disk or app data")
        roots = [str(r) for r in self.roots()]
        if str(real) not in roots:
            self.config.set("files.roots", roots + [str(real)])
            self.config.save()
        return str(real)

    def remove(self, folder: str) -> None:
        real = pathlib.Path(os.path.expanduser(folder)).resolve()
        self.config.set("files.roots", [str(r) for r in self.roots() if r != real])
        self.config.save()

    def inside(self, path: str) -> bool:
        return any(root in pathlib.Path(path).parents for root in self.roots())

    # ---- what to index ---------------------------------------------------------

    def walk(self) -> Iterator[Found]:
        """Every file to index in the shared folders."""
        self.cloud_only = 0
        repos: set[str] = set()                     # folders inside a git checkout
        for root in self.roots():
            for folder, dirs, names in os.walk(root, followlinks=False):
                in_repo = folder in repos or ".git" in dirs or ".git" in names
                dirs[:] = [d for d in dirs if self._enter(folder, d)]
                if in_repo:
                    repos.update(os.path.join(folder, d) for d in dirs)
                for name in names:
                    found = self._file(pathlib.Path(folder) / name, in_repo)
                    if found is not None:
                        yield found

    def check(self, path: pathlib.Path) -> Found | None:
        """The same rules for one path (a file event), without walking."""
        path = path.parent.resolve() / path.name          # roots are resolved (/var is /private/var on macOS)
        root = next((r for r in self.roots() if r in path.parents), None)
        if root is None:
            return None
        relative = path.relative_to(root).parts[:-1]
        folder = root
        for part in relative:
            if not self._enter(str(folder), part):
                return None
            folder = folder / part
        in_repo = any((parent / ".git").exists() for parent in [path.parent, *path.parent.parents]
                      if parent == root or root in parent.parents)
        return self._file(path, in_repo)

    def _enter(self, folder: str, name: str) -> bool:
        if name in SKIP_DIRS or name in SECRET_DIRS or name.startswith("."):
            return False
        if os.path.splitext(name)[1].lower() in PACKAGES or pathlib.Path(folder, name) in self.private_dirs():
            return False
        if self.config.get("files.cloud_files", False):
            return True
        try:
            remote = in_cloud_only(os.lstat(os.path.join(folder, name)))
        except OSError:
            return False
        self.cloud_only += remote
        return not remote

    def _file(self, path: pathlib.Path, in_repo: bool) -> Found | None:
        kind = kind_of(path)
        if path.name.startswith(".") or kind is None or is_secret(path) or path.is_symlink():
            return None
        if in_repo and kind == "text" and path.suffix.lower() not in PROSE \
                and not self.config.get("files.code_in_repos", False):
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        limit = int(self.config.get("files.max_file_mb", 200) or 200) * 1024 * 1024
        if not 0 < stat.st_size <= limit:
            return None
        remote = not self.config.get("files.cloud_files", False) and in_cloud_only(stat)
        self.cloud_only += remote
        return Found(path, stat, remote, in_repo)
