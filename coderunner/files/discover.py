"""Which files the assistant may index and read: the shared folders, and the rules applied inside them.

Only regular files inside the folders the owner shared are read, after resolving symlinks. Secrets (key
files, .env files, ~/.ssh, keychains, password databases), app data under ~/Library, hidden files, `*.noindex`
folders and CodeRunner's own state are never indexed or read. Files kept only in iCloud (macOS "Optimize
Storage") are downloaded to be read when they are small enough (files.cloud_file_mb, 20 MB; files.cloud_files
reads them all), then given back to iCloud; larger ones are indexed by name, since reading one downloads it.
Source code inside git checkouts is left to code tools unless files.code_in_repos is on (a few checkouts hold
hundreds of thousands of files); everywhere, generated and minified files and source files over 350 KB are
skipped, as code search engines do. App bundles and libraries (Photos, Music...) are single documents to macOS
and are skipped.

Folders are found by their names too: every folder the walk enters below a shared one, except inside a git
checkout, where folders are the code's layout (`src`, `utils`, `tests`); the checkout's own folder is one.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
from dataclasses import dataclass, field
from typing import Iterator

from ..core import paths
from . import attached
from .kinds import MAX_CODE_BYTES, PROSE, is_code, is_generated, is_screenshot, kind_of

SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store", "Keychains", ".coderunner"}
SECRET_FILES = (".env", ".env.*", "*.env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials",
                "credentials.*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "*.pem", "*.key", "*.p12",
                "*.pfx", "*.jks", "*.keystore", "*.kdbx", "*.keychain", "*.keychain-db", "*.ovpn", "*.gpg", "*.asc",
                "*.p8", "*.mobileprovision")
PACKAGES = {".app", ".bundle", ".framework", ".photoslibrary", ".musiclibrary", ".tvlibrary", ".imovielibrary",
            ".fcpbundle", ".logicx", ".band", ".lrlibrary", ".xcodeproj", ".xcworkspace", ".xcarchive", ".plugin",
            ".kext", ".appex", ".photolibrary", ".aplibrary", ".dtbase2", ".sparsebundle", ".pkg", ".mpkg",
            ".noindex", ".movpkg"}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".tox", "target", "build", "dist", ".next", ".cache", ".Trash", "DerivedData", "Pods", "Carthage",
             "bower_components", "vendor", "vendors", "third_party", "third-party", "site-packages", "coverage",
             "__generated__"}
DATALESS = 0x40000000      # st_flags: the content is only in iCloud, and reading the file downloads it
DEFAULT_CLOUD_MB = 20      # the largest file only iCloud holds that is downloaded to be read


class FileAccessError(PermissionError):
    pass


def is_secret(path: pathlib.Path) -> bool:
    if any(part in SECRET_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILES)


def in_cloud_only(stat: os.stat_result) -> bool:
    return bool(getattr(stat, "st_flags", 0) & DATALESS)


COPY_SAMPLE = 65536         # bytes hashed from the start, middle and end of a file to tell copies


def content_fingerprint(path: str) -> bytes | None:
    """The file's size and the SHA-256 of bytes from its start, middle and end; None for a file only iCloud
    holds (reading it would download it) or one that is gone."""
    import hashlib
    try:
        stat = os.stat(path)
        if in_cloud_only(stat):
            return None
        with open(path, "rb") as handle:
            if stat.st_size <= 3 * COPY_SAMPLE:
                sample = handle.read()
            else:
                sample = b""
                for start in (0, (stat.st_size - COPY_SAMPLE) // 2, stat.st_size - COPY_SAMPLE):
                    handle.seek(start)
                    sample += handle.read(COPY_SAMPLE)
        return stat.st_size.to_bytes(8, "little") + hashlib.sha256(sample).digest()
    except OSError:
        return None


@dataclass
class Found:
    path: pathlib.Path
    stat: os.stat_result
    remote: bool            # only iCloud holds the content and it is too large to fetch: index the name only
    in_repo: bool = False   # inside a git checkout: mostly other people's code and docs
    repo: str | None = None     # the checkout's root, for titling code by its path inside it
    screenshot: bool = False    # read with the documents, and not held back on battery
    evicted: bool = False       # only iCloud held the content: it is downloaded to be read, then given back
    folder: bool = False        # a folder, found by its name and the kinds of file in it
    holds: tuple = field(default_factory=tuple)   # a folder: the kinds of file directly in it

    @property
    def kind(self) -> str | None:
        return "folder" if self.folder else kind_of(self.path)

    @property
    def device(self) -> int:
        return int(getattr(self.stat, "st_dev", 0) or 0)

    @property
    def inode(self) -> int:
        return int(getattr(self.stat, "st_ino", 0) or 0)

    @property
    def mtime(self) -> float:
        # A folder's creation time stands in for its modification time, which changes each time a file comes or
        # goes: the folder is read again when it is renamed or moved, or a kind of file comes or goes.
        if self.folder:
            return float(getattr(self.stat, "st_birthtime", None) or self.stat.st_mtime)
        return self.stat.st_mtime

    @property
    def size(self) -> int:
        return 0 if self.folder else self.stat.st_size

    @property
    def profile(self) -> str:
        """What the file's pieces depend on besides its bytes: a folder's kinds of file; a table or Office file
        read for its contents (t1, x1), so only those are read again when that reading arrives."""
        if self.folder:
            return " ".join(self.holds)
        suffix = self.path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            return "t1"
        if suffix in (".xlsx", ".pptx", ".numbers", ".key", ".pages"):
            return "x1"
        return ""


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

    def cloud_limit(self) -> int:
        """The largest file only iCloud holds that is downloaded to be read, in bytes (0: none)."""
        if self.config.get("files.cloud_files", False):
            return 1 << 62
        mb = self.config.get("files.cloud_file_mb", DEFAULT_CLOUD_MB)
        return int(float(mb if mb is not None else DEFAULT_CLOUD_MB) * 1024 * 1024)

    # ---- sharing ---------------------------------------------------------------

    def resolve(self, path: str) -> pathlib.Path:
        """The real path of a shared-folder file the assistant may read, or FileAccessError."""
        real = pathlib.Path(os.path.expanduser(attached.on_host(str(path)))).resolve()
        if attached.is_attached(real):          # a file the owner handed over in the app, kept under the home
            return real
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

    def walk(self, starts: list[tuple[pathlib.Path, str | None]] | None = None) -> Iterator[Found]:
        """Every file and folder to index in the shared folders (or under `starts`, each with the checkout it is
        in, if any)."""
        self.cloud_only = 0
        roots = self.roots()
        begin = starts if starts is not None else [(root, None) for root in roots]
        for start, parent_repo in begin:
            stack: list[tuple[str, str | None]] = [(str(start), parent_repo)]
            while stack:
                folder, repo = stack.pop()
                try:
                    with os.scandir(folder) as entries:
                        listed = list(entries)
                except OSError:
                    continue
                names = {e.name for e in listed}
                parent_repo = repo
                if repo is None and ".git" in names:
                    repo = folder
                files = [e for e in listed if e.is_file(follow_symlinks=False)]
                if parent_repo is None and pathlib.Path(folder) not in roots:
                    found = self.folder_entry(pathlib.Path(folder), [e.name for e in files])
                    if found is not None:
                        yield found
                for entry in files:
                    found = self._file(pathlib.Path(entry.path), repo)
                    if found is not None:
                        yield found
                for entry in sorted((e for e in listed if e.is_dir(follow_symlinks=False)), key=lambda e: e.name,
                                    reverse=True):
                    if self._enter(folder, entry.name):
                        stack.append((entry.path, repo))

    def folder_entry(self, path: pathlib.Path, names: list[str] | None = None) -> Found | None:
        """A folder, to find by its name and the kinds of file in it. None for one only iCloud holds (its listing
        would be downloaded) or one that cannot be read."""
        try:
            stat = os.lstat(path)
        except OSError:
            return None
        if in_cloud_only(stat) and self.cloud_limit() == 0:
            return None
        if names is None:
            try:
                with os.scandir(path) as entries:
                    names = [e.name for e in entries if e.is_file(follow_symlinks=False)]
            except OSError:
                return None
        kinds = {kind_of(pathlib.Path(n)) for n in names if not n.startswith(".")}
        kinds.discard(None)
        holds = tuple(k for k in ("doc", "text", "image", "video", "audio") if k in kinds)
        if any(is_code(n) for n in names if not n.startswith(".")):
            holds += ("code",)
        return Found(path, stat, False, folder=True, holds=holds)

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
        repo = next((str(parent) for parent in [path.parent, *path.parent.parents]
                     if (parent == root or root in parent.parents) and (parent / ".git").exists()), None)
        return self._file(path, repo)

    def folder_start(self, path: pathlib.Path) -> tuple[pathlib.Path, str | None] | None:
        """Whether a folder (a folder event) is one a walk would enter, and the checkout it is in."""
        path = path.parent.resolve() / path.name
        root = next((r for r in self.roots() if r in path.parents), None)
        if root is None or not path.is_dir():
            return None
        folder = root
        repo = None
        for part in path.relative_to(root).parts:
            if repo is None and (folder / ".git").exists():
                repo = str(folder)
            if not self._enter(str(folder), part):
                return None
            folder = folder / part
        return path, repo

    def _enter(self, folder: str, name: str) -> bool:
        if name in SKIP_DIRS or name in SECRET_DIRS or name.startswith("."):
            return False
        if os.path.splitext(name)[1].lower() in PACKAGES or pathlib.Path(folder, name) in self.private_dirs():
            return False
        try:
            remote = in_cloud_only(os.lstat(os.path.join(folder, name)))
        except OSError:
            return False
        if remote and self.cloud_limit() == 0:
            # Listing a folder only iCloud holds downloads its listing: not when nothing in it would be read anyway.
            self.cloud_only += 1
            return False
        return True

    def _file(self, path: pathlib.Path, repo: str | None) -> Found | None:
        kind = kind_of(path)
        if path.name.startswith(".") or kind is None or is_secret(path) or path.is_symlink():
            return None
        if kind == "text" and is_generated(path.name):
            return None
        in_repo = repo is not None
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
        if is_code(path) and stat.st_size > MAX_CODE_BYTES:
            return None
        dataless = in_cloud_only(stat)
        remote = dataless and stat.st_size > self.cloud_limit()
        self.cloud_only += remote
        found = Found(path, stat, remote, in_repo, repo)
        found.evicted = dataless and not remote
        found.screenshot = kind in ("image", "video") and is_screenshot(path)
        return found
