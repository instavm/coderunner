"""Files the owner attaches to a task in the app: pictures, videos, recordings, documents.

Each is kept under its own id in a folder the sandbox also mounts (assets/outputs, /app/uploads/outputs inside
it), so the assistant can read it with the file tools and code in the sandbox can work on it. The owner gave the
file on purpose, so it is readable without being in a shared folder; nothing else under CodeRunner's home is.

Code in the sandbox can write to that folder too. What is served back or read is therefore always a regular
file directly inside its id's folder: a link planted there leads nowhere.
"""

from __future__ import annotations

import mimetypes
import os
import pathlib
import re
import secrets
import unicodedata
import urllib.parse

from ..core import paths
from .kinds import kind_of

SANDBOX_DIR = "/app/uploads/outputs/attached"
ID = re.compile(r"[0-9a-f]{12}")
MAX_PER_TASK = 8
SHOWN = 4                       # pictures put in front of the model with the message itself; the rest by file_view
# What the app may show in place, by its own type. Anything else (HTML and SVG above all, which would run as
# the daemon's own page) is only ever handed over as a download.
INLINE = re.compile(r"(image/(jpeg|png|gif|webp|heic|heif|avif|bmp|tiff)|video/[\w.+-]+|audio/[\w.+-]+|application/pdf)")
KINDS = {"text": "text", "doc": "document", "image": "image", "audio": "audio", "video": "video"}


class TooLarge(Exception):
    pass


def attached_dir() -> pathlib.Path:
    return paths.home() / "assets" / "outputs" / "attached"


def safe_name(name: str) -> str:
    """The file's own name, as one harmless path part."""
    name = unicodedata.normalize("NFC", str(name or "")).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if c.isprintable() and c not in '<>:"|?*').strip().lstrip(".")
    stem, dot, suffix = name.rpartition(".")
    if dot and stem:
        name = stem[:100] + "." + suffix[:12]
    return name[:120] or "file"


def limit_bytes(config) -> int:
    mb = config.get("files.max_file_mb") if config is not None else None
    return int(float(mb or 200) * 1024 * 1024)


async def save(chunks, name: str, limit: int) -> dict:
    """Write an upload as it arrives; TooLarge once it passes the limit, ValueError for an empty one."""
    upload_id = secrets.token_hex(6)
    folder = attached_dir() / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    path, part, size = folder / safe_name(name), folder / ".part", 0
    try:
        with open(part, "wb") as handle:
            async for chunk in chunks:
                size += len(chunk)
                if size > limit:
                    raise TooLarge(f"larger than {limit // (1024 * 1024)} MB")
                handle.write(chunk)
        if not size:
            raise ValueError("the file is empty")
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        if not any(folder.iterdir()):
            folder.rmdir()
        raise
    return describe(upload_id)


def path_of(upload_id: str) -> pathlib.Path | None:
    """The one file kept under this id, or None: no such id, or what is there is not a plain file of ours."""
    if not ID.fullmatch(str(upload_id or "")):
        return None
    folder = attached_dir() / upload_id
    try:
        found = [p for p in folder.iterdir() if not p.name.startswith(".")]
    except OSError:
        return None
    if len(found) != 1 or found[0].is_symlink() or not found[0].is_file():
        return None
    return found[0] if found[0].resolve().parent == folder.resolve() else None


def describe(upload_id: str) -> dict | None:
    path = path_of(upload_id)
    if path is None:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    kind = KINDS.get(kind_of(path) or "", "file")
    return {"id": upload_id, "name": path.name, "mime": mime, "size": path.stat().st_size, "kind": kind,
            "inline": bool(INLINE.fullmatch(mime)), "url": f"/api/uploads/{upload_id}"}


def on_host(path: str) -> str:
    """A path as code in the sandbox sees an attached file, as the host's own path for it; anything else as it
    is. The model is told both and names either to the file tools."""
    text = str(path)
    if text.startswith(SANDBOX_DIR + "/"):
        upload_id, _, name = text[len(SANDBOX_DIR) + 1:].partition("/")
        if ID.fullmatch(upload_id) and name and "/" not in name:
            return str(attached_dir() / upload_id / name)
    return text


# ---- what a task made: files code in the sandbox left in its outputs folder --------------------------------

PRODUCED_MAX = 8


def outputs_dir() -> pathlib.Path:
    return paths.home() / "assets" / "outputs"


def output_path(name: str) -> pathlib.Path | None:
    """A plain file directly in the outputs folder by its name, or None (a link, a folder, a name with a path)."""
    if not name or name != pathlib.PurePath(name).name or name.startswith("."):
        return None
    path = outputs_dir() / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != outputs_dir().resolve():
        return None
    return path


def outputs_now() -> dict[str, float]:
    """The files in the outputs folder with when each last changed (taken before a task, compared after)."""
    try:
        return {p.name: p.stat().st_mtime for p in outputs_dir().iterdir()
                if not p.name.startswith(".") and not p.is_symlink() and p.is_file()}
    except OSError:
        return {}


def produced(before: dict[str, float]) -> list[dict]:
    """What appeared or changed in the outputs folder since `before`, as the app shows it."""
    found = []
    for name, changed in sorted(outputs_now().items()):
        path = output_path(name)
        if path is None or before.get(name) == changed:
            continue
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        found.append({"name": name, "mime": mime, "size": path.stat().st_size, "kind": KINDS.get(kind_of(path) or "", "file"),
                      "inline": bool(INLINE.fullmatch(mime)), "url": f"/api/outputs/{urllib.parse.quote(name)}"})
    return found[:PRODUCED_MAX]


def is_attached(real: pathlib.Path) -> bool:
    """Whether this resolved path is a file the owner attached (the file tools' gate asks)."""
    base = attached_dir().resolve()
    if real.parent.parent != base:
        return False
    kept = path_of(real.parent.name)
    return kept is not None and kept.resolve() == real


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{max(n // 1000, 1)} KB"


async def for_the_model(ids: list[str], media, vision: bool, sandbox: bool) -> tuple[str, list, list[dict]]:
    """What a task's attachments add to the owner's message: a note naming each file, where it is and how to
    use it, the pictures the model is shown straight away (as (mime, base64) JPEGs), and what the app shows of
    them. Each file's line says what to do with that file: a small model that was only told "use the file
    tools" read a video's frame and never listened to it."""
    import base64
    shown, lines, images = [], [], []
    for upload_id in ids[:MAX_PER_TASK]:
        info, path = describe(upload_id), path_of(upload_id)
        if info is None:
            continue
        shown.append(info)
        kind, seen, length = info["kind"], False, ""
        if vision and kind in ("image", "video") and len(images) < SHOWN and media is not None:
            try:
                data = await media.image(path, at=1.0 if kind == "video" else None)
                images.append(("image/jpeg", base64.b64encode(data).decode()))
                seen = True
            except Exception:           # a picture that cannot be converted is still there for file_view
                pass
        if kind == "video" and media is not None:
            try:
                seconds = await media.duration(path)
                length = f", {seconds:.0f} s long" if seconds else ""
            except Exception:
                pass
        how = {"image": "You are shown it with this message." if seen else
                        "file_view shows it to you." if vision else "This model cannot see pictures.",
               "video": ("You are shown one frame of it (at 1 s) with this message; file_view shows other moments. "
                         if seen else "file_view shows a frame of it at a time. " if vision else "") +
                        "Its sound is not in the frame: file_listen gives what is said in it.",
               "audio": "file_listen gives what is said in it.",
               "document": "file_read reads it.", "text": "file_read reads it."}.get(kind, "")
        lines.append(f"- {info['name']} ({kind}, {_size(info['size'])}{length}) at {path}. {how}".rstrip())
    if not lines:
        return "", [], []
    note = "Files the owner attached to this message:\n" + "\n".join(lines)
    if sandbox:
        note += f"\nCode in the sandbox finds each one under {SANDBOX_DIR}/<its id>/<its name>."
    return note, images, shown
