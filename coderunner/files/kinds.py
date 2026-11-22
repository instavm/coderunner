"""What kind of file a path is, how embedding models want text phrased, and the words a kind is described with."""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
import subprocess
import sys

TEXT = {".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".conf", ".log", ".tex", ".bib", ".srt", ".vtt", ".html", ".htm", ".xml", ".svg", ".css",
        ".scss", ".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".kts",
        ".swift", ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm", ".cs", ".rb", ".php", ".pl", ".lua", ".r", ".jl",
        ".scala", ".dart", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".graphql", ".proto", ".vue", ".svelte",
        ".zig", ".nim", ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".groovy", ".gradle", ".cmake", ".tf"}
PROSE = {".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".tex", ".bib", ".srt", ".vtt"}
DATA = {".csv", ".tsv", ".json", ".jsonl", ".log", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}
TABLES = {".csv", ".tsv"}                                          # read as tables, their rows with columns named
DOCS = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".pages", ".xlsx", ".pptx", ".numbers", ".key"}
OFFICE = {".xlsx", ".pptx", ".numbers", ".key", ".pages"}         # read from their parts, or the preview they carry
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".avif"}
AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".flac", ".ogg", ".opus", ".caf", ".m4b"}
VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
KINDS = ("text", "doc", "image", "audio", "video")
FOLDER = "folder"
MEDIA = ("image", "audio", "video")

# How each embedding model family wants queries and documents phrased, and the similarity above which a
# memory is clearly relevant to a task (measured on labelled examples; scales differ between models).
# Source code in a programming language: EmbeddingGemma 2 is trained to match it to questions asked with its
# code retrieval prompt (Google's model card), so code passages are searched with that one.
CODE = {".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".kts", ".swift",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm", ".cs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".scala",
        ".dart", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".graphql", ".proto", ".vue", ".svelte", ".zig",
        ".nim", ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".groovy", ".gradle", ".cmake", ".tf"}
CODE_NAMES = {"makefile", "dockerfile"}
MAX_CODE_BYTES = 350 * 1024          # as GitHub's code search: larger source files are skipped
# Lock files and generated or minified sources (a subset of GitHub Linguist's rules).
GENERATED = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock", "gemfile.lock", "poetry.lock",
             "pipfile.lock", "composer.lock", "go.sum", "podfile.lock", "package.resolved", "*.min.js", "*.min.css",
             "*.map", "*.pb.go", "*_pb2.py", "*.pb.swift", "*.g.dart", "*.generated.*", "*.designer.cs", "*.bundle.js")
PROFILES = {
    "embeddinggemma": {"query": "task: search result | query: ", "doc": "title: {title} | text: ", "recall": 0.62,
                       "code": "task: code retrieval | query: "},
    "": {"query": "", "doc": "", "recall": 0.5},
}
# The system's words for common file types, as Spotlight's Kind column has them, so "penguin video" can tell
# the video from the photo and "costa rica pics" the pictures.
DESCRIPTIONS = {
    ".pdf": "PDF document", ".docx": "Word document", ".doc": "Word document", ".rtf": "rich text document",
    ".odt": "OpenDocument text", ".pages": "Pages document", ".xlsx": "Excel spreadsheet", ".numbers": "Numbers spreadsheet",
    ".pptx": "PowerPoint presentation", ".key": "Keynote presentation", ".txt": "plain text", ".md": "Markdown text",
    ".markdown": "Markdown text", ".csv": "comma-separated values table", ".tsv": "tab-separated values table",
    ".json": "JSON data", ".log": "log", ".html": "web page", ".htm": "web page", ".png": "PNG image", ".jpg": "JPEG image",
    ".jpeg": "JPEG image", ".heic": "HEIF image", ".heif": "HEIF image", ".gif": "GIF image", ".webp": "WebP image",
    ".tiff": "TIFF image", ".tif": "TIFF image", ".bmp": "bitmap image", ".svg": "SVG image", ".mp4": "MPEG-4 movie, video",
    ".mov": "QuickTime movie, video", ".m4v": "MPEG-4 movie, video", ".mkv": "Matroska video", ".webm": "WebM video",
    ".avi": "AVI video", ".mp3": "MP3 audio recording", ".m4a": "MPEG-4 audio recording", ".wav": "waveform audio recording",
    ".aiff": "AIFF audio recording", ".aif": "AIFF audio recording", ".flac": "FLAC audio recording", ".caf": "Core Audio recording",
    ".ogg": "Ogg audio recording", ".opus": "Opus audio recording",
}
NOUNS = {"text": "text", "doc": "document", "image": "image", "audio": "audio recording", "video": "video",
         FOLDER: "folder"}
PLURALS = {"text": "text files", "doc": "documents", "image": "images", "audio": "audio", "video": "videos",
           "code": "code"}


def embedding_profile(model: str) -> dict:
    family = next((name for name in PROFILES if name and name in (model or "").lower()), "")
    return PROFILES[family]


# A small file that is a credential: an API key, a token, a private key. Told by what it holds, since such
# a file is often named anything (a key saved as "antkey" in Downloads was readable by the file tools).
KEY_FILE_BYTES = 8192
CREDENTIAL = re.compile(
    rb"sk-ant-[A-Za-z0-9_-]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    rb"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{35}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def holds_key(path: pathlib.Path) -> bool:
    """Whether this is a small file with a credential in it (never indexed, never read to a model)."""
    try:
        if not 0 < path.stat().st_size <= KEY_FILE_BYTES:
            return False
        with open(path, "rb") as handle:
            return CREDENTIAL.search(handle.read(KEY_FILE_BYTES)) is not None
    except OSError:
        return False


def kind_of(path: pathlib.Path) -> str | None:
    name = path.name.lower()
    if name in CODE_NAMES:
        return "text"
    suffix = path.suffix.lower()
    for kind, suffixes in (("text", TEXT), ("doc", DOCS), ("image", IMAGES), ("audio", AUDIO), ("video", VIDEO)):
        if suffix in suffixes:
            return kind
    return None


def is_code(path: str | pathlib.Path) -> bool:
    path = pathlib.Path(path)
    return path.suffix.lower() in CODE or path.name.lower() in CODE_NAMES


def is_generated(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatch(lower, pattern) for pattern in GENERATED)


def kind_description(path: pathlib.Path, kind: str | None = None) -> str:
    """"PDF document", "HEIF image", "QuickTime movie, video"; "source code" for code."""
    if is_code(path):
        return "source code"
    kind = kind or kind_of(path) or ""
    system = DESCRIPTIONS.get(path.suffix.lower(), "")
    noun = NOUNS.get(kind, "")
    if not system:
        return noun
    if not noun or noun.split()[0] in system.lower():
        return system
    return f"{system}, {noun}"


def holds_words(holds: list[str] | str) -> str:
    """"folder of images and documents" from the kinds a folder holds."""
    kinds = holds.split() if isinstance(holds, str) else list(holds)
    plural = [PLURALS[k] for k in ("doc", "text", "code", "image", "video", "audio") if k in kinds]
    return "folder of " + " and ".join(plural) if plural else "folder"


_screenshot_folder: str | None = None


def screenshot_folder() -> str:
    """Where macOS saves screenshots (Screenshot app > Options), the Desktop unless changed."""
    global _screenshot_folder
    if _screenshot_folder is None:
        folder = os.path.join(os.path.expanduser("~"), "Desktop")
        if sys.platform == "darwin":
            try:
                out = subprocess.run(["defaults", "read", "com.apple.screencapture", "location"], capture_output=True,
                                     text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    folder = os.path.expanduser(out.stdout.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
        _screenshot_folder = os.path.realpath(folder)
    return _screenshot_folder


def is_screenshot(path: pathlib.Path) -> bool:
    """macOS names screenshots "Screenshot 2026-10-16 at 19.21.07.png" ("Screen Shot" before macOS 13; other
    tools keep the word too); anything in a screenshot folder other than the Desktop counts as well."""
    lower = path.name.lower()
    if lower.startswith(("screenshot", "screen shot", "cleanshot", "screen recording")):
        return True
    folder = screenshot_folder()
    return folder != os.path.realpath(os.path.join(os.path.expanduser("~"), "Desktop")) \
        and os.path.realpath(str(path.parent)) == folder
