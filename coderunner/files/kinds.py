"""What kind of file a path is, and how embedding models want text phrased."""

from __future__ import annotations

import pathlib

TEXT = {".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".log", ".tex", ".bib", ".srt", ".vtt", ".html", ".htm", ".xml", ".svg", ".css", ".scss",
        ".py", ".ipynb", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h",
        ".cc", ".cpp", ".hpp", ".m", ".mm", ".cs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".scala", ".dart",
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".graphql", ".proto", ".vue", ".svelte", ".zig", ".nim"}
PROSE = {".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".tex", ".bib", ".srt", ".vtt"}
DOCS = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".pages"}         # converted to text by local tools
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tiff", ".tif"}
AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".flac", ".ogg", ".opus", ".caf"}
VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
KINDS = ("text", "doc", "image", "audio", "video")

# How each embedding model family wants queries and documents phrased, and the similarity above which a
# memory is clearly relevant to a task (measured on labelled examples; scales differ between models).
PROFILES = {
    "embeddinggemma": {"query": "task: search result | query: ", "doc": "title: {title} | text: ", "recall": 0.62},
    "": {"query": "", "doc": "", "recall": 0.5},
}


def embedding_profile(model: str) -> dict:
    family = next((name for name in PROFILES if name and name in (model or "").lower()), "")
    return PROFILES[family]


def kind_of(path: pathlib.Path) -> str | None:
    suffix = path.suffix.lower()
    for kind, suffixes in (("text", TEXT), ("doc", DOCS), ("image", IMAGES), ("audio", AUDIO), ("video", VIDEO)):
        if suffix in suffixes:
            return kind
    return None
