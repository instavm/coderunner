"""Turning a file into pieces to index: its name, its text, its pictures and its sound.

Each piece has a place in the file (a line, a page, a time), words for the keyword index and what the
embedding model reads. Text is cut into overlapping passages, PDFs are read page by page (scanned pages
through on-device OCR on macOS), pictures are embedded as pictures and their text read with OCR, sound is
embedded in overlapping 30-second windows, and video as frames plus sound. Text made mostly of numbers
(data exports, logs of figures) means little to an embedding model and is matched by keywords only.

A passage with nothing to say (a note that reads "Untitled", the residue of markup, a run of base64) keeps its
words for keyword search but gets no vector: in a large index such passages sit near every question. Tables
(CSV, TSV, spreadsheets) are read as rows with their columns named; Excel, PowerPoint and iWork files are read
from their parts or the preview they carry. Generated and minified code is left out, as code search engines do.

Every conversion runs a local tool with the path as an argument (never shell text): ffmpeg, the macOS
helper (Vision, PDFKit), textutil, unzip-free zip reading, or pdftotext where installed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import pathlib
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..core import native
from . import tables
from .kinds import DATA, DOCS, OFFICE, TABLES, holds_words, is_code, kind_description

HELPER_SOURCE = pathlib.Path(__file__).with_name("helper.swift")
NAME_ONLY = "name only"    # part label of a file indexed by its name, without reading it
TEXT_CHUNK = 1500          # characters per passage
TEXT_OVERLAP = 200
MAX_CHUNKS = 400           # passages per file
MAX_PAGES = 1000
WINDOW = 30                # seconds of sound per embedding
WINDOW_STEP = 25           # windows overlap by 5 s, so a phrase on a boundary is in one of them
MAX_WINDOWS = 40           # the first 20 minutes of a recording
MAX_FRAMES = 32            # a frame a second, thinned evenly to this many, as EmbeddingGemma 2 reads video natively;
                           # with one averaged "whole video" piece, 14 of 14 questions found their video against 13
                           # with 24 frames 10 s apart (and margins +0.108 against +0.089), checked against Google's model
SILENCE_DB = -60.0          # sound windows quieter than this (RMS) are not embedded
NATIVE_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}   # sent to the embedding model unchanged
NATIVE_IMAGE_BYTES = 20 * 1024 * 1024
WORDY = 0.35               # share of letters below which text is data, matched by keywords only
OCR_MIN_WORDS = 3
LONGEST_WORD = 64          # a run this long without a space is not a word: base64, a hash, a minified stretch
MIN_WORDS = 3              # a passage must hold this many words with a letter to be worth a vector
MAX_DATA_PASSAGES = 8      # data (exports, logs, configs): the first passages say what it is; the rest is rows alike
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SEPARATORS = re.compile(r"[_\-.]+")

# A transcriber returns (start second, text) pieces of what is said in a recording.
Transcriber = Callable[[pathlib.Path], Awaitable[list[tuple[float, str]]]]


def untimed(transcribe: Callable[[pathlib.Path], Awaitable[str]]) -> Transcriber:
    """A transcriber from a speech engine that returns plain text, without times."""
    async def transcriber(path: pathlib.Path) -> list[tuple[float, str]]:
        return [(0.0, await transcribe(path))]
    return transcriber


@dataclass
class Piece:
    part: str               # where in the file: "page 3", "from line 120", "at 1:30", "name"...
    text: str               # words for the keyword index, and the snippet shown
    embed: Any              # what the embedding model reads (text, {"image": b64}, {"audio": b64}); None: keywords only
    modality: str           # "name", "text", "image" or "audio": scores are compared within each
    loc: float | None = None   # page number or second, to open the file at the match
    average_of: str | None = None  # instead of being embedded: the mean of the file's pieces whose part starts so
    copied: tuple | None = None    # (embedding blob, code blob) taken from a twin file's piece instead of embedding


class Skipped(Exception):
    """A file left out on purpose (generated or minified code): still found by its name, not read again until
    it changes."""


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def audible(wav: bytes, floor_db: float = SILENCE_DB) -> bool:
    """Whether 16-bit WAV sound is louder than floor_db (RMS, dBFS). A video's empty track embedded as a sound
    window outranked every real one for 17% of questions in a check against Google's reference model."""
    import array
    import io
    import wave
    try:
        with wave.open(io.BytesIO(wav)) as sound:
            if sound.getsampwidth() != 2:
                return True
            samples = array.array("h", sound.readframes(sound.getnframes()))
    except (wave.Error, EOFError):
        return True
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return False
    rms = math.sqrt(math.sumprod(samples, samples) / len(samples))
    return rms > 32768 * 10 ** (floor_db / 20)


def wordy(text: str) -> bool:
    """Whether text is language rather than figures: an embedding helps only for language."""
    visible = [c for c in text if not c.isspace()]
    return bool(visible) and sum(c.isalpha() for c in visible) / len(visible) >= WORDY


def has_words(passage: str) -> bool:
    """Whether a passage holds MIN_WORDS words with a letter in them, runs over LONGEST_WORD characters not counted."""
    found = 0
    for word in passage.split():
        if len(word) <= LONGEST_WORD and any(c.isalpha() for c in word):
            found += 1
            if found >= MIN_WORDS:
                return True
    return False


def readable(passage: str) -> str | None:
    """What the model reads of a passage, or None when nothing in it is worth a vector. Runs over LONGEST_WORD
    characters are left out (no one asks for them by meaning, and a passage of them sits near every question),
    and what remains must hold three words with a letter in them. A note that says only "# Untitled", the
    arrows left of an HTML comment, a table of numbers: kept for keyword search, not embedded. In an index of
    200,000 passages such near-empty ones were the closest neighbour of almost any question."""
    if not has_words(passage):
        return None
    if any(len(word) > LONGEST_WORD for word in passage.split()):
        return re.sub(r"\S{%d,}" % (LONGEST_WORD + 1), "", passage)
    return passage


def readable_name(name: str) -> str | None:
    """A name's words, without UUIDs and runs of hex digits mixing letters and numbers (hashes):
    "Export-9f2c1a7e-3b61-…" reads "Export". None when nothing is left. Folders named by a hash alone (an
    editor's cache) came within the top ten for one question in six, as near-empty passages did."""
    words = [w for w in SEPARATORS.sub(" ", UUID.sub(" ", name)).split()
             if not (len(w) >= 4 and all(c in "0123456789abcdefABCDEF" for c in w)
                     and any(c.isdigit() for c in w) and any(c.isalpha() for c in w))]
    return " ".join(words) if words else None


def not_worth_indexing(text: str) -> str | None:
    """Why a source file is left out, as code search engines do: generated (a "Code generated … DO NOT EDIT" or
    "@generated" header), minified (a line over 4,096 bytes), or nearly empty."""
    head = text[:1024].lower()
    if "@generated" in head or ("code generated" in head and "do not edit" in head) or "autogenerated" in head \
            or "auto-generated" in head:
        return "generated code"
    visible = 0
    for line in text.split("\n"):
        if len(line.encode("utf-8", "replace")) > 4096:
            return "minified (a line over 4,096 bytes)"
        visible += sum(1 for c in line if c > " ")
    return "nearly empty" if visible < 64 else None


def passages(text: str) -> list[tuple[int, str]]:
    """(first line, passage) of overlapping slices of a text."""
    found = []
    for start in range(0, max(len(text), 1), TEXT_CHUNK - TEXT_OVERLAP):
        piece = text[start:start + TEXT_CHUNK]
        if piece.strip():
            found.append((text.count("\n", 0, start) + 1, piece))
        if len(found) >= MAX_CHUNKS:
            break
    return found


class Media:
    """Local conversions: ffmpeg, macOS tools and the native helper."""

    def __init__(self):
        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        self.sips = shutil.which("sips")
        self.textutil = shutil.which("textutil")
        self.pdftotext = shutil.which("pdftotext")
        self._helper: str | None | bool = False      # False: not looked for yet
        self._ocr: asyncio.subprocess.Process | None = None
        self._ocr_lock = asyncio.Lock()

    async def helper(self) -> str | None:
        if self._helper is False:
            self._helper = await native.ensure(HELPER_SOURCE, "files-helper")
        return self._helper or None

    async def _run(self, *argv: str, timeout: float = 120) -> bytes:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{pathlib.Path(argv[0]).name} timed out")
        if proc.returncode:
            raise RuntimeError(f"{pathlib.Path(argv[0]).name} failed: {err.decode(errors='replace')[-300:]}")
        return out

    # ---- text ----------------------------------------------------------------

    async def pages(self, path: pathlib.Path, read_scans: bool = True) -> list[str]:
        """The text of each page of a PDF; scanned pages are read with OCR where the macOS helper runs."""
        helper = await self.helper()
        if helper:
            out = json.loads(await self._run(helper, "pdf", str(path), str(MAX_PAGES), "1" if read_scans else "0",
                                             timeout=600) or b"{}")
            if "error" in out:
                raise RuntimeError(out["error"])
            return list(out.get("pages") or [])
        if self.pdftotext:
            text = (await self._run(self.pdftotext, "-layout", "-q", str(path), "-")).decode(errors="replace")
            return text.split("\f")[:MAX_PAGES]
        raise RuntimeError("PDFs need macOS (with the Xcode command line tools) or poppler's pdftotext")

    async def text(self, path: pathlib.Path) -> str:
        """The whole text of a text file or document, as the assistant reads it."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = await self.pages(path)
            return "\n".join(f"--- page {n} ---\n{page}" for n, page in enumerate(pages, 1))
        if suffix in DOCS:
            if not self.textutil:
                raise RuntimeError("documents need macOS textutil")
            return (await self._run(self.textutil, "-convert", "txt", "-stdout", str(path))).decode(errors="replace")
        return path.read_text(errors="replace")

    async def read_text_in(self, path: pathlib.Path) -> str:
        """Words in a picture (on-device OCR), or "" where the helper is unavailable."""
        helper = await self.helper()
        if not helper:
            return ""
        async with self._ocr_lock:
            if self._ocr is None or self._ocr.returncode is not None:
                self._ocr = await asyncio.create_subprocess_exec(
                    helper, "ocr", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL)
            try:
                self._ocr.stdin.write(str(path).replace("\n", " ").encode() + b"\n")
                await self._ocr.stdin.drain()
                line = await asyncio.wait_for(self._ocr.stdout.readline(), 60)
                return json.loads(line or b"{}").get("text", "")
            except (asyncio.TimeoutError, ValueError, OSError, BrokenPipeError):
                self._ocr.kill()
                self._ocr = None
                return ""

    async def close(self) -> None:
        if self._ocr is not None and self._ocr.returncode is None:
            self._ocr.stdin.close()
            try:
                await asyncio.wait_for(self._ocr.wait(), 5)
            except asyncio.TimeoutError:
                self._ocr.kill()
        self._ocr = None

    # ---- pictures and sound --------------------------------------------------

    async def image(self, path: pathlib.Path, at: float | None = None, size: int = 1024) -> bytes:
        """A JPEG no larger than size pixels: the image itself, or a video frame at `at` seconds."""
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "frame.jpg"
            if self.ffmpeg and (at is not None or path.suffix.lower() not in (".heic", ".heif")):
                argv = [self.ffmpeg, "-loglevel", "error", "-y"]
                if at is not None:
                    argv += ["-ss", f"{at:.2f}"]
                # -q:v 2: ffmpeg's default JPEG quality moved a frame's embedding by as much as 0.07 in cosine
                # (0.928 to the original at worst); at 2 it stays within 0.014.
                argv += ["-i", str(path), "-frames:v", "1", "-vf",
                         f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease",
                         "-q:v", "2", str(out)]
                await self._run(*argv)
            elif self.sips:
                await self._run(self.sips, "-s", "format", "jpeg", "-s", "formatOptions", "best", "-Z", str(size),
                                str(path), "--out", str(out))
            else:
                raise RuntimeError("install ffmpeg to read images and video")
            return out.read_bytes()

    async def duration(self, path: pathlib.Path) -> float:
        if not self.ffprobe:
            return 0.0
        out = await self._run(self.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path))
        try:
            return float(out.decode().strip() or 0)
        except ValueError:
            return 0.0

    async def frames(self, path: pathlib.Path, count: int, length: float, size: int = 1024) -> list[tuple[float, bytes]]:
        """`count` frames spread evenly over a video, as JPEGs no larger than size pixels, in one ffmpeg run."""
        if count <= 1 or not length:
            return [(length / 2, await self.image(path, at=length / 2, size=size))]
        with tempfile.TemporaryDirectory() as tmp:
            step = length / count
            await self._run(self.ffmpeg, "-loglevel", "error", "-y", "-i", str(path), "-vf",
                            f"fps=1/{step:.4f}:start_time={step / 2:.4f}:round=down,"     # default rounding drops the last
                            f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease",
                            "-frames:v", str(count), "-q:v", "2", str(pathlib.Path(tmp) / "frame%03d.jpg"),
                            timeout=600)
            files = sorted(pathlib.Path(tmp).glob("frame*.jpg"))
            return [(step * (i + 0.5), f.read_bytes()) for i, f in enumerate(files)]

    async def streams(self, path: pathlib.Path) -> set[str]:
        """What a recording holds ("video", "audio"): screen recordings have no sound, some .mp4 no picture."""
        if not self.ffprobe:
            return {"video", "audio"}
        out = await self._run(self.ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(path))
        return {line.strip() for line in out.decode().split() if line.strip()}

    async def sound(self, path: pathlib.Path, start: float, seconds: float) -> bytes:
        """16 kHz mono WAV of part of an audio or video file."""
        if not self.ffmpeg:
            raise RuntimeError("install ffmpeg to read audio and video")
        return await self._run(self.ffmpeg, "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}",
                               "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-")


class Extractor:
    """The pieces of one file, given what the embedding model accepts (text, images, audio)."""

    def __init__(self, media: Media, *, doc_prefix: str = "", capabilities: set[str] | None = None,
                 media_on: bool = True, ocr_on: bool = True, transcriber: Transcriber | None = None):
        self.media = media
        self.doc_prefix = doc_prefix
        self.capabilities = capabilities or set()
        self.media_on = media_on
        self.ocr_on = ocr_on
        self.transcriber = transcriber

    def _text(self, title: str, text: str) -> Any:
        if not wordy(text):
            return None
        worth = readable(text)
        return self.doc_prefix.format(title=title) + worth if worth is not None else None

    def name(self, path: pathlib.Path, remote: bool = False, kind: str | None = None) -> Piece:
        """The piece that finds a file by its name. Names are often the best description of a file ("boarding
        pass.pdf"); separators read as spaces. The file's kind goes with it, as Spotlight's Kind does
        ("QuickTime movie", "HEIF image"), so "penguin video" can tell the video from the photo."""
        words = SEPARATORS.sub(" ", path.stem).strip() or path.name
        description = kind_description(path, kind)
        part = NAME_ONLY + ": the file is in iCloud, opening it downloads it" if remote else "name"
        return Piece(part, f"{path.name} {path.parent.name} {description}",
                     self.doc_prefix.format(title=path.name) + f"{words} ({description})", "name")

    def folder_name(self, path: pathlib.Path, holds) -> Piece:
        """A folder's name, as a file's, with the folder it is in for keyword search; "folder" goes with it, so
        "tax folder" finds the folder rather than the files in it. The model reads the name's words without
        hashes and ids and the kinds of file in the folder: "Garden (folder of images)" tells the photos from
        "Garden (folder of videos)", and answers "costa rica pics". A name with no words left keeps them for
        keyword search and gets no vector."""
        words = readable_name(path.name)
        embed = self.doc_prefix.format(title=words) + f"{words} ({holds_words(holds)})" if words else None
        return Piece("name", f"{path.name} {path.parent.name} folder", embed, "name")

    @staticmethod
    def own_name(path: pathlib.Path, kind: str | None = None) -> str:
        """A name piece's words for the ranking by names: the file's or folder's own name and kind, without the
        folder it is in."""
        if kind == "folder":
            return f"{path.name} folder"
        return f"{path.name} {kind_description(path, kind)}"

    async def pieces(self, path: pathlib.Path, kind: str, repo: str | None = None) -> list[Piece]:
        title = path.name
        found = [self.name(path, kind=kind)]
        if kind == "text":
            found += await self._textfile(path, repo)
        elif kind == "doc":
            found += await self._document(path)
        elif kind == "image":
            found += await self._picture(path)
        elif kind in ("audio", "video"):
            found += await self._recording(path, kind)
        return found

    async def _textfile(self, path: pathlib.Path, repo: str | None) -> list[Piece]:
        suffix = path.suffix.lower()
        text = await self.media.text(path)
        title = path.name
        if suffix in TABLES:
            # In a worker thread: a large sheet took 32 s, and on the event loop that froze every task and the
            # app's live view with it.
            def describe(raw: str) -> str:
                table = tables.Table.parse(raw, "\t" if suffix == ".tsv" else ",")
                return table.described(path.name) if table is not None else raw
            text = await asyncio.to_thread(describe, text)
        if is_code(path):
            reason = not_worth_indexing(text)
            if reason:
                raise Skipped(reason)
            # Code passages are titled with the file's path inside its repository ("crates/sync/src/download.rs"):
            # the folders say what a file is part of, which its bare name does not.
            if repo and str(path).startswith(repo + "/"):
                title = str(path)[len(repo) + 1:]
        limit = MAX_DATA_PASSAGES if suffix in DATA else MAX_CHUNKS
        return [Piece(f"from line {line}", chunk, self._text(title, chunk), "text", line)
                for line, chunk in passages(text)[:limit]]

    async def _document(self, path: pathlib.Path) -> list[Piece]:
        title = path.name
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return await self._pdf(path)
        if suffix == ".xlsx":
            found = []
            for sheet, table in await asyncio.to_thread(tables.sheets, path):
                described = table.described(f"{title}, sheet {sheet}")
                found += [Piece(f"sheet {sheet}", text, self._text(title, text), "text", None)
                          for _, text in passages(described)[:MAX_DATA_PASSAGES - len(found)]]
                if len(found) >= MAX_DATA_PASSAGES:
                    break
            return found
        if suffix == ".pptx":
            found = []
            for number, words in await asyncio.to_thread(tables.slides, path):
                found += [Piece(f"slide {number}", text, self._text(title, text), "text", number)
                          for _, text in passages(words)]
                if len(found) >= MAX_CHUNKS:
                    break
            return found
        if suffix in OFFICE:
            return await self._iwork(path)
        return [Piece(f"from line {line}", text, self._text(title, text), "text", line)
                for line, text in passages(await self.media.text(path))]

    async def _pdf(self, path: pathlib.Path) -> list[Piece]:
        title = path.name
        found = []
        for number, page in enumerate(await self.media.pages(path, read_scans=self.ocr_on), 1):
            for _, text in passages(page):
                found.append(Piece(f"page {number}", text, self._text(title, text), "text", number))
            if len(found) >= MAX_CHUNKS:
                break
        return found

    async def _iwork(self, path: pathlib.Path) -> list[Piece]:
        """An iWork file through the preview it carries: a PDF of the whole document (files from before 2013),
        or a picture of its first page or slide, read by OCR."""
        preview = await asyncio.to_thread(tables.iwork_preview, path)
        if preview is None:
            return []
        kind, data = preview
        with tempfile.TemporaryDirectory() as tmp:
            copy = pathlib.Path(tmp) / (path.name + (".pdf" if kind == "pdf" else ".jpg"))
            copy.write_bytes(data)
            if kind == "pdf":
                pages = await self.media.pages(copy, read_scans=self.ocr_on)
                return [Piece(f"page {n}", text, self._text(path.name, text), "text", n)
                        for n, page in enumerate(pages, 1) for _, text in passages(page)][:MAX_CHUNKS]
            if not self.ocr_on:
                return []
            words = (await self.media.read_text_in(copy)).strip()
        if len(words.split()) < OCR_MIN_WORDS:
            return []
        return [Piece("first page", text, self._text(path.name, text), "text", 1) for _, text in passages(words)]

    async def _picture(self, path: pathlib.Path) -> list[Piece]:
        found = []
        if self.media_on and "vision" in self.capabilities:
            # The model's own image processing is used as is; formats it cannot decode (HEIC, TIFF, BMP, GIF)
            # and very large files are converted to JPEG first.
            if path.suffix.lower() in NATIVE_IMAGES and path.stat().st_size <= NATIVE_IMAGE_BYTES:
                raw = path.read_bytes()
            else:
                raw = await self.media.image(path, size=2048)
            found.append(Piece("image", path.name, {"image": base64.b64encode(raw).decode()}, "image"))
        if self.ocr_on:
            words = (await self.media.read_text_in(path)).strip()
            if len(words.split()) >= OCR_MIN_WORDS:
                found.append(Piece("text in the image", words, self._text(path.name, words[:TEXT_CHUNK]), "text"))
        return found

    async def _recording(self, path: pathlib.Path, kind: str) -> list[Piece]:
        found: list[Piece] = []
        if not (self.media_on and self.media.ffmpeg):
            return found
        length = await self.media.duration(path)
        streams = await self.media.streams(path)
        if kind == "video" and "vision" in self.capabilities and "video" in streams:
            count = min(MAX_FRAMES, max(1, int(length))) if length else 1
            for at, frame in await self.media.frames(path, count, length):
                found.append(Piece(f"frame at {clock(at)}", path.name, {"image": base64.b64encode(frame).decode()},
                                   "image", at))
            if sum(piece.part.startswith("frame at") for piece in found) > 1:
                found.append(Piece("the whole video", path.name, None, "image", average_of="frame at"))
        if "audio" in self.capabilities and "audio" in streams:
            windows = max(1, min(MAX_WINDOWS, math.ceil(max(length - WINDOW, 0) / WINDOW_STEP) + 1)) if length else 1
            for i in range(windows):
                start = i * WINDOW_STEP
                sound = await self.media.sound(path, start, WINDOW)
                if len(sound) > 1000 and audible(sound):      # a silent track outranked real sound
                    found.append(Piece(f"sound at {clock(start)}", path.name,
                                       {"audio": base64.b64encode(sound).decode()}, "audio", start))
        if self.transcriber is not None:
            # What is said becomes text: searchable by its words, and by meaning as text.
            for start, words in await self.transcriber(path):
                if words.strip():
                    found.append(Piece(f"said at {clock(start)}", words, self._text(path.name, words), "text", start))
        return found
