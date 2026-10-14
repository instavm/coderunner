"""Turning a file into pieces to index: its name, its text, its pictures and its sound.

Each piece has a place in the file (a line, a page, a time), words for the keyword index and what the
embedding model reads. Text is cut into overlapping passages, PDFs are read page by page (scanned pages
through on-device OCR on macOS), pictures are embedded as pictures and their text read with OCR, sound is
embedded in overlapping 30-second windows, and video as frames plus sound. Text made mostly of numbers
(data exports, logs of figures) means little to an embedding model and is matched by keywords only.

Every conversion runs a local tool with the path as an argument (never shell text): ffmpeg, the macOS
helper (Vision, PDFKit), textutil, or pdftotext where installed.
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
from .kinds import DOCS

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
        return self.doc_prefix.format(title=title) + text if wordy(text) else None

    def name(self, path: pathlib.Path, remote: bool = False) -> Piece:
        # Names are often the best description of a file ("boarding pass.pdf"); separators read as spaces.
        words = re.sub(r"[_\-.]+", " ", path.stem).strip() or path.name
        part = NAME_ONLY + ": the file is in iCloud, opening it downloads it" if remote else "name"
        return Piece(part, f"{path.name} {path.parent.name}", self.doc_prefix.format(title=path.name) + words,
                     "name")

    async def pieces(self, path: pathlib.Path, kind: str) -> list[Piece]:
        title = path.name
        found = [self.name(path)]
        if kind == "text":
            found += [Piece(f"from line {line}", text, self._text(title, text), "text", line)
                      for line, text in passages(await self.media.text(path))]
        elif kind == "doc":
            found += await self._document(path)
        elif kind == "image":
            found += await self._picture(path)
        elif kind in ("audio", "video"):
            found += await self._recording(path, kind)
        return found

    async def _document(self, path: pathlib.Path) -> list[Piece]:
        title = path.name
        if path.suffix.lower() != ".pdf":
            return [Piece(f"from line {line}", text, self._text(title, text), "text", line)
                    for line, text in passages(await self.media.text(path))]
        found = []
        for number, page in enumerate(await self.media.pages(path, read_scans=self.ocr_on), 1):
            for _, text in passages(page):
                found.append(Piece(f"page {number}", text, self._text(title, text), "text", number))
            if len(found) >= MAX_CHUNKS:
                break
        return found

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
