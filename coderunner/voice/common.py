"""What the speech engines share: errors, transcript segments and audio conversion."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import wave
from typing import Callable

from ..core.process import run

STT_HINT = "To set up speech to text, run `coderunner voice setup` on the assistant's computer."
TTS_HINT = ("Spoken replies need ffmpeg and a text-to-speech engine: macOS `say`, `piper` with "
            "voice.piper_model set, or `kokoro`.")

Progress = Callable[[str], None]


class VoiceUnavailable(RuntimeError):
    pass


class UnsupportedLanguage(RuntimeError):
    pass


@dataclasses.dataclass
class Segment:
    """Words spoken between `start` and `end` seconds of a recording."""

    start: float
    end: float
    text: str


def phrases(output: str) -> list[Segment]:
    """Segments from a Swift helper's output: one JSON line per phrase, {"start", "end", "text"}."""
    found = []
    for line in output.splitlines():
        if line.strip():
            item = json.loads(line)
            found.append(Segment(round(float(item["start"]), 2), round(float(item["end"]), 2),
                                 " ".join(item["text"].split())))
    return found


def joined(segments: list[Segment]) -> str:
    return " ".join(" ".join(segment.text for segment in segments).split())


def seconds(wav: pathlib.Path) -> float:
    with wave.open(str(wav)) as audio:
        return audio.getnframes() / audio.getframerate()


async def ffmpeg(binary: str, args: list[str]) -> None:
    # Only local files: a crafted "audio" playlist must not make ffmpeg fetch URLs.
    argv = [binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file", *args]
    code, _, err = await run(argv, 120)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {err.strip()[-500:]}")


async def to_wav(binary: str, source, wav: pathlib.Path) -> None:
    """16 kHz mono 16-bit WAV, what every engine reads."""
    await ffmpeg(binary, ["-i", str(source), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
