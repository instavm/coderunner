"""Apple's on-device speech model (SpeechAnalyzer, macOS 26 and later), through a Swift helper.

It transcribes one set language (the Mac's own unless configured) from those Apple supports; the
system downloads a language's model the first time it is needed. Results come per phrase with times.
The helper is built on first use by coderunner.core.native.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

from ..core import native
from ..core.process import run
from .common import Segment, UnsupportedLanguage, VoiceUnavailable

SOURCE = pathlib.Path(__file__).with_name("apple_speech.swift")
NAME = "coderunner-speech"
MIN_MACOS = 26
UNSUPPORTED = 3          # the helper's exit code when this Mac or language has no speech model


def macos_major() -> int:
    """The macOS major version (26 for macOS 26), or 0 on other systems."""
    if sys.platform != "darwin":
        return 0
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        return 0


class AppleSpeech:
    name = "apple"

    def __init__(self, language: str = "auto", timeout: float = 300):
        self.language = language
        self.timeout = timeout

    def helper(self) -> pathlib.Path:
        return native.helper_path(SOURCE, NAME)

    def possible(self) -> bool:
        return macos_major() >= MIN_MACOS and native.can_build(SOURCE, NAME)

    async def build(self) -> str:
        if macos_major() < MIN_MACOS:
            raise VoiceUnavailable(f"Apple speech needs macOS {MIN_MACOS}")
        helper = await native.ensure(SOURCE, NAME)
        if helper is None:
            raise VoiceUnavailable("the Apple speech helper could not be built (Xcode command line tools: "
                                   "xcode-select --install)")
        return helper

    async def _call(self, *args: str, timeout: float) -> str:
        code, out, err = await run([await self.build(), *args], timeout)
        if code == UNSUPPORTED:
            raise UnsupportedLanguage(err.strip())
        if code != 0:
            raise RuntimeError(f"apple speech failed ({code}): {err.strip()[-500:]}")
        return out

    async def prepare(self) -> None:
        """Make sure the language's model is installed (the system downloads it if needed)."""
        await self._call("prepare", self.language, timeout=1800)

    async def segments(self, wav: pathlib.Path, work: pathlib.Path) -> list[Segment]:
        out = await self._call("transcribe", str(wav), self.language, timeout=self.timeout)
        found = []
        for line in out.splitlines():
            if line.strip():
                item = json.loads(line)
                found.append(Segment(float(item["start"]), float(item["end"]), " ".join(item["text"].split())))
        return found
