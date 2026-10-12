"""Spoken replies: macOS `say`, or the `piper` / `kokoro` CLIs, rendered to OGG/Opus by ffmpeg."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

from ..core.process import find_binary, run
from .common import TTS_HINT, VoiceUnavailable, ffmpeg


class Speaker:
    def __init__(self, get, tmp_root: str | None = None):
        self._get = get               # voice.<key> lookup
        self.tmp_root = tmp_root

    def _bin(self, name: str) -> str | None:
        return find_binary(name, self._get(name))

    def engine(self) -> tuple[str, str] | None:
        """The text-to-speech engine, as (name, executable)."""
        preferred = self._get("tts")
        options = [
            ("say", self._bin("say") if sys.platform == "darwin" else None),
            ("piper", self._bin("piper") if self._get("piper_model") else None),
            ("kokoro", self._bin("kokoro")),
        ]
        for name, binary in options:
            if binary and preferred in (None, "", name):
                return name, binary
        return None

    def available(self) -> bool:
        return bool(self._bin("ffmpeg") and self.engine())

    async def speak(self, text: str) -> str:
        """Render text to an OGG/Opus file and return its path. Call discard(path) when done."""
        converter, tts = self._bin("ffmpeg"), self.engine()
        if not converter or not tts:
            raise VoiceUnavailable(TTS_HINT)
        name, binary = tts
        work = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-tts-", dir=self.tmp_root))  # mode 0700
        try:
            source = work / "text.txt"
            source.write_text(text)
            stdin = None
            if name == "say":
                raw = work / "speech.aiff"
                argv = [binary, "-o", str(raw), "-f", str(source)]  # a file, so text can't be read as flags
                if self._get("say_voice"):
                    argv += ["-v", self._get("say_voice")]
            elif name == "piper":
                raw = work / "speech.wav"
                argv, stdin = [binary, "--model", self._get("piper_model"), "--output_file", str(raw)], text.encode()
            else:
                raw = work / "speech.wav"
                argv = [binary, "-i", str(source), "-o", str(raw)]
                if self._get("kokoro_voice"):
                    argv += ["-m", self._get("kokoro_voice")]
            code, _, err = await run(argv, float(self._get("timeout") or 300), stdin=stdin)
            if code != 0 or not raw.exists():
                raise RuntimeError(f"{name} failed ({code}): {err.strip()[-500:]}")
            out = work / "speech.ogg"
            await ffmpeg(converter, ["-i", str(raw), "-c:a", "libopus", "-b:a", "32k", "-ac", "1",
                                     "-application", "voip", str(out)])
            source.unlink()
            raw.unlink()
            return str(out)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise

    @staticmethod
    def discard(path) -> None:
        """Remove a file returned by speak() together with its private directory."""
        parent = pathlib.Path(path).parent
        if parent.name.startswith("coderunner-tts-"):
            shutil.rmtree(parent, ignore_errors=True)
