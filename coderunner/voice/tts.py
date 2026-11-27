"""Spoken replies: LFM3-Omni (/tts), macOS `say`, or the `piper` / `kokoro` CLIs, rendered to OGG/Opus by ffmpeg."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

from ..core.process import find_binary, run
from .common import TTS_HINT, VoiceUnavailable, ffmpeg
from .omni import clean_for_speech, speech_lang, synthesize


class Speaker:
    def __init__(self, get, tmp_root: str | None = None, config=None):
        self._get = get               # voice.<key> lookup
        self.tmp_root = tmp_root
        self.config = config

    def _bin(self, name: str) -> str | None:
        return find_binary(name, self._get(name))

    def _lfm_ready(self) -> bool:
        model = self._get("omni_model")
        if not model:
            return False
        try:
            from ..models.runtimes import VoiceRuntime
            return VoiceRuntime(self.config).ready() if self.config is not None else False
        except Exception:
            return False

    def engine(self) -> tuple[str, str] | None:
        """The text-to-speech engine, as (name, executable-or-label)."""
        preferred = self._get("tts")
        options = [
            ("lfm", "lfm" if self._lfm_ready() else None),
            ("say", self._bin("say") if sys.platform == "darwin" else None),
            ("piper", self._bin("piper") if self._get("piper_model") else None),
            ("kokoro", self._bin("kokoro")),
        ]
        for name, binary in options:
            if binary and preferred in (None, "", name, "auto"):
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
            cleaned = clean_for_speech(text)
            if name == "lfm":
                from ..models.runtimes import VoiceRuntime
                runtime = VoiceRuntime(self.config)
                voice = self._get("voice") or "female"
                lang = speech_lang(self.config) if self.config is not None else "en"
                pcm, rate = await synthesize(runtime, cleaned, voice=voice, lang=lang)
                raw = work / "speech.pcm"
                raw.write_bytes(pcm)
                wav = work / "speech.wav"
                await ffmpeg(converter, ["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", str(raw), str(wav)])
                raw.unlink(missing_ok=True)
                source_path = wav
            else:
                source = work / "text.txt"
                source.write_text(cleaned)
                stdin = None
                if name == "say":
                    raw = work / "speech.aiff"
                    argv = [binary, "-o", str(raw), "-f", str(source)]
                    if self._get("say_voice"):
                        argv += ["-v", self._get("say_voice")]
                elif name == "piper":
                    raw = work / "speech.wav"
                    argv, stdin = [binary, "--model", self._get("piper_model"), "--output_file", str(raw)], cleaned.encode()
                else:
                    raw = work / "speech.wav"
                    argv = [binary, "-i", str(source), "-o", str(raw)]
                    if self._get("kokoro_voice"):
                        argv += ["-m", self._get("kokoro_voice")]
                code, _, err = await run(argv, float(self._get("timeout") or 300), stdin=stdin)
                if code != 0 or not raw.exists():
                    raise RuntimeError(f"{name} failed ({code}): {err.strip()[-500:]}")
                source.unlink(missing_ok=True)
                source_path = raw
            out = work / "speech.ogg"
            await ffmpeg(converter, ["-i", str(source_path), "-c:a", "libopus", "-b:a", "32k", "-ac", "1",
                                     "-application", "voip", str(out)])
            source_path.unlink(missing_ok=True)
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
