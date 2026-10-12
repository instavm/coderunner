"""The voice service the daemon, channels, web page and file index use: picks a speech-to-text
engine, converts audio for it, sets engines up, and speaks replies."""

from __future__ import annotations

import asyncio
import pathlib
import shutil
import tempfile

from ..core.process import find_binary, run
from . import installed
from .apple import AppleSpeech
from .common import STT_HINT, Progress, Segment, VoiceUnavailable, joined, to_wav
from .qwen import QwenASR
from .tts import Speaker


class Voice:
    """Speech to text with the first engine that works: Apple speech, Qwen3-ASR, then tools found installed.

    `voice.stt` names one engine instead of auto. An engine that fails or lacks the language hands over
    to the next.
    """

    def __init__(self, config=None, tmp_root: str | None = None):
        self.config = config
        self.tmp_root = tmp_root
        self.speaker = Speaker(self._get, tmp_root)

    def _get(self, key: str, default=None):
        return self.config.get(f"voice.{key}", default) if self.config is not None else default

    @property
    def language(self) -> str:
        return str(self._get("language") or "auto")

    @property
    def timeout(self) -> float:
        return float(self._get("timeout") or 300)

    @property
    def apple(self) -> AppleSpeech:
        return AppleSpeech(self.language, self.timeout)

    @property
    def qwen(self) -> QwenASR:
        return QwenASR(self._get("asr_model"), self.timeout)

    def ffmpeg(self) -> str | None:
        return find_binary("ffmpeg", self._get("ffmpeg"))

    # Engine choice --------------------------------------------------------
    def engines(self) -> list:
        """Speech-to-text engines ready to use, in the order they are tried."""
        found: list = []
        if self.apple.possible():
            found.append(self.apple)
        if self.qwen.ready():
            found.append(self.qwen)
        found += installed.find(self._get)
        choice = self._get("stt") or "auto"
        return found if choice == "auto" else [engine for engine in found if engine.name == choice]

    def available(self) -> bool:
        return bool(self.ffmpeg() and self.engines())

    def status(self) -> dict:
        tts = self.speaker.engine()
        return {"ready": self.available(), "engines": [engine.name for engine in self.engines()],
                "language": self.language, "asr_model": self.qwen.model, "speaks_with": tts[0] if tts else None}

    # Transcription --------------------------------------------------------
    async def segments(self, path) -> list[Segment]:
        """What was said in an audio or video file, as timed segments (phrases, or pieces of about 30 s)."""
        converter, engines = self.ffmpeg(), self.engines()
        if not converter or not engines:
            raise VoiceUnavailable(STT_HINT)
        work = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-stt-", dir=self.tmp_root))  # mode 0700
        try:
            wav = work / "audio.wav"
            await to_wav(converter, path, wav)
            failure: Exception | None = None
            for engine in engines:
                try:
                    return await engine.segments(wav, work)
                except (RuntimeError, OSError) as exc:      # the next engine gets a chance
                    failure = exc
            raise failure if failure else VoiceUnavailable(STT_HINT)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    async def transcribe(self, path) -> str:
        return joined(await self.segments(path))

    async def timed(self, path) -> list[tuple[float, str]]:
        """(start second, text) pieces for the file index; nothing while speech to text is not set up."""
        if not self.available():
            return []
        return [(segment.start, segment.text) for segment in await self.segments(path)]

    # Setup ----------------------------------------------------------------
    async def setup(self, progress: Progress | None = None, engine: str | None = None, hub=None) -> dict:
        """Make speech to text ready: Apple speech where it serves the language, else Qwen3-ASR.

        engine: "apple" or "qwen3-asr" to set up only that one; Qwen3-ASR is otherwise downloaded
        only when Apple speech is missing or does not know the configured language.
        """
        say = progress or (lambda message: None)
        done: dict = {"ffmpeg": self.ffmpeg()}
        apple_ok = False
        if engine in (None, "apple"):
            try:
                await self.apple.prepare()
                apple_ok = True
                done["apple"] = "ready"
            except (RuntimeError, OSError) as exc:
                done["apple"] = str(exc) or "unavailable"
        if engine == "qwen3-asr" or (engine is None and not apple_ok):
            qwen = self.qwen
            await asyncio.get_running_loop().run_in_executor(None, lambda: qwen.install(say, hub))
            done["qwen3-asr"] = qwen.model
        if not done["ffmpeg"]:
            done["ffmpeg"] = await self._install_ffmpeg(say)
        done["ok"] = self.available()
        done["engines"] = [found.name for found in self.engines()]
        return done

    async def _install_ffmpeg(self, say: Progress) -> str | None:
        brew = find_binary("brew")
        if not brew:
            return None
        say("installing ffmpeg with Homebrew")
        code, _, err = await run([brew, "install", "ffmpeg"], 1800)
        if code != 0:
            say(f"Homebrew could not install ffmpeg: {err.strip()[-300:]}")
        return self.ffmpeg()

    # Text to speech -------------------------------------------------------
    def tts(self) -> tuple[str, str] | None:
        return self.speaker.engine()

    def can_speak(self) -> bool:
        return self.speaker.available()

    async def speak(self, text: str) -> str:
        return await self.speaker.speak(text)

    discard = staticmethod(Speaker.discard)
