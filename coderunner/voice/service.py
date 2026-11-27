"""The voice service the daemon, channels, web page and file index use: picks a speech-to-text
engine, converts audio for it, sets engines up, and speaks replies."""

from __future__ import annotations

import asyncio
import datetime
import functools
import json
import pathlib
import shutil
import tempfile
import time

from ..core import paths
from ..core.config import Config
from ..core.process import find_binary, run
from . import installed
from .apple import AppleSpeech
from .common import STT_HINT, Progress, Segment, VoiceUnavailable, joined, to_wav
from .parakeet import ParakeetSpeech
from .qwen import QwenASR
from .tts import Speaker

# Voice setup times Parakeet and Apple speech on this sentence, spoken by `say`, on the owner's Mac.
CHECK_SENTENCE = "Please book a table for two at seven tonight, by the window."
LIGHT_ENGINES = ("parakeet-coreml", "apple")    # off the GPU: run beside a model there
NOTICEABLE_S = 0.5       # waiting this much longer for the words of a voice message is noticed


def _check_path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "speech.json"


@functools.cache
def _machine() -> dict:
    from ..models.bench import fingerprint
    from ..models.hardware import detect
    return fingerprint(detect())


def parakeet_first() -> bool:
    """Parakeet goes before Apple speech unless voice setup measured it noticeably slower on this machine."""
    try:
        saved = json.loads(_check_path().read_text())
    except (OSError, ValueError):
        return True
    return bool(saved.get("parakeet_first", True)) if saved.get("machine") == _machine() else True


class Voice:
    """Speech to text with the first engine that works: Parakeet on Core ML, Apple speech, Qwen3-ASR, then
    tools found installed.

    `voice.stt` names one engine instead of auto. An engine that fails or lacks the language hands over
    to the next.
    """

    def __init__(self, config=None, tmp_root: str | None = None):
        self.config = config
        self.tmp_root = tmp_root
        self.speaker = Speaker(self._get, tmp_root, config=config)

    def _get(self, key: str, default=None):
        return self.config.get(f"voice.{key}", default) if self.config is not None else default

    @property
    def language(self) -> str:
        return str(self._get("language") or "auto")

    @property
    def timeout(self) -> float:
        return float(self._get("timeout") or 300)

    @property
    def parakeet(self) -> ParakeetSpeech:
        return ParakeetSpeech(self.language, self.timeout)

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
        parakeet = self.parakeet
        ready = parakeet.ready()
        first = ready and parakeet_first()
        found: list = [parakeet] if first else []
        if self.apple.possible():
            found.append(self.apple)
        if ready and not first:
            found.append(parakeet)
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
    async def segments(self, path, light: bool = False) -> list[Segment]:
        """What was said in an audio or video file, as timed segments (phrases, or pieces of about 30 s).
        light: only engines that leave the GPU to a model already on it (Parakeet on the Neural Engine, Apple
        speech)."""
        converter, engines = self.ffmpeg(), self.engines()
        if light:
            engines = [engine for engine in engines if engine.name in LIGHT_ENGINES]
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

    async def transcribe(self, path, light: bool = False) -> str:
        return joined(await self.segments(path, light))

    async def timed(self, path) -> list[tuple[float, str]]:
        """(start second, text) pieces for the file index; nothing while speech to text is not set up."""
        if not self.available():
            return []
        return [(segment.start, segment.text) for segment in await self.segments(path)]

    # Setup ----------------------------------------------------------------
    async def setup(self, progress: Progress | None = None, engine: str | None = None, hub=None) -> dict:
        """Make speech to text ready: Parakeet where the Mac and the language allow, Apple speech where it
        serves the language, else Qwen3-ASR.

        engine: "parakeet-coreml", "apple" or "qwen3-asr" to set up only that one; Qwen3-ASR is otherwise
        downloaded only when neither Parakeet nor Apple speech serves the configured language.
        """
        say = progress or (lambda message: None)
        loop = asyncio.get_running_loop()
        done: dict = {"ffmpeg": self.ffmpeg()}
        parakeet, parakeet_ok, apple_ok = self.parakeet, False, False
        if engine == "parakeet-coreml" or (engine is None and parakeet.possible() and parakeet.serves()):
            config = self.config if self.config is not None else Config({})
            try:
                if not parakeet.possible():
                    raise VoiceUnavailable("Parakeet on Core ML needs a Mac with Apple silicon and macOS 15 or later")
                await loop.run_in_executor(None, lambda: parakeet.install(config, say, hub))
                parakeet_ok = True
                done["parakeet-coreml"] = "ready"
            except Exception as exc:            # noqa: BLE001 - Apple speech or Qwen3-ASR is set up instead
                done["parakeet-coreml"] = str(exc) or "unavailable"
        if engine in (None, "apple"):
            try:
                await self.apple.prepare()
                apple_ok = True
                done["apple"] = "ready"
            except (RuntimeError, OSError) as exc:
                done["apple"] = str(exc) or "unavailable"
        if engine == "qwen3-asr" or (engine is None and not apple_ok and not parakeet_ok):
            qwen = self.qwen
            await loop.run_in_executor(None, lambda: qwen.install(say, hub))
            done["qwen3-asr"] = qwen.model
        if parakeet_ok:
            done["speed_check"] = await self.compare(say)
        if not done["ffmpeg"]:
            done["ffmpeg"] = await self._install_ffmpeg(say)
        done["ok"] = self.available()
        done["engines"] = [found.name for found in self.engines()]
        return done

    async def compare(self, progress: Progress) -> dict:
        """Time Parakeet and Apple speech on one sentence spoken by `say` on this machine, and save which goes
        first: Parakeet, unless it was noticeably slower or heard nothing."""
        tool = find_binary("say")
        if not tool:
            return {}
        work = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-stt-", dir=self.tmp_root))
        try:
            clip = work / "check.wav"
            code, _, _ = await run([tool, "-o", str(clip), "--file-format=WAVE", "--data-format=LEI16@16000",
                                    CHECK_SENTENCE], 60)
            if code != 0 or not clip.exists():
                return {}
            progress("timing Parakeet and Apple speech on this Mac")
            result: dict = {"machine": _machine(), "date": datetime.date.today().isoformat()}
            for engine in (self.parakeet, self.apple):
                if engine.name == "apple" and not engine.possible():
                    continue
                times, text = [], ""
                try:
                    for _ in range(2):              # the second run is what later messages see
                        start = time.monotonic()
                        text = joined(await engine.segments(clip, work))
                        times.append(time.monotonic() - start)
                except (RuntimeError, OSError) as exc:
                    result[engine.name] = {"error": str(exc)[-200:]}
                    continue
                result[engine.name] = {"seconds": round(min(times), 2), "text": text}
            mine, apple = result.get("parakeet-coreml", {}), result.get("apple", {})
            result["parakeet_first"] = bool(mine.get("text")) and (
                "seconds" not in apple or mine["seconds"] <= apple["seconds"] + NOTICEABLE_S)
            _check_path().parent.mkdir(parents=True, exist_ok=True)
            _check_path().write_text(json.dumps(result, indent=1))
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)

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
