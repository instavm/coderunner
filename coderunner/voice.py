"""Local speech: transcribe voice notes and, optionally, speak replies.

Audio is processed on this machine only. Speech-to-text uses the first engine
found: whisper.cpp's `whisper-cli` (with a ggml model), `parakeet-mlx`, then
`mlx_whisper` (the MLX engines download their model from Hugging Face once,
on first use). Text-to-speech uses macOS `say`, or the `piper` / `kokoro`
CLIs. ffmpeg does every format conversion.

Config keys (all optional) under `voice`: whisper_model, language, ffmpeg,
whisper_cli, parakeet_mlx, parakeet_model, mlx_whisper, mlx_whisper_model,
tts (say|piper|kokoro), say_voice, piper, piper_model, kokoro, kokoro_voice,
timeout. Binary keys take a path when the tool is not on PATH.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import shutil
import sys
import tempfile

from . import paths

# Daemons started by launchd get a minimal PATH, so also look where Homebrew, pipx and uv install tools.
EXTRA_PATHS = ["/opt/homebrew/bin", "/usr/local/bin", str(pathlib.Path.home() / ".local" / "bin")]

STT_HINT = ("Voice notes need a local speech-to-text engine and ffmpeg. Install ffmpeg (`brew install ffmpeg`) and "
            "one of: `brew install whisper-cpp` plus a ggml model saved as "
            "~/.coderunner/assistant/models/ggml-base.en.bin, `uv tool install parakeet-mlx`, "
            "or `uv tool install mlx-whisper`.")
TTS_HINT = ("Spoken replies need ffmpeg and a text-to-speech engine: macOS `say`, `piper` with "
            "voice.piper_model set, or `kokoro`.")


class VoiceUnavailable(RuntimeError):
    pass


def find_binary(name: str, override: str | None = None) -> str | None:
    """Resolve an executable by name (PATH plus common install dirs) or by explicit path."""
    candidate = os.path.expanduser(override or name)
    if os.sep in candidate:
        return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None
    return shutil.which(candidate, path=os.pathsep.join([os.environ.get("PATH", ""), *EXTRA_PATHS]))


async def run(argv: list[str], timeout: float, stdin: bytes | None = None) -> tuple[int, str, str]:
    """Run argv directly (never through a shell); the process is killed on timeout or cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def terminate(proc: asyncio.subprocess.Process | None, grace: float = 5) -> None:
    """Stop a long-running child: SIGTERM, then SIGKILL after `grace` seconds."""
    if proc is None or proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


class Voice:
    def __init__(self, config=None, tmp_root: str | None = None):
        self.config = config
        self.tmp_root = tmp_root

    def _get(self, key: str, default=None):
        return self.config.get(f"voice.{key}", default) if self.config is not None else default

    def _bin(self, name: str) -> str | None:
        return find_binary(name, self._get(name.replace("-", "_")))

    def _workdir(self, prefix: str) -> pathlib.Path:
        return pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=self.tmp_root))  # mode 0700

    @property
    def timeout(self) -> float:
        return float(self._get("timeout", 300))

    # Speech to text -------------------------------------------------------
    def whisper_model(self) -> str | None:
        configured = self._get("whisper_model")
        if configured:
            path = pathlib.Path(configured).expanduser()
            return str(path) if path.is_file() else None
        models = sorted((paths.assistant_dir() / "models").glob("ggml-*.bin"))
        return str(models[0]) if models else None

    def engine(self) -> tuple[str, str] | None:
        """The speech-to-text engine that would be used, as (name, executable)."""
        whisper = self._bin("whisper-cli")
        if whisper and self.whisper_model():
            return "whisper-cli", whisper
        for name in ("parakeet-mlx", "mlx_whisper"):
            binary = self._bin(name)
            if binary:
                return name, binary
        return None

    def available(self) -> bool:
        return bool(self._bin("ffmpeg") and self.engine())

    async def transcribe(self, path) -> str:
        ffmpeg, engine = self._bin("ffmpeg"), self.engine()
        if not ffmpeg or not engine:
            raise VoiceUnavailable(STT_HINT)
        name, binary = engine
        work = self._workdir("coderunner-stt-")
        try:
            wav = work / "audio.wav"
            await self._ffmpeg(ffmpeg, ["-i", str(path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
            language = self._get("language", "auto")
            if name == "whisper-cli":
                argv = [binary, "-m", self.whisper_model(), "-f", str(wav), "-l", language,
                        "-nt", "-np", "-otxt", "-of", str(work / "audio")]
            else:
                argv = [binary, str(wav), "--output-dir", str(work), "--output-format", "txt"]
                model = self._get("parakeet_model" if name == "parakeet-mlx" else "mlx_whisper_model")
                if model:
                    argv += ["--model", model]
                if name == "mlx_whisper" and language != "auto":
                    argv += ["--language", language]
            code, out, err = await run(argv, self.timeout)
            if code != 0:
                raise RuntimeError(f"{name} failed ({code}): {err.strip()[-500:]}")
            result = work / "audio.txt"
            text = result.read_text(errors="replace") if result.exists() else out
            return " ".join(text.split())
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # Text to speech -------------------------------------------------------
    def tts(self) -> tuple[str, str] | None:
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

    def can_speak(self) -> bool:
        return bool(self._bin("ffmpeg") and self.tts())

    async def speak(self, text: str) -> str:
        """Render text to an OGG/Opus file and return its path. Call discard(path) when done."""
        ffmpeg, tts = self._bin("ffmpeg"), self.tts()
        if not ffmpeg or not tts:
            raise VoiceUnavailable(TTS_HINT)
        name, binary = tts
        work = self._workdir("coderunner-tts-")
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
            code, _, err = await run(argv, self.timeout, stdin=stdin)
            if code != 0 or not raw.exists():
                raise RuntimeError(f"{name} failed ({code}): {err.strip()[-500:]}")
            out = work / "speech.ogg"
            await self._ffmpeg(ffmpeg, ["-i", str(raw), "-c:a", "libopus", "-b:a", "32k", "-ac", "1",
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

    async def _ffmpeg(self, ffmpeg: str, args: list[str]) -> None:
        # Only local files: a crafted "audio" playlist must not make ffmpeg fetch URLs.
        argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file", *args]
        code, _, err = await run(argv, 120)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {err.strip()[-500:]}")
