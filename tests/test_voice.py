import asyncio
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

from coderunner import voice as voice_module
from coderunner.config import Config
from coderunner.voice import Voice, VoiceUnavailable, find_binary, run

FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None)
needs_ffmpeg = unittest.skipUnless(FFMPEG, "ffmpeg is not installed")

# Fake engines report the format they were given, so the test can check ffmpeg's conversion.
FAKE_WHISPER = r'''
import pathlib, sys, wave
args = sys.argv[1:]
wav, prefix = args[args.index("-f") + 1], args[args.index("-of") + 1]
assert pathlib.Path(args[args.index("-m") + 1]).is_file()
with wave.open(wav) as w:
    pathlib.Path(prefix + ".txt").write_text(f" rate={w.getframerate()}\n channels={w.getnchannels()} \n")
'''
FAKE_DIR_ENGINE = r'''
import pathlib, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index("--output-dir") + 1]) / (pathlib.Path(args[0]).stem + ".txt")
out.write_text("parakeet heard you")
'''
FAKE_FAILING = "import sys\nsys.stderr.write('model exploded')\nsys.exit(3)\n"
FAKE_TTS = r'''
import json, math, pathlib, struct, sys, wave
args = sys.argv[1:]
source = args[args.index("-f" if "-f" in args else "-i") + 1]
log = {"argv": args, "text": pathlib.Path(source).read_text()}
pathlib.Path(__file__).with_suffix(".log").write_text(json.dumps(log))
out = args[args.index("-o") + 1]
with wave.open(out, "wb") as w:  # WAV bytes even for say's .aiff name; ffmpeg probes the content
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
    w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 8))) for i in range(11025)))
'''


def write_tone(path, rate=44100, channels=2, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        samples = (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(rate * seconds)))
        w.writeframes(b"".join(struct.pack("<h", sample) * channels for sample in samples))


class VoiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.work = self.dir / "work"
        self.work.mkdir()
        self.models = self.dir / "home" / "assistant" / "models"
        self.models.mkdir(parents=True)
        # Only our fake tools are visible, whatever is installed on this machine.
        for patcher in (mock.patch.dict(os.environ, {"PATH": str(self.bin), "CODERUNNER_HOME": str(self.dir / "home")}),
                        mock.patch.object(voice_module, "EXTRA_PATHS", [])):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def tool(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)
        return str(path)

    def voice(self, **settings):
        return Voice(Config({"voice": {"ffmpeg": FFMPEG or "", **settings}}), tmp_root=str(self.work))


class EngineDetectionTests(VoiceTestCase):
    async def test_nothing_installed(self):
        voice = self.voice()
        self.assertIsNone(voice.engine())
        self.assertFalse(voice.available())
        with self.assertRaisesRegex(VoiceUnavailable, "whisper-cpp"):
            await voice.transcribe(self.dir / "note.ogg")

    def test_engine_preference_order(self):
        self.tool("mlx_whisper", FAKE_DIR_ENGINE)
        self.assertEqual(self.voice().engine()[0], "mlx_whisper")
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        self.assertEqual(self.voice().engine()[0], "parakeet-mlx")
        self.tool("whisper-cli", FAKE_WHISPER)
        self.assertEqual(self.voice().engine()[0], "parakeet-mlx")  # whisper-cli is useless without a model
        (self.models / "ggml-base.en.bin").write_bytes(b"model")
        self.assertEqual(self.voice().engine(), ("whisper-cli", str(self.bin / "whisper-cli")))
        self.assertEqual(self.voice(whisper_model=str(self.dir / "missing.bin")).engine()[0], "parakeet-mlx")

    def test_ffmpeg_is_required(self):
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        self.assertFalse(Voice(Config({"voice": {"ffmpeg": str(self.dir / "nope")}})).available())

    def test_find_binary(self):
        script = self.tool("thing", "print(1)")
        self.assertEqual(find_binary("thing"), script)
        self.assertEqual(find_binary("other", script), script)
        self.assertIsNone(find_binary("thing", str(self.dir / "absent")))
        (self.bin / "plain").write_text("not executable")
        self.assertIsNone(find_binary("plain"))

    async def test_run_kills_on_timeout(self):
        with self.assertRaises(asyncio.TimeoutError):
            await run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.3)
        self.assertEqual(await run([sys.executable, "-c", "print('hi')"], 10), (0, "hi\n", ""))


@needs_ffmpeg
class TranscribeTests(VoiceTestCase):
    async def test_whisper_cli_gets_16k_mono_wav_and_temp_files_are_removed(self):
        self.tool("whisper-cli", FAKE_WHISPER)
        (self.models / "ggml-tiny.bin").write_bytes(b"model")
        note = self.dir / "note.wav"
        write_tone(note)
        voice = self.voice()
        self.assertTrue(voice.available())
        self.assertEqual(await voice.transcribe(note), "rate=16000 channels=1")
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertTrue(note.exists())

    async def test_parakeet_output_dir_and_failures(self):
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        note = self.dir / "note.wav"
        write_tone(note, rate=8000, channels=1)
        self.assertEqual(await self.voice().transcribe(note), "parakeet heard you")
        self.tool("parakeet-mlx", FAKE_FAILING)
        with self.assertRaisesRegex(RuntimeError, "model exploded"):
            await self.voice().transcribe(note)
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_unreadable_audio_raises(self):
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        bogus = self.dir / "bogus.ogg"
        bogus.write_bytes(b"definitely not audio")
        with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
            await self.voice().transcribe(bogus)
        self.assertEqual(list(self.work.iterdir()), [])


@needs_ffmpeg
class SpeakTests(VoiceTestCase):
    async def check_ogg(self, voice, tool):
        text = "-v hello there; $(rm -rf ~)"
        path = await voice.speak(text)
        log = json.loads((self.bin / f"{tool}.log").read_text())
        self.assertEqual(log["text"], text)
        self.assertNotIn(text, log["argv"])
        data = pathlib.Path(path).read_bytes()
        self.assertTrue(data.startswith(b"OggS"))
        self.assertIn(b"OpusHead", data[:200])
        self.assertEqual([p.name for p in pathlib.Path(path).parent.iterdir()], ["speech.ogg"])
        voice.discard(path)
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_kokoro_to_opus(self):
        self.tool("kokoro", FAKE_TTS)
        voice = self.voice(tts="kokoro")
        self.assertTrue(voice.can_speak())
        await self.check_ogg(voice, "kokoro")

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    async def test_say_reads_text_from_a_file(self):
        self.tool("say", FAKE_TTS)
        voice = self.voice()
        self.assertEqual(voice.tts()[0], "say")
        await self.check_ogg(voice, "say")

    async def test_no_tts_engine(self):
        voice = self.voice(tts="piper")  # piper also needs voice.piper_model
        self.tool("piper", FAKE_TTS)
        self.assertFalse(voice.can_speak())
        with self.assertRaises(VoiceUnavailable):
            await voice.speak("hi")

    @unittest.skipUnless(sys.platform == "darwin" and os.path.exists("/usr/bin/say"), "needs macOS say")
    async def test_real_say(self):
        voice = Voice(Config({"voice": {"ffmpeg": FFMPEG, "say": "/usr/bin/say", "tts": "say"}}),
                      tmp_root=str(self.work))
        path = await voice.speak("Testing one two.")
        self.assertTrue(pathlib.Path(path).read_bytes().startswith(b"OggS"))
        voice.discard(path)


if __name__ == "__main__":
    unittest.main()
