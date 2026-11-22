import array
import asyncio
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import tarfile
import tempfile
import unittest
import wave
from unittest import mock

from coderunner.core import native, process
from coderunner.core.config import Config
from coderunner.voice import Segment, Voice, VoiceUnavailable, find_binary, run
from coderunner.voice import apple as apple_module
from coderunner.voice import parakeet as parakeet_module
from coderunner.voice import qwen as qwen_module
from coderunner.voice import service as service_module
from coderunner.voice.parakeet import ParakeetSpeech
from coderunner.voice.qwen import QwenASR, asr_transcript, cut_points

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
# Apple's helper: `transcribe <wav> <language>` or `prepare <language>`; exit 3 means no model for the language.
FAKE_APPLE = r'''
import json, pathlib, sys, wave
with pathlib.Path(__file__).with_suffix(".log").open("a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
if sys.argv[-1] == "hi":
    sys.stderr.write("no on-device speech model for hi"); sys.exit(3)
if sys.argv[1] == "transcribe":
    with wave.open(sys.argv[2]) as w:
        print(json.dumps({"start": 0, "end": 0.5, "text": f"Apple heard {w.getframerate()} Hz,"}))
        print(json.dumps({"start": 0.5, "end": 1.0, "text": " twice."}))
'''
# Parakeet's helper: `transcribe <wav> <model folder> <language>` or `prepare <model folder>`.
FAKE_PARAKEET = r'''
import json, pathlib, sys, wave
with pathlib.Path(__file__).with_name("parakeet.log").open("a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
assert pathlib.Path(sys.argv[-1 if sys.argv[1] == "prepare" else 3], "Encoder.mlmodelc/weights/weight.bin").is_file()
if sys.argv[1] == "transcribe":
    with wave.open(sys.argv[2]) as w:
        print(json.dumps({"start": 0.24, "end": 0.6000000000000001, "text": f"Parakeet heard {w.getframerate()} Hz."}))
        print(json.dumps({"start": 0.7, "end": 1.0, "text": " Twice. "}))
'''
FAKE_SAY = r'''
import sys, wave
args = sys.argv[1:]
assert "--data-format=LEI16@16000" in args
with wave.open(args[args.index("-o") + 1], "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 16000)
'''
REAL_PARAKEET_POSSIBLE = ParakeetSpeech.possible
FAKE_LLAMA = r'''
import pathlib, sys, wave
args = sys.argv[1:]
assert pathlib.Path(args[args.index("-m") + 1]).is_file() and pathlib.Path(args[args.index("--mmproj") + 1]).is_file()
assert args[args.index("-p") + 1] == "<__media__>"
with wave.open(args[args.index("--audio") + 1]) as w:
    print(f"language English<asr_text>Qwen heard {w.getframerate()} Hz\n  on {w.getnchannels()} channel.")
'''
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
        # Only our fake tools are visible, whatever is installed on this machine, and Apple speech and
        # Parakeet are off unless a test turns them on.
        for patcher in (mock.patch.dict(os.environ, {"PATH": str(self.bin), "CODERUNNER_HOME": str(self.dir / "home")}),
                        mock.patch.object(process, "EXTRA_PATHS", []),
                        mock.patch.object(apple_module, "macos_major", lambda: 0),
                        mock.patch.object(ParakeetSpeech, "possible", staticmethod(lambda: False)),
                        mock.patch.object(parakeet_module, "mac_language", lambda: "en"),
                        mock.patch.object(service_module, "_machine", lambda: {"chip": "Apple M3 Pro"})):
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

    def macos_26(self):
        patcher = mock.patch.object(apple_module, "macos_major", lambda: 26)
        patcher.start()
        self.addCleanup(patcher.stop)

    def apple(self, voice, body=FAKE_APPLE):
        """A built Apple helper on a macOS 26 Mac."""
        helper = voice.apple.helper()
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(f"#!{sys.executable}\n{body}")
        helper.chmod(0o755)
        self.macos_26()
        return helper

    def apple_silicon(self):
        patcher = mock.patch.object(ParakeetSpeech, "possible", staticmethod(lambda: True))
        patcher.start()
        self.addCleanup(patcher.stop)

    def parakeet(self, body=FAKE_PARAKEET):
        """Parakeet's helper and model files, as `coderunner voice setup` leaves them, on an Apple silicon Mac."""
        helper = parakeet_module.helper_path()
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(f"#!{sys.executable}\n{body}")
        helper.chmod(0o755)
        for name, _ in parakeet_module.MODEL["files"]:
            (ParakeetSpeech.model_dir() / name).parent.mkdir(parents=True, exist_ok=True)
            (ParakeetSpeech.model_dir() / name).write_bytes(b"coreml")
        self.apple_silicon()
        return helper

    def llama(self, body=FAKE_LLAMA):
        cli = QwenASR.llama_dir() / f"llama-{qwen_module.LLAMA_RELEASE}" / "llama-mtmd-cli"
        cli.parent.mkdir(parents=True, exist_ok=True)
        cli.write_text(f"#!{sys.executable}\n{body}")
        cli.chmod(0o755)
        return str(cli)

    def qwen(self, voice, body=FAKE_LLAMA):
        """llama.cpp and the Qwen3-ASR files, as `coderunner voice setup` leaves them."""
        cli = self.llama(body)
        QwenASR.models_dir().mkdir(parents=True, exist_ok=True)
        for name, _ in qwen_module.MODELS[voice.qwen.model]["files"]:
            (QwenASR.models_dir() / name).write_bytes(b"gguf")
        return cli


class EngineDetectionTests(VoiceTestCase):
    async def test_nothing_installed(self):
        voice = self.voice()
        self.assertEqual(voice.engines(), [])
        self.assertFalse(voice.available())
        with self.assertRaisesRegex(VoiceUnavailable, "coderunner voice setup"):
            await voice.transcribe(self.dir / "note.ogg")

    def test_engine_preference_order(self):
        names = lambda voice: [engine.name for engine in voice.engines()]  # noqa: E731
        self.tool("mlx_whisper", FAKE_DIR_ENGINE)
        self.assertEqual(names(self.voice()), ["mlx_whisper"])
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        self.tool("whisper-cli", FAKE_WHISPER)
        self.assertEqual(names(self.voice()), ["parakeet-mlx", "mlx_whisper"])  # whisper-cli needs a model
        (self.models / "ggml-base.en.bin").write_bytes(b"model")
        self.assertEqual(names(self.voice()), ["whisper-cli", "parakeet-mlx", "mlx_whisper"])
        self.assertEqual(names(self.voice(whisper_model=str(self.dir / "missing.bin")))[0], "parakeet-mlx")
        # Qwen3-ASR once set up, and Apple speech on macOS 26, come before the engines found installed.
        voice = self.voice()
        self.qwen(voice)
        self.assertEqual(names(voice), ["qwen3-asr", "whisper-cli", "parakeet-mlx", "mlx_whisper"])
        self.apple(voice)
        self.assertEqual(names(voice)[:2], ["apple", "qwen3-asr"])
        self.assertEqual(names(self.voice(stt="qwen3-asr")), ["qwen3-asr"])
        self.assertEqual(self.voice(stt="nonsense").engines(), [])
        self.assertEqual(voice.status()["engines"], names(voice))

    def test_parakeet_comes_first_on_apple_silicon(self):
        names = lambda voice: [engine.name for engine in voice.engines()]  # noqa: E731
        voice = self.voice()
        self.qwen(voice)
        self.apple(voice)
        self.assertEqual(names(voice), ["apple", "qwen3-asr"])
        self.parakeet()
        self.assertEqual(names(voice), ["parakeet-coreml", "apple", "qwen3-asr"])
        self.assertEqual(names(self.voice(stt="parakeet-coreml")), ["parakeet-coreml"])
        (ParakeetSpeech.model_dir() / "parakeet_vocab.json").unlink()     # a setup that did not finish
        self.assertEqual(names(voice), ["apple", "qwen3-asr"])

    def test_parakeet_follows_apple_where_setup_measured_it_slower(self):
        names = lambda voice: [engine.name for engine in voice.engines()]  # noqa: E731
        voice = self.voice()
        self.apple(voice)
        self.parakeet()
        check = service_module._check_path()
        check.parent.mkdir(parents=True, exist_ok=True)
        check.write_text(json.dumps({"machine": {"chip": "Apple M3 Pro"}, "parakeet_first": False}))
        self.assertEqual(names(voice), ["apple", "parakeet-coreml"])
        # Measured on another machine (a copied CodeRunner folder): the measured default holds.
        check.write_text(json.dumps({"machine": {"chip": "Apple M4 Max"}, "parakeet_first": False}))
        self.assertEqual(names(voice), ["parakeet-coreml", "apple"])

    def test_parakeet_needs_apple_silicon_and_macos_15(self):
        from types import SimpleNamespace
        for system, machine, major, expected in (("darwin", "arm64", 15, True), ("darwin", "arm64", 14, False),
                                                 ("darwin", "x86_64", 26, False), ("linux", "aarch64", 0, False)):
            with mock.patch.object(parakeet_module, "sys", SimpleNamespace(platform=system)), \
                    mock.patch.object(parakeet_module, "platform", SimpleNamespace(machine=lambda machine=machine: machine)), \
                    mock.patch.object(parakeet_module, "macos_major", lambda major=major: major):
                self.assertEqual(REAL_PARAKEET_POSSIBLE(), expected, (system, machine, major))

    def test_parakeet_languages(self):
        for language, code, serves in (("en-IN", "en", True), ("de_DE", "de", True), ("pt-BR", "pt", True),
                                       ("auto", "en", True), ("hi", "hi", False), ("ja-JP", "ja", False)):
            parakeet = ParakeetSpeech(language)
            self.assertEqual((parakeet.code(), parakeet.serves()), (code, serves), language)
        with mock.patch.object(parakeet_module, "mac_language", lambda: "hi"):
            self.assertFalse(ParakeetSpeech("auto").serves())
        self.assertEqual(len(parakeet_module.LANGUAGES), 25)

    def test_apple_needs_macos_26_and_a_way_to_build(self):
        voice = self.voice()
        self.assertFalse(voice.apple.possible())               # macOS 15 or Linux
        self.macos_26()
        self.assertFalse(voice.apple.possible())               # no helper and no Swift compiler
        if sys.platform == "darwin":
            self.tool("xcrun", "")
            self.assertTrue(voice.apple.possible())

    def test_asr_model_follows_memory(self):
        with mock.patch.object(qwen_module, "memory_gb", lambda: 8.0):
            self.assertEqual(self.voice().qwen.model, "qwen3-asr-0.6b")
        with mock.patch.object(qwen_module, "memory_gb", lambda: 18.0):
            self.assertEqual(self.voice().qwen.model, "qwen3-asr-1.7b")
            self.assertEqual(self.voice(asr_model="qwen3-asr-0.6b").qwen.model, "qwen3-asr-0.6b")

    def test_asr_transcript(self):
        self.assertEqual(asr_transcript("language English<asr_text>Hello  there.\n"), "Hello there.")
        self.assertEqual(asr_transcript("plain text"), "plain text")

    def test_long_recordings_are_cut_in_pauses(self):
        rate = 100
        loud = [1000] * (rate * 10)
        quiet = [0] * (rate // 5)
        samples = array.array("h", loud * 3 + quiet + loud * 3 + quiet + loud)    # pauses at 30.0 s and 60.2 s
        cuts = cut_points(samples, rate, chunk=35, slack=8)
        self.assertEqual(len(cuts), 2)
        self.assertTrue(30 * rate <= cuts[0] <= 30.2 * rate, cuts)
        self.assertTrue(60.2 * rate <= cuts[1] <= 60.4 * rate, cuts)
        self.assertEqual(cut_points(array.array("h", loud), rate), [])

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
        self.assertEqual(await voice.segments(note), [Segment(0.0, 1.0, "rate=16000 channels=1")])
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

    async def test_apple_speech_gives_timed_phrases(self):
        voice = self.voice(language="en-IN")
        helper = self.apple(voice)
        note = self.dir / "note.wav"
        write_tone(note)
        self.assertEqual(await voice.segments(note), [Segment(0.0, 0.5, "Apple heard 16000 Hz,"),
                                                      Segment(0.5, 1.0, "twice.")])
        self.assertEqual(await voice.transcribe(note), "Apple heard 16000 Hz, twice.")
        logged = helper.with_suffix(".log").read_text().split()
        self.assertEqual((logged[0], logged[2]), ("transcribe", "en-IN"))
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_parakeet_gives_timed_phrases(self):
        voice = self.voice(language="en-IN")
        helper = self.parakeet()
        apple = self.apple(voice)
        note = self.dir / "note.wav"
        write_tone(note)
        self.assertEqual(await voice.segments(note), [Segment(0.24, 0.6, "Parakeet heard 16000 Hz."),
                                                      Segment(0.7, 1.0, "Twice.")])
        logged = (helper.parent / "parakeet.log").read_text().split()
        self.assertEqual((logged[0], logged[2], logged[3]), ("transcribe", str(ParakeetSpeech.model_dir()), "en"))
        self.assertFalse(apple.with_suffix(".log").exists())
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_languages_parakeet_lacks_go_to_apple_then_qwen(self):
        voice = self.voice(language="hi")
        helper = self.parakeet()
        self.apple(voice)
        self.qwen(voice)
        note = self.dir / "note.wav"
        write_tone(note)
        self.assertEqual(await voice.transcribe(note), "Qwen heard 16000 Hz on 1 channel.")
        self.assertFalse((helper.parent / "parakeet.log").exists())      # not even started for Hindi
        with self.assertRaisesRegex(RuntimeError, "Parakeet does not transcribe hi"):
            await self.voice(language="hi", stt="parakeet-coreml").transcribe(note)

    async def test_failing_parakeet_hands_over_to_apple(self):
        voice = self.voice()
        self.parakeet(FAKE_FAILING)
        self.apple(voice)
        note = self.dir / "note.wav"
        write_tone(note)
        self.assertEqual(await voice.transcribe(note), "Apple heard 16000 Hz, twice.")
        with self.assertRaisesRegex(RuntimeError, "parakeet failed.*model exploded"):
            await self.voice(stt="parakeet-coreml").transcribe(note)

    async def test_timed_pieces_for_the_file_index(self):
        voice = self.voice()
        self.assertEqual(await voice.timed(self.dir / "absent.wav"), [])     # not set up: nothing, no error
        self.apple(voice)
        note = self.dir / "memo.wav"
        write_tone(note)
        self.assertEqual(await voice.timed(note), [(0.0, "Apple heard 16000 Hz,"), (0.5, "twice.")])

    async def test_languages_apple_lacks_go_to_qwen(self):
        voice = self.voice(language="hi")
        self.apple(voice)
        self.qwen(voice)
        note = self.dir / "note.wav"
        write_tone(note, rate=8000)
        self.assertEqual(await voice.transcribe(note), "Qwen heard 16000 Hz on 1 channel.")
        # Without a fallback the reason is reported.
        with self.assertRaisesRegex(RuntimeError, "no on-device speech model for hi"):
            await self.voice(language="hi", stt="apple").transcribe(note)

    async def test_failing_engine_hands_over(self):
        voice = self.voice()
        self.apple(voice, FAKE_FAILING)
        self.qwen(voice)
        note = self.dir / "note.wav"
        write_tone(note)
        self.assertEqual(await voice.transcribe(note), "Qwen heard 16000 Hz on 1 channel.")
        self.llama(FAKE_FAILING)
        with self.assertRaisesRegex(RuntimeError, "qwen3-asr failed.*model exploded"):
            await self.voice(stt="qwen3-asr").transcribe(note)
        self.llama("print('a log line but no transcript')")
        with self.assertRaisesRegex(RuntimeError, "no transcript"):
            await self.voice(stt="qwen3-asr").transcribe(note)

    async def test_qwen_segments_long_recordings(self):
        voice = self.voice(stt="qwen3-asr")
        self.qwen(voice)
        note = self.dir / "long.wav"
        write_tone(note, rate=16000, channels=1, seconds=1.0)
        with mock.patch.object(qwen_module, "CHUNK", 0.4), mock.patch.object(qwen_module, "SLACK", 0.2):
            segments = await voice.segments(note)
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(all(0.2 <= s.end - s.start <= 0.4 for s in segments[:-1]), segments)
        self.assertEqual((segments[0].start, segments[-1].end), (0.0, 1.0))
        self.assertTrue(all(a.end == b.start for a, b in zip(segments, segments[1:], strict=False)))
        self.assertTrue(all(s.text == "Qwen heard 16000 Hz on 1 channel." for s in segments))
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_unreadable_audio_raises(self):
        self.tool("parakeet-mlx", FAKE_DIR_ENGINE)
        bogus = self.dir / "bogus.ogg"
        bogus.write_bytes(b"definitely not audio")
        with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
            await self.voice().transcribe(bogus)
        self.assertEqual(list(self.work.iterdir()), [])


def tarball(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeStream:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self, size):
        yield self.data


class FakeHub:
    def __init__(self, license="cc-by-4.0"):
        self.calls, self.revisions, self.license = [], [], license

    def download(self, repo, file, dest, sha256=None, progress=None, revision="main"):
        self.calls.append((repo, file, sha256))
        self.revisions.append(revision)
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dest).write_bytes(b"gguf")
        return pathlib.Path(dest)

    def resolve_license(self, repo):
        return {"spdx": self.license, "name": None, "link": f"https://huggingface.co/{repo}", "repo": repo,
                "gated": False}


# SwiftPM building the Parakeet helper: checks the package it is given and leaves the helper where SwiftPM would.
FAKE_SWIFT_BUILD = r'''
import pathlib, shutil, sys
args = sys.argv[1:]
pathlib.Path(__file__).with_name("swift.log").write_text(" ".join(args))
package = pathlib.Path(args[args.index("--package-path") + 1])
assert '"FluidAudio-0.17.5/Sources/FluidAudio"' in (package / "Package.swift").read_text()
assert (package / "FluidAudio-0.17.5" / "Sources" / "FluidAudio").is_dir()
assert "coderunner-parakeet transcribe" in (package / "Helper" / "main.swift").read_text()
out = pathlib.Path(args[args.index("--scratch-path") + 1]) / "release" / "coderunner-parakeet"
out.parent.mkdir(parents=True)
shutil.copy(pathlib.Path(__file__).with_name("parakeet-fake"), out)
'''
FLUIDAUDIO_TARBALL = {"FluidAudio-0.17.5/Sources/FluidAudio/ASR.swift": "",
                      "FluidAudio-0.17.5/Sources/FastClusterWrapper/fastcluster.cpp": "",
                      "FluidAudio-0.17.5/Sources/MachTaskSelfWrapper/mach.c": ""}


FAKE_SWIFTC = r'''
import pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
out.write_text("#!/bin/sh\necho built\n"); out.chmod(0o755)
with pathlib.Path(__file__).with_suffix(".log").open("a") as log:
    log.write(sys.argv[1] + "\n")
'''


class SetupTests(VoiceTestCase):
    def install(self, archive, sha256):
        build = ("macos-arm64", sha256)
        with mock.patch.object(QwenASR, "build_for_host", staticmethod(lambda: build)), \
                mock.patch("httpx.stream", lambda *args, **kwargs: FakeStream(archive)):
            return QwenASR().install_llama()

    def test_llama_build_is_checked_before_unpacking(self):
        archive = tarball({f"llama-{qwen_module.LLAMA_RELEASE}/llama-mtmd-cli": "#!/bin/sh\n"})
        cli = self.install(archive, hashlib.sha256(archive).hexdigest())
        self.assertEqual(QwenASR().cli(), cli)
        self.assertEqual([p.name for p in QwenASR.llama_dir().iterdir()], [f"llama-{qwen_module.LLAMA_RELEASE}"])
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.install(archive, "0" * 64)
        with mock.patch.object(QwenASR, "build_for_host", staticmethod(lambda: None)):
            with self.assertRaisesRegex(VoiceUnavailable, "no build"):
                QwenASR().install_llama()

    async def test_setup_downloads_qwen_only_when_apple_cannot_serve(self):
        hub, installs = FakeHub(), []

        def install_llama(qwen, progress=None):
            installs.append(qwen.model)
            return self.llama()

        with mock.patch.object(QwenASR, "install_llama", install_llama):
            # Apple speech has the language: nothing to download.
            voice = self.voice()
            self.apple(voice)
            result = await voice.setup(hub=hub)
            self.assertEqual((result["apple"], result["engines"], hub.calls, installs), ("ready", ["apple"], [], []))
            # Hindi: no Apple model, so llama.cpp and Qwen3-ASR are fetched, with pinned checksums.
            voice = self.voice(language="hi")
            result = await voice.setup(hub=hub)
            self.assertIn("no on-device speech model", result["apple"])
            spec = qwen_module.MODELS[voice.qwen.model]
            self.assertEqual(hub.calls, [(spec["repo"], name, sha) for name, sha in spec["files"]])
            self.assertEqual((result["engines"], len(installs)), (["apple", "qwen3-asr"], 1))
            self.assertEqual(result["ok"], bool(FFMPEG))

    def fluidaudio(self, archive):
        """FluidAudio's release served by a fake GitHub, with its pinned checksum."""
        for patcher in (mock.patch.dict(parakeet_module.FLUIDAUDIO, {"sha256": hashlib.sha256(archive).hexdigest()}),
                        mock.patch("httpx.stream", lambda *args, **kwargs: FakeStream(archive))):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_setup_builds_parakeet_and_times_it_against_apple(self):
        hub = FakeHub()
        self.tool("parakeet-fake", FAKE_PARAKEET)
        self.tool("swift", FAKE_SWIFT_BUILD)
        self.tool("say", FAKE_SAY)
        self.fluidaudio(tarball(FLUIDAUDIO_TARBALL))
        voice = self.voice()
        self.apple(voice)
        self.apple_silicon()
        result = await voice.setup(hub=hub)
        self.assertEqual((result["parakeet-coreml"], result["apple"]), ("ready", "ready"))
        # Model files at the pinned revision, each with its pinned checksum; nothing for Qwen3-ASR.
        model = parakeet_module.MODEL
        self.assertEqual(hub.calls, [(model["repo"], name, sha) for name, sha in model["files"]])
        self.assertEqual(set(hub.revisions), {model["revision"]})
        self.assertNotIn("qwen3-asr", result)
        swift = (self.bin / "swift.log").read_text().split()
        self.assertEqual(swift[:6], ["build", "-c", "release", "--product", "coderunner-parakeet", "--package-path"])
        # Only the helper is kept; it was loaded once before the first message.
        helper = parakeet_module.helper_path()
        self.assertEqual(sorted(p.name for p in helper.parent.iterdir() if "parakeet" in p.name),
                         [helper.name, "parakeet.log"])
        self.assertEqual((helper.parent / "parakeet.log").read_text().splitlines()[0],
                         f"prepare {ParakeetSpeech.model_dir()}")
        check = result["speed_check"]
        self.assertEqual(check["parakeet-coreml"]["text"], "Parakeet heard 16000 Hz. Twice.")
        self.assertIn("seconds", check["apple"])
        self.assertTrue(check["parakeet_first"])
        self.assertEqual(json.loads(service_module._check_path().read_text()), check)
        self.assertEqual(result["engines"], ["parakeet-coreml", "apple"])
        # Noticeably slower than Apple speech on this machine: Apple speech goes first.
        with mock.patch.object(service_module, "NOTICEABLE_S", -1.0):
            self.assertFalse((await voice.compare(lambda message: None))["parakeet_first"])
        self.assertEqual([engine.name for engine in voice.engines()], ["apple", "parakeet-coreml"])
        self.assertEqual(list(self.work.iterdir()), [])

    async def test_setup_leaves_parakeet_out_for_languages_it_lacks(self):
        hub = FakeHub()
        self.apple_silicon()
        with mock.patch.object(QwenASR, "install_llama", lambda qwen, progress=None: self.llama()):
            result = await self.voice(language="hi").setup(hub=hub)
        self.assertNotIn("parakeet-coreml", result)
        self.assertEqual({repo for repo, _, _ in hub.calls}, {qwen_module.MODELS[self.voice().qwen.model]["repo"]})
        self.assertEqual(result["engines"], ["qwen3-asr"])

    async def test_parakeet_licence_is_checked_before_downloading(self):
        hub = FakeHub(license="cc-by-nc-4.0")
        voice = self.voice()
        self.apple(voice)
        self.apple_silicon()
        result = await voice.setup(hub=hub)
        self.assertRegex(result["parakeet-coreml"], "licence.*commercial")
        self.assertEqual((hub.calls, result["engines"]), ([], ["apple"]))

    def test_fluidaudio_is_checked_before_building(self):
        self.tool("swift", "import sys\nsys.exit(9)")
        self.fluidaudio(tarball(FLUIDAUDIO_TARBALL))
        with mock.patch.dict(parakeet_module.FLUIDAUDIO, {"sha256": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                ParakeetSpeech().build(lambda message: None)
        self.assertEqual(list(parakeet_module.helper_path().parent.iterdir()), [])
        (self.bin / "swift").unlink()
        with self.assertRaisesRegex(VoiceUnavailable, "command line tools"):
            ParakeetSpeech().build(lambda message: None)

    @unittest.skipUnless(sys.platform == "darwin", "Swift helpers are built on macOS")
    async def test_apple_helper_is_built_once_and_a_failure_is_remembered(self):
        voice = self.voice()
        self.macos_26()
        self.tool("swiftc", FAKE_SWIFTC)
        first = await voice.apple.build()
        self.assertEqual(first, str(voice.apple.helper()))
        self.assertEqual(await voice.apple.build(), first)
        self.assertEqual((self.bin / "swiftc.log").read_text(), "-O\n")
        # A changed source that fails to build is remembered, so it is not rebuilt on every message.
        source = self.dir / "apple_speech.swift"
        source.write_text("broken")
        self.tool("swiftc", "import sys\nsys.stderr.write('syntax error'); sys.exit(1)")
        with mock.patch.object(apple_module, "SOURCE", source):
            self.assertTrue(voice.apple.possible())
            with self.assertRaisesRegex(VoiceUnavailable, "could not be built"):
                await voice.apple.build()
            self.assertFalse(voice.apple.possible())
            failed = native.helper_path(source, apple_module.NAME)
            self.assertIn("syntax error", failed.with_name(failed.name + ".failed").read_text())


@unittest.skipUnless(os.environ.get("CODERUNNER_REAL_VOICE") and FFMPEG and os.path.exists("/usr/bin/say"),
                     "set CODERUNNER_REAL_VOICE=1 on a Mac to transcribe real speech")
class RealSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_spoken_sentence_round_trip(self):
        """macOS `say` speaks a sentence; the engine voice setup chose writes it down."""
        with tempfile.TemporaryDirectory() as tmp:
            home = os.environ.get("CODERUNNER_HOME", tmp)
            with mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}):
                voice = Voice(Config({"voice": {"ffmpeg": FFMPEG, "language": "en-US"}}), tmp_root=tmp)
                await voice.setup()
                speech = pathlib.Path(tmp) / "speech.aiff"
                await run(["/usr/bin/say", "-o", str(speech), "Book a table for two at seven tonight."], 60)
                segments = await voice.segments(speech)
        text = " ".join(segment.text for segment in segments).lower()
        for word in ("book", "table", "tonight"):
            self.assertIn(word, text)
        self.assertTrue("seven" in text or "7" in text, text)    # engines may write numbers as digits
        self.assertTrue(all(0 <= s.start <= s.end <= 4 for s in segments), segments)


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
