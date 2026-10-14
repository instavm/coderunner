"""Parakeet on Core ML: speech to text on the Neural Engine of Apple silicon Macs (macOS 15 and later).

The model is Parakeet Ultra, Moondream's post-training of NVIDIA's Parakeet TDT 0.6B v3 (CC-BY-4.0), in
FluidInference's Core ML build: 25 European languages, with punctuation and capitals. Of the Core ML speech
models it scores best on the Open ASR Leaderboard's seven English sets (5.80% average WER; Parakeet TDT v3
6.26%, Parakeet Unified 6.27%), and on FLEURS in all 25 languages (9.55% average; v3 11.62%).

FluidAudio (Apache-2.0) runs it. Voice setup downloads FluidAudio's pinned source release, checks its SHA-256,
builds a small helper from parakeet_speech.swift with SwiftPM (Xcode command line tools) and keeps only the
helper; the model files are downloaded at a pinned revision, each checked against its SHA-256. Every
transcription is one short process: the model loads from Core ML's cache in about a tenth of a second, so
nothing stays in memory between voice messages.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

from ..core import paths
from ..core.process import run
from .apple import macos_major
from .common import STT_HINT, Progress, Segment, UnsupportedLanguage, VoiceUnavailable, phrases

SOURCE = pathlib.Path(__file__).with_name("parakeet_speech.swift")
NAME = "coderunner-parakeet"
MIN_MACOS = 15
BUILD_TIMEOUT = 1800
LANGUAGES = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())
# FluidAudio's source release; the SHA-256 of GitHub's tarball of the tag.
FLUIDAUDIO = {"version": "0.17.5", "sha256": "15c5055094d1a5d0b4d7495409970adbf83fa66937bdeb482bbe6897b5e58690",
              "url": "https://github.com/FluidInference/FluidAudio/archive/refs/tags/v0.17.5.tar.gz"}
MODEL = {"repo": "FluidInference/parakeet-ultra-coreml", "revision": "95eaa59a39d4394f047a4dc5cce480388a60d1b6",
         "files": (  # what FluidAudio loads for Parakeet Ultra: 631 MB
             ("Decoder.mlmodelc/analytics/coremldata.bin", "fe92b6cfaa012abd5248c0bc877832f19807015abffc60d87b8ccc8ccb48b3b5"),
             ("Decoder.mlmodelc/coremldata.bin", "3b06e66768f0df7e21795f50e2b29300e33eeb1a2579dc42c695279c2d308497"),
             ("Decoder.mlmodelc/model.mil", "956f600207f88396017ca5c96cfa3acd5bfee60835a5b44b762e033a8fb28955"),
             ("Decoder.mlmodelc/weights/weight.bin", "02a0d219f281b9665bc10c8768649403b2eebbbf4b44c627041d948e0de11bb4"),
             ("Encoder.mlmodelc/analytics/coremldata.bin", "d87101d824d6723cf95304da33755c3c60e762663b9ef2b0c4bd0aa166a09a0d"),
             ("Encoder.mlmodelc/coremldata.bin", "397a84a4062f563cbc5f56077c674f09a61d85be5090f61d2f1932afb92ac0fe"),
             ("Encoder.mlmodelc/model.mil", "f5d601568a4171d99a314c0fe3f6bc67715da2623732a3fb566e050ea83e5848"),
             ("Encoder.mlmodelc/weights/weight.bin", "315ba01f33cadbf601d43ac7f5c86208b7aa75fdaa34705c9869d5abe3521c9b"),
             ("JointDecisionv3.mlmodelc/analytics/coremldata.bin", "68d38ca646aebafa7a9329e2efda50f5767c49e89fdb5f77f212072bb66f97c4"),
             ("JointDecisionv3.mlmodelc/coremldata.bin", "5e3af5a4ce686f6c237cadbd9284e10d333bc0e1546633cd0e430e6194044bc4"),
             ("JointDecisionv3.mlmodelc/model.mil", "791b3c3cf3eb2079c84623fc880f6bba008d1366e5f8b03e9a8ed8bd4d7194a0"),
             ("JointDecisionv3.mlmodelc/weights/weight.bin", "3f310b85b82341c53ec383025ab094a4e462ee1c592e1ad7c6bfe39cff66ca25"),
             ("Preprocessor.mlmodelc/analytics/coremldata.bin", "c9beeb989c8d66f8be11df59bc6df277ec76cee404f6865b46243835ef562f6d"),
             ("Preprocessor.mlmodelc/coremldata.bin", "dbde3f2300842c1fd51ef3ff948a0bcffe65ffd2dca10707f2509f32c1d65b1d"),
             ("Preprocessor.mlmodelc/metadata.json", "2a98699e22d279dd37fa1d238aeb1c6db1df0d6fad687775324157689d8f3acf"),
             ("Preprocessor.mlmodelc/model.mil", "4b8518a956450fec57f06c2a21bdffc26973f7f1fa6842fb38fe917f896b6b93"),
             ("Preprocessor.mlmodelc/weights/weight.bin", "129b76e3aeafa8afa3ea76d995b964b145fe83700d579f6ff42c4c38fa0968ea"),
             ("parakeet_vocab.json", "7ec60e05f1b24480736ec0eed40900f4626bce1fa9a60fd700ec7e2a59198735"))}
# FluidAudio's library built from its sources, without its command line tool, its tests and the prebuilt
# text-normalization library its text to speech uses (an 87 MB download speech to text does not need).
PACKAGE = """// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "coderunner-parakeet",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "FastClusterWrapper", path: "@SRC@/Sources/FastClusterWrapper", publicHeadersPath: "include"),
        .target(name: "MachTaskSelfWrapper", path: "@SRC@/Sources/MachTaskSelfWrapper", publicHeadersPath: "include"),
        .target(name: "FluidAudio", dependencies: ["FastClusterWrapper", "MachTaskSelfWrapper"],
                path: "@SRC@/Sources/FluidAudio", exclude: ["ASR/Parakeet/Unified/benchmark.md"],
                resources: [.process("TTS/LuxTts/G2p/Resources")]),
        .executableTarget(name: "coderunner-parakeet", dependencies: ["FluidAudio"], path: "Helper"),
    ],
    cxxLanguageStandard: .cxx17
)
"""


@functools.cache
def mac_language() -> str:
    """The Mac's own language ("en" for en_IN), which Apple speech also follows for "auto"."""
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleLocale"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip().replace("-", "_").split("_")[0].lower()


def helper_path() -> pathlib.Path:
    """Where the helper built from this source and FluidAudio release lives."""
    digest = hashlib.sha256(SOURCE.read_bytes() + PACKAGE.encode() + FLUIDAUDIO["version"].encode()).hexdigest()
    return paths.assistant_dir() / "bin" / f"{NAME}-{digest[:12]}"


def swift() -> list[str] | None:
    if shutil.which("swift"):
        return ["swift"]
    if shutil.which("xcrun"):
        return ["xcrun", "swift"]
    return None


class ParakeetSpeech:
    name = "parakeet-coreml"

    def __init__(self, language: str = "auto", timeout: float = 300):
        self.language = language
        self.timeout = timeout

    # Where it runs --------------------------------------------------------
    @staticmethod
    def possible() -> bool:
        """Apple silicon (for the Neural Engine) and macOS 15 or later."""
        return sys.platform == "darwin" and platform.machine() == "arm64" and macos_major() >= MIN_MACOS

    def code(self) -> str:
        wanted = mac_language() if self.language in ("", "auto") else self.language
        return wanted.replace("_", "-").split("-")[0].lower()

    def serves(self) -> bool:
        return self.code() in LANGUAGES

    # Files ----------------------------------------------------------------
    @staticmethod
    def model_dir() -> pathlib.Path:
        return paths.assistant_dir() / "models" / "speech" / "parakeet-ultra"

    @staticmethod
    def helper() -> pathlib.Path | None:
        """The helper built from this source or, until voice setup rebuilds it, from an earlier one."""
        current = helper_path()
        if current.exists():
            return current
        older = sorted((path for path in current.parent.glob(f"{NAME}-*") if os.access(path, os.X_OK)),
                       key=lambda path: path.stat().st_mtime) if current.parent.exists() else []
        return older[-1] if older else None

    def downloaded(self) -> bool:
        return all((self.model_dir() / name).is_file() for name, _ in MODEL["files"])

    def ready(self) -> bool:
        return self.possible() and self.helper() is not None and self.downloaded()

    # Transcription --------------------------------------------------------
    async def segments(self, wav: pathlib.Path, work: pathlib.Path) -> list[Segment]:
        if not self.serves():
            raise UnsupportedLanguage(f"Parakeet does not transcribe {self.code() or self.language}")
        helper = self.helper()
        if helper is None or not self.downloaded():
            raise VoiceUnavailable(STT_HINT)
        code, out, err = await run([str(helper), "transcribe", str(wav), str(self.model_dir()), self.code()], self.timeout)
        if code != 0:
            raise RuntimeError(f"parakeet failed ({code}): {err.strip()[-500:]}")
        return phrases(out)

    # Installation ---------------------------------------------------------
    def install(self, config, progress: Progress | None = None, hub=None) -> None:
        """Licence check, model files, then the helper, loaded once so Core ML compiles the model for this Mac."""
        from ..models import licenses
        from ..models.hub import Hub
        say = progress or (lambda message: None)
        hub = hub or Hub()
        decision, why, _ = licenses.check(MODEL["repo"], config, hub)
        if decision != "allow":
            raise VoiceUnavailable(f"Parakeet's licence: {why}")
        for name, sha256 in MODEL["files"]:
            hub.download(MODEL["repo"], name, self.model_dir() / name, revision=MODEL["revision"], sha256=sha256,
                         progress=say)
        if not helper_path().exists():
            self.build(say)
        say("preparing Parakeet for the Neural Engine (once, about half a minute)")
        done = subprocess.run([str(self.helper()), "prepare", str(self.model_dir())], capture_output=True, text=True,
                              timeout=BUILD_TIMEOUT)
        if done.returncode != 0:
            raise RuntimeError(f"Parakeet could not load its model: {done.stderr.strip()[-300:]}")

    def build(self, progress: Progress) -> pathlib.Path:
        compiler = swift()
        if compiler is None:
            raise VoiceUnavailable("building the Parakeet helper needs the Xcode command line tools "
                                   "(xcode-select --install)")
        import httpx
        target = helper_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        work = pathlib.Path(tempfile.mkdtemp(prefix="parakeet-build-", dir=target.parent))
        try:
            archive = work / "fluidaudio.tar.gz"
            progress(f"downloading FluidAudio {FLUIDAUDIO['version']}")
            digest = hashlib.sha256()
            with httpx.stream("GET", FLUIDAUDIO["url"], follow_redirects=True,
                              timeout=httpx.Timeout(30, read=300)) as response:
                response.raise_for_status()
                with open(archive, "wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        digest.update(chunk)
                        handle.write(chunk)
            if digest.hexdigest() != FLUIDAUDIO["sha256"]:
                raise RuntimeError(f"FluidAudio {FLUIDAUDIO['version']} does not match its pinned checksum")
            with tarfile.open(archive) as tar:
                tar.extractall(work, filter="data")     # refuses absolute paths, links out, devices
            source = f"FluidAudio-{FLUIDAUDIO['version']}"
            (work / "Package.swift").write_text(PACKAGE.replace("@SRC@", source))
            (work / "Helper").mkdir()
            shutil.copy(SOURCE, work / "Helper" / "main.swift")
            progress("building the Parakeet helper with SwiftPM (a minute or two)")
            done = subprocess.run([*compiler, "build", "-c", "release", "--product", NAME, "--package-path", str(work),
                                   "--scratch-path", str(work / ".build")], capture_output=True, text=True,
                                  timeout=BUILD_TIMEOUT)
            product = work / ".build" / "release" / NAME
            if done.returncode != 0 or not product.exists():
                raise RuntimeError(f"the Parakeet helper did not build: {(done.stderr or done.stdout).strip()[-500:]}")
            partial = target.with_name(f".{target.name}.{os.getpid()}")
            shutil.copy2(product, partial)
            os.replace(partial, target)
            for old in target.parent.glob(f"{NAME}-*"):
                if old != target:
                    old.unlink(missing_ok=True)
            return target
        finally:
            shutil.rmtree(work, ignore_errors=True)
