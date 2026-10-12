"""Qwen3-ASR on llama.cpp's official build: the open speech-to-text model, used where Apple's is not.

Qwen3-ASR (Apache 2.0) covers 30 languages and 22 Chinese dialects and detects the language itself.
Of the speech models llama.cpp runs (Qwen3-ASR, Voxtral Mini, Qwen Omni, Gemma 4 E2B/E4B) it scores
best on the Open ASR Leaderboard: 5.75% average English WER for 1.7B and 6.60% for 0.6B (Voxtral Mini
3B 7.84%). The pinned llama.cpp build and the model are downloaded, checksummed, into the CodeRunner
folder. Recordings longer than CHUNK seconds are cut at quiet moments; each piece is one segment.
"""

from __future__ import annotations

import array
import hashlib
import os
import pathlib
import platform
import sys
import wave

from ..core import paths
from ..core.process import run
from .common import STT_HINT, Progress, Segment, VoiceUnavailable

# llama.cpp's official build, checked against the SHA-256 GitHub publishes for each asset.
LLAMA_RELEASE = "b11564"
LLAMA_BUILDS = {  # (os, arch) -> (asset, sha256)
    ("darwin", "arm64"): ("macos-arm64", "fa7897448562888373f1c4d393dc860da78873e0fbc230fd17312e0e972af844"),
    ("darwin", "x86_64"): ("macos-x64", "64489d9ef02d7b0fc40dce0fc26b9286f52d514e96ea60a8b7c50af6ad425a39"),
    ("linux", "x86_64"): ("ubuntu-x64", "349c1499f28d93da836ba8c60160c2e77545f6cf30b4a038a0cf040ff0a949ef"),
    ("linux", "aarch64"): ("ubuntu-arm64", "92a42da53d621c1650465b73bf4fcdcdb90e656aa9997d2590a6206b27619932"),
}
# Open ASR Leaderboard (09-10-2026), average English WER: 1.7B 5.75, 0.6B 6.60. Files: (name, sha256).
MODELS = {
    "qwen3-asr-1.7b": {"repo": "ggml-org/Qwen3-ASR-1.7B-GGUF", "files": (
        ("Qwen3-ASR-1.7B-Q8_0.gguf", "58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57"),
        ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d"))},
    "qwen3-asr-0.6b": {"repo": "ggml-org/Qwen3-ASR-0.6B-GGUF", "files": (
        ("Qwen3-ASR-0.6B-Q8_0.gguf", "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971"),
        ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d"))},
}
SMALL_MACHINE_GB = 8.5       # the 0.6B model at or below this much memory
ASR_PREFIX = "<asr_text>"    # Qwen3-ASR writes "language English<asr_text>the transcript"
CHUNK = 30.0                 # seconds per piece of a long recording
SLACK = 5.0                  # a cut moves up to this far back to land in a pause


def memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return 0.0


def asr_transcript(output: str) -> str:
    """The words from Qwen3-ASR's output, without its language tag."""
    text = output.split(ASR_PREFIX, 1)[1] if ASR_PREFIX in output else output
    return " ".join(text.split())


def cut_points(samples, rate: int, chunk: float = CHUNK, slack: float = SLACK) -> list[int]:
    """Sample offsets splitting a recording into pieces of at most `chunk` seconds, each cut in the
    quietest tenth of a second within `slack` seconds before the limit."""
    step, cuts, start = max(rate // 10, 1), [], 0
    while len(samples) - start > chunk * rate:
        end = start + int(chunk * rate)
        quietest = min(range(end - int(slack * rate), end - step + 1, step),
                       key=lambda i: sum(map(abs, samples[i:i + step])))
        start = quietest + step // 2
        cuts.append(start)
    return cuts


class QwenASR:
    name = "qwen3-asr"

    def __init__(self, model: str | None = None, timeout: float = 300):
        self.configured = model
        self.timeout = timeout

    @property
    def model(self) -> str:
        if self.configured in MODELS:
            return self.configured
        return "qwen3-asr-0.6b" if 0 < memory_gb() <= SMALL_MACHINE_GB else "qwen3-asr-1.7b"

    # Files ----------------------------------------------------------------
    @staticmethod
    def models_dir() -> pathlib.Path:
        return paths.assistant_dir() / "models" / "speech"

    @staticmethod
    def llama_dir() -> pathlib.Path:
        return paths.assistant_dir() / "runtimes" / f"llama-{LLAMA_RELEASE}"

    def files(self) -> list[pathlib.Path] | None:
        """The model and its audio encoder when both are downloaded."""
        found = [self.models_dir() / name for name, _ in MODELS[self.model]["files"]]
        return found if all(path.is_file() for path in found) else None

    def cli(self) -> str | None:
        found = sorted(self.llama_dir().glob("**/llama-mtmd-cli")) if self.llama_dir().exists() else []
        return str(found[0]) if found and os.access(found[0], os.X_OK) else None

    def ready(self) -> bool:
        return bool(self.cli() and self.files())

    # Installation ---------------------------------------------------------
    @staticmethod
    def build_for_host() -> tuple[str, str] | None:
        arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64e", "arm64")
        if sys.platform == "linux" and arch == "arm64":
            arch = "aarch64"
        return LLAMA_BUILDS.get((sys.platform, arch))

    def install_llama(self, progress: Progress | None = None) -> str:
        build = self.build_for_host()
        if build is None:
            raise VoiceUnavailable("llama.cpp publishes no build for this platform")
        import tarfile

        import httpx
        asset, sha256 = build
        name = f"llama-{LLAMA_RELEASE}-bin-{asset}.tar.gz"
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{name}"
        target = self.llama_dir()
        target.mkdir(parents=True, exist_ok=True)
        archive = target / name
        if progress:
            progress(f"downloading {name}")
        digest = hashlib.sha256()
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as response:
            response.raise_for_status()
            with open(archive, "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    digest.update(chunk)
                    handle.write(chunk)
        if digest.hexdigest() != sha256:
            archive.unlink()
            raise RuntimeError(f"{name} does not match its published checksum")
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")     # refuses absolute paths, links out, devices
        archive.unlink()
        cli = self.cli()
        if not cli:
            raise RuntimeError(f"{name} has no llama-mtmd-cli")
        return cli

    def download(self, progress: Progress | None = None, hub=None) -> list[pathlib.Path]:
        from ..models.hub import Hub
        spec = MODELS[self.model]
        hub = hub or Hub()
        return [hub.download(spec["repo"], name, self.models_dir() / name, sha256=sha256, progress=progress)
                for name, sha256 in spec["files"]]

    def install(self, progress: Progress | None = None, hub=None) -> None:
        if not self.cli():
            self.install_llama(progress)
        self.download(progress, hub)

    # Transcription --------------------------------------------------------
    def pieces(self, wav: pathlib.Path, work: pathlib.Path) -> list[tuple[float, float, pathlib.Path]]:
        """(start, end, file) for each piece of the recording; short ones are a single piece."""
        with wave.open(str(wav)) as audio:
            rate, params = audio.getframerate(), audio.getparams()
            samples = array.array("h", audio.readframes(audio.getnframes()))
        if sys.byteorder == "big":
            samples.byteswap()
        cuts = cut_points(samples, rate, CHUNK, SLACK)
        if not cuts:
            return [(0.0, len(samples) / rate, wav)]
        found = []
        for index, (start, end) in enumerate(zip([0, *cuts], [*cuts, len(samples)])):
            piece = work / f"piece-{index:04d}.wav"
            with wave.open(str(piece), "wb") as out:
                out.setparams(params)
                out.writeframes(samples[start:end].tobytes())
            found.append((start / rate, end / rate, piece))
        return found

    async def _transcribe(self, cli: str, model: pathlib.Path, mmproj: pathlib.Path, wav: pathlib.Path) -> str:
        # The user turn is the audio alone, as Qwen3-ASR was trained (an empty prompt starts a chat
        # session instead). The transcript is printed through llama.cpp's log, so logging stays on;
        # the rest of the log goes to stderr.
        argv = [cli, "-m", str(model), "--mmproj", str(mmproj), "--audio", str(wav), "-p", "<__media__>",
                "--temp", "0", "-n", "1024", "--no-warmup"]
        code, out, err = await run(argv, self.timeout)
        if code != 0:
            raise RuntimeError(f"qwen3-asr failed ({code}): {err.strip()[-500:]}")
        if ASR_PREFIX not in out:        # seen when the machine was short of memory
            raise RuntimeError(f"qwen3-asr wrote no transcript: {err.strip()[-300:]}")
        return asr_transcript(out)

    async def segments(self, wav: pathlib.Path, work: pathlib.Path) -> list[Segment]:
        cli, files = self.cli(), self.files()
        if not cli or not files:
            raise VoiceUnavailable(STT_HINT)
        model, mmproj = files
        found = []
        for start, end, piece in self.pieces(wav, work):
            text = await self._transcribe(cli, model, mmproj, piece)
            if text:
                found.append(Segment(round(start, 2), round(end, 2), text))
        return found
