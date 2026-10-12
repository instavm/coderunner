"""Speech-to-text tools the owner installed before voice setup existed: whisper.cpp's `whisper-cli`
(with a ggml model), `parakeet-mlx` and `mlx_whisper`. Each returns the whole recording as one segment.
"""

from __future__ import annotations

import pathlib

from ..core import paths
from ..core.process import find_binary, run
from .common import Segment, seconds


class InstalledEngine:
    def __init__(self, name: str, binary: str, settings):
        self.name = name
        self.binary = binary
        self._get = settings          # voice.<key> lookup

    def argv(self, wav: pathlib.Path, work: pathlib.Path) -> list[str]:
        language = self._get("language") or "auto"
        if self.name == "whisper-cli":
            return [self.binary, "-m", whisper_model(self._get), "-f", str(wav), "-l", language,
                    "-nt", "-np", "-otxt", "-of", str(work / "audio")]
        argv = [self.binary, str(wav), "--output-dir", str(work), "--output-format", "txt"]
        model = self._get("parakeet_model" if self.name == "parakeet-mlx" else "mlx_whisper_model")
        if model:
            argv += ["--model", model]
        if self.name == "mlx_whisper" and language != "auto":
            argv += ["--language", language]
        return argv

    async def segments(self, wav: pathlib.Path, work: pathlib.Path) -> list[Segment]:
        code, out, err = await run(self.argv(wav, work), float(self._get("timeout") or 300))
        if code != 0:
            raise RuntimeError(f"{self.name} failed ({code}): {err.strip()[-500:]}")
        result = work / "audio.txt"
        text = " ".join((result.read_text(errors="replace") if result.exists() else out).split())
        return [Segment(0.0, round(seconds(wav), 2), text)] if text else []


def whisper_model(get) -> str | None:
    configured = get("whisper_model")
    if configured:
        path = pathlib.Path(configured).expanduser()
        return str(path) if path.is_file() else None
    models = sorted((paths.assistant_dir() / "models").glob("ggml-*.bin"))
    return str(models[0]) if models else None


def find(get) -> list[InstalledEngine]:
    """Installed engines in order of preference; whisper-cli counts only with a model."""
    found = []
    whisper = find_binary("whisper-cli", get("whisper_cli"))
    if whisper and whisper_model(get):
        found.append(InstalledEngine("whisper-cli", whisper, get))
    for name in ("parakeet-mlx", "mlx_whisper"):
        binary = find_binary(name, get(name.replace("-", "_")))
        if binary:
            found.append(InstalledEngine(name, binary, get))
    return found
