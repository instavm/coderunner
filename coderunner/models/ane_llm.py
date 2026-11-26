"""Apple Neural Engine (ANE) & Apple Intelligence on-device fast distiller/sub-worker support.

Provides:
1. Apple Intelligence FoundationModels bridge (macOS 26+): compiles `foundation_helper.swift` once and
   uses `SystemLanguageModel.default` when Apple Intelligence is ready on the Mac (zero extra GPU RAM).
2. Core ML ANE small-model candidate registry (MiniCPM-2B / Qwen3-1.7B / Qwen3.5-2B palettized Core ML
   builds) and a qualification benchmark harness for measuring ANE distillation speed & latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from ..core import paths

HELPER_SRC = pathlib.Path(__file__).with_name("foundation_helper.swift")

# Permissively licensed small-model candidates targeted for Apple Neural Engine (Core ML / ANEMLL) distillation.
ANE_DISTILLER_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "apple-foundation-models",
        "name": "Apple Intelligence (FoundationModels)",
        "runtime": "apple-foundation",
        "license": "system",
        "params_b": 3.0,
        "est_ram_gb": 0.0,
        "notes": "Built into macOS 26+ on Apple Silicon; zero download and zero extra GPU VRAM when enabled.",
    },
    {
        "id": "minicpm4-0.5b-ane",
        "name": "MiniCPM4 0.5B (Core ML ANE)",
        "runtime": "coreml-ane",
        "license": "apache-2.0",
        "repo": "openbmb/MiniCPM4-0.5B",
        "params_b": 0.5,
        "est_ram_gb": 0.6,
        "notes": "Ultra-fast Apache 2.0 0.5B distiller suited for palettized INT4/INT6 Core ML ANE execution.",
    },
    {
        "id": "minicpm-2b-ane",
        "name": "MiniCPM 2B (Core ML ANE)",
        "runtime": "coreml-ane",
        "license": "apache-2.0",
        "repo": "openbmb/MiniCPM-2B-sft-bf16",
        "params_b": 2.4,
        "est_ram_gb": 1.6,
        "notes": "Apache 2.0 2B class model for ANE-offloaded page extraction and summarization.",
    },
    {
        "id": "qwen3-1.7b-ane",
        "name": "Qwen3 1.7B (Core ML ANE)",
        "runtime": "coreml-ane",
        "license": "apache-2.0",
        "repo": "Qwen/Qwen3-1.7B",
        "params_b": 1.7,
        "est_ram_gb": 1.2,
        "notes": "Apache 2.0 1.7B model with strong structured extraction; runs on ANE via Core ML palettization.",
    },
)


def foundation_supported() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 26
    except ValueError:
        return False


def _helper_binary() -> pathlib.Path | None:
    if not foundation_supported() or not HELPER_SRC.is_file() or not shutil.which("swiftc"):
        return None
    digest = hashlib.sha256(HELPER_SRC.read_bytes()).hexdigest()[:12]
    out = paths.assistant_dir() / "bin" / f"foundation-helper-{digest}"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    done = subprocess.run(["swiftc", "-O", str(HELPER_SRC), "-o", str(tmp)],
                          capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(out)
    out.chmod(0o755)
    return out


def probe_foundation() -> dict[str, Any]:
    """Check whether macOS 26+ Apple Intelligence FoundationModels is ready on this Mac."""
    if not foundation_supported():
        return {"supported": False, "available": False, "reason": "requires Apple Silicon macOS 26+"}
    binary = _helper_binary()
    if binary is None:
        return {"supported": False, "available": False, "reason": "Swift toolchain unavailable"}
    try:
        done = subprocess.run([str(binary), "--probe"], capture_output=True, text=True, timeout=15)
        data = json.loads(done.stdout.strip().splitlines()[-1]) if done.stdout.strip() else {}
        return data if isinstance(data, dict) else {"supported": True, "available": False}
    except Exception as exc:
        return {"supported": True, "available": False, "reason": str(exc)[:160]}


async def distill_with_foundation(text: str, focus: str, title: str = "", timeout: float = 15.0) -> str:
    """Distill fetched page text using Apple Intelligence FoundationModels when ready; returns '' if unavailable."""
    binary = _helper_binary()
    if binary is None:
        return ""
    proc = await asyncio.create_subprocess_exec(
        str(binary), "--focus", f"{focus} ({title})" if title else focus,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(text[:10000].encode("utf-8")), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return ""
    if proc.returncode != 0 or not out:
        return ""
    try:
        data = json.loads(out.decode("utf-8", errors="replace").strip().splitlines()[-1])
        return str(data.get("text") or "").strip() if data.get("ok") else ""
    except ValueError:
        return ""


async def benchmark_distiller(sample_text: str, focus: str) -> dict[str, Any]:
    """Benchmark available on-device ANE / Apple FoundationModels distillation latency and compression."""
    status = probe_foundation()
    out: dict[str, Any] = {
        "foundation": status,
        "candidates": list(ANE_DISTILLER_CANDIDATES),
        "measured": False,
    }
    if status.get("available"):
        t0 = time.monotonic()
        distilled = await distill_with_foundation(sample_text, focus)
        elapsed = round(time.monotonic() - t0, 2)
        out.update(
            measured=bool(distilled),
            engine="apple-foundation-models",
            seconds=elapsed,
            input_chars=len(sample_text),
            output_chars=len(distilled),
        )
    return out
