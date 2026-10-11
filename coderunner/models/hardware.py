"""Detect memory, accelerator and bandwidth so models can be sized to the host.

The memory budget follows llmfit/Jan/whichllm: on Apple silicon the GPU may
wire at most `iogpu.wired_limit_mb` (about 75% of RAM by default) and the
sandbox VM, macOS and the owner's apps need the rest. All sizes are GiB.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import re
import shutil
import subprocess
from typing import Callable

GiB = 1024 ** 3

# Unified-memory bandwidth in GB/s. Ultra chips are two Max dies.
APPLE_BANDWIDTH = {
    "M1": 68, "M1 Pro": 200, "M1 Max": 400,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
    "M5": 153,
}
# Binned Max chips (fewer GPU cores, narrower bus) only ship with 36 GB.
BINNED_MAX = {"M3 Max": 300, "M4 Max": 410}
TIER_SCALE = {"": 1, "Pro": 2, "Max": 4, "Ultra": 8}
NVIDIA_BANDWIDTH = {
    "5090": 1792, "5080": 960, "5070 Ti": 896, "5070": 672, "5060": 448,
    "4090": 1008, "4080": 717, "4070 Ti": 504, "4070": 504, "4060 Ti": 288, "4060": 272,
    "3090": 936, "3080": 760, "3070": 448, "3060": 360,
    "H100": 3350, "A100": 1555, "L40": 864, "L4": 300, "A6000": 768,
}
DEFAULT_BANDWIDTH = {"apple": 100, "nvidia": 300, "amd": 300, "none": 50}


@dataclasses.dataclass
class Hardware:
    os: str
    arch: str
    chip: str = ""
    ram_gb: float = 0.0
    gpu: str = "none"            # apple | nvidia | amd | none
    vram_gb: float = 0.0
    gpu_limit_gb: float = 0.0
    bandwidth_gbs: float = 50.0
    wsl: bool = False
    budget_gb: float = 0.0       # memory a model (weights + KV + buffers) may use
    sandbox_memory_gb: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in dataclasses.asdict(self).items()}


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _read(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read()
    except OSError:
        return ""


def apple_budget(ram_gb: float, gpu_limit_gb: float, sandbox_gb: float) -> float:
    return max(0.0, min(gpu_limit_gb, ram_gb - sandbox_gb - 2.5 - 0.1 * ram_gb))


def cpu_budget(ram_gb: float) -> float:
    return max(0.0, min(0.8 * ram_gb, ram_gb - 2))


def apple_bandwidth(chip: str, ram_gb: float = 0) -> float:
    match = re.search(r"\b(M\d+)(?:\s+(Pro|Max|Ultra))?", chip)
    if not match:
        return DEFAULT_BANDWIDTH["apple"]
    gen, tier = match.group(1), match.group(2) or ""
    if tier == "Ultra":
        return 2 * APPLE_BANDWIDTH.get(f"{gen} Max", APPLE_BANDWIDTH.get(gen, 100) * 4)
    name = f"{gen} {tier}".strip()
    if name in BINNED_MAX and 0 < ram_gb <= 36:
        return BINNED_MAX[name]
    if name in APPLE_BANDWIDTH:
        return APPLE_BANDWIDTH[name]
    # Newer chips: scale the base chip by tier until the table catches up.
    return APPLE_BANDWIDTH[gen] * TIER_SCALE[tier] if gen in APPLE_BANDWIDTH else DEFAULT_BANDWIDTH["apple"]


def nvidia_bandwidth(name: str) -> float:
    for key, value in NVIDIA_BANDWIDTH.items():  # ordered so "4070 Ti" wins over "4070"
        if key in name:
            return value
    return DEFAULT_BANDWIDTH["nvidia"]


def _nvidia(run) -> tuple[str, float, float] | None:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            gpus.append((parts[0], float(parts[1]) / 1024, float(parts[2]) / 1024))
        except (IndexError, ValueError):
            continue
    return max(gpus, key=lambda g: g[2]) if gpus else None  # the single GPU with most free memory


def _amd(run) -> tuple[str, float, float] | None:
    out = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    try:
        cards = json.loads(out or "")
    except json.JSONDecodeError:
        return None
    best = None
    for name, card in cards.items():
        try:
            total = int(card["VRAM Total Memory (B)"]) / GiB
            used = int(card["VRAM Total Used Memory (B)"]) / GiB
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or total - used > best[2]:
            best = (name, total, total - used)
    return best


def detect(sandbox_memory_gb: float = 1, *, run: Callable[[list[str]], str | None] | None = None,
           read: Callable[[str], str] | None = None, system: str | None = None,
           machine: str | None = None) -> Hardware:
    run, read = run or _run, read or _read
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    hw = Hardware(os=system, arch=machine, sandbox_memory_gb=float(sandbox_memory_gb))

    if system == "darwin":
        hw.chip = (run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
        try:
            hw.ram_gb = int((run(["sysctl", "-n", "hw.memsize"]) or "0").strip()) / GiB
        except ValueError:
            pass
        if machine == "arm64" or hw.chip.startswith("Apple"):
            try:
                wired_mb = int((run(["sysctl", "-n", "iogpu.wired_limit_mb"]) or "0").strip())
            except ValueError:
                wired_mb = 0
            hw.gpu = "apple"
            hw.gpu_limit_gb = wired_mb / 1024 if wired_mb > 0 else 0.75 * hw.ram_gb
            hw.vram_gb = hw.gpu_limit_gb
            hw.bandwidth_gbs = apple_bandwidth(hw.chip, hw.ram_gb)
            hw.budget_gb = apple_budget(hw.ram_gb, hw.gpu_limit_gb, hw.sandbox_memory_gb)
            return hw
        hw.budget_gb = cpu_budget(hw.ram_gb)
        return hw

    if system == "linux":
        match = re.search(r"MemTotal:\s+(\d+)\s*kB", read("/proc/meminfo"))
        hw.ram_gb = int(match.group(1)) / 1024 ** 2 if match else 0.0
        hw.wsl = "microsoft" in read("/proc/version").lower()
        cpu = re.search(r"model name\s*:\s*(.+)", read("/proc/cpuinfo"))
        hw.chip = cpu.group(1).strip() if cpu else ""

    for kind, probe in (("nvidia", _nvidia), ("amd", _amd)):
        found = probe(run)
        if found:
            name, total, free = found
            hw.gpu, hw.chip, hw.vram_gb, hw.gpu_limit_gb = kind, name, total, free
            hw.bandwidth_gbs = nvidia_bandwidth(name) if kind == "nvidia" else DEFAULT_BANDWIDTH["amd"]
            hw.budget_gb = max(0.0, free - 1)
            return hw
    hw.bandwidth_gbs = DEFAULT_BANDWIDTH["none"]
    hw.budget_gb = cpu_budget(hw.ram_gb)
    return hw
