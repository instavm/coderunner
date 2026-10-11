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
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400, "M3 Ultra": 819,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
    "M5": 153,
}
# Binned Max chips (fewer GPU cores, narrower bus) only ship with 36 GB.
BINNED_MAX = {"M3 Max": 300, "M4 Max": 410}
TIER_SCALE = {"": 1, "Pro": 2, "Max": 4, "Ultra": 8}
# Matched in order, so more specific names come first.
NVIDIA_BANDWIDTH = {
    "GB10": 273, "Thor": 273, "GB200": 8000, "B300": 8000, "B200": 8000, "H200": 4800, "GH200": 4000,
    "H100 PCIe": 2000, "H100": 3350, "H800": 3350, "A100-SXM4-80GB": 2039, "A100 80GB": 1935, "A100": 1555,
    "RTX PRO 6000": 1792, "RTX 6000 Ada": 960, "L40S": 864, "L40": 864, "L4": 300, "A6000": 768, "A5000": 768,
    "5090": 1792, "5080": 960, "5070 Ti": 896, "5070": 672, "5060 Ti": 448, "5060": 448,
    "4090": 1008, "4080": 717, "4070 Ti": 504, "4070": 504, "4060 Ti": 288, "4060": 272,
    "3090": 936, "3080": 760, "3070": 448, "3060": 360,
}
# NVIDIA systems whose GPU shares the system's memory (nvidia-smi reports no dedicated VRAM).
NVIDIA_UNIFIED = ("GB10", "Thor", "Orin")
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
    gpus: int = 0                # NVIDIA/AMD cards counted in vram_gb
    compute_cap: float = 0.0     # NVIDIA compute capability (8.9 Ada, 9.0 Hopper, 10.x/12.x Blackwell)
    unified: bool = False        # GPU shares system memory (Apple silicon, DGX Spark / GB10, Jetson)

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


def unified_budget(ram_gb: float, sandbox_gb: float) -> float:
    """Linux unified-memory systems (DGX Spark): no wired-memory cap, but the OS and the sandbox share the pool."""
    return max(0.0, ram_gb - sandbox_gb - 4 - 0.05 * ram_gb)


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


def _nvidia(run) -> list[dict]:
    """Every NVIDIA GPU: name, total and free memory in GiB (None on unified-memory systems), compute capability."""
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
               "--format=csv,noheader,nounits"])
    if out is None:   # older drivers do not know compute_cap
        out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue

        def number(index: int) -> float | None:
            try:
                return float(parts[index])
            except (IndexError, ValueError):     # "[N/A]" on GB10 and Jetson: memory is shared
                return None
        total, free = number(1), number(2)
        gpus.append({"name": parts[0], "total": total / 1024 if total is not None else None,
                     "free": free / 1024 if free is not None else None, "cap": number(3) or 0.0})
    return gpus


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
            hw.gpu, hw.unified = "apple", True
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

    gpus = _nvidia(run)
    if gpus:
        names = [g["name"] for g in gpus]
        hw.gpu, hw.chip, hw.gpus = "nvidia", names[0] if len(set(names)) == 1 else " + ".join(names), len(gpus)
        hw.compute_cap = min(g["cap"] for g in gpus)
        hw.bandwidth_gbs = min(nvidia_bandwidth(name) for name in names)
        if any(marker in name for name in names for marker in NVIDIA_UNIFIED) or any(g["total"] is None for g in gpus):
            # DGX Spark / GB10, Jetson: one pool of memory for the OS, the sandbox and the model.
            hw.unified = True
            hw.vram_gb = hw.gpu_limit_gb = hw.ram_gb
            hw.budget_gb = unified_budget(hw.ram_gb, hw.sandbox_memory_gb)
        else:
            # llama.cpp and Ollama split a model across cards; keep ~1 GiB free on each for buffers.
            hw.vram_gb = sum(g["total"] for g in gpus)
            hw.gpu_limit_gb = sum(g["free"] for g in gpus)
            hw.budget_gb = max(0.0, hw.gpu_limit_gb - len(gpus))
        return hw
    found = _amd(run)
    if found:
        name, total, free = found
        hw.gpu, hw.chip, hw.vram_gb, hw.gpu_limit_gb, hw.gpus = "amd", name, total, free, 1
        hw.bandwidth_gbs = DEFAULT_BANDWIDTH["amd"]
        hw.budget_gb = max(0.0, free - 1)
        return hw
    hw.bandwidth_gbs = DEFAULT_BANDWIDTH["none"]
    hw.budget_gb = cpu_budget(hw.ram_gb)
    return hw
