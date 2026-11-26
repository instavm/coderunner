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
# Prompt reading is bound by compute, not bandwidth. Until this machine is measured, reading speed is
# guessed as (factor x bandwidth) / active parameters in billions. Apple: measured on an M3 Pro, where
# Qwen3.5-9B and Bonsai 27B both read at about 14.5 x 150 GB/s / params. The others are rough.
READ_PER_BANDWIDTH = {"apple": 14.5, "nvidia": 60.0, "amd": 30.0, "none": 8.0}


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
    gpu_cores: int = 0           # Apple GPU core count when detected via ioreg
    compute_cap: float = 0.0     # NVIDIA compute capability (8.9 Ada, 9.0 Hopper, 10.x/12.x Blackwell)
    unified: bool = False        # GPU shares system memory (Apple silicon, DGX Spark / GB10, Jetson)
    ram_bandwidth_gbs: float = 80.0   # system RAM, for MoE experts kept in RAM (desktop DDR5 dual channel)
    nodes: int = 1               # machines pooled over llama.cpp RPC (models.cluster)
    # Measured on this machine by `coderunner models bench` (bench.py) or Tier 0 probe (probe.py).
    gen_factor: float = 1.0      # measured / estimated generation speed
    read_k: float = 0.0          # prompt reading: tokens/s x billions of active parameters (0: estimate)
    measured: str = ""           # what the measurement used, e.g. "qwen3.5-9b on ollama, 2026-10-12"
    # Apple Neural Engine telemetry from Tier 0 probe (probe.py), when available.
    ane_cores: int = 0
    ane_tops: float = 0.0
    ane_verified: bool | None = None
    gpu_contention_pct: float | None = None

    def to_dict(self) -> dict:
        out = {k: round(v, 2) if isinstance(v, float) else v for k, v in dataclasses.asdict(self).items()}
        if self.gpu == "apple" and self.ram_gb > 0:
            out["tier"] = ram_tier(self.ram_gb)
        return out


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


def os_app_reserve_gb(ram_gb: float) -> float:
    """Host memory reserved for macOS, Electron, Chrome and active Core ML buffers.

    Grows sub-linearly with installed RAM and plateaus around 16 GiB so 256-512 GB Macs do not
    lose 25-51 GiB to a linear percentage tax, while 8 GB Macs keep a lean ~2.3 GiB floor.
    """
    return round(min(16.0, max(2.2, 2.0 + 0.06 * min(ram_gb, 64.0) + 0.025 * max(0.0, ram_gb - 64.0))), 2)


def default_apple_gpu_limit(ram_gb: float) -> float:
    """macOS default Metal wired ceiling when iogpu.wired_limit_mb is 0 (auto): ~2/3 on <=24 GB, 3/4 on >=32 GB."""
    return (2.0 / 3.0 if ram_gb <= 24.5 else 0.75) * ram_gb


def safe_wired_limit_gb(ram_gb: float, sandbox_gb: float = 2.0) -> float:
    """Safe raised iogpu.wired_limit_mb ceiling (in GiB) leaving OS, apps, sandbox and 2 GiB safety margin."""
    return max(0.0, min(0.94 * ram_gb, ram_gb - os_app_reserve_gb(ram_gb) - sandbox_gb - 2.0))


def apple_budget(ram_gb: float, gpu_limit_gb: float, sandbox_gb: float, ane_gb: float = 0.0) -> float:
    return max(0.0, min(gpu_limit_gb, ram_gb - sandbox_gb - os_app_reserve_gb(ram_gb) - ane_gb))


# Unified RAM tier profiles for Apple Silicon (8 GB through 512 GB).
APPLE_TIERS: tuple[dict, ...] = (
    {"min_gb": 0,   "tier_gb": 8,   "label": "8 GB",   "target_ctx": 16384,  "parallel_tasks": 1, "sandbox_gb": 1.0, "sandbox_on_demand": True,  "driver_mode": "off",         "dedicated_judge": False, "dual_resident": False, "no_mmap": False},
    {"min_gb": 14,  "tier_gb": 16,  "label": "16 GB",  "target_ctx": 32768,  "parallel_tasks": 2, "sandbox_gb": 1.5, "sandbox_on_demand": True,  "driver_mode": "exclusive",   "dedicated_judge": False, "dual_resident": False, "no_mmap": False},
    {"min_gb": 17,  "tier_gb": 18,  "label": "18 GB",  "target_ctx": 32768,  "parallel_tasks": 2, "sandbox_gb": 2.0, "sandbox_on_demand": True,  "driver_mode": "exclusive",   "dedicated_judge": False, "dual_resident": False, "no_mmap": False},
    {"min_gb": 22,  "tier_gb": 24,  "label": "24 GB",  "target_ctx": 32768,  "parallel_tasks": 2, "sandbox_gb": 2.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": False, "no_mmap": False},
    {"min_gb": 30,  "tier_gb": 32,  "label": "32 GB",  "target_ctx": 65536,  "parallel_tasks": 2, "sandbox_gb": 2.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": False, "no_mmap": False},
    {"min_gb": 34,  "tier_gb": 36,  "label": "36 GB",  "target_ctx": 65536,  "parallel_tasks": 3, "sandbox_gb": 3.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": False, "no_mmap": False},
    {"min_gb": 44,  "tier_gb": 48,  "label": "48 GB",  "target_ctx": 65536,  "parallel_tasks": 3, "sandbox_gb": 4.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": False, "no_mmap": False},
    {"min_gb": 60,  "tier_gb": 64,  "label": "64 GB",  "target_ctx": 131072, "parallel_tasks": 3, "sandbox_gb": 4.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
    {"min_gb": 90,  "tier_gb": 96,  "label": "96 GB",  "target_ctx": 131072, "parallel_tasks": 4, "sandbox_gb": 4.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
    {"min_gb": 120, "tier_gb": 128, "label": "128 GB", "target_ctx": 131072, "parallel_tasks": 4, "sandbox_gb": 8.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
    {"min_gb": 180, "tier_gb": 192, "label": "192 GB", "target_ctx": 131072, "parallel_tasks": 4, "sandbox_gb": 8.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
    {"min_gb": 240, "tier_gb": 256, "label": "256 GB", "target_ctx": 262144, "parallel_tasks": 6, "sandbox_gb": 8.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
    {"min_gb": 480, "tier_gb": 512, "label": "512 GB", "target_ctx": 262144, "parallel_tasks": 6, "sandbox_gb": 8.0, "sandbox_on_demand": False, "driver_mode": "co_resident", "dedicated_judge": True,  "dual_resident": True,  "no_mmap": True},
)


def ram_tier(ram_gb: float) -> dict:
    """Return the Apple Silicon unified RAM tier profile for `ram_gb` (8 GB through 512 GB)."""
    picked = APPLE_TIERS[0]
    for tier in APPLE_TIERS:
        if ram_gb >= tier["min_gb"]:
            picked = tier
    return dict(picked)


def parallel_tasks(config=None, hw: Hardware | None = None) -> int:
    """Shared local task concurrency and server slot capacity policy across scheduling and llama-server.

    Explicit `agent.parallel_tasks` wins; otherwise Apple Silicon tiers scale from 1 (8 GB) to 6 (256-512 GB),
    defaulting to 2 when neither config nor hardware is provided or on non-Apple hosts.
    """
    set_by_owner = config.get("agent.parallel_tasks") if config is not None else None
    if set_by_owner:
        return max(1, int(set_by_owner))
    if hw is None and config is not None:
        hw = detect(float(config.get("sandbox.memory_gb") or 2))
    if hw is not None and getattr(hw, "gpu", "") == "apple" and getattr(hw, "ram_gb", 0):
        return int(ram_tier(hw.ram_gb)["parallel_tasks"])
    return 2


def unified_budget(ram_gb: float, sandbox_gb: float) -> float:
    """Linux unified-memory systems (DGX Spark): no wired-memory cap, but the OS and the sandbox share the pool."""
    return max(0.0, ram_gb - sandbox_gb - 4 - 0.05 * ram_gb)


def cpu_budget(ram_gb: float) -> float:
    return max(0.0, min(0.8 * ram_gb, ram_gb - 2))


def apple_gpu_cores(run: Callable[[list[str]], str | None] | None = None) -> int:
    """Apple GPU core count via ioreg (distinguishes binned 30-core M3 Max 36/96 GB from 40-core M3 Max)."""
    out = (run or _run)(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"]) or ""
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return int(match.group(1)) if match else 0


def apple_bandwidth(chip: str, ram_gb: float = 0, gpu_cores: int = 0) -> float:
    match = re.search(r"\b(M\d+)(?:\s+(Pro|Max|Ultra))?", chip)
    if not match:
        return DEFAULT_BANDWIDTH["apple"]
    gen, tier = match.group(1), match.group(2) or ""
    name = f"{gen} {tier}".strip()
    # Binned Max chips: 30-core M3 Max ships in 36 GB and 96 GB (300 GB/s); 32-core M4 Max in 36 GB (410 GB/s).
    if name == "M3 Max" and ((0 < gpu_cores <= 30) or (gpu_cores == 0 and 0 < ram_gb <= 36)):
        return BINNED_MAX["M3 Max"]
    if name == "M4 Max" and ((0 < gpu_cores <= 32) or (gpu_cores == 0 and 0 < ram_gb <= 36)):
        return BINNED_MAX["M4 Max"]
    if name in APPLE_BANDWIDTH:
        return APPLE_BANDWIDTH[name]
    if tier == "Ultra":
        return 2 * APPLE_BANDWIDTH.get(f"{gen} Max", APPLE_BANDWIDTH.get(gen, 100) * 4)
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


def swap_used_gb(*, run: Callable[[list[str]], str | None] | None = None, system: str | None = None) -> float | None:
    """Current swap in use (GiB) on macOS via `sysctl vm.swapusage`; None when unavailable."""
    system = (system or platform.system()).lower()
    if system != "darwin":
        return None
    out = (run or _run)(["sysctl", "-n", "vm.swapusage"]) or ""
    match = re.search(r"used\s*=\s*([\d.]+)\s*([MGK])", out, re.I)
    if not match:
        return None
    val, unit = float(match.group(1)), match.group(2).upper()
    return round(val / 1024 if unit == "M" else val / (1024 ** 2) if unit == "K" else val, 2)


def live_memory(ram_gb: float = 0, *, run: Callable[[list[str]], str | None] | None = None,
                read: Callable[[str], str] | None = None, system: str | None = None) -> dict:
    """Live memory and swap pressure snapshot: {free_gb, reclaimable_gb, swap_used_gb, wired_gb, pressure}."""
    run_fn, read_fn = run or _run, read or _read
    system = (system or platform.system()).lower()
    free = available_gb(ram_gb, run=run_fn, read=read_fn, system=system)
    info: dict = {"free_gb": round(free, 2) if free is not None else None,
                  "reclaimable_gb": None, "swap_used_gb": None, "wired_gb": None, "pressure": "unknown"}
    if system == "darwin":
        info["swap_used_gb"] = swap_used_gb(run=run_fn, system=system)
        vm = run_fn(["vm_stat"]) or ""
        page_match = re.search(r"page size of (\d+) bytes", vm)
        if page_match:
            page = int(page_match.group(1))
            def pages(label: str) -> int:
                m = re.search(rf"{re.escape(label)}:\s+(\d+)", vm)
                return int(m.group(1)) if m else 0
            reclaimable = (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")
                           + pages("Pages purgeable")) * page / GiB
            wired = pages("Pages wired down") * page / GiB
            info["reclaimable_gb"] = round(reclaimable, 2)
            info["wired_gb"] = round(wired, 2)
    swap = info["swap_used_gb"] or 0.0
    if free is not None:
        info["pressure"] = "high" if (swap >= 2.5 or (ram_gb and free / max(ram_gb, 1) < 0.2)) else \
            "moderate" if (swap >= 1.0 or (ram_gb and free / max(ram_gb, 1) < 0.35)) else "low"
    return info


def available_gb(ram_gb: float = 0, *, run: Callable[[list[str]], str | None] | None = None,
                 read: Callable[[str], str] | None = None, system: str | None = None) -> float | None:
    """GiB a new process can take right now without the machine swapping, or None where that cannot be read.
    macOS: the kernel's own figure (kern.memorystatus_level, the percentage `memory_pressure` prints) of the
    memory installed. Linux: MemAvailable. A plan says what fits the machine; this says what fits it now, with
    whatever else is open."""
    run, read = run or _run, read or _read
    system = (system or platform.system()).lower()
    try:
        if system == "darwin":
            if not ram_gb:
                ram_gb = int((run(["sysctl", "-n", "hw.memsize"]) or "0").strip()) / GiB
            level = int((run(["sysctl", "-n", "kern.memorystatus_level"]) or "").strip())
            return ram_gb * level / 100 if ram_gb and 0 <= level <= 100 else None
        if system == "linux":
            match = re.search(r"MemAvailable:\s+(\d+)\s*kB", read("/proc/meminfo"))
            return int(match.group(1)) * 1024 / GiB if match else None
    except (OSError, ValueError):
        return None
    return None


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
            hw.gpu_cores = apple_gpu_cores(run)
            hw.gpu_limit_gb = wired_mb / 1024 if wired_mb > 0 else default_apple_gpu_limit(hw.ram_gb)
            hw.vram_gb = hw.gpu_limit_gb
            hw.bandwidth_gbs = apple_bandwidth(hw.chip, hw.ram_gb, hw.gpu_cores)
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
