"""Tier 0 pre-download bare-metal Mac probe (Metal GPU + Apple Neural Engine + SSD + HF CDN).

Compiles and caches `probe.swift` via `core/native.py` using only public macOS frameworks
(Metal, MetalPerformanceShaders, CoreML, Foundation) — no SIP changes, no sudo, no coremltools venv.
Generates a deterministic ANE-native 4D channel-first 1x1 conv `.mlmodel` locally in <20 ms so no
binary model weights live in git. Results are cached per (machine fingerprint, macOS build).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time
from typing import Callable

import httpx

from ..core import native, paths
from . import bench, hardware
from .hub import endpoint

SOURCE = pathlib.Path(__file__).with_name("probe.swift")
HELPER_NAME = "coderunner-mac-probe"
PROBE_VERSION = 1
CONTENTION_YIELD_PCT = 8.0


# ---- Minimal pure-Python Core ML NeuralNetwork (.mlmodel) protobuf emitter ------------------------------
# Emits a valid Core ML spec-v4 NeuralNetwork with N 1x1 FP16 Conv2D + ReLU layers over [C, 1, W]
# (channel-first 4D layout native to Apple Neural Engine SRAM), requiring zero external packages.

def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _len_field(field_number: int, payload: bytes) -> bytes:
    return _tag(field_number, 2) + _varint(len(payload)) + payload


def _varint_field(field_number: int, value: int) -> bytes:
    return _tag(field_number, 0) + _varint(value)


def _str_field(field_number: int, text: str) -> bytes:
    return _len_field(field_number, text.encode("utf-8"))


def _feature_desc(name: str, channels: int, width: int) -> bytes:
    # ArrayFeatureType: shape=[C, 1, W] (field 1 packed/repeated int64), dataType=FLOAT16 (65552, field 2)
    shape = b"".join(_varint_field(1, dim) for dim in (channels, 1, width))
    arr_type = shape + _varint_field(2, 65552)
    # FeatureType.multiArrayType = field 5
    ftype = _len_field(5, arr_type)
    # FeatureDescription: name=1, type=3
    return _str_field(1, name) + _len_field(3, ftype)


def _conv_layer(name: str, inp: str, out: str, channels: int, weight_bytes: bytes) -> bytes:
    # WeightParams: float16Value = field 2 (bytes)
    weights = _len_field(2, weight_bytes)
    # ConvolutionLayerParams:
    #   outputChannels=1, kernelChannels=2, nGroups=10, kernelSize=20, stride=30,
    #   dilationFactor=40, valid=50, isDeconvolution=60, hasBias=70, weights=90
    conv = (
        _varint_field(1, channels)
        + _varint_field(2, channels)
        + _varint_field(10, 1)
        + _varint_field(20, 1) + _varint_field(20, 1)
        + _varint_field(30, 1) + _varint_field(30, 1)
        + _varint_field(40, 1) + _varint_field(40, 1)
        + _len_field(50, b"")
        + _varint_field(60, 0)
        + _varint_field(70, 0)
        + _len_field(90, weights)
    )
    # NeuralNetworkLayer: name=1, input=2, output=3, convolution=100
    return _str_field(1, name) + _str_field(2, inp) + _str_field(3, out) + _len_field(100, conv)


def _relu_layer(name: str, inp: str, out: str) -> bytes:
    # ActivationParams.ReLU = field 10 (empty); NeuralNetworkLayer.activation = field 130
    act = _len_field(10, b"")
    return _str_field(1, name) + _str_field(2, inp) + _str_field(3, out) + _len_field(130, act)


def build_ane_probe_mlmodel(dest: pathlib.Path, channels: int = 256, width: int = 256, layers: int = 8) -> pathlib.Path:
    """Write a deterministic ANE-native 1x1 FP16 Conv2D+ReLU .mlmodel (~1 MB) with zero external deps."""
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic small FP16 values (~0.0156, 0x2200) so activations stay finite across layers.
    n_weights = channels * channels
    weight_bytes = struct.pack("<H", 0x2200) * n_weights
    nn_layers = []
    cur = "x"
    for idx in range(layers):
        mid = f"c{idx}"
        nxt = "y" if idx == layers - 1 else f"r{idx}"
        nn_layers.append(_len_field(1, _conv_layer(f"conv_{idx}", cur, mid, channels, weight_bytes)))
        nn_layers.append(_len_field(1, _relu_layer(f"relu_{idx}", mid, nxt)))
        cur = nxt
    nn = b"".join(nn_layers) + _varint_field(10, 1)  # arrayInputShapeMapping = EXACT_ARRAY_MAPPING (1)
    desc = _len_field(1, _feature_desc("x", channels, width)) + _len_field(10, _feature_desc("y", channels, width))
    # Model: specificationVersion=1 (7 -> FP16 multiarray I/O + FP16 weights, macOS 13+), description=2, neuralNetwork=500
    model = _varint_field(1, 7) + _len_field(2, desc) + _len_field(500, nn)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(model)
    os.replace(tmp, dest)
    return dest


def macos_build() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        out = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True, timeout=5)
        return (out.stdout or "").strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def probe_hf_cdn_mb_s(timeout: float = 4.0) -> float | None:
    """Measure Hugging Face CDN range-read throughput (MB/s) over ~2 MB for honest download ETAs."""
    url = f"{endpoint()}/Qwen/Qwen3.5-4B/resolve/main/tokenizer.json"
    try:
        start = time.monotonic()
        total = 0
        with httpx.Client(timeout=httpx.Timeout(timeout, read=timeout), follow_redirects=True,
                          headers={"User-Agent": "coderunner"}) as client:
            with client.stream("GET", url, headers={"Range": "bytes=0-2097151"}) as resp:
                if resp.status_code not in (200, 206):
                    return None
                for chunk in resp.iter_bytes(131072):
                    total += len(chunk)
                    if time.monotonic() - start >= 3.0:
                        break
        elapsed = time.monotonic() - start
        return round((total / 1e6) / elapsed, 2) if total >= 65536 and elapsed > 0 else None
    except httpx.HTTPError:
        return None


def sample_macmon(seconds: float = 1.5) -> dict | None:
    """Optional sudoless Apple Silicon power reading via `macmon` when installed on PATH."""
    if sys.platform != "darwin" or not shutil.which("macmon"):
        return None
    try:
        proc = subprocess.Popen(["macmon", "pipe", "-i", "250", "-s", str(max(2, int(seconds / 0.25)))],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        out, _ = proc.communicate(timeout=seconds + 5)
    except (OSError, subprocess.SubprocessError):
        return None
    samples = []
    for line in (out or "").splitlines():
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not samples:
        return None
    def avg(key: str) -> float | None:
        vals = [float(s[key]) for s in samples if isinstance(s.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None
    return {"cpu_w": avg("cpu_power"), "gpu_w": avg("gpu_power"), "ane_w": avg("ane_power"), "ram_w": avg("ram_power")}


def _path() -> pathlib.Path:
    return paths.assistant_dir() / "models" / "probe.json"


def load(hw: hardware.Hardware) -> dict | None:
    """Return cached Tier 0 hardware + ANE probe for this exact machine and macOS build."""
    try:
        data = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != PROBE_VERSION or data.get("machine") != bench.fingerprint(hw):
        return None
    if sys.platform == "darwin" and data.get("macos_build") and data.get("macos_build") != macos_build():
        return None
    return data


def save(record: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)


def apply(hw: hardware.Hardware, probed: dict | None) -> hardware.Hardware:
    """Apply measured Tier 0 bandwidth, Metal working set, prefill compute, and ANE results to Hardware."""
    if not probed or probed.get("noisy"):
        return hw
    updates: dict = {}
    bw = float(probed.get("bandwidth_gbs") or 0)
    if hw.gpu == "apple" and bw > 20.0:
        updates["bandwidth_gbs"] = round(bw, 1)
    ws = float(probed.get("recommended_max_working_set_gb") or 0)
    if hw.gpu == "apple" and ws > 2.0 and not hw.measured:
        # Respect an explicit raised iogpu.wired_limit_mb when larger than default working set.
        if ws > hw.gpu_limit_gb or abs(ws - hw.gpu_limit_gb) / max(hw.gpu_limit_gb, 1.0) < 0.25:
            updates["gpu_limit_gb"] = round(ws, 2)
            updates["vram_gb"] = round(ws, 2)
            updates["budget_gb"] = round(hardware.apple_budget(hw.ram_gb, ws, hw.sandbox_memory_gb), 2)
    tflops = float(probed.get("gpu_fp16_tflops") or 0)
    if hw.gpu == "apple" and tflops > 0.5 and not hw.read_k:
        # Empirically calibrated: ~140-165 tok/s*B per effective FP16 TFLOP on llama.cpp Metal prefill.
        updates["read_k"] = round(tflops * 150.0, 1)
    if probed.get("ane_cores") is not None:
        updates["ane_cores"] = int(probed.get("ane_cores") or 0)
    if probed.get("ane_fp16_tops"):
        updates["ane_tops"] = float(probed["ane_fp16_tops"])
    if probed.get("ane_verified") is not None:
        updates["ane_verified"] = bool(probed["ane_verified"])
    if probed.get("gpu_contention_pct") is not None:
        updates["gpu_contention_pct"] = float(probed["gpu_contention_pct"])
    return dataclasses.replace(hw, **updates) if updates else hw


def run(hw: hardware.Hardware | None = None, *, include_cdn: bool = True, force: bool = False,
        progress: Callable[[str], None] | None = None) -> dict:
    """Run Tier 0 bare-metal Metal + ANE probe on macOS (or return cached clean measurement)."""
    hw = hw or hardware.detect()
    if not force:
        cached = load(hw)
        if cached and not cached.get("noisy"):
            return cached
    if sys.platform != "darwin" or hw.gpu != "apple":
        return {"ok": False, "supported": False, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "date": datetime.date.today().isoformat(), "error": "Tier 0 Metal/ANE probe requires Apple silicon"}
    if progress:
        progress("measuring GPU bandwidth, Metal compute and Neural Engine")
    if not SOURCE.is_file():
        return {"ok": False, "supported": False, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "date": datetime.date.today().isoformat(),
                "error": "Tier 0 probe source (probe.swift) is missing from this installation; using chip specifications"}
    binary = native.build(SOURCE, HELPER_NAME)
    if not binary:
        return {"ok": False, "supported": False, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "date": datetime.date.today().isoformat(),
                "error": "Xcode Command Line Tools (swiftc) not installed; using chip specifications"}
    build_id = macos_build()
    cache_root = paths.assistant_dir() / "cache" / "ane-probe"
    model_file = build_ane_probe_mlmodel(cache_root / f"ane_conv512_l8_v{PROBE_VERSION}.mlmodel",
                                         channels=512, width=512, layers=8)
    compiled_cache = cache_root / f"ane_conv512_l8_v{PROBE_VERSION}-{build_id or 'mac'}.mlmodelc"
    try:
        done = subprocess.run([binary, "--ane-model", str(model_file), "--compiled-cache", str(compiled_cache),
                               "--channels", "512", "--width", "512", "--layers", "8"],
                              capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "supported": True, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "error": f"probe timed out or failed: {exc}"}
    if done.returncode != 0 or not done.stdout.strip():
        return {"ok": False, "supported": True, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "error": (done.stderr or done.stdout or "probe failed")[-300:]}
    try:
        measured = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "supported": True, "version": PROBE_VERSION, "machine": bench.fingerprint(hw),
                "error": f"probe returned invalid JSON: {exc}"}
    cdn_mb_s = probe_hf_cdn_mb_s() if include_cdn else None
    power = sample_macmon(1.0)
    contention = measured.get("gpu_contention_pct")
    record = {
        "ok": True,
        "supported": True,
        "version": PROBE_VERSION,
        "date": datetime.date.today().isoformat(),
        "macos_build": build_id,
        "machine": bench.fingerprint(hw),
        **measured,
        "hf_cdn_mb_s": cdn_mb_s,
        "yield_during_llm_turn": bool(contention is not None and contention > CONTENTION_YIELD_PCT),
        **({"power": power} if power else {}),
    }
    # Never overwrite a clean cached measurement with a thermally throttled or Low Power Mode run.
    existing = load(hw)
    if not record.get("noisy") or not existing or existing.get("noisy"):
        save(record)
    return record
