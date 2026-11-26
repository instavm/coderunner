import asyncio
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import httpx

from coderunner.core import paths
from coderunner.core.config import Config
from coderunner.safety import decision as decision_module
from coderunner.models import bench, catalog, evidence, fit, hardware, licenses, manager, runtimes, smoketest
from coderunner.models import hub as hub_module
from coderunner.models.hub import Hub, HubError

GiB = 1024 ** 3
REAL_MEASURE = bench.measure


async def no_measure(*args, **kwargs):
    raise ConnectionError("not timed in this test")


class TempHome(unittest.TestCase):
    """Keeps every test away from the real ~/.coderunner and from timing real servers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home), "HF_TOKEN": "",
                                           "HF_HOME": str(self.home / "hf")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)
        timing = mock.patch.object(bench, "measure", no_measure)       # setup times models through bench
        timing.start()
        self.addCleanup(timing.stop)
        # Chat models on CodeRunner's llama.cpp is Apple silicon's default, and would install llama.cpp from the
        # network; tests about it turn it on with fake runtimes.
        llama = mock.patch.object(manager, "_chat_on_llama", return_value=False)
        llama.start()
        self.addCleanup(llama.stop)
        # Setup fetches its own Ollama where none is installed, and skips downloads the disk has no room for:
        # tests about either turn it on, so the others neither download nor depend on this machine's disk.
        # ...nor on what happens to be open on it: free memory is unknown unless a test says how much there is.
        from coderunner.models import coreml
        for target, name, value in ((runtimes.OllamaRuntime, "can_install", False), (manager, "_free_gb", 1e6),
                                    (hardware, "available_gb", None), (coreml, "supported", False)):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def config(self, **models):
        # Tests never reach Hugging Face; discovery has its own tests with a fake hub.
        return Config({"models": {"discover": False, **models}}, path=self.home / "config.json")


def fake_run(outputs):
    def run(cmd):
        return outputs.get(cmd[0] if cmd[0] != "sysctl" else cmd[-1])
    return run


def apple(ram_gb, chip="Apple M3 Pro", wired_mb=0, sandbox=4):
    run = fake_run({"machdep.cpu.brand_string": chip + "\n", "hw.memsize": f"{int(ram_gb * GiB)}\n",
                    "iogpu.wired_limit_mb": f"{wired_mb}\n"})
    return hardware.detect(sandbox, run=run, system="Darwin", machine="arm64")


def linux(ram_gb, outputs=None, wsl=False):
    files = {"/proc/meminfo": f"MemTotal:       {int(ram_gb * 1024 * 1024)} kB\n",
             "/proc/version": "Linux version 6.6.87.2-microsoft-standard-WSL2" if wsl else "Linux version 6.8.0-generic",
             "/proc/cpuinfo": "model name\t: AMD Ryzen 9 7950X\n"}
    return hardware.detect(4, run=fake_run(outputs or {}), read=lambda p: files.get(p, ""),
                           system="Linux", machine="x86_64")


def nvidia(vram_gb, free_gb=None, name="NVIDIA GeForce RTX 4090"):
    free = free_gb if free_gb is not None else vram_gb - 0.5
    return linux(64, {"nvidia-smi": f"{name}, {int(vram_gb * 1024)}, {int(free * 1024)}\n"})


# ---- hardware -----------------------------------------------------------------

class HardwareTests(unittest.TestCase):
    def test_free_memory_right_now_is_read_from_the_kernel(self):
        answers = {"hw.memsize": str(18 * 1024 ** 3), "kern.memorystatus_level": "34"}
        self.assertAlmostEqual(hardware.available_gb(run=lambda cmd: answers[cmd[-1]], system="Darwin"), 6.12, places=2)
        self.assertAlmostEqual(hardware.available_gb(24, run=lambda cmd: "50", system="Darwin"), 12.0)
        self.assertIsNone(hardware.available_gb(18, run=lambda cmd: "", system="Darwin"))
        meminfo = "MemTotal:       32000000 kB\nMemAvailable:   12582912 kB\n"
        self.assertAlmostEqual(hardware.available_gb(read=lambda path: meminfo, system="Linux"), 12.0)
        self.assertIsNone(hardware.available_gb(read=lambda path: "MemTotal: 1 kB\n", system="Linux"))
        self.assertIsNone(hardware.available_gb(system="Windows"))

    def test_apple_parsing_and_budget(self):
        hw = apple(18)
        self.assertEqual((hw.gpu, hw.chip, hw.bandwidth_gbs), ("apple", "Apple M3 Pro", 150))
        self.assertAlmostEqual(hw.ram_gb, 18)
        self.assertAlmostEqual(hw.gpu_limit_gb, 12.0)            # no wired limit set: ~2/3 of RAM on <=24 GB
        self.assertAlmostEqual(hw.budget_gb, hardware.apple_budget(18, 12.0, 4))
        self.assertEqual(hw.to_dict()["tier"]["tier_gb"], 18)

    def test_apple_wired_limit_caps_budget(self):
        hw = apple(128, "Apple M4 Max", wired_mb=64 * 1024)
        self.assertAlmostEqual(hw.gpu_limit_gb, 64)
        self.assertAlmostEqual(hw.budget_gb, 64)
        self.assertEqual(hw.bandwidth_gbs, 546)

    def test_apple_bandwidth_table(self):
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Max", 36), 300)   # binned
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Max", 96, gpu_cores=30), 300)  # binned 96 GB M3 Max
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Max", 64), 400)
        self.assertEqual(hardware.apple_bandwidth("Apple M2 Ultra", 192), 800)
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Ultra", 256), 819)             # exact Ultra lookup
        self.assertEqual(hardware.apple_bandwidth("Apple M1", 8), 68)
        self.assertEqual(hardware.apple_bandwidth("Apple M5", 16), 153)
        self.assertEqual(hardware.apple_bandwidth("Apple M5 Pro", 24), 306)  # scaled until listed
        self.assertEqual(hardware.apple_bandwidth("Apple M9", 16), 100)

    def test_small_mac_has_almost_no_budget(self):
        self.assertAlmostEqual(apple(8, "Apple M1").budget_gb, hardware.apple_budget(8, hardware.default_apple_gpu_limit(8), 4))
        self.assertGreater(hardware.apple_budget(8, hardware.default_apple_gpu_limit(8), 1), 4.0)   # lazy 1 GB VM fits 4B/9B
        self.assertEqual(hardware.apple_budget(4, 3, 4), 0.0)

    def test_ram_tiers_and_sublinear_reserve_from_8gb_to_512gb(self):
        for gb in (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512):
            tier = hardware.ram_tier(gb)
            self.assertEqual(tier["tier_gb"], gb)
        self.assertLessEqual(hardware.os_app_reserve_gb(512), 16.0)
        self.assertGreater(hardware.safe_wired_limit_gb(256, 8), 220.0)
        self.assertGreater(hardware.safe_wired_limit_gb(512, 8), 470.0)
        # Live memory parses vm_stat + swapusage cleanly.
        fake_outputs = {
            "hw.memsize": str(18 * GiB),
            "kern.memorystatus_level": "40",
            "vm.swapusage": "total = 4096.00M  used = 2816.00M  free = 1280.00M",
            "vm_stat": "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 50000.\nPages inactive: 100000.\nPages speculative: 1000.\nPages purgeable: 5000.\nPages wired down: 200000.\n",
        }
        mem = hardware.live_memory(18, run=lambda cmd: fake_outputs.get(cmd[-1] if cmd[0] == "sysctl" else cmd[0]), system="Darwin")
        self.assertAlmostEqual(mem["swap_used_gb"], 2.75, places=2)
        self.assertEqual(mem["pressure"], "high")
        self.assertGreater(mem["reclaimable_gb"], 2.0)

    def test_nvidia(self):
        hw = nvidia(24)
        self.assertEqual((hw.gpu, hw.chip, hw.bandwidth_gbs, hw.gpus, hw.unified),
                         ("nvidia", "NVIDIA GeForce RTX 4090", 1008, 1, False))
        self.assertAlmostEqual(hw.vram_gb, 24)
        self.assertAlmostEqual(hw.budget_gb, 23.5 - 1)            # free VRAM - 1 GiB per card

    def test_several_nvidia_cards_are_pooled(self):
        # llama.cpp and Ollama split a model across cards; the slowest card sets the pace.
        out = "NVIDIA GeForce RTX 3060, 12288, 11000, 8.6\nNVIDIA GeForce RTX 4090, 24564, 23900, 8.9\n"
        hw = linux(64, {"nvidia-smi": out})
        self.assertEqual((hw.gpus, hw.bandwidth_gbs, hw.compute_cap), (2, 360, 8.6))
        self.assertAlmostEqual(hw.vram_gb, (12288 + 24564) / 1024)
        self.assertAlmostEqual(hw.budget_gb, (11000 + 23900) / 1024 - 2)
        four = linux(512, {"nvidia-smi": "NVIDIA H200, 143771, 143000, 9.0\n" * 4})
        self.assertEqual((four.gpus, four.bandwidth_gbs), (4, 4800))
        self.assertGreater(four.budget_gb, 550)

    def test_unified_memory_nvidia_systems(self):
        # DGX Spark reports no dedicated VRAM: the model shares the 128 GB pool with the OS and sandbox.
        spark = linux(128, {"nvidia-smi": "NVIDIA GB10, [N/A], [N/A], 12.1\n"})
        self.assertEqual((spark.gpu, spark.unified, spark.bandwidth_gbs, spark.compute_cap), ("nvidia", True, 273, 12.1))
        self.assertAlmostEqual(spark.budget_gb, hardware.unified_budget(128, 4))
        self.assertGreater(spark.budget_gb, 100)
        thor = linux(128, {"nvidia-smi": "NVIDIA Thor, [N/A], [N/A], 11.0\n"})
        self.assertTrue(thor.unified)
        self.assertEqual(hardware.nvidia_bandwidth("NVIDIA GeForce RTX 4070 Ti SUPER"), 504)

    def test_amd(self):
        out = json.dumps({"card0": {"VRAM Total Memory (B)": str(16 * GiB), "VRAM Total Used Memory (B)": str(GiB)}})
        hw = linux(32, {"rocm-smi": out})
        self.assertEqual(hw.gpu, "amd")
        self.assertAlmostEqual(hw.budget_gb, 14)

    def test_cpu_only(self):
        hw = linux(32)
        self.assertEqual((hw.gpu, hw.wsl, hw.chip), ("none", False, "AMD Ryzen 9 7950X"))
        self.assertAlmostEqual(hw.budget_gb, min(0.8 * 32, 30))
        self.assertAlmostEqual(linux(8).budget_gb, 6)              # RAM - 2 GB wins on small machines

    def test_wsl2_uses_vm_memory(self):
        hw = linux(16, wsl=True)
        self.assertTrue(hw.wsl)
        self.assertAlmostEqual(hw.ram_gb, 16)
        self.assertAlmostEqual(hw.budget_gb, 12.8)
        self.assertTrue(linux(16, {"nvidia-smi": "NVIDIA GeForce RTX 4070, 12282, 11800\n"}, wsl=True).gpu == "nvidia")


# ---- sizing and recommendations ---------------------------------------------------

class FitTests(TempHome):
    def test_kv_cache_math(self):
        q9 = catalog.get("qwen3.5-9b")
        # 2 (K,V) * 8 full-attention layers * 4 kv heads * 256 dims * 32768 tokens * 2 bytes = 1 GiB
        self.assertAlmostEqual(fit.kv_gib(q9, 32768), 1.0)
        self.assertAlmostEqual(fit.kv_gib(q9, 16384), 0.5)
        g31 = catalog.get("gemma-4-31b")
        swa = 2 * 50 * 16 * 256 * 1024 * 2 / GiB                   # sliding layers stop at the window
        self.assertAlmostEqual(fit.kv_gib(g31, 65536) - fit.kv_gib(g31, 32768), 2 * 10 * 4 * 512 * 32768 * 2 / GiB)
        self.assertAlmostEqual(fit.kv_gib(g31, 1024), 2 * 10 * 4 * 512 * 1024 * 2 / GiB + swa)
        unknown = {"params_b": 8}
        self.assertAlmostEqual(fit.kv_gib(unknown, 8192), 8192 * 131072 / GiB)

    def test_need_fit_and_speed(self):
        hw = apple(18)
        est = fit.estimate(catalog.get("qwen3.5-9b"), hw, 32768, "llama.cpp")
        need = (5.68 + 0.92) * 1e9 / GiB + 1.0 + fit.OVERHEAD_GIB
        self.assertAlmostEqual(est["need_gb"], round(need, 2))
        self.assertEqual(est["fit"], "good")                        # 8.15 <= 0.85 * 9.7
        self.assertAlmostEqual(est["tok_s"], round(0.55 * 150 / 5.68, 1))
        moe = fit.estimate(catalog.get("qwen3.6-35b-a3b"), hw, 32768, "llama.cpp")
        self.assertEqual(moe["fit"], "no")
        self.assertAlmostEqual(moe["tok_s"], round(0.55 * 150 / (20.42 * 3 / 35.95), 1))
        tight = hardware.Hardware("darwin", "arm64", budget_gb=need / 0.9, bandwidth_gbs=100)
        self.assertEqual(fit.estimate(catalog.get("qwen3.5-9b"), tight, 32768, "llama.cpp")["fit"], "tight")

    def test_context_drops_before_tier(self):
        entry = catalog.get("qwen3.5-9b")
        need16 = fit.estimate(entry, hardware.Hardware("darwin", "arm64", budget_gb=100), 16384, "llama.cpp")["need_gb"]
        hw = hardware.Hardware("darwin", "arm64", budget_gb=need16 / 0.85 + 0.01, bandwidth_gbs=200)
        best = fit.recommend(hw, self.config(), runtime="llama.cpp")[0]
        self.assertEqual((best["id"], best["ctx"], best["fit"]), ("qwen3.5-9b", 16384, "good"))

    def assertSensible(self, picks):
        for c in picks:
            self.assertGreaterEqual(c["params_b"], 4)
            self.assertIn("chat", catalog.get(c["id"])["roles"])
            self.assertNotEqual(c["fit"], "no")

    def test_apple_recommendations(self):
        cfg = self.config()
        # Verified Nov 2026 SOTA across Apple Silicon RAM tiers (8 GB through 512 GB):
        # Bonsai 2.5 27B leads 18-24 GB; Inkling-2-Mini & Qwen3.8-27B lead 32-64 GB; Ornith-1.5-122B leads 128 GB;
        # MiMo-V2.6-Flash-RL leads 256 GB; DeepSeek-V4.1-Flash-1017 & Inkling-2-Small lead 512 GB.
        expected = {
            "ollama": {16: "qwen3.5-9b", 18: "qwen3.5-9b", 24: "qwen3.5-9b", 64: "qwen3.8-27b", 128: "ornith-1.5-122b"},
            "llama.cpp": {16: "qwen3.5-9b", 18: "bonsai-2.5-27b", 24: "bonsai-2.5-27b", 64: "bonsai-2.5-27b",
                          128: "ornith-1.5-122b", 256: "mimo-v2.6-flash-rl", 512: "mimo-v2.6-flash-rl"},
        }
        chips = {8: "Apple M1", 16: "Apple M2", 18: "Apple M3 Pro", 24: "Apple M4 Pro", 32: "Apple M2 Pro",
                 64: "Apple M1 Max", 128: "Apple M4 Max", 256: "Apple M3 Ultra", 512: "Apple M3 Ultra"}
        for ram, chip in chips.items():
            for runtime in ("ollama", "llama.cpp"):
                with self.subTest(ram=ram, runtime=runtime):
                    picks = fit.recommend(apple(ram, chip), cfg, runtime=runtime)
                    self.assertSensible(picks)
                    if ram == 8:
                        self.assertEqual(picks, [])                 # 4 GB full VM + OS leave no room for a 4B model
                        continue
                    if ram in expected[runtime]:
                        self.assertEqual(picks[0]["id"], expected[runtime][ram])
                    self.assertTrue(picks[0]["vision"])
                    self.assertEqual(picks[0]["fit"], "good")
        # Across runtimes, Bonsai 2.5 27B leads 18-24 GB Macs with a 1-2 GB sandbox, and Qwen3.8-27B /
        # Bonsai 2.5 27B / Inkling-2-Mini sit at the top of 32-64 GB Macs.
        two = fit.recommend(apple(16, "Apple M2", sandbox=1), cfg)
        self.assertIn(two[0]["id"], ("bonsai-2.5-27b", "bonsai-2.5-27b-compact", "qwen3.5-9b"))
        big = fit.recommend(apple(64, "Apple M1 Max"), cfg, runtime="llama.cpp")
        self.assertIn("qwen3.8-27b", [c["id"] for c in big[:5]])
        self.assertIn("inkling-2-mini", [c["id"] for c in big[:5]])
        # Presets map 64 GB to Fast (Inkling-2-Mini) / Balanced / Smartest (Qwen3.8-27B).
        tier_presets = {p["key"]: p for p in fit.presets(apple(64, "Apple M1 Max"), big)}
        self.assertEqual(tier_presets["fast"]["model"], "inkling-2-mini")
        self.assertIn(tier_presets["smartest"]["model"], ("qwen3.8-27b", "qwen3.8-27b-q6"))
        for ram, chip in ((18, "Apple M3 Pro"), (24, "Apple M4 Pro")):
            with self.subTest(ram=ram):
                picks = fit.recommend(apple(ram, chip, sandbox=1), cfg)
                self.assertEqual((picks[0]["id"], picks[0]["vision"]), ("bonsai-2.5-27b", True))
                self.assertIn("prism", catalog.get(picks[0]["id"])["artifacts"])
                nine = next(c for c in picks if c["id"] == "qwen3.5-9b")
                self.assertGreater(picks[0]["ability"], nine["ability"] + 0.2)
                self.assertGreater(picks[0]["step_s"], nine["step_s"])
        # 512 GB M3 Ultra fits DeepSeek-V4.1-Flash-1017, Inkling-2-Small, and MiMo-V2.6-Flash-RL cleanly.
        ultra512 = [c["id"] for c in fit.recommend(apple(512, "Apple M3 Ultra", sandbox=8), cfg, runtime="llama.cpp")[:4]]
        self.assertIn("deepseek-v4.1-flash-1017", ultra512)
        self.assertIn("inkling-2-small", ultra512)
        # The 1-bit 27B fits too, but its published scores keep it below Qwen3.5-9B.
        picks18 = fit.recommend(apple(18), cfg)
        onebit = next(c for c in picks18 if c["id"] == "bonsai-27b-1bit")
        self.assertLess(onebit["ability"], next(c for c in picks18 if c["id"] == "qwen3.5-9b")["ability"])
        self.assertIn(f"benchmark ability {onebit['ability']:+.2f}", onebit["reasons"])
        self.assertIn(fit.recommend(apple(16, "Apple M1", sandbox=1), cfg)[0]["id"],
                      ("bonsai-2.5-27b-compact", "qwen3.5-9b"))
        compact = fit.candidate(catalog.get("bonsai-2-27b-compact"), apple(18), cfg, "chat", "prism")
        self.assertLess(compact["tok_s"], fit.MIN_TOK_S)        # measured factor applied
        self.assertEqual(fit.recommend(apple(18), cfg, runtime="ollama")[0]["id"], "qwen3.5-9b")
        order = [c["id"] for c in fit.recommend(apple(32, "Apple M2 Pro"), cfg, runtime="llama.cpp")]
        self.assertEqual(order[:2], ["inkling-2-mini", "bonsai-2.5-27b"])

    def test_nvidia_recommendations(self):
        cfg = self.config()
        rtx4070 = nvidia(12, name="NVIDIA GeForce RTX 4070")
        picks12 = fit.recommend(rtx4070, cfg, runtime="llama.cpp")
        self.assertSensible(picks12)
        # With 64 GB of RAM, a mixture-of-experts model keeps its experts in system RAM (llama.cpp --cpu-moe).
        self.assertEqual((picks12[0]["id"], picks12[0]["offload"]), ("inkling-2-mini", True))
        self.assertIn("experts in system RAM", " ".join(picks12[0]["reasons"]))
        self.assertLess(picks12[0]["need_gb"], rtx4070.budget_gb)
        self.assertGreater(picks12[0]["tok_s"], fit.MIN_TOK_S)
        # Ollama cannot place experts in RAM, and a machine without spare RAM cannot either.
        self.assertEqual(fit.recommend(rtx4070, cfg, runtime="ollama")[0]["id"], "qwen3.5-9b")
        small = linux(16, {"nvidia-smi": "NVIDIA GeForce RTX 4070, 12288, 11776\n"})
        self.assertEqual(fit.recommend(small, cfg, runtime="llama.cpp")[0]["id"], "qwen3.5-9b")
        spark = linux(128, {"nvidia-smi": "NVIDIA GB10, [N/A], [N/A], 12.1\n"})
        self.assertFalse(any(c.get("offload") for c in fit.recommend(spark, cfg, runtime="llama.cpp")))
        # On 24 GB, Qwen3.8-27B fits only tightly, so Bonsai 2.5 / 9B that fit well come first.
        picks24 = fit.recommend(nvidia(24), cfg, runtime="llama.cpp")
        self.assertSensible(picks24)
        self.assertIn(picks24[0]["id"], ("bonsai-2.5-27b", "qwen3.5-9b"))
        self.assertEqual(next(c for c in picks24 if c["id"] == "qwen3.8-27b")["fit"], "tight")

    def test_eight_gb_macs_run_a_compact_model(self):
        for chip in ("Apple M1", "Apple M3"):
            with self.subTest(chip=chip):
                picks = fit.recommend(apple(8, chip, sandbox=1), self.config())
                self.assertTrue(picks)
                self.assertIn(picks[0]["id"], ("bonsai-2-4b", "bonsai-2-9b", "qwen3.5-4b-compact"))
                # Under a tighter 1.5 GB sandbox budget, qwen3.5-4b-compact drops its vision projector to still fit.
                tight_picks = fit.recommend(apple(8, chip, sandbox=1.5), self.config())
                compact = next(c for c in tight_picks if c["id"] == "qwen3.5-4b-compact")
                self.assertFalse(compact["vision"])
                self.assertIn("text only: the vision projector does not fit", compact["reasons"])

    def test_small_models_never_chat(self):
        rejected = fit.recommend(apple(128, "Apple M4 Max"), self.config(), include_rejected=True)
        self.assertNotIn("qwen3.5-0.8b", [c["id"] for c in rejected])
        self.assertEqual(fit.recommend(apple(18), self.config(), role="guard")[0]["id"], "qwen3.5-0.8b")
        # One multimodal model for memory and files where the runtime can serve it.
        self.assertEqual(fit.recommend(apple(18), self.config(), role="embedding", runtime="ollama")[0]["id"],
                         "embeddinggemma-2")

    def test_license_policy_applies_to_recommendations(self):
        cfg = self.config(allow_licenses=[])
        for c in fit.recommend(apple(18), cfg, role="embedding", include_rejected=True):
            if c["id"] == "embeddinggemma-300m":
                self.assertEqual(c["license"], "ask")
        picks = fit.recommend(apple(18), cfg, role="embedding")
        self.assertEqual(picks[0]["id"], "embeddinggemma-2")         # permissive licenses rank first
        self.assertEqual(picks[1]["id"], "qwen3-embedding-0.6b")

    def test_catalog_is_well_formed(self):
        ids = set()
        for e in catalog.models():
            self.assertNotIn(e["id"], ids)
            ids.add(e["id"])
            self.assertTrue(e["artifacts"])
            self.assertRegex(e.get("verified", ""), r"^2026-\d\d-\d\d$")
            self.assertIn(licenses.classify(e["license"]["spdx"]), ("allow", "ask"))
            if "chat" in e["roles"]:
                self.assertGreaterEqual(e["params_b"], 4)
                # Ternary models need Prism's build; everything else ships for Ollama and llama.cpp.
                if "prism" in e["artifacts"]:
                    self.assertTrue({"sha256", "file", "repo"} <= set(e["artifacts"]["prism"]))
                else:
                    self.assertIn("gguf", e["artifacts"])


# ---- licenses and hub ------------------------------------------------------------

def hf_transport(routes, calls=None):
    def handler(request: httpx.Request):
        if calls is not None:
            calls.append(request)
        key = request.url.path
        if key not in routes:
            return httpx.Response(404, json={"error": "not found"})
        value = routes[key]
        return value(request) if callable(value) else httpx.Response(200, json=value)
    return httpx.MockTransport(handler)


class LicenseTests(TempHome):
    def test_classify(self):
        for spdx in ("apache-2.0", "MIT", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense"):
            self.assertEqual(licenses.classify(spdx), "allow", spdx)
        for spdx, name in (("gemma", None), ("llama3.1", None), ("llama4", None), ("openrail++", None),
                           ("creativeml-openrail-m", None), ("other", "qwen"), ("nvidia-open-model-license", None)):
            self.assertEqual(licenses.classify(spdx, name), "ask", spdx)
        for spdx, name in (("cc-by-nc-4.0", None), ("cc-by-nc-sa-4.0", None), ("other", "qwen-research"),
                           ("apple-amlr", None), ("other", None), (None, None), ("", None), ("weird-1.0", None)):
            self.assertEqual(licenses.classify(spdx, name), "block", (spdx, name))

    def test_check_and_acceptance(self):
        entry = {"id": "x", "license": {"spdx": "gemma", "link": "https://ai.google.dev/gemma/terms"}}
        self.assertEqual(licenses.check(entry, self.config())[0], "ask")
        self.assertEqual(licenses.check(entry, self.config(accepted_licenses=["gemma"]))[0], "allow")
        self.assertEqual(licenses.check(entry, self.config(allow_licenses=["gemma"]))[0], "allow")
        nc = {"id": "y", "license": {"spdx": "cc-by-nc-4.0"}}
        decision, reason, _ = licenses.check(nc, self.config(accepted_licenses=["cc-by-nc-4.0"]))
        self.assertEqual(decision, "block")
        self.assertIn("commercial", reason)
        other = {"id": "z", "license": {"spdx": "other", "name": "deepseek"}}
        self.assertEqual(licenses.check(other, self.config(accepted_licenses=["deepseek"]))[0], "allow")
        self.assertEqual(licenses.key("other", "DeepSeek"), "deepseek")

    def test_mirror_without_license_resolves_through_base_model(self):
        calls = []
        routes = {
            "/api/models/someone/Model-GGUF": {"id": "someone/Model-GGUF", "cardData": {},
                                               "tags": ["gguf", "base_model:quantized:org/Model"], "gated": False},
            "/api/models/org/Model": {"id": "org/Model", "gated": "manual",
                                      "cardData": {"license": "gemma", "license_link": "LICENSE.md"}, "tags": []},
            "/api/models/mirror/NoLicense": {"id": "mirror/NoLicense", "cardData": {}, "tags": []},
            "/api/models/mirror/Other": {"id": "mirror/Other", "cardData": {"base_model": ["org/Apache"]}, "tags": []},
            "/api/models/org/Apache": {"cardData": {"license": "apache-2.0"}, "tags": ["license:apache-2.0"]},
        }
        hub = Hub(httpx.Client(transport=hf_transport(routes, calls)), cache_dir=self.home / "cache")
        info = hub.resolve_license("someone/Model-GGUF")
        self.assertEqual((info["spdx"], info["repo"], info["gated"]), ("gemma", "org/Model", False))
        self.assertEqual(info["link"], f"{hub_module.endpoint()}/org/Model/blob/main/LICENSE.md")
        self.assertEqual(licenses.check("someone/Model-GGUF", self.config(), hub)[0], "ask")
        self.assertEqual(licenses.check("mirror/Other", self.config(), hub)[0], "allow")
        decision, reason, _ = licenses.check("mirror/NoLicense", self.config(), hub)
        self.assertEqual(decision, "block")
        self.assertIn("no license", reason)
        self.assertEqual(licenses.check("mirror/NoLicense", self.config(accepted_licenses=["mirror/nolicense"]), hub)[0],
                         "allow")
        before = len(calls)
        hub.resolve_license("someone/Model-GGUF")                    # cached on disk
        self.assertEqual(len(calls), before)

    def test_gated_repo_mentions_token(self):
        routes = {"/api/models/org/Gated": {"gated": "manual", "cardData": {"license": "llama3.1"}, "tags": []}}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        decision, reason, _ = licenses.check("org/Gated", self.config(), hub)
        self.assertEqual(decision, "ask")
        self.assertIn("HF_TOKEN", reason)


class HubTests(TempHome):
    def test_split_gguf_manifest(self):
        seen_agents = []

        def manifest(request):
            seen_agents.append(request.headers["user-agent"])
            return httpx.Response(200, json={
                "ggufFile": {"rfilename": "Q4_K_M/M-Q4_K_M-00001-of-00003.gguf", "size": 10,
                             "lfs": {"sha256": "a" * 64}},
                "mmprojFile": {"rfilename": "mmproj-F16.gguf", "size": 5, "lfs": {"sha256": "m" * 64}}})
        siblings = [{"rfilename": f"Q4_K_M/M-Q4_K_M-0000{i}-of-00003.gguf", "size": i * 100,
                     "lfs": {"sha256": str(i) * 64}} for i in (1, 2, 3)]
        routes = {"/v2/org/M-GGUF/manifests/Q4_K_M": manifest,
                  "/api/models/org/M-GGUF": {"sha": "abc123", "siblings": siblings}}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        files = hub.gguf_files("org/M-GGUF", "Q4_K_M")
        self.assertEqual(seen_agents, ["llama-cpp"])
        self.assertEqual([f["file"] for f in files], [s["rfilename"] for s in siblings] + ["mmproj-F16.gguf"])
        self.assertEqual([f["sha256"][0] for f in files], ["1", "2", "3", "m"])
        self.assertEqual(files[-1]["kind"], "mmproj")
        self.assertEqual(len(hub.gguf_files("org/M-GGUF", "Q4_K_M", vision=False)), 3)

    def test_a_rate_limit_says_when_to_try_again_and_yesterdays_answer_still_serves(self):
        limited = {"/api/models/org/M": lambda request: httpx.Response(
            429, headers={"ratelimit": '"api";r=0;t=252', "ratelimit-policy": '"fixed window";"api";q=500;w=300'},
            json={"error": "We had to rate limit your IP"})}
        hub = Hub(httpx.Client(transport=hf_transport(limited)), cache_dir=self.home / "cache")
        with self.assertRaises(hub_module.RateLimited) as raised:
            hub.model_info("org/M")
        self.assertEqual(raised.exception.seconds, 252)
        self.assertIn("try again in 4 min, or set HF_TOKEN", str(raised.exception))
        stale = self.home / "cache" / "info-org_M-main.json"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(json.dumps({"sha": "from yesterday"}))
        os.utime(stale, (1, 1))                                       # far older than the cache's day
        self.assertEqual(hub.model_info("org/M"), {"sha": "from yesterday"})

    def test_a_model_named_by_its_address_is_fetched_checked_and_served_with_its_arguments(self):
        # Bonsai 2 27B for llama.cpp itself: one file from Ollama's registry by address and checksum, the vision
        # projector from the maker's repository. Setup from the app stopped at once on this entry (KeyError:
        # 'repo'), which a run through the API with another model had not touched.
        entry = catalog.get("bonsai-2-27b")
        art = entry["artifacts"]["gguf"]
        self.assertTrue(art["url"].endswith(art["sha256"]) and "repo" not in art)
        fetched = []

        class Registry:
            def fetch(self, url, dest, sha256, size=None, progress=None, parallel=None):
                fetched.append((url, sha256, size, parallel))
                pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(dest).write_bytes(b"gguf")
                return pathlib.Path(dest)

            def download(self, repo, file, dest, revision="main", sha256=None, size=None, progress=None):
                fetched.append((repo, file, sha256))
                pathlib.Path(dest).write_bytes(b"mmproj")
                return pathlib.Path(dest)

            def resolve_license(self, repo, depth=3):
                raise AssertionError("a build without a repository has no licence to look up there")

        rt = runtimes.LlamaCppRuntime(self.config(), hub=Registry())
        with mock.patch.object(rt, "binary", return_value="/usr/bin/true"), \
                mock.patch.object(rt, "start"), mock.patch.object(rt, "stop"), \
                mock.patch.object(manager, "_prism", return_value=None):
            result = manager.setup(self.config(), model_id="bonsai-2-27b", runtime=rt, hw=apple(18, sandbox=1),
                                   hub=Registry(), progress=lambda m: None, embedding=False, decision=False,
                                   smoke=fake_smoke({}), measure=False)
        self.assertTrue(result["ok"], result)
        self.assertEqual(fetched[0], (art["url"], art["sha256"], art["bytes"], art["streams"]))
        self.assertEqual(fetched[1][:2], (art["mmproj_repo"], art["mmproj"]))
        active = rt._state()["active"]
        self.assertEqual((active["id"], active["args"]), ("bonsai-2-27b", art["server_args"]))
        command = rt.command(active)
        self.assertEqual(command[-len(art["server_args"]):], art["server_args"])
        self.assertIn("--mmproj", command)
        # Only where it was run: an NVIDIA machine is not offered this build.
        self.assertFalse(fit.candidate(entry, nvidia(24), self.config(), "chat", "llama.cpp")["ok"])
        self.assertTrue(fit.candidate(entry, apple(24, "Apple M4 Pro", sandbox=1), self.config(), "chat", "llama.cpp")["ok"])

    def test_a_server_that_does_not_start_keeps_the_download(self):
        class WillNotStart(FakeLlama):
            def start(self, timeout=60):
                raise RuntimeError("llama-server exited with code 1")
        rt, said = WillNotStart(), []
        result = manager.setup(self.config(), model_id="qwen3.5-9b", runtime=rt, hw=apple(18), progress=said.append,
                               embedding=False, decision=False, smoke=fake_smoke({}), hub=ApacheHub())
        self.assertFalse(result["ok"])
        self.assertIn("llama-server exited with code 1", result["error"])
        self.assertEqual((rt.pulled, rt.removed), (["qwen3.5-9b"], []))          # fetched, and not thrown away
        self.assertIn("loading qwen3.5-9b into memory", said)
        self.assertTrue(any("its download is kept" in line for line in said))
        # What the catalog passes to llama.cpp for Bonsai 2 27B are options the pinned build takes.
        args = catalog.get("bonsai-2-27b")["artifacts"]["gguf"]["server_args"]
        self.assertEqual(args[args.index("--load-mode") + 1], "none")
        self.assertNotIn("--no-mmap", args)

    def test_a_refused_licence_lookup_is_that_models_failure_not_a_crash(self):
        class Limited:
            def resolve_license(self, repo, depth=3):
                raise hub_module.RateLimited(120)
        rt = FakeLlama()
        result = manager.setup(self.config(), model_id="qwen3.5-9b", runtime=rt, hw=apple(18), hub=Limited(),
                               progress=lambda m: None, embedding=False, decision=False, smoke=fake_smoke({}))
        self.assertFalse(result["ok"])
        self.assertIn("No model could be set up: Hugging Face is limiting requests", result["error"])
        self.assertEqual(rt.pulled, [])

    def test_a_file_at_a_plain_address_comes_in_as_many_ranges_as_asked_and_resumes_with_more(self):
        blob = os.urandom(hub_module.PARALLEL_MIN + 12_345)
        seen = []

        def serve(request):
            self.assertNotIn("authorization", request.headers)             # the Hub's token stays with the Hub
            first, last = request.headers["range"][6:].split("-")
            seen.append(int(first))
            return httpx.Response(206, content=blob[int(first):int(last) + 1])

        transport = httpx.MockTransport(lambda request: serve(request))
        dest = self.home / "models" / "registry" / "model.gguf"
        part, state = dest.with_name("model.gguf.part"), dest.with_name("model.gguf.part.json")
        dest.parent.mkdir(parents=True)
        half = len(blob) // 2
        part.write_bytes(blob[:1000] + bytes(half - 1000) + blob[half:half + 5000])      # two ranges, both begun
        state.write_text(json.dumps({"size": len(blob), "ranges": [[0, half, 1000], [half, len(blob), 5000]]}))
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret"}):
            got = Hub(httpx.Client(transport=transport), cache_dir=False).fetch(
                "https://registry.example/v2/blobs/sha256:abc", dest, hashlib.sha256(blob).hexdigest(), len(blob),
                parallel=8)
        self.assertEqual(got.read_bytes(), blob)
        self.assertEqual(len(seen), 8)                                    # four pieces of each unfinished half
        self.assertEqual(sorted(seen)[:2], [1000, 1000 + (half - 1000) // 4])
        self.assertFalse(part.exists() or state.exists())
        self.assertEqual(hub_module._resplit([[0, 10, 10], [10, 20, 4]], 3), [[0, 10, 10], [10, 14, 4], [14, 16, 0],
                                                                              [16, 18, 0], [18, 20, 0]])

    def test_resumable_download_verifies_sha256(self):
        blob = os.urandom(300_000)
        digest = hashlib.sha256(blob).hexdigest()
        ranges = []

        def resolve(request):
            self.assertEqual(request.headers.get("authorization"), "Bearer hf_test")
            start = int(request.headers.get("range", "bytes=0-")[6:-1] or 0)
            ranges.append(start)
            status = 206 if start else 200
            return httpx.Response(status, content=blob[start:], headers={"content-length": str(len(blob) - start)})

        routes = {"/org/M/resolve/abc/model.gguf": resolve}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        dest = self.home / "models" / "model.gguf"
        dest.parent.mkdir(parents=True)
        (dest.parent / "model.gguf.part").write_bytes(blob[:100_000])  # an interrupted earlier attempt
        messages = []
        bad = self.home / "models" / "bad.gguf"
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            hub.download("org/M", "model.gguf", dest, revision="abc", sha256=digest, size=len(blob),
                         progress=messages.append)
            with self.assertRaises(HubError):
                hub.download("org/M", "model.gguf", bad, revision="abc", sha256="0" * 64)
        self.assertEqual(ranges, [100_000, 0])
        self.assertEqual(dest.read_bytes(), blob)
        self.assertFalse((dest.parent / "model.gguf.part").exists())
        self.assertTrue(messages and messages[-1].startswith("downloading model.gguf: 100%"))
        self.assertFalse(bad.exists())
        self.assertFalse((bad.parent / "bad.gguf.part").exists())

    def test_a_download_carries_on_when_the_connection_drops(self):
        blob = os.urandom(3_500_000)
        starts = []

        class Dropping(httpx.SyncByteStream):
            def __init__(self, data):
                self.data = data

            def __iter__(self):                 # drops 1.2 MB into every response, as the CDN did
                yield self.data[:1_200_000]
                if len(self.data) > 1_200_000:
                    raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

        def resolve(request):
            start = int(request.headers.get("range", "bytes=0-")[6:-1] or 0)
            starts.append(start)
            return httpx.Response(206 if start else 200, stream=Dropping(blob[start:]),
                                  headers={"content-length": str(len(blob) - start)})

        hub = Hub(httpx.Client(transport=hf_transport({"/org/M/resolve/abc/model.gguf": resolve})), cache_dir=False)
        dest = self.home / "models" / "model.gguf"
        messages = []
        with mock.patch("coderunner.models.hub.time.sleep"):
            hub.download("org/M", "model.gguf", dest, revision="abc", sha256=hashlib.sha256(blob).hexdigest(),
                         size=len(blob), progress=messages.append)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts, sorted(set(starts)))                    # each attempt carries on further
        self.assertGreaterEqual(len(starts), 3)
        self.assertEqual(dest.read_bytes(), blob)
        self.assertTrue(any("resuming" in m for m in messages))

    @staticmethod
    def ranged(blob, seen, drop_at=None):
        """A CDN answering byte ranges; with drop_at, the range starting there drops after 1.2 MB."""
        lock = threading.Lock()

        class Dropping(httpx.SyncByteStream):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield self.data[:1_200_000]
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

        def resolve(request):
            first, last = request.headers["range"][6:].split("-")
            start, end = int(first), int(last) + 1
            with lock:
                seen.append(start)
            if start == drop_at:
                return httpx.Response(206, stream=Dropping(blob[start:end]))
            return httpx.Response(206, content=blob[start:end])
        return resolve

    def test_a_large_download_comes_in_ranges_at_once(self):
        blob, seen = os.urandom(1_000_003), []
        hub = Hub(httpx.Client(transport=hf_transport({"/org/M/resolve/abc/model.gguf": self.ranged(blob, seen)})),
                  cache_dir=False)
        dest = self.home / "models" / "model.gguf"
        with mock.patch.object(hub_module, "PARALLEL_MIN", 1000):
            hub.download("org/M", "model.gguf", dest, revision="abc", sha256=hashlib.sha256(blob).hexdigest(),
                         size=len(blob))
        self.assertEqual(sorted(seen), [0, 250_000, 500_001, 750_002])
        self.assertEqual(dest.read_bytes(), blob)
        self.assertEqual(sorted(p.name for p in dest.parent.iterdir()), ["model.gguf"])

    def test_an_interrupted_ranged_download_carries_on_in_each_range(self):
        blob, seen = os.urandom(8_000_000), []
        third = 4_000_000                                   # where the third of four ranges starts
        routes = {"/org/M/resolve/abc/model.gguf": self.ranged(blob, seen, drop_at=third)}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        dest = self.home / "models" / "model.gguf"
        digest = hashlib.sha256(blob).hexdigest()
        with mock.patch.object(hub_module, "PARALLEL_MIN", 1000), mock.patch.object(hub_module, "RESUMES", 0):
            with self.assertRaisesRegex(HubError, "run again to resume"):
                hub.download("org/M", "model.gguf", dest, revision="abc", sha256=digest, size=len(blob))
            saved = json.loads((dest.parent / "model.gguf.part.json").read_text())["ranges"]
            self.assertEqual(saved[2][2], 1 << 20)          # what came before the drop is kept
            seen.clear()
            routes["/org/M/resolve/abc/model.gguf"] = self.ranged(blob, seen)
            hub.download("org/M", "model.gguf", dest, revision="abc", sha256=digest, size=len(blob))
        self.assertEqual(sorted(seen), sorted(start + done for start, end, done in saved if done < end - start))
        self.assertEqual(dest.read_bytes(), blob)
        self.assertFalse((dest.parent / "model.gguf.part.json").exists())

    def test_a_server_without_ranges_sends_the_file_in_one_stream(self):
        blob = os.urandom(100_000)
        routes = {"/org/M/resolve/abc/model.gguf": lambda request: httpx.Response(200, content=blob)}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        dest = self.home / "models" / "model.gguf"
        with mock.patch.object(hub_module, "PARALLEL_MIN", 1000):
            hub.download("org/M", "model.gguf", dest, revision="abc", sha256=hashlib.sha256(blob).hexdigest(),
                         size=len(blob))
        self.assertEqual(dest.read_bytes(), blob)
        self.assertEqual(sorted(p.name for p in dest.parent.iterdir()), ["model.gguf"])

    def test_auth_errors_explain_hf_token(self):
        routes = {"/api/models/org/Gated": lambda r: httpx.Response(401, json={"error": "gated"})}
        hub = Hub(httpx.Client(transport=hf_transport(routes)), cache_dir=False)
        with self.assertRaisesRegex(HubError, "HF_TOKEN"):
            hub.model_info("org/Gated")


# ---- smoke test ---------------------------------------------------------------------

def tool_call(name, args):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


GOOD = {
    "add": {"tool_calls": [tool_call("add_numbers", {"a": 3, "b": 7})]},
    "weather": {"tool_calls": [tool_call("get_weather", {"city": "Paris"})]},
    "ready": {"content": "<think>short</think>Ready."},
    "colour": {"content": "Red"},
}


class FakeOpenAI:
    """A tiny OpenAI-compatible server answering from a script keyed by prompt."""

    def __init__(self, script):
        self.script, self.requests = script, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                content = body["messages"][-1]["content"]
                text = content if isinstance(content, str) else content[0]["text"]
                key = next(k for k in ("add", "weather", "ready", "colour") if k in text.lower())
                reply = outer.script[key]
                if reply == 500:
                    payload, status = {"error": "boom"}, 500
                else:
                    message = {"role": "assistant", "content": reply.get("content", ""),
                               **({"tool_calls": reply["tool_calls"]} if "tool_calls" in reply else {})}
                    payload, status = {"choices": [{"message": message, "finish_reason": "stop"}],
                                       "usage": {"completion_tokens": 20}}, 200
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class SmokeTestTests(unittest.TestCase):
    def run_against(self, overrides=None, vision=False):
        server = FakeOpenAI({**GOOD, **(overrides or {})})
        self.addCleanup(server.close)
        result = asyncio.run(smoketest.run(server.url, "m", vision=vision))
        return result, {c["name"]: c for c in result["checks"]}, server

    def test_passes_with_correct_tool_use(self):
        result, checks, server = self.run_against(vision=True)
        self.assertTrue(result["passed"], result)
        self.assertEqual(list(checks), ["tool_arguments", "tool_selection", "no_tool", "vision"])
        self.assertGreater(result["tok_s"], 0)
        self.assertEqual(server.requests[0]["temperature"], 0)
        image = server.requests[3]["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/png;base64,"))

    def test_text_form_tool_calls_are_accepted(self):
        text = '<tool_call>{"name": "add_numbers", "arguments": {"a": 3, "b": 7}}</tool_call>'
        result, checks, _ = self.run_against({"add": {"content": text}})
        self.assertTrue(checks["tool_arguments"]["passed"], checks["tool_arguments"])

    def test_wrong_arguments_fail(self):
        # The failure seen with a 1.7B model: it reported 13 instead of 10.
        result, checks, _ = self.run_against({"add": {"tool_calls": [tool_call("add_numbers", {"a": 3, "b": 10})]}})
        self.assertFalse(result["passed"])
        self.assertFalse(checks["tool_arguments"]["passed"])
        self.assertIn('"b": 10', checks["tool_arguments"]["detail"])
        self.assertTrue(checks["tool_selection"]["passed"])

    def test_string_arguments_and_wrong_tool_fail(self):
        _, checks, _ = self.run_against({
            "add": {"tool_calls": [tool_call("add_numbers", {"a": "3", "b": "7"})]},
            "weather": {"tool_calls": [tool_call("add_numbers", {"a": 1, "b": 2})]},
        })
        self.assertFalse(checks["tool_arguments"]["passed"])
        self.assertFalse(checks["tool_selection"]["passed"])

    def test_tool_call_when_none_needed_and_missing_call_fail(self):
        _, checks, _ = self.run_against({
            "ready": {"tool_calls": [tool_call("get_weather", {"city": "ready"})]},
            "add": {"content": "3 + 7 = 10"},
        })
        self.assertFalse(checks["no_tool"]["passed"])
        self.assertIn("no tool call", checks["tool_arguments"]["detail"])

    def test_vision_and_server_errors_are_reported(self):
        result, checks, _ = self.run_against({"colour": {"content": "Blue"}, "weather": 500}, vision=True)
        self.assertFalse(checks["vision"]["passed"])
        self.assertFalse(checks["tool_selection"]["passed"])
        self.assertIn("500", checks["tool_selection"]["detail"])
        self.assertTrue(checks["tool_arguments"]["passed"])
        self.assertFalse(result["passed"])

    def test_red_png_is_valid(self):
        png = smoketest.solid_png(64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        pos, chunks = 8, {}
        while pos < len(png):
            length = struct.unpack(">I", png[pos:pos + 4])[0]
            kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + length]
            crc = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])[0]
            self.assertEqual(crc, zlib.crc32(kind + data))
            chunks[kind] = data
            pos += 12 + length
        self.assertEqual(struct.unpack(">II", chunks[b"IHDR"][:8]), (64, 64))
        pixels = zlib.decompress(chunks[b"IDAT"])
        self.assertEqual(len(pixels), 64 * (1 + 64 * 3))
        self.assertEqual(pixels[1:4], b"\xff\x00\x00")


# ---- runtimes -----------------------------------------------------------------------------

class RuntimeTests(TempHome):
    def test_ollama_modelfile_alias(self):
        rt = runtimes.OllamaRuntime()
        entry = catalog.get("qwen3.5-4b")
        self.assertEqual(rt.modelfile("qwen3.5:4b", 32768), "FROM qwen3.5:4b\nPARAMETER num_ctx 32768\n")
        with mock.patch.object(rt, "binary", return_value="/usr/local/bin/ollama"):
            self.assertEqual(rt.alias_command(entry, "/tmp/m.Modelfile"),
                             ["/usr/local/bin/ollama", "create", "coderunner-qwen3.5-4b", "-f", "/tmp/m.Modelfile"])
            seen = {}

            def fake_run(cmd, env=None, **kwargs):
                seen.update(cmd=cmd, host=env["OLLAMA_HOST"], modelfile=pathlib.Path(cmd[-1]).read_text())
                return mock.Mock(returncode=0, stdout="success", stderr="")
            with mock.patch.object(runtimes.subprocess, "run", fake_run):
                self.assertEqual(rt.prepare(entry, 16384), "coderunner-qwen3.5-4b")
        self.assertEqual(seen["cmd"][1:3], ["create", "coderunner-qwen3.5-4b"])
        self.assertEqual(seen["host"], "127.0.0.1:11434")
        self.assertEqual(seen["modelfile"], "FROM qwen3.5:4b-q4_K_M\nPARAMETER num_ctx 16384\n")
        self.assertTrue(pathlib.Path(seen["cmd"][-1]).is_relative_to(self.home))
        self.assertEqual(rt.base_url, "http://127.0.0.1:11434/v1")

    def test_ollama_alias_failure_is_reported(self):
        rt = runtimes.OllamaRuntime()
        failed = mock.Mock(returncode=1, stdout="", stderr="Error: model not found")
        with mock.patch.object(runtimes.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                rt.prepare(catalog.get("qwen3.5-4b"), 8192)

    def test_the_driver_keeps_its_conversation_while_the_step_judge_asks_it(self):
        rt = runtimes.DriverRuntime()
        entry = catalog.get("fara1.5-4b")
        rt._save({"models": {entry["id"]: {"model": "/m/f.gguf", "mmproj": "/m/mmproj.gguf"}}})
        with mock.patch.object(rt, "running", return_value=False):
            rt.prepare(entry, 16384)
        with mock.patch.object(rt, "binary", return_value="llama-server"):
            cmd = " ".join(rt.command(rt._state()["active"]))
        self.assertIn("-np 2 --kv-unified", cmd)            # the judge gets a slot of its own
        self.assertIn("--port 8083", cmd)

    def test_llama_server_command(self):
        rt = runtimes.LlamaCppRuntime()
        entry = catalog.get("qwen3.5-9b")
        rt._save({"models": {entry["id"]: {"model": "/m/q.gguf", "mmproj": "/m/mmproj.gguf"}}})
        with mock.patch.object(rt, "running", return_value=False):
            self.assertEqual(rt.prepare(entry, 32768), "coderunner-qwen3.5-9b")
        with mock.patch.object(rt, "binary", return_value="llama-server"):
            cmd = rt.command(rt._state()["active"])
        self.assertEqual(cmd, ["llama-server", "-m", "/m/q.gguf", "--mmproj", "/m/mmproj.gguf", "--jinja", "-c", "32768",
                               "--host", "127.0.0.1", "--port", "8081", "--cache-ram", "0", "-np", "3",
                               "--kv-unified", "--slot-save-path", str(paths.assistant_dir() / "run" / "slots"),
                               "--alias", "coderunner-qwen3.5-9b", "--sleep-idle-seconds", "600"])
        self.assertTrue(rt.vision(entry))
        self.assertEqual(rt.base_url, "http://127.0.0.1:8081/v1")
        with self.assertRaisesRegex(RuntimeError, "not been downloaded"):
            rt.prepare(catalog.get("qwen3.5-4b"), 8192)

    def test_llama_server_offload_and_cluster_flags(self):
        cfg = self.config(cluster=[{"host": "10.0.0.2", "memory_gb": 128}, {"host": "10.0.0.3", "port": 50060}])
        rt = runtimes.LlamaCppRuntime(cfg)
        entry = catalog.get("qwen3.6-35b-a3b")
        rt._save({"models": {entry["id"]: {"model": "/m/q.gguf"}}})
        with mock.patch.object(rt, "running", return_value=False):
            rt.prepare(entry, 32768, offload=True)
        with mock.patch.object(rt, "binary", return_value="llama-server"):
            cmd = rt.command(rt._state()["active"])
        self.assertIn("--cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--rpc") + 1], "10.0.0.2:50052,10.0.0.3:50060")
        with mock.patch.object(rt, "running", return_value=False):
            rt.prepare(entry, 32768)
        with mock.patch.object(rt, "binary", return_value="llama-server"):
            self.assertNotIn("--cpu-moe", rt.command(rt._state()["active"]))

    def test_a_second_home_can_serve_its_models_on_other_ports(self):
        from coderunner.models import coreml
        ports = lambda cfg: [runtimes.LlamaCppRuntime(cfg).port, runtimes.PrismRuntime(cfg).port,  # noqa: E731
                             runtimes.DriverRuntime(cfg).port, coreml.CoreMLRuntime(cfg).port]
        self.assertEqual(ports(self.config()), [8081, 8082, 8083, 8084])
        self.assertEqual(ports(self.config(port_base=8181)), [8181, 8182, 8183, 8184])
        self.assertEqual(runtimes.LlamaCppRuntime(self.config(port_base=8181)).base_url, "http://127.0.0.1:8181/v1")
        self.assertEqual(runtimes.LlamaCppRuntime(self.config(), port=9000).port, 9000)

    def test_choose_runtime(self):
        self.assertIsInstance(runtimes.choose_runtime(self.config(runtime="llama.cpp")), runtimes.LlamaCppRuntime)
        external = runtimes.choose_runtime(self.config(runtime="openai", base_url="http://10.0.0.2:1234/v1/"))
        self.assertEqual((external.name, external.base_url), ("openai", "http://10.0.0.2:1234/v1"))
        with mock.patch.object(runtimes.OllamaRuntime, "available", return_value=False), \
                mock.patch.object(runtimes.LlamaCppRuntime, "available", return_value=True):
            self.assertEqual(runtimes.choose_runtime(self.config()).name, "llama.cpp")
        with mock.patch.object(runtimes.OllamaRuntime, "available", return_value=False), \
                mock.patch.object(runtimes.LlamaCppRuntime, "available", return_value=False):
            self.assertIsNone(runtimes.choose_runtime(self.config()))
        with self.assertRaises(ValueError):
            runtimes.choose_runtime(self.config(runtime="nope"))
        self.assertIn("brew install ollama", runtimes.install_hint(apple(18)))
        self.assertIn("https://ollama.com/install.sh", runtimes.install_hint(linux(16)))


# ---- manager ------------------------------------------------------------------------------

class FakeRuntime(runtimes.Runtime):
    name = "ollama"
    base_url = "http://127.0.0.1:9/v1"

    def __init__(self):
        super().__init__()
        self.pulled, self.prepared, self.offloaded, self.removed = [], [], [], []

    def available(self):
        return True

    def running(self):
        return True

    def pull(self, entry, progress=None):
        self.pulled.append(entry["id"])

    def remove(self, entry):
        self.removed.append(entry["id"])

    def unload(self, model):
        self.unloaded = [*getattr(self, "unloaded", []), model]

    def prepare(self, entry, ctx, offload=False):
        self.prepared.append((entry["id"], ctx))
        if offload:
            self.offloaded.append(entry["id"])
        return self.model_id(entry)


class MemoryNowTests(TempHome):
    """A plan says what fits the machine; setup also looks at what is free at the moment it loads a model."""

    def test_a_model_that_fits_the_machine_but_not_this_moment_is_not_loaded(self):
        rt, cfg, said = FakeRuntime(), self.config(), []
        rt.start = mock.Mock()
        hw = apple(24, "Apple M4 Pro", sandbox=1)
        first = fit.recommend(hw, cfg, runtime="ollama")[0]
        with mock.patch.object(hardware, "available_gb", return_value=first["need_gb"] - 3.6):
            result = manager.setup(cfg, runtime=rt, hw=hw, progress=said.append, smoke=fake_smoke({}), measure=False,
                                   embedding=False, decision=False)
        short = result["memory_short"]
        self.assertEqual((result["ok"], short["model"], short["need_gb"]), (False, first["id"], round(first["need_gb"], 1)))
        self.assertEqual(short["free_gb"], round(first["need_gb"] - 3.6, 1))
        self.assertRegex(result["error"], r"^Not enough memory free: .+ needs [\d.]+ GB and [\d.]+ GB is free\. Close some apps")
        # Nothing was loaded, the download stays for the next try, and no smaller model was settled for.
        self.assertEqual((rt.prepared, rt.removed, cfg.get("models.chat") or ""), ([], [], ""))
        self.assertTrue(any("its download is kept" in line for line in said))
        if short["smaller"]:
            smaller = next(c for c in fit.recommend(hw, cfg, runtime="ollama") if c["id"] == short["smaller"])
            self.assertLessEqual(smaller["need_gb"] + manager.MEMORY_MARGIN_GB, short["free_gb"] + 0.05)
            self.assertIn(f"--model {short['smaller']}", result["error"])
            self.assertEqual(result["next"]["model"], short["smaller"])     # offered with what it would download
            self.assertGreater(result["next"]["download_gb"], 0)

    def test_a_margin_is_left_beside_the_model_and_one_asked_for_by_name_is_loaded_anyway(self):
        hw = apple(24, "Apple M4 Pro", sandbox=1)
        first = fit.recommend(hw, self.config(), runtime="ollama")[0]
        # 8.8 GB free for a 9.7 GB model was let through once, and swapped: as much as it needs is not enough.
        with mock.patch.object(hardware, "available_gb", return_value=first["need_gb"] + 0.5):
            tight = manager.setup(self.config(), runtime=FakeRuntime(), hw=hw, progress=lambda m: None,
                                  smoke=fake_smoke({}), measure=False, embedding=False, decision=False)
        self.assertEqual((tight["ok"], tight["memory_short"]["model"]), (False, first["id"]))
        with mock.patch.object(hardware, "available_gb", return_value=first["need_gb"] + manager.MEMORY_MARGIN_GB + 0.1):
            roomy = manager.setup(self.config(), runtime=FakeRuntime(), hw=hw, progress=lambda m: None,
                                  smoke=fake_smoke({}), measure=False, embedding=False, decision=False)
        self.assertTrue(roomy["ok"], roomy)
        rt, said = FakeRuntime(), []
        with mock.patch.object(hardware, "available_gb", return_value=2.0):
            asked = manager.setup(self.config(), model_id=first["id"], runtime=rt, hw=hw, progress=said.append,
                                  smoke=fake_smoke({}), measure=False, embedding=False, decision=False)
        self.assertTrue(asked["ok"], asked)
        self.assertTrue(any("loading it anyway because it was requested" in line for line in said))
        # The app names the model the owner chose on the first screen; that is no order to load it regardless.
        rt = FakeRuntime()
        with mock.patch.object(hardware, "available_gb", return_value=2.0):
            chosen = manager.setup(self.config(), model_id=first["id"], runtime=rt, hw=hw, progress=lambda m: None,
                                   smoke=fake_smoke({}), measure=False, embedding=False, decision=False, insist=False)
        self.assertEqual((chosen["ok"], chosen["memory_short"]["model"], rt.prepared), (False, first["id"], []))

    def test_nothing_is_started_before_setup_has_chosen_a_model(self):
        # An Ollama the owner had installed for something else was started by the daemon on a Mac that had not
        # been set up yet (and would use llama.cpp).
        rt, cfg = FakeRuntime(), self.config()
        rt.running, rt.start = (lambda: False), mock.Mock()
        with mock.patch.object(manager, "choose_runtime", return_value=rt):
            manager.ensure_running(cfg)
            rt.start.assert_not_called()
            cfg.set("models.chat", "coderunner-qwen3.5-9b")
            manager.ensure_running(cfg)
            rt.start.assert_called_once()

    def test_from_the_app_a_model_that_does_not_work_out_is_not_replaced_by_another_download(self):
        class FirstWillNotStart(FakeLlama):
            def installed(self, entry):
                return entry["id"] == self.first

            def start(self, timeout=60):
                if self.prepared and self.prepared[-1][0] == self.first:
                    raise RuntimeError("the server exited with code 1")
        hw, cfg = apple(24, "Apple M4 Pro", sandbox=1), self.config()
        ranked = [c["id"] for c in fit.recommend(hw, cfg, runtime="llama.cpp")]
        rt = FirstWillNotStart()
        rt.first = ranked[0]
        stopped = manager.setup(cfg, runtime=rt, hw=hw, progress=lambda m: None, smoke=fake_smoke({}), measure=False,
                                embedding=False, decision=False, other_downloads=False, hub=ApacheHub())
        self.assertEqual((stopped["ok"], rt.pulled), (False, []))                 # nothing was fetched unasked
        self.assertEqual(stopped["next"]["model"], ranked[1])
        self.assertGreater(stopped["next"]["download_gb"], 0)
        self.assertRegex(stopped["error"], r"could not be set up: the server exited with code 1$")
        # `coderunner setup` in a terminal, which prints what it does, still goes on to the next model.
        rt = FirstWillNotStart()
        rt.first = ranked[0]
        went_on = manager.setup(self.config(), runtime=rt, hw=hw, progress=lambda m: None, smoke=fake_smoke({}),
                                measure=False, embedding=False, decision=False, hub=ApacheHub())
        self.assertEqual((went_on["ok"], rt.pulled), (True, [ranked[1]]))

    def test_from_the_app_a_model_that_fails_its_checks_is_said_so_and_the_next_is_offered(self):
        class OnlyFirstIsHere(FakeLlama):
            def installed(self, entry):
                return entry["id"] == self.first
        hw = apple(24, "Apple M4 Pro", sandbox=1)
        ranked = [c["id"] for c in fit.recommend(hw, self.config(), runtime="llama.cpp")]
        rt = OnlyFirstIsHere()
        rt.first = ranked[0]
        failing = fake_smoke({rt.model_id(catalog.get(ranked[0])): False})
        result = manager.setup(self.config(), runtime=rt, hw=hw, progress=lambda m: None, smoke=failing, measure=False,
                               embedding=False, decision=False, other_downloads=False, hub=ApacheHub())
        self.assertEqual((result["ok"], rt.pulled, result["next"]["model"]), (False, [], ranked[1]))
        self.assertTrue(result["error"].endswith("could not be set up: it did not pass its checks on this computer"),
                        result["error"])

    def test_from_the_app_a_download_the_disk_has_no_room_for_is_refused_in_a_sentence(self):
        rt = FakeRuntime()
        with mock.patch.object(manager, "_free_gb", return_value=6.0):
            result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=lambda m: None, smoke=fake_smoke({}),
                                   measure=False, embedding=False, decision=False, other_downloads=False)
        self.assertEqual((result["ok"], rt.pulled), (False, []))
        self.assertEqual(result["disk_short"]["room_gb"], 1.0)
        self.assertRegex(result["error"], r"^Not enough disk space: .+ needs [\d.]+ GB and 1\.0 GB can be used\. Free [\d.]+ GB and try again\.$")
        self.assertLess(len(result["error"]), 110)


class FakeLlama(FakeRuntime):
    name = "llama.cpp"
    base_url = "http://127.0.0.1:8/v1"


def fake_smoke(outcomes):
    calls = []

    async def smoke(base_url, model, vision=False, api_key=""):
        calls.append((model, vision))
        passed = outcomes.get(model, True)
        checks = [{"name": "tool_arguments", "passed": passed, "detail": "", "seconds": 0.1}]
        if vision:
            checks.append({"name": "vision", "passed": outcomes.get(model + ":vision", True), "detail": "", "seconds": 0.1})
        return {"passed": all(c["passed"] for c in checks), "checks": checks, "tok_s": 12.0}
    smoke.calls = calls
    return smoke


def fake_measure(speed):
    """A bench.measure stand-in: speed(model) -> (read tok/s, generation tok/s)."""
    calls = []

    async def measure(base_url, model, api_key="", **_):
        calls.append(model)
        read, gen = speed(model)
        return {"prompt_tokens": 4000, "written_tokens": 128, "first_token_s": round(4000 / read, 2),
                "read_tok_s": read, "gen_tok_s": gen, "step_s": round(fit.step_seconds(read, gen), 1)}
    measure.calls = calls
    return measure


class ManagerTests(TempHome):
    def test_plan_reports_choice_and_alternatives(self):
        result = manager.plan(self.config(), hw=apple(18), runtime=FakeRuntime())
        self.assertEqual(result["model"]["id"], "qwen3.5-9b")
        self.assertEqual(result["license"]["decision"], "allow")
        self.assertIn("qwen3.5-4b", [c["id"] for c in result["alternatives"]])
        self.assertIn("qwen3.8-27b", [c["id"] for c in result["rejected"]])
        self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")
        self.assertIsNone(manager.plan(self.config(), hw=apple(8, "Apple M1"), runtime=FakeRuntime())["model"])

    def test_a_requested_chat_model_runs_on_llama_cpp_on_apple_silicon(self):
        ollama, llama = FakeRuntime(), FakeLlama()
        with mock.patch.object(manager, "_chat_on_llama", return_value=True), \
                mock.patch.object(manager, "choose_runtime", return_value=ollama), \
                mock.patch.object(manager, "_llama", return_value=llama), \
                mock.patch.object(manager, "_prism", return_value=None):
            done = manager.setup(self.config(), model_id="qwen3.5-9b", hw=apple(18), progress=lambda m: None,
                                 embedding=False, decision=False, smoke=fake_smoke({}), hub=ApacheHub())
        self.assertTrue(done["ok"], done)
        self.assertEqual((done["runtime"], llama.pulled, ollama.pulled), ("llama.cpp", ["qwen3.5-9b"], []))

    def test_a_build_other_than_the_planned_gguf_is_refused(self):
        # Ollama's plain qwen3.5:9b tag became an MLX build on Apple silicon (16.3 GB loaded on 18 GB).
        rt = FakeRuntime()
        rt.build_format = lambda entry: "safetensors"
        result = manager.setup(self.config(), model_id="qwen3.5-9b", runtime=rt, hw=apple(18), progress=lambda m: None,
                               embedding=False, decision=False, smoke=fake_smoke({}))
        self.assertFalse(result["ok"])
        self.assertIn("not the GGUF build", result["attempts"][0]["error"])
        self.assertEqual(rt.removed, ["qwen3.5-9b"])                 # its download is deleted
        self.assertTrue(catalog.get("qwen3.5-9b")["artifacts"]["ollama"]["tag"].endswith("-q4_K_M"))

    def test_decision_model_only_where_it_fits_beside_the_chat_model(self):
        # tev1 4B judged 66 labelled browser steps with no missed commit; it runs beside the chat and
        # embedding models when they leave room (24 GB here), otherwise the chat model judges (18 GB).
        small = manager.plan(self.config(), hw=apple(18, sandbox=1), runtime=FakeRuntime())
        self.assertIsNone(small["decision"])
        big = manager.plan(self.config(), hw=apple(48, "Apple M4 Pro", sandbox=1), runtime=FakeRuntime())
        self.assertEqual(big["decision"]["id"], "tev1-4b")
        entry = catalog.get("tev1-4b")
        self.assertEqual((entry["decision"]["missed"], entry["decision"]["false_asks"]), (0, 7))
        rt, cfg = FakeRuntime(), self.config()

        async def answers(*args, **kwargs):
            return 0.0
        with mock.patch.object(decision_module.DecisionModel, "probability", answers):
            result = manager.setup(cfg, runtime=rt, hw=apple(48, "Apple M4 Pro", sandbox=1), progress=lambda m: None,
                                   smoke=fake_smoke({}), measure=False)
        self.assertTrue(result["ok"], result)
        self.assertIn("tev1-4b", rt.pulled)
        self.assertEqual((cfg.get("models.decision"), cfg.get("models.decision_api"), cfg.get("models.decision_threshold")),
                         ("tev1:4b", "systemone", 0.7))
        self.assertEqual(cfg.get("models.decision_url"), "")              # same runtime as the chat model
        # Where it does not fit, setup clears any earlier choice so the chat model judges.
        with mock.patch.object(decision_module.DecisionModel, "probability", answers):
            manager.setup(cfg, runtime=FakeRuntime(), hw=apple(18, sandbox=1), progress=lambda m: None,
                          smoke=fake_smoke({}), measure=False)
        self.assertEqual(cfg.get("models.decision"), "")
        # A decision model on Ollama is reached there even when another runtime serves the chat model.
        judge = decision_module.DecisionModel.from_config(self.config(chat="bonsai", decision="tev1:4b",
                                                                      decision_url="http://127.0.0.1:11434/v1"),
                                                          base_url="http://127.0.0.1:8082/v1")
        self.assertEqual((judge.base_url, judge.api), ("http://127.0.0.1:11434/v1", "systemone"))
        own = decision_module.DecisionModel.from_config(self.config(chat="bonsai"), base_url="http://127.0.0.1:8082/v1")
        self.assertEqual((own.base_url, own.api), ("http://127.0.0.1:8082/v1", "chat"))

    def test_gpu_memory_limit_note_needs_a_clearly_better_model(self):
        # On 96 GB, raising the GPU wired limit unlocks Ornith-1.5-122B (77 GB); on 128 GB it already fits.
        p96 = manager.plan(self.config(), hw=apple(96, "Apple M2 Max", sandbox=1), runtime=FakeRuntime())
        self.assertEqual(p96["wired_limit_hint"]["model"], "ornith-1.5-122b")
        p128 = manager.plan(self.config(), hw=apple(128, "Apple M4 Max", sandbox=1), runtime=FakeRuntime())
        self.assertFalse(p128["notes"])

    def test_spark_cluster_pools_memory(self):
        spark = linux(128, {"nvidia-smi": "NVIDIA GB10, [N/A], [N/A], 12.1\n"})
        cfg = self.config(cluster=[{"host": "spark-2.local", "memory_gb": 128}])
        hw = manager._hardware(cfg, spark)
        self.assertEqual(hw.nodes, 2)
        self.assertAlmostEqual(hw.budget_gb, spark.budget_gb + 128 * 0.85 - 4)
        self.assertEqual(manager.plan(cfg, hw=spark, runtime=FakeRuntime())["hardware"]["nodes"], 2)

    def test_experts_in_ram_run_on_llama_cpp_next_to_ollama(self):
        rtx4070 = nvidia(12, name="NVIDIA GeForce RTX 4070")
        with mock.patch.object(manager, "choose_runtime", return_value=FakeRuntime()), \
                mock.patch.object(manager, "_prism", return_value=None), \
                mock.patch.object(runtimes.LlamaCppRuntime, "available", return_value=False):
            result = manager.plan(self.config(), hw=rtx4070)
            self.assertEqual(result["model"]["id"], "qwen3.5-9b")
            self.assertIn("inkling-2-mini would rank higher but needs llama.cpp", " ".join(result["notes"]))
            with mock.patch.object(manager, "WORTH_A_NOTE", 0.0):
                result = manager.plan(self.config(), hw=rtx4070)
        self.assertIn("inkling-2-mini would rank higher but needs llama.cpp", " ".join(result["notes"]))
        main, llama = FakeRuntime(), FakeLlama()
        with mock.patch.object(manager, "choose_runtime", return_value=main), \
                mock.patch.object(manager, "_prism", return_value=None), \
                mock.patch.object(manager, "_llama", return_value=llama):
            result = manager.plan(self.config(), hw=rtx4070)
            self.assertEqual((result["model"]["id"], result["model"]["runtime"]), ("inkling-2-mini", "llama.cpp"))
            self.assertEqual(result["runtime"]["name"], "llama.cpp")
            self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")        # still from Ollama
            done = manager.setup(self.config(), hw=rtx4070, progress=lambda m: None, embedding=False,
                                 smoke=fake_smoke({}), hub=ApacheHub())     # the GGUF repo's live licence
        self.assertTrue(done["ok"], done)
        self.assertEqual((llama.offloaded, main.prepared), (["inkling-2-mini"], []))

    def test_driver_setup_installs_llama_server_when_missing(self):
        installed = []

        def install(rt, progress=None):
            installed.append(True)
        with mock.patch.object(manager, "_hardware", return_value=apple(18)), \
                mock.patch.object(runtimes.DriverRuntime, "binary", lambda rt: "/bin/llama-server" if installed else None), \
                mock.patch.object(runtimes.DriverRuntime, "install", install), \
                mock.patch.object(runtimes.DriverRuntime, "installed", return_value=True), \
                mock.patch.object(runtimes.DriverRuntime, "prepare", return_value="coderunner-fara1.5-4b"):
            result = manager.setup_driver(self.config(), progress=lambda m: None)
        self.assertTrue(result["ok"], result)
        self.assertEqual(installed, [True])      # without the server the driver could never start

    def test_an_interrupted_download_is_kept_for_resuming(self):
        class Flaky(FakeRuntime):
            def pull(self, entry, progress=None):
                raise RuntimeError("Download interrupted (received 5345964237 bytes, expected 7206168928)")
        rt = Flaky()
        result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=lambda m: None, embedding=False,
                               smoke=fake_smoke({}))
        self.assertFalse(result["ok"])
        self.assertIn("interrupted", result["attempts"][0]["error"])
        self.assertEqual(rt.removed, [])            # the partial file stays; running setup again resumes it

    def test_setup_falls_back_and_writes_config(self):
        rt, smoke = FakeRuntime(), fake_smoke({"coderunner-qwen3.5-9b": False, "coderunner-qwen3.5-4b:vision": False})
        cfg = self.config()
        result = manager.setup(cfg, runtime=rt, hw=apple(18), progress=lambda m: None, embedding=False, smoke=smoke)
        self.assertTrue(result["ok"], result)
        self.assertEqual(rt.pulled, ["qwen3.5-9b", "qwen3.5-4b"])
        self.assertEqual(result["model"], "coderunner-qwen3.5-4b")
        self.assertFalse(result["vision"])                            # vision check failed: text only
        saved = json.loads((self.home / "config.json").read_text())["models"]
        self.assertEqual((saved["runtime"], saved["base_url"], saved["chat"], saved["vision"], saved["context"]),
                         ("ollama", rt.base_url, "coderunner-qwen3.5-4b", False, 32768))
        ep = manager.endpoint(Config.load(self.home / "config.json"))
        self.assertEqual((ep.base_url, ep.model, ep.vision), (rt.base_url, "coderunner-qwen3.5-4b", False))

    def test_setup_stops_after_two_fallbacks(self):
        rt = FakeRuntime()
        smoke = fake_smoke({f"coderunner-{e['id']}": False for e in catalog.by_role("chat")})
        result = manager.setup(self.config(), runtime=rt, hw=apple(128, "Apple M4 Max"), progress=lambda m: None,
                               embedding=False, smoke=smoke)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["attempts"]), 1 + manager.MAX_FALLBACKS)
        self.assertFalse((self.home / "config.json").exists())

    def test_forced_model_is_not_replaced(self):
        rt, smoke = FakeRuntime(), fake_smoke({"coderunner-qwen3.5-4b": False})
        result = manager.setup(self.config(), model_id="qwen3.5-4b", runtime=rt, hw=apple(18),
                               progress=lambda m: None, embedding=False, smoke=smoke)
        self.assertFalse(result["ok"])
        self.assertEqual(rt.pulled, ["qwen3.5-4b"])
        too_small = manager.setup(self.config(), model_id="qwen3.5-0.8b", runtime=rt, hw=apple(18),
                                  progress=lambda m: None, smoke=smoke)
        self.assertFalse(too_small["ok"])

    def test_license_acceptance_is_recorded(self):
        entry = dict(catalog.get("qwen3.5-4b"), license={"spdx": "gemma", "link": "https://ai.google.dev/gemma/terms"})
        real_get = catalog.get
        with mock.patch.object(catalog, "get", lambda i: entry if i == "qwen3.5-4b" else real_get(i)):
            cfg = self.config()
            blocked = manager.setup(cfg, model_id="qwen3.5-4b", runtime=FakeRuntime(), hw=apple(18),
                                    progress=lambda m: None, embedding=False, smoke=fake_smoke({}))
            self.assertFalse(blocked["ok"])
            self.assertEqual(blocked["needs_license"]["model"], "qwen3.5-4b")
            accepted = manager.setup(cfg, model_id="qwen3.5-4b", runtime=FakeRuntime(), hw=apple(18), accept_license=True,
                                     progress=lambda m: None, embedding=False, smoke=fake_smoke({}))
        self.assertTrue(accepted["ok"], accepted)
        self.assertIn("gemma", Config.load(self.home / "config.json").get("models.accepted_licenses"))

    def test_external_endpoint(self):
        cfg = self.config(runtime="openai", base_url="http://10.0.0.2:1234/v1", chat="served-model")
        with mock.patch.object(runtimes.OpenAIRuntime, "running", return_value=True):
            result = manager.setup(cfg, hw=apple(18), progress=lambda m: None, smoke=fake_smoke({}))
        self.assertTrue(result["ok"], result)
        with mock.patch.dict(os.environ, {"CODERUNNER_MODEL_API_KEY": "k"}):
            ep = manager.endpoint(cfg)
        self.assertEqual((ep.base_url, ep.model, ep.api_key), ("http://10.0.0.2:1234/v1", "served-model", "k"))

    def test_no_runtime_gives_install_hint(self):
        with mock.patch.object(runtimes.OllamaRuntime, "available", return_value=False), \
                mock.patch.object(runtimes.LlamaCppRuntime, "available", return_value=False):
            result = manager.setup(self.config(), hw=apple(18), progress=lambda m: None)
        self.assertFalse(result["ok"])
        self.assertIn("brew install ollama", result["error"])


# ---- measuring this machine ------------------------------------------------------------------

def sse(*events) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"


class BenchTests(TempHome):
    def test_measure_reads_the_stream_and_prefers_llama_cpp_timings(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            chunk = {"choices": [{"delta": {"content": "Delhi"}}]}
            usage = {"choices": [], "usage": {"prompt_tokens": 4100, "completion_tokens": 128}}
            if body["max_tokens"] == bench.WRITE_TOKENS and len(requests) == 4:    # second run, as llama.cpp
                usage["timings"] = {"prompt_n": 4100, "prompt_per_second": 812.5, "predicted_n": 128,
                                    "predicted_per_second": 21.3}
            return httpx.Response(200, content=sse(chunk, chunk, chunk, usage),
                                  headers={"content-type": "text/event-stream"})
        transport = httpx.MockTransport(handler)
        plain = asyncio.run(REAL_MEASURE("http://127.0.0.1:9/v1", "m", transport=transport))
        self.assertEqual((plain["prompt_tokens"], plain["written_tokens"]), (4100, 128))
        self.assertGreater(plain["read_tok_s"], 0)
        timed = asyncio.run(REAL_MEASURE("http://127.0.0.1:9/v1", "m", transport=transport))
        self.assertEqual((timed["read_tok_s"], timed["gen_tok_s"]), (812.5, 21.3))
        self.assertEqual(timed["step_s"], round(fit.step_seconds(812.5, 21.3), 1))
        # The page is new every time (no prompt cache helps), long enough, and asks for no reasoning.
        pages = [r["messages"][-1]["content"] for r in requests if r["max_tokens"] == bench.WRITE_TOKENS]
        self.assertNotEqual(pages[0], pages[1])
        self.assertGreater(len(pages[0]), 7000)            # about 3,900 tokens
        self.assertEqual(requests[1]["reasoning_effort"], "none")

    def test_measurements_calibrate_every_candidate_on_this_machine(self):
        cfg, hw, entry = self.config(), apple(18), catalog.get("qwen3.5-9b")
        spec = fit.candidate(entry, hw, cfg, "chat", "ollama")
        self.assertFalse(manager._hardware(cfg, apple(18)).measured)
        measured = {"read_tok_s": spec["read_tok_s"] / 2, "gen_tok_s": spec["tok_s"] / 2, "prompt_tokens": 4000,
                    "first_token_s": 1.0, "step_s": fit.step_seconds(spec["read_tok_s"] / 2, spec["tok_s"] / 2)}
        cal = bench.calibration(hw, entry, "ollama", measured)
        self.assertAlmostEqual(cal["gen_factor"], 0.5, places=2)
        bench.save(cal)
        here = manager._hardware(cfg, apple(18))
        self.assertIn("qwen3.5-9b on ollama", here.measured)
        again = fit.candidate(entry, here, cfg, "chat", "ollama")
        self.assertAlmostEqual(again["tok_s"], spec["tok_s"] / 2, delta=0.1)
        self.assertAlmostEqual(again["read_tok_s"], spec["read_tok_s"] / 2, delta=1)
        self.assertIn("(measured here)", " ".join(again["reasons"]))
        # Other models get the same machine factors; another machine ignores them.
        four, four_spec = (fit.candidate(catalog.get("qwen3.5-4b"), h, cfg, "chat", "ollama") for h in (here, hw))
        self.assertAlmostEqual(four["read_tok_s"], four_spec["read_tok_s"] / 2, delta=1)
        self.assertFalse(manager._hardware(cfg, apple(24, "Apple M4 Pro")).measured)
        # A wildly off measurement says more about the run than the machine.
        wild = bench.calibration(hw, entry, "ollama", dict(measured, gen_tok_s=0.01, read_tok_s=1e6))
        self.assertEqual(wild["gen_factor"], bench.FACTOR_RANGE[0])
        self.assertLessEqual(wild["read_k"], bench.FACTOR_RANGE[1] * spec["read_tok_s"] * entry["params_b"] + 1)

    def test_setup_times_the_pick_and_moves_on_when_it_is_far_slower_here(self):
        # On a 128 GB M4 Max the dense 27B is picked by its scores (~10 s per agent step estimated). If this
        # Measure on a 64 GB Mac where Qwen3.8-27B is top pick on Ollama; if it measures far slower than
        # estimated, setup switches to the next faster candidate and deletes the first.
        cfg, hw = self.config(), apple(64, "Apple M4 Max")

        def twelfth(model):
            est = fit.candidate(catalog.get(model.removeprefix("coderunner-")), hw, cfg, "chat", "ollama")
            return est["read_tok_s"] / 12, est["tok_s"] / 12
        rt, measure = FakeRuntime(), fake_measure(twelfth)
        result = manager.setup(cfg, runtime=rt, hw=hw, progress=lambda m: None, embedding=False, decision=False,
                               smoke=fake_smoke({}), measure=measure)
        self.assertTrue(result["ok"], result)
        self.assertEqual(measure.calls[0], "coderunner-qwen3.8-27b")
        self.assertEqual(len(measure.calls), 2)
        self.assertEqual((result["entry"], rt.removed), (result["attempts"][1]["model"], ["qwen3.8-27b"]))
        self.assertGreater(result["attempts"][0]["speed"]["step_s"], fit.MAX_STEP_S)
        self.assertLess(result["attempts"][1]["speed"]["step_s"], fit.MAX_STEP_S)
        self.assertIn("slower on this machine", result["attempts"][0]["error"])
        self.assertTrue(bench.load(hw))
        # Measured as estimated: the pick stays, and plans now use the measurement.
        rt = FakeRuntime()
        (self.home / "assistant" / "models" / "bench.json").unlink()
        same = fake_measure(lambda m: (fit.candidate(catalog.get(m.removeprefix("coderunner-")), hw, cfg, "chat",
                                                     "ollama")["read_tok_s"], 18.0))
        result = manager.setup(self.config(), runtime=rt, hw=hw, progress=lambda m: None, embedding=False,
                               decision=False, smoke=fake_smoke({}), measure=same)
        self.assertEqual((result["entry"], rt.removed, len(same.calls)), ("qwen3.8-27b", [], 1))
        self.assertIn("qwen3.8-27b", manager.plan(self.config(), hw=hw, runtime=FakeRuntime())["hardware"]["measured"])

    def test_bench_command_times_the_configured_model_and_reports(self):
        cfg = self.config(runtime="prism", chat="coderunner-bonsai-2-27b", base_url="http://127.0.0.1:8082/v1",
                          context=32768)
        hw = apple(18, sandbox=1)
        endpoint = manager.Endpoint(base_url="http://127.0.0.1:8082/v1", model="coderunner-bonsai-2-27b")
        measure = fake_measure(lambda m: (70.0, 9.0))
        with mock.patch.object(manager, "ensure_running", return_value=endpoint):
            result = manager.bench_here(cfg, measure=measure, hw=hw)
        self.assertTrue(result["ok"], result)
        self.assertEqual(measure.calls, ["coderunner-bonsai-2-27b"])
        self.assertEqual((result["calibration"]["model"], result["calibration"]["runtime"]), ("bonsai-2-27b", "prism"))
        self.assertIn("| prompt reading | 70 tok/s", result["report"])
        self.assertTrue(bench.load(hw))
        with mock.patch.object(manager, "ensure_running", return_value=manager.Endpoint(base_url="", model="")):
            self.assertFalse(manager.bench_here(self.config(), measure=measure, hw=hw)["ok"])

    def test_report_is_shareable(self):
        cfg, hw, entry = self.config(), apple(18), catalog.get("qwen3.5-9b")
        cal = bench.calibration(hw, entry, "ollama", {"read_tok_s": 240.0, "gen_tok_s": 12.6, "prompt_tokens": 4100,
                                                       "first_token_s": 17.1, "step_s": 20.2})
        picks = fit.recommend(bench.apply(hw, cal), cfg, runtime="ollama")[:3]
        text = bench.report(bench.apply(hw, cal), cal, picks, {"name": "ollama", "version": "0.40.3"})
        for part in ("Apple M3 Pro", "18 GB RAM", "ollama 0.40.3", "| prompt reading | 240 tok/s", "qwen3.5-9b"):
            self.assertIn(part, text)
        for private in (str(self.home), os.environ.get("USER") or "nobody-here", str(pathlib.Path.home())):
            self.assertNotIn(private, text)
        self.assertIn("Not measured yet", bench.report(hw, None, picks))


class OllamaVersionTests(TempHome):
    def test_versions_are_parsed_and_compared(self):
        self.assertEqual(runtimes.parse_version("ollama version is 0.40.3"), (0, 40, 3))
        self.assertEqual(runtimes.parse_version("0.36.0-rc2"), (0, 36, 0))
        self.assertIsNone(runtimes.parse_version("unknown"))
        rt = runtimes.OllamaRuntime(self.config())
        with mock.patch.object(rt, "version", return_value=(0, 32, 15)):
            self.assertTrue(rt.outdated())
        with mock.patch.object(rt, "version", return_value=(0, 40, 3)):
            self.assertFalse(rt.outdated())
        with mock.patch.object(rt, "version", return_value=None):
            self.assertFalse(rt.outdated())            # nothing installed: setup's install hint covers it

    def test_old_ollama_gets_coderunners_own_copy(self):
        rt = runtimes.OllamaRuntime(self.config())
        versions = iter([(0, 32, 15), (0, 32, 15), (0, 40, 3)])

        def install(progress=None):
            rt.managed_dir.mkdir(parents=True)
            (rt.managed_dir / "ollama").write_text("#!/bin/sh\n")
            (rt.managed_dir / "ollama").chmod(0o755)
            return str(rt.managed_dir / "ollama")
        with mock.patch.object(rt, "version", side_effect=lambda: next(versions)), \
                mock.patch.object(rt, "running", return_value=False), \
                mock.patch.object(rt, "can_install", return_value=True), \
                mock.patch.object(rt, "install", side_effect=install) as installed:
            rt.ensure_current(lambda m: None)
        installed.assert_called_once()
        self.assertEqual(rt.binary(), str(rt.managed_dir / "ollama"))
        self.assertIn(f"ollama-{rt.RELEASE}", rt.binary())

    def test_an_old_server_someone_else_runs_is_explained(self):
        rt = runtimes.OllamaRuntime(self.config())
        with mock.patch.object(rt, "version", return_value=(0, 32, 15)), \
                mock.patch.object(rt, "available", return_value=True), \
                mock.patch.object(rt, "running", return_value=True), \
                mock.patch.object(rt, "stop", return_value=False), \
                mock.patch.object(rt, "pull", side_effect=AssertionError("setup must stop before pulling")):
            with self.assertRaisesRegex(RuntimeError, "Ollama 0.32.15 is running .* quit it"):
                rt.ensure_current()
            result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=lambda m: None)
        self.assertFalse(result["ok"])
        self.assertIn("0.36.0 or later", result["error"])

    def test_install_refuses_a_build_that_does_not_match_its_checksum(self):
        rt = runtimes.OllamaRuntime(self.config())

        class Response:
            headers = {"content-length": "23"}

            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                yield b"not the published build"

        class Stream:
            def __enter__(self):
                return Response()

            def __exit__(self, *exc):
                return False

        with mock.patch.object(runtimes.platform, "system", return_value="Darwin"), \
                mock.patch.object(runtimes.httpx, "stream", return_value=Stream()):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                rt.install()
        self.assertFalse((rt.managed_dir / "ollama").exists())
        self.assertEqual(list(rt.managed_dir.glob("*.tgz")), [])

    def test_there_is_a_pinned_build_for_macs_and_linux_and_none_for_windows(self):
        build = runtimes.OllamaRuntime.build
        self.assertEqual(build("Darwin", "arm64")[0], "ollama-darwin.tgz")
        self.assertEqual(build("Linux", "x86_64")[0], "ollama-linux-amd64.tar.zst")
        self.assertEqual(build("Linux", "AMD64")[0], "ollama-linux-amd64.tar.zst")
        self.assertEqual(build("Linux", "arm64")[0], "ollama-linux-arm64.tar.zst")
        self.assertEqual(build("Darwin", "aarch64")[0], "ollama-darwin.tgz")
        self.assertIsNone(build("Windows", "AMD64"))
        for asset, sha256 in runtimes.OllamaRuntime.BUILDS.values():
            self.assertRegex(sha256, r"^[0-9a-f]{64}$", asset)

    def test_the_linux_archive_keeps_its_binary_under_bin(self):
        rt = runtimes.OllamaRuntime(self.config())
        with mock.patch.object(runtimes.shutil, "which", return_value=None), \
                mock.patch.object(rt, "APP_BINARY", "/nonexistent/ollama"):
            self.assertIsNone(rt.binary())
            (rt.managed_dir / "bin").mkdir(parents=True)
            (rt.managed_dir / "bin" / "ollama").write_text("#!/bin/sh\n")
            (rt.managed_dir / "bin" / "ollama").chmod(0o755)
            self.assertEqual(rt.binary(), str(rt.managed_dir / "bin" / "ollama"))

    @unittest.skipUnless(shutil.which("zstd") and shutil.which("tar"), "needs the zstd and tar commands")
    def test_a_zstd_archive_is_unpacked_as_the_linux_release_is(self):
        work = self.home / "work"
        (work / "src" / "bin").mkdir(parents=True)
        (work / "src" / "bin" / "ollama").write_text("#!/bin/sh\necho ollama version is 0.40.3\n")
        subprocess.run(["tar", "-cf", str(work / "o.tar"), "-C", str(work / "src"), "bin"], check=True)
        subprocess.run(["zstd", "-q", str(work / "o.tar"), "-o", str(work / "o.tar.zst")], check=True)
        (work / "out").mkdir()
        runtimes._unpack(work / "o.tar.zst", work / "out")
        self.assertIn("0.40.3", (work / "out" / "bin" / "ollama").read_text())
        (work / "bad.tar.zst").write_bytes(b"not an archive")
        with self.assertRaisesRegex(RuntimeError, "install zstd"):
            runtimes._unpack(work / "bad.tar.zst", work / "out")


    def test_a_zstd_archive_is_unpacked_without_the_systems_zstd(self):
        try:
            import zstandard
        except ImportError:
            self.skipTest("needs the zstandard package (installed with CodeRunner on Linux before Python 3.14)")
        import io
        import tarfile
        work = self.home / "work2"
        work.mkdir()
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode="w") as tar:
            data = b"#!/bin/sh\necho ollama version is 0.40.3\n"
            info = tarfile.TarInfo("bin/ollama")
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
        (work / "o.tar.zst").write_bytes(zstandard.ZstdCompressor().compress(packed.getvalue()))
        (work / "out").mkdir()
        with mock.patch.object(runtimes.shutil, "which", return_value=None):          # no tar, no zstd
            runtimes._unpack(work / "o.tar.zst", work / "out")
            self.assertTrue(os.access(work / "out" / "bin" / "ollama", os.X_OK))
            (work / "bad.tar.zst").write_bytes(b"not an archive")
            with self.assertRaisesRegex(RuntimeError, "could not be unpacked"):
                runtimes._unpack(work / "bad.tar.zst", work / "out")


class OwnRuntimeTests(TempHome):
    """A machine with nothing to run models with gets CodeRunner's own copy of Ollama."""

    def bare(self, can_install=True):
        stack = contextlib.ExitStack()
        for target, name, value in ((runtimes.OllamaRuntime, "available", False),
                                    (runtimes.LlamaCppRuntime, "available", False),
                                    (runtimes.OllamaRuntime, "can_install", can_install)):
            stack.enter_context(mock.patch.object(target, name, return_value=value))
        return stack

    def test_setup_plans_for_its_own_ollama_and_says_so(self):
        with self.bare(), mock.patch.object(manager, "_prism", return_value=None):
            result = manager.plan(self.config(), hw=apple(18))
        self.assertEqual((result["runtime"]["name"], result["runtime"]["available"]), ("ollama", False))
        self.assertEqual(result["model"]["id"], "qwen3.5-9b")
        self.assertIn("setup fetches its own copy of Ollama", result["notes"][0])
        with self.bare(can_install=False):                               # Windows: no pinned build
            result = manager.plan(self.config(), hw=apple(18))
        self.assertIsNone(result["runtime"])
        self.assertIn("ollama.com/download", " ".join(result["notes"]))

    def test_the_copy_is_installed_once_and_only_where_ollama_is_missing(self):
        rt, said = runtimes.OllamaRuntime(self.config()), []
        state = {"installed": False}

        def install(progress=None):
            state["installed"] = True
        with mock.patch.object(rt, "available", side_effect=lambda: state["installed"]), \
                mock.patch.object(rt, "can_install", return_value=True), \
                mock.patch.object(rt, "install", side_effect=install) as installed:
            self.assertIsNone(manager._own_ollama(rt, apple(18), said.append))
            self.assertIsNone(manager._own_ollama(rt, apple(18), said.append))       # there now: nothing to do
        installed.assert_called_once()
        self.assertIn("CodeRunner's own copy of Ollama", said[0])
        llama = FakeLlama()
        llama.available = lambda: False
        self.assertIsNone(manager._own_ollama(llama, apple(18), said.append))          # another runtime's business

    def test_a_failed_download_is_reported_with_what_to_do(self):
        with self.bare(), mock.patch.object(runtimes.OllamaRuntime, "install", side_effect=RuntimeError("offline")):
            result = manager.setup(self.config(), hw=apple(18), progress=lambda m: None)
        self.assertFalse(result["ok"])
        self.assertIn("could not be installed (offline)", result["error"])
        self.assertIn("brew install ollama", result["error"])


class NoGpuTests(TempHome):
    def test_a_machine_without_a_gpu_is_told_models_will_be_slow_before_it_downloads_one(self):
        notes = manager.plan(self.config(), hw=linux(32), runtime=FakeRuntime())["notes"]
        self.assertRegex(" ".join(notes), r"No GPU was found here \(AMD Ryzen 9 7950X\), so models run on the processor, "
                                          r"slowly: about \d+ tokens a second and \d+ s a step with ")
        said = []
        manager.setup(self.config(), runtime=FakeRuntime(), hw=linux(32), progress=said.append, embedding=False,
                      decision=False, smoke=fake_smoke({}), measure=False)
        self.assertTrue(said[0].startswith("warning: No GPU was found here"), said[:1])
        for hw in (apple(18), nvidia(24)):
            self.assertNotIn("No GPU", " ".join(manager.plan(self.config(), hw=hw, runtime=FakeRuntime())["notes"]))


class DiskRoomTests(TempHome):
    """Downloads never take the disk under files.min_free_gb; setup moves on to a model that fits."""

    def free(self, gb):
        return mock.patch.object(manager, "_free_gb", return_value=gb)

    def test_room_is_what_is_free_less_the_floor(self):
        with self.free(11.4):
            self.assertEqual([round(x, 1) for x in manager._room_gb(self.config())], [6.4, 11.4, 5.0])
            self.assertIsNone(manager._no_room(self.config(), 6.0))
            self.assertIn("a 7.8 GB download, and 6.4 GB can be used (11.4 GB free, 5 GB kept free)",
                          manager._no_room(self.config(), 7.8))
            cfg = Config({"files": {"min_free_gb": 1}}, path=self.home / "config.json")
            self.assertIsNone(manager._no_room(cfg, 7.8))

    def test_setup_skips_a_model_the_disk_has_no_room_for_and_takes_the_next(self):
        rt, said = FakeRuntime(), []
        with self.free(11.0):                 # 6 GB of room: Qwen3.5-9B (6.6 GB) does not fit, the 4B model does
            result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=said.append,
                                   embedding=False, decision=False, smoke=fake_smoke({}), measure=False)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["entry"], rt.pulled), ("qwen3.5-4b", ["qwen3.5-4b"]))
        self.assertTrue(any(line.startswith("skipping qwen3.5-9b") and "6.0 GB can be used" in line for line in said))
        self.assertRegex(said[-1], r"qwen3.5-9b ranks higher but the disk has no room for it: free \d\.\d GB more")
        self.assertIn("qwen3.5-9b", result["skipped_for_disk"][0])

    def test_nothing_is_downloaded_when_no_model_fits_the_disk(self):
        rt = FakeRuntime()
        with self.free(6.0):
            result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=lambda m: None,
                                   embedding=False, decision=False, smoke=fake_smoke({}), measure=False)
            asked = manager.setup(self.config(), model_id="qwen3.5-9b", runtime=rt, hw=apple(18),
                                  progress=lambda m: None, embedding=False, decision=False, smoke=fake_smoke({}))
        self.assertEqual(rt.pulled, [])
        for outcome in (result, asked):
            self.assertFalse(outcome["ok"])
            self.assertIn("Not enough disk space", outcome["error"])
            self.assertLess(len(outcome["error"]), 120)                         # said in one short sentence or two
            self.assertNotIn("min_free_gb", outcome["error"])
        self.assertIn("qwen3.5-9b", asked["error"])
        self.assertRegex(asked["error"], r"needs 6\.\d GB and 1\.0 GB can be used\. Free 5\.\d GB")
        self.assertRegex(result["error"], r"qwen3.5-4b.*Free \d\.\d GB")       # the least to free is named

    def test_a_model_already_here_needs_no_room(self):
        rt = FakeRuntime()
        rt.installed = lambda entry: True
        with self.free(0.5):
            result = manager.setup(self.config(), runtime=rt, hw=apple(18), progress=lambda m: None,
                                   embedding=False, decision=False, smoke=fake_smoke({}), measure=False)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["entry"], rt.pulled), ("qwen3.5-9b", []))

    def test_the_plan_says_when_its_pick_does_not_fit_the_disk(self):
        with self.free(8.0):
            notes = manager.plan(self.config(), hw=apple(18), runtime=FakeRuntime())["notes"]
        self.assertRegex(" ".join(notes), r"qwen3.5-9b is a 6\.\d GB download, and 3\.0 GB can be used \(8\.0 GB free")
        self.assertEqual(manager.plan(self.config(), hw=apple(18), runtime=FakeRuntime())["notes"], [])


@unittest.skipUnless(os.environ.get("CODERUNNER_LIVE_MODELS") == "1", "set CODERUNNER_LIVE_MODELS=1 for live HF calls")
class LiveHubTests(unittest.TestCase):
    def test_manifest_and_license(self):
        hub = Hub(cache_dir=False)
        files = hub.gguf_files("unsloth/Qwen3.5-4B-GGUF", "Q4_K_M")
        self.assertEqual([f["kind"] for f in files], ["model", "mmproj"])
        self.assertTrue(files[0]["file"].endswith("Q4_K_M.gguf"))
        self.assertEqual(len(files[0]["sha256"]), 64)
        self.assertEqual(hub.resolve_license("ggml-org/embeddinggemma-300M-GGUF")["spdx"], "gemma")


if __name__ == "__main__":
    unittest.main()


# ---- Prism's llama.cpp build for ternary Bonsai models -------------------------------

class FakePrism(FakeRuntime):
    name = "prism"
    base_url = "http://127.0.0.1:8/v1"


class BenchCandidatesTests(TempHome):
    def test_measures_the_best_candidates_keeps_the_best_and_deletes_the_rest(self):
        from coderunner.models import tasks
        ollama, prism, hw = FakeRuntime(), FakePrism(), apple(18, "Apple M3 Pro", sandbox=1)
        outcomes = {"bonsai-2.5-27b": (5, 760.0), "bonsai-2.5-27b-compact": (5, 920.0),
                    "bonsai-2-27b": (4, 820.0), "bonsai-2-27b-compact": (4, 990.0), "qwen3.5-9b": (3, 410.0)}
        served = []

        async def run_tasks(config, progress=None):
            model = manager.entry_id(config)
            served.append(model)
            passed, seconds = outcomes[model]
            record = {"suite": tasks.SUITE, "date": "2026-11-26", "machine": bench.fingerprint(hw), "model": model,
                      "runtime": config.get("models.runtime"), "runtime_version": "", "build": model,
                      "passed": passed, "total": 5, "seconds": seconds, "median_request_s": 5.0, "slowed": False,
                      "tasks": [{"name": t.name, "passed": True, "seconds": seconds / 5} for t in tasks.TASKS]}
            tasks.save(record)
            return {"ok": True, **record}

        def choose(config, hw=None, name=None):
            return prism if (name or config.get("models.runtime")) == "prism" else ollama
        with mock.patch.object(manager, "choose_runtime", side_effect=choose), \
                mock.patch.object(manager, "_prism", return_value=prism):
            cfg = self.config()
            result = manager.bench_candidates(cfg, count=3, hw=hw, progress=lambda m: None, run_tasks=run_tasks,
                                              smoke=fake_smoke({}), decision=False, embedding=False)
        self.assertEqual(served[:2], ["bonsai-2.5-27b", "bonsai-2.5-27b-compact"])
        # Of the models that passed all 5 tasks, the fastest wins and the rest are removed.
        self.assertEqual(result["model"]["id"], "bonsai-2.5-27b")
        self.assertEqual(result["model"]["tasks"]["passed"], 5)
        self.assertEqual(manager.entry_id(cfg), "bonsai-2.5-27b")
        self.assertIn("bonsai-2.5-27b-compact", prism.removed)


class MeasuredTests(TempHome):
    def test_a_slowed_run_counts_until_a_clean_one(self):
        from coderunner.models import tasks
        hw = apple(18, "Apple M3 Pro", sandbox=1)

        def run(model, seconds, slowed, passed=5):
            tasks.save({"suite": tasks.SUITE, "date": "2026-10-13", "machine": bench.fingerprint(hw), "model": model,
                        "runtime": "prism", "runtime_version": "b1", "build": f"{model}-{slowed}", "passed": passed,
                        "total": 5, "seconds": seconds, "slowed": slowed, "median_request_s": 5.0,
                        "tasks": [{"name": t.name, "passed": True, "seconds": seconds / 5} for t in tasks.TASKS]})
        run("bonsai-2-27b-compact", 1113.0, True)
        found = manager._measured(hw)
        self.assertEqual(found[("bonsai-2-27b-compact", "prism")]["seconds"], 1113.0)     # not "unmeasured"
        run("bonsai-2-27b-compact", 900.0, False)
        run("bonsai-2-27b-compact", 1500.0, True)                     # a later slowed run does not replace it
        self.assertEqual(manager._measured(hw)[("bonsai-2-27b-compact", "prism")]["seconds"], 900.0)


class ApacheHub:
    """Answers a repo's licence lookup without the network (tests must not reach Hugging Face)."""

    def resolve_license(self, repo):
        return {"repo": repo, "spdx": "apache-2.0", "name": "apache-2.0", "link": f"https://huggingface.co/{repo}"}


class PrismTests(TempHome):
    def test_pinned_builds_per_platform(self):
        self.assertEqual(runtimes.PrismRuntime.build_for(apple(18))[0], "macos-arm64")
        self.assertEqual(runtimes.PrismRuntime.build_for(nvidia(24))[0], "linux-cuda-12.4-x64")
        self.assertIsNone(runtimes.PrismRuntime.build_for(hardware.Hardware(os="darwin", arch="x86_64")))
        self.assertIsNone(runtimes.PrismRuntime.build_for(hardware.Hardware(os="windows", arch="x86_64")))

    def test_server_flags(self):
        rt = runtimes.PrismRuntime(self.config(), hw=apple(18))
        cmd = rt.command({"model": "/m.gguf", "mmproj": "/p.gguf", "ctx": 32768, "alias": "bonsai-2-27b"})
        joined = " ".join(cmd)
        for flag in ("--mmproj /p.gguf", "--jinja", "-ctk q8_0", "-ctv q8_0", "--load-mode none", "-c 32768",
                     "-np 3", "--kv-unified"):
            self.assertIn(flag, joined)
        self.assertIn(json.dumps({"reasoning_effort": "medium"}), cmd)
        self.assertIn("--sleep-idle-seconds 600", joined)             # frees the model's memory when unused
        awake = runtimes.PrismRuntime(self.config(idle_unload_minutes=0), hw=apple(18))
        self.assertNotIn("--sleep-idle-seconds", awake.command({"model": "/m.gguf", "ctx": 8192, "alias": "b"}))
        self.assertEqual(runtimes.OllamaRuntime(self.config())._env()["OLLAMA_KEEP_ALIVE"], "600s")
        # Locked weights cannot be reclaimed; with a second model loaded an 18 GB Mac froze. Opt-in only.
        locked = runtimes.PrismRuntime(self.config(lock_weights=True), hw=apple(18))
        self.assertIn("--load-mode mlock", " ".join(locked.command({"model": "/m.gguf", "ctx": 8192,
                                                                     "alias": "bonsai-2-27b"})))

    def test_install_refuses_a_build_that_does_not_match_its_checksum(self):
        rt = runtimes.PrismRuntime(self.config(), hw=apple(18))

        class Response:
            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                yield b"not the published build"

        class Stream:
            def __enter__(self):
                return Response()

            def __exit__(self, *exc):
                return False

        with mock.patch.object(runtimes.httpx, "stream", return_value=Stream()):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                rt.install()
        self.assertIsNone(rt.binary())
        self.assertEqual(list(rt.bin_dir.glob("*.tar.gz")), [])

    def test_removing_one_build_keeps_the_files_another_uses(self):
        rt = runtimes.PrismRuntime(self.config(), hw=apple(18))
        regular, compact = catalog.get("bonsai-2-27b"), catalog.get("bonsai-2-27b-compact")
        folder = rt.root / regular["artifacts"]["prism"]["repo"].replace("/", "--")
        folder.mkdir(parents=True)
        files = {name: folder / name for name in (regular["artifacts"]["prism"]["file"], compact["artifacts"]["prism"]["file"],
                                                  regular["artifacts"]["prism"]["mmproj"])}
        for path in files.values():
            path.write_bytes(b"x")
        shared = folder / regular["artifacts"]["prism"]["mmproj"]
        rt._save({"models": {"bonsai-2-27b": {"model": str(folder / regular["artifacts"]["prism"]["file"]), "mmproj": str(shared)},
                             "bonsai-2-27b-compact": {"model": str(folder / compact["artifacts"]["prism"]["file"]),
                                                      "mmproj": str(shared)}}})
        rt.remove(compact)
        self.assertFalse((folder / compact["artifacts"]["prism"]["file"]).exists())
        self.assertTrue(shared.exists() and (folder / regular["artifacts"]["prism"]["file"]).exists())
        # An interrupted download of a build is deleted with it.
        partial = folder / (compact["artifacts"]["prism"]["file"] + ".part")
        ranges = folder / (compact["artifacts"]["prism"]["file"] + ".part.json")
        partial.write_bytes(b"x")
        ranges.write_text("{}")
        rt.remove(compact)
        self.assertFalse(partial.exists() or ranges.exists())
        self.assertTrue(shared.exists())

    def test_memory_estimate_counts_the_8_bit_cache(self):
        entry = catalog.get("bonsai-2-27b")
        est = fit.estimate(entry, apple(18), 32768, "prism")
        f16 = fit.kv_gib(entry, 32768)
        self.assertAlmostEqual(est["need_gb"], round((7.21 + 0.63) * fit.GB + f16 / 2 + fit.OVERHEAD_GIB, 2), places=2)

    def test_plan_and_setup_rank_across_runtimes(self):
        ollama, prism = FakeRuntime(), FakePrism()
        with mock.patch.object(manager, "choose_runtime", return_value=ollama), \
                mock.patch.object(manager, "_prism", return_value=prism):
            # Bonsai 2 27B keeps 98.2% of Qwen3.8-27B's scores (Prism's measurement), so it leads on 18 and
            # 24 GB and runs on Prism's runtime; Qwen3.5-9B stays an alternative.
            for ram, chip in ((18, "Apple M3 Pro"), (24, "Apple M4 Pro")):
                result = manager.plan(self.config(), hw=apple(ram, chip, sandbox=1))
                self.assertEqual((result["model"]["id"], result["model"]["runtime"]), ("bonsai-2.5-27b", "prism"))
                self.assertEqual(result["runtime"]["name"], "prism")
                self.assertIn("qwen3.5-9b", [c["id"] for c in result["alternatives"]])
            self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")      # from Ollama beside it
            cfg = self.config()
            setup = manager.setup(cfg, hw=apple(24, "Apple M4 Pro", sandbox=1), progress=lambda m: None,
                                  embedding=False, decision=False, smoke=fake_smoke({}))
            self.assertTrue(setup["ok"], setup)
            self.assertEqual((setup["entry"], setup["runtime"]), ("bonsai-2.5-27b", "prism"))
            self.assertEqual(prism.pulled, ["bonsai-2.5-27b"])
            self.assertEqual(ollama.pulled, [])
            self.assertEqual(cfg.get("models.runtime"), "prism")
        # Without the published retention figure, the generic 2-bit loss would put it below Qwen3.5-9B.
        entry = catalog.get("bonsai-2-27b")
        generic = evidence.ability({**entry, "evidence": {"repo": "Qwen/Qwen3.8-27B"}}, 2.13)
        self.assertLess(generic, evidence.ability(catalog.get("qwen3.5-9b"), 4.8) + 0.15)
        self.assertAlmostEqual(evidence.ability(entry, 2.13),
                               evidence.abilities()["Qwen/Qwen3.8-27B"] - evidence.RETENTION_LOSS * 0.018, places=3)
        # A runtime the owner chose is not second-guessed.
        result = manager.plan(self.config(), hw=apple(18), runtime=FakeRuntime())
        self.assertEqual(result["model"]["id"], "qwen3.5-9b")

    def test_setups_own_runtime_pick_does_not_stop_later_comparisons(self):
        ollama, prism = FakeRuntime(), FakePrism()

        def choose(config, hw=None, name=None):
            return prism if (name or config.get("models.runtime")) == "prism" else ollama
        real_prism = manager._prism
        with mock.patch.object(manager, "choose_runtime", side_effect=choose), \
                mock.patch.object(manager, "_prism", side_effect=lambda c, h, rt: real_prism(c, h, rt) and prism):
            with mock.patch.object(runtimes.PrismRuntime, "available", return_value=True):
                cfg = self.config(runtime="prism")          # written by an earlier setup
                ranked = manager.plan(cfg, hw=apple(18, "Apple M3 Pro", sandbox=1))
                self.assertEqual(ranked["model"]["id"], "bonsai-2.5-27b")
                self.assertIn("qwen3.5-9b", [c["id"] for c in ranked["alternatives"]])
                self.assertTrue(all(c["step_s"] < 1e9 for c in ranked["alternatives"]))
                cfg.set("models.runtime_pinned", True)        # the owner's own choice
                pinned = manager.plan(cfg, hw=apple(18, "Apple M3 Pro", sandbox=1))
                self.assertNotIn("qwen3.5-9b", [c["id"] for c in pinned["alternatives"]])
