import asyncio
import hashlib
import json
import os
import pathlib
import struct
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import httpx

from coderunner.core.config import Config
from coderunner.safety import decision as decision_module
from coderunner.models import bench, catalog, evidence, fit, hardware, licenses, manager, runtimes, smoketest
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
    def test_apple_parsing_and_budget(self):
        hw = apple(18)
        self.assertEqual((hw.gpu, hw.chip, hw.bandwidth_gbs), ("apple", "Apple M3 Pro", 150))
        self.assertAlmostEqual(hw.ram_gb, 18)
        self.assertAlmostEqual(hw.gpu_limit_gb, 13.5)            # no wired limit set: 75% of RAM
        self.assertAlmostEqual(hw.budget_gb, 18 - 4 - 2.5 - 1.8)  # RAM - sandbox - 2.5 - 10%
        self.assertEqual(hw.to_dict()["budget_gb"], 9.7)

    def test_apple_wired_limit_caps_budget(self):
        hw = apple(128, "Apple M4 Max", wired_mb=64 * 1024)
        self.assertAlmostEqual(hw.gpu_limit_gb, 64)
        self.assertAlmostEqual(hw.budget_gb, 64)
        self.assertEqual(hw.bandwidth_gbs, 546)

    def test_apple_bandwidth_table(self):
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Max", 36), 300)   # binned
        self.assertEqual(hardware.apple_bandwidth("Apple M3 Max", 64), 400)
        self.assertEqual(hardware.apple_bandwidth("Apple M2 Ultra", 192), 800)
        self.assertEqual(hardware.apple_bandwidth("Apple M1", 8), 68)
        self.assertEqual(hardware.apple_bandwidth("Apple M5", 16), 153)
        self.assertEqual(hardware.apple_bandwidth("Apple M5 Pro", 24), 306)  # scaled until listed
        self.assertEqual(hardware.apple_bandwidth("Apple M9", 16), 100)

    def test_small_mac_has_almost_no_budget(self):
        self.assertAlmostEqual(apple(8, "Apple M1").budget_gb, 8 - 4 - 2.5 - 0.8)
        self.assertEqual(hardware.apple_budget(4, 3, 4), 0.0)

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
        # Ranked by published benchmarks first, then speed: on 64-128 GB Macs Qwen3.8-27B's scores outweigh
        # the 35B-A3B's faster steps (Ollama and llama.cpp builds only here; Prism's builds are below).
        expected = {16: "qwen3.5-4b", 18: "qwen3.5-9b", 24: "qwen3.5-9b", 64: "qwen3.8-27b", 128: "qwen3.8-27b"}
        chips = {8: "Apple M1", 16: "Apple M2", 18: "Apple M3 Pro", 24: "Apple M4 Pro", 32: "Apple M2 Pro",
                 64: "Apple M1 Max", 128: "Apple M4 Max"}
        for ram, chip in chips.items():
            for runtime in ("ollama", "llama.cpp"):
                with self.subTest(ram=ram, runtime=runtime):
                    picks = fit.recommend(apple(ram, chip), cfg, runtime=runtime)
                    self.assertSensible(picks)
                    if ram == 8:
                        self.assertEqual(picks, [])                 # VM + OS leave no room for a 4B model
                        continue
                    if ram in expected:
                        self.assertEqual(picks[0]["id"], expected[ram])
                    self.assertTrue(picks[0]["vision"])
                    self.assertEqual(picks[0]["fit"], "good")
        # Across runtimes, Bonsai 2 27B (Prism's build) leads on 18-24 GB Macs with the lite sandbox: Prism
        # measured it at 98.2% of Qwen3.8-27B's scores, far above Qwen3.5-9B, which outweighs its slower steps
        # (22 s against 12 s on an M3 Pro). On a 16 GB M2 its steps would take 32 s: the two tie, the faster wins.
        two = fit.recommend(apple(16, "Apple M2", sandbox=1), cfg)
        self.assertEqual(two[0]["id"], "qwen3.5-9b")
        self.assertEqual(round(fit.utility(two[0]), 1), round(fit.utility(two[1]), 1))
        for ram, chip in ((18, "Apple M3 Pro"), (24, "Apple M4 Pro")):
            with self.subTest(ram=ram):
                picks = fit.recommend(apple(ram, chip, sandbox=1), cfg)
                self.assertEqual((picks[0]["id"], picks[0]["vision"]), ("bonsai-2-27b", True))
                self.assertIn("prism", catalog.get(picks[0]["id"])["artifacts"])
                nine = next(c for c in picks if c["id"] == "qwen3.5-9b")
                self.assertGreater(picks[0]["ability"], nine["ability"] + 0.2)
                self.assertGreater(picks[0]["step_s"], nine["step_s"])
        # The 1-bit 27B fits too, but its own published scores (MMMU-Pro 60.5 against 75.8 for the model it
        # was made from) put it below Qwen3.5-9B; a base M1's steps would be too slow for Bonsai 2.
        picks18 = fit.recommend(apple(18), cfg)
        onebit = next(c for c in picks18 if c["id"] == "bonsai-27b-1bit")
        self.assertLess(onebit["ability"], next(c for c in picks18 if c["id"] == "qwen3.5-9b")["ability"])
        self.assertIn(f"benchmark ability {onebit['ability']:+.2f}", onebit["reasons"])
        self.assertEqual(fit.recommend(apple(16, "Apple M1", sandbox=1), cfg)[0]["id"], "qwen3.5-9b")
        compact = fit.candidate(catalog.get("bonsai-2-27b-compact"), apple(18), cfg, "chat", "prism")
        self.assertLess(compact["tok_s"], fit.MIN_TOK_S)        # measured factor applied
        self.assertEqual(fit.recommend(apple(18), cfg, runtime="ollama")[0]["id"], "qwen3.5-9b")
        # Qwen3.5-9B scores above Gemma 4 26B-A4B, which is faster but no longer outranks it on speed.
        self.assertEqual(fit.recommend(apple(32, "Apple M2 Pro"), cfg, runtime="llama.cpp")[0]["id"], "qwen3.5-9b")

    def test_nvidia_recommendations(self):
        cfg = self.config()
        rtx4070 = nvidia(12, name="NVIDIA GeForce RTX 4070")
        picks12 = fit.recommend(rtx4070, cfg, runtime="llama.cpp")
        self.assertSensible(picks12)
        # With 64 GB of RAM, a mixture-of-experts model keeps its experts in system RAM (llama.cpp --cpu-moe).
        self.assertEqual((picks12[0]["id"], picks12[0]["offload"]), ("qwen3.6-35b-a3b", True))
        self.assertIn("experts in system RAM", " ".join(picks12[0]["reasons"]))
        self.assertLess(picks12[0]["need_gb"], rtx4070.budget_gb)
        self.assertGreater(picks12[0]["tok_s"], fit.MIN_TOK_S)
        # Ollama cannot place experts in RAM, and a machine without spare RAM cannot either.
        self.assertEqual(fit.recommend(rtx4070, cfg, runtime="ollama")[0]["id"], "qwen3.5-9b")
        small = linux(16, {"nvidia-smi": "NVIDIA GeForce RTX 4070, 12288, 11776\n"})
        self.assertEqual(fit.recommend(small, cfg, runtime="llama.cpp")[0]["id"], "qwen3.5-9b")
        spark = linux(128, {"nvidia-smi": "NVIDIA GB10, [N/A], [N/A], 12.1\n"})
        self.assertFalse(any(c.get("offload") for c in fit.recommend(spark, cfg, runtime="llama.cpp")))
        # On 24 GB, Qwen3.8-27B fits only tightly (94% of the budget), so the 9B that fits well comes first.
        picks24 = fit.recommend(nvidia(24), cfg, runtime="llama.cpp")
        self.assertSensible(picks24)
        self.assertEqual((picks24[0]["id"], picks24[0]["vision"]), ("qwen3.5-9b", True))
        self.assertEqual(next(c for c in picks24 if c["id"] == "qwen3.8-27b")["fit"], "tight")

    def test_eight_gb_macs_run_a_compact_model_text_only(self):
        for chip in ("Apple M1", "Apple M3"):
            with self.subTest(chip=chip):
                top = fit.recommend(apple(8, chip, sandbox=1), self.config())[0]
                self.assertEqual((top["id"], top["vision"]), ("qwen3.5-4b-compact", False))
                self.assertIn("text only: the vision projector does not fit", top["reasons"])

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
                    self.assertIn("ollama", e["artifacts"])
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
        self.assertEqual(info["link"], "https://huggingface.co/org/Model/blob/main/LICENSE.md")
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
        self.assertEqual(seen["modelfile"], "FROM qwen3.5:4b\nPARAMETER num_ctx 16384\n")
        self.assertTrue(pathlib.Path(seen["cmd"][-1]).is_relative_to(self.home))
        self.assertEqual(rt.base_url, "http://127.0.0.1:11434/v1")

    def test_ollama_alias_failure_is_reported(self):
        rt = runtimes.OllamaRuntime()
        failed = mock.Mock(returncode=1, stdout="", stderr="Error: model not found")
        with mock.patch.object(runtimes.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                rt.prepare(catalog.get("qwen3.5-4b"), 8192)

    def test_llama_server_command(self):
        rt = runtimes.LlamaCppRuntime()
        entry = catalog.get("qwen3.5-9b")
        rt._save({"models": {entry["id"]: {"model": "/m/q.gguf", "mmproj": "/m/mmproj.gguf"}}})
        with mock.patch.object(rt, "running", return_value=False):
            self.assertEqual(rt.prepare(entry, 32768), "coderunner-qwen3.5-9b")
        with mock.patch.object(rt, "binary", return_value="llama-server"):
            cmd = rt.command(rt._state()["active"])
        self.assertEqual(cmd, ["llama-server", "-m", "/m/q.gguf", "--mmproj", "/m/mmproj.gguf", "--jinja", "-c", "32768",
                               "--host", "127.0.0.1", "--port", "8081", "--cache-ram", "0", "-np", "2",
                               "--kv-unified", "--alias", "coderunner-qwen3.5-9b"])
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

    def test_decision_model_only_where_it_fits_beside_the_chat_model(self):
        # tev1 4B judged 66 labelled browser steps with no missed commit; it runs beside the chat and
        # embedding models when they leave room (24 GB here), otherwise the chat model judges (18 GB).
        small = manager.plan(self.config(), hw=apple(18, sandbox=1), runtime=FakeRuntime())
        self.assertIsNone(small["decision"])
        big = manager.plan(self.config(), hw=apple(24, "Apple M4 Pro", sandbox=1), runtime=FakeRuntime())
        self.assertEqual(big["decision"]["id"], "tev1-4b")
        entry = catalog.get("tev1-4b")
        self.assertEqual((entry["decision"]["missed"], entry["decision"]["false_asks"]), (0, 7))
        rt, cfg = FakeRuntime(), self.config()

        async def answers(*args, **kwargs):
            return 0.0
        with mock.patch.object(decision_module.DecisionModel, "probability", answers):
            result = manager.setup(cfg, runtime=rt, hw=apple(24, "Apple M4 Pro", sandbox=1), progress=lambda m: None,
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
        # Raising the limit on 96 GB would fit Qwen3.5-122B-A10B, which scores no better than the 27B.
        for ram, chip in ((96, "Apple M2 Max"), (128, "Apple M4 Max")):
            with self.subTest(ram=ram):
                self.assertFalse(manager.plan(self.config(), hw=apple(ram, chip, sandbox=1),
                                              runtime=FakeRuntime())["notes"])

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
            self.assertEqual((result["model"]["id"], result["notes"]), ("qwen3.5-9b", []))   # gain too small to nag
            with mock.patch.object(manager, "WORTH_A_NOTE", 0.0):
                result = manager.plan(self.config(), hw=rtx4070)
        self.assertIn("qwen3.6-35b-a3b would rank higher but needs llama.cpp", " ".join(result["notes"]))
        main, llama = FakeRuntime(), FakeLlama()
        with mock.patch.object(manager, "choose_runtime", return_value=main), \
                mock.patch.object(manager, "_prism", return_value=None), \
                mock.patch.object(manager, "_llama", return_value=llama):
            result = manager.plan(self.config(), hw=rtx4070)
            self.assertEqual((result["model"]["id"], result["model"]["runtime"]), ("qwen3.6-35b-a3b", "llama.cpp"))
            self.assertEqual(result["runtime"]["name"], "llama.cpp")
            self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")        # still from Ollama
            done = manager.setup(self.config(), hw=rtx4070, progress=lambda m: None, embedding=False,
                                 smoke=fake_smoke({}), hub=ApacheHub())     # the GGUF repo's live licence
        self.assertTrue(done["ok"], done)
        self.assertEqual((llama.offloaded, main.prepared), (["qwen3.6-35b-a3b"], []))

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
        smoke = fake_smoke({f"coderunner-{m}": False for m in ("qwen3.5-122b-a10b", "qwen3.8-27b", "gpt-oss-120b",
                                                                   "qwen3.6-35b-a3b", "gemma-4-31b")})
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
        # machine runs at a twelfth of the estimate, a 27B step takes over the 90 s limit: setup moves on to the
        # best remaining candidate under it and deletes the 27B it downloaded.
        cfg, hw = self.config(), apple(128, "Apple M4 Max")

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
        outcomes = {"bonsai-2-27b": (5, 820.0), "bonsai-2-27b-compact": (5, 990.0), "qwen3.5-9b": (3, 410.0)}
        served = []

        async def run_tasks(config, progress=None):
            model = manager.entry_id(config)
            served.append(model)
            passed, seconds = outcomes[model]
            record = {"suite": tasks.SUITE, "date": "2026-10-13", "machine": bench.fingerprint(hw), "model": model,
                      "runtime": config.get("models.runtime"), "runtime_version": "", "build": model,
                      "passed": passed, "total": 5, "seconds": seconds, "median_request_s": 5.0, "paging": False,
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
        self.assertEqual(served, ["bonsai-2-27b", "bonsai-2-27b-compact", "qwen3.5-9b"])
        # Qwen3.5-9B was quickest but passed fewer tasks; of the two that passed all, the faster one.
        self.assertEqual(result["model"]["id"], "bonsai-2-27b")
        self.assertEqual(result["model"]["tasks"]["passed"], 5)
        self.assertEqual(manager.entry_id(cfg), "bonsai-2-27b")
        self.assertEqual((prism.removed, ollama.removed), (["bonsai-2-27b-compact"], ["qwen3.5-9b"]))


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
                     "-np 2", "--kv-unified"):
            self.assertIn(flag, joined)
        self.assertIn(json.dumps({"reasoning_effort": "medium"}), cmd)
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
                self.assertEqual((result["model"]["id"], result["model"]["runtime"]), ("bonsai-2-27b", "prism"))
                self.assertEqual(result["runtime"]["name"], "prism")
                self.assertIn("qwen3.5-9b", [c["id"] for c in result["alternatives"]])
            self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")      # from Ollama beside it
            cfg = self.config()
            setup = manager.setup(cfg, hw=apple(24, "Apple M4 Pro", sandbox=1), progress=lambda m: None,
                                  embedding=False, decision=False, smoke=fake_smoke({}))
            self.assertTrue(setup["ok"], setup)
            self.assertEqual((setup["entry"], setup["runtime"]), ("bonsai-2-27b", "prism"))
            self.assertEqual(prism.pulled, ["bonsai-2-27b"])
            self.assertEqual(ollama.pulled, [])
            self.assertEqual(cfg.get("models.runtime"), "prism")
        # Without the published retention figure, the generic 2-bit loss would put it below Qwen3.5-9B.
        entry = catalog.get("bonsai-2-27b")
        generic = evidence.ability({**entry, "evidence": {"repo": "Qwen/Qwen3.8-27B"}}, 2.13)
        self.assertLess(generic, evidence.ability(catalog.get("qwen3.5-9b"), 4.8) + 0.05)
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
                self.assertEqual(ranked["model"]["id"], "bonsai-2-27b")
                self.assertIn("qwen3.5-9b", [c["id"] for c in ranked["alternatives"]])
                self.assertTrue(all(c["step_s"] < 1e9 for c in ranked["alternatives"]))
                cfg.set("models.runtime_pinned", True)        # the owner's own choice
                pinned = manager.plan(cfg, hw=apple(18, "Apple M3 Pro", sandbox=1))
                self.assertNotIn("qwen3.5-9b", [c["id"] for c in pinned["alternatives"]])
