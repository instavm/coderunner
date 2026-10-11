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

from coderunner.config import Config
from coderunner.models import catalog, fit, hardware, licenses, manager, runtimes, smoketest
from coderunner.models.hub import Hub, HubError

GiB = 1024 ** 3


class TempHome(unittest.TestCase):
    """Keeps every test away from the real ~/.coderunner."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home), "HF_TOKEN": "",
                                           "HF_HOME": str(self.home / "hf")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)

    def config(self, **models):
        return Config({"models": models} if models else {}, path=self.home / "config.json")


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
        out = "NVIDIA GeForce RTX 3060, 12288, 11000\nNVIDIA GeForce RTX 4090, 24564, 23900\n"
        hw = linux(64, {"nvidia-smi": out})
        self.assertEqual((hw.gpu, hw.chip, hw.bandwidth_gbs), ("nvidia", "NVIDIA GeForce RTX 4090", 1008))
        self.assertAlmostEqual(hw.vram_gb, 24564 / 1024)
        self.assertAlmostEqual(hw.budget_gb, 23900 / 1024 - 1)    # free VRAM - 1 GiB
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
        expected = {16: "qwen3.5-4b", 18: "qwen3.5-9b", 24: "qwen3.5-9b", 64: "qwen3.8-27b", 128: "qwen3.5-122b-a10b"}
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
        top18 = fit.recommend(apple(18), cfg)[0]
        self.assertEqual((top18["id"], top18["ctx"], top18["vision"]), ("qwen3.5-9b", 32768, True))
        self.assertEqual(fit.recommend(apple(32, "Apple M2 Pro"), cfg, runtime="llama.cpp")[0]["id"], "gemma-4-26b-a4b")

    def test_nvidia_recommendations(self):
        cfg = self.config()
        picks12 = fit.recommend(nvidia(12, name="NVIDIA GeForce RTX 4070"), cfg, runtime="llama.cpp")
        self.assertSensible(picks12)
        self.assertEqual(picks12[0]["id"], "qwen3.5-9b")
        picks24 = fit.recommend(nvidia(24), cfg, runtime="llama.cpp")
        self.assertSensible(picks24)
        self.assertGreaterEqual(picks24[0]["params_b"], 20)
        self.assertTrue(picks24[0]["vision"])

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
                               "--host", "127.0.0.1", "--port", "8081", "--cache-ram", "0", "-np", "1",
                               "--alias", "coderunner-qwen3.5-9b"])
        self.assertTrue(rt.vision(entry))
        self.assertEqual(rt.base_url, "http://127.0.0.1:8081/v1")
        with self.assertRaisesRegex(RuntimeError, "not been downloaded"):
            rt.prepare(catalog.get("qwen3.5-4b"), 8192)

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
        self.pulled, self.prepared = [], []

    def available(self):
        return True

    def running(self):
        return True

    def pull(self, entry, progress=None):
        self.pulled.append(entry["id"])

    def prepare(self, entry, ctx):
        self.prepared.append((entry["id"], ctx))
        return self.model_id(entry)


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


class ManagerTests(TempHome):
    def test_plan_reports_choice_and_alternatives(self):
        result = manager.plan(self.config(), hw=apple(18), runtime=FakeRuntime())
        self.assertEqual(result["model"]["id"], "qwen3.5-9b")
        self.assertEqual(result["license"]["decision"], "allow")
        self.assertIn("qwen3.5-4b", [c["id"] for c in result["alternatives"]])
        self.assertIn("qwen3.8-27b", [c["id"] for c in result["rejected"]])
        self.assertEqual(result["embedding"]["id"], "embeddinggemma-2")
        self.assertIsNone(manager.plan(self.config(), hw=apple(8, "Apple M1"), runtime=FakeRuntime())["model"])

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
