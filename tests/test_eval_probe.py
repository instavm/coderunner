"""Tests for Tier 0 GPU/ANE probe, StackSpec combo evaluation harness, and ModelPool."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.core.config import Config
from coderunner.eval import (
    StackSpec,
    compare_markdown,
    default_candidates,
    estimate_stack_memory,
    load_history,
    plain_summary,
    qualify_ane_judge,
    save_run,
)
from coderunner.eval.suites import tier1_gate
from coderunner.models import catalog, hardware, hub as hub_module, pool, probe


def _apple(ram_gb: float, chip: str = "Apple M4 Max", sandbox: float = 4.0) -> hardware.Hardware:
    gl = hardware.default_apple_gpu_limit(ram_gb)
    return hardware.Hardware(
        os="darwin", arch="arm64", chip=chip, ram_gb=ram_gb, gpu="apple", vram_gb=gl, gpu_limit_gb=gl,
        bandwidth_gbs=hardware.apple_bandwidth(chip, ram_gb), unified=True, sandbox_memory_gb=sandbox,
        budget_gb=hardware.apple_budget(ram_gb, gl, sandbox),
    )


class ProbeAndEvalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_ane_mlmodel_emitter_and_probe_apply(self):
        dest = self.home / "test_probe.mlmodel"
        out = probe.build_ane_probe_mlmodel(dest, channels=64, width=32, layers=2)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1024)
        hw = _apple(18, "Apple M3 Pro", sandbox=2)
        probed = {
            "ok": True, "version": probe.PROBE_VERSION, "noisy": False,
            "bandwidth_gbs": 148.5, "gpu_fp16_tflops": 4.4, "recommended_max_working_set_gb": 13.32,
            "ane_cores": 16, "ane_fp16_tops": 4.5, "ane_verified": True, "gpu_contention_pct": 1.2,
        }
        updated = probe.apply(hw, probed)
        self.assertAlmostEqual(updated.bandwidth_gbs, 148.5)
        self.assertEqual(updated.ane_cores, 16)
        self.assertTrue(updated.ane_verified)
        self.assertAlmostEqual(updated.ane_tops, 4.5)
        self.assertAlmostEqual(updated.read_k, 660.0)
        # Throttled/noisy runs never overwrite Hardware estimates.
        self.assertEqual(probe.apply(hw, {**probed, "noisy": True}).bandwidth_gbs, hw.bandwidth_gbs)

    def test_stack_memory_and_tier_candidates_across_8gb_to_512gb(self):
        cfg = Config({"models": {"discover": False}}, path=self.home / "config.json")
        # 18 GB: Planner + Fara1.5-4B runs in exclusive mode (takes turns)
        hw18 = _apple(18, "Apple M3 Pro", sandbox=2)
        s18 = StackSpec(id="b25+fara4", label="Bonsai 2.5 + Fara 4B", planner="bonsai-2.5-27b", driver="fara1.5-4b",
                        driver_mode="co_resident")
        mem18 = estimate_stack_memory(s18, hw18, cfg)
        self.assertTrue(mem18["fits"])
        self.assertTrue(mem18["driver_exclusive"])

        # 64 GB: Inkling-2-Mini + Fara1.5-9B + dedicated judge-4b co-reside cleanly
        hw64 = _apple(64, "Apple M4 Max", sandbox=4)
        s64 = StackSpec(id="ink+fara9+judge", label="Inkling + Fara 9B + Judge", planner="inkling-2-mini",
                        driver="fara1.5-9b", driver_mode="co_resident", decision="judge-4b")
        mem64 = estimate_stack_memory(s64, hw64, cfg)
        self.assertTrue(mem64["fits"])
        self.assertFalse(mem64["driver_exclusive"])

        # 8 GB through 512 GB generate tier-appropriate single & multi-model candidates
        for ram, chip in ((8, "Apple M2"), (64, "Apple M4 Max"), (256, "Apple M3 Ultra"), (512, "Apple M3 Ultra")):
            cands = default_candidates(cfg, hw=_apple(ram, chip, sandbox=1 if ram == 8 else 4))
            self.assertTrue(cands, f"expected candidates for {ram} GB")

    def test_persistent_history_and_pareto_report(self):
        hw = _apple(18, "Apple M3 Pro", sandbox=2)
        run_a = save_run({
            "ok": True, "suite": "tier2", "signature": "llama.cpp|qwen3.5-9b|32768|-|-|self|coreml",
            "machine": {"os": "darwin", "chip": "Apple M3 Pro", "ram_gb": 18, "gpu": "apple", "gpus": 0},
            "stack": {"id": "qwen3.5-9b", "label": "Qwen3.5 9B"},
            "memory": {"total_gb": 8.1, "comfort": "comfortable"},
            "passed": 5, "total": 5, "seconds": 360.0, "step_s": 10.0, "cache_hit_ratio": 0.89,
            "swap_delta_gb": 0.1, "missed_commits": 0,
        })
        run_b = save_run({
            "ok": True, "suite": "tier2", "signature": "llama.cpp|bonsai-2.5-27b|32768|-|-|self|coreml",
            "machine": {"os": "darwin", "chip": "Apple M3 Pro", "ram_gb": 18, "gpu": "apple", "gpus": 0},
            "stack": {"id": "bonsai-2.5-27b", "label": "Bonsai 2.5 27B"},
            "memory": {"total_gb": 9.7, "comfort": "tight"},
            "passed": 5, "total": 5, "seconds": 720.0, "step_s": 19.8, "cache_hit_ratio": 0.85,
            "swap_delta_gb": 0.4, "missed_commits": 0,
        })
        loaded = load_history(hw=hw, suite="tier2")
        self.assertEqual(len(loaded), 2)
        summary = plain_summary([run_b, run_a], preference="balanced")
        self.assertEqual(summary["winner"]["id"], "qwen3.5-9b")
        md = compare_markdown([run_b, run_a])
        self.assertIn("Qwen3.5 9B (best)", md)
        self.assertIn("89%", md)

    def test_ane_judge_qualification_gate(self):
        good = qualify_ane_judge({
            "max_tokens": 1024, "compile_cold_s": 30.0, "warm_load_s": 0.8, "peak_rss_mb": 1200,
            "ane_plan_cost_pct": 92.0, "median_ms": 45.0, "missed_commits": 0, "false_alarm_rate": 0.05,
        })
        self.assertTrue(good["qualified"])
        bad = qualify_ane_judge({
            "max_tokens": 96, "compile_cold_s": 120.0, "warm_load_s": 5.0, "peak_rss_mb": 7500,
            "ane_plan_cost_pct": 40.0, "median_ms": 200.0, "missed_commits": 2, "false_alarm_rate": 0.4,
        })
        self.assertFalse(bad["qualified"])
        self.assertGreaterEqual(len(bad["reasons"]), 6)

    def test_model_pool_role_routing_and_yield(self):
        cfg = Config({"models": {"chat": "coderunner-inkling-2-mini", "escalator": "qwen3.8-27b",
                                 "driver": {"model": "fara1.5-9b", "exclusive": False}},
                      "files": {"yield_during_llm_turn": True}}, path=self.home / "c.json")
        mp = pool.ModelPool(cfg, hw=_apple(64, "Apple M4 Max", sandbox=4))
        rp = mp.role_plan()
        self.assertEqual((rp.planner, rp.escalator, rp.driver, rp.driver_exclusive),
                         ("inkling-2-mini", "qwen3.8-27b", "fara1.5-9b", False))
        self.assertTrue(mp.should_escalate(stuck=2))
        self.assertFalse(mp.should_escalate(stuck=0, failed=False))
        self.assertTrue(mp.should_yield_indexing(llm_busy=True))
        self.assertFalse(mp.should_yield_indexing(llm_busy=False))

    def test_preflight_gguf_header_and_tool_template(self):
        def handler(req: httpx.Request):
            if req.url.path.endswith(".gguf"):
                return httpx.Response(206, content=b"GGUF\x03\x00\x00\x00general.architecture\x00qwen35\x00")
            if req.url.path.endswith("tokenizer_config.json"):
                return httpx.Response(200, json={"chat_template": "{% if tools %}tool_call{% endif %}"})
            return httpx.Response(404)
        h = hub_module.Hub(httpx.Client(transport=httpx.MockTransport(handler)), cache_dir=False)
        res = h.preflight_gguf("org/Model-GGUF", "Model-Q4_K_M.gguf", base_repo="org/Model")
        self.assertTrue(res["ok"])
        self.assertTrue(res["magic_ok"])
        self.assertTrue(res["has_tool_template"])
        self.assertEqual(res["architecture"], "qwen35")

    def test_unified_catalog_lookup_reads_discovered_models(self):
        path = self.home / "assistant" / "models" / "discovered.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{
            "id": "hf-lab-new-30b", "repo": "lab/New-30B", "family": "qwen3.5", "roles": ["chat"],
            "params_b": 30.0, "active_params_b": 30.0, "ctx_max": 32768,
            "gguf_repo": "unsloth/New-30B-GGUF",
            "quants": {"Q4_K_M": {"file": "New-30B-Q4_K_M.gguf", "size_gb": 18.0}},
            "license": {"spdx": "apache-2.0", "base_repo": "lab/New-30B"},
        }]))
        found = catalog.lookup("hf-lab-new-30b")
        self.assertEqual(found["artifacts"]["gguf"]["quant"], "Q4_K_M")

    async def test_tier1_gate_checks_nested_batch_and_cache_ratio(self):
        def handler(req: httpx.Request):
            body = json.loads(req.content)
            if body.get("stream"):
                payload = (
                    'data: {"choices":[{"delta":{"content":"IndiGo is cheapest."}}],'
                    '"usage":{"prompt_tokens":3900,"completion_tokens":16},'
                    '"timings":{"prompt_per_second":240.0,"predicted_per_second":22.0}}\n\n'
                    'data: [DONE]\n\n'
                )
                return httpx.Response(200, content=payload.encode(), headers={"content-type": "text/event-stream"})
            tools = body.get("tools") or []
            names = [t["function"]["name"] for t in tools]
            if "browser_batch" in names:
                return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "browser_batch",
                                 "arguments": json.dumps({"steps": [{"action": "type", "ref": 12, "value": "Delhi"},
                                                                    {"action": "click", "ref": 15}]})},
                }]}}]})
            if "add_numbers" in names and "get_weather" in names:
                prompt = body["messages"][-1]["content"]
                if "ready" in prompt.lower():
                    return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})
                if "weather" in prompt.lower():
                    return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
                        "id": "c2", "type": "function",
                        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    }]}}]})
                return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
                    "id": "c3", "type": "function",
                    "function": {"name": "add_numbers", "arguments": json.dumps({"a": 3, "b": 7})},
                }]}}]})
            if len(body.get("messages") or []) >= 4:
                return httpx.Response(200, json={"choices": [{"message": {"content": "Air India"}}],
                                                 "timings": {"cache_n": 2400, "prompt_n": 100}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}}]})

        transport = httpx.MockTransport(handler)
        res = await tier1_gate("http://127.0.0.1:8081/v1", "test-model", vision=False, transport=transport)
        self.assertTrue(res["ok"], res)
        self.assertAlmostEqual(res["cache_hit_ratio"], 0.96, places=2)
        self.assertTrue(any(c["name"] == "nested_batch_tool" and c["passed"] for c in res["checks"]))


if __name__ == "__main__":
    unittest.main()
