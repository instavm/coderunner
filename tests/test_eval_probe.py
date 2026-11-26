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
            "ane_plan_cost_pct": 92.0, "median_ms": 45.0, "missed_commits": 0, "false_alarm_rate": 0.0,
        })
        self.assertTrue(good["qualified"], good)
        unmeasured = qualify_ane_judge({
            "max_tokens": 1024, "compile_cold_s": 30.0, "warm_load_s": 0.8, "peak_rss_mb": 1200,
            "ane_plan_cost_pct": 92.0, "median_ms": 45.0, "missed_commits": None, "false_alarm_rate": 0.0,
        })
        self.assertFalse(unmeasured["qualified"])
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
            if len(body.get("messages") or []) >= 4 and not body.get("max_tokens") == 8:
                return httpx.Response(200, json={"choices": [{"message": {"content": "Air India"}}],
                                                 "timings": {"cache_n": 2400, "prompt_n": 100}})
            if body.get("max_tokens") == 8 and "Evaluate the supplied decision task" in str(body["messages"][0].get("content")):
                state_text = str(body["messages"][-1].get("content") or "").lower()
                is_commit = any(w in state_text for w in ("pay", "send email", "transfer", "place order", "delete account", "post reply"))
                return httpx.Response(200, json={"choices": [{"message": {"content": "B" if is_commit else "A"}}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}}]})

        transport = httpx.MockTransport(handler)
        res = await tier1_gate("http://127.0.0.1:8081/v1", "test-model", vision=False, transport=transport)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["safety_measured"])
        self.assertEqual(res["missed_commits"], 0)
        self.assertEqual(res["false_alarm_rate"], 0.0)
        self.assertAlmostEqual(res["cache_hit_ratio"], 0.96, places=2)
        self.assertTrue(any(c["name"] == "nested_batch_tool" and c["passed"] for c in res["checks"]))
        self.assertTrue(any(c["name"] == "safety_commit_judge" and c["passed"] for c in res["checks"]))

        # Unmeasured safety must never qualify as safe or win comparisons.
        from coderunner.eval.telemetry import score_run
        unmeasured = score_run({"passed": 5, "total": 5, "missed_commits": None, "safety_measured": False})
        self.assertFalse(unmeasured["safe"])
        self.assertFalse(unmeasured["safety_measured"])
        self.assertIsNone(plain_summary([{"stack": {"id": "x"}, "passed": 5, "total": 5, "missed_commits": None}])["winner"])

    def test_pyproject_ships_eval_and_probe_swift_and_preflight_caps_200_stream(self):
        pyproject = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        self.assertIn('"coderunner.eval"', pyproject)
        self.assertIn('"probe.swift"', pyproject)
        # Missing probe.swift falls back cleanly without raising FileNotFoundError.
        with mock.patch.object(probe, "SOURCE", self.home / "missing.swift"), \
                mock.patch("sys.platform", "darwin"):
            res = probe.run(hw=_apple(18, "Apple M3 Pro", sandbox=2), force=True)
            self.assertFalse(res["ok"])
            self.assertIn("probe.swift", res["error"])

        # Streaming preflight stops after 64 KB even when a server ignores Range and streams 200 OK.
        read_chunks = 0

        class CountingStream(httpx.SyncByteStream):
            def __iter__(self):
                nonlocal read_chunks
                yield b"GGUF\x03\x00\x00\x00general.architecture\x00qwen35\x00" + (b"\x00" * 16340)
                read_chunks += 1
                for _ in range(40):
                    read_chunks += 1
                    yield b"\x00" * 16384

        def stream_handler(req: httpx.Request):
            if req.url.path.endswith(".gguf"):
                return httpx.Response(200, stream=CountingStream())
            return httpx.Response(404)

        h = hub_module.Hub(httpx.Client(transport=httpx.MockTransport(stream_handler)), cache_dir=False)
        res = h.preflight_gguf("org/Big-GGUF", "Big-Q4_K_M.gguf")
        self.assertTrue(res["magic_ok"])
        self.assertLessEqual(read_chunks, 6)

    async def test_run_stack_validation_isolation_and_runtime_restore(self):
        from coderunner.eval import runner
        from coderunner.models.runtimes import LlamaCppRuntime

        cfg_path = self.home / "owner-config.json"
        cfg = Config({
            "models": {"runtime": "llama.cpp", "chat": "coderunner-qwen3.5-9b", "context": 32768,
                       "base_url": "http://127.0.0.1:8081/v1", "discover": False},
            "sandbox": {"memory_gb": 2},
        }, path=cfg_path)
        cfg.save()
        original_text = cfg_path.read_text()

        # Unsupported escalator stack is rejected explicitly.
        hw64 = _apple(64, "Apple M4 Max", sandbox=4)
        with mock.patch("coderunner.models.manager._hardware", return_value=hw64):
            rej = await runner.run_stack(cfg, StackSpec(id="dual", label="Dual", planner="qwen3.5-9b",
                                                        escalator="qwen3.8-27b", embedding="none"), suite="tier1")
            self.assertFalse(rej["ok"])
            self.assertIn("escalator", rej["error"].lower())

            # Different context triggers setup on a temporary config and restores snapshotted runtime state.
            rt = LlamaCppRuntime(cfg, hw=hw64)
            rt._save({"active": {"id": "qwen3.5-9b", "ctx": 32768}})
            before_state = rt.state_path.read_bytes()

            def fake_setup(work_cfg, model_id=None, runtime=None, hw=None, progress=None,
                           embedding=False, decision=False, measure=False, ctx=None, **kw):
                self.assertNotEqual(work_cfg.path, cfg_path)
                self.assertEqual(ctx, 65536)
                work_cfg.set("models.context", ctx)
                work_cfg.save()
                LlamaCppRuntime(work_cfg, hw=hw)._save({"active": {"id": model_id, "ctx": ctx}})
                return {"ok": True, "ctx": ctx}

            with mock.patch("coderunner.models.manager.setup", side_effect=fake_setup) as setup_mock, \
                    mock.patch("coderunner.models.manager.ensure_running",
                               return_value=mock.Mock(base_url="http://127.0.0.1:8081/v1", model="coderunner-qwen3.5-9b",
                                                      vision=False, api_key="")) as ensure_mock, \
                    mock.patch.object(LlamaCppRuntime, "stop"), \
                    mock.patch("coderunner.eval.runner.tier1_gate",
                               return_value={"ok": True, "passed": 5, "total": 5, "checks": [],
                                             "step_s": 5.0, "safety_measured": True, "missed_commits": 0,
                                             "false_alarm_rate": 0.0}):
                res = await runner.run_stack(
                    cfg, StackSpec(id="q64k", label="Qwen 64K", planner="qwen3.5-9b",
                                   ctx=65536, decision="self", embedding="none"),
                    suite="tier1",
                )
            self.assertTrue(res["ok"], res)
            setup_mock.assert_called_once()
            self.assertEqual(cfg_path.read_text(), original_text)
            self.assertEqual(rt.state_path.read_bytes(), before_state)
            self.assertGreaterEqual(ensure_mock.call_count, 2)

    async def test_judge_runtime_recovery_managed_mlx_slots_and_indexer_yield(self):
        from types import SimpleNamespace
        from coderunner.files.index import FileIndex
        from coderunner.models import manager, runtimes
        from coderunner.safety.decision import DecisionModel
        from coderunner.server import tasks as server_tasks

        # 1. Shared hardware parallel_tasks matches LlamaCppRuntime -np and server tasks.limit(svc).
        cfg8 = Config({"sandbox": {"memory_gb": 1}}, path=self.home / "c8.json")
        hw8 = _apple(8, "Apple M2", sandbox=1)
        svc8 = SimpleNamespace(config=cfg8, hardware=hw8, remote=None, llm=None)
        rt8 = runtimes.LlamaCppRuntime(cfg8, hw=hw8)
        cmd8 = rt8.command({"model": "/m.gguf", "ctx": 16384, "alias": "a"})
        self.assertEqual(server_tasks.limit(svc8), 1)
        self.assertEqual(cmd8[cmd8.index("-np") + 1], str(server_tasks.limit(svc8) + 1))

        cfg64 = Config({"sandbox": {"memory_gb": 4}}, path=self.home / "c64.json")
        hw64 = _apple(64, "Apple M4 Max", sandbox=4)
        svc64 = SimpleNamespace(config=cfg64, hardware=hw64, remote=None, llm=None)
        rt64 = runtimes.LlamaCppRuntime(cfg64, hw=hw64)
        cmd64 = rt64.command({"model": "/m.gguf", "ctx": 65536, "alias": "a"})
        self.assertEqual(server_tasks.limit(svc64), 3)
        self.assertEqual(cmd64[cmd64.index("-np") + 1], str(server_tasks.limit(svc64) + 1))

        # 2. Managed MLX vs external owner-managed MLX.
        managed_cfg = Config({"models": {"runtime": "mlx"}}, path=self.home / "mlx-m.json")
        ext_cfg = Config({"models": {"runtime": "mlx", "base_url": "http://127.0.0.1:8080/v1"}},
                         path=self.home / "mlx-e.json")
        mlx_m = runtimes.MLXRuntime(managed_cfg)
        mlx_e = runtimes.MLXRuntime(ext_cfg)
        self.assertTrue(mlx_m.managed())
        self.assertTrue(manager._is_managed_local(mlx_m))
        self.assertFalse(mlx_e.managed())
        self.assertFalse(manager._is_managed_local(mlx_e))

        # 3. Dedicated JudgeRuntime starts in ensure_running() and DecisionModel recovers on ConnectError.
        jrt = runtimes.JudgeRuntime(cfg64)
        jrt._save({"active": {"id": "judge-4b", "alias": "coderunner-judge-4b", "model": "/m/j.gguf", "ctx": 4096}})
        jcfg = Config({
            "models": {"runtime": "llama.cpp", "chat": "coderunner-qwen3.5-9b",
                       "base_url": "http://127.0.0.1:8081/v1",
                       "decision": "coderunner-judge-4b", "decision_api": "chat",
                       "decision_url": jrt.base_url},
        }, path=self.home / "jcfg.json")
        with mock.patch.object(runtimes.LlamaCppRuntime, "running", return_value=True), \
                mock.patch.object(runtimes.JudgeRuntime, "running", return_value=False), \
                mock.patch.object(runtimes.JudgeRuntime, "start") as jstart:
            manager.ensure_running(jcfg)
            jstart.assert_called_once()

        calls = 0

        def flaky_judge(req: httpx.Request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("connection refused", request=req)
            return httpx.Response(200, json={"choices": [{"message": {"content": "B"}}]})

        dm = DecisionModel(jrt.base_url, "coderunner-judge-4b", api="chat",
                           transport=httpx.MockTransport(flaky_judge))
        recovered = []

        async def do_recover():
            recovered.append(True)
            return True

        dm.recover = do_recover
        p = await dm.probability("Pay now", "Commit?", {"safe": "No", "commit": "Yes"}, {"commit"})
        await dm.aclose()
        self.assertEqual((p, len(recovered), calls), (1.0, 1, 2))

        # 4. FileIndex honors ModelPool.should_yield_indexing during busy LLM turns.
        from coderunner.core.audit import AuditLog
        from coderunner.core.store import Store

        mp = pool.ModelPool(Config({"files": {"yield_during_llm_turn": True}}, path=self.home / "y.json"), hw=hw64)
        idx = FileIndex(cfg64, Store(self.home / "s.db"), AuditLog(self.home / "a.jsonl"),
                        should_yield=lambda: mp.should_yield_indexing(llm_busy=True))
        dummy = SimpleNamespace(path=self.home / "doc.txt", folder=False, remote=False, evicted=False,
                                kind="text", repo=False)
        done, failed = await idx._index([dummy])
        self.assertEqual((done, failed), (0, 0))
        self.assertEqual(len(idx._queue), 1)

    async def test_tier2_combo_tasks_runs_in_isolated_subprocess(self):
        from coderunner.eval import suites

        before_home = os.environ.get("CODERUNNER_HOME")
        cfg = Config({"models": {"chat": "", "base_url": ""}}, path=self.home / "t2.json")
        res = await suites.tier2_combo_tasks(cfg, names=["nonexistent"], progress=lambda _: None)
        self.assertFalse(res.get("ok"))
        self.assertEqual(os.environ.get("CODERUNNER_HOME"), before_home)


if __name__ == "__main__":
    unittest.main()
