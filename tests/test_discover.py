import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from coderunner.core import paths
from coderunner.core.config import Config
from coderunner.models import catalog, discover, evidence, manager
from test_models import FakeLlama, FakeRuntime, apple, linux, nvidia

GiB = 1e9


class FakeHub:
    """Serves listings, configs and file trees the way the Hugging Face API does."""

    def __init__(self, data):
        self.data = data

    def _get_json(self, url, key, headers=None):
        for fragment, value in self.data.items():
            if fragment in url:
                return value
        raise discover.HubError(f"{url} returned 404")

    def model_info(self, repo, revision=None):
        return self._get_json(f"/api/models/{repo}?blobs=true", "")

    def resolve_license(self, repo, depth=3):
        return {"spdx": "apache-2.0", "link": f"https://huggingface.co/{repo}"}


def tree(*files):
    return [{"path": name, "size": int(size * GiB)} for name, size in files]


DENSE = {"model_type": "qwen3_5", "num_hidden_layers": 64, "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                                                  "full_attention"] * 16,
         "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "max_position_embeddings": 262144}
MOE = {"model_type": "deepseek_v4", "text_config": {"num_hidden_layers": 43, "num_attention_heads": 64, "kv_lora_rank": 512,
                       "qk_rope_head_dim": 64, "n_routed_experts": 256, "num_experts_per_tok": 8,
                       "n_shared_experts": 1, "max_position_embeddings": 1048576}}


# Published scores as the Hub's evalResults give them, for the fake lab's models.
STRONG = {"Idavidrein/gpqa:diamond": 95.0, "harborframework/terminal-bench-2.1:terminalbench_2_1": 92.0,
          "hkust-nlp/Toolathlon:toolathlon_verified": 80.0}
LIKE_QWEN38_27B = {"Idavidrein/gpqa:diamond": 89.0, "harborframework/terminal-bench-2.1:terminalbench_2_1": 73.0,
                   "ScaleAI/SWE-bench_Pro:SWE_Bench_Pro": 61.0}
LIKE_GRANITE_30B = {"TIGER-Lab/MMLU-Pro:mmlu_pro": 77.6, "harborframework/terminal-bench-2.1:terminalbench_2_1": 29.2,
                    "ScaleAI/SWE-bench_Pro:SWE_Bench_Pro": 33.3, "SWE-bench/SWE-bench_Verified:swe_bench_%_resolved": 57.0}


def hub():
    return FakeHub({
        "models?author=lab": [
            {"id": "lab/Agent-27B", "createdAt": "2026-09-20T00:00:00Z", "pipeline_tag": "image-text-to-text"},
            {"id": "lab/Big-300B-A30B", "createdAt": "2026-09-01T00:00:00Z", "pipeline_tag": "text-generation"},
            {"id": "lab/Agent-27B-Base", "createdAt": "2026-09-20T00:00:00Z", "pipeline_tag": "text-generation"},
            {"id": "lab/Old-7B", "createdAt": "2024-01-01T00:00:00Z", "pipeline_tag": "text-generation"}],
        "/api/models/lab/Agent-27B?": {"safetensors": {"total": 27_000_000_000}},
        "/api/models/lab/Big-300B-A30B?": {"safetensors": {"total": 300_000_000_000}},
        "lab/Agent-27B/resolve/main/config.json": DENSE,
        "lab/Big-300B-A30B/resolve/main/config.json": MOE,
        "models/unsloth/Agent-27B-GGUF/tree": tree(
            ("Agent-27B-Q8_0.gguf", 28.6), ("Agent-27B-UD-Q4_K_XL.gguf", 17.0), ("Agent-27B-UD-Q2_K_XL.gguf", 9.6),
            ("mmproj-F16.gguf", 0.9), ("Agent-27B-MTP-Q8_0.gguf", 1.1)),
        "models/unsloth/Big-300B-A30B-GGUF/tree": tree(
            ("UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00003.gguf", 50), ("UD-Q4_K_XL/Big-UD-Q4_K_XL-00002-of-00003.gguf", 50),
            ("UD-Q4_K_XL/Big-UD-Q4_K_XL-00003-of-00003.gguf", 55), ("UD-Q2_K_XL/Big-UD-Q2_K_XL-00001-of-00002.gguf", 50),
            ("UD-Q2_K_XL/Big-UD-Q2_K_XL-00002-of-00002.gguf", 47), ("Big-Q8_0.gguf", 10.9)),
        "models/mlx-community/Agent-27B-4bit/tree": [{"path": "model.safetensors", "size": int(16 * GiB)}],
    })


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": self.tmp.name})
        self.env.start()
        self.config = Config({}, path=pathlib.Path(self.tmp.name) / "config.json")
        with mock.patch.object(discover.time, "time", return_value=1791700000):     # 2026-10-11
            self.models = {m["repo"]: m for m in discover.Discovery(hub()).run(("lab",))}

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_reads_recent_chat_models_and_their_real_builds(self):
        self.assertEqual(set(self.models), {"lab/Agent-27B", "lab/Big-300B-A30B"})   # no base or old models
        agent = self.models["lab/Agent-27B"]
        self.assertEqual(agent["arch"], {"full_attn_layers": 16, "kv_heads": 4, "head_dim": 256})
        self.assertEqual(set(agent["quants"]), {"Q8_0", "UD-Q4_K_XL", "UD-Q2_K_XL"})   # not the MTP head
        self.assertAlmostEqual(agent["mmproj"]["size_gb"], 0.9, places=2)
        self.assertEqual(agent["mlx"]["repo"], "mlx-community/Agent-27B-4bit")
        self.assertIn("vision", agent["roles"])
        big = self.models["lab/Big-300B-A30B"]
        self.assertAlmostEqual(big["quants"]["UD-Q4_K_XL"]["size_gb"], 155, places=0)      # shards summed
        self.assertNotIn("Q8_0", big["quants"])                                            # 10.9 GB is not 300B
        self.assertEqual(big["arch"]["kv_heads"], 1)                                       # compressed attention
        self.assertLess(big["active_params_b"], 0.2 * big["params_b"])                     # MoE

    def test_picks_the_best_build_that_fits_each_machine(self):
        agent = self.models["lab/Agent-27B"]
        on24 = discover.best_build(agent, nvidia(24), self.config, "llama.cpp")
        self.assertEqual(on24["artifacts"]["gguf"]["quant"], "UD-Q4_K_XL")
        on48 = discover.best_build(agent, nvidia(48), self.config, "llama.cpp")
        self.assertEqual(on48["artifacts"]["gguf"]["quant"], "Q8_0")
        self.assertEqual(on48["artifacts"]["ollama"]["tag"], "hf.co/unsloth/Agent-27B-GGUF:Q8_0")
        mac = discover.best_build(agent, apple(36, "Apple M3 Max", sandbox=1), self.config, "mlx")
        self.assertEqual(mac["artifacts"]["mlx"]["repo"], "mlx-community/Agent-27B-4bit")
        self.assertIsNone(discover.best_build(agent, apple(8, "Apple M2", sandbox=1), self.config, "llama.cpp"))
        big = self.models["lab/Big-300B-A30B"]
        studio = discover.best_build(big, apple(512, "Apple M3 Ultra", sandbox=1), self.config, "llama.cpp")
        self.assertEqual(studio["artifacts"]["gguf"]["quant"], "UD-Q4_K_XL")
        spark = discover.best_build(big, nvidia(24), self.config, "llama.cpp")
        self.assertIsNone(spark)

    def test_precision_is_not_bought_with_speed(self):
        agent = self.models["lab/Agent-27B"]
        # Q8_0 fits a 64 GB M4 Max but runs at ~10 tok/s; the 4-bit build stays above 15.
        mac = discover.best_build(agent, apple(64, "Apple M4 Max", sandbox=1), self.config, "llama.cpp")
        self.assertEqual(mac["artifacts"]["gguf"]["quant"], "UD-Q4_K_XL")

    def ranked(self, hw):
        rt = FakeRuntime()
        with mock.patch.object(discover, "cached", return_value=list(self.models.values())):
            extra = manager._discovered(Config({}, path=pathlib.Path(self.tmp.name) / "r.json"), hw, rt)
        return [c["id"] for c in manager._ranked(hw, self.config, rt, None, extra) if c["ok"]]

    def test_published_scores_decide_where_discovered_models_rank(self):
        hw = nvidia(48)
        # No published scores: after every model that has some, however big.
        order = self.ranked(hw)
        self.assertGreater(order.index("hf-lab-agent-27b"), order.index("qwen3.5-4b"))
        # Scores like Granite 4.2 30B's: below Qwen3.5-9B despite three times the parameters.
        evidence.save({"lab/Agent-27B": LIKE_GRANITE_30B})
        order = self.ranked(hw)
        self.assertGreater(order.index("hf-lab-agent-27b"), order.index("qwen3.5-9b"))
        # Scores above every catalog model: first.
        evidence.save({"lab/Agent-27B": STRONG})
        self.assertEqual(self.ranked(hw)[0], "hf-lab-agent-27b")

    def test_scores_are_read_from_hub_eval_results(self):
        records = [{"data": {"dataset": {"id": "Idavidrein/gpqa", "task_id": "diamond"}, "value": 84.0}},
                   {"data": {"dataset": {"id": "Idavidrein/gpqa", "task_id": "diamond"}, "value": 86.0}},
                   {"data": {"dataset": {"id": "cais/hle", "task_id": "hle"}, "value": 55.3}},     # settings vary
                   {"data": {"dataset": {"id": "TIGER-Lab/MMLU-Pro", "task_id": "mmlu_pro"}, "value": "n/a"}}]
        self.assertEqual(evidence.scores_from(records), {"Idavidrein/gpqa:diamond": 85.0})
        hub = FakeHub({"/api/models/lab/Agent-27B?expand": {"evalResults": records}})
        self.assertEqual(evidence.fetch(hub, ["lab/Agent-27B", "lab/Missing"]),
                         {"lab/Agent-27B": {"Idavidrein/gpqa:diamond": 85.0}})

    def test_quantized_builds_lose_ability(self):
        known = evidence.abilities()
        entry = {"license": {"base_repo": "Qwen/Qwen3.8-27B"}}
        self.assertEqual(evidence.ability(entry, 4.8), known["Qwen/Qwen3.8-27B"])
        self.assertAlmostEqual(evidence.ability(entry, 2.1), known["Qwen/Qwen3.8-27B"] - 0.3)
        # Scores published for the quantized build itself are used as they are.
        onebit = catalog.get("bonsai-27b-1bit")
        self.assertEqual(evidence.ability(onebit, 1.1), known["prism-ml/Bonsai-27B-gguf"])
        self.assertLess(known["prism-ml/Bonsai-27B-gguf"], known["Qwen/Qwen3.6-27B"] - 0.5)     # what 1 bit costs
        self.assertIsNone(evidence.ability({"repo": "lab/Unknown"}, 4.8))

    def test_cached_models_are_filtered_again(self):
        path = paths.assistant_dir() / "models" / "discovered.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        stale = [*self.models.values(), {**self.models["lab/Agent-27B"], "repo": "lab/Fara2-27B"},
                 {**self.models["lab/Agent-27B"], "repo": "lab/Tiny-1B", "params_b": 1.0}]
        path.write_text(json.dumps(stale))
        self.assertEqual({m["repo"] for m in discover.cached()}, set(self.models))

    def test_models_ollama_cannot_serve_and_known_breakages(self):
        big, studio = self.models["lab/Big-300B-A30B"], apple(512, "Apple M3 Ultra", sandbox=1)
        self.assertIsNone(discover.best_build(big, studio, self.config, "ollama"))      # no parser for its tools
        self.assertIsNotNone(discover.best_build(big, studio, self.config, "llama.cpp"))
        spark = linux(128, {"nvidia-smi": "NVIDIA GB10, [N/A], [N/A], 12.1\n"})
        self.assertIn("28132", discover.broken(big, spark))
        self.assertIsNone(discover.best_build(big, spark, self.config, "llama.cpp"))
        self.assertIsNone(discover.broken(big, studio))
        self.assertIsNone(discover.broken({**big, "repo": "lab/Qwen3.8-Flash-Next"}, studio))
        thor = linux(128, {"nvidia-smi": "NVIDIA Thor, [N/A], [N/A], 11.0\n"})
        self.assertIn("27763", discover.broken({**big, "repo": "lab/Qwen3.8-Flash-Next"}, thor))

    def test_discovered_models_route_to_llama_cpp(self):
        evidence.save({"lab/Big-300B-A30B": STRONG, "lab/Agent-27B": LIKE_QWEN38_27B})
        cfg = Config({"models": {"discover": True}}, path=pathlib.Path(self.tmp.name) / "c3.json")
        studio = apple(512, "Apple M3 Ultra", sandbox=1)
        with mock.patch.object(discover, "cached", return_value=list(self.models.values())), \
                mock.patch.object(manager, "choose_runtime", return_value=FakeRuntime()), \
                mock.patch.object(manager, "_prism", return_value=None):
            with mock.patch.object(manager, "_llama", return_value=FakeLlama()):
                result = manager.plan(cfg, hw=studio)
            found = {c["id"]: c for c in [result["model"], *result["alternatives"]]}
            self.assertEqual(found["hf-lab-big-300b-a30b"]["runtime"], "llama.cpp")
            # On Apple silicon chat models run on CodeRunner's llama.cpp server; on NVIDIA, Ollama's CUDA build.
            self.assertEqual(found["hf-lab-agent-27b"]["runtime"], "llama.cpp")
            self.assertTrue(manager._chat_on_llama(studio))
            self.assertFalse(manager._chat_on_llama(nvidia(32, name="NVIDIA GeForce RTX 5090")))
            with mock.patch.object(manager, "_llama", return_value=None):
                result = manager.plan(cfg, hw=studio)
            rejected = {c["id"]: c for c in result["rejected"]}
            self.assertIn("needs llama.cpp", rejected["hf-lab-big-300b-a30b"]["reasons"][0])
            self.assertIn("hf-lab-big-300b-a30b would rank higher but needs llama.cpp", " ".join(result["notes"]))
            with mock.patch.object(manager, "_llama", return_value=None):      # no llama.cpp to be had
                done = manager.setup(cfg, model_id="hf-lab-big-300b-a30b", hw=studio, progress=lambda m: None,
                                     embedding=False)
            self.assertFalse(done["ok"])
            self.assertIn("brew install llama.cpp", done["error"])

    def test_big_macs_hear_when_a_higher_gpu_limit_fits_a_better_model(self):
        evidence.save({"lab/Big-300B-A30B": STRONG})
        cfg = Config({"models": {"discover": True}}, path=pathlib.Path(self.tmp.name) / "c4.json")
        with mock.patch.object(discover, "cached", return_value=list(self.models.values())), \
                mock.patch.object(manager, "choose_runtime", return_value=FakeRuntime()), \
                mock.patch.object(manager, "_prism", return_value=None), \
                mock.patch.object(manager, "_llama", return_value=FakeLlama()):
            result = manager.plan(cfg, hw=apple(128, "Apple M4 Max", sandbox=1))
        self.assertNotEqual(result["model"]["id"], "hf-lab-big-300b-a30b")       # 97 GB does not fit 96
        notes = " ".join(result["notes"])
        self.assertIn("Raising the GPU memory limit allows hf-lab-big-300b-a30b", notes)
        self.assertIn("sudo sysctl iogpu.wired_limit_mb=122880", notes)

    def test_discovered_models_rank_among_catalog_models_and_reach_setup(self):
        placed = discover.place([discover.best_build(self.models["lab/Agent-27B"], nvidia(24), self.config,
                                                     "llama.cpp")], catalog.models())
        self.assertTrue(placed[0]["id"].startswith("hf-lab-agent-27b"))
        evidence.save({"lab/Agent-27B": LIKE_QWEN38_27B})
        cfg = Config({"models": {"discover": True}}, path=pathlib.Path(self.tmp.name) / "c2.json")
        with mock.patch.object(discover, "cached", return_value=list(self.models.values())), \
                mock.patch.object(manager, "choose_runtime", return_value=FakeRuntime()), \
                mock.patch.object(manager, "_prism", return_value=None):
            result = manager.plan(cfg, hw=nvidia(24))
            ids = [result["model"]["id"]] + [c["id"] for c in result["alternatives"]]
            self.assertIn("hf-lab-agent-27b", ids)
            rt = FakeRuntime()
            with mock.patch.object(manager, "choose_runtime", return_value=rt):
                done = manager.setup(cfg, model_id="hf-lab-agent-27b", hw=nvidia(24), progress=lambda m: None,
                                     embedding=False, smoke=_passing)
            self.assertTrue(done["ok"], done)
            self.assertEqual(rt.pulled, ["hf-lab-agent-27b"])


async def _passing(base_url, model, vision=False, api_key=""):
    return {"passed": True, "checks": [{"name": "tool_arguments", "passed": True, "detail": "", "seconds": 0.1}],
            "tok_s": 20}


if __name__ == "__main__":
    unittest.main()
