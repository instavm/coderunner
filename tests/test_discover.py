import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from coderunner import paths
from coderunner.config import Config
from coderunner.models import catalog, discover, manager
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

    def test_clearly_bigger_models_rank_ahead_of_curated_ones(self):
        studio = apple(512, "Apple M3 Ultra", sandbox=1)
        big = discover.best_build(self.models["lab/Big-300B-A30B"], studio, self.config, "llama.cpp")
        agent = discover.best_build(self.models["lab/Agent-27B"], studio, self.config, "llama.cpp")
        best_curated = min(e["rank"] for e in catalog.models() if "chat" in e["roles"])
        placed = {e["id"]: e["rank"] for e in discover.place([big, agent], catalog.models())}
        self.assertLess(placed["hf-lab-big-300b-a30b"], best_curated)
        self.assertGreater(placed["hf-lab-agent-27b"], catalog.get("qwen3.8-27b")["rank"])    # curated wins ties

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
        cfg = Config({"models": {"discover": True}}, path=pathlib.Path(self.tmp.name) / "c3.json")
        studio = apple(512, "Apple M3 Ultra", sandbox=1)
        with mock.patch.object(discover, "cached", return_value=list(self.models.values())), \
                mock.patch.object(manager, "choose_runtime", return_value=FakeRuntime()), \
                mock.patch.object(manager, "_prism", return_value=None):
            with mock.patch.object(manager, "_llama", return_value=FakeLlama()):
                result = manager.plan(cfg, hw=studio)
            found = {c["id"]: c for c in [result["model"], *result["alternatives"]]}
            self.assertEqual(found["hf-lab-big-300b-a30b"]["runtime"], "llama.cpp")
            self.assertEqual(found["hf-lab-agent-27b"]["runtime"], "ollama")
            with mock.patch.object(manager, "_llama", return_value=None):
                result = manager.plan(cfg, hw=studio)
            rejected = {c["id"]: c for c in result["rejected"]}
            self.assertIn("needs llama.cpp", rejected["hf-lab-big-300b-a30b"]["reasons"][0])
            self.assertIn("hf-lab-big-300b-a30b would rank higher but needs llama.cpp", " ".join(result["notes"]))
            done = manager.setup(cfg, model_id="hf-lab-big-300b-a30b", hw=studio, progress=lambda m: None,
                                 embedding=False)
            self.assertFalse(done["ok"])
            self.assertIn("brew install llama.cpp", done["error"])

    def test_discovered_models_rank_among_catalog_models_and_reach_setup(self):
        placed = discover.place([discover.best_build(self.models["lab/Agent-27B"], nvidia(24), self.config,
                                                     "llama.cpp")], catalog.models())
        self.assertTrue(placed[0]["id"].startswith("hf-lab-agent-27b"))
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
