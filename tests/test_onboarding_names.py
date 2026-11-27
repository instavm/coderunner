"""What the first-run screens call models and how they offer them."""

import unittest

from coderunner.models.manager import pretty
from coderunner.server.setup import merge_presets


class ModelNames(unittest.TestCase):
    def test_discovered_models_lose_their_publisher(self):
        self.assertEqual(pretty("hf-inclusionai-zwz2-4b"), "Zwz2 4B")
        self.assertEqual(pretty("hf-qwen-qwen3.5-9b"), "Qwen3.5 9B")
        self.assertEqual(pretty("hf-unsloth-qwen3.6-27b-gguf"), "Qwen3.6 27B")
        self.assertEqual(pretty("hf-deepseek-ai-deepseek-v4-flash"), "Deepseek V4 Flash")

    def test_known_names(self):
        self.assertEqual(pretty("lfm3-omni-2b"), "LFM3-Omni 2B")
        self.assertEqual(pretty("bonsai-2.5-27b"), "Bonsai 2.5 27B")
        self.assertEqual(pretty("bonsai-2-9b"), "Bonsai 2 9B")


class Presets(unittest.TestCase):
    def test_one_card_per_model(self):
        presets = [{"key": "fast", "label": "Fast", "summary": "Quickest", "model": "hf-inclusionai-zwz2-4b"},
                   {"key": "balanced", "label": "Balanced", "summary": "Balance", "model": "bonsai-2.5-27b"},
                   {"key": "smartest", "label": "Smartest", "summary": "Accuracy", "model": "bonsai-2.5-27b"}]
        merged = merge_presets(presets)
        self.assertEqual([(p["label"], p["name"]) for p in merged],
                         [("Fast", "Zwz2 4B"), ("Best for this Mac", "Bonsai 2.5 27B")])
        self.assertEqual(merged[1]["key"], "balanced")

    def test_distinct_models_keep_their_cards(self):
        presets = [{"key": k, "label": k.title(), "summary": "", "model": m}
                   for k, m in (("fast", "bonsai-2-4b"), ("balanced", "bonsai-2-9b"), ("smartest", "bonsai-2.5-27b"))]
        self.assertEqual([p["label"] for p in merge_presets(presets)], ["Fast", "Balanced", "Smartest"])


if __name__ == "__main__":
    unittest.main()
