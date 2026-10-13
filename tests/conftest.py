"""Tests never download models: Hugging Face is pointed at a closed local port unless live calls are asked for.

A test that reached the real hub by mistake downloaded 11 GB of Qwen3.5-9B into temporary folders before its
run was stopped.
"""

import os

if os.environ.get("CODERUNNER_LIVE_MODELS") != "1":
    os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"
