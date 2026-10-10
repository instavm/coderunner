"""Local model management: detect the machine, pick a model that fits, serve it
on 127.0.0.1 behind an OpenAI-compatible API and verify tool calling."""

from __future__ import annotations

from .catalog import get as get_model, models as catalog_models
from .fit import estimate, recommend
from .hardware import Hardware, detect
from .licenses import check as check_license, classify as classify_license
from .manager import Endpoint, endpoint, ensure_running, plan, setup
from .runtimes import LlamaCppRuntime, OllamaRuntime, OpenAIRuntime, Runtime, choose_runtime, install_hint

__all__ = [
    "Endpoint", "Hardware", "LlamaCppRuntime", "OllamaRuntime", "OpenAIRuntime", "Runtime",
    "catalog_models", "check_license", "choose_runtime", "classify_license", "detect", "endpoint",
    "ensure_running", "estimate", "get_model", "install_hint", "plan", "recommend", "setup",
]
