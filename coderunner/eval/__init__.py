"""Unified StackSpec evaluation and benchmark harness for single models and multi-model combinations."""

from .report import compare_markdown, plain_summary
from .runner import compare_stacks, run_stack, run_tier0, run_tier1
from .spec import StackSpec, default_candidates, estimate_stack_memory, stack_from_config
from .suites import qualify_ane_judge
from .telemetry import load_history, save_run

__all__ = [
    "StackSpec",
    "compare_markdown",
    "compare_stacks",
    "default_candidates",
    "estimate_stack_memory",
    "load_history",
    "plain_summary",
    "qualify_ane_judge",
    "run_stack",
    "run_tier0",
    "run_tier1",
    "save_run",
    "stack_from_config",
]
