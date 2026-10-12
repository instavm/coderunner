"""Measure this machine with a served model, and calibrate the planner's estimates with it.

The planner starts from spec-sheet numbers: memory bandwidth per chip for writing, and a rough compute
factor per accelerator for reading prompts. One timed request on the owner's machine replaces both:
a page-sized prompt (what an agent step reads) and a short answer (what it writes). The result is
saved per machine and applied to every candidate: writing speed scales with the bytes each token
reads, prompt reading with the active parameters.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import platform
import random
import time

import httpx

from .. import __version__
from ..core import paths
from . import catalog, fit
from .hardware import Hardware

PAGE_LINES = 100            # about 3,900 tokens of page-like text (Qwen3.5 tokenizer)
WRITE_TOKENS = 128
FACTOR_RANGE = (0.25, 2.0)  # a measurement outside this says more about the run than the machine
CITIES = ("Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Goa", "Jaipur", "Kochi", "Lucknow",
          "Dubai", "Singapore", "Bangkok", "London", "Paris", "Tokyo", "Sydney", "Doha", "Colombo", "Kathmandu")


def page(seed: int, lines: int = PAGE_LINES) -> str:
    """A flight-results page as the browser tool renders it; new each time, so no cache can help."""
    rnd = random.Random(seed)
    out = [f"# Flights, search {seed}"]
    for i in range(lines):
        a, b = rnd.sample(CITIES, 2)
        kind = rnd.choice(("link", "button", "option"))
        line = (f'[{i + 10}] {kind} "{a} to {b}, {rnd.randint(1, 28)} Nov, '
                f'{rnd.choice(("IndiGo", "Air India", "Vistara", "Akasa"))} {rnd.randint(5, 23)}:{rnd.randint(10, 59)}, '
                f'INR {rnd.randint(2100, 19900)}"')
        out.append(line + (f" -> /flights/{a.lower()}-{b.lower()}/{rnd.randint(1000, 9999)}" if kind == "link" else ""))
    return "\n".join(out)


async def _stream(client: httpx.AsyncClient, base_url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            # No reasoning: Ollama reads reasoning_effort, llama.cpp chat_template_kwargs.
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    start = time.monotonic()
    first = last = None
    chunks, usage, timings = 0, {}, {}
    async with client.stream("POST", f"{base_url}/chat/completions", json=body) as response:
        if response.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {response.status_code}: {(await response.aread()).decode()[:200]}")
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                    last = time.monotonic()
                    first = first or last
                    chunks += 1
    end = time.monotonic()
    return {"start": start, "first": first or end, "last": last or end, "chunks": chunks, "usage": usage,
            "timings": timings, "chars": sum(len(m["content"]) for m in messages)}


async def measure(base_url: str, model: str, api_key: str = "", timeout: float = 900, transport=None) -> dict:
    """Prompt reading and generation speed of a served model, in tokens per second."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False, transport=transport) as client:
        base_url = base_url.rstrip("/")
        await _stream(client, base_url, model, [{"role": "user", "content": "Say hi."}], 4)   # load the weights
        seed = int(time.time() * 1000)
        run = await _stream(client, base_url, model, [
            {"role": "system", "content": f"Session {seed}. You help the owner book travel."},
            {"role": "user", "content": page(seed) + "\n\nSummarize these flights in detail, cheapest first."}],
            WRITE_TOKENS)
    usage, timings = run["usage"], run["timings"]
    prompt = int(usage.get("prompt_tokens") or timings.get("prompt_n") or run["chars"] / 3.5)
    written = int(usage.get("completion_tokens") or timings.get("predicted_n") or run["chunks"])
    first = run["first"] - run["start"]
    result = {"prompt_tokens": prompt, "written_tokens": written, "first_token_s": round(first, 2),
              "read_tok_s": round(prompt / first, 1) if first > 0 else 0.0,
              "gen_tok_s": round((written - 1) / (run["last"] - run["first"]), 1)
              if written > 1 and run["last"] > run["first"] else 0.0}
    # llama.cpp's own timings leave out the network and the first token.
    if timings.get("prompt_per_second"):
        result["read_tok_s"] = round(float(timings["prompt_per_second"]), 1)
    if timings.get("predicted_per_second"):
        result["gen_tok_s"] = round(float(timings["predicted_per_second"]), 1)
    result["step_s"] = round(fit.step_seconds(result["read_tok_s"], result["gen_tok_s"]), 1)
    return result


def fingerprint(hw: Hardware) -> dict:
    return {"os": hw.os, "chip": hw.chip, "ram_gb": round(hw.ram_gb), "gpu": hw.gpu, "gpus": hw.gpus}


def calibration(hw: Hardware, entry: dict, runtime: str | None, measured: dict, ctx: int = 32768) -> dict:
    """Machine factors from one measured model: measured / spec-sheet generation, and reading per parameter."""
    spec = dataclasses.replace(hw, gen_factor=1.0, read_k=0.0, measured="")
    est = fit.estimate(entry, spec, ctx, runtime)
    art = catalog.artifact(entry, runtime) or {}
    active = float(entry.get("active_params_b") or entry.get("params_b") or 1)
    low, high = FACTOR_RANGE
    gen_factor = measured["gen_tok_s"] / est["tok_s"] if est["tok_s"] else 1.0
    read_k = measured["read_tok_s"] * active / art.get("read_factor", 1.0)
    spec_read_k = est["read_tok_s"] * active / art.get("read_factor", 1.0)
    return {"machine": fingerprint(hw), "date": datetime.date.today().isoformat(), "model": entry["id"],
            "runtime": runtime or "", "measured": measured,
            "estimated": {"tok_s": est["tok_s"], "read_tok_s": est["read_tok_s"], "step_s": est["step_s"]},
            "gen_factor": round(min(max(gen_factor, low), high), 3),
            "read_k": round(min(max(read_k, low * spec_read_k), high * spec_read_k), 1)}


def _path():
    return paths.assistant_dir() / "models" / "bench.json"


def save(cal: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1))


def load(hw: Hardware) -> dict | None:
    """The saved calibration, if it was measured on this machine."""
    try:
        cal = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    return cal if cal.get("machine") == fingerprint(hw) else None


def apply(hw: Hardware, cal: dict | None) -> Hardware:
    if not cal:
        return hw
    return dataclasses.replace(hw, gen_factor=float(cal["gen_factor"]), read_k=float(cal["read_k"]),
                               measured=f"{cal['model']} on {cal['runtime'] or 'the configured runtime'}, {cal['date']}")


def report(hw: Hardware, cal: dict | None, picks: list[dict], runtime: dict | None = None) -> str:
    """A summary the owner can paste into an issue or a chat: no names, paths or tokens."""
    lines = ["## CodeRunner hardware report", "",
             f"- CodeRunner {__version__}, {platform.system()} {platform.release()} ({hw.arch})",
             f"- {hw.chip or 'unknown chip'}, {hw.ram_gb:.0f} GB RAM, GPU: {hw.gpu}"
             + (f" x{hw.gpus}" if hw.gpus > 1 else "") + (f", {hw.vram_gb:.0f} GB VRAM" if not hw.unified and hw.vram_gb else "")
             + (" (unified memory)" if hw.unified else ""),
             f"- Spec bandwidth {hw.bandwidth_gbs:.0f} GB/s; model budget {hw.budget_gb:.1f} GB"
             + (f"; {hw.nodes} machines pooled" if hw.nodes > 1 else "")]
    if runtime:
        lines.append(f"- Runtime: {runtime.get('name')} {runtime.get('version') or ''}".rstrip())
    if cal:
        m, e = cal["measured"], cal["estimated"]
        lines += ["", f"### Measured ({cal['date']}, {cal['model']} on {cal['runtime']})", "",
                  "| | measured | spec estimate |", "| --- | --- | --- |",
                  f"| prompt reading | {m['read_tok_s']:.0f} tok/s | {e['read_tok_s']:.0f} tok/s |",
                  f"| generation | {m['gen_tok_s']:.1f} tok/s | {e['tok_s']:.1f} tok/s |",
                  f"| agent step ({fit.STEP_READ_TOKENS} read, {fit.STEP_WRITE_TOKENS} written) | {m['step_s']:.1f} s | {e['step_s']:.1f} s |",
                  f"| first token after {m['prompt_tokens']} prompt tokens | {m['first_token_s']:.1f} s | |",
                  "", f"Calibration: generation x{cal['gen_factor']:.2f}, reading {cal['read_k']:.0f} tok/s per billion active parameters."]
    else:
        lines += ["", "Not measured yet: run `coderunner models bench` after `coderunner setup`."]
    if picks:
        lines += ["", "### Plan", "", "| model | runtime | fit | context | tok/s | read tok/s | s per step | ability |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in picks:
            ability = "" if c.get("ability") is None else f"{c['ability']:+.2f}"
            lines.append(f"| {c['id']} | {c.get('runtime') or ''} | {c['fit']} | {c['ctx'] // 1024}K | {c['tok_s']:.0f} | "
                         f"{c['read_tok_s']:.0f} | {c['step_s']:.0f} | {ability} |")
    return "\n".join(lines) + "\n"
