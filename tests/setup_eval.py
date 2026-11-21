"""A first run of `coderunner setup`, measured: what it fetches, how long each step takes and where it stops.

    python tests/setup_eval.py                        # this machine as it is, stopping at the first download
    python tests/setup_eval.py --bare                 # as on a machine with no Ollama, llama.cpp or Homebrew
    python tests/setup_eval.py --download runtimes    # fetch and start runtimes, stop at the first model
    python tests/setup_eval.py --download all         # the whole setup (needs the disk space it lists)
    python tests/setup_eval.py --served               # also time the model this home already serves
    python tests/setup_eval.py --min-free 1           # as with files.min_free_gb set to 1 (default 5)

Setup runs for real in a throwaway CODERUNNER_HOME, so the owner's assistant is not touched; servers the run
starts are stopped at the end. Before it runs, the plan's downloads are listed with their sizes against the free
disk, and a few seconds of the model's download give the connection's speed. `--served` adds the steps a run
without downloads cannot reach (waking the model, the smoke test, the speed measurement), timed against the
model the owner's home already serves; nothing is saved there.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from coderunner.core import paths  # noqa: E402
from coderunner.core.config import Config  # noqa: E402
from coderunner.models import bench, catalog, coreml, discover, hardware, manager, runtimes, smoketest  # noqa: E402
from coderunner.models import hub as hub_module  # noqa: E402
from coderunner.voice import qwen  # noqa: E402

HIDDEN = ("ollama", "llama-server", "brew")       # what a bare machine does not have
PORTS = {11434: "Ollama", 8081: "llama.cpp", 8082: "Prism's llama.cpp", 8084: "the Core ML embedding server"}
SPEED_SECONDS = 6
STEPS: list[dict] = []


class Stopped(BaseException):
    """Raised in place of a download the run was told not to make (not an Exception: setup must not catch it)."""


def timed(function, label):
    """`function`, with its time and outcome added to STEPS; label may be made from the call's arguments."""
    def note(args, start, error=""):
        STEPS.append({"step": label(args) if callable(label) else label,
                      "seconds": round(time.monotonic() - start, 1), **({"error": error} if error else {})})

    if asyncio.iscoroutinefunction(function):
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                result = await function(*args, **kwargs)
            except Exception as exc:
                note(args, start, str(exc)[:200])
                raise
            note(args, start)
            return result
    else:
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                note(args, start, str(exc)[:200])
                raise
            note(args, start)
            return result
    return wrapper


def stopper(what):
    def stop(*args, **kwargs):
        raise Stopped(what(args) if callable(what) else what)
    return stop


def _entry(args) -> str:
    return args[1]["id"] if len(args) > 1 and isinstance(args[1], dict) else "?"


@contextlib.contextmanager
def instrumented(download: str):
    """Setup's steps timed; downloads the run may not make stop it instead."""
    ollama, llama, prism, core = (runtimes.OllamaRuntime, runtimes.LlamaCppRuntime, runtimes.PrismRuntime,
                                  coreml.CoreMLRuntime)
    steps = [
        (hardware, "detect", "detect the hardware"),
        (discover, "cached", "discover recent models on Hugging Face"),
        (ollama, "install", f"download Ollama {ollama.RELEASE}"),
        (ollama, "start", "start Ollama"),
        (ollama, "pull", lambda a: f"ollama pull {_entry(a)}"),
        (llama, "install", f"download llama.cpp {qwen.LLAMA_RELEASE}"),
        (llama, "pull", lambda a: f"download {_entry(a)} (GGUF)"),
        (llama, "start", "load the model in llama.cpp"),
        (prism, "install", f"download Prism's llama.cpp {prism.RELEASE}"),
        (prism, "pull", lambda a: f"download {_entry(a)} (Prism)"),
        (prism, "start", "load the model in Prism's llama.cpp"),
        (core, "install", "install the Core ML packages"),
        (core, "pull", lambda a: f"download {_entry(a)} (Core ML)"),
        (coreml, "warm", "compile the embedding model for the Neural Engine"),
        (coreml, "compare", "time embeddings: Neural Engine against GPU"),
        (manager, "_setup_decision", "set up the decision model"),
    ]
    blocked = []
    if download in ("none", "runtimes"):
        blocked += [(ollama, "pull", lambda a: f"ollama pull {_entry(a)}"),
                    (hub_module.Hub, "download", lambda a: f"download {a[1]}/{a[2]}" if len(a) > 2 else "a download"),
                    (core, "pull", lambda a: f"download {_entry(a)} (Core ML)")]
    if download == "none":
        blocked += [(ollama, "install", f"download Ollama {ollama.RELEASE}"),
                    (llama, "install", f"download llama.cpp {qwen.LLAMA_RELEASE}"),
                    (prism, "install", f"download Prism's llama.cpp {prism.RELEASE}"),
                    (core, "install", "install the Core ML packages")]
    with contextlib.ExitStack() as stack:
        for owner, name, what in blocked:               # under the timers, so a stopped step is still listed
            stack.enter_context(mock.patch.object(owner, name, stopper(what)))
        for owner, name, label in steps:
            stack.enter_context(mock.patch.object(owner, name, timed(getattr(owner, name), label)))
        yield


@contextlib.contextmanager
def bare_machine():
    """No Ollama (command or app), no llama-server and no Homebrew, as on a Mac or PC nobody developed on."""
    which = shutil.which
    with mock.patch.object(shutil, "which", lambda cmd, *a, **k: None if os.path.basename(str(cmd)) in HIDDEN
                           else which(cmd, *a, **k)), \
            mock.patch.object(runtimes.OllamaRuntime, "APP_BINARY", "/nonexistent/ollama"):
        yield


@contextlib.contextmanager
def clean_home():
    before = os.environ.get("CODERUNNER_HOME")
    folder = tempfile.mkdtemp(prefix="cr-setup-")
    os.environ["CODERUNNER_HOME"] = folder
    try:
        paths.ensure_dirs()
        yield pathlib.Path(folder)
    finally:
        config = Config.load()
        for rt in (runtimes.LlamaCppRuntime(config), runtimes.PrismRuntime(config), coreml.CoreMLRuntime(config),
                   runtimes.OllamaRuntime(config)):
            with contextlib.suppress(Exception):
                rt.stop()                       # only what this run started: each stops by its own pid file
        if before is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = before
        shutil.rmtree(folder, ignore_errors=True)


def _size_gb(url: str) -> float | None:
    try:
        response = httpx.head(url, follow_redirects=True, timeout=15)
        return int(response.headers["content-length"]) / 1e9 if response.status_code == 200 else None
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def downloads(plan: dict, hw: hardware.Hardware, config) -> list[dict]:
    """Everything a first run of setup fetches for this plan, with sizes in GB (None where unknown)."""
    found: list[dict] = []
    ollama = runtimes.OllamaRuntime(config)
    if (ollama.binary() is None or ollama.outdated()) and ollama.can_install():
        asset = ollama.build()[0]
        found.append({"what": f"Ollama {ollama.RELEASE}", "gb": _size_gb(
            f"https://github.com/ollama/ollama/releases/download/v{ollama.RELEASE}/{asset}")})
    model = plan.get("model") or {}
    name = model.get("runtime") or ""
    if name == "prism" and runtimes.PrismRuntime(config, hw=hw).binary() is None:
        build = runtimes.PrismRuntime.build_for(hw)
        if build:
            release = runtimes.PrismRuntime.RELEASE
            found.append({"what": "Prism's llama.cpp build", "gb": _size_gb(
                f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{release}/llama-{release}-bin-{build[0]}.tar.gz")})
    if name == "llama.cpp" and runtimes.LlamaCppRuntime(config).binary() is None:
        build = qwen.QwenASR.build_for_host()
        if build:
            found.append({"what": "llama.cpp build", "gb": _size_gb(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{qwen.LLAMA_RELEASE}/"
                f"llama-{qwen.LLAMA_RELEASE}-bin-{build[0]}.tar.gz")})
    for role, key in (("assistant model", "model"), ("embedding model (Ollama)", "embedding"),
                      ("decision model", "decision")):
        cand = plan.get(key)
        if not cand:
            continue
        try:
            art = catalog.artifact(catalog.get(cand["id"]), cand.get("runtime")) or {}
        except KeyError:
            art = {}                    # a discovered model: its size is in the plan's reasons only
        size = art.get("size_gb")
        if size is not None and cand.get("vision"):
            size += art.get("mmproj_gb", 0.0)
        found.append({"what": f"{role}: {cand['id']}", "gb": size, "art": art})
    if coreml.supported():
        entry = next((e for e in catalog.by_role("embedding") if coreml.CoreMLRuntime.artifact(e)), None)
        if entry:
            found.append({"what": f"embedding model (Core ML): {entry['id']}",
                          "gb": coreml.CoreMLRuntime.artifact(entry).get("size_gb")})
            found.append({"what": "Core ML packages", "gb": 0.12})
    return found


def speed_mb_s(art: dict) -> float | None:
    """The connection's speed, from a few seconds of the assistant model's download in as many streams as setup
    uses (nothing is kept)."""
    from concurrent.futures import ThreadPoolExecutor
    if not art.get("repo") or not art.get("file"):
        return None
    url = f"{hub_module.endpoint()}/{art['repo']}/resolve/main/{art['file']}"
    streams, piece = hub_module.PARALLEL, 1 << 30

    def fetch(index: int) -> int:
        done, start = 0, time.monotonic()
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=30,
                              headers={"Range": f"bytes={index * piece}-{(index + 1) * piece - 1}"}) as response:
                if response.status_code not in (200, 206):
                    return 0
                for chunk in response.iter_bytes(1 << 18):
                    done += len(chunk)
                    if time.monotonic() - start > SPEED_SECONDS:
                        break
        except httpx.HTTPError:
            pass
        return done

    start = time.monotonic()
    with ThreadPoolExecutor(streams) as pool:
        done = sum(pool.map(fetch, range(streams)))
    return done / 1e6 / max(time.monotonic() - start, 0.1) if done else None


def served() -> dict:
    """The steps after the downloads, on the model the owner's home serves: waking it, the smoke test, its speed."""
    config = Config.load()
    ep = manager.endpoint(config)
    if not ep.model or not ep.base_url:
        return {"error": "this home has no model set up"}
    out: dict = {"model": ep.model}
    start = time.monotonic()
    smoke = asyncio.run(smoketest.run(ep.base_url, ep.model, vision=ep.vision, api_key=ep.api_key))
    out["smoke_s"] = round(time.monotonic() - start, 1)
    out["smoke"] = {c["name"]: {"passed": c["passed"], "seconds": c["seconds"]} for c in smoke["checks"]}
    start = time.monotonic()
    out["measured"] = asyncio.run(bench.measure(ep.base_url, ep.model, api_key=ep.api_key))
    out["measure_s"] = round(time.monotonic() - start, 1)
    return out


def run(args) -> dict:
    report: dict = {"mode": "bare" if args.bare else "as is", "download": args.download}
    if args.served:
        report["served"] = served()
    busy = [f"{name} on {port}" for port, name in PORTS.items()
            if runtimes._ok(f"http://127.0.0.1:{port}/health") or runtimes._ok(f"http://127.0.0.1:{port}/api/version")]
    if busy:
        report["warning"] = ("already answering here, so the run would use them instead of starting its own: "
                             + ", ".join(busy))
    with clean_home() as home, (bare_machine() if args.bare else contextlib.nullcontext()), \
            instrumented(args.download):
        config = Config.load()
        if args.min_free is not None:
            config.set("files.min_free_gb", args.min_free)
        if args.no_discover:
            config.set("models.discover", False)
        start, calls = time.monotonic(), hub_module.Hub.calls
        plan = manager.plan(config)
        report["plan_s"] = round(time.monotonic() - start, 1)
        report["plan_requests"] = hub_module.Hub.calls - calls
        hw = hardware.Hardware(**{k: v for k, v in plan["hardware"].items()})
        report["hardware"] = bench.fingerprint(hw)
        report["plan"] = {"runtime": (plan["runtime"] or {}).get("name"), "notes": plan["notes"],
                          "model": plan["model"] and {k: plan["model"][k] for k in
                                                     ("id", "runtime", "ctx", "need_gb", "step_s", "ability")},
                          "next": [c["id"] for c in plan["alternatives"][:3]]}
        items = downloads(plan, hw, config)
        report["downloads"] = [{"what": i["what"], "gb": i["gb"]} for i in items]
        report["download_gb"] = round(sum(i["gb"] or 0 for i in items), 2)
        report["free_gb"] = round(shutil.disk_usage(home).free / 1e9, 1)
        art = next((i.get("art") for i in items if i["what"].startswith("assistant model")), None) or {}
        speed = None if args.no_speed else speed_mb_s(art)
        if speed:
            report["speed_mb_s"] = round(speed, 1)
            report["download_min"] = round(report["download_gb"] * 1000 / speed / 60, 1)
        STEPS.clear()
        lines: list[str] = []
        start = time.monotonic()
        try:
            result = manager.setup(config, model_id=args.model, smoke=timed(smoketest.run, "smoke test"),
                                   measure=timed(bench.measure, "time the model"),
                                   progress=lambda m: lines.append(f"{time.monotonic() - start:6.1f}s  {m}"))
            report["outcome"] = "set up " + result["entry"] if result.get("ok") else "failed: " + str(result.get("error"))
        except Stopped as stop:
            report["outcome"] = f"stopped before a download it was not allowed: {stop}"
        report["setup_s"] = round(time.monotonic() - start, 1)
        report["steps"] = list(STEPS)
        own = runtimes.OllamaRuntime(config)
        if own._managed() and not own.running():        # the copy setup fetched: does it start and answer?
            began = time.monotonic()
            try:
                own.start()
                version = ".".join(map(str, own.version() or ()))
                report["own_ollama"] = f"started and answered as {version} in {time.monotonic() - began:.1f} s"
            except Exception as exc:
                report["own_ollama"] = f"did not start: {exc}"
        report["log"] = lines
    return report


def show(report: dict) -> None:
    print(f"First run of setup ({report['mode']}; downloads allowed: {report['download']})")
    hw = report["hardware"]
    print(f"  machine: {hw['chip']}, {hw['ram_gb']} GB, GPU {hw['gpu']}")
    if report.get("warning"):
        print(f"  note: {report['warning']}")
    plan = report["plan"]
    model = plan["model"] or {}
    print(f"  plan ({report['plan_s']} s, {report.get('plan_requests', 0)} requests to Hugging Face's API): {model.get('id', 'no model')} on {model.get('runtime') or plan['runtime'] or 'no runtime'}"
          + (f", {model['step_s']:.0f} s a step by estimate" if model else "") + f"; then {', '.join(plan['next']) or '-'}")
    for note in plan["notes"]:
        print(f"    plan note: {note}")
    print("  downloads of a first run:")
    for item in report["downloads"]:
        print(f"    {'?' if item['gb'] is None else format(item['gb'], '6.2f')} GB  {item['what']}")
    print(f"    {report['download_gb']:6.2f} GB in all; {report['free_gb']} GB free on the disk"
          + (f"; about {report['download_min']:.0f} min at {report['speed_mb_s']:.0f} MB/s" if report.get("speed_mb_s") else ""))
    print(f"  setup ran {report['setup_s']} s: {report['outcome']}")
    for step in report["steps"]:
        print(f"    {step['seconds']:6.1f} s  {step['step']}" + (f"  [{step['error']}]" if step.get("error") else ""))
    if report.get("own_ollama"):
        print(f"  CodeRunner's own Ollama {report['own_ollama']}")
    if report.get("served"):
        s = report["served"]
        if s.get("error"):
            print(f"  served model: {s['error']}")
        else:
            m = s["measured"]
            print(f"  served model {s['model']}: smoke test {s['smoke_s']} s "
                  f"({', '.join(k + (' ok' if v['passed'] else ' FAIL') for k, v in s['smoke'].items())}); "
                  f"timing {s['measure_s']} s: reads {m['read_tok_s']:.0f} tok/s, writes {m['gen_tok_s']:.1f} tok/s, "
                  f"{m['step_s']:.0f} s a step, first token after {m['first_token_s']:.0f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bare", action="store_true", help="as on a machine with no Ollama, llama.cpp or Homebrew")
    parser.add_argument("--download", choices=["none", "runtimes", "all"], default="none")
    parser.add_argument("--model", help="set up this model instead of the plan's pick")
    parser.add_argument("--served", action="store_true", help="also time the model this home already serves")
    parser.add_argument("--no-speed", action="store_true", help="do not sample the connection's speed")
    parser.add_argument("--min-free", type=float, help="GB of disk downloads must leave free (files.min_free_gb)")
    parser.add_argument("--no-discover", action="store_true", help="the catalog only: no model discovery")
    parser.add_argument("--json", help="also write the report to this file")
    args = parser.parse_args()
    report = run(args)
    show(report)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
