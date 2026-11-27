"""`coderunner` command line."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from . import __version__
from .core import paths
from .core.config import Config
from .server.gateway import admin_request


CHAT_TIMEOUT = 4 * 3600


def _admin(config: Config, method: str, path: str, body: Any = None, timeout: float = 900) -> Any:
    base_url = f"http://127.0.0.1:{config.get('gateway.admin_port')}"
    return admin_request(method, path, base_url=base_url, token=paths.admin_token(), json_body=body,
                         timeout=timeout)


def _daemon_up(config: Config) -> bool:
    try:
        # A request this home's token must sign, answered without waiting on anything: /api/status asks the
        # sandbox for its health, 3 s when the VM is not up, and while the daemon was starting that passed the
        # timeout, so `coderunner chat` said the daemon was not running.
        _admin(config, "GET", "/api/approvals", timeout=5)
        return True
    except Exception:
        return False


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str) if not isinstance(value, str) else value)


# ---- setup and lifecycle ---------------------------------------------------

def cmd_setup(args, config: Config) -> int:
    paths.ensure_dirs()
    paths.api_token()
    paths.admin_token()
    if args.name:
        config.set("owner.name", args.name)
    if args.network:
        config.set("sandbox.network", args.network)
    config.save()
    if args.no_models:
        print("Skipped model setup. Run `coderunner models setup` later.")
        return 0
    from .models import manager
    if args.search:
        result = manager.setup_search(config, progress=print)
        _print(result)
        if not result.get("ok"):
            return 1
        print("\nSearch mode is set up: the file index and the search tools, no chat model and no sandbox.\n"
              "Next:\n  coderunner up\n  coderunner files add ~/Documents\n"
              "  coderunner files sources add mail        # also messages, notes, calendar\n"
              "  claude mcp add coderunner-search -- coderunner mcp --tools search")
        return 0
    result = manager.setup(config, model_id=args.model, runtime=args.runtime, accept_license=args.accept_license,
                           progress=print)
    _print(result)
    if not result.get("ok"):
        return 1
    if not args.no_voice:
        import asyncio

        from .voice import Voice
        print("\nSetting up speech to text (voice messages and `coderunner chat --voice`)…")
        try:
            _print(asyncio.run(Voice(config).setup(progress=print)))
        except Exception as exc:          # voice is optional; the assistant works without it
            print(f"Voice setup failed ({exc}); retry later with `coderunner voice setup`.")
    print("\nNext: `coderunner up` to start the assistant, then `coderunner chat`.")
    return 0


def _pidfile():
    return paths.assistant_dir() / "run" / "daemon.pid"


def cmd_up(args, config: Config) -> int:
    from .sandbox import driver_for
    driver = driver_for(config)
    state = driver.status()
    if state == "missing" and config.get("sandbox.driver") != "none":
        print(NO_SANDBOX)
    _stopped_mark().unlink(missing_ok=True)
    if _daemon_up(config):
        if state not in ("running", "missing", "unavailable"):
            print(f"Starting sandbox ({driver.name})…")
            driver.start()
        print("Daemon already running.")
        return 0
    # Start the daemon first so the desktop app opens in ~1s while ComputerKeeper brings the sandbox VM up
    # asynchronously in the background, instead of blocking the splash screen for 15–30s on VM boot.
    from .server import service
    if service.status().get("installed"):
        service.restart()
    else:
        log = open(paths.assistant_dir() / "logs" / "daemon.log", "ab")
        process = subprocess.Popen([sys.executable, "-m", "coderunner", "daemon"], stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        _pidfile().write_text(str(process.pid))
    for _ in range(150):
        if _daemon_up(config):
            print(f"✅ CodeRunner is running. Owner UI: http://127.0.0.1:{config.get('gateway.admin_port')}/ "
                  "(pair a browser with `coderunner remote pair`).")
            return 0
        time.sleep(0.4)
    print(f"❌ The daemon did not start; see {paths.assistant_dir() / 'logs' / 'daemon.log'}")
    return 1


def _stopped_mark():
    """Left by `coderunner down`: the owner stopped the assistant, so the desktop app does not bring it back
    by itself (it offers a button). Any start clears it."""
    return paths.assistant_dir() / "run" / "stopped-by-owner"


def cmd_down(args, config: Config) -> int:
    from .server import service
    with contextlib.suppress(OSError):
        _stopped_mark().parent.mkdir(parents=True, exist_ok=True)
        _stopped_mark().write_text("")
    info = service.status()
    if args.uninstall and info.get("installed"):
        print(service.uninstall())
    elif info.get("pid"):
        # A clean exit is not restarted by the login service until the next `coderunner up` or login.
        os.kill(info["pid"], signal.SIGTERM)
    pidfile = _pidfile()
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        pidfile.unlink(missing_ok=True)
    if args.sandbox:
        from .sandbox import driver_for
        driver_for(config).stop()
    print("Stopped.")
    return 0


BUILD_SPACE_GB = 26                          # install.sh's own figure, read from it when it is there


def build_space_gb() -> int:
    """The free space install.sh wants for a build, from install.sh itself: a lower figure here let a reset delete
    the old sandbox and then have the installer refuse the build (2026-11-26: 20 here, 26 there, 19 free)."""
    import re
    from .sandbox import reset
    try:
        found = re.search(r"^BUILD_SPACE_GB=(\d+)", reset.installer().read_text(), re.M)
    except OSError:
        found = None
    return int(found.group(1)) if found else BUILD_SPACE_GB
NO_SANDBOX = "No sandbox is installed. Run ./install.sh (without --search) to create one."


def cmd_sandbox(args, config: Config) -> int:
    """The sandbox VM's lifecycle, the same on Apple `container` and Docker."""
    from .sandbox import driver_for
    driver = driver_for(config)
    state = driver.status()
    if state == "unavailable":
        print("No sandbox runtime here (Apple `container` on macOS, Docker elsewhere).")
        return 1
    if args.action == "status":
        print(f"{state} ({driver.name})")
        return 0
    if state == "missing" and args.action != "reset":
        print(NO_SANDBOX)
        return 1
    if args.action == "start":
        if state == "running":
            print("The sandbox is already running.")
            return 0
        ok = driver.start()
        print("Started." if ok else "❌ The sandbox did not start; see `coderunner sandbox logs`.")
        return 0 if ok else 1
    if args.action == "stop":
        if state != "running":
            print("The sandbox is not running.")
            return 0
        ok = driver.stop()
        print("Stopped; its filesystem is kept (Python kernel state is not)." if ok
              else "❌ The sandbox did not stop.")
        return 0 if ok else 1
    if args.action in ("shell", "logs"):
        if state != "running" and args.action == "shell":
            print("The sandbox is not running; start it with `coderunner sandbox start`.")
            return 1
        argv = (driver.exec_argv(args.cmdline or ["/bin/bash"], tty=sys.stdin.isatty())
                if args.action == "shell" else driver.logs_argv(args.follow))
        os.execvp(argv[0], argv)              # does not return
        return 1
    # reset: delete the VM and its filesystem, then create it again with the installer (sandbox/reset.py), keeping
    # the old VM's settings where none are given. Only the sandbox is recreated (--no-assistant), not models.
    from .sandbox import reset
    options, env = reset.options_for(driver, state, cpus=args.cpus, memory=args.memory, network=args.network,
                                     mounts=args.mount, build=args.build,
                                     desktop=(paths.assistant_dir() / "config.json").exists())
    free_gb = shutil.disk_usage(pathlib.Path.home()).free / 1e9
    # The old VM's disk is deleted before the installer builds (and checks again), so its space counts as free.
    disk = getattr(driver, "disk_bytes", lambda: None)() if state != "missing" else None
    free_gb += (disk or 0) / 1e9
    need_gb = build_space_gb()
    if args.build and free_gb < need_gb:            # the installer refuses then, after the old sandbox is gone
        print(f"Building the image needs about {need_gb} GB free; {free_gb:.0f} GB is free, counting the old sandbox's disk. Nothing was "
              "deleted; free some space, or reset without --build to reuse the current image.")
        return 1
    if state != "missing" and not args.yes:
        answer = input("Delete the sandbox and everything in its filesystem (host mounts are untouched)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing was deleted.")
            return 1
    return reset.recreate(driver, state, options, env, lambda line: print(line, flush=True))


def cmd_app(args, config: Config) -> int:
    """Open the desktop app (it starts the daemon itself when that is not running)."""
    from . import appshell
    process = appshell.launch(progress=print)
    print(f"Opened the CodeRunner app (process {process.pid}); its log is {paths.assistant_dir() / 'logs' / 'app.log'}.")
    return 0


def cmd_daemon(args, config: Config) -> int:
    from .server.daemon import main
    main()
    return 0


def cmd_status(args, config: Config) -> int:
    if _daemon_up(config):
        _print(_admin(config, "GET", "/api/status"))
        return 0
    from .sandbox import driver_for
    driver = driver_for(config)
    _print({"daemon": "stopped", "sandbox": {"driver": driver.name, "state": driver.status()},
            "model": {"runtime": config.get("models.runtime"), "chat": config.get("models.chat")}})
    return 1


def cmd_check(args, config: Config) -> int:
    """Run the core tasks on this install, as the app would, and say which worked (server/check.py)."""
    if not _daemon_up(config):
        print("The assistant is not running. Start it with `coderunner up`, then run this again.")
        return 1
    if args.last:
        shown = _admin(config, "GET", "/api/check")
        result = (shown["real"]["result"] if args.real else shown.get("research") if args.research
                  else shown.get("world") if args.world else shown.get("everyday") if args.everyday
                  else shown.get("check") or shown.get("result"))
        if not result:
            print("No check has been run yet.")
            return 1
        return _print_check(result)
    try:
        _admin(config, "POST", "/api/check", {"names": args.names, "long": args.long, "real": args.real,
                                              "suite": "research" if args.research else "world" if args.world
                                              else "everyday" if args.everyday else "",
                                              "minutes": args.minutes})
    except RuntimeError as exc:
        try:
            print(json.loads(str(exc).split(": ", 1)[1])["error"])
        except (ValueError, KeyError, IndexError):
            print(exc)
        return 1
    shown = 0
    try:
        while True:
            try:
                state = _admin(config, "GET", "/api/check", timeout=30)
            except Exception:           # the daemon is busy for a moment (a scan of the shared folders, say)
                time.sleep(2)
                continue
            for line in state["lines"][shown:]:
                print(line, flush=True)
            shown = len(state["lines"])
            if state["state"] != "running":
                break
            time.sleep(2)
    except KeyboardInterrupt:
        _admin(config, "POST", "/api/check", {"action": "stop"})
        print("\nStopping the check…")
        return 130
    return 0 if (state.get("result") or {}).get("ok") else 1


def _print_check(result: dict) -> int:
    print(f"{result['date']}  {result['model']}")
    for task in result["tasks"]:
        took = f"{task['seconds']} s" if task.get("seconds") is not None else ""
        asked = f" ({task['approvals']} approval{'s' if task['approvals'] != 1 else ''} asked)" if task.get("approvals") else ""
        print(f"  {task['state'].replace('_', ' '):<10}{task['name']:<11}{took:>7}  {task['why']}{asked}")
        if task.get("seconds_final") not in (None, task.get("seconds")):       # it carried on past its limit
            print(f"  {'  then ' + task['state_final'].replace('_', ' '):<21}{str(task['seconds_final']) + ' s':>7}"
                  f"  {task['why_final']}")
    print(f"{result['passed']} of {result['ran']} passed"
          + (f", {result['needs_you']} stopped at a step that is yours" if result.get("needs_you") else "")
          + (f", {result['skipped']} skipped" if result["skipped"] else ""))
    return 0 if result.get("ok") else 1


def cmd_doctor(args, config: Config) -> int:
    from .models import hardware
    from .sandbox import driver_for
    hw = hardware.detect()
    driver = driver_for(config)
    from .safety import selfcheck
    security = selfcheck.run(config)
    report = {"version": __version__, "hardware": hw.to_dict(), "sandbox_driver": driver.name,
              "sandbox_state": driver.status(), "daemon": _daemon_up(config),
              "browser": _browser_report(config), "security": security,
              "api_token_file": str(paths.api_token_path()), "config": str(config.path)}
    _print(report)
    failed = [c["check"] for c in security if not c["ok"] and c["severity"] == "high"]
    if failed:
        print(f"Security: {len(failed)} problem(s): {'; '.join(failed)}. See the fixes above.")
    return 1 if failed else 0


def _browser_report(config: Config) -> dict:
    from .browser import extension
    report: dict[str, Any] = {"mode": config.get("browser.mode") or "dedicated"}
    if report["mode"] == "chrome":
        report["host_registered"] = extension.installed()
        try:
            report["extension"] = (_admin(config, "GET", "/api/status", timeout=5).get("browser") or {}).get("extension")
        except Exception:
            report["extension"] = "unknown: the assistant is not running"
    return report


CONNECT_STEPS = """The CodeRunner extension is ready to load. In Chrome (or Brave, Edge):
  1. Open chrome://extensions and turn on Developer mode (top right).
  2. Click "Load unpacked" and choose:
       {extension_dir}
  3. Restart the assistant: coderunner down && coderunner up
The assistant then opens its tabs in a "CodeRunner" tab group in a window of its own. It can only
see and use the tabs in that group: drag a tab out to take it back, or in to hand it over.
Check the connection with: coderunner browser status"""


def _computer_setup(config: Config) -> int:
    """Give the sandbox VM its own desktop and Chrome, and let the assistant browse there."""
    from .sandbox import computer
    if not computer.setup(config, say=print):
        return 1
    print("Done. The assistant now browses on the sandbox's own computer, which you can watch and take over in the "
          "app. Restart the assistant: coderunner down && coderunner up")
    return 0


def cmd_browser(args, config: Config) -> int:
    from .browser import extension
    if args.action == "computer":
        return _computer_setup(config)
    if args.action == "connect":
        try:
            result = extension.install(args.browser or None)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 2
        if not result["manifests"]:
            print("No Chrome, Brave, Edge or Chromium profile folder found. Start the browser once, or name it "
                  "with --browser chrome|brave|edge|chromium.")
            return 1
        config.set("browser.mode", "chrome")
        config.save()
        for manifest in result["manifests"]:
            print(f"Registered the native host for {manifest['browser']}: {manifest['path']}")
        print(CONNECT_STEPS.format(extension_dir=result["extension_dir"]))
        return 0
    if args.action == "disconnect":
        for path in extension.uninstall():
            print(f"Removed {path}")
        config.set("browser.mode", "dedicated")
        config.save()
        print("The assistant uses its own browser profile again (restart it: coderunner down && coderunner up). "
              "You can remove the CodeRunner extension in chrome://extensions.")
        return 0
    _print(_browser_report(config))
    return 0


# ---- chat --------------------------------------------------------------------

def _approval_prompt(config: Config, approval: dict) -> None:
    print(f"\n🔐 Approval needed: {approval['summary']}\n   ({approval['kind']} · {approval['target']})")
    choices = "[o]nce, [n]o" if approval["risk"] == "high" else "[o]nce, [s]ession, [d]ay, [a]lways, [n]o"
    answer = input(f"   Allow? {choices}: ").strip().lower()[:1]
    scope = {"o": "once", "s": "session", "d": "day", "a": "always"}.get(answer)
    try:
        _admin(config, "POST", f"/api/approvals/{approval['id']}", {"allow": scope is not None,
                                                                     "scope": scope or "once"})
    except RuntimeError as exc:
        print(f"   ({exc})")


def _chat_once(config: Config, message: str, conversation: str) -> str:
    result: dict = {}

    def worker():
        try:
            # Long tasks (a browser search with a local model) can take far longer than an API call.
            result["reply"] = _admin(config, "POST", "/api/chat", {"message": message, "conversation": conversation},
                                     timeout=CHAT_TIMEOUT)["reply"]
        except Exception as exc:
            result["reply"] = f"Error: {exc}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    seen: set[str] = set()
    while thread.is_alive():
        try:
            for approval in _admin(config, "GET", "/api/approvals", timeout=5):
                if approval["id"] not in seen:
                    seen.add(approval["id"])
                    if sys.stdin.isatty():
                        _approval_prompt(config, approval)
                    else:
                        print(f"Approval needed: {approval['summary']} "
                              f"(coderunner approvals approve {approval['id']})", file=sys.stderr)
        except Exception:
            pass
        thread.join(1)
    return result.get("reply", "")


def _record(voice) -> str | None:
    """Record from the default microphone until Enter is pressed; the path of the recording, or None."""
    import tempfile
    ffmpeg = voice.ffmpeg()
    if not ffmpeg:
        print("Recording needs ffmpeg; run `coderunner voice setup`.")
        return None
    source = ["-f", "avfoundation", "-i", ":default"] if sys.platform == "darwin" else ["-f", "pulse", "-i", "default"]
    path = os.path.join(tempfile.mkdtemp(prefix="coderunner-mic-"), "message.wav")
    proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", *source, "-ac", "1", "-ar", "16000", path],
                            stdin=subprocess.PIPE)
    try:
        input("  listening… press Enter to send ")
    finally:
        if proc.poll() is None:
            proc.communicate(b"q", timeout=10)        # ffmpeg finishes the file when it reads q
    return path if os.path.exists(path) and os.path.getsize(path) > 1000 else None


def _speak_aloud(voice, text: str) -> None:
    tts = voice.tts()
    if tts and tts[0] == "say" and text:
        subprocess.run([tts[1]], input=text.encode(), check=False)   # text on stdin, never as arguments


def _voice_chat(config: Config, conversation: str) -> int:
    import asyncio
    import shutil

    from .voice import Voice
    voice = Voice(config)
    if not voice.available():
        print("Speech to text is not set up yet. Run `coderunner voice setup` first.")
        return 1
    print("Talk to your assistant: press Enter to start speaking, Enter again to send. Ctrl-D to exit.")
    while True:
        try:
            input("\n🎤 ")
            path = _record(voice)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not path:
            print("Nothing was recorded (is the microphone allowed for this terminal?).")
            continue
        try:
            text = asyncio.run(voice.transcribe(path)).strip()
        finally:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        if not text:
            print("I couldn't make out any words.")
            continue
        print(f"› {text}")
        reply = _chat_once(config, text, conversation)
        print(reply)
        _speak_aloud(voice, reply)


def cmd_chat(args, config: Config) -> int:
    if not _daemon_up(config):
        print("The daemon is not running. Start it with `coderunner up`.")
        return 1
    if args.voice:
        return _voice_chat(config, args.conversation)
    if args.message:
        print(_chat_once(config, " ".join(args.message), args.conversation))
        return 0
    print("Chat with your assistant. Ctrl-D to exit.")
    while True:
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line:
            print(_chat_once(config, line, args.conversation))


# ---- approvals, grants, audit --------------------------------------------------

def cmd_approvals(args, config: Config) -> int:
    if args.action == "list":
        _print(_admin(config, "GET", "/api/approvals"))
    else:
        _print(_admin(config, "POST", f"/api/approvals/{args.id}",
                      {"allow": args.action == "approve", "scope": args.scope}))
    return 0


def cmd_grants(args, config: Config) -> int:
    if args.action == "revoke":
        _print(_admin(config, "DELETE", f"/api/grants/{args.id}"))
    else:
        _print(_admin(config, "GET", "/api/grants"))
    return 0


def cmd_audit(args, config: Config) -> int:
    for record in _admin(config, "GET", f"/api/audit?limit={args.limit}"):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.pop("ts")))
        print(stamp, record.pop("event"), json.dumps(record, default=str)[:300])
    return 0


# ---- models ------------------------------------------------------------------

def cmd_claude(args, config: Config) -> int:
    """Remote mode: the planner on Anthropic's API with the owner's key. Not private, and said so."""
    from .models import remote
    if args.action == "status":
        now = remote.active(config)
        print(f"Remote mode is on: {now['name']} on Anthropic's API. {remote.WARNING}" if now else
              "Remote mode is off: the model runs on this computer."
              + (" An Anthropic key is kept." if remote.key() else ""))
        return 0
    try:
        if args.action == "on":
            if args.key_file:
                remote.keep_key_file(args.key_file)
            elif not remote.key():
                remote.keep_key(getpass.getpass("Anthropic API key (not shown): "))
            if _daemon_up(config):
                _admin(config, "POST", "/api/remote", {"action": "on", "model": args.model})
            else:
                remote.turn_on(config, args.model)
            print(f"Remote mode is on: {remote.MODELS[args.model]}.\n{remote.WARNING}\n"
                  "File search and voice stay on this computer. `coderunner claude off` goes back to the local model.")
        else:
            if _daemon_up(config):
                _admin(config, "POST", "/api/remote", {"action": "off", "forget": args.forget})
            else:
                remote.turn_off(config, forget=args.forget)
            print("Remote mode is off: the model runs on this computer." + (" The key was deleted." if args.forget else ""))
    except (ValueError, RuntimeError) as exc:
        print(str(exc).split(": ", 1)[-1] if str(exc).startswith("400") else exc)
        return 1
    return 0


def cmd_models(args, config: Config) -> int:
    from .models import manager
    if args.action == "plan":
        _print(manager.plan(config))
    elif args.action == "setup":
        result = manager.setup(config, model_id=args.model, runtime=args.runtime,
                               accept_license=args.accept_license, progress=print)
        _print(result)
        return 0 if result.get("ok") else 1
    elif args.action == "driver":
        result = manager.setup_driver(config, model_id=args.model, progress=print, off=args.off)
        _print(result)
        return 0 if result.get("ok") else 1
    elif args.action == "discover":
        from .models import discover, evidence, hardware
        hw = hardware.detect(float(config.get("sandbox.memory_gb") or 1))
        models = discover.cached(progress=print, refresh=True, guess_mlx=hw.gpu == "apple")
        rows = []
        for model in models:
            for runtime in ("ollama", "llama.cpp", "mlx") if hw.gpu == "apple" else ("ollama", "llama.cpp"):
                entry = discover.best_build(model, hw, config, runtime)
                if entry:
                    art = next(iter(entry["artifacts"].values()))
                    rows.append({"model": model["repo"], "released": model["released"], "params_b": model["params_b"],
                                 "runtime": runtime, "build": art.get("quant") or art.get("repo"),
                                 "size_gb": art["size_gb"], "license": model["license"]["spdx"],
                                 # From published benchmark scores; None when the model has none yet.
                                 "ability": evidence.ability(entry, art.get("bits"))})
                    break
        rows.sort(key=lambda r: -(r["ability"] if r["ability"] is not None else -9))
        _print({"found": len(models), "fit_this_machine": rows})
    elif args.action == "test":
        import asyncio
        from .models import smoketest
        endpoint = manager.endpoint(config)
        _print(asyncio.run(smoketest.run(endpoint.base_url, endpoint.model, vision=endpoint.vision)))
    elif args.action == "bench" and args.probe:
        from . import eval as eval_pkg
        result = eval_pkg.run_tier0(config, force=True, include_cdn=True, progress=print)
        _print(result)
        return 0 if result.get("ok") else 1
    elif args.action == "bench" and args.compare is not None:
        import asyncio
        from . import eval as eval_pkg
        specs = eval_pkg.default_candidates(config)
        if args.compare and args.compare != ["auto"]:
            wanted = set(args.compare)
            specs = [s for s in specs if s.id in wanted or s.planner in wanted] or [
                eval_pkg.StackSpec.from_dict({"planner": name}) for name in args.compare
            ]
        result = asyncio.run(eval_pkg.compare_stacks(config, specs[:4], suite=args.suite, progress=print))
        if result.get("markdown"):
            print(result["markdown"], end="")
        else:
            _print(result)
        return 0 if result.get("ok") else 1
    elif args.action == "bench" and args.embedding:
        from .models import coreml
        used = manager.setup_coreml_embedding(config, progress=print, remeasure=True)
        config.save()
        record = coreml.load(manager._hardware(config, None))
        if record:
            print(f"Neural Engine {record['coreml_per_s']} passages/s, GPU {record['gpu_per_s']} passages/s, "
                  f"vectors agree at cosine {record['cosine']}")
        print(f"Text embeddings: {'Core ML (Neural Engine)' if used == 'coreml' else 'GPU'}")
        return 0
    elif args.action == "bench" and args.tasks and args.candidates:
        from .models import tasks
        result = manager.bench_candidates(config, count=args.candidates, progress=print)
        machine = tasks.fingerprint(tasks.hardware.detect(float(config.get("sandbox.memory_gb") or 1)))
        print(tasks.report([r for r in tasks.load() if r["machine"] == machine and r["suite"] == tasks.SUITE]))
        print(f"Assistant model: {(result['model'] or {}).get('id', 'none')}")
        return 0
    elif args.action == "bench" and args.tasks:
        import asyncio
        from .models import tasks
        result = asyncio.run(tasks.run(config, names=args.only, visible=args.visible))
        if not result.get("ok"):
            print(result["error"])
            return 1
        print(tasks.report([r for r in tasks.load() if r["machine"] == result["machine"] and r["suite"] == tasks.SUITE]))
        return 0 if result["passed"] == result["total"] else 1
    elif args.action == "bench":
        result = manager.bench_here(config, model=args.model)
        if not result.get("ok"):
            print(result["error"])
            return 1
        if args.report:
            print(result["report"], end="")
        else:
            _print({key: result[key] for key in ("model", "measured", "calibration")} |
                   {"plan": result["plan"]["model"], "decision": result["plan"]["decision"]})
    return 0


# ---- secrets -----------------------------------------------------------------

def cmd_secrets(args, config: Config) -> int:
    from .safety.secrets_store import SecretStore
    store = SecretStore()
    if args.action == "set":
        if not args.host:
            print("Name at least one --host the secret may be sent to.")
            return 2
        value = getpass.getpass(f"Value for {args.name} (input hidden): ")
        store.set(args.name, value)
        config.set(f"secrets.{args.name}", {"hosts": args.host})
        config.save()
        print(f"Saved. Sandbox code can use it as a header value: {{{{secret:{args.name}}}}} "
              f"(only for {', '.join(args.host)}).")
    elif args.action == "delete":
        store.delete(args.name)
        secrets_cfg = config.get("secrets", {}) or {}
        secrets_cfg.pop(args.name, None)
        config.set("secrets", secrets_cfg)
        config.save()
        print("Deleted.")
    else:
        _print({name: (config.get("secrets", {}) or {}).get(name, {}) for name in store.names()})
    return 0


# ---- channels, connectors, remote, service, schedules -------------------------

def cmd_channels(args, config: Config) -> int:
    if args.action == "list":
        _print(config.get("channels", {}))
        return 0
    if args.action == "enable":
        settings = config.get(f"channels.{args.name}", {}) or {}
        settings["enabled"] = True
        for item in args.set or []:
            key, _, value = item.partition("=")
            try:
                settings[key] = json.loads(value)
            except json.JSONDecodeError:
                settings[key] = value
        config.set(f"channels.{args.name}", settings)
        if args.name == "telegram" and not args.no_token:
            from .safety.secrets_store import SecretStore
            SecretStore().set("telegram_bot_token", getpass.getpass("Telegram bot token from @BotFather: "))
        config.save()
        print(f"Enabled {args.name}. Restart the daemon (`coderunner up` after `coderunner down`), then run "
              f"`coderunner channels pair {args.name}`.")
    elif args.action == "disable":
        config.set(f"channels.{args.name}.enabled", False)
        config.save()
        print(f"Disabled {args.name}.")
    elif args.action == "pair":
        from .channels.base import Channel
        from .core.store import Store
        from .core.audit import AuditLog

        class _Svc:  # pairing codes only need the shared store
            store = Store()
            audit = AuditLog()
        channel = Channel({}, _Svc, None)
        channel.name = args.name
        code = channel.start_pairing()
        print(f"From the account you want to own this assistant, send:  pair {code}\n(valid for 10 minutes)")
    return 0


def cmd_connectors(args, config: Config) -> int:
    from .connectors.macos import SENDERS
    enabled = list(config.get("connectors.enabled", []) or [])
    sending = list(config.get("connectors.send", []) or [])
    if getattr(args, "send", False) and args.name not in SENDERS:
        print(f"--send is for: {', '.join(SENDERS)}.")
        return 1
    if args.action == "enable" and args.name:
        if args.name not in enabled:
            enabled.append(args.name)
        if args.send and args.name not in sending:
            sending.append(args.name)
            print(f"The assistant can now send from {args.name}. It asks you before every message, showing who it "
                  "goes to and its whole text.")
    elif args.action == "disable" and args.name:
        if args.send:                           # sending off, the connector stays
            sending = [name for name in sending if name != args.name]
        else:
            enabled = [name for name in enabled if name != args.name]
            sending = [name for name in sending if name != args.name]
    elif args.action == "test":
        import asyncio
        from . import connectors
        print(f"Running a harmless {args.name} read so macOS can ask for permission…")
        try:
            print(asyncio.run(connectors.check(args.name)))
        except Exception as exc:
            print(f"Failed: {exc}")
            return 1
        return 0
    config.set("connectors.enabled", enabled)
    config.set("connectors.send", sending)
    config.save()
    _print({"enabled": enabled, "send": sending})
    return 0


def cmd_remote(args, config: Config) -> int:
    from .server import remote
    if args.action == "enable":
        url = remote.enable(int(config.get("gateway.admin_port")))
        config.set("remote.provider", "tailscale")
        if args.owner:
            config.set("remote.owners", args.owner)
        config.save()
        print(f"Reachable from your tailnet at {url}\nPair your phone: run `coderunner remote pair` and open {url}pair")
    elif args.action == "disable":
        print(remote.disable(admin_port=int(config.get("gateway.admin_port"))))
        config.set("remote.provider", "")
        config.save()
        print("Remote access disabled.")
    elif args.action == "pair":
        result = _admin(config, "POST", "/api/pair")
        print(f"Pairing code: {result['code']}  (valid {result['expires_in'] // 60} minutes, single use)")
    elif args.action == "devices":
        _print(_admin(config, "GET", "/api/devices"))
    else:
        _print(remote.status(int(config.get("gateway.admin_port"))))
    return 0


def cmd_service(args, config: Config) -> int:
    from .server import service
    if args.action == "install":
        _print(service.install())
    elif args.action == "uninstall":
        _print(service.uninstall())
    elif args.action == "restart":
        _print(service.restart())
    else:
        _print(service.status())
    return 0


def cmd_schedule(args, config: Config) -> int:
    if args.action == "add":
        _print(_admin(config, "POST", "/api/schedules", {"name": args.name, "spec": args.spec,
                                                         "prompt": args.prompt}))
    elif args.action == "remove":
        _print(_admin(config, "DELETE", f"/api/schedules/{args.id}"))
    else:
        _print(_admin(config, "GET", "/api/schedules"))
    return 0


def cmd_files(args, config: Config) -> int:
    target = " ".join(args.target or [])
    if args.action == "search":
        return _find(config, target, args.kind, args.limit, args.json)
    if args.action == "add":
        _print(_admin(config, "POST", "/api/files", {"add": os.path.abspath(os.path.expanduser(target))}))
    elif args.action == "remove":
        _print(_admin(config, "POST", "/api/files", {"remove": os.path.abspath(os.path.expanduser(target))}))
    elif args.action == "sources":
        # coderunner files sources [add|remove <mail|messages|notes|calendar>]
        words = args.target or []
        if len(words) == 2 and words[0] in ("add", "remove"):
            _print(_admin(config, "POST", "/api/files", {f"source_{words[0]}": words[1]}))
        elif words:
            print("Usage: coderunner files sources [add|remove mail|messages|notes|calendar]")
            return 2
        else:
            _print((_admin(config, "GET", "/api/files") or {}).get("sources", {}))
    elif args.action == "index":
        _print(_admin(config, "POST", "/api/files", {"budget": 100000}, timeout=86400))
    else:
        _print(_admin(config, "GET", "/api/files"))
    return 0


def cmd_find(args, config: Config) -> int:
    return _find(config, " ".join(args.query), args.kind, args.limit, args.json)


def _find(config: Config, query: str, kind: str | None, limit: int, as_json: bool) -> int:
    """Search the shared folders: through the daemon (index and model already loaded), else directly."""
    from urllib.parse import urlencode
    if not query.strip():
        print("Say what to look for, e.g. coderunner find \"boarding pass for the Delhi trip\"")
        return 2
    start = time.monotonic()
    params = {"q": query, "limit": limit, **({"kind": kind} if kind else {})}
    try:
        # Straight to the daemon: asking /api/status first (whether it runs) took 3 s of a 3.3 s search.
        found = _admin(config, "GET", f"/api/files?{urlencode(params)}", timeout=60)
    except Exception:              # no daemon (or not this home's): search the index here
        import asyncio
        found = asyncio.run(_find_here(config, query, kind, limit))
    if as_json or not isinstance(found, list):
        _print(found)
        return 0
    from .files.extract import clock
    home = os.path.expanduser("~")
    for number, item in enumerate(found, 1):
        path = "~" + item["path"][len(home):] if item["path"].startswith(home + os.sep) else item["path"]
        where = (f"page {item['page']}" if "page" in item else f"line {item['line']}" if "line" in item
                 else f"at {clock(item['at'])}" if "at" in item else "")
        notes = [item.get("kind", ""), where, "in iCloud" if item.get("in_icloud") else ""]
        print(f"{number:2}. {path}  ({', '.join(note for note in notes if note)})")
        if item.get("copies"):
            print("    same file at " + ", ".join("~" + p[len(home):] if p.startswith(home + os.sep) else p
                                                for p in item["copies"]))
        if item.get("versions"):
            print("    also at " + ", ".join("~" + p[len(home):] if p.startswith(home + os.sep) else p
                                           for p in item["versions"]))
        if item.get("snippet"):
            print(f"    {item['snippet'][:160]}")
    print("Nothing matched in the shared folders." if not found else "", f"({(time.monotonic() - start) * 1000:.0f} ms)")
    return 0


async def _find_here(config: Config, query: str, kind: str | None, limit: int) -> list[dict]:
    import asyncio
    from .core.audit import AuditLog
    from .files import Embedder, FileIndex
    from .core.store import Store
    model = config.get("files.embedding") or config.get("models.embedding")
    url = config.get("models.embedding_url") or "http://127.0.0.1:11434"
    embedder = Embedder(url, model) if model else None
    if embedder is not None:
        from .models import manager

        async def recover() -> bool:             # the Core ML server exits when idle; start it again
            return await asyncio.to_thread(manager.restart_embedding, config)
        embedder.recover = recover
    index = FileIndex(config, Store(), AuditLog(paths.audit_path()), embedder)
    try:
        return await index.search(query, kind, limit)
    finally:
        if embedder is not None:
            await embedder.aclose()


def cmd_voice(args, config: Config) -> int:
    import asyncio

    from .voice import Voice
    if args.language:
        config.set("voice.language", args.language)
        config.save()
    voice = Voice(config)
    if args.action == "setup":
        result = asyncio.run(voice.setup(progress=print, engine=args.engine))
        _print(result)
        return 0 if result.get("ok") else 1
    if args.action == "transcribe":
        if not args.file:
            print("Say which audio file to transcribe.")
            return 2
        print(asyncio.run(voice.transcribe(args.file)))
        return 0
    _print(voice.status())
    return 0


def cmd_token(args, config: Config) -> int:
    print(paths.admin_token() if args.admin else paths.api_token())
    return 0


def cmd_mcp(args, config: Config) -> int:
    from .server.mcp_bridge import main
    main(tools=args.tools)
    return 0


# ---- parser --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coderunner", description="Private local assistant runtime.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="Detect hardware, pick and verify a local model")
    p.add_argument("--name", help="What the assistant should call you")
    p.add_argument("--model"), p.add_argument("--runtime")
    p.add_argument("--network", choices=["default", "none", "proxy"])
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--no-voice", action="store_true", help="Skip speech-to-text setup")
    p.add_argument("--search", action="store_true",
                   help="Search mode: the embedding model and the file index only (no chat model, no sandbox)")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("up", help="Start the sandbox and the assistant daemon")
    p.set_defaults(func=cmd_up)
    p = sub.add_parser("down", help="Stop the assistant daemon")
    p.add_argument("--sandbox", action="store_true", help="Also stop the sandbox VM")
    p.add_argument("--uninstall", action="store_true", help="Also remove the login service")
    p.set_defaults(func=cmd_down)
    p = sub.add_parser("sandbox", help="Start, stop, open a shell in, or recreate the sandbox VM")
    sandbox = p.add_subparsers(dest="action", required=True)
    sandbox.add_parser("status")
    sandbox.add_parser("start")
    sandbox.add_parser("stop", help="Stop it; the filesystem is kept")
    sp = sandbox.add_parser("shell", help="A shell (or the given command) inside the sandbox")
    sp.add_argument("cmdline", nargs=argparse.REMAINDER, metavar="command")
    sp = sandbox.add_parser("logs")
    sp.add_argument("-f", "--follow", action="store_true")
    sp = sandbox.add_parser("reset", help="Delete the sandbox and its filesystem and create it again; settings "
                                          "not given are kept from the old one")
    sp.add_argument("--yes", action="store_true", help="Do not ask before deleting")
    sp.add_argument("--cpus"), sp.add_argument("--memory")
    sp.add_argument("--mount", action="append", help="HOST:/workspace/NAME[:rw], repeatable; replaces the old mounts")
    sp.add_argument("--network", choices=["default", "none", "proxy"])
    sp.add_argument("--build", action="store_true", help="Build the image from this checkout")
    p.set_defaults(func=cmd_sandbox)
    sub.add_parser("app", help="Open the desktop app: tasks, the computer's screen, and first-run setup"
                   ).set_defaults(func=cmd_app)
    sub.add_parser("daemon", help="Run the daemon in the foreground").set_defaults(func=cmd_daemon)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="Show hardware and environment checks").set_defaults(func=cmd_doctor)
    p = sub.add_parser("check", help="Do the core tasks once on this install, as the app would, and say which worked")
    p.add_argument("names", nargs="*", help="only these (answer, picture, document, video, code, files, website, "
                                            "flights, booking, appliance, errands, driver)")
    p.add_argument("--long", action="store_true", help="also the long ones: flights, booking, appliance, errands, driver")
    p.add_argument("--last", action="store_true", help="show the last run's result instead of running")
    p.add_argument("--real", action="store_true",
                   help="twenty tasks and five forms on real websites (a trip planned, prices compared, a cart taken "
                        "to the step before payment, a court case search...): nothing is signed in to, paid for "
                        "or submitted")
    p.add_argument("--research", action="store_true",
                   help="twelve questions answered from several real pages, most with an answer that can be checked")
    p.add_argument("--world", action="store_true",
                   help="thirteen tasks away from India (San Francisco homes, London rents, US tax, new films and "
                        "series by their reviews), several in German, "
                        "French, Spanish, Japanese or Portuguese and answered in that language")
    p.add_argument("--everyday", action="store_true",
                   help="fourteen things people ask a personal assistant for (a dentist or a table booked up to the "
                        "owner's details, a job application up to the CV, dinners turned into a priced shopping list, "
                        "subscriptions found in a bank statement, a morning brief and price watches set up as routines)")
    p.add_argument("--minutes", type=int, default=0, help="one time limit for every task, in minutes")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("chat", help="Talk to the assistant; approvals are asked here")
    p.add_argument("message", nargs="*")
    p.add_argument("--conversation", default="cli")
    p.add_argument("--voice", action="store_true", help="Speak instead of typing; replies are read aloud on macOS")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("approvals")
    p.add_argument("action", choices=["list", "approve", "deny"], nargs="?", default="list")
    p.add_argument("id", nargs="?")
    p.add_argument("--scope", default="once", choices=["once", "session", "hour", "day", "always"])
    p.set_defaults(func=cmd_approvals)
    p = sub.add_parser("grants")
    p.add_argument("action", choices=["list", "revoke"], nargs="?", default="list")
    p.add_argument("id", nargs="?")
    p.set_defaults(func=cmd_grants)
    p = sub.add_parser("audit")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("models", help="plan, set up, test, discover, or bench (time the model on this machine)")
    p.add_argument("action", choices=["plan", "setup", "test", "discover", "bench", "driver"])
    p.add_argument("--off", action="store_true", help="with driver: switch the browser driver off")
    p.add_argument("--model"), p.add_argument("--runtime")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--report", action="store_true", help="bench: print a summary to share (no names or paths)")
    p.add_argument("--tasks", action="store_true",
                   help="bench: run sample tasks end to end (local test sites and documents) with the served model")
    p.add_argument("--only", action="append", help="bench --tasks: run only this task (flights, evening, library, documents, appliance)")
    p.add_argument("--visible", action="store_true", help="bench --tasks: show the browser window")
    p.add_argument("--embedding", action="store_true",
                   help="bench: time text embeddings on the Neural Engine (Core ML) against the GPU and use the faster")
    p.add_argument("--probe", action="store_true",
                   help="bench: measure bare-metal GPU bandwidth, FP16 compute, Neural Engine TOPS and GPU+ANE isolation")
    p.add_argument("--compare", nargs="*", metavar="STACK",
                   help="bench: compare single models or multi-model combinations side by side")
    p.add_argument("--suite", default="tier1", choices=["tier1", "tier2", "ane"],
                   help="bench --compare: suite to run (tier1 90s gate, tier2 multi-domain tasks, or ane)")
    p.add_argument("--candidates", type=int, default=0,
                   help="bench --tasks: also measure the best N candidates not measured on this machine yet, "
                        "one at a time, keep the best and delete the others' downloads")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("claude", help="Use Claude on Anthropic's API as the model, with your own key (not private)")
    p.add_argument("action", choices=["on", "off", "status"], nargs="?", default="status")
    p.add_argument("--key-file", help="on: a file that holds your Anthropic API key (it is copied to CodeRunner's own "
                                      "storage, readable only by you; otherwise the key is asked for, unseen)")
    p.add_argument("--model", default="claude-haiku-5-5", choices=["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"])
    p.add_argument("--forget", action="store_true", help="off: also delete the kept key")
    p.set_defaults(func=cmd_claude)

    p = sub.add_parser("secrets", help="Secrets injected at the network boundary, never shown to the sandbox")
    p.add_argument("action", choices=["list", "set", "delete"], nargs="?", default="list")
    p.add_argument("name", nargs="?")
    p.add_argument("--host", action="append", help="Host allowed to receive this secret (repeatable)")
    p.set_defaults(func=cmd_secrets)

    p = sub.add_parser("channels")
    p.add_argument("action", choices=["list", "enable", "disable", "pair"], nargs="?", default="list")
    p.add_argument("name", nargs="?", choices=["telegram", "imessage", "signal", "whatsapp"])
    p.add_argument("--set", action="append", help="key=value channel setting (repeatable)")
    p.add_argument("--no-token", action="store_true")
    p.set_defaults(func=cmd_channels)

    p = sub.add_parser("connectors")
    p.add_argument("action", choices=["list", "enable", "disable", "test"], nargs="?", default="list")
    p.add_argument("name", nargs="?", choices=["calendar", "reminders", "notes", "mail", "contacts", "settings",
                                               "phone"])
    p.add_argument("--send", action="store_true",
                   help="with enable: also let the assistant send (mail), each message approved by you; with "
                        "disable: switch only sending off")
    p.set_defaults(func=cmd_connectors)

    p = sub.add_parser("remote", help="Private access from your devices over Tailscale")
    p.add_argument("action", choices=["status", "enable", "disable", "pair", "devices"], nargs="?",
                   default="status")
    p.add_argument("--owner", action="append", help="Tailscale login allowed to use paired devices")
    p.set_defaults(func=cmd_remote)

    p = sub.add_parser("service", help="Run the daemon at login")
    p.add_argument("action", choices=["install", "uninstall", "restart", "status"], nargs="?", default="status")
    p.set_defaults(func=cmd_service)

    p = sub.add_parser("schedule")
    p.add_argument("action", choices=["list", "add", "remove"], nargs="?", default="list")
    p.add_argument("--name"), p.add_argument("--spec"), p.add_argument("--prompt")
    p.add_argument("--id", type=int)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("browser", help="Browse on the sandbox's own computer, in a CodeRunner tab group in your everyday "
                                       "Chrome, or with a separate profile")
    p.add_argument("action", choices=["status", "computer", "connect", "disconnect"], nargs="?", default="status")
    p.add_argument("--browser", action="append", help="chrome, brave, edge, chromium... (default: all installed)")
    p.set_defaults(func=cmd_browser)

    kinds = ["text", "doc", "image", "audio", "video"]
    p = sub.add_parser("files", help="Folders the assistant may search and read (text, code, images, audio, video)")
    p.add_argument("action", choices=["status", "add", "remove", "sources", "index", "search"], nargs="?",
                   default="status")
    p.add_argument("target", nargs="*", help="folder to add or remove, a source (sources add mail), or what to search for")
    p.add_argument("--kind", choices=kinds)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_files)
    p = sub.add_parser("find", help="Find files in the shared folders by describing them")
    p.add_argument("query", nargs="+")
    p.add_argument("--kind", choices=kinds)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("voice", help="On-device speech to text: Parakeet on the Neural Engine, Apple speech, else "
                                     "Qwen3-ASR on llama.cpp")
    p.add_argument("action", choices=["status", "setup", "transcribe"], nargs="?", default="status")
    p.add_argument("file", nargs="?", help="audio file to transcribe")
    p.add_argument("--engine", choices=["parakeet-coreml", "apple", "qwen3-asr"], help="set up only this engine")
    p.add_argument("--language", help='spoken language, e.g. "en-IN" or "hi"; "auto" follows the system')
    p.set_defaults(func=cmd_voice)

    p = sub.add_parser("token", help="Print the sandbox API token (or --admin)")
    p.add_argument("--admin", action="store_true")
    p.set_defaults(func=cmd_token)
    p = sub.add_parser("mcp", help="MCP stdio bridge to the assistant's guarded tools")
    p.add_argument("--tools", choices=["all", "search"], default="all",
                   help="all of the assistant's tools, or only search, reading and memory")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    config = Config.load()
    needs_name = {"secrets": ("set", "delete"), "channels": ("enable", "disable", "pair"),
                  "connectors": ("enable", "disable", "test")}
    if args.command in needs_name and args.action in needs_name[args.command] and not args.name:
        print(f"`coderunner {args.command} {args.action}` needs a name.")
        return 2
    if args.command in ("approvals", "grants") and args.action != "list" and not args.id:
        print("An id is required.")
        return 2
    try:
        return args.func(args, config) or 0
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
