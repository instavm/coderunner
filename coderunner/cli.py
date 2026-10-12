"""`coderunner` command line."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from . import __version__, paths
from .config import Config
from .gateway import admin_request


CHAT_TIMEOUT = 4 * 3600


def _admin(config: Config, method: str, path: str, body: Any = None, timeout: float = 900) -> Any:
    base_url = f"http://127.0.0.1:{config.get('gateway.admin_port')}"
    return admin_request(method, path, base_url=base_url, token=paths.admin_token(), json_body=body,
                         timeout=timeout)


def _daemon_up(config: Config) -> bool:
    try:
        _admin(config, "GET", "/api/status", timeout=5)
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
    result = manager.setup(config, model_id=args.model, runtime=args.runtime, accept_license=args.accept_license,
                           progress=print)
    _print(result)
    if not result.get("ok"):
        return 1
    print("\nNext: `coderunner up` to start the assistant, then `coderunner chat`.")
    return 0


def _pidfile():
    return paths.assistant_dir() / "run" / "daemon.pid"


def cmd_up(args, config: Config) -> int:
    from .sandbox import driver_for
    driver = driver_for(config)
    state = driver.status()
    if state not in ("running", "missing", "unavailable"):
        print(f"Starting sandbox ({driver.name})…")
        driver.start()
    elif state == "missing":
        print("No sandbox container yet. Run ./install.sh first.")
    if _daemon_up(config):
        print("Daemon already running.")
        return 0
    from . import service
    if service.status().get("installed"):
        service.restart()
    else:
        log = open(paths.assistant_dir() / "logs" / "daemon.log", "ab")
        process = subprocess.Popen([sys.executable, "-m", "coderunner", "daemon"], stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        _pidfile().write_text(str(process.pid))
    for _ in range(60):
        if _daemon_up(config):
            print(f"✅ CodeRunner is running. Owner UI: http://127.0.0.1:{config.get('gateway.admin_port')}/ "
                  "(pair a browser with `coderunner remote pair`).")
            return 0
        time.sleep(1)
    print(f"❌ The daemon did not start; see {paths.assistant_dir() / 'logs' / 'daemon.log'}")
    return 1


def cmd_down(args, config: Config) -> int:
    from . import service
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


def cmd_daemon(args, config: Config) -> int:
    from .daemon import main
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


def cmd_doctor(args, config: Config) -> int:
    from .models import hardware
    from .sandbox import driver_for
    hw = hardware.detect()
    driver = driver_for(config)
    report = {"version": __version__, "hardware": hw.to_dict(), "sandbox_driver": driver.name,
              "sandbox_state": driver.status(), "daemon": _daemon_up(config),
              "browser": _browser_report(config),
              "api_token_file": str(paths.api_token_path()), "config": str(config.path)}
    _print(report)
    return 0


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


def cmd_browser(args, config: Config) -> int:
    from .browser import extension
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


def cmd_chat(args, config: Config) -> int:
    if not _daemon_up(config):
        print("The daemon is not running. Start it with `coderunner up`.")
        return 1
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

def cmd_models(args, config: Config) -> int:
    from .models import manager
    if args.action == "plan":
        _print(manager.plan(config))
    elif args.action == "setup":
        result = manager.setup(config, model_id=args.model, runtime=args.runtime,
                               accept_license=args.accept_license, progress=print)
        _print(result)
        return 0 if result.get("ok") else 1
    elif args.action == "discover":
        from .models import discover, evidence, hardware
        hw = hardware.detect(float(config.get("sandbox.memory_gb") or 1))
        models = discover.cached(progress=print, refresh=True)
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
    from .secrets_store import SecretStore
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
            from .secrets_store import SecretStore
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
        from .store import Store
        from .audit import AuditLog

        class _Svc:  # pairing codes only need the shared store
            store = Store()
            audit = AuditLog()
        channel = Channel({}, _Svc, None)
        channel.name = args.name
        code = channel.start_pairing()
        print(f"From the account you want to own this assistant, send:  pair {code}\n(valid for 10 minutes)")
    return 0


def cmd_connectors(args, config: Config) -> int:
    enabled = list(config.get("connectors.enabled", []) or [])
    if args.action == "enable" and args.name not in enabled:
        enabled.append(args.name)
    elif args.action == "disable" and args.name in enabled:
        enabled.remove(args.name)
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
    config.save()
    _print({"enabled": enabled})
    return 0


def cmd_remote(args, config: Config) -> int:
    from . import remote
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
    from . import service
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
    if args.action == "add":
        _print(_admin(config, "POST", "/api/files", {"add": os.path.abspath(os.path.expanduser(args.target))}))
    elif args.action == "remove":
        _print(_admin(config, "POST", "/api/files", {"remove": os.path.abspath(os.path.expanduser(args.target))}))
    elif args.action == "index":
        _print(_admin(config, "POST", "/api/files", {"budget": 100000}, timeout=86400))
    elif args.action == "search":
        from urllib.parse import quote
        _print(_admin(config, "GET", f"/api/files?q={quote(args.target or '')}"))
    else:
        _print(_admin(config, "GET", "/api/files"))
    return 0


def cmd_token(args, config: Config) -> int:
    print(paths.admin_token() if args.admin else paths.api_token())
    return 0


def cmd_mcp(args, config: Config) -> int:
    from .mcp_bridge import main
    main()
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
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("up", help="Start the sandbox and the assistant daemon")
    p.set_defaults(func=cmd_up)
    p = sub.add_parser("down", help="Stop the assistant daemon")
    p.add_argument("--sandbox", action="store_true", help="Also stop the sandbox VM")
    p.add_argument("--uninstall", action="store_true", help="Also remove the login service")
    p.set_defaults(func=cmd_down)
    sub.add_parser("daemon", help="Run the daemon in the foreground").set_defaults(func=cmd_daemon)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="Show hardware and environment checks").set_defaults(func=cmd_doctor)

    p = sub.add_parser("chat", help="Talk to the assistant; approvals are asked here")
    p.add_argument("message", nargs="*")
    p.add_argument("--conversation", default="cli")
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
    p.add_argument("action", choices=["plan", "setup", "test", "discover", "bench"])
    p.add_argument("--model"), p.add_argument("--runtime")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--report", action="store_true", help="bench: print a summary to share (no names or paths)")
    p.set_defaults(func=cmd_models)

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
    p.add_argument("name", nargs="?", choices=["calendar", "reminders", "notes", "mail", "contacts"])
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

    p = sub.add_parser("browser", help="Use a CodeRunner tab group in your everyday Chrome, or a separate profile")
    p.add_argument("action", choices=["status", "connect", "disconnect"], nargs="?", default="status")
    p.add_argument("--browser", action="append", help="chrome, brave, edge, chromium... (default: all installed)")
    p.set_defaults(func=cmd_browser)

    p = sub.add_parser("files", help="Folders the assistant may search and read (text, code, images, audio, video)")
    p.add_argument("action", choices=["status", "add", "remove", "index", "search"], nargs="?", default="status")
    p.add_argument("target", nargs="?", help="folder to add or remove, or a search query")
    p.set_defaults(func=cmd_files)

    p = sub.add_parser("token", help="Print the sandbox API token (or --admin)")
    p.add_argument("--admin", action="store_true")
    p.set_defaults(func=cmd_token)
    sub.add_parser("mcp", help="MCP stdio bridge to the assistant's guarded tools").set_defaults(func=cmd_mcp)
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
