"""The always-on host daemon: wires models, sandbox, browser, policy and channels."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import signal
import sys
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import uvicorn

from ..core import paths
from ..agent import Agent, RunContext, build_registry
from ..core.audit import AuditLog
from ..core.config import Config
from ..safety.egress import EgressGuard, EgressProxy
from .gateway import admin_app, sandbox_app
from ..safety.guard import Guard
from ..core.llm import LLM
from ..safety.policy import Approval, Policy
from ..sandbox import SandboxClient, driver_for
from ..sandbox.owner_terminal import OwnerTerminal
from ..safety.secrets_store import SecretStore
from ..core.store import Store

log = logging.getLogger("coderunner")


@dataclass
class Services:
    config: Config
    store: Store
    audit: AuditLog
    policy: Policy
    secrets: SecretStore
    api_token: Callable[[], str]
    admin_token: Callable[[], str]
    llm: LLM | None = None
    llm_api_key: str = ""
    embedding_model: str = ""
    embedding_url: str = ""          # where embeddings are served, when not by the chat model's runtime
    vision: bool = False
    remote: dict | None = None      # remote mode, when on: {"provider", "model", "name"} (models/remote.py)
    guard: Guard = field(default_factory=Guard)
    egress: EgressGuard | None = None
    sandbox: SandboxClient | None = None
    computer: Any = None            # sandbox.keeper.Keeper: keeps the VM and its desktop up by itself
    terminal: Any = field(default_factory=OwnerTerminal)     # the owner typing in the VM's terminal
    browser: Any = None
    chrome: Any = None              # browser.extension.ChromeBridge: the owner's browser via the extension
    app_browser: Any = None         # browser.app.AppBridge: the CodeRunner app's own browser, on the Mac
    memory: Any = None
    scheduler: Any = None
    voice: Any = None
    files: Any = None
    connectors: list = field(default_factory=list)
    channels: dict = field(default_factory=dict)
    registry: Any = None
    turns: Any = None               # server.tasks.Turns: one model turn at a time between running tasks
    agent: Agent | None = None
    driver: Any = None              # agent.driver.Driver: a screenshot-driven model for browser tasks


class _Server(uvicorn.Server):
    """The daemon owns signal handling for all of its servers."""

    def capture_signals(self):
        return contextlib.nullcontext()


def _bind_error(host: str, port: int) -> str | None:
    """Probe the address first: uvicorn logs an error and exits when it cannot bind."""
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as exc:
        return f"{host}:{port}: {exc.strerror or exc}"
    return None


async def notify_desktop(title: str, text: str) -> None:
    """macOS notification; the text is passed as an argument, never as script source."""
    if platform.system() != "Darwin":
        return
    script = ["-e", "on run argv", "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
              "-e", "end run"]
    with contextlib.suppress(OSError, asyncio.TimeoutError):
        process = await asyncio.create_subprocess_exec("osascript", *script, title, text[:200],
                                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.wait_for(process.wait(), 10)


RESTART_AFTER_S = 4         # after setup from the app: time for the app to read the outcome


class Daemon:
    def __init__(self, config: Config | None = None):
        paths.ensure_dirs()
        self.config = config or Config.load()
        self.started = time.time()
        self._locks: dict[str, asyncio.Lock] = {}
        self._servers: list[uvicorn.Server] = []
        self._tasks: list[asyncio.Task] = []
        self._routine_runs: dict[int, str] = {}     # routine id -> the task of its latest run in an app thread
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.restart_requested = False
        self.svc = self._build()
        from .setup import SetupJob
        self.setup = SetupJob(self.config, self.svc.audit, on_done=self._after_setup)
        from .check import CheckJob
        self.check = CheckJob(self)             # `coderunner check`: the core tasks, done once and checked

    def restart_soon(self) -> None:
        """Start again in place, shortly: a setting everything was built around has changed (remote mode)."""
        self._after_setup({})

    def _after_setup(self, result: dict) -> None:
        """Setup from the app chose a model. Everything here was built for the
        configuration before it, so the daemon starts again with the new one, a few seconds on, once the app
        has read the outcome. Called from the job's thread."""
        self.restart_requested = True
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._loop.call_later, RESTART_AFTER_S, self._stop.set)

    async def _warm(self) -> None:
        """Let the model read the instructions and tool list now, not when the first task arrives."""
        if not self.config.get("models.chat") or getattr(self.svc, "remote", None):   # search mode, or not ours to warm
            return
        try:
            await self.svc.agent.warm()
            self.svc.audit.write("model.warmed")
        except Exception as exc:  # the first task then reads them itself
            self.svc.audit.write("model.warm_failed", error=str(exc)[:200])

    async def _start_sandbox(self) -> None:
        """Bring up the sandbox VM and its desktop at start, in the background (sandbox/keeper.py): after a restart
        macOS brings back the daemon (its login service) but not the VM, and the owner should see the computer's
        screen as soon as the app opens, without a button to press. A sandbox never installed is left alone."""
        if await self.computer.ensure(desktop=False, timeout=600):
            await self._vm_timezone()

    def _disk_status(self) -> dict:
        keeper = self.svc.computer
        try:
            used, budget = keeper.disk_gb(), keeper.budget_gb()
        except Exception:
            return {"disk_gb": None, "disk_budget_gb": None}
        number = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None  # noqa: E731
        return {"disk_gb": number(used), "disk_budget_gb": number(budget)}

    def _tasks_busy(self) -> bool:
        """A task running or waiting for a place, or a check under way: the VM is not to be made again now."""
        tasks = getattr(self, "tasks", None)
        if tasks is None:
            return True                     # not ready to tell yet
        check = getattr(self, "check", None)
        return bool(tasks.running or tasks.waiting or getattr(check, "state", "") == "running")

    def _owner_away(self) -> bool:
        """The owner is not using the computer: not typing in its terminal, not in control of its screen, and
        away from the Mac itself (no keyboard, mouse or trackpad for a while). Unknown counts as present."""
        from ..sandbox.keeper import Keeper, mac_idle_seconds
        svc = self.svc
        if svc.terminal is not None and svc.terminal.typing():
            return False
        if getattr(svc.browser, "owner_control", False):
            return False
        idle = mac_idle_seconds()
        return idle is not None and idle >= Keeper.OWNER_AWAY_S

    def _computer_mode(self) -> None:
        """The keeper gave the sandbox its desktop and moved the assistant's browser there."""
        self.svc.browser.restart = self._restart_vm_browser

    async def _restart_vm_browser(self) -> None:
        """Stop Chrome on the sandbox's desktop; its relay starts a fresh one on the next connection. Used by
        the browser broker when Chrome no longer answers (after the VM ran out of memory, new tabs hung)."""
        argv = self.driver.exec_argv(["pkill", "-f", "google-chrome|/opt/google/chrome/chrome"], tty=False)
        if argv:
            await asyncio.to_thread(subprocess.run, argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
            await asyncio.sleep(1.5)
            self.svc.audit.write("browser.vm_chrome_restarted")

    async def _vm_timezone(self) -> None:
        """The sandbox's clock in the owner's time zone: they watch that screen, and it showed UTC."""
        import re
        zone = self.config.get("owner.timezone") or ""
        if not zone:
            with contextlib.suppress(OSError):
                zone = os.path.realpath("/etc/localtime").split("zoneinfo/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z_]+(/[A-Za-z0-9_+-]+){0,2}", zone or ""):
            return
        with contextlib.suppress(Exception):            # a nicety: never in the way of starting
            argv = self.driver.exec_argv(["sh", "-c", f"[ -e /usr/share/zoneinfo/{zone} ] && ln -sf /usr/share/zoneinfo/{zone} "
                                                      f"/etc/localtime && echo {zone} > /etc/timezone"], tty=False)
            if argv:
                await asyncio.to_thread(subprocess.run, argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=15)

    async def _restart_embedding(self) -> bool:
        """Start the embedding server again when it has gone (one restart at a time)."""
        lock = self._locks.setdefault("embedding", asyncio.Lock())
        async with lock:
            from ..models import manager
            restarted = await asyncio.to_thread(manager.restart_embedding, self.config)
            self.svc.audit.write("embedding.restart", ok=restarted)
            return restarted

    async def _restart_model(self) -> bool:
        """Restart a crashed or stuck local model runtime (one restart at a time)."""
        lock = self._locks.setdefault("model", asyncio.Lock())
        async with lock:
            from ..models import manager
            self.svc.audit.write("model.restart")
            try:
                return await asyncio.to_thread(manager.restart, self.config)
            except Exception as exc:  # the request then fails with the runtime's own error
                self.svc.audit.write("model.restart_failed", error=str(exc)[:300])
                return False

    # ---- construction ----------------------------------------------------

    def _build(self) -> Services:
        store = Store()
        audit = AuditLog()
        svc = Services(config=self.config, store=store, audit=audit,
                       policy=Policy(self.config, store, audit), secrets=SecretStore(),
                       api_token=paths.api_token, admin_token=paths.admin_token)
        svc.egress = EgressGuard(self.config, svc.policy, audit)

        from ..models import manager, remote
        endpoint = manager.endpoint(self.config)
        svc.embedding_model = endpoint.embedding_model or ""
        from ..safety.decision import ActionCheck, DecisionModel
        runtime = {"base_url": endpoint.base_url, "api_key": endpoint.api_key}
        svc.remote = remote.active(self.config)
        claude = remote.client(self.config) if svc.remote else None
        if claude is not None:
            # Remote mode: Claude, or the owner's own model server, plans, sees, judges browser steps and screens
            # untrusted text. File search and voice keep their local models; the local chat model is left set
            # up, unloaded.
            svc.llm, svc.llm_api_key, svc.vision = claude, "", True
            if svc.remote["provider"] == remote.SERVER:
                judge = DecisionModel(claude.base_url, claude.model, api="chat", api_key=remote.server_key())
                if claude.slots > 1:
                    judge.slot = getattr(claude, "slot_base", 0) + claude.slots - 1   # its own, after the tasks'
                    # (tasks.limit), within this install's share of a shared server
                    judge.planner = claude              # which keeps the tasks' slots through each judgement

                def renamed(name: str) -> None:
                    """The server renamed its model (an upgrade): the step judge and the settings follow."""
                    judge.model = name
                    svc.remote.update(model=name, name=name)
                    self.config.set("models.remote", {**(self.config.get("models.remote") or {}), "model": name})
                    audit.write("remote.renamed", model=name)
                claude.renamed = renamed
            else:
                judge = DecisionModel(claude.base_url, claude.model, api="chat", claude=claude)
            # The injection guard stays off unless asked for (models.remote.guard): a general model asked the
            # guard's question took the page snapshot's own reading hints for instructions and flagged 7 ordinary
            # pages in about 80. Wrapping, taint, approvals and the leak rule are what stop a steered task.
            svc.guard = Guard(judge if (self.config.get("models.remote") or {}).get("guard") else None, audit=audit)
        else:
            svc.llm = LLM(endpoint.base_url, endpoint.model, endpoint.api_key)
            svc.llm.recover = self._restart_model
            svc.llm_api_key = endpoint.api_key
            svc.vision = endpoint.vision
            judge = None
            # Injection scoring needs a fast dedicated model; the assistant's own model would slow every step.
            svc.guard = Guard(DecisionModel.from_config(self.config, model=self.config.get("models.guard") or "",
                                                        dedicated_only=True, **runtime), audit=audit)

        driver = driver_for(self.config)
        self.driver = driver
        url = self.config.get("sandbox.url") or driver.default_url()
        svc.sandbox = SandboxClient(url, paths.api_token())

        from ..browser import BrowserBroker
        if os.name == "posix":
            from ..browser.extension import ChromeBridge, refresh
            svc.chrome = ChromeBridge(audit=audit)
            try:
                if refresh():
                    audit.write("browser.extension_updated")
            except OSError as exc:
                audit.write("browser.extension_update_failed", error=str(exc)[:200])
        decider = ActionCheck(judge, audit=audit) if judge is not None else \
            ActionCheck.from_config(self.config, audit, **runtime)
        if decider is not None and claude is None and decider.model.base_url == svc.llm.base_url.rstrip("/"):
            from .tasks import limit
            decider.model.slot = limit(svc)     # the slot after the tasks' (runtimes.py starts one more)
        svc.browser = BrowserBroker(self.config, svc.policy, audit, decider=decider, chrome=svc.chrome,
                                    sandbox=(url, paths.api_token()))

        if self.config.get("browser.mode") == "computer":
            svc.browser.restart = self._restart_vm_browser      # what the broker falls back to when Chrome is stuck
        from ..sandbox.keeper import Keeper
        from ..sandbox.reset import recreate_quietly
        svc.computer = self.computer = Keeper(self.config, driver, svc.sandbox, audit, on_mode=self._computer_mode,
                                              busy=self._tasks_busy, owner_away=self._owner_away,
                                              recreate=recreate_quietly)
        svc.browser.computer = svc.computer                     # a task that needs the screen asks for it
        if claude is not None:
            # A model with room to read gets more of each page at once, and links whole enough to cite: a local
            # 9B model is given 12,000 characters and addresses cut to 60.
            from ..browser import page
            svc.browser.page_chars, page.LINK_CHARS = claude.page_chars, claude.link_chars
        if self.config.get("browser.mode") == "computer" and sys.platform == "darwin" \
                and self.config.get("browser.mac_switch", True) is not False:
            # A site that turns the sandbox's Chrome away may go on in the browser on the Mac, with the owner's OK.
            from ..browser.app import AppBridge, MacLauncher
            from ..browser.switch import BrowserSwitch
            svc.app_browser = AppBridge(audit=audit)
            mac_launcher = MacLauncher(svc.app_browser)
            mac = BrowserBroker(self.config, svc.policy, audit, launcher=mac_launcher, decider=decider, mode="mac")
            mac.page_chars = svc.browser.page_chars
            svc.browser = BrowserSwitch(svc.browser, mac, svc.policy, audit, launcher=mac_launcher,
                                        bridge=svc.app_browser, config=self.config)
        from ..agent.memory import Memory
        # Embeddings can come from a different runtime than the chat model, e.g. EmbeddingGemma in
        # Ollama beside a chat model served by llama.cpp.
        embedding_url = self.config.get("models.embedding_url") or endpoint.base_url
        embed_llm = svc.llm if embedding_url == endpoint.base_url else LLM(embedding_url, svc.embedding_model)
        if embed_llm is not svc.llm:
            embed_llm.recover = self._restart_embedding
        svc.embedding_url = "" if embedding_url == endpoint.base_url else embedding_url.rstrip("/")
        embedder = None
        if svc.embedding_model:
            async def embedder(texts: list[str]) -> list[list[float]]:
                return await embed_llm.embed(texts, model=svc.embedding_model)
        from ..files import embedding_profile
        profile = embedding_profile(svc.embedding_model)
        svc.memory = Memory(store, audit, embedder, query_prefix=profile["query"], doc_prefix=profile["doc"])

        from ..files import Embedder, FileIndex
        from ..models.runtimes import idle_seconds
        # Multimodal embeddings (text, images, audio) come from Ollama's native embed API.
        file_model = self.config.get("files.embedding") or svc.embedding_model
        ollama = self.config.get("models.runtime") in ("ollama", "auto") or bool(self.config.get("models.embedding_url"))
        # Ollama frees the model after the idle time like every runtime; a negative time (idle unload off) keeps it.
        file_embedder = Embedder(embedding_url, file_model, keep_alive=f"{idle_seconds(self.config) or -60}s") \
            if file_model and ollama else None
        if file_embedder is not None and embed_llm is not svc.llm:
            file_embedder.recover = self._restart_embedding
        svc.files = FileIndex(self.config, store, audit, file_embedder)

        from ..agent.scheduler import Scheduler
        svc.scheduler = Scheduler(store, audit, self._run_schedule, self._deliver_schedule,
                                  tz=self.config.get("owner.timezone") or None)

        from ..voice import Voice
        svc.voice = Voice(self.config)      # usable once `coderunner voice setup` has run, no restart needed
        svc.files.transcriber = svc.voice.timed                         # used with files.transcribe on

        from ..connectors import load_connectors
        svc.connectors = load_connectors(self.config)

        from ..channels import load_channels
        svc.channels = load_channels(self.config, svc, self._on_channel_message)

        svc.driver = self._driver(svc)
        svc.registry = build_registry(svc)
        from .tasks import Turns
        svc.turns = Turns()
        svc.agent = Agent(svc, svc.llm, svc.registry)
        svc.policy.add_listener(self._announce_approval)
        return svc

    def _driver(self, svc: Services):
        """The browser driver (models.driver), when one is set up: a model that acts from screenshots."""
        settings = self.config.get("models.driver") or {}
        from ..models import remote
        served = getattr(svc, "remote", None)
        on_server = bool(served) and served["provider"] == remote.SERVER and bool(settings.get("model"))
        # The step judge on the driver's model here, without the driver working the pages (off by default:
        # models.remote.local_judge), so no judgement goes to the server between a task's steps.
        local_judge = on_server and not settings.get("with_server") and \
            bool((self.config.get("models.remote") or {}).get("local_judge"))
        # Claude drives the browser itself. With the owner's model server, the local driver can work the pages
        # (models.driver.with_server) while the server plans: nothing on this Mac has to make room for it.
        if not local_judge and (not settings.get("model") or served and (served["provider"] != remote.SERVER
                                                                          or not settings.get("with_server"))):
            return None
        from ..agent.driver import Driver, Swap
        from ..models.runtimes import DriverRuntime, choose_runtime
        runtime = DriverRuntime(self.config)
        alias = (runtime._state().get("active") or {}).get("alias") or settings["model"]
        swap = Swap(svc, None, runtime) if served else \
            Swap(svc, choose_runtime(self.config), runtime) if settings.get("exclusive") else None
        decider = getattr(getattr(svc, "browser", None), "decider", None)
        if served and decider is not None:
            # The step judge asks the driver too: a judge call on the server's other slot made the server drop
            # the task's slot, read again at the next step (27 of the 37 long re-reads in a 30-minute trip plan).
            from ..safety.decision import DecisionModel
            local = DecisionModel(runtime.base_url, alias, api="chat")
            local.prepare = lambda: asyncio.to_thread(runtime.start)
            local.fallback = decider.model
            decider.model = local
        if local_judge:
            return None
        from ..models import catalog
        try:
            fmt = settings.get("format") or catalog.get(settings["model"])["driver"]["format"]
        except (KeyError, TypeError):
            fmt = "fara1.5"
        return Driver(svc, LLM(runtime.base_url, alias), steps=int(settings.get("steps") or 30), swap=swap,
                      runtime=runtime, format=fmt)

    # ---- message handling --------------------------------------------------

    async def handle_message(self, channel: str, chat_id: str, text: str, *, origin: str = "",
                             sender_is_owner: bool = True, images: list | None = None,
                             tainted: bool = False, events=None, task: str = "", remote: bool = False) -> str:
        """remote: this one message on Claude (the owner's retry of a task the local model did not finish)."""
        conversation = f"{channel}:{chat_id}"
        if not remote and not self.config.get("models.chat") and not getattr(getattr(self, "svc", None), "remote", None):
            return ("No local chat model is set up (search mode). Use the search tools from your own agent over "
                    "MCP (`coderunner mcp --tools search`), or run `coderunner setup` to add a model.")
        agent = await self.claude_agent() if remote else self.svc.agent
        lock = self._locks.setdefault(conversation, asyncio.Lock())
        async with lock:
            ctx = RunContext(conversation=conversation, origin=origin or conversation, channel=channel,
                             chat_id=chat_id, origin_is_owner=sender_is_owner, events=events,
                             deadline=getattr(events, "deadline", 0.0) or 0.0,    # a task with a time limit
                             stop_at=getattr(events, "stop_at", 0.0) or 0.0)
            inbox = getattr(events, "inbox", None)
            if isinstance(inbox, list):
                ctx.inbox = inbox           # the owner's messages sent while it works (server.tasks)
            if tainted:
                ctx.taint()
            if remote:
                self.svc.audit.write("task.remote", conversation=conversation, model=agent.llm.model)
            return await agent.run(ctx, text, images=images)

    async def claude_agent(self):
        """An agent on Anthropic's API for one task the owner sent there, with the kept key, beside the local model
        (remote mode itself stays off: the next task is local again). It plans, sees and works the web pages
        itself, so the local screenshot driver is left out; the step judge, approvals, wrapping and the leak rule
        are the assistant's own, unchanged. Built once and kept, like the local agent."""
        from ..models import remote
        if (getattr(self.svc, "remote", None) or {}).get("provider") == remote.PROVIDER:
            return self.svc.agent                       # remote mode is on: every task is on Claude already
        if not remote.key():
            raise ValueError("No Anthropic key is kept.")
        chosen = remote.claude_model(self.config)
        kept = getattr(self, "_claude", None)
        if kept is None or kept.llm.model != chosen:
            await asyncio.to_thread(remote.install_sdk, lambda line: None)
            from ..core.claude import Claude
            from ..agent.loop import Agent
            from ..agent.tools import ToolRegistry
            tools = [tool for name, tool in self.svc.registry.tools.items() if name != "browser_task"]
            self._claude = kept = Agent(self.svc, Claude(remote.key(), chosen), ToolRegistry(self.svc, tools))
        return kept

    async def _on_channel_message(self, message) -> str | None:
        text = message.text
        if message.audio_path:
            if self.svc.voice is None or not self.svc.voice.available():
                return "Voice messages aren't set up on this computer yet."
            text = (await self.svc.voice.transcribe(message.audio_path)).strip()
            if not text:
                return "I couldn't make out that voice message."
        return await self.handle_message(message.channel, message.chat_id, text,
                                         origin=f"{message.channel}:{message.sender}",
                                         sender_is_owner=message.sender_is_owner, images=message.images)

    async def _announce_approval(self, approval: Approval) -> None:
        if self.config.get("owner.desktop_notifications", True):
            await notify_desktop("CodeRunner needs approval", approval.summary)
        for channel in self.svc.channels.values():
            try:
                await channel.notify_approval(approval)
            except Exception as exc:
                self.svc.audit.write("channel.error", channel=channel.name, error=str(exc))

    async def _run_schedule(self, schedule: dict) -> str:
        channel = schedule.get("channel") or "schedule"
        chat_id = schedule.get("chat_id") or str(schedule["id"])
        tainted = bool(self.svc.store.get_kv(f"schedule:tainted:{schedule['id']}"))
        if channel == "app" and schedule.get("chat_id") and getattr(self, "tasks", None) is not None:
            return await self._run_routine_in_app(schedule, tainted)
        return await self.handle_message(channel, chat_id, schedule["prompt"], origin=f"schedule:{schedule['id']}",
                                         tainted=tainted)

    async def _run_routine_in_app(self, schedule: dict, tainted: bool) -> str:
        """A routine made in an app conversation runs as a task in that conversation: the thread shows the run
        and its answer like any task's, and the sidebar lists the conversation as working. The scheduler waits
        for it to end; its time limit stops the waiting, not the task, which the owner can stop in the thread."""
        runs = self._routine_runs
        earlier = self.tasks.get(runs.get(schedule["id"], "")) or {}
        if earlier.get("status") in ("running", "queued"):
            self.svc.audit.write("schedule.skipped", id=schedule["id"], reason="the last run is still going")
            return ""
        task = self.tasks.start(f"app:{schedule['chat_id']}", schedule["prompt"], f"schedule:{schedule['id']}",
                                tainted=tainted, routine={"id": schedule["id"], "name": schedule.get("name", ""),
                                                          "spec": schedule.get("spec", "")})
        runs[schedule["id"]] = task["id"]
        ended = await asyncio.shield(self.tasks.wait(task["id"])) or {}
        if ended.get("status") == "failed":
            raise RuntimeError("the run failed; its thread says how")
        # The answer is in the thread already; it is passed on only for the desktop notification.
        return (ended.get("reply") or "") if ended.get("status") == "done" else ""

    async def _deliver_schedule(self, schedule: dict, text: str) -> None:
        channel = self.svc.channels.get(schedule.get("channel") or "")
        if channel is not None and schedule.get("chat_id"):
            await channel.send(schedule["chat_id"], text)
        elif self.config.get("owner.desktop_notifications", True):
            await notify_desktop(f"CodeRunner: {schedule.get('name', 'task')}", text)

    # ---- status ------------------------------------------------------------

    async def status(self) -> dict:
        svc = self.svc
        return {
            "uptime": round(time.time() - self.started),
            "model": {"base_url": svc.llm.base_url, "model": svc.llm.model, "vision": svc.vision,
                      "embedding": svc.embedding_model},
            # Remote mode: Claude does the planning, the web pages and the step judging, and the app says so.
            "remote": svc.remote and {**svc.remote, "usage": dict(getattr(svc.llm, "usage", {}) or {}),
                                      "step_s": getattr(svc.llm, "step_seconds", lambda: None)()},
            "models": {"chat": svc.remote["model"] if svc.remote else self.config.get("models.chat") or svc.llm.model,
                       "driver": svc.remote["model"] if svc.remote and svc.driver is None else
                       (self.config.get("models.driver") or {}).get("model") or "",
                       "decision": svc.remote["model"] if svc.remote else self.config.get("models.decision") or "",
                       "embedding": self.config.get("models.embedding") or svc.embedding_model,
                       "local_chat": self.config.get("models.chat") or ""},
            "sandbox": {"driver": self.driver.name, "url": svc.sandbox.base_url, "healthy": await svc.sandbox.health(),
                        # what its disk takes on the Mac (None where not measured), and the budget it is kept to
                        **self._disk_status()},
            "network": self.config.get("sandbox.network"),
            "channels": sorted(svc.channels),
            "connectors": sorted({c.connector for c in svc.connectors}),
            "voice": [engine.name for engine in svc.voice.engines()] if svc.voice is not None else [],
            "pending_approvals": len(svc.policy.pending()),
            "tools": sorted(svc.registry.tools),
            "browser": {"mode": svc.browser.mode, "running": svc.browser.running(),
                        **({"extension": svc.chrome.status()} if svc.chrome is not None else {}),
                        **({"app": svc.app_browser.status()} if svc.app_browser is not None else {})},
        }

    # ---- serving -----------------------------------------------------------

    def _sandbox_bind(self) -> str:
        bind = self.config.get("gateway.sandbox_bind", "auto")
        return self.driver.gateway_address() if bind == "auto" else bind

    async def _serve_with_retry(self, app, host_fn: Callable[[], str], port: int, label: str) -> None:
        """The VM bridge address only exists while the sandbox network is up, so keep trying.

        Only the first failure in a row is audited, so a stopped sandbox does not flood the log.
        """
        failing = False
        while True:
            host = host_fn()
            error = _bind_error(host, port)
            if error is None:
                server = _Server(uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on"))
                self._servers.append(server)
                try:
                    await server.serve()
                except (OSError, SystemExit):
                    error = f"{host}:{port}: could not listen"
                if server.should_exit:
                    return
                self._servers.remove(server)
            if error and not failing:
                self.svc.audit.write("gateway.bind_failed", surface=label, host=host, port=port, error=error)
            failing = error is not None
            await asyncio.sleep(10)

    async def _proxy_with_retry(self, servers: list) -> None:
        """Like the HTTP surfaces, the egress proxy waits for the VM bridge address to exist."""
        proxy = EgressProxy(self.svc.egress, self.svc.audit, paths.api_token)
        failing = False
        while True:
            try:
                servers.append(await proxy.start(self._sandbox_bind(), int(self.config.get("gateway.proxy_port"))))
                return
            except OSError as exc:
                if not failing:
                    self.svc.audit.write("gateway.bind_failed", surface="proxy", error=str(exc))
                failing = True
                await asyncio.sleep(10)

    async def run(self) -> None:
        cfg, svc = self.config, self.svc
        loop = self._loop = asyncio.get_running_loop()
        stop = self._stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)

        from ..models import manager
        try:
            await asyncio.to_thread(manager.ensure_running, cfg)
        except Exception as exc:
            svc.audit.write("models.start_failed", error=str(exc))
        self._tasks.append(asyncio.create_task(self._warm()))
        self._tasks.append(asyncio.create_task(self._start_sandbox()))

        from .tasks import EventBus, Tasks
        self.bus = EventBus(svc.store)
        self.tasks = Tasks(self, self.bus)
        admin = admin_app(svc, self)
        self._tasks.append(asyncio.create_task(self._serve_with_retry(
            admin, lambda: cfg.get("gateway.admin_bind", "127.0.0.1"), int(cfg.get("gateway.admin_port")), "admin")))
        self._tasks.append(asyncio.create_task(self._serve_with_retry(
            sandbox_app(svc), self._sandbox_bind, int(cfg.get("gateway.sandbox_port")), "sandbox")))
        proxy_servers: list = []
        if cfg.get("sandbox.network") == "proxy":
            self._tasks.append(asyncio.create_task(self._proxy_with_retry(proxy_servers)))
        self._tasks.append(asyncio.create_task(svc.scheduler.run_forever()))
        self._tasks.append(asyncio.create_task(svc.files.run_forever()))
        self._tasks.append(asyncio.create_task(self.computer.watch_disk()))     # the VM's disk kept small
        if svc.chrome is not None:
            try:
                await svc.chrome.start()
            except OSError as exc:
                svc.audit.write("browser.extension_bridge_failed", error=str(exc))
        if svc.app_browser is not None:
            try:
                await svc.app_browser.start()
            except OSError as exc:
                svc.audit.write("browser.app_bridge_failed", error=str(exc))
        for channel in svc.channels.values():
            try:
                await channel.start()
            except Exception as exc:
                svc.audit.write("channel.error", channel=channel.name, error=str(exc))
        svc.audit.write("daemon.started", channels=sorted(svc.channels), network=cfg.get("sandbox.network"))
        log.info("CodeRunner daemon running; owner UI at http://%s:%s/", cfg.get("gateway.admin_bind"),
                 cfg.get("gateway.admin_port"))
        # Tasks a restart (an update, a crash, the Mac sleeping) left unfinished go on from their journals.
        try:
            self.tasks.resume()
        except Exception as exc:
            svc.audit.write("task.resume_failed", error=f"{type(exc).__name__}: {exc}"[:300])

        await stop.wait()
        svc.audit.write("daemon.stopping")
        # Running tasks stop where they are and keep their journals, before the browser and servers go under them.
        self.tasks.shutdown()
        if self.tasks.running:
            await asyncio.wait(list(self.tasks.running.values()), timeout=10)
        for server in self._servers:
            server.should_exit = True
        # Let the HTTP servers finish their own shutdown before cancelling the rest.
        await asyncio.wait(self._tasks[:2], timeout=10)
        for channel in svc.channels.values():
            with contextlib.suppress(Exception):
                await channel.stop()
        for proxy_server in proxy_servers:
            proxy_server.close()
        with contextlib.suppress(Exception):
            await svc.browser.stop()
        if svc.chrome is not None:
            with contextlib.suppress(Exception):
                await svc.chrome.stop()
        if svc.app_browser is not None:
            with contextlib.suppress(Exception):
                await svc.app_browser.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await svc.sandbox.aclose()
        await svc.llm.aclose()


    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()


def only_one():
    """The lock a home's one daemon holds while it runs, or None when another daemon has it. A second daemon on
    the same home was not refused: it loaded the model again and waited for the port, and took over with whatever
    code it had when the first one stopped. The lock goes with the process (and is taken again by a daemon that
    starts again in place)."""
    try:
        import fcntl
    except ImportError:             # not a POSIX system: no lock to take
        return True
    paths.ensure_dirs()
    handle = open(paths.assistant_dir() / "run" / "daemon.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.truncate(0)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lock = only_one()
    if lock is None:
        log.error("Another CodeRunner daemon is already running for %s; this one stops. `coderunner down` stops "
                  "the other.", paths.home())
        raise SystemExit(1)
    daemon = Daemon()
    asyncio.run(daemon.run())
    if daemon.restart_requested:
        # In place, so a login service or `coderunner up` keeps the process it started.
        log.info("starting again with the new configuration")
        os.execv(sys.executable, [sys.executable, "-m", "coderunner", "daemon"])


if __name__ == "__main__":
    main()
