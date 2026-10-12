"""The always-on host daemon: wires models, sandbox, browser, policy and channels."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import uvicorn

from . import paths
from .agent import Agent, RunContext, build_registry
from .audit import AuditLog
from .config import Config
from .egress import EgressGuard, EgressProxy
from .gateway import admin_app, sandbox_app
from .guard import Guard
from .llm import LLM
from .policy import Approval, Policy
from .sandbox import SandboxClient, driver_for
from .secrets_store import SecretStore
from .store import Store

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
    vision: bool = False
    guard: Guard = field(default_factory=Guard)
    egress: EgressGuard | None = None
    sandbox: SandboxClient | None = None
    browser: Any = None
    chrome: Any = None              # browser.extension.ChromeBridge: the owner's browser via the extension
    memory: Any = None
    scheduler: Any = None
    voice: Any = None
    files: Any = None
    connectors: list = field(default_factory=list)
    channels: dict = field(default_factory=dict)
    registry: Any = None
    agent: Agent | None = None


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


class Daemon:
    def __init__(self, config: Config | None = None):
        paths.ensure_dirs()
        self.config = config or Config.load()
        self.started = time.time()
        self._locks: dict[str, asyncio.Lock] = {}
        self._servers: list[uvicorn.Server] = []
        self._tasks: list[asyncio.Task] = []
        self._stop: asyncio.Event | None = None
        self.svc = self._build()

    # ---- construction ----------------------------------------------------

    def _build(self) -> Services:
        store = Store()
        audit = AuditLog()
        svc = Services(config=self.config, store=store, audit=audit,
                       policy=Policy(self.config, store, audit), secrets=SecretStore(),
                       api_token=paths.api_token, admin_token=paths.admin_token)
        svc.egress = EgressGuard(self.config, svc.policy, audit)

        from .models import manager
        endpoint = manager.endpoint(self.config)
        svc.llm = LLM(endpoint.base_url, endpoint.model, endpoint.api_key)
        svc.llm_api_key = endpoint.api_key
        svc.vision = endpoint.vision
        svc.embedding_model = endpoint.embedding_model or ""
        from .decision import ActionCheck, DecisionModel
        runtime = {"base_url": endpoint.base_url, "api_key": endpoint.api_key}
        # Injection scoring needs a fast dedicated model; the assistant's own model would slow every step.
        svc.guard = Guard(DecisionModel.from_config(self.config, model=self.config.get("models.guard") or "",
                                                    dedicated_only=True, **runtime), audit=audit)

        driver = driver_for(self.config)
        self.driver = driver
        url = self.config.get("sandbox.url") or driver.default_url()
        svc.sandbox = SandboxClient(url, paths.api_token())

        from .browser import BrowserBroker
        if os.name == "posix":
            from .browser.extension import ChromeBridge
            svc.chrome = ChromeBridge(audit=audit)
        svc.browser = BrowserBroker(self.config, svc.policy, audit,
                                    decider=ActionCheck.from_config(self.config, audit, **runtime), chrome=svc.chrome)

        from .memory import Memory
        # Embeddings can come from a different runtime than the chat model, e.g. EmbeddingGemma in
        # Ollama beside a chat model served by llama.cpp.
        embedding_url = self.config.get("models.embedding_url") or endpoint.base_url
        embed_llm = svc.llm if embedding_url == endpoint.base_url else LLM(embedding_url, svc.embedding_model)
        embedder = None
        if svc.embedding_model:
            async def embedder(texts: list[str]) -> list[list[float]]:
                return await embed_llm.embed(texts, model=svc.embedding_model)
        from .files import embedding_profile
        profile = embedding_profile(svc.embedding_model)
        svc.memory = Memory(store, audit, embedder, query_prefix=profile["query"], doc_prefix=profile["doc"])

        from .files import Embedder, FileIndex
        # Multimodal embeddings (text, images, audio) come from Ollama's native embed API.
        file_model = self.config.get("files.embedding") or svc.embedding_model
        ollama = self.config.get("models.runtime") in ("ollama", "auto") or bool(self.config.get("models.embedding_url"))
        svc.files = FileIndex(self.config, store, audit,
                              Embedder(embedding_url, file_model) if file_model and ollama else None)

        from .scheduler import Scheduler
        svc.scheduler = Scheduler(store, audit, self._run_schedule, self._deliver_schedule,
                                  tz=self.config.get("owner.timezone") or None)

        from .voice import Voice
        voice = Voice(self.config)
        svc.voice = voice if voice.available() else None
        if svc.voice is not None:
            from .files.extract import untimed
            svc.files.transcriber = untimed(svc.voice.transcribe)       # used with files.transcribe on

        from .connectors import load_connectors
        svc.connectors = load_connectors(self.config)

        from .channels import load_channels
        svc.channels = load_channels(self.config, svc, self._on_channel_message)

        svc.registry = build_registry(svc)
        svc.agent = Agent(svc, svc.llm, svc.registry)
        svc.policy.add_listener(self._announce_approval)
        return svc

    # ---- message handling --------------------------------------------------

    async def handle_message(self, channel: str, chat_id: str, text: str, *, origin: str = "",
                             sender_is_owner: bool = True, images: list | None = None,
                             tainted: bool = False) -> str:
        conversation = f"{channel}:{chat_id}"
        lock = self._locks.setdefault(conversation, asyncio.Lock())
        async with lock:
            ctx = RunContext(conversation=conversation, origin=origin or conversation, channel=channel,
                             chat_id=chat_id, origin_is_owner=sender_is_owner)
            if tainted:
                ctx.taint()
            return await self.svc.agent.run(ctx, text, images=images)

    async def _on_channel_message(self, message) -> str | None:
        text = message.text
        if message.audio_path:
            if self.svc.voice is None:
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
        return await self.handle_message(channel, chat_id, schedule["prompt"], origin=f"schedule:{schedule['id']}",
                                         tainted=tainted)

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
            "sandbox": {"driver": self.driver.name, "url": svc.sandbox.base_url, "healthy": await svc.sandbox.health()},
            "network": self.config.get("sandbox.network"),
            "channels": sorted(svc.channels),
            "connectors": sorted({c.connector for c in svc.connectors}),
            "pending_approvals": len(svc.policy.pending()),
            "tools": sorted(svc.registry.tools),
            "browser": {"mode": svc.browser.mode, "running": svc.browser.running(),
                        **({"extension": svc.chrome.status()} if svc.chrome is not None else {})},
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
        loop = asyncio.get_running_loop()
        stop = self._stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)

        from .models import manager
        try:
            await asyncio.to_thread(manager.ensure_running, cfg)
        except Exception as exc:
            svc.audit.write("models.start_failed", error=str(exc))

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
        if svc.chrome is not None:
            try:
                await svc.chrome.start()
            except OSError as exc:
                svc.audit.write("browser.extension_bridge_failed", error=str(exc))
        for channel in svc.channels.values():
            try:
                await channel.start()
            except Exception as exc:
                svc.audit.write("channel.error", channel=channel.name, error=str(exc))
        svc.audit.write("daemon.started", channels=sorted(svc.channels), network=cfg.get("sandbox.network"))
        log.info("CodeRunner daemon running; owner UI at http://%s:%s/", cfg.get("gateway.admin_bind"),
                 cfg.get("gateway.admin_port"))

        await stop.wait()
        svc.audit.write("daemon.stopping")
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
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await svc.sandbox.aclose()
        await svc.llm.aclose()


    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(Daemon().run())


if __name__ == "__main__":
    main()
