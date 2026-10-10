"""Tools the assistant can call, each wired through policy.

Tool handlers receive validated JSON arguments and a RunContext. Output from
third parties (web pages, mail, notes, other people's messages) is marked
untrusted: it is wrapped, screened, and taints the run so later sensitive
actions need fresh approval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..policy import Action

if TYPE_CHECKING:
    from .loop import RunContext

Handler = Callable[[dict, "RunContext"], Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    untrusted: bool = False        # output carries third-party content
    sandbox_output: bool = False   # output comes from code running in the sandbox

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _params(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}


class ToolRegistry:
    def __init__(self, svc, tools: list[Tool]):
        self.svc = svc
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters,
                 "untrusted": t.untrusted} for t in self.tools.values()]

    async def invoke(self, name: str, raw_arguments: Any, ctx: "RunContext") -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}"
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                return f"Error: arguments were not valid JSON ({exc}). Call {name} again with valid JSON."
        else:
            arguments = raw_arguments or {}
        if not isinstance(arguments, dict):
            return "Error: arguments must be a JSON object."
        missing = [key for key in tool.parameters.get("required", []) if key not in arguments]
        if missing:
            return f"Error: missing required argument(s): {', '.join(missing)}."
        unknown = set(arguments) - set(tool.parameters.get("properties", {}))
        for key in unknown:
            arguments.pop(key)
        self.svc.audit.write("tool.call", tool=name, origin=ctx.origin, arguments=arguments, tainted=ctx.tainted)
        try:
            result = await tool.handler(arguments, ctx)
        except Exception as exc:
            self.svc.audit.write("tool.error", tool=name, error=str(exc))
            return f"Error: {name} failed: {exc}"
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        limit = int(self.svc.config.get("agent.tool_output_chars", 12000))
        if len(text) > limit:
            text = text[:limit] + f"\n…[truncated {len(text) - limit} characters]"
        if tool.untrusted or tool.sandbox_output:
            ctx.taint(content=tool.untrusted)
            text, score = await self.svc.guard.wrap(name, text)
            if score >= self.svc.guard.threshold:
                self.svc.audit.write("guard.flagged", tool=name, score=score, origin=ctx.origin)
        return text


# ---- sandbox -------------------------------------------------------------

def sandbox_tools(svc) -> list[Tool]:
    async def run_python(args, ctx):
        session = args.get("session") or "main"
        if not await svc.policy.check(Action("sandbox.exec", session, f"Run in the sandbox:\n{args['code']}",
                                             tainted=ctx.content_tainted), origin=ctx.origin):
            return "Error: the owner did not approve running this code after untrusted content was read."
        if ctx.content_tainted and svc.egress is not None:
            # Code written after reading untrusted content must not reuse standing egress grants.
            svc.egress.taint()
        await svc.sandbox.start_session(session)
        result = await svc.sandbox.execute(args["code"], session_id=session)
        return (result.get("stdout", "") + (("\n" + result["stderr"]) if result.get("stderr") else "")).strip() \
            or "(no output)"

    async def run_shell(args, ctx):
        code = ("import subprocess\n"
                f"r = subprocess.run({args['command']!r}, shell=True, capture_output=True, text=True, timeout=600)\n"
                "print(r.stdout[-20000:]); print(r.stderr[-5000:]); print('exit code', r.returncode)")
        return await run_python({"code": code, "session": "shell"}, ctx)

    return [
        Tool("run_python", "Run Python in the isolated sandbox VM. State persists per session. Use this for any "
             "calculation, data processing, file work or package installs. Print what you need to see.",
             _params({"code": STR, "session": STR}, ["code"]), run_python, sandbox_output=True),
        Tool("run_shell", "Run a shell command inside the isolated sandbox VM (not on the owner's computer).",
             _params({"command": STR}, ["command"]), run_shell, sandbox_output=True),
    ]


# ---- browser -------------------------------------------------------------

def browser_tools(svc) -> list[Tool]:
    b = svc.browser

    def _result(value: Any) -> Any:
        if isinstance(value, dict) and value.get("ok") is False:
            return f"Error: {value.get('error', 'browser action failed')}"
        return value

    async def open_(args, ctx):
        return _result(await b.navigate(args["url"], tainted=ctx.tainted))

    async def snapshot(args, ctx):
        result = _result(await b.snapshot(tainted=ctx.tainted))
        if isinstance(result, dict) and result.get("changed") is False:
            result["note"] = "Page unchanged since the last snapshot."
        return result

    async def click(args, ctx):
        return _result(await b.click(int(args["ref"]), tainted=ctx.tainted))

    async def type_(args, ctx):
        return _result(await b.type(int(args["ref"]), args["text"], submit=bool(args.get("submit")),
                                    tainted=ctx.tainted))

    async def press(args, ctx):
        return _result(await b.press(args["key"], tainted=ctx.tainted))

    async def scroll(args, ctx):
        return _result(await b.scroll(args.get("direction", "down"), tainted=ctx.tainted))

    async def back(args, ctx):
        return _result(await b.back(tainted=ctx.tainted))

    async def tabs(args, ctx):
        return _result(await b.tabs(tainted=ctx.tainted))

    async def screenshot(args, ctx):
        result = _result(await b.screenshot(tainted=ctx.tainted))
        if isinstance(result, str):
            return result
        if not result.get("changed", True):
            return "Screen unchanged since the last screenshot."
        if not svc.vision:
            return "The current model cannot see images; use browser_snapshot instead."
        ctx.images.append((result["mime"], result["data_b64"]))
        return "Screenshot attached to the next message."

    async def handoff(args, ctx):
        result = _result(await b.handoff(args["reason"], tainted=ctx.tainted))
        if isinstance(result, dict):
            return "The owner finished the step." if result.get("completed") else \
                "The owner did not complete the step. Do not retry; explain what is needed."
        return result

    tools = [
        Tool("browser_open", "Open a URL in the owner's CodeRunner browser window (a dedicated profile the owner "
             "can watch). New sites need the owner's approval.", _params({"url": STR}, ["url"]), open_, True),
        Tool("browser_snapshot", "Read the current page as text with numbered interactive elements [N].",
             _params({}), snapshot, True),
        Tool("browser_click", "Click element [ref] from the latest snapshot.", _params({"ref": INT}, ["ref"]), click,
             True),
        Tool("browser_type", "Type text into element [ref]. Never use for passwords or one-time codes; use "
             "browser_handoff instead.", _params({"ref": INT, "text": STR, "submit": BOOL}, ["ref", "text"]), type_, True),
        Tool("browser_press", "Press a key: Enter, Tab, Escape, ArrowDown, ArrowUp, Backspace, PageDown, PageUp.",
             _params({"key": STR}, ["key"]), press, True),
        Tool("browser_scroll", "Scroll the page up or down.", _params({"direction": {"type": "string", "enum": [
            "up", "down"]}}), scroll, True),
        Tool("browser_back", "Go back to the previous page.", _params({}), back, True),
        Tool("browser_tabs", "List open tabs.", _params({}), tabs, True),
        Tool("browser_handoff", "Ask the owner to complete a step themselves in the browser window (log in, "
             "password, 2FA, CAPTCHA, payment details). Waits until they finish.",
             _params({"reason": STR}, ["reason"]), handoff),
    ]
    if svc.vision:
        tools.append(Tool("browser_screenshot", "Take a screenshot of the current page to look at it.",
                          _params({}), screenshot, True))
    return tools


# ---- memory and schedules ------------------------------------------------

def memory_tools(svc) -> list[Tool]:
    async def save(args, ctx):
        if not await svc.policy.check(Action("memory.write", "memory", f"Remember: {args['text']}",
                                             tainted=ctx.tainted), origin=ctx.origin):
            return "Error: the owner did not approve saving this memory."
        return await svc.memory.save(args["text"], args.get("tags", ""))

    async def search(args, ctx):
        return await svc.memory.search(args["query"], int(args.get("limit", 5)))

    async def forget(args, ctx):
        what = f"id {args['id']}" if args.get("id") is not None else f"matching '{args.get('query', '')}'"
        if not await svc.policy.check(Action("memory.forget", "memory", f"Forget memories {what}",
                                             tainted=ctx.tainted), origin=ctx.origin):
            return "Error: the owner did not approve forgetting these memories."
        count = svc.memory.forget(id=args.get("id"), query=args.get("query"))
        return f"Forgot {count} memories."

    return [
        Tool("memory_save", "Remember a durable fact or preference the owner told you. Never store secrets.",
             _params({"text": STR, "tags": STR}, ["text"]), save),
        Tool("memory_search", "Search what you remember about the owner.", _params({"query": STR, "limit": INT},
                                                                                  ["query"]), search),
        Tool("memory_forget", "Forget memories by id or matching a query, when the owner asks.",
             _params({"id": INT, "query": STR}), forget),
    ]


def schedule_tools(svc) -> list[Tool]:
    async def create(args, ctx):
        summary = f"Schedule '{args['name']}' ({args['spec']}) to run:\n{args['prompt']}"
        if not await svc.policy.check(Action("schedule.create", args["spec"], summary, dict(args),
                                             tainted=ctx.tainted), origin=ctx.origin):
            return "Error: the owner did not approve this schedule."
        created = svc.scheduler.add(args["name"], args["spec"], args["prompt"], ctx.channel, ctx.chat_id)
        if ctx.tainted:
            # A prompt written after reading untrusted content keeps running as untrusted.
            svc.store.set_kv(f"schedule:tainted:{created['id']}", True)
        return created

    async def list_(args, ctx):
        return svc.scheduler.list()

    async def delete(args, ctx):
        return {"removed": svc.scheduler.remove(int(args["id"]))}

    return [
        Tool("schedule_create", "Run a task later or repeatedly. spec examples: 'every 2h', 'daily 08:00', "
             "'weekdays 09:30', 'weekly mon 09:00', 'at 2026-10-12 18:00', 'in 30m'. The result is sent to this "
             "conversation.", _params({"name": STR, "spec": STR, "prompt": STR}, ["name", "spec", "prompt"]), create),
        Tool("schedule_list", "List scheduled tasks.", _params({}), list_),
        Tool("schedule_delete", "Delete a scheduled task by id.", _params({"id": INT}, ["id"]), delete),
    ]


# ---- connectors and channels --------------------------------------------

def connector_tools(svc) -> list[Tool]:
    tools = []
    for spec in svc.connectors:
        async def handler(args, ctx, spec=spec):
            kind = "connector.write" if spec.write else "connector.read"
            summary = spec.describe(args) if hasattr(spec, "describe") else f"{spec.name} {json.dumps(args)[:300]}"
            if not await svc.policy.check(Action(kind, spec.connector, summary, dict(args), tainted=ctx.tainted),
                                          origin=ctx.origin):
                return f"Error: {spec.connector} access was not allowed."
            return await spec.call(args)
        tools.append(Tool(spec.name, spec.description, spec.parameters, handler, untrusted=spec.untrusted))
    return tools


def channel_tools(svc) -> list[Tool]:
    async def send(args, ctx):
        channel = svc.channels.get(args["channel"])
        if channel is None:
            return f"Error: channel {args['channel']!r} is not configured."
        # The owner must see exactly what will be sent, never a truncated preview.
        summary = f"Send via {args['channel']} to {args['to']}:\n{args['text']}"
        if not await svc.policy.check(Action("channel.send", f"{args['channel']}:{args['to']}", summary,
                                             dict(args), tainted=ctx.tainted), origin=ctx.origin):
            return "Error: the owner did not approve sending this message."
        await channel.send(args["to"], args["text"])
        return "Sent."

    if not svc.channels:
        return []
    return [Tool("send_message", "Send a message to someone other than the owner. Always needs the owner's "
                 "approval.", _params({"channel": STR, "to": STR, "text": STR}, ["channel", "to", "text"]), send)]


def build_registry(svc) -> ToolRegistry:
    tools: list[Tool] = []
    if svc.sandbox is not None:
        tools += sandbox_tools(svc)
    if svc.browser is not None:
        tools += browser_tools(svc)
    if svc.memory is not None:
        tools += memory_tools(svc)
    if svc.scheduler is not None:
        tools += schedule_tools(svc)
    tools += connector_tools(svc)
    tools += channel_tools(svc)
    return ToolRegistry(svc, tools)
