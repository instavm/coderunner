"""Setup from the app: the plan for this machine and setup itself, and the sandbox's computer, each as a background
job whose progress the app shows. They run the same manager.plan, manager.setup and sandbox.computer.setup as
`coderunner setup` and `coderunner browser computer`, so the app and the terminal cannot differ."""

from __future__ import annotations

import threading
from typing import Callable

KEEP = 400          # progress lines kept; the app shows the latest
SHOWN = 80


class Job:
    """One run at a time in the background; what it says is kept for the app to show."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = "idle"             # idle | running | done | failed
        self.lines: list[str] = []
        self.result: dict | None = None

    def _begin(self, target, *args) -> bool:
        with self._lock:
            if self.state == "running":
                return False
            self.state, self.lines, self.result = "running", [], None
        threading.Thread(target=target, args=args, daemon=True, name=type(self).__name__).start()
        return True

    def _say(self, message: str) -> None:
        with self._lock:
            self.lines.append(str(message))
            del self.lines[:-KEEP]

    def _finish(self, result: dict) -> None:
        with self._lock:
            self.result = result
            self.state = "done" if result.get("ok") else "failed"

    def progress(self) -> dict:
        with self._lock:
            return {"state": self.state, "lines": self.lines[-SHOWN:], "line_count": len(self.lines),
                    "result": self.result}


class ComputerJob(Job):
    """`coderunner browser computer` from the app: the desktop and Chrome installed in the sandbox VM."""

    def __init__(self, config, audit=None, on_done: Callable[[dict], None] | None = None, run=None):
        super().__init__()
        self.config, self.audit, self.on_done, self._run_setup = config, audit, on_done, run

    def public(self) -> dict:
        return self.progress()

    def start(self) -> bool:
        return self._begin(self._run)

    def _run(self) -> None:
        from ..sandbox import computer
        try:
            ok = (self._run_setup or computer.setup)(self.config, say=self._say)
        except Exception as exc:
            self._say(f"Error: {type(exc).__name__}: {exc}")
            ok = False
        self._finish({"ok": bool(ok)})
        if self.audit is not None:
            self.audit.write("computer.setup_finished", ok=bool(ok))
        if ok and self.on_done is not None:
            self.on_done({"ok": True})


class SetupJob(Job):
    def __init__(self, config, audit=None, on_done: Callable[[dict], None] | None = None, plan=None, setup=None,
                 voice: bool = True):
        from ..models import manager
        super().__init__()
        self.config, self.audit, self.on_done, self.voice = config, audit, on_done, voice
        self._plan, self._setup = plan or manager.plan, setup or manager.setup
        self.planned: dict | None = None

    def public(self) -> dict:
        return {"configured": bool(self.config.get("models.chat")), "model": self.config.get("models.chat") or "",
                **self.progress(), "plan": self.planned}

    def plan(self) -> dict:
        """What setup would do here and what it would fetch. Slow the first time in a day (model discovery)."""
        from ..models import manager
        planned = self._plan(self.config)
        items = manager.downloads(self.config, planned)
        room, free, _ = manager._room_gb(self.config)
        model, hw = planned.get("model"), planned["hardware"]
        summary = {
            "hardware": {"chip": hw.get("chip") or "unknown chip", "ram_gb": round(hw.get("ram_gb") or 0),
                         "gpu": hw.get("gpu"), "measured": hw.get("measured") or ""},
            "model": model and {"id": model["id"], "runtime": model.get("runtime"), "ctx": model.get("ctx"),
                                "step_s": model.get("step_s"), "vision": model.get("vision"),
                                "need_gb": model.get("need_gb"), "tok_s": model.get("tok_s"),
                                "reasons": model.get("reasons", [])[:4]},
            "alternatives": [{"id": c["id"], "runtime": c.get("runtime"), "step_s": c.get("step_s")}
                             for c in planned.get("alternatives", [])[:4]],
            "notes": planned.get("notes", []), "downloads": items,
            "download_gb": round(sum(item["gb"] or 0 for item in items), 2),
            "free_gb": round(free, 1), "room_gb": round(max(room, 0), 1),
        }
        with self._lock:
            self.planned = summary
        return summary

    def start(self, model: str | None = None, accept_license: bool = False) -> bool:
        """Run setup in the background; False when one is already running."""
        return self._begin(self._run, model, accept_license)

    def _run(self, model: str | None, accept_license: bool) -> None:
        try:
            result = self._setup(self.config, model_id=model or None, accept_license=accept_license,
                                 progress=self._say)
        except Exception as exc:        # setup reports its failures; this is one it did not foresee
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if result.get("ok") and self.voice:
            self._say("setting up speech to text")
            try:
                import asyncio

                from ..voice import Voice
                asyncio.run(Voice(self.config).setup(progress=self._say))
            except Exception as exc:    # voice is optional, as in `coderunner setup`
                self._say(f"voice setup failed ({exc}); retry later with `coderunner voice setup`")
        keep = ("ok", "entry", "model", "runtime", "ctx", "vision", "error", "needs_license", "skipped_for_disk")
        self._finish({key: result.get(key) for key in keep if key in result})
        if self.audit is not None:
            self.audit.write("setup.finished", ok=bool(result.get("ok")), model=result.get("entry") or "",
                             error=str(result.get("error") or "")[:300])
        if result.get("ok") and self.on_done is not None:
            self.on_done(result)
