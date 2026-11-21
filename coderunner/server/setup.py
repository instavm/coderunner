"""Setup from the app: the plan for this machine, and setup itself as a background job whose progress the app
shows. It is the same manager.plan and manager.setup that `coderunner setup` runs, so the two cannot differ."""

from __future__ import annotations

import threading
from typing import Callable

KEEP = 400          # progress lines kept; the app shows the latest
SHOWN = 80


class SetupJob:
    def __init__(self, config, audit=None, on_done: Callable[[dict], None] | None = None, plan=None, setup=None,
                 voice: bool = True):
        from ..models import manager
        self.config, self.audit, self.on_done, self.voice = config, audit, on_done, voice
        self._plan, self._setup = plan or manager.plan, setup or manager.setup
        self._lock = threading.Lock()
        self.state = "idle"             # idle | running | done | failed
        self.lines: list[str] = []
        self.result: dict | None = None
        self.planned: dict | None = None

    def public(self) -> dict:
        with self._lock:
            return {"configured": bool(self.config.get("models.chat")), "model": self.config.get("models.chat") or "",
                    "state": self.state, "lines": self.lines[-SHOWN:], "line_count": len(self.lines),
                    "result": self.result, "plan": self.planned}

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
        with self._lock:
            if self.state == "running":
                return False
            self.state, self.lines, self.result = "running", [], None
        threading.Thread(target=self._run, args=(model, accept_license), daemon=True, name="setup").start()
        return True

    def _say(self, message: str) -> None:
        with self._lock:
            self.lines.append(str(message))
            del self.lines[:-KEEP]

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
        with self._lock:
            self.result = {key: result.get(key) for key in keep if key in result}
            self.state = "done" if result.get("ok") else "failed"
        if self.audit is not None:
            self.audit.write("setup.finished", ok=bool(result.get("ok")), model=result.get("entry") or "",
                             error=str(result.get("error") or "")[:300])
        if result.get("ok") and self.on_done is not None:
            self.on_done(result)
