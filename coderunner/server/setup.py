"""Setup from the app: the briefing for this machine and setup itself, and the sandbox's computer, each as a
background job whose progress the app shows. They run the same manager.plan, manager.setup and
sandbox.computer.setup as `coderunner setup` and `coderunner browser computer`, so the app and the terminal
cannot differ."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import sys
import threading
from typing import Callable

KEEP = 400          # progress lines kept; the app shows the latest
SHOWN = 80

# Setup's progress lines, put for the app into what an owner follows: each stage says what is being fetched or
# done in a few plain words (the planning model, the vision part), never "the model" and never a file's name,
# and in the words the first screen uses for the assistant's jobs. A line belongs to the first rule that matches it; a rule may be tied to the stage setup is already in,
# so that a download during voice setup is the voice model's.
#   (stage, pattern, the stage a line must already be in or None, what is happening)
RULES = (
    ("engine", r"^(no model runtime|downloading (Ollama|llama-))", None, "Downloading the model engine"),
    ("voice", r"^(setting up speech to text|voice setup failed)", None, "Setting up voice"),
    ("voice", r"^(downloading FluidAudio|building the Parakeet helper)", None, "Preparing voice"),
    ("voice", r"^preparing Parakeet", None, "Preparing voice"),
    ("voice", r"^timing Parakeet", None, "Testing voice"),
    ("voice", r"^downloading ", "voice", "Downloading the voice model"),
    ("search", r"^installing Core ML packages", None, "Installing file search"),
    ("search", r"^preparing the embedding model", None, "Preparing file search"),
    ("search", r"^timing embeddings", None, "Testing file search"),
    ("search", r"^(embeddings:|embedding model|removed the Core ML|Core ML embeddings)", None, "Setting up file search"),
    ("search", r"^(downloading |pulling |ollama pull)", "search", "Downloading the file search model"),
    ("checker", r"^(decision model|no decision model)", None, "Setting up the safety check"),
    ("planner", r"^downloading .*mmproj", None, "Downloading the vision part"),
    ("planner", r"^(downloading |pulling |ollama pull|\S+: [\d.]+ GB to download)", None, "Downloading the planning model"),
    ("planner", r"^skipping ", None, "Choosing a model that fits the disk"),
    ("planner", r"^\S+ via \S+:", None, "Choosing the planning model"),
    ("check", r"^loading ", None, "Loading the planning model"),
    ("check", r"^(smoke testing|\s+(pass|FAIL) )", None, "Testing tools and vision"),
    ("speed", r"^(timing \S+ on this machine|measured |could not time)", None, "Measuring speed"),
)
RULES = tuple((stage, re.compile(pattern), within, doing) for stage, pattern, within, doing in RULES)
ORDER = ("engine", "planner", "check", "speed", "search", "checker", "voice")
DONE = {"engine": "Model engine ready", "planner": "Planning model downloaded", "check": "Tools and vision work",
        "speed": "Speed measured", "search": "File search ready", "checker": "Safety check ready", "voice": "Voice ready"}
PERCENT = re.compile(r"(\d+)% of ([\d.]+) GB")
MEASURED = re.compile(r"measured ([\d.]+) tok/s reading, ([\d.]+) tok/s writing: ([\d.]+) s per agent step")
PARAKEET_GB = 0.63          # what FluidAudio loads for Parakeet Ultra (voice/parakeet.py)


# A download in the briefing's words: what it is for. Its name is in the list of jobs above it.
DOWNLOADS = (("assistant model", "Planning model"), ("embedding model", "File search model"),
             ("decision model", "Safety check model"), ("voice model", "Voice model"),
             ("core ml packages", "File search runtime"), ("ollama", "Model engine (Ollama)"),
             ("llama.cpp", "Model engine"), ("prism", "Model engine"))


def _short_path(path: pathlib.Path) -> str:
    """A folder as the setup screen shows it: ~ for the home folder, and only its end when it is long (the
    whole path is the row's tooltip)."""
    text = str(path)
    home = str(pathlib.Path.home())
    if text == home or text.startswith(home + os.sep):
        text = "~" + text[len(home):]
    if len(text) > 44:
        text = "…" + os.sep + os.sep.join(path.parts[-2:])
    return text


def _disk_gb(path: pathlib.Path) -> float:
    """Decimal GB of the whole disk this folder is (or will be) on; 0 when it cannot be read."""
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).total / 1e9
    except OSError:
        return 0.0


def _download_label(what: str) -> str:
    low = what.lower()
    return next((label for start, label in DOWNLOADS if low.startswith(start)), what[:1].upper() + what[1:])


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
            self._follow(str(message))

    def _follow(self, line: str) -> None:
        """Where a line puts the job among its stages (a job without stages keeps none)."""

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
        self._stage = ""                        # the stage the latest line belongs to
        self._seen: dict[str, dict] = {}        # stage -> what it is doing, how far, and how it ended

    # ---- progress as stages --------------------------------------------------

    def _follow(self, line: str) -> None:
        for stage, pattern, within, doing in RULES:
            if pattern.search(line) and within in (None, self._stage):
                # A later stage does not hand a line back to an earlier one: a file fetched after the planning
                # model was checked belongs to what is being set up now.
                if self._stage and ORDER.index(stage) < ORDER.index(self._stage):
                    stage, doing = self._stage, self._seen[self._stage]["label"]
                self._stage = stage
                seen = self._seen.setdefault(stage, {"label": doing, "detail": "", "percent": None, "of_gb": None})
                seen["label"] = doing
                break
        if not self._stage:
            return
        seen = self._seen[self._stage]
        found = PERCENT.search(line)
        seen["percent"], seen["of_gb"] = (int(found.group(1)), float(found.group(2))) if found else (None, None)
        seen["detail"] = line.strip()[:160]
        measured = MEASURED.search(line)
        if measured:
            seen["done"] = f"Speed measured: {float(measured.group(3)):.0f} s per step"
        if "vision check failed" in line and "check" in self._seen:
            self._seen["check"]["done"] = "Tools work (text only on this computer)"

    def stages(self) -> list[dict]:
        """The stages reached so far, in order: each says what it is doing while it runs (with a percentage
        while something downloads) and what came of it once it is done. Stages not reached yet are not listed,
        as which of them a setup has depends on the machine."""
        out = []
        for key in ORDER:
            seen = self._seen.get(key)
            if seen is None:
                continue
            current = key == self._stage
            state = "done" if not current or self.state == "done" else "failed" if self.state == "failed" else "running"
            label = seen.get("done") or DONE[key] if state == "done" else seen["label"]
            out.append({"key": key, "label": label, "state": state, "percent": seen["percent"],
                        "of_gb": seen["of_gb"], "detail": seen["detail"]})
        return out

    def public(self) -> dict:
        with self._lock:
            stages = self.stages()
        return {"configured": bool(self.config.get("models.chat")), "model": self.config.get("models.chat") or "",
                **self.progress(), "stages": stages, "plan": self.planned}

    # ---- the briefing before anything is fetched -------------------------------

    def plan(self) -> dict:
        """What the app shows before setup starts: this computer, how fast it will be, the model for each job,
        what will be downloaded, where to, and what that leaves on the disk. Slow the first time in a day (model
        discovery)."""
        from ..models import manager
        from ..models.runtimes import models_dir
        planned = self._plan(self.config)
        items = manager.downloads(self.config, planned)
        roles = self._roles(planned, items)
        for item in items:
            item["label"] = _download_label(item["what"])
        room, free, floor = manager._room_gb(self.config)
        model, hw = planned.get("model"), planned["hardware"]
        total = round(sum(item["gb"] or 0 for item in items), 2)
        summary = {
            "hardware": {"chip": hw.get("chip") or "unknown chip", "ram_gb": round(hw.get("ram_gb") or 0),
                         "gpu": hw.get("gpu"), "budget_gb": round(hw.get("budget_gb") or 0, 1),
                         "unified": bool(hw.get("unified")), "vram_gb": round(hw.get("vram_gb") or 0),
                         "measured": hw.get("measured") or "", "os": "mac" if sys.platform == "darwin" else "linux"},
            "speed": model and {"measured": bool(hw.get("measured") or model.get("tasks")),
                                "read_tok_s": model.get("read_tok_s"), "tok_s": model.get("tok_s"),
                                "step_s": model.get("step_s"), "bandwidth_gbs": hw.get("bandwidth_gbs"),
                                "when": hw.get("measured") or "", "tasks": model.get("tasks")},
            "model": model and {"id": model["id"], "name": manager.pretty(model["id"]), "runtime": model.get("runtime"),
                                "ctx": model.get("ctx"), "step_s": model.get("step_s"), "vision": model.get("vision"),
                                "need_gb": model.get("need_gb"), "tok_s": model.get("tok_s"),
                                "reasons": model.get("reasons", [])[:4]},
            "roles": roles,
            "alternatives": [{"id": c["id"], "name": manager.pretty(c["id"]), "runtime": c.get("runtime"),
                              "step_s": c.get("step_s")} for c in planned.get("alternatives", [])[:4]],
            "notes": planned.get("notes", []), "downloads": items, "download_gb": total,
            "free_gb": round(free, 1), "room_gb": round(max(room, 0), 1), "floor_gb": floor,
            "after_gb": round(free - total, 1), "disk_gb": round(_disk_gb(models_dir(self.config)), 1),
            "models_dir": str(models_dir(self.config)), "models_dir_short": _short_path(models_dir(self.config)),
            "models_dir_default": not self.config.get("models.dir"),
        }
        with self._lock:
            self.planned = summary
        return summary

    def _roles(self, planned: dict, items: list[dict]) -> list[dict]:
        """A model for each job the assistant has, as the plan has them for this machine. `items` gains what
        the plan itself does not list (the voice model)."""
        from ..models import coreml, manager
        sizes = {item["what"].split(": ", 1)[-1]: item["gb"] for item in items if ": " in item["what"]}
        roles: list[dict] = []
        model = planned.get("model")
        where = {"llama.cpp": "llama.cpp, on the GPU", "prism": "Prism's llama.cpp, on the GPU",
                 "ollama": "Ollama", "mlx": "MLX", "openai": "a server you run"}
        if planned["hardware"].get("gpu") == "none":
            where = {key: value.replace(", on the GPU", ", on the processor") for key, value in where.items()}
        if model:
            roles.append({"key": "planner", "short": "Planning", "job": "Plans each task and carries it out",
                          "model": manager.pretty(model["id"]), "on": where.get(model.get("runtime") or "", "this computer"),
                          "memory_gb": model.get("need_gb"), "download_gb": sizes.get(model["id"], 0)})
            sees = bool(model.get("vision"))
            roles.append({"key": "eyes", "short": "Vision", "job": "Reads pictures, screenshots and the screen",
                          "model": "The planning model" if sees else "Not on this computer",
                          "on": "" if sees else "the model that fits here reads text only",
                          "download_gb": 0, "shared": sees, "missing": not sees})
        if coreml.supported() and (self.config.get("models.embedding_runtime") or "auto") != "gpu":
            entry = next((e for e in manager.catalog.by_role("embedding") if coreml.CoreMLRuntime.artifact(e)), None)
            if entry:
                roles.append({"key": "search", "short": "File search", "job": "Finds things in your files, by what they are about",
                              "model": manager.pretty(entry["id"]), "on": "Core ML, on the Neural Engine",
                              "download_gb": round(sum(i["gb"] or 0 for i in items if "Core ML" in i["what"]), 2)})
        elif planned.get("embedding"):
            pick = planned["embedding"]
            roles.append({"key": "search", "short": "File search", "job": "Finds things in your files, by what they are about",
                          "model": manager.pretty(pick["id"]), "on": "Ollama", "download_gb": sizes.get(pick["id"], 0)})
        if self.voice:
            roles.append(self._voice(items))
        pick = planned.get("decision")
        roles.append({"key": "checker", "short": "Safety check", "job": "Checks each step before it pays, books, sends or deletes",
                      "model": manager.pretty(pick["id"]) if pick else "The planning model", "on": "Ollama" if pick else "",
                      "download_gb": sizes.get(pick["id"], 0) if pick else 0, "shared": not pick})
        return [role for role in roles if role]

    def _voice(self, items: list[dict]) -> dict | None:
        try:
            from ..voice import Voice
            voice = Voice(self.config)
            parakeet = voice.parakeet
            if parakeet.possible() and parakeet.serves():
                need = 0 if parakeet.ready() else PARAKEET_GB
                if need:
                    items.append({"what": "voice model: Parakeet Ultra", "gb": need})
                return {"key": "voice", "short": "Voice", "job": "Hears you when you talk to it", "model": "Parakeet Ultra",
                        "on": "Core ML, on the Neural Engine", "download_gb": need}
            if voice.apple.possible():
                return {"key": "voice", "short": "Voice", "job": "Hears you when you talk to it", "model": "Apple's on-device speech",
                        "on": "built into macOS", "download_gb": 0}
        except Exception:               # the briefing goes on without a voice row
            return None
        return None

    def set_location(self, path: str) -> dict:
        """Keep model files in another folder (an empty path: back to the default), then plan again, since the
        free space is that folder's disk's."""
        with self._lock:
            if self.state == "running":
                raise ValueError("Setup is running; the folder can be changed once it has finished.")
        path = (path or "").strip()
        if path:
            folder = pathlib.Path(path).expanduser()
            if not folder.is_absolute():
                raise ValueError("Give the folder's full path, such as /Volumes/Disk/CodeRunner.")
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"That folder cannot be made: {exc.strerror or exc}") from exc
            if not os.access(folder, os.W_OK):
                raise ValueError("That folder cannot be written to.")
            path = str(folder)
        self.config.set("models.dir", path)
        self.config.save()
        return self.plan()

    # ---- setup -----------------------------------------------------------------

    def start(self, model: str | None = None, accept_license: bool = False) -> bool:
        """Run setup in the background; False when one is already running."""
        with self._lock:
            if self.state != "running":
                self._stage, self._seen = "", {}
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
