"""The desktop app's menu bar: what /api/tray lists, and the shell's icon, popover and quick ask."""

import asyncio
import json
import pathlib
import re
import shutil
import struct
import subprocess
import unittest

import httpx

from coderunner import appshell
from coderunner.server import tasks as tasks_module
from coderunner.server.gateway import admin_app
from fakes import Svc
from test_gateway import ADMIN, FakeDaemon

WEB = pathlib.Path(__file__).resolve().parent.parent / "coderunner" / "server" / "web" / "app"


class TrayListTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = Svc({"models": {"chat": "test"}})
        self.daemon = FakeDaemon(self.svc)
        self.gates: dict[str, asyncio.Event] = {}

        async def handle(channel, chat_id, text, origin="", events=None, task="", images=None):
            events({"type": "note", "text": f"Checking Skyscanner for the 12th. Then the {chat_id} fares."})
            if chat_id == "fares":              # this one uses the computer's browser
                events({"type": "activity", "event": "browser.open", "url": "https://www.skyscanner.co.in/"})
            await self.gates.setdefault(chat_id, asyncio.Event()).wait()
            return f"The cheapest for {chat_id} is ₹4,812."
        self.daemon.handle_message = handle
        self.bus = self.daemon.bus = tasks_module.EventBus(self.svc.store)
        self.tasks = self.daemon.tasks = tasks_module.Tasks(self.daemon, self.bus)

    def add(self, conversation, status, created, finished=None, reply=""):
        self.svc.store.execute("INSERT INTO tasks(id, conversation, prompt, status, reply, created, finished) "
                               "VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (f"t-{conversation}", conversation, f"ask {conversation}", status, reply, created, finished))

    async def test_needs_you_first_then_running_then_recent_each_with_its_line(self):
        self.add("app:old", "done", 10, 20, "An old answer " + "word " * 60)
        self.add("app:broke", "failed", 30, 40)
        self.add("app:check-world-1", "running", 50)              # the check's own threads are not the owner's
        running = self.tasks.start("app:fares", "find fares", "owner")
        asking = self.tasks.start("app:book", "book it", "owner")
        await asyncio.sleep(0.02)
        self.bus.publish({"type": "activity", "event": "approval.requested", "id": "a1", "task": asking["id"],
                          "conversation": "app:book"})
        listed = self.tasks.tray()
        order = [t["conversation"] for t in listed["threads"]]
        self.assertEqual(order, ["app:book", "app:fares", "app:broke", "app:old"])
        self.assertEqual((listed["running"], listed["waiting"]), (2, 1))
        book, fares, broke, old = listed["threads"]
        self.assertTrue(book["waiting"])
        self.assertEqual(book["line"], "")                      # "Needs you" is said by the popover
        self.assertEqual(fares["line"], "Checking Skyscanner for the 12th")
        self.assertEqual(fares["status"], "running")
        self.assertEqual(fares["task"], running["id"])
        self.assertIsNone(fares["finished"])
        self.assertGreater(fares["started"], 0)
        self.assertEqual((fares["computer"], book["computer"], listed["computer"]), (True, False, True))
        self.assertEqual((broke["status"], broke["finished"]), ("failed", 40))
        self.assertLessEqual(len(old["reply"]), 160)
        self.assertTrue(old["reply"].startswith("An old answer"))
        # Answered and finished: no longer waiting, its line gone, the answer's start kept.
        self.bus.publish({"type": "activity", "event": "approval.decided", "id": "a1", "task": asking["id"],
                          "conversation": "app:book"})
        self.gates["book"] = self.gates.get("book") or asyncio.Event()
        self.gates["book"].set()
        await asyncio.sleep(0.05)
        listed = self.tasks.tray()
        self.assertEqual([t["conversation"] for t in listed["threads"]][:1], ["app:fares"])
        book = next(t for t in listed["threads"] if t["conversation"] == "app:book")
        self.assertEqual((book["status"], book["waiting"], book["line"]), ("done", False, ""))
        self.assertIn("₹4,812", book["reply"])
        self.assertGreaterEqual(book["finished"], book["started"])
        self.assertEqual((listed["running"], listed["waiting"]), (1, 0))
        self.gates.setdefault("fares", asyncio.Event()).set()
        await asyncio.sleep(0.05)
        self.assertFalse(self.tasks.tray()["computer"])         # nothing running uses it: no screen in the popover

    async def test_the_route_needs_the_token_and_lists_at_most_a_dozen(self):
        for i in range(20):
            self.add(f"app:t{i}", "done", i, i + 1)
        app = admin_app(self.svc, self.daemon)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            self.assertEqual((await client.get("/api/tray")).status_code, 401)
            response = await client.get("/api/tray", headers={"Authorization": "Bearer " + ADMIN})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["threads"]), 12)
        self.assertEqual(data["threads"][0]["conversation"], "app:t19")
        self.assertEqual((data["running"], data["waiting"]), (0, 0))


def png_size(path: pathlib.Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


class TrayShellTests(unittest.TestCase):
    def setUp(self):
        self.app = appshell.app_dir()
        self.main = (self.app / "main.js").read_text()

    def test_the_icons_are_the_mark_at_16_points_template_where_the_bar_tints_them(self):
        folder = self.app / "tray"
        for name in ("trayTemplate", "trayDotTemplate", "trayNeedsYou", "trayNeedsYouDark"):
            self.assertEqual(png_size(folder / f"{name}.png"), (16, 16))
            self.assertEqual(png_size(folder / f"{name}@2x.png"), (32, 32))
            self.assertIn(f'"{name}"', self.main)
        self.assertIn('if (name.endsWith("Template")) image.setTemplateImage(true);', self.main)
        self.assertIn('tray.setTitle(running >= 2 ? String(running) : ""', self.main)

    def test_closing_the_window_on_a_mac_leaves_the_app_in_the_menu_bar(self):
        self.assertIn('app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });', self.main)
        self.assertIn("if (!MAC) app.quit();", self.main)                # elsewhere closing the window quits, as before
        self.assertIn('{label: "Quit CodeRunner", role: "quit"}', self.main)
        self.assertIn("app.requestSingleInstanceLock()", self.main)
        self.assertIn('app.on("activate"', self.main)

    def test_the_quick_ask_shortcut_is_a_setting_and_a_taken_one_is_no_shortcut(self):
        self.assertIn('path.join(app.getPath("userData"), "settings.json")', self.main)
        self.assertIn('quickAskShortcut: "Alt+Space"', self.main)
        register = self.main[self.main.index("function registerQuickAsk()"):]
        register = register[:register.index("\n}\n")]
        self.assertIn("try { globalShortcut.register(shortcut", register)
        self.assertIn('app.on("will-quit", () => globalShortcut.unregisterAll());', self.main)

    def test_the_shell_asks_the_daemon_with_the_token_and_the_pages_never_hold_it(self):
        self.assertIn("fetch(`${ORIGIN}/api/tray`, {headers: daemonHeaders()", self.main)
        self.assertIn("fetch(`${ORIGIN}/api/events?since=-1`, {headers: daemonHeaders()", self.main)
        for name in ("tray.js", "ask.js", "tray.html", "ask.html"):
            text = (WEB / name).read_text()
            self.assertNotIn("Authorization", text, name)
            self.assertNotIn("admin-token", text, name)
            self.assertNotIn("Bearer", text, name)

    def test_the_pages_put_words_in_as_text(self):
        for name in ("tray.js", "ask.js"):
            code = (WEB / name).read_text()
            self.assertNotRegex(code, r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write", name)
            self.assertNotRegex(code, r"\balert\(", name)
            self.assertIn('from "./render.js"', code)
        tray = (WEB / "tray.js").read_text()
        self.assertIn("shell.openThread?.(thread.conversation)", tray)     # a thread opens in the app, no approvals here
        self.assertNotIn("/api/approvals", tray)
        self.assertIn('"Needs you"', tray)
        self.assertIn("threads.slice(0, 5)", tray)
        ask = (WEB / "ask.js").read_text()
        self.assertIn('request.open("POST", "/api/uploads")', ask)
        self.assertIn('conversation: "", attachments: files.map((file) => file.id)', ask)
        for name in ("tray.html", "ask.html"):
            markup = (WEB / name).read_text()
            self.assertIn('placeholder="What should I get done?"', markup)
            self.assertIn('href="app.css"', markup)
            self.assertIn('class="logo ', markup)                         # the mark in brand blue
            self.assertEqual(markup.count('type="submit"'), 1)              # one button, round and blue
            self.assertIn('class="send-round"', markup)
        style = (WEB / "tray.css").read_text()
        self.assertIn(':root[data-theme="dark"]', style)
        self.assertIn("prefers-color-scheme: dark", style)

    def test_the_main_page_opens_a_thread_by_id_and_leaves_notifications_to_the_shell(self):
        page = (WEB / "app.js").read_text()
        self.assertIn("window.coderunnerShell?.onShowThread?.((id) => openById(id));", page)
        self.assertIn("location.hash.match(/^#c=([^&]+)$/)", page)
        self.assertIn('window.coderunnerShell?.threadOpen?.(id || "");', page)
        self.assertIn("if (window.coderunnerShell?.notifies || document.hasFocus()", page)
        self.assertIn('note.on("click", () => { notes.delete(note); showMain(conversation); });', self.main)
        self.assertIn("`${ORIGIN}/app/` + (win.pending ? `#${win.pending}` : \"\")", self.main)
        self.assertIn('computer ? "computer=expanded" : conversation ? `c=${encodeURIComponent(conversation)}` : ""', self.main)
        # The popover's expand button: the computer as large as the window, on its screen.
        self.assertIn('location.hash === "#computer=expanded"', page)
        self.assertIn("window.coderunnerShell?.onShowComputer?.(() => expandComputer());", page)
        expand = page[page.index("function expandComputer()"):]
        expand = expand[:expand.index("\n}\n")]
        for step in ("showComputer(true);", 'setView("screen", false);', "setExpanded(true);"):
            self.assertIn(step, expand)

    def test_the_popover_shows_the_computer_live_only_while_it_is_busy_and_seen(self):
        tray = (WEB / "tray.js").read_text()
        self.assertIn('import RFB from "./novnc/core/rfb.js";', tray)
        self.assertIn("/api/computer/screen", tray)
        self.assertIn('api("/api/computer")', tray)                # a look, never a start (POST brings it up)
        self.assertNotIn('"/api/computer", {method: "POST"', tray)
        self.assertIn("rfb.viewOnly = true;", tray)
        self.assertIn("showLive(Boolean(data.computer));", tray)
        self.assertIn("live.want = busy && !document.hidden;", tray)
        self.assertIn("if (document.hidden) showLive(false);", tray)   # put away: the connection closes
        self.assertIn("shell.openComputer?.()", tray)
        self.assertIn("trayState.computer ? POPOVER.withScreen : POPOVER.height", self.main)
        self.assertIn("panelWindow(ASK.bar + 2 * ASK.room, ASK.height + 2 * ASK.room, true)", self.main)
        self.assertIn("const ASK = {bar: 680, height: 64", self.main)

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_every_script_parses(self):
        for path in [self.app / "main.js", self.app / "preload.js", self.app / "browser.js", *(WEB / name for name in
                                                                         ("app.js", "render.js", "tray.js", "ask.js"))]:
            done = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, f"{path.name}: {done.stderr[-400:]}")

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_the_icon_says_needs_you_over_working_over_finished_unseen(self):
        start = self.main.index("function drawTray()")
        draw = self.main[start:self.main.index("\n}\n", start) + 3]
        script = """
const MAC = true, calls = [];
const nativeTheme = {shouldUseDarkColors: false};
const trayImage = (name) => name;
const tray = {isDestroyed: () => false, setImage: (n) => calls.push(["image", n]), setTitle: (t) => calls.push(["title", t]),
              setToolTip: (t) => calls.push(["tip", t])};
const trayState = {running: 0, waiting: 0, unseen: new Set(), image: ""};
""" + draw + """
const look = (running, waiting, unseen, dark) => {
  calls.length = 0; Object.assign(trayState, {running, waiting, unseen: new Set(unseen)}); nativeTheme.shouldUseDarkColors = dark;
  trayState.image = ""; drawTray(); return Object.fromEntries(calls);
};
console.log(JSON.stringify([look(0, 0, [], false), look(1, 0, [], false), look(3, 0, [], false), look(0, 0, ["c"], false),
                            look(2, 1, [], false), look(1, 1, [], true)]));
"""
        done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        idle, one, three, unseen, asks, asks_dark = json.loads(done.stdout)
        self.assertEqual((idle["image"], idle["title"], idle["tip"]), ("trayTemplate", "", "CodeRunner"))
        self.assertEqual((one["image"], one["title"]), ("trayDotTemplate", ""))
        self.assertEqual((three["image"], three["title"], three["tip"]), ("trayDotTemplate", "3", "CodeRunner: 3 working"))
        self.assertEqual((unseen["image"], unseen["title"]), ("trayDotTemplate", ""))
        self.assertEqual((asks["image"], asks["title"], asks["tip"]), ("trayNeedsYou", "2", "CodeRunner needs you"))
        self.assertEqual(asks_dark["image"], "trayNeedsYouDark")
        self.assertTrue(re.search(r"const POPOVER = \{width: 360, height: 420", self.main))


if __name__ == "__main__":
    unittest.main()
