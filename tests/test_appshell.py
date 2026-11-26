import subprocess
import shutil
"""`coderunner app`: Electron from the checkout or CodeRunner's own pinned copy, and the app opened with it."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from coderunner import appshell


class AppShellTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)

    def fake_electron(self, folder):
        binary = appshell._binary(folder)
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        return str(binary)

    def test_the_pinned_electron_is_the_one_the_app_was_written_for(self):
        package = json.loads((appshell.app_dir() / "package.json").read_text())
        self.assertEqual(package["devDependencies"]["electron"], appshell.ELECTRON)
        self.assertEqual(appshell.build("Darwin", "arm64")[0], "darwin-arm64")
        self.assertEqual(appshell.build("Linux", "AMD64")[0], "linux-x64")
        self.assertEqual(appshell.build("Linux", "arm64")[0], "linux-arm64")
        self.assertEqual(appshell.build("Darwin", "aarch64")[0], "darwin-arm64")
        self.assertIsNone(appshell.build("Windows", "AMD64"))
        for asset, sha256 in appshell.BUILDS.values():
            self.assertRegex(sha256, r"^[0-9a-f]{64}$", asset)

    def test_the_page_can_ask_the_shell_for_a_folder_and_for_nothing_else(self):
        import re
        main, preload = ((appshell.app_dir() / name).read_text() for name in ("main.js", "preload.js"))
        self.assertIn('preload: path.join(__dirname, "preload.js")', main)
        self.assertIn("contextIsolation: true, nodeIntegration: false, sandbox: true", main)
        # The Browser tab's: where the browser goes, the owner's buttons, what it shows (main window only).
        asks = ["pick-folder", "start-assistant", "show-in-finder", "open-system-settings", "app-settings",
                "open-thread", "open-app", "open-computer", "open-settings", "hide-panel", "task-started",
                "thread-open", "developer-mode", "browser-view", "browser-act", "browser-state"]
        self.assertEqual(re.findall(r'ipcMain\.(?:handle|on)\("([\w-]+)"', main), asks)
        self.assertEqual(re.findall(r'ipcRenderer\.(?:invoke|send)\("([\w-]+)"', preload), asks)
        # What the shell says to the pages: which thread to open, that a panel was shown, developer mode turned on or off.
        self.assertEqual(re.findall(r'ipcRenderer\.on\("([\w-]+)"', preload),
                         ["show-thread", "show-computer", "open-settings", "panel-shown", "test-this-mac", "developer-mode",
                          "browser-state", "browser-front"])
        self.assertIn('ipcMain.handle("developer-mode", (event) => fromDaemon(event) && developerMode());', main)
        asks.remove("developer-mode")
        # The Browser tab's asks are heard from the main window's own page only, the daemon's.
        self.assertIn("const mainPage = (event) => fromDaemon(event) && mainWindow && !mainWindow.isDestroyed() && "
                      "event.sender === mainWindow.webContents;", main)
        for ask in ("browser-act", "browser-state"):
            self.assertRegex(main, r'ipcMain\.handle\("%s", \(event[^)]*\) => \(mainPage\(event\)' % ask)
            asks.remove(ask)
        self.assertEqual(re.findall(r'exposeInMainWorld\("(\w+)"', preload), ["coderunnerShell"])
        self.assertIn("if (asker !== ORIGIN) return null;", main)        # only the daemon's own page is answered
        self.assertIn("return asker === ORIGIN ? startDaemon() : false;", main)
        # Every IPC handler and message is strictly origin-gated to the daemon's own pages.
        for ask in ("show-in-finder", "open-system-settings", "app-settings"):
            self.assertRegex(main, r'ipcMain\.handle\("%s", \(event[^)]*\) => \{\s*if \([^)]*fromDaemon\(event\)' % ask)
        for ask in ("open-thread", "open-app", "open-computer", "open-settings", "hide-panel", "task-started", "thread-open"):
            self.assertRegex(main, r'ipcMain\.on\("%s", \(event[^)]*\) => \{\s*(?:if \(fromDaemon\(event\)|const win = .*\n\s+if '
                                   r'\(fromDaemon\(event\)|if \(!fromDaemon\(event\))' % ask)

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_the_shell_lets_the_daemons_page_hear_the_microphone_and_nothing_else(self):
        main = (appshell.app_dir() / "main.js").read_text()
        rules = main[main.index("  const own = (url) =>"):main.index("  // The one thing the page asks of the shell")]
        script = """
const ORIGIN = "http://127.0.0.1:8223", handlers = {};
const session = {defaultSession: {setPermissionRequestHandler: (f) => { handlers.ask = f; },
                                  setPermissionCheckHandler: (f) => { handlers.check = f; }}};
""" + rules + """
const page = {getURL: () => ORIGIN + "/app/"}, other = {getURL: () => "https://evil.example/"};
const ask = (contents, permission, details) => { let got; handlers.ask(contents, permission, (ok) => { got = ok; }, details); return got; };
console.log(JSON.stringify([
  ask(page, "media", {mediaTypes: ["audio"], requestingUrl: ORIGIN + "/app/"}),
  ask(page, "media", {mediaTypes: ["audio", "video"], requestingUrl: ORIGIN + "/app/"}),
  ask(page, "media", {mediaTypes: ["video"]}),
  ask(page, "media", {}),
  ask(other, "media", {mediaTypes: ["audio"], requestingUrl: "https://evil.example/"}),
  ask(page, "media", {mediaTypes: ["audio"], requestingUrl: "https://evil.example/frame"}),
  ask(page, "notifications", {}), ask(page, "geolocation", {}), ask(page, "clipboard-read", {}),
  handlers.check(page, "media", ORIGIN, {mediaType: "audio"}),
  handlers.check(page, "media", ORIGIN, {mediaType: "video"}),
  handlers.check(other, "media", "https://evil.example", {mediaType: "audio"}),
  handlers.check(page, "geolocation", ORIGIN, {}),
]));
"""
        done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertEqual(json.loads(done.stdout), [True, False, False, False, False, False, True, False, False,
                                                   True, False, False, False])

    def test_dictation_puts_the_words_in_the_box_and_never_sends_them(self):
        web = appshell.app_dir().parent / "coderunner" / "server" / "web" / "app"
        page, markup = (web / "app.js").read_text(), (web / "index.html").read_text()
        code = page[page.index("// ---- dictation"):page.index('document.querySelectorAll(".example")')]
        self.assertIn('fetch("/api/voice", {method: "POST"', code)
        self.assertIn("box.setRangeText(", code)                 # at the cursor, after what is there
        self.assertNotIn("requestSubmit", code)                  # the owner reads it and sends it
        self.assertNotRegex(code, r"\balert\(")
        self.assertIn("getUserMedia({audio: true})", code)       # sound only
        self.assertIn('$("mic").hidden = !(status.voice || []).length', page)    # only where a voice engine is set up
        self.assertRegex(markup, r'<button id="mic"[^>]*hidden')

    def test_the_shell_brings_a_dead_daemon_back_unless_the_owner_stopped_it(self):
        import argparse
        from unittest import mock

        from coderunner import cli
        main = (appshell.app_dir() / "main.js").read_text()
        self.assertIn("if (misses >= 2 && !stoppedByOwner()) startDaemon();", main)
        self.assertIn('path.join(ASSISTANT, "run", "stopped-by-owner")', main)
        # `coderunner down` leaves the mark the shell looks for, and any start clears it.
        with mock.patch.dict("os.environ", {"CODERUNNER_HOME": str(self.home)}), \
                mock.patch("coderunner.server.service.status", return_value={}), mock.patch("builtins.print"):
            mark = cli._stopped_mark()
            cli.cmd_down(argparse.Namespace(uninstall=False, sandbox=False), mock.Mock())
            self.assertTrue(mark.exists())
            with mock.patch.object(cli, "_daemon_up", return_value=True), mock.patch("coderunner.sandbox.driver_for"):
                cli.cmd_up(argparse.Namespace(), mock.Mock())
            self.assertFalse(mark.exists())

    def test_the_app_shows_no_raw_errors_terminal_commands_or_alerts(self):
        page = (appshell.app_dir().parent / "coderunner" / "server" / "web" / "app" / "app.js").read_text()
        code = "\n".join(line for line in page.splitlines() if not line.strip().startswith("//"))
        self.assertNotRegex(code, r"\balert\(")
        self.assertNotIn("Failed to fetch", code)
        self.assertNotRegex(code, r"is not reachable: \$\{")
        self.assertNotRegex(code, r"`coderunner |coderunner (up|setup|down)\b")
        gateway = (appshell.app_dir().parent / "coderunner" / "server" / "gateway.py").read_text()
        facing = gateway[gateway.index("async def computer("):gateway.index("async def computer_screen(")]
        self.assertNotIn("__name__", facing)
        self.assertNotIn("coderunner up", facing)
        self.assertIn("No model is set up yet.", gateway)

    def test_a_high_risk_approval_is_shown_whole_and_the_owners_turn_has_its_own_button(self):
        import re
        web = appshell.app_dir().parent / "coderunner" / "server" / "web" / "app"
        page, style, markup = ((web / name).read_text() for name in ("app.js", "app.css", "index.html"))
        # A message to be sent is never cut to the two lines a script's preview gets: it scrolls inside its card.
        self.assertRegex(page, r'const grave = e\.risk === "high" && !act;\s+if \(grave\) this\.showWhole\(card')
        body = re.search(r"^\.ask-body \{[^}]*\}", style, re.M).group(0)
        self.assertNotIn("line-clamp", body)
        self.assertIn("overflow-y: auto", body)
        whole = page[page.index("  showWhole(card, summary) {"):page.index("  finish(e) {")]
        self.assertIn("not in what you asked, nor in your contacts", whole)      # the caution, set apart
        self.assertIn("ask-caution", whole)
        self.assertRegex(whole, r"\^\(To\|Cc\|Bcc")                              # who it reaches, a line each
        # The owner's step is said once, and the large screen has the button that hands it back.
        self.assertIn('id="turnDone"', markup)
        self.assertIn('$("turnDone").onclick = finish', page)
        # Finished check tasks are on the Check screen, not among the owner's own.
        self.assertIn('c.conversation.startsWith("app:check-") && c.status !== "running"', page)

    @unittest.skipIf(shutil.which("node") is None, "node not installed")
    def test_the_computer_has_a_terminal_tab_with_xterm_kept_here(self):
        root = appshell.app_dir().parent
        web = root / "coderunner" / "server" / "web" / "app"
        page, style, markup = ((web / name).read_text() for name in ("app.js", "app.css", "index.html"))
        # Composed tabs in the workspace header: sandbox screen, terminal, Mac browser, and system overview.
        self.assertIn('<button id="viewScreen" class="ctab on" type="button" role="tab" aria-selected="true" title="Sandbox screen">', markup)
        self.assertIn('<button id="viewTerminal" class="ctab" type="button" role="tab" aria-selected="false" title="Sandbox terminal">', markup)
        self.assertIn('<button id="viewBrowser" class="ctab" type="button" role="tab" aria-selected="false" hidden', markup)
        self.assertIn('<button id="viewMac" class="ctab" type="button" role="tab" aria-selected="false" title="System and models">', markup)
        self.assertIn('<div id="term" hidden><div id="termNote" class="term-note">Connecting…</div></div>', markup)
        # xterm.js from the app's own files, never a CDN, loaded only when the tab opens.
        for name in ("xterm.js", "addon-fit.js", "xterm.css", "LICENSE.txt"):
            self.assertTrue((web / "xterm" / name).is_file(), name)
        self.assertIn("xterm.js v5.5.0", (root / "THIRD_PARTY_NOTICES.md").read_text())
        self.assertNotRegex(page + markup, r"https?://(cdn|unpkg|cdnjs)")
        self.assertIn('loadScript("xterm/xterm.js")', page)
        self.assertIn("/api/computer/terminal", page)
        # Light or dark with the app: the theme follows the page's setting, then the system's, live.
        self.assertIn('darkQuery.addEventListener?.("change", applyTermTheme)', page)
        self.assertIn('attributeFilter: ["data-theme"]', page)
        self.assertIn('fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12.5', page)
        self.assertIn("#term { position: absolute; inset: 0; background: var(--surface); }", style)
        # The tab is remembered where storage works, and the page works where it does not.
        self.assertRegex(page, r"try \{ localStorage\.setItem\(VIEW_KEY, view\); \} catch")

    def test_the_thread_renderer_makes_messages_marks_and_cards_and_never_html(self):
        root = appshell.app_dir().parent
        done = subprocess.run(["node", str(root / "tests" / "js" / "render_test.mjs"),
                               str(root / "coderunner" / "server" / "web" / "app" / "render.js")],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "ok"), done.stderr[-800:])
        page = (root / "coderunner" / "server" / "web" / "app" / "render.js").read_text()
        self.assertNotIn("innerHTML", page)
        code = "\n".join(line for line in page.splitlines() if not line.strip().startswith("//"))
        self.assertNotRegex(code, r"favicon|googleusercontent|icons\.duckduckgo")     # no icon service is asked

    def test_the_checkouts_own_electron_is_used_before_coderunners_copy(self):
        app = self.home / "checkout" / "app"
        self.assertIsNone(appshell.installed(app))
        managed = self.fake_electron(appshell.managed_dir())
        self.assertEqual(appshell.installed(app), managed)
        own = self.fake_electron(app / "node_modules" / "electron" / "dist")
        self.assertEqual(appshell.installed(app), own)

    def test_a_download_that_does_not_match_its_checksum_is_refused(self):
        class Response:
            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                yield b"not the published build"

        class Stream:
            def __enter__(self):
                return Response()

            def __exit__(self, *exc):
                return False

        with mock.patch.object(appshell, "build", return_value=("darwin-arm64", "0" * 64)), \
                mock.patch.object(appshell.httpx, "stream", return_value=Stream()):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                appshell.install()
        self.assertEqual(list(appshell.managed_dir().iterdir()), [])
        with mock.patch.object(appshell, "build", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no build for this platform"):
                appshell.install()

    def test_the_app_opens_on_this_home_in_a_process_of_its_own(self):
        binary = self.fake_electron(appshell.managed_dir())
        with mock.patch.object(appshell.subprocess, "Popen") as popen, \
                mock.patch.object(appshell, "installed", return_value=binary), \
                mock.patch.object(appshell, "bundled", side_effect=lambda found, app=None: found):
            appshell.launch()
        argv, options = popen.call_args[0][0], popen.call_args[1]
        self.assertEqual(argv, [binary, str(appshell.app_dir())])
        self.assertEqual(options["env"]["CODERUNNER_HOME"], str(self.home))
        self.assertTrue(options["start_new_session"])
        with mock.patch.object(appshell, "app_dir", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "from a checkout"):
                appshell.launch()

    def test_on_a_mac_the_app_is_named_coderunner_with_its_icon_not_electron(self):
        import plistlib
        binary = self.fake_electron(self.home / "dist")
        contents = pathlib.Path(binary).parents[1]
        (contents / "Resources").mkdir()
        (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleName": "Electron", "CFBundleIdentifier":
                                                              "com.github.Electron", "CFBundleExecutable": "Electron"}))
        real_run, signed = subprocess.run, []

        def run(argv, **kwargs):
            if argv[0] == "/usr/bin/codesign":
                signed.append(argv)
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return real_run(argv, **kwargs)
        with mock.patch.object(appshell.subprocess, "run", run):
            made = appshell.bundled(binary, appshell.app_dir())
            again = appshell.bundled(binary, appshell.app_dir())
        target = appshell.bundle_dir()
        self.assertEqual(made, str(target / "Contents" / "MacOS" / "Electron"))
        info = plistlib.loads((target / "Contents" / "Info.plist").read_bytes())
        self.assertEqual((info["CFBundleName"], info["CFBundleDisplayName"], info["CFBundleIdentifier"],
                          info["CFBundleIconFile"]), ("CodeRunner", "CodeRunner", "io.instavm.coderunner", "coderunner.icns"))
        self.assertTrue((target / "Contents" / "Resources" / "coderunner.icns").is_file())
        self.assertEqual(again, made)
        self.assertEqual(len(signed), 1)                        # made once, signed again on this Mac once
        # Signing that fails: the plain Electron is used rather than an app macOS would refuse to start.
        shutil.rmtree(target)

        def refuse(argv, **kwargs):
            if argv[0] == "/usr/bin/codesign":
                raise subprocess.CalledProcessError(1, argv)
            return real_run(argv, **kwargs)
        with mock.patch.object(appshell.subprocess, "run", refuse):
            self.assertEqual(appshell.bundled(binary, appshell.app_dir()), binary)
        self.assertFalse(target.exists())

    def test_a_zip_is_unpacked_with_its_links_and_executable_bits(self):
        import stat
        import zipfile
        archive = self.home / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            tool = zipfile.ZipInfo("electron")
            tool.external_attr = (stat.S_IFREG | 0o755) << 16
            bundle.writestr(tool, "#!/bin/sh\n")
            link = zipfile.ZipInfo("current")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(link, "electron")
        out = self.home / "out"
        out.mkdir()
        with mock.patch.object(appshell.platform, "system", return_value="Linux"):
            appshell._unzip(archive, out)
        self.assertTrue(os.access(out / "electron", os.X_OK))
        self.assertEqual(os.readlink(out / "current"), "electron")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../outside", "x")
        with mock.patch.object(appshell.platform, "system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "outside its folder"):
                appshell._unzip(archive, out)

    def test_app_settings_checks_shortcut_return_value_and_persists_developer_mode(self):
        main_js = (appshell.app_dir() / "main.js").read_text()
        self.assertIn("registered = Boolean(globalShortcut.register(candidate", main_js)
        self.assertIn("globalShortcut.register(cur.quickAskShortcut", main_js)
        self.assertIn("next.developerMode = update.developerMode", main_js)


if __name__ == "__main__":
    unittest.main()
