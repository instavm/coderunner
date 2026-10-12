import os
import pathlib
import plistlib
import subprocess
import tempfile
import unittest
from unittest import mock

from coderunner.server import service

LABEL = "io.instavm.coderunner"

LAUNCHCTL_PRINT = """gui/501/io.instavm.coderunner = {
\tactive count = 1
\tpath = /Users/me/Library/LaunchAgents/io.instavm.coderunner.plist
\ttype = LaunchAgent
\tstate = running

\tprogram = /opt/venv/bin/python
\targuments = {
\t\t/opt/venv/bin/python
\t\t-m
\t\tcoderunner
\t\tdaemon
\t}
\tpid = 4242
\tlast exit code = (never exited)
}
"""


class FakeRunner:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        for prefix, result in self.results.items():
            if tuple(argv[:len(prefix)]) == prefix:
                if isinstance(result, list):
                    result = result.pop(0) if len(result) > 1 else result[0]
                code, out, err = result
                return subprocess.CompletedProcess(argv, code, out, err)
        return subprocess.CompletedProcess(argv, 0, "", "")


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.root / "home"), "USER": "me"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class LaunchAgentTests(ServiceTestCase):
    def agent(self, runner):
        return service.LaunchAgent(directory=self.root / "LaunchAgents", runner=runner,
                                   python="/opt/venv/bin/python", uid=501, sleep=lambda s: None)

    def test_install_writes_plist_and_bootstraps(self):
        runner = FakeRunner({("launchctl", "bootout"): (3, "", "Boot-out failed: 3: No such process")})
        message = self.agent(runner).install()
        path = self.root / "LaunchAgents" / f"{LABEL}.plist"
        self.assertEqual(runner.calls, [
            ["launchctl", "bootout", f"gui/501/{LABEL}"],
            ["launchctl", "bootstrap", "gui/501", str(path)],
        ])
        self.assertIn(str(path), message)
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)
        plist = plistlib.loads(path.read_bytes())
        logs = self.root / "home" / "assistant" / "logs"
        self.assertEqual(plist["Label"], LABEL)
        self.assertEqual(plist["ProgramArguments"], ["/opt/venv/bin/python", "-m", "coderunner", "daemon"])
        self.assertIs(plist["RunAtLoad"], True)
        self.assertEqual(plist["KeepAlive"], {"SuccessfulExit": False})
        self.assertEqual(plist["ProcessType"], "Interactive")
        self.assertEqual(plist["LimitLoadToSessionType"], "Aqua")
        self.assertEqual(plist["ThrottleInterval"], 10)
        self.assertEqual(plist["Umask"], 63)
        self.assertEqual(plist["WorkingDirectory"], str(self.root / "home" / "assistant"))
        self.assertEqual(plist["StandardOutPath"], str(logs / "daemon.out.log"))
        self.assertEqual(plist["StandardErrorPath"], str(logs / "daemon.err.log"))
        self.assertTrue(logs.is_dir())
        env = plist["EnvironmentVariables"]
        self.assertEqual(env["CODERUNNER_HOME"], str(self.root / "home"))
        self.assertTrue(env["PATH"].startswith("/opt/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"))

    def test_home_is_only_set_when_overridden(self):
        del os.environ["CODERUNNER_HOME"]
        plist = self.agent(FakeRunner()).definition()
        self.assertNotIn("CODERUNNER_HOME", plist["EnvironmentVariables"])

    def test_bootstrap_retries_once_then_reports(self):
        runner = FakeRunner({("launchctl", "bootstrap"): [(5, "", "Bootstrap failed: 5: Input/output error"),
                                                         (0, "", "")]})
        self.agent(runner).install()
        self.assertEqual([c[1] for c in runner.calls], ["bootout", "bootstrap", "bootstrap"])
        runner = FakeRunner({("launchctl", "bootstrap"): (5, "", "Bootstrap failed: 5: Input/output error")})
        with self.assertRaises(service.ServiceError) as caught:
            self.agent(runner).install()
        self.assertIn("Input/output error", str(caught.exception))

    def test_status_parses_launchctl_print(self):
        runner = FakeRunner({("launchctl", "print"): (0, LAUNCHCTL_PRINT, "")})
        info = self.agent(runner).status()
        self.assertEqual(runner.calls, [["launchctl", "print", f"gui/501/{LABEL}"]])
        self.assertTrue(info["loaded"] and info["running"])
        self.assertEqual(info["pid"], 4242)
        self.assertEqual(info["state"], "running")
        runner = FakeRunner({("launchctl", "print"): (113, "", "Could not find service")})
        info = self.agent(runner).status()
        self.assertFalse(info["loaded"] or info["running"] or info["installed"])

    def test_restart_and_uninstall(self):
        runner = FakeRunner()
        agent = self.agent(runner)
        agent.install()
        self.assertEqual(agent.restart(), f"Restarted {LABEL}.")
        self.assertEqual(runner.calls[-1], ["launchctl", "kickstart", "-k", f"gui/501/{LABEL}"])
        self.assertIn("Removed", agent.uninstall())
        self.assertEqual(runner.calls[-1], ["launchctl", "bootout", f"gui/501/{LABEL}"])
        self.assertFalse(agent.path.exists())
        self.assertIn("was not installed", agent.uninstall())
        failing = FakeRunner({("launchctl", "kickstart"): (113, "", "Could not find service")})
        with self.assertRaises(service.ServiceError):
            self.agent(failing).restart()


class SystemdTests(ServiceTestCase):
    def unit(self, runner, python="/usr/bin/python3"):
        return service.SystemdUnit(directory=self.root / "systemd", runner=runner, python=python)

    def test_install_writes_unit_and_enables(self):
        runner = FakeRunner()
        message = self.unit(runner).install()
        text = (self.root / "systemd" / "coderunner.server.service").read_text()
        self.assertIn("ExecStart=/usr/bin/python3 -m coderunner daemon\n", text)
        self.assertIn("Restart=on-failure\n", text)
        self.assertIn("RestartSec=2\n", text)
        self.assertIn("WantedBy=default.target\n", text)
        self.assertIn(f"Environment=CODERUNNER_HOME={self.root / 'home'}\n", text)
        self.assertEqual(runner.calls, [["systemctl", "--user", "daemon-reload"],
                                        ["systemctl", "--user", "enable", "--now", "coderunner.server.service"]])
        self.assertIn("loginctl enable-linger me", message)

    def test_paths_with_spaces_and_percent_are_quoted(self):
        text = self.unit(FakeRunner(), python="/home/me/my env/bin/python%1").definition()
        self.assertIn('ExecStart="/home/me/my env/bin/python%%1" -m coderunner daemon', text)

    def test_failures_status_restart_uninstall(self):
        runner = FakeRunner({("systemctl", "--user", "enable"): (1, "", "Failed to connect to bus")})
        with self.assertRaises(service.ServiceError) as caught:
            self.unit(runner).install()
        self.assertIn("Failed to connect to bus", str(caught.exception))
        runner = FakeRunner({("systemctl", "--user", "show"): (
            0, "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=777\n", "")})
        unit = self.unit(runner)
        info = unit.status()
        self.assertEqual(runner.calls[-1], ["systemctl", "--user", "show", "coderunner.server.service",
                                            "--property=LoadState,ActiveState,SubState,MainPID"])
        self.assertTrue(info["running"] and info["loaded"])
        self.assertEqual(info["pid"], 777)
        unit.restart()
        self.assertEqual(runner.calls[-1], ["systemctl", "--user", "restart", "coderunner.server.service"])
        unit.install()
        unit.uninstall()
        self.assertEqual(runner.calls[-2:], [["systemctl", "--user", "disable", "--now", "coderunner.server.service"],
                                             ["systemctl", "--user", "daemon-reload"]])
        self.assertFalse(unit.path.exists())


class ManagerTests(ServiceTestCase):
    def test_platform_selection(self):
        runner = FakeRunner()
        self.assertIsInstance(service.manager(system="Darwin", runner=runner, uid=1), service.LaunchAgent)
        self.assertIsInstance(service.manager(system="Linux", runner=runner, uid=1, sleep=None), service.SystemdUnit)
        with self.assertRaises(service.ServiceError):
            service.manager(system="Windows")

    def test_default_directories(self):
        self.assertEqual(service.LaunchAgent(runner=FakeRunner()).path,
                         pathlib.Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist")
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "xdg")}):
            self.assertEqual(service.SystemdUnit(runner=FakeRunner()).path,
                             self.root / "xdg" / "systemd" / "user" / "coderunner.server.service")


if __name__ == "__main__":
    unittest.main()
