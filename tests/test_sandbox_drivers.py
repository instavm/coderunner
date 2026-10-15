import json
import subprocess
import unittest
from unittest import mock

from coderunner.core.config import Config
from coderunner.sandbox import drivers


class FakeContainerCLI:
    """Apple's `container` command: the service can be stopped, and inspect reports state as 1.5 does."""

    def __init__(self, service=False, state="stopped"):
        self.service, self.state, self.calls = service, state, []

    def __call__(self, args, timeout=60):
        self.calls.append(args)
        rc, out = 0, ""
        if args[:3] == ["container", "system", "status"]:
            rc = 0 if self.service else 1
        elif args[:3] == ["container", "system", "start"]:
            self.service = True
        elif args[:2] == ["container", "inspect"]:
            if not self.service:
                rc = 1
            else:
                out = json.dumps([{"status": {"state": self.state, "networks": []},
                                   "configuration": {"networks": [{"network": "default"}]}}])
        elif args[:2] == ["container", "start"]:
            rc = 0 if self.service else 1
            self.state = "running" if self.service else self.state
        return subprocess.CompletedProcess(args, rc, out, "")


class AppleDriverTests(unittest.TestCase):
    def test_a_stopped_service_is_started_with_the_sandbox(self):
        cli = FakeContainerCLI(service=False)
        driver = drivers.AppleDriver(Config({}))
        with mock.patch.object(drivers, "_run", cli):
            self.assertEqual(driver.status(), "service-stopped")         # not "missing": nothing to reinstall
            self.assertTrue(driver.start())
            self.assertEqual(driver.status(), "running")
        self.assertIn(["container", "system", "start", "--disable-kernel-install", "--timeout", "60"], cli.calls)

    def test_state_from_a_status_object_or_a_string(self):
        driver = drivers.AppleDriver(Config({}))
        with mock.patch.object(drivers, "_run", FakeContainerCLI(service=True, state="stopped")):
            self.assertEqual(driver.status(), "stopped")
        older = subprocess.CompletedProcess([], 0, json.dumps([{"status": "running"}]), "")
        with mock.patch.object(drivers, "_run", return_value=older):
            self.assertEqual(driver.status(), "running")


if __name__ == "__main__":
    unittest.main()
