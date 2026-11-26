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

    def test_the_service_is_restarted_only_when_no_other_container_runs(self):
        driver = drivers.AppleDriver(Config({}))

        def listing(*ids, rc=0):
            cli = FakeContainerCLI(service=True)
            plain = cli.__call__

            def run(args, timeout=60):
                if args[:2] == ["container", "list"]:
                    cli.calls.append(args)
                    out = json.dumps([{"configuration": {"id": i}, "status": "running"} for i in ids])
                    return subprocess.CompletedProcess(args, rc, out, "")
                return plain(args, timeout)
            return cli, run

        cli, run = listing(driver.container, "postgres")      # the owner's own container would be stopped too
        with mock.patch.object(drivers, "_run", run):
            self.assertFalse(driver.restart_service())
        self.assertNotIn(["container", "system", "stop"], cli.calls)
        cli, run = listing("x", rc=1)                          # a list that cannot be read: left alone as well
        with mock.patch.object(drivers, "_run", run):
            self.assertFalse(driver.restart_service())
        cli, run = listing(driver.container)                   # ours alone: stopped and started
        with mock.patch.object(drivers, "_run", run):
            self.assertTrue(driver.restart_service())
        self.assertIn(["container", "system", "stop"], cli.calls)



class NoneDriverTests(unittest.TestCase):
    def test_search_mode_has_no_sandbox(self):
        """sandbox.driver none (search mode): a driver that is never available, so nothing is started."""
        from coderunner.core.config import Config
        from coderunner.sandbox.drivers import Driver, driver_for
        driver = driver_for(Config({"sandbox": {"driver": "none"}}, path=None))
        self.assertIs(type(driver), Driver)
        self.assertEqual((driver.name, driver.available(), driver.status()), ("none", False, "unavailable"))


class FakeDriver:
    name = "apple"

    def __init__(self, state="running"):
        self.state, self.deleted, self.stopped = state, False, False

    def status(self):
        return self.state

    def stop(self):
        self.stopped = True
        return True

    def delete(self):
        self.deleted = True
        return True

    def exec_argv(self, command, tty):
        return drivers.AppleDriver(Config({})).exec_argv(command, tty)

    def logs_argv(self, follow):
        return drivers.AppleDriver(Config({})).logs_argv(follow)

    def settings(self):
        return {"cpus": "4", "memory": "2g", "profile": "lite", "network": "none",
                "mounts": ["/p:/workspace/p:rw"], "home": "/h", "image": "coderunner-local:latest"}


class SandboxCommandTests(unittest.TestCase):
    """`coderunner sandbox` covers what the README used to do with `container` commands, on either runtime."""

    def run_cli(self, argv, driver, **patches):
        import contextlib
        import io
        import os
        import shutil
        import tempfile

        from coderunner import cli
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"CODERUNNER_HOME": tmp}), \
                mock.patch("coderunner.sandbox.driver_for", return_value=driver), contextlib.redirect_stdout(out):
            with contextlib.ExitStack() as stack:
                mocks = {name: stack.enter_context(mock.patch(target, **kw)) for name, (target, kw) in patches.items()}
                return cli.main(argv), out.getvalue(), mocks

    def test_exec_and_logs_use_the_runtime_command(self):
        apple, docker = drivers.AppleDriver(Config({})), drivers.DockerDriver(Config({}))
        self.assertEqual(apple.exec_argv(["/bin/bash"], tty=True), ["container", "exec", "-it", "coderunner", "/bin/bash"])
        self.assertEqual(docker.exec_argv(["ls"], tty=False), ["docker", "exec", "-i", "coderunner", "ls"])
        self.assertEqual(docker.logs_argv(follow=True), ["docker", "logs", "--follow", "coderunner"])
        cli = FakeContainerCLI(service=False)
        with mock.patch.object(drivers, "_run", cli):
            self.assertTrue(apple.delete())                     # starts the service first, or delete cannot run
        self.assertEqual(cli.calls[1][:3], ["container", "system", "start"])
        self.assertEqual(cli.calls[-1], ["container", "delete", "--force", "coderunner"])

    def test_shell_replaces_the_process_with_exec(self):
        code, _, mocks = self.run_cli(["sandbox", "shell", "python3", "-V"], FakeDriver(),
                                      execvp=("os.execvp", {}))
        mocks["execvp"].assert_called_once()
        self.assertEqual(mocks["execvp"].call_args.args[1][-2:], ["python3", "-V"])

    def test_a_missing_sandbox_points_to_the_installer(self):
        code, out, _ = self.run_cli(["sandbox", "stop"], FakeDriver("missing"))
        self.assertEqual(code, 1)
        self.assertIn("./install.sh", out)

    def test_reset_asks_then_deletes_and_runs_the_installer_with_the_options(self):
        driver = FakeDriver("stopped")
        code, out, _ = self.run_cli(["sandbox", "reset"], driver, ask=("builtins.input", {"return_value": "n"}))
        self.assertEqual((code, driver.deleted), (1, False))
        code, _, mocks = self.run_cli(["sandbox", "reset", "--yes", "--cpus", "8"], driver,
                                      call=("subprocess.call", {"return_value": 0}))
        self.assertEqual(code, 0)
        self.assertTrue(driver.deleted)
        argv = mocks["call"].call_args.args[0]
        self.assertTrue(argv[0].endswith("install.sh"))
        # the old network, mounts, profile and home are kept: a reset must not reopen a `none` sandbox
        self.assertEqual(argv[1:], ["--cpus", "8", "--memory", "2g", "--network", "none", "--profile", "lite",
                                    "--mount", "/p:/workspace/p:rw", "--no-assistant", "--image", "coderunner-local:latest"])
        self.assertEqual(mocks["call"].call_args.kwargs["env"]["CODERUNNER_HOME"], "/h")

    def test_a_build_without_space_deletes_nothing(self):
        import collections
        driver = FakeDriver("running")
        code, out, _ = self.run_cli(["sandbox", "reset", "--yes", "--build"], driver, disk=(
            "shutil.disk_usage", {"return_value": collections.namedtuple("u", "total used free")(0, 0, 4e9)}))
        self.assertEqual((code, driver.deleted), (1, False))
        self.assertIn("Nothing was deleted", out)
        # The old sandbox's disk is deleted before the build, so it counts toward the space a build needs.
        driver = FakeDriver("running")
        driver.disk_bytes = lambda: 14e9
        code, out, mocks = self.run_cli(["sandbox", "reset", "--yes", "--build"], driver, disk=(
            "shutil.disk_usage", {"return_value": collections.namedtuple("u", "total used free")(0, 0, 7e9)}),
            call=("subprocess.call", {"return_value": 0}), carry=("coderunner.sandbox.computer.carry_out", {}))
        self.assertEqual((code, driver.deleted), (0, True))
        self.assertIn("--build", mocks["call"].call_args.args[0])

    def test_settings_are_read_back_from_inspect(self):
        info = [{"configuration": {
            "resources": {"cpus": 4, "memoryInBytes": 2 << 30}, "image": {"reference": "coderunner-local:latest"},
            "initProcess": {"environment": ["CODERUNNER_PROFILE=lite", "PATH=/bin"]},
            "networks": [{"network": "coderunner-offline"}],
            "mounts": [{"source": "/h/assets/outputs", "destination": "/app/uploads/outputs", "options": []},
                       {"source": "/src", "destination": "/workspace/src", "options": ["ro"]},
                       {"source": "/out", "destination": "/workspace/out", "options": []}]}}]
        ok = subprocess.CompletedProcess([], 0, json.dumps(info), "")
        with mock.patch.object(drivers, "_run", return_value=ok):
            got = drivers.AppleDriver(Config({})).settings()
        self.assertEqual(got, {"cpus": "4", "memory": "2g", "profile": "lite", "network": "none", "home": "/h",
                               "image": "coderunner-local:latest",
                               "mounts": ["/src:/workspace/src", "/out:/workspace/out:rw"]})

if __name__ == "__main__":
    unittest.main()
