"""The assistant's computer kept up by the daemon itself (sandbox/keeper.py): started, given its desktop and its
Chrome, and repaired a step at a time, against a fake VM that is both the driver and the sandbox's server."""

import asyncio
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.sandbox import keeper as keeper_module
from coderunner.sandbox.keeper import DISK_FULL, FAILED, READY, Keeper, classify

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FakeVM:
    """Apple `container` and the sandbox server in one, with the faults seen on the owner's Mac.

    listed     what `container inspect` says (running, stopped, missing, service-stopped)
    stuck      listed as running, but exec, stop and start answer "container is not running"; only a restart of
               the container service clears it
    io         the VM's root disk returns I/O errors (the server then answers 409 for the desktop)
    installed  the desktop's packages are in the VM
    wedged     the desktop session will not come up again by itself (only ending it helps)
    """

    name = "apple"
    base_url = "http://coderunner.local:8222"

    def __init__(self, listed="running", stuck=False, io=False, installed=True, wedged=False, starts=True,
                 vm_disk_full=False):
        self.listed, self.stuck, self.io, self.installed, self.wedged = listed, stuck, io, installed, wedged
        self.starts, self.vm_disk_full = starts, vm_disk_full
        self.session = False
        self.chrome = False
        self.calls: list[str] = []
        self.deleted = False

    # ---- driver ---------------------------------------------------------------

    def status(self):
        return self.listed

    def start(self):
        self.calls.append("start")
        if self.stuck or not self.starts:
            return False
        self.listed = "running"
        return True

    def stop(self):
        self.calls.append("stop")
        if self.stuck:
            return False
        self.listed, self.session, self.chrome = "stopped", False, False
        return True

    def restart_service(self):
        self.calls.append("service")
        self.stuck = self.io = False       # the VM's disk was intact after the service came back
        self.listed, self.session, self.chrome = "stopped", False, False
        return True

    def delete(self):
        self.deleted = True
        return True

    def run_in(self, command, timeout=30, stdin=None):
        script = command[-1]
        if self.stuck or self.listed != "running":
            return 1, "Error: container is not running"
        if script == keeper_module.END_SESSION:
            self.calls.append("session")
            self.session = self.chrome = self.wedged = False
            return 0, ""
        if self.vm_disk_full:
            return 3, "touch: cannot touch '/var/tmp/.coderunner-probe': No space left on device"
        if self.io:
            return 3, "touch: cannot touch '/var/tmp/.coderunner-probe': Input/output error"
        return 0, "desktop-installed\n" if self.installed else "desktop-missing\n"

    # ---- the sandbox's server -------------------------------------------------

    async def health(self):
        return self.listed == "running"

    async def desktop(self, start=False, chrome=False):
        if self.listed != "running":
            raise httpx.ConnectError("connection refused")
        if self.io or not self.installed:
            return {"available": False, "running": False, "error": "This sandbox has no desktop yet."}
        if start and not self.wedged:
            self.session = True
            if chrome:
                self.calls.append("chrome")
                self.chrome = True
        return {"available": True, "running": self.session, "port": 5900, "password": "pw", "chrome": self.chrome}


def make(vm, mode="computer", free_gb=50.0, installer=None, **kwargs):
    tmp = pathlib.Path(tempfile.mkdtemp())
    config = Config({"browser": {"mode": mode}}, path=tmp / "config.json")
    audit = AuditLog(tmp / "audit.jsonl")
    waits: list[float] = []

    async def sleep(seconds):
        waits.append(seconds)
        await asyncio.sleep(0)

    async def vnc(host, port):
        return vm.session

    keeper = Keeper(config, vm, vm, audit, installer=installer, disk_free_gb=lambda: free_gb, vnc_check=vnc,
                    sleep=sleep, **kwargs)
    keeper.waits = waits
    return keeper


def events(keeper, name=None):
    rows = keeper.audit.tail(200)
    return [r for r in rows if name is None or r["event"] == name]


class AutomaticTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stopped_vm_is_started_and_its_screen_and_chrome_come_up(self):
        vm = FakeVM(listed="stopped")
        keeper = make(vm)
        self.assertEqual(keeper.public(), {"state": "working", "message": "Getting the computer ready…"})
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual((keeper.state, vm.session, vm.chrome), (READY, True, True))
        self.assertEqual(vm.calls, ["start", "chrome"])          # one Chrome, the one the tasks use
        self.assertEqual([r["was"] for r in events(keeper, "sandbox.started")], ["stopped"])
        self.assertEqual(keeper.public(), {"state": "ready", "message": ""})

    async def test_chrome_is_only_started_for_the_computer_browser_and_its_failure_is_not_a_repair(self):
        vm = FakeVM()
        keeper = make(vm, mode="chrome")
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertFalse(vm.chrome)                              # the owner's own browser: no second Chrome
        vm = FakeVM()
        keeper = make(vm)
        real = vm.desktop

        async def no_chrome(start=False, chrome=False):
            if chrome:
                raise httpx.ReadTimeout("chrome hung")
            return await real(start=start)
        vm.desktop = no_chrome
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual((vm.calls, len(events(keeper, "computer.chrome_failed"))), ([], 1))

    async def test_a_nearly_full_mac_disk_is_said_and_nothing_is_tried(self):
        vm = FakeVM(listed="stopped")
        keeper = make(vm, free_gb=0.8)
        self.assertFalse(await keeper.ensure(desktop=True))
        self.assertEqual(keeper.state, DISK_FULL)
        self.assertIn("disk is almost full", keeper.public()["message"])
        self.assertEqual(vm.calls, [])
        self.assertEqual(events(keeper, "computer.disk_full")[0]["where"], "host")
        self.assertIsNone(keeper.kick())                         # looked at again only after a while

    async def test_no_sandbox_installed_is_left_alone(self):
        vm = FakeVM(listed="missing")
        keeper = make(vm)
        self.assertFalse(await keeper.ensure(desktop=True))
        self.assertEqual((keeper.public()["state"], vm.calls), ("absent", []))

    async def test_a_vm_without_its_desktop_gets_it_installed_once_and_the_browser_moves_there(self):
        vm = FakeVM(installed=False)
        installs = []

        def installer(driver, say):
            installs.append(driver)
            say("Desktop installed")
            vm.installed = True
            return True
        moved = []
        keeper = make(vm, mode="dedicated", installer=installer, on_mode=lambda: moved.append(True))
        keeper.config.path = pathlib.Path(tempfile.mkdtemp()) / "config.json"
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual((len(installs), vm.calls[:2], moved), (1, ["stop", "start"], [True]))
        self.assertEqual(keeper.config.get("browser.mode"), "computer")
        self.assertTrue(events(keeper, "computer.installed")[0]["ok"])
        # The owner's own Chrome stays their browser.
        vm, installs[:] = FakeVM(installed=False), []
        keeper = make(vm, mode="chrome", installer=installer)
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(keeper.config.get("browser.mode"), "chrome")

    async def test_a_failing_install_is_not_repeated_within_an_attempt_and_ends_in_one_plain_sentence(self):
        vm = FakeVM(installed=False)
        installs = []
        keeper = make(vm, installer=lambda driver, say: installs.append(1) or say("E: network down") or False)
        self.assertFalse(await keeper.ensure(desktop=True))
        self.assertEqual(keeper.state, FAILED)
        self.assertEqual(len(installs), 1)
        self.assertEqual(keeper.waits[:4], [5, 15, 45, 120])     # tried again quietly, waiting longer each time
        message = keeper.public()["message"]
        self.assertEqual(message, "The assistant's computer could not be started. It will try again shortly.")
        self.assertNotRegex(message, r"coderunner|E: |Error")

    async def test_the_package_carries_the_files_the_install_needs(self):
        from coderunner.sandbox import computer
        folder = computer.files_dir()
        self.assertEqual(folder, ROOT / "coderunner" / "sandbox" / "vm")
        for name, _ in computer.FILES:
            self.assertEqual((folder / name).read_bytes(), (ROOT / name).read_bytes(), name)
        self.assertEqual((folder / "desktop-setup.sh").read_bytes(), (ROOT / "desktop-setup.sh").read_bytes())
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn('"coderunner.sandbox" = ["vm/*"]', pyproject)

    async def test_a_task_that_only_needs_the_vm_does_not_wait_for_the_desktop(self):
        vm = FakeVM(listed="stopped")
        gate = asyncio.Event()
        real = vm.desktop

        async def slow_desktop(start=False, chrome=False):
            await gate.wait()
            return await real(start=start, chrome=chrome)
        vm.desktop = slow_desktop
        keeper = make(vm)
        self.assertTrue(await asyncio.wait_for(keeper.ensure(desktop=False), 5))
        self.assertNotEqual(keeper.state, READY)
        gate.set()
        self.assertTrue(await keeper.ensure(desktop=True))


class RepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_wedged_desktop_session_is_restarted_inside_the_vm(self):
        vm = FakeVM(wedged=True)
        keeper = make(vm)
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(vm.calls[0], "session")
        repairs = events(keeper, "computer.repair")
        self.assertEqual([(r["step"], r["problem"], r["ok"]) for r in repairs], [("session", "session", True)])
        self.assertEqual(events(keeper, "computer.repaired")[0]["step"], "session")

    async def test_a_screen_that_does_not_answer_restarts_the_session(self):
        vm = FakeVM()
        keeper = make(vm)
        answered = iter([False])

        async def vnc(host, port):
            return next(answered, True)
        keeper._vnc_check = vnc
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(events(keeper, "computer.repair")[0]["problem"], "screen")

    async def test_a_server_that_stops_answering_gets_the_vm_stopped_and_started(self):
        vm = FakeVM()
        keeper = make(vm)
        answers = iter([False] * 30)
        real = vm.health

        async def health():
            return next(answers, None) is None and await real()
        vm.health = health
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(vm.calls[:2], ["stop", "start"])
        self.assertEqual([(r["step"], r["problem"]) for r in events(keeper, "computer.repair")],
                         [("container", "unhealthy")])

    async def test_the_full_disk_case_a_vm_listed_running_that_takes_no_commands_gets_the_service_restarted(self):
        # 2026-11-25: the Mac's disk filled, the VM's disk returned I/O errors, `container list` said running while
        # exec, stop and kill failed, and /v1/desktop answered 409. Stopping and starting the service fixed it.
        vm = FakeVM(stuck=True, io=True)
        keeper = make(vm)
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(vm.calls, ["stop", "start", "service", "start", "chrome"])
        self.assertEqual([(r["step"], r["problem"], r["ok"]) for r in events(keeper, "computer.repair")],
                         [("container", "stuck", False), ("service", "stuck", True)])
        self.assertFalse(vm.deleted)

    async def test_io_errors_inside_a_vm_that_takes_commands_restart_it(self):
        vm = FakeVM(io=True)
        real_start = vm.start

        def start():
            vm.io = False                      # a fresh boot remounts the disk
            return real_start()
        vm.start = start
        keeper = make(vm)
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertEqual(events(keeper, "computer.repair")[0]["problem"], "io")

    async def test_the_vms_own_full_disk_is_said_not_repaired(self):
        vm = FakeVM(io=True, vm_disk_full=True)
        keeper = make(vm)
        self.assertFalse(await keeper.ensure(desktop=True))
        self.assertEqual(keeper.public(), {"state": "disk_full",
                                           "message": "The assistant's computer has run out of disk space."})
        self.assertEqual(vm.calls, [])

    async def test_repairs_do_not_loop_for_ever_and_never_delete_the_vm(self):
        vm = FakeVM(stuck=True, io=True)
        vm.restart_service = lambda: vm.calls.append("service") or False     # the service will not come back
        clock = [1000.0]
        keeper = make(vm, clock=lambda: clock[0])
        self.assertFalse(await keeper.ensure(desktop=True, timeout=30))
        self.assertEqual(keeper.state, FAILED)
        self.assertEqual(vm.calls.count("service"), 1)          # at most once an hour
        self.assertEqual(len(events(keeper, "computer.repair_skipped")), 4)
        self.assertFalse(vm.deleted)
        self.assertIsNone(keeper.kick())                        # resting: asking again tries nothing
        clock[0] += Keeper.REST_S + 1
        self.assertIsNotNone(keeper.kick())
        await keeper._task
        self.assertEqual(events(keeper, "computer.unavailable")[-1]["rest_s"], 2 * Keeper.REST_S)

    async def test_a_driver_that_raises_counts_as_a_failed_step(self):
        vm = FakeVM(stuck=True, io=True)

        def boom():
            raise subprocess.TimeoutExpired("container", 120)
        vm.stop = boom
        keeper = make(vm)
        self.assertTrue(await keeper.ensure(desktop=True))
        self.assertIn("service", vm.calls)
        self.assertTrue(events(keeper, "computer.call_failed"))

    def test_what_the_probe_inside_the_vm_finds(self):
        self.assertEqual(classify(0, "desktop-installed\n"), "installed")
        self.assertEqual(classify(0, "desktop-missing\n"), "missing")
        self.assertEqual(classify(3, ""), "io")
        self.assertEqual(classify(1, "rm: Input/output error"), "io")
        self.assertEqual(classify(1, "Error: container is not running"), "stuck")
        self.assertEqual(classify(None, "timed out"), "stuck")
        self.assertEqual(classify(3, "No space left on device"), "vm_disk_full")


class BackdropTests(unittest.TestCase):
    def test_a_stock_desktop_gets_the_gradient_once_and_an_owners_backdrop_is_kept(self):
        sys.path.insert(0, str(ROOT))
        try:
            import desktop
        finally:
            sys.path.remove(str(ROOT))
        xml = desktop.backdrop_xml()
        for n in range(4):
            self.assertIn(f'name="workspace{n}"', xml)
        self.assertIn('<property name="monitorVNC-0" type="empty">', xml)
        self.assertIn('<property name="image-style" type="int" value="0"/>', xml)
        self.assertIn('<property name="color-style" type="int" value="2"/>', xml)
        self.assertRegex(xml, r'name="rgba1" type="array">\s*<value type="double" value="0.059"/>\s*'
                              r'<value type="double" value="0.090"/>\s*<value type="double" value="0.200"/>\s*'
                              r'<value type="double" value="1"/>')
        self.assertRegex(xml, r'name="rgba2" type="array">\s*<value type="double" value="0.200"/>\s*'
                              r'<value type="double" value="0.380"/>\s*<value type="double" value="0.870"/>')
        home = pathlib.Path(tempfile.mkdtemp())
        settings = home / ".config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml"
        self.assertTrue(desktop.apply_backdrop(home))                         # a desktop with no settings yet
        self.assertEqual(settings.read_text(), xml)
        settings.write_text("<channel>xfce-shapes.svg</channel>")
        self.assertFalse(desktop.apply_backdrop(home))                        # once only
        home = pathlib.Path(tempfile.mkdtemp())
        settings = home / ".config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml"
        settings.parent.mkdir(parents=True)
        settings.write_text('<property name="last-image" value="/usr/share/backgrounds/xfce/xfce-shapes.svg"/>')
        self.assertTrue(desktop.apply_backdrop(home))                         # the stock backdrop is replaced
        home = pathlib.Path(tempfile.mkdtemp())
        settings = home / ".config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml"
        settings.parent.mkdir(parents=True)
        settings.write_text('<property name="last-image" value="/home/box/Pictures/beach.jpg"/>')
        self.assertFalse(desktop.apply_backdrop(home))                        # the owner's own is kept
        self.assertIn("beach.jpg", settings.read_text())

    def test_the_installer_writes_the_same_backdrop(self):
        sys.path.insert(0, str(ROOT))
        try:
            import desktop
        finally:
            sys.path.remove(str(ROOT))
        script = (ROOT / "desktop-setup.sh").read_text()
        part = script[script.index('desktop_xml="'):script.index("\nfi\n", script.index('desktop_xml="')) + 4]
        home = pathlib.Path(tempfile.mkdtemp())
        (home / ".config/xfce4/xfconf/xfce-perchannel-xml").mkdir(parents=True)
        subprocess.run(["bash", "-c", f'set -euo pipefail\nhome="{home}"\n{part}'], check=True)
        written = (home / ".config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml").read_text()
        self.assertEqual(written, desktop.backdrop_xml())
        self.assertTrue((home / ".config/xfce4" / desktop.BACKDROP_MARK).exists())


class PersistenceTests(unittest.TestCase):
    """entrypoint.sh keeps the Chrome profile, Downloads and the workspace on the data volume (/data)."""

    def run_entry(self, data, home, workspace):
        script = (ROOT / "entrypoint.sh").read_text()
        part = script[:script.index("# Lite profile")]
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "CODERUNNER_DATA": str(data),
               "CODERUNNER_DESKTOP_HOME": str(home), "CODERUNNER_WORKSPACE": str(workspace),
               "CODERUNNER_DESKTOP_USER": "nobody-here"}
        subprocess.run(["bash", "-c", part], check=True, env=env)

    def test_what_the_owner_would_miss_moves_to_the_data_volume_and_stays_there(self):
        root = pathlib.Path(tempfile.mkdtemp())
        data, home, workspace = root / "data", root / "home", root / "workspace"
        data.mkdir()
        profile = home / ".config" / "coderunner-chrome"
        profile.mkdir(parents=True)
        (profile / "Cookies").write_text("signed in")
        (home / "Downloads").mkdir()
        (home / "Downloads" / "report.pdf").write_text("pdf")
        workspace.mkdir()
        (workspace / "notes.txt").write_text("notes")
        self.run_entry(data, home, workspace)
        self.assertTrue(profile.is_symlink())
        self.assertEqual((data / "chrome-profile" / "Cookies").read_text(), "signed in")
        self.assertEqual((home / "Downloads" / "report.pdf").read_text(), "pdf")
        self.assertEqual((data / "workspace" / "notes.txt").read_text(), "notes")
        self.run_entry(data, home, workspace)                                 # every start: nothing changes
        self.assertEqual((profile / "Cookies").read_text(), "signed in")
        # The VM created again from a new image: fresh, empty folders; the data volume brings everything back.
        for path in (profile, home / "Downloads", workspace):
            path.unlink()
        profile.mkdir()
        workspace.mkdir()
        self.run_entry(data, home, workspace)
        self.assertEqual((profile / "Cookies").read_text(), "signed in")
        self.assertEqual((workspace / "notes.txt").read_text(), "notes")
        self.assertEqual((home / "Downloads" / "report.pdf").read_text(), "pdf")

    def test_without_a_data_volume_nothing_changes(self):
        root = pathlib.Path(tempfile.mkdtemp())
        home, workspace = root / "home", root / "workspace"
        (home / "Downloads").mkdir(parents=True)
        workspace.mkdir()
        script = (ROOT / "entrypoint.sh").read_text()
        part = script[:script.index("# Lite profile")]
        env = {"PATH": "/usr/bin:/bin", "CODERUNNER_DESKTOP_HOME": str(home), "CODERUNNER_WORKSPACE": str(workspace)}
        # /data is neither given nor mounted here.
        subprocess.run(["bash", "-c", part.replace('DATA="${CODERUNNER_DATA:-/data}"', f'DATA="{root}/none"')],
                       check=True, env=env)
        self.assertFalse((home / "Downloads").is_symlink())
        self.assertFalse(workspace.is_symlink())

    def test_a_reset_carries_the_profile_downloads_and_workspace_of_a_vm_from_before_the_volume(self):
        from coderunner.sandbox import computer
        root = pathlib.Path(tempfile.mkdtemp())
        old, new, shims = root / "old", root / "new", root / "bin"
        shims.mkdir()
        (shims / "getent").write_text("#!/bin/sh\necho 'box:x:1000:1000::/home/box:/bin/bash'\n")
        (shims / "getent").chmod(0o755)
        profile = old / "home/box/.config/coderunner-chrome"
        (profile / "Default/Cache").mkdir(parents=True)
        (profile / "Default/Cookies").write_text("signed in")
        (profile / "Default/Cache/data_0").write_text("cache")
        (old / "home/box/Downloads").mkdir(parents=True)
        (old / "home/box/Downloads/report.pdf").write_text("pdf")
        (old / "workspace").mkdir()
        (old / "workspace/notes.txt").write_text("notes")
        # The new VM already links its folders into the data volume (entrypoint.sh).
        data = new / "data"
        for name in ("chrome-profile", "downloads", "workspace"):
            (data / name).mkdir(parents=True)
        (new / "home/box/.config").mkdir(parents=True)
        (new / "home/box/.config/coderunner-chrome").symlink_to(data / "chrome-profile")
        (new / "home/box/Downloads").symlink_to(data / "downloads")
        (new / "workspace").symlink_to(data / "workspace")

        class Driver:
            def __init__(self, at):
                self.at = at

            def exec_argv(self, command, tty):
                script = command[2].replace("cd / ", f"cd {self.at} ").replace("-C / ", f"-C {self.at} ")
                if "GNU" not in subprocess.run(["tar", "--version"], capture_output=True, text=True).stdout:
                    # The VM's tar is GNU's; bsdtar (macOS) extracts through a linked folder only with -P.
                    script = script.replace("--keep-directory-symlink", "-P")
                return ["env", f"PATH={shims}:/usr/bin:/bin", "sh", "-c", script]

        carried, said = root / "carry.tar", []
        self.assertTrue(computer.carry_out(Driver(old), carried, said.append), said)
        self.assertTrue(computer.carry_in(Driver(new), carried, said.append), said)
        self.assertEqual((data / "chrome-profile/Default/Cookies").read_text(), "signed in")
        self.assertFalse((data / "chrome-profile/Default/Cache").exists())          # caches are left behind
        self.assertEqual((data / "downloads/report.pdf").read_text(), "pdf")
        self.assertEqual((data / "workspace/notes.txt").read_text(), "notes")
        self.assertTrue((new / "workspace").is_symlink())
        self.assertFalse(carried.exists())
        # A VM already on the volume has nothing to carry.
        self.assertIsNone(computer.carry_out(Driver(new), carried, said.append))

    def test_reset_copies_out_before_deleting_and_back_after_recreating(self):
        import argparse
        import os
        from coderunner import cli
        from coderunner.sandbox import computer
        order = []

        class Driver:
            name = "apple"

            def status(self):
                return "running"

            def settings(self):
                return {}

            def delete(self):
                order.append("delete")
                return True

        def out(driver, target, say=print):
            order.append("out")
            target.write_bytes(b"tar")
            return True

        def back(driver, source, say=print):
            order.append("in")
            source.unlink()
            return True
        home = tempfile.mkdtemp()
        args = argparse.Namespace(action="reset", cpus=None, memory=None, network=None, mount=None, build=False,
                                  yes=True)
        with mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}), \
                mock.patch("coderunner.sandbox.driver_for", return_value=Driver()), \
                mock.patch.object(computer, "carry_out", out), mock.patch.object(computer, "carry_in", back), \
                mock.patch.object(cli.subprocess, "call", side_effect=lambda *a, **k: order.append("install") or 0), \
                mock.patch("builtins.print"):
            self.assertEqual(cli.cmd_sandbox(args, mock.Mock()), 0)
        self.assertEqual(order, ["out", "delete", "install", "in"])

    def test_the_installer_gives_the_vm_the_data_volume_and_the_desktop_image(self):
        install = (ROOT / "install.sh").read_text()
        self.assertEqual(len(re.findall(r'--volume "\$DATA_VOLUME:/data"', install)), 2)     # Apple and Docker
        self.assertNotIn("volume delete", install)
        self.assertIn("--build-arg DESKTOP=1", install)
        self.assertEqual(install.count("build $(build_args)"), 4)


if __name__ == "__main__":
    unittest.main()
