"""The assistant's computer kept small on the Mac: the VM made again past its disk budget (sandbox/keeper.py) only
when idle, once a day and with room to spare; Chrome's cache capped; caches cleared at each start (entrypoint.sh);
build leftovers removed (install.sh); and the measurements the budget rests on (sandbox/drivers.py)."""

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from coderunner.core.config import Config
from coderunner.sandbox import drivers, reset
from coderunner.sandbox.keeper import READY, TIDIED, Keeper
from test_keeper import FakeVM, events, make

ROOT = pathlib.Path(__file__).resolve().parent.parent
GB = 10 ** 9


class SizedVM(FakeVM):
    """The fake VM with a disk on the Mac that grows, and an image of a known size."""

    def __init__(self, disk_gb=12.0, image_gb=6.0, **kwargs):
        super().__init__(**kwargs)
        self.disk_gb, self.image_gb = disk_gb, image_gb

    def disk_bytes(self):
        return int(self.disk_gb * GB)

    def image_bytes(self, image):
        return int(self.image_gb * GB) if image == "coderunner-local:latest" else None

    def settings(self):
        return {"image": "coderunner-local:latest"}


class Budget:
    """What the daemon tells the keeper: a task running, the owner at the Mac, and the clock."""

    def __init__(self):
        self.busy, self.away, self.now = False, True, 1_000_000.0
        self.made: list = []


def budgeted(vm, free_gb=50.0, budget=8, tidy_file=None, state=None, ok=True):
    state = state or Budget()

    def recreate(driver):
        state.made.append(driver)
        vm.disk_gb = vm.image_gb + 0.5          # the VM as made again: its image and a fresh desktop
        return ok

    tidy_file = tidy_file or pathlib.Path(tempfile.mkdtemp()) / "sandbox-tidy.json"
    keeper = make(vm, free_gb=free_gb, busy=lambda: state.busy, owner_away=lambda: state.away,
                  recreate=recreate, tidy_file=tidy_file, wall=lambda: state.now)
    keeper.config.set("sandbox.disk_budget_gb", budget)
    keeper.want_desktop = False
    return keeper, state


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_within_the_budget_nothing_happens(self):
        keeper, state = budgeted(SizedVM(disk_gb=7.5))
        self.assertEqual(await keeper.check_disk(), "within")
        self.assertEqual(state.made, [])
        self.assertEqual(events(keeper, "sandbox.tidy_skipped"), [])

    async def test_past_the_budget_and_idle_the_vm_is_made_again_audited_and_said_once(self):
        vm = SizedVM(disk_gb=14.0)
        keeper, state = budgeted(vm)
        self.assertEqual(await keeper.check_disk(), "tidying")
        self.assertEqual(keeper.public()["message"], "Tidying the assistant's computer…")
        self.assertEqual(await keeper._task, READY)
        self.assertEqual(state.made, [vm])
        self.assertFalse(vm.deleted)                         # the keeper deletes nothing itself: reset does
        tidy = events(keeper, "sandbox.tidy")
        self.assertEqual((tidy[0]["used_gb"], tidy[0]["budget_gb"]), (14.0, 8.0))
        done = events(keeper, "sandbox.tidied")[0]
        self.assertEqual((done["ok"], done["was_gb"], done["used_gb"]), (True, 14.0, 6.5))
        self.assertEqual(keeper.public_note()["text"], TIDIED)
        self.assertEqual(keeper.public(), {"state": "ready", "message": ""})
        state.now += 86400                                   # the note is kept for a day, then dropped
        self.assertIsNone(keeper.public_note())

    async def test_never_during_a_task_nor_while_the_owner_is_at_the_mac(self):
        keeper, state = budgeted(SizedVM(disk_gb=14.0))
        state.busy = True
        self.assertEqual(await keeper.check_disk(), "task")
        self.assertEqual(await keeper.check_disk(), "task")
        state.busy, state.away = False, False
        self.assertEqual(await keeper.check_disk(), "owner")
        self.assertEqual(state.made, [])
        # Audited when the reason changes, not every hour.
        self.assertEqual([r["reason"] for r in events(keeper, "sandbox.tidy_skipped")], ["task", "owner"])
        state.away = True
        self.assertEqual(await keeper.check_disk(), "tidying")
        await keeper._task
        self.assertEqual(len(state.made), 1)

    async def test_without_the_daemons_signals_it_never_acts(self):
        keeper = make(SizedVM(disk_gb=14.0), tidy_file=pathlib.Path(tempfile.mkdtemp()) / "t.json")
        self.assertEqual(await keeper.check_disk(), "unwired")

    async def test_at_most_once_a_day_even_across_a_restart(self):
        vm = SizedVM(disk_gb=14.0)
        keeper, state = budgeted(vm)
        await keeper.check_disk()
        await keeper._task
        vm.disk_gb = 14.0                                    # grown again at once
        self.assertEqual(await keeper.check_disk(), "today")
        again, _ = budgeted(vm, tidy_file=keeper._tidy_file, state=state)       # the daemon restarted
        self.assertEqual(await again.check_disk(), "today")
        state.now += 86400 + 1
        self.assertEqual(await again.check_disk(), "tidying")
        await again._task
        self.assertEqual(len(state.made), 2)

    async def test_a_fresh_vm_above_the_budget_is_not_made_again_for_nothing(self):
        vm = SizedVM(disk_gb=14.0, image_gb=8.4)             # an image that alone is past the budget
        keeper, state = budgeted(vm)
        await keeper.check_disk()
        await keeper._task                                   # made again: 8.9 GB
        state.now += 3600
        self.assertEqual(await keeper.check_disk(), "fresh")  # its size as made is kept as the start
        self.assertEqual(json.loads(keeper._tidy_file.read_text())["fresh_gb"], 8.9)
        state.now += 2 * 86400
        vm.disk_gb = 9.5
        self.assertEqual(await keeper.check_disk(), "fresh")  # grown by less than a GB
        vm.disk_gb = 10.2
        self.assertEqual(await keeper.check_disk(), "tidying")
        await keeper._task

    async def test_skipped_when_the_mac_lacks_the_room_to_make_it(self):
        keeper, state = budgeted(SizedVM(disk_gb=14.0, image_gb=6.0), free_gb=7.0)    # needs 6 + 2
        self.assertEqual(await keeper.check_disk(), "low_disk")
        skipped = events(keeper, "sandbox.tidy_skipped")[0]
        self.assertEqual((skipped["free_gb"], skipped["need_gb"]), (7.0, 8.0))
        self.assertEqual(state.made, [])

    async def test_a_budget_of_zero_turns_it_off_and_an_unmeasured_disk_is_left_alone(self):
        keeper, state = budgeted(SizedVM(disk_gb=40.0), budget=0)
        self.assertEqual(await keeper.check_disk(), "off")
        vm = SizedVM()
        vm.disk_bytes = lambda: None                         # Docker: not measured here
        keeper, state = budgeted(vm)
        self.assertEqual(await keeper.check_disk(), "unmeasured")
        self.assertIsNone(keeper.disk_gb())

    async def test_a_failed_recreate_is_audited_and_not_tried_again_that_day(self):
        vm = SizedVM(disk_gb=14.0)
        keeper, state = budgeted(vm, ok=False)
        await keeper.check_disk()
        await keeper._task
        self.assertFalse(events(keeper, "sandbox.tidied")[0]["ok"])
        self.assertIsNone(keeper.public_note())
        vm.disk_gb = 14.0
        self.assertEqual(await keeper.check_disk(), "today")

    async def test_a_task_asking_for_the_vm_meanwhile_waits_for_it(self):
        keeper, state = budgeted(SizedVM(disk_gb=14.0))
        await keeper.check_disk()
        self.assertIs(keeper.kick(), keeper._task)           # ensure() waits on the same attempt
        self.assertTrue(await keeper.ensure(desktop=False, timeout=5))

    def test_the_daemon_counts_the_owner_present_while_typing_in_control_or_at_the_mac(self):
        from coderunner.server.daemon import Daemon
        from coderunner.sandbox import keeper as keeper_module
        terminal = types.SimpleNamespace(typing=lambda: False)
        browser = types.SimpleNamespace(owner_control=False)
        fake = types.SimpleNamespace(svc=types.SimpleNamespace(terminal=terminal, browser=browser))
        with mock.patch.object(keeper_module, "mac_idle_seconds", return_value=1800.0):
            self.assertTrue(Daemon._owner_away(fake))
            browser.owner_control = True
            self.assertFalse(Daemon._owner_away(fake))
            browser.owner_control, terminal.typing = False, lambda: True
            self.assertFalse(Daemon._owner_away(fake))
        terminal.typing = lambda: False
        with mock.patch.object(keeper_module, "mac_idle_seconds", return_value=30.0):
            self.assertFalse(Daemon._owner_away(fake))
        with mock.patch.object(keeper_module, "mac_idle_seconds", return_value=None):
            self.assertFalse(Daemon._owner_away(fake))       # unknown counts as present
        tasks = types.SimpleNamespace(running={}, waiting=[])
        fake = types.SimpleNamespace(tasks=tasks, check=types.SimpleNamespace(state="idle"))
        self.assertFalse(Daemon._tasks_busy(fake))
        tasks.waiting.append(("queued",))
        self.assertTrue(Daemon._tasks_busy(fake))
        self.assertTrue(Daemon._tasks_busy(types.SimpleNamespace()))   # before the tasks exist: busy


class ShownTests(unittest.TestCase):
    def test_the_app_shows_the_size_in_one_line_and_the_note_once(self):
        from coderunner.server.daemon import Daemon
        keeper = mock.Mock(disk_gb=mock.Mock(return_value=6.4), budget_gb=mock.Mock(return_value=8.0))
        fake = types.SimpleNamespace(svc=types.SimpleNamespace(computer=keeper))
        self.assertEqual(Daemon._disk_status(fake), {"disk_gb": 6.4, "disk_budget_gb": 8.0})
        fake.svc.computer = mock.Mock(spec=Keeper)           # a fake keeper's mocks are not numbers
        self.assertEqual(Daemon._disk_status(fake), {"disk_gb": None, "disk_budget_gb": None})
        fake.svc.computer = None
        self.assertEqual(Daemon._disk_status(fake), {"disk_gb": None, "disk_budget_gb": None})
        app = (ROOT / "coderunner/server/web/app/app.js").read_text()
        self.assertIn('["Computer disk", diskLine(status.sandbox)]', app)
        self.assertIn("`${used} GB of ${budget} GB`", app)
        self.assertIn("if (info.note) noteOnce(info.note);", app)
        gateway = (ROOT / "coderunner/server/gateway.py").read_text()
        self.assertIn('shown["note"] = note', gateway)


class RecreateTests(unittest.TestCase):
    def test_the_keeper_never_deletes_without_an_installer_to_make_it_again(self):
        driver = mock.Mock(status=mock.Mock(return_value="running"), settings=mock.Mock(return_value={}))
        with mock.patch.object(reset, "installer", return_value=pathlib.Path("/nonexistent/install.sh")):
            self.assertFalse(reset.recreate_quietly(driver))
        driver.delete.assert_not_called()

    def test_the_keeper_makes_it_again_like_a_reset_with_the_installers_output_in_a_log(self):
        home = tempfile.mkdtemp()
        driver = mock.Mock(status=mock.Mock(return_value="stopped"),
                           settings=mock.Mock(return_value={"profile": "lite", "image": "coderunner-local:latest",
                                                            "network": "default"}))
        with mock.patch.dict(os.environ, {"CODERUNNER_HOME": home}), \
                mock.patch("subprocess.call", return_value=0) as call:
            self.assertTrue(reset.recreate_quietly(driver))
        driver.delete.assert_called_once()
        argv = call.call_args.args[0]
        self.assertTrue(argv[0].endswith("install.sh"))
        self.assertEqual(argv[1:], ["--network", "default", "--profile", "lite", "--no-assistant",
                                    "--image", "coderunner-local:latest"])
        self.assertIs(call.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(call.call_args.kwargs["stdout"].name.endswith("sandbox-recreate.log"))
        install = (ROOT / "install.sh").read_text()
        self.assertNotIn("volume delete", install)           # the data volume outlives every VM


class DriverMeasureTests(unittest.TestCase):
    def test_the_vm_disk_is_its_allocated_blocks_not_its_size_on_paper(self):
        root = pathlib.Path(tempfile.mkdtemp())
        disk = root / "containers" / "coderunner" / "rootfs.ext4"
        disk.parent.mkdir(parents=True)
        with open(disk, "wb") as f:
            f.truncate(50 * GB)                              # sparse, like the VM's disk
            f.write(b"x" * (4 << 20))
        driver = drivers.AppleDriver(Config({}))
        driver.root = root
        used = driver.disk_bytes()
        self.assertGreaterEqual(used, 4 << 20)
        self.assertLess(used, 64 << 20)
        driver.container = "other"
        self.assertIsNone(driver.disk_bytes())
        self.assertIsNone(drivers.DockerDriver(Config({})).disk_bytes())

    def test_the_image_size_is_its_unpacked_snapshot_or_an_estimate(self):
        root = pathlib.Path(tempfile.mkdtemp())
        arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(os.uname().machine, os.uname().machine)
        listing = [{"configuration": {"name": "coderunner-local:latest"},
                    "variants": [{"digest": "sha256:" + "a" * 64, "platform": {"os": "linux", "architecture": arch},
                                  "size": 2 * GB}]},
                   {"configuration": {"name": "docker.io/library/python@sha256:" + "b" * 64}, "variants": []}]
        ok = subprocess.CompletedProcess([], 0, json.dumps(listing), "")
        driver = drivers.AppleDriver(Config({}))
        driver.root = root
        with mock.patch.object(drivers, "_run", return_value=ok):
            self.assertEqual(driver.image_bytes("coderunner-local:latest"), 6 * GB)    # no snapshot: 3x
            snapshot = root / "snapshots" / ("a" * 64)
            snapshot.mkdir(parents=True)
            (snapshot / "snapshot").write_bytes(b"y" * (1 << 20))
            self.assertGreaterEqual(driver.image_bytes("docker.io/library/coderunner-local:latest"), 1 << 20)
            self.assertLess(driver.image_bytes("coderunner-local:latest"), 8 << 20)
            self.assertIsNone(driver.image_bytes("instavm/coderunner:v2"))


class ChromeCacheTests(unittest.TestCase):
    def test_chrome_runs_with_its_disk_cache_capped(self):
        sys.path.insert(0, str(ROOT))
        try:
            import desktop
        finally:
            sys.path.remove(str(ROOT))
        argv = desktop.chrome_argv(pathlib.Path("/home/box/.config/coderunner-chrome"))
        self.assertEqual(argv[0], desktop.CHROME)
        self.assertIn("--disk-cache-size=268435456", argv)
        self.assertIn("--user-data-dir=/home/box/.config/coderunner-chrome", argv)
        self.assertIn("--remote-debugging-pipe", argv)


class StartTidyTests(unittest.TestCase):
    """entrypoint.sh clears caches at each start, and only caches."""

    def write(self, path, text="x", age_days=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if age_days:
            old = path.stat().st_mtime - age_days * 86400
            os.utime(path, (old, old))
        return path

    def test_caches_go_and_the_owners_data_stays(self):
        root = pathlib.Path(tempfile.mkdtemp())
        home = root / "home" / "box"
        gone = [self.write(root / "tmp" / "coderunner-jobs" / "1" / "out"), self.write(root / "tmp" / "a.txt"),
                self.write(root / "root" / ".cache" / "pip" / "http" / "wheel"),
                self.write(root / "root" / ".npm" / "_cacache" / "index"),
                self.write(root / "var" / "lib" / "apt" / "lists" / "deb.debian.org_dists_bookworm_Release"),
                self.write(root / "var" / "cache" / "apt" / "archives" / "jq_1.6.deb"),
                self.write(root / "var" / "cache" / "apt" / "pkgcache.bin"),
                self.write(home / ".cache" / "coderunner-chrome" / "Default" / "Cache" / "Cache_Data" / "data_1"),
                self.write(home / ".cache" / "pip" / "wheel"), self.write(home / ".cache" / "Xvnc.log"),
                self.write(home / ".config" / "coderunner-chrome" / "Default" / "Cache" / "data_0"),
                self.write(home / ".config" / "coderunner-chrome" / "Default" / "Code Cache" / "js" / "a"),
                self.write(home / ".config" / "coderunner-chrome" / "Default" / "GPUCache" / "data_0")]
        kept = [self.write(home / ".config" / "coderunner-chrome" / "Default" / "Cookies", "signed in"),
                self.write(home / ".config" / "coderunner-chrome" / "Local State"),
                self.write(home / "Downloads" / "statement.pdf", age_days=90),   # old, but the owner's
                self.write(root / "workspace" / "notes.txt"),
                self.write(root / "root" / ".cache" / "ms-playwright" / "chromium" / "chrome"),
                self.write(root / "var" / "lib" / "dpkg" / "status")]
        script = (ROOT / "entrypoint.sh").read_text()
        part = script[:script.index("# Lite profile")]
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "CODERUNNER_TIDY_ROOT": str(root),
               "CODERUNNER_DESKTOP_HOME": str(home), "CODERUNNER_DATA": str(root / "no-data-volume")}
        subprocess.run(["bash", "-c", part], check=True, env=env)
        self.assertEqual([p for p in gone if p.exists()], [])
        self.assertEqual([p for p in kept if not p.exists()], [])
        self.assertTrue((root / "tmp").is_dir())
        self.assertEqual((home / ".config/coderunner-chrome/Default/Cookies").read_text(), "signed in")

    def test_it_runs_only_inside_the_sandbox(self):
        script = (ROOT / "entrypoint.sh").read_text()
        self.assertIn('if [ -n "$TIDY_ROOT" ] || [ -f /app/server.py ]; then', script)
        self.assertLess(script.index("tidy_caches \"$TIDY_ROOT\""), script.index("exec uvicorn"))


class BuildLeftoverTests(unittest.TestCase):
    def test_install_sh_parses(self):
        for name in ("install.sh", "entrypoint.sh", "desktop-setup.sh"):
            done = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{name}: {done.stderr}")

    def test_after_a_build_the_builder_dangling_and_base_images_go_and_tagged_ones_stay(self):
        install = (ROOT / "install.sh").read_text()
        functions = "\n".join(install[install.index(f"{name}() {{"):install.index("\n}\n", install.index(f"{name}() {{")) + 3]
                              for name in ("remove_builder", "base_image_digests", "remove_build_leftovers"))
        dockerfile = (ROOT / "Dockerfile").read_text()
        digests = [line.split("@")[1].split()[0] for line in dockerfile.splitlines() if line.startswith("FROM ")]
        self.assertEqual(len(digests), 2)
        tmp = pathlib.Path(tempfile.mkdtemp())
        log = tmp / "calls"
        listed = ["coderunner-local:latest", f"node@{digests[0]}", f"python@{digests[1]}",
                  "python@sha256:" + "c" * 64, "ghcr.io/apple/containerization/vminit:0.40.1"]
        shim = tmp / "container"
        shim.write_text("#!/bin/sh\necho \"$*\" >> " + str(log) + "\n"
                        "if [ \"$1 $2\" = \"image list\" ]; then printf '%s\\n' " +
                        " ".join(f"'{n}'" for n in listed) + "; fi\n")
        shim.chmod(0o755)
        subprocess.run(["bash", "-c", f'{functions}\nIMAGE=coderunner-local:latest\nremove_build_leftovers'],
                       check=True, env={"PATH": f"{tmp}:/usr/bin:/bin", "SCRIPT_DIR": str(ROOT)})
        calls = log.read_text().splitlines()
        self.assertIn("builder delete", calls)
        self.assertIn("image prune", calls)                  # dangling only: no --all
        deleted = [c.split()[-1] for c in calls if c.startswith("image delete")]
        self.assertEqual(deleted, [f"node@{digests[0]}", f"python@{digests[1]}"])
        self.assertEqual(install.count("remove_build_leftovers\n"), 2)       # after both Apple builds
        self.assertEqual(install.count("docker image prune -f"), 2)           # dangling only, after Docker builds
        self.assertNotIn("image prune --all", install)
        self.assertNotIn("image prune -a", install)


class ImageTests(unittest.TestCase):
    def test_the_dockerfile_leaves_out_caches_and_what_is_unused(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("pip install --no-cache-dir --require-hashes", dockerfile)
        self.assertIn("playwright install chromium", dockerfile)
        self.assertIn("path-exclude=/usr/share/doc/*", dockerfile)
        self.assertIn("path-include=/usr/share/doc/*/copyright", dockerfile)
        self.assertIn("default-jre-headless", dockerfile)
        self.assertNotIn("cargo", dockerfile)
        for line in dockerfile.splitlines():
            if "apt-get install" in line:
                self.assertIn("--no-install-recommends", line)
        setup = (ROOT / "desktop-setup.sh").read_text()
        self.assertIn("--no-install-recommends", setup)
        self.assertIn("rm -rf /var/lib/apt/lists/*", setup)
        # A home a reset made before the account existed is root's; the account takes its top folder back.
        self.assertIn('chown "$USER_NAME:" "$(getent passwd "$USER_NAME" | cut -d: -f6)"', setup)


if __name__ == "__main__":
    unittest.main()
