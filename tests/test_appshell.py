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
        self.assertEqual(re.findall(r'ipcMain\.(?:handle|on)\("([\w-]+)"', main), ["pick-folder"])
        self.assertEqual(re.findall(r'ipcRenderer\.\w+\("([\w-]+)"', preload), ["pick-folder"])
        self.assertEqual(re.findall(r'exposeInMainWorld\("(\w+)"', preload), ["coderunnerShell"])
        self.assertIn("if (asker !== ORIGIN) return null;", main)        # only the daemon's own page is answered

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
                mock.patch.object(appshell, "installed", return_value=binary):
            appshell.launch()
        argv, options = popen.call_args[0][0], popen.call_args[1]
        self.assertEqual(argv, [binary, str(appshell.app_dir())])
        self.assertEqual(options["env"]["CODERUNNER_HOME"], str(self.home))
        self.assertTrue(options["start_new_session"])
        with mock.patch.object(appshell, "app_dir", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "from a checkout"):
                appshell.launch()

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


if __name__ == "__main__":
    unittest.main()
