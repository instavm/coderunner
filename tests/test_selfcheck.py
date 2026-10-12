import os
import pathlib
import shutil
import tempfile
import unittest

from coderunner.core import paths
from coderunner.core.config import Config
from coderunner.safety import selfcheck


class SelfCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self._home = os.environ.get("CODERUNNER_HOME")
        os.environ["CODERUNNER_HOME"] = str(self.tmp)
        paths.ensure_dirs()

    def tearDown(self):
        if self._home is None:
            os.environ.pop("CODERUNNER_HOME", None)
        else:
            os.environ["CODERUNNER_HOME"] = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def failures(self, data):
        return {c["check"] for c in selfcheck.run(Config(data, path=self.tmp / "config.json"))
                if not c["ok"] and c["severity"] == "high"}

    def test_defaults_pass(self):
        self.assertEqual(self.failures({}), set())
        info = [c for c in selfcheck.run(Config({}, path=self.tmp / "config.json")) if c["severity"] == "info"]
        self.assertEqual([c["check"] for c in info], ["sandbox network goes through the egress proxy"])

    def test_what_exposes_the_owner_is_reported(self):
        self.assertEqual(self.failures({
            "gateway": {"admin_bind": "0.0.0.0", "sandbox_bind": "0.0.0.0"},
            "policy": {"rules": [{"kind": "browser.sensitive", "target": "*", "decision": "allow"}]},
            "models": {"base_url": "https://llm.example.com/v1"},
            "remote": {"provider": "tailscale", "owners": []},
        }), {"owner API on this machine only", "sandbox API not on every network",
             "committing steps still need your approval", "model runs on this machine",
             "remote access limited to your own login"})

    def test_readable_credentials_are_reported(self):
        token = paths.assistant_dir() / "admin-token"
        token.write_text("secret")
        os.chmod(token, 0o644)
        self.assertEqual(self.failures({}), {"credentials readable only by you"})
