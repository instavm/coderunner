"""Unit tests for the sandbox server: request validation, the browser tool's URL guard, skill archives, session
ids and the health check. They import server.py outside the image (CODERUNNER_SHARED_DIR points at a temporary
folder; the Jupyter URL at a port where nothing listens), with the image's requirements installed."""

import asyncio
import os
import pathlib
import shutil
import tempfile
import unittest
import zipfile
from unittest import mock

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHARED = pathlib.Path(tempfile.mkdtemp(prefix="coderunner-test-"))
os.environ["CODERUNNER_SHARED_DIR"] = str(SHARED)
os.environ["CODERUNNER_JUPYTER_URL"] = "http://127.0.0.1:1"        # nothing listens here
os.environ.setdefault("CODERUNNER_API_TOKEN", "fixture-credential-not-for-use-123456789")
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
import server  # noqa: E402


class HostOriginTests(unittest.TestCase):
    def test_host_header_parsing(self):
        hostname = server._header_hostname
        self.assertEqual(hostname("coderunner.local:8222"), "coderunner.local")
        self.assertEqual(hostname("127.0.0.1"), "127.0.0.1")
        self.assertEqual(hostname("Localhost:8222"), "localhost")
        self.assertIsNone(hostname("user:pw@localhost"))
        self.assertIsNone(hostname("localhost:8222/path"))
        self.assertIsNone(hostname("localhost:notaport"))
        self.assertEqual(hostname("http://localhost:3000", origin=True), "localhost")
        self.assertIsNone(hostname("file://localhost", origin=True))
        self.assertIsNone(hostname("http://evil.example/?x=1", origin=True))

    def test_foreign_host_or_origin_is_refused(self):
        scope = {"type": "http", "headers": [(b"host", b"coderunner.local:8222")]}
        self.assertTrue(server.HostOriginValidator._is_allowed(scope))
        scope["headers"] = [(b"host", b"evil.example")]
        self.assertFalse(server.HostOriginValidator._is_allowed(scope))
        scope["headers"] = [(b"host", b"localhost:8222"), (b"origin", b"http://evil.example")]
        self.assertFalse(server.HostOriginValidator._is_allowed(scope))
        scope["headers"] = [(b"host", b"localhost:8222"), (b"origin", b"http://localhost:5173")]
        self.assertTrue(server.HostOriginValidator._is_allowed(scope))


class BrowserUrlTests(unittest.TestCase):
    def test_only_public_http_urls_are_opened(self):
        guard = server.public_url
        self.assertIn("only http", guard("file:///etc/passwd"))
        self.assertIn("only http", guard("ftp://example.com/"))
        self.assertIn("no host", guard("http://"))
        for url in ("http://127.0.0.1:8888/api/kernels", "http://localhost/", "http://10.0.0.5/",
                    "http://169.254.169.254/latest/meta-data/", "http://192.168.64.1:8223/", "http://[::1]/",
                    "http://0.0.0.0/"):
            with self.subTest(url=url):
                self.assertIn("not a public address", guard(url))
        self.assertIn("not a public address", guard("http://100.64.1.2/"))    # shared space (CGNAT, tailnets)
        self.assertIn("not a public address", guard("http://224.0.0.1/"))
        self.assertIsNone(guard("https://93.184.216.34/"))          # a public address, no DNS needed
        self.assertIn("does not resolve", guard("http://no-such-host.invalid/"))
        # The same decision with addresses the caller resolved, as every request the page makes is checked.
        self.assertIn("not a public address", guard("http://example.com/", addresses={"93.184.216.34", "10.0.0.1"}))
        self.assertIsNone(guard("http://example.com/", addresses={"93.184.216.34"}))
        self.assertIsNone(asyncio.run(server.checked_url("https://93.184.216.34/")))
        self.assertIn("not a public address", asyncio.run(server.checked_url("http://127.0.0.1:8888/")))


class SkillArchiveTests(unittest.TestCase):
    def setUp(self):
        server.USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, server.USER_SKILLS_DIR, True)

    def archive(self, name, members):
        path = server.USER_SKILLS_DIR / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, data in members:
                archive.writestr(member, data)
        return path

    def test_archives_cannot_escape_or_explode(self):
        slip = self.archive("slip.zip", [("../outside.txt", "x")])
        with self.assertRaises(ValueError):
            server._extract_skill_archive(slip)
        self.assertFalse((server.USER_SKILLS_DIR.parent / "outside.txt").exists())
        with mock.patch.object(server, "MAX_SKILL_ARCHIVE_BYTES", 10):
            big = self.archive("big.zip", [("skill/SKILL.md", "x" * 100)])
            with self.assertRaises(ValueError):
                server._extract_skill_archive(big)
        good = self.archive("good.zip", [("my-skill/SKILL.md", "---\nname: my-skill\ndescription: Does a thing\n---\n")])
        server._extract_skill_archive(good)
        self.assertTrue((server.USER_SKILLS_DIR / "my-skill" / "SKILL.md").is_file())

    def test_a_bad_archive_is_set_aside_and_listing_goes_on(self):
        (server.USER_SKILLS_DIR / "broken.zip").write_bytes(b"not a zip")
        locked = self.archive("locked.zip", [("locked-skill/SKILL.md", "x")])
        with mock.patch.object(server, "_extract_skill_archive", side_effect=RuntimeError("File is encrypted")):
            listing = asyncio.run(server.list_skills())
        self.assertTrue((server.USER_SKILLS_DIR / "locked.zip.bad").exists())      # a password-protected zip
        self.assertFalse(locked.exists())
        self.archive("ok.zip", [("ok-skill/SKILL.md", "---\nname: ok-skill\ndescription: Fine\n---\n")])
        listing = asyncio.run(server.list_skills())
        self.assertIn("ok-skill", listing)
        self.assertTrue((server.USER_SKILLS_DIR / "broken.zip.bad").exists())
        self.assertFalse((server.USER_SKILLS_DIR / "ok.zip").exists())


class SessionIdTests(unittest.TestCase):
    def test_session_ids_are_validated_before_a_kernel_is_taken(self):
        manager = server.PythonSessionManager()
        with mock.patch.object(server.kernel_pool, "initialize", mock.AsyncMock()), \
                mock.patch.object(server.kernel_pool, "get_available_kernel", mock.AsyncMock(return_value="k1")) as taken:
            for bad in ("-leading", "has space", "x" * 65, "../etc", 42):      # "" means "pick one for me"
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    asyncio.run(manager.start(bad))
            taken.assert_not_awaited()
            info = asyncio.run(manager.start("job.1"))
            self.assertEqual(info["session_id"], "job.1")
            with self.assertRaises(server.SessionConflictError):
                asyncio.run(manager.start("job.1"))


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_degraded_without_jupyter_and_needs_no_token(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8222") as client:
            response = await client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["jupyter"], False)
            # Every other route needs the token, and a foreign host is refused before anything else.
            self.assertEqual((await client.post("/execute", json={"code": "1"})).status_code, 401)
            self.assertEqual((await client.get("/health", headers={"Host": "evil.example"})).status_code, 403)


if __name__ == "__main__":
    unittest.main()
