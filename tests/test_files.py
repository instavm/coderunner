import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import httpx

from coderunner.agent import RunContext, build_registry
from coderunner.audit import AuditLog
from coderunner.config import Config
from coderunner.files import Embedder, FileAccessError, FileIndex
from coderunner.store import Store
from fakes import Svc

CONCEPTS = ["invoice", "dentist", "holiday", "image", "audio"]


def fake_embed_server(capabilities=("embedding",)):
    """Ollama's /api/show and /api/embed: a vector per concept word, media get their own axis."""
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": list(capabilities)})
        vectors = []
        for item in body["input"]:
            seen.append(item)
            vector = [0.01] * len(CONCEPTS)
            if isinstance(item, dict):
                vector[CONCEPTS.index("image" if "image" in item else "audio")] += 1
            else:
                for i, word in enumerate(CONCEPTS):
                    vector[i] += item.lower().count(word)
            vectors.append(vector)
        return httpx.Response(200, json={"embeddings": vectors})
    return httpx.MockTransport(handler), seen


class FileIndexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.home = self.tmp / "crhome"
        self.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        self.env.start()
        self.docs = self.tmp / "Documents"
        (self.docs / "taxes").mkdir(parents=True)
        (self.docs / "taxes" / "invoice-2026.txt").write_text("Invoice 2026-41\nTotal due: 420 USD\n")
        (self.docs / "notes.md").write_text("Dentist on Friday at 3pm.\n" * 3)
        (self.docs / "code.py").write_text("def holiday_planner():\n    return 'holiday'\n")
        (self.docs / ".env").write_text("API_KEY=sk-secret")
        (self.docs / "server.pem").write_text("-----BEGIN PRIVATE KEY-----")
        (self.docs / ".hidden").mkdir()
        (self.docs / ".hidden" / "notes.txt").write_text("hidden invoice")
        (self.docs / "node_modules").mkdir()
        (self.docs / "node_modules" / "readme.md").write_text("invoice library")
        outside = self.tmp / "outside.txt"
        outside.write_text("private invoice elsewhere")
        (self.docs / "link.txt").symlink_to(outside)
        self.config = Config({"files": {"roots": [str(self.docs)]}}, path=self.tmp / "config.json")
        self.store = Store(":memory:")
        self.audit = AuditLog(self.tmp / "audit.jsonl")

    async def asyncTearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def index(self, embedder=None):
        return FileIndex(self.config, self.store, self.audit, embedder)

    async def test_only_shared_non_secret_files_are_indexed_and_readable(self):
        index = self.index()
        result = await index.index_once()
        self.assertEqual(result["indexed"], 3)
        paths = {pathlib.Path(r["path"]).name for r in self.store.query("SELECT path FROM files")}
        self.assertEqual(paths, {"invoice-2026.txt", "notes.md", "code.py"})
        self.assertEqual(index.resolve(str(self.docs / "notes.md")).name, "notes.md")
        for path in (self.docs / ".env", self.docs / "server.pem", self.docs / ".hidden" / "notes.txt",
                     self.docs / "link.txt", self.tmp / "outside.txt", self.home / "api-token", self.docs):
            with self.subTest(path=path.name), self.assertRaises(FileAccessError):
                index.resolve(str(path))

    async def test_folders_that_cannot_be_shared(self):
        index = self.index()
        for folder in ("/", str(pathlib.Path.home() / "Library"), str(self.home)):
            self.home.mkdir(parents=True, exist_ok=True)
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                index.add_root(folder)
        extra = self.tmp / "Music"
        extra.mkdir()
        self.assertEqual(index.add_root(str(extra)), str(extra.resolve()))
        self.assertIn(str(extra.resolve()), self.config.get("files.roots"))

    async def test_keyword_search_and_changes(self):
        index = self.index()
        await index.index_once()
        found = await index.search("Total due")
        self.assertEqual(pathlib.Path(found[0]["path"]).name, "invoice-2026.txt")
        self.assertIn("420 USD", found[0]["snippet"])
        # Changed and removed files are picked up on the next pass.
        (self.docs / "notes.md").write_text("Moved the dentist to Monday, and a total due reminder.")
        (self.docs / "taxes" / "invoice-2026.txt").unlink()
        result = await index.index_once()
        self.assertEqual((result["indexed"], result["removed"]), (1, 1))
        self.assertEqual([pathlib.Path(r["path"]).name for r in await index.search("total due")], ["notes.md"])
        # Unsharing a folder forgets what was indexed from it.
        self.assertEqual(index.remove_root(str(self.docs)), 2)
        self.assertEqual(await index.search("dentist"), [])

    async def test_meaning_search_with_embeddings(self):
        transport, seen = fake_embed_server()
        embedder = Embedder("http://ollama/v1", "embeddinggemma-2", transport=transport)
        index = self.index(embedder)
        await index.index_once()
        self.assertTrue(any(isinstance(item, str) and item.startswith("title: notes.md | text: ") for item in seen))
        found = await index.search("when is my dentist")
        self.assertEqual(pathlib.Path(found[0]["path"]).name, "notes.md")
        self.assertTrue(seen[-1].startswith("task: search result | query: "))
        await embedder.aclose()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for media")
    async def test_images_and_audio_are_embedded_as_media(self):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64",
                        "-frames:v", "1", str(self.docs / "red.png")], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                        str(self.docs / "tone.wav")], check=True)
        transport, seen = fake_embed_server(("embedding", "vision", "audio"))
        embedder = Embedder("http://ollama/v1", "embeddinggemma-2", transport=transport)
        index = self.index(embedder)
        await index.index_once()
        self.assertTrue(any(isinstance(item, dict) and "image" in item for item in seen))
        self.assertTrue(any(isinstance(item, dict) and "audio" in item for item in seen))
        found = await index.search("image", limit=20)
        self.assertIn("image", {r["kind"] for r in found})
        self.assertEqual(len({r["path"] for r in found}), len(found))      # one result per file
        # Without vision support the image is indexed by name only.
        transport, seen = fake_embed_server(("embedding",))
        text_only = Embedder("http://ollama/v1", "embeddinggemma-2", transport=transport)
        self.store.execute("DELETE FROM files")
        await self.index(text_only).index_once()
        self.assertFalse(any(isinstance(item, dict) for item in seen))
        await embedder.aclose()
        await text_only.aclose()

    async def test_tools_mark_the_task_as_holding_private_untrusted_data(self):
        svc = Svc()
        svc.files = self.index()
        await svc.files.index_once()
        registry = build_registry(svc)
        ctx = RunContext(conversation="f1")
        found = await registry.invoke("files_search", {"query": "invoice"}, ctx)
        self.assertIn("invoice-2026.txt", found)
        self.assertTrue(found.startswith('<untrusted source="files_search">'))
        self.assertEqual((ctx.content_tainted, ctx.private_sources), (True, {"files"}))
        text = await registry.invoke("file_read", {"path": str(self.docs / "taxes" / "invoice-2026.txt")}, ctx)
        self.assertIn("Total due: 420 USD", text)
        denied = await registry.invoke("file_read", {"path": str(self.docs / ".env")}, ctx)
        self.assertIn("never read", denied)
        self.assertNotIn("sk-secret", denied)


if __name__ == "__main__":
    unittest.main()
