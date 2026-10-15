import asyncio
import json
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import httpx

from coderunner.core import native
from coderunner.agent import RunContext, build_registry
from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.files import Embedder, FileAccessError, FileIndex, pack, unpack
from coderunner.files import power, search
from coderunner.files.extract import HELPER_SOURCE, passages, wordy
from coderunner.files.vectors import VectorIndex, code_bytes, signs
from coderunner.core.store import Store
from fakes import Svc

CONCEPTS = ["invoice", "dentist", "holiday", "image", "audio"]
SWIFT = native._compiler() is not None and os.uname().sysname == "Darwin"


def fake_embed_server(capabilities=("embedding",), digest="sha256:aaaa1111bbbb2222"):
    """Ollama's /api/show, /api/tags and /api/embed: a vector per concept word, media on their own axis."""
    seen, requests = [], []

    def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "embeddinggemma-2:latest", "digest": digest}]})
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": list(capabilities)})
        requests.append(len(body["input"]))
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
    transport = httpx.MockTransport(handler)
    transport.requests = requests
    return transport, seen


class FileIndexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.home = self.tmp / "crhome"
        self.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.home)})
        self.env.start()
        charging = mock.patch.object(power, "state", mock.AsyncMock(return_value=(False, 100, False)))
        charging.start()                # on the charger unless a test says otherwise (media waits on battery)
        self.addCleanup(charging.stop)
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
        # OCR needs the macOS helper (built in its own test); the others read without it.
        self.config = Config({"files": {"roots": [str(self.docs)], "ocr": False}}, path=self.tmp / "config.json")
        self.store = Store(":memory:")
        self.audit = AuditLog(self.tmp / "audit.jsonl")

    async def asyncTearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def index(self, embedder=None):
        return FileIndex(self.config, self.store, self.audit, embedder)

    def embedder(self, capabilities=("embedding",), digest="sha256:aaaa1111bbbb2222"):
        transport, seen = fake_embed_server(capabilities, digest)
        return Embedder("http://ollama/v1", "embeddinggemma-2", transport=transport), seen, transport

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

    async def test_cloud_only_files_and_bundles_are_left_alone(self):
        (self.docs / "Photos Library.photoslibrary" / "originals").mkdir(parents=True)
        (self.docs / "Photos Library.photoslibrary" / "originals" / "trip.txt").write_text("photo notes")
        (self.docs / "Tool.app" / "Contents").mkdir(parents=True)
        (self.docs / "Tool.app" / "Contents" / "readme.md").write_text("bundle insides")
        (self.docs / "in-icloud.txt").write_text("cloud!!")                  # 7 bytes: stands for a cloud-only file
        (self.docs / "repo" / ".git").mkdir(parents=True)
        (self.docs / "repo" / "src").mkdir()
        (self.docs / "repo" / "src" / "main.c").write_text("int main() { return 0; }")
        (self.docs / "repo" / "README.md").write_text("How to build the invoice tool")
        index = self.index()
        cloud = mock.patch("coderunner.files.discover.in_cloud_only", lambda stat: stat.st_size == 7)
        with cloud:
            result = await index.index_once()
        self.assertEqual((result["indexed"], result["cloud_only"]), (5, 1))
        parts = {}
        for row in self.store.query("SELECT path, part FROM file_chunks"):
            parts.setdefault(pathlib.Path(row["path"]).name, []).append(row["part"])
        self.assertNotIn("trip.txt", parts)                                  # inside a Photos library
        self.assertNotIn("readme.md", parts)                                 # inside an app
        self.assertNotIn("main.c", parts)                                    # source code in a git checkout
        self.assertIn("README.md", parts)                                    # ...but its docs are kept
        self.assertEqual(len(parts["in-icloud.txt"]), 1)                     # not downloaded: its name only
        self.assertTrue(parts["in-icloud.txt"][0].startswith("name only"))
        self.assertEqual(index.status()["cloud_only_skipped"], 1)
        self.config.set("files.cloud_files", True)                          # the owner opts in to downloads
        with cloud:
            self.assertEqual((await index.index_once())["indexed"], 1)
        parts = [r["part"] for r in self.store.query("SELECT part FROM file_chunks WHERE path LIKE '%in-icloud.txt'")]
        self.assertEqual(parts, ["name", "from line 1"])

    async def test_cloud_names_are_findable_at_once_and_embedded_after_local_files(self):
        for n in range(5):
            (self.docs / f"scan-{n}.pdf").write_text("x" * 7)               # 7 bytes: only in iCloud
        embedder, seen, _ = self.embedder()
        index = self.index(embedder)
        cloud = mock.patch("coderunner.files.discover.in_cloud_only", lambda stat: stat.st_size == 7)
        with cloud:
            await index._scan()
            self.assertEqual([f.remote for f in index._queue], [False] * 3 + [True] * 5)    # local first
            first = await index.index_once(budget=4)
        self.assertTrue(first["more"])
        # Named in the keyword index straight away, before any vector exists for them.
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM file_chunks WHERE modality = 'name' "
                                        "AND code IS NULL")["n"], 1)      # 3 local files + 1 name
        with cloud:
            while (await index.index_once(budget=4))["more"]:
                pass
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM file_chunks WHERE code IS NULL")["n"], 0)
        self.assertTrue(any(isinstance(item, str) and item.endswith("scan 3") for item in seen))
        await embedder.aclose()

    async def test_vectors_are_half_precision_with_sign_codes(self):
        self.assertEqual(list(unpack(pack([0.5, -0.25]))), [0.5, -0.25])
        self.assertEqual(len(pack([0.1] * 768)), 1536)
        self.assertEqual(list(unpack("[0.5, -0.25]")), [0.5, -0.25])      # rows from before
        self.assertEqual(signs([0.3, -0.1, 0.2]), 0b101)
        self.assertEqual(len(code_bytes([0.1] * 768)), 96)

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
        self.assertEqual(found[0]["line"], 1)
        # Changed and removed files are picked up on the next pass.
        (self.docs / "notes.md").write_text("Moved the dentist to Monday, and a total due reminder.")
        (self.docs / "taxes" / "invoice-2026.txt").unlink()
        result = await index.index_once()
        self.assertEqual((result["indexed"], result["removed"]), (1, 1))
        self.assertEqual([pathlib.Path(r["path"]).name for r in await index.search("total due")], ["notes.md"])
        # Unsharing a folder forgets what was indexed from it.
        self.assertEqual(index.remove_root(str(self.docs)), 2)
        self.assertEqual(await index.search("dentist"), [])

    async def test_copies_of_a_file_count_once(self):
        original = self.docs / "taxes" / "invoice-2026.txt"
        (self.docs / "invoice-2026 (1).txt").write_bytes(original.read_bytes())
        (self.docs / "invoice-2027.txt").write_text("Invoice 2026-42\nTotal due: 420 USD\n")   # same size, other bytes
        index = self.index()
        await index.index_once()
        found = await index.search("Total due")
        self.assertEqual(len(found), 2)
        copied = next(r for r in found if "copies" in r)
        self.assertEqual({pathlib.Path(p).name for p in [copied["path"], *copied["copies"]]},
                         {"invoice-2026.txt", "invoice-2026 (1).txt"})
        self.assertEqual(pathlib.Path(next(r for r in found if r is not copied)["path"]).name, "invoice-2027.txt")
        # A copy only iCloud holds is not read to compare it (that would download it).
        index.searcher._fingerprints.clear()
        with mock.patch("coderunner.files.search.in_cloud_only", lambda stat: True):
            self.assertEqual(len(await index.search("Total due")), 3)

    async def test_meaning_search_with_embeddings(self):
        embedder, seen, _ = self.embedder()
        index = self.index(embedder)
        await index.index_once()
        self.assertTrue(any(isinstance(item, str) and item.startswith("title: notes.md | text: ") for item in seen))
        found = await index.search("when is my dentist")
        self.assertEqual(pathlib.Path(found[0]["path"]).name, "notes.md")
        # The question goes once as a search and once as Google's code retrieval prompt, for code passages.
        self.assertEqual(seen[-2:], ["task: search result | query: when is my dentist",
                                     "task: code retrieval | query: when is my dentist"])
        from coderunner.files.kinds import is_code
        self.assertTrue(is_code("tools/build.rs") and is_code("App.SWIFT") and not is_code("notes.md"))
        await embedder.aclose()

    async def test_many_files_are_embedded_in_few_requests(self):
        for n in range(150):
            (self.docs / f"note-{n}.txt").write_text(f"holiday plan {n}")
        embedder, seen, transport = self.embedder()
        await self.index(embedder).index_once(budget=1000)
        # 153 files, each a name and a passage: about 306 inputs, sent 64 at a time.
        self.assertGreaterEqual(sum(transport.requests), 300)
        self.assertLessEqual(len(transport.requests), 12)
        await embedder.aclose()

    async def test_a_new_embedding_model_means_indexing_again(self):
        embedder, _, _ = self.embedder(digest="sha256:aaaa1111")
        index = self.index(embedder)
        await index.index_once()
        self.assertEqual(index.status()["by_kind"]["text"]["files"], 3)
        other, seen, _ = self.embedder(digest="sha256:bbbb2222")
        index = self.index(other)
        result = await index.index_once()
        self.assertEqual(result["indexed"], 3)                       # everything again, with the new weights
        self.assertTrue(any("reindex" in e["event"] for e in self.audit.tail()))
        await embedder.aclose()
        await other.aclose()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for media")
    async def test_a_new_media_encoder_re_embeds_only_media(self):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64",
                        "-frames:v", "1", str(self.docs / "red.png")], check=True)
        before, _, _ = self.embedder(("embedding", "vision"), digest="text:75b97ab036ca|media:969600645b52")
        index = self.index(before)
        await index.index_once()
        text_ids = {r["id"] for r in self.store.query("SELECT id FROM file_chunks c JOIN files f ON f.path = c.path "
                                                      "WHERE f.kind = 'text'")}
        after, seen, _ = self.embedder(("embedding", "vision"), digest="text:75b97ab036ca|media:coreml-audio-vision")
        index = self.index(after)
        result = await index.index_once()
        self.assertEqual(result["indexed"], 1)                                # the picture, nothing else
        self.assertTrue(any(isinstance(item, dict) and "image" in item for item in seen))
        self.assertFalse(any(isinstance(item, str) and "Dentist" in item for item in seen))
        kept = {r["id"] for r in self.store.query("SELECT id FROM file_chunks c JOIN files f ON f.path = c.path "
                                                  "WHERE f.kind = 'text'")}
        self.assertEqual(kept, text_ids)                                      # text vectors untouched
        other, _, _ = self.embedder(("embedding", "vision"), digest="text:ffffffffffff|media:coreml-audio-vision")
        index = self.index(other)
        self.assertEqual((await index.index_once())["indexed"], 4)            # new text weights: everything
        for embedder in (before, after, other):
            await embedder.aclose()

    async def test_file_events_bring_changes_in_first(self):
        index = self.index()
        await index.index_once()
        new = self.docs / "trip.md"
        new.write_text("Holiday in Goa")
        (self.docs / "notes.md").unlink()
        await index.changed({str(new), str(self.docs / "notes.md"), str(self.docs / ".env")})
        self.assertEqual([f.path.name for f in index._queue], ["trip.md"])          # never the secret
        self.assertIsNone(self.store.one("SELECT path FROM files WHERE path LIKE '%notes.md'"))
        await index.index_once()
        self.assertEqual(pathlib.Path((await index.search("Goa"))[0]["path"]).name, "trip.md")

    async def test_notes_first_then_media_then_code_then_checkouts(self):
        repo = self.docs / "aaa-cloned-project"                      # walked first by name on most systems
        (repo / ".git").mkdir(parents=True)
        (repo / "README.md").write_text("A library someone else wrote.")
        (self.docs / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
        index = self.index()
        await index._scan()
        order = [f.path.name for f in index._queue]
        self.assertEqual(set(order[:2]), {"invoice-2026.txt", "notes.md"})
        self.assertEqual(order[2:], ["photo.png", "code.py", "README.md"])

    async def test_a_file_queued_twice_is_indexed_once(self):
        index = self.index()
        await index._scan()                                      # the walk queues every file
        await index.changed({str(self.docs / "notes.md")})       # and an event queues one of them again
        result = await index.index_once()
        self.assertEqual((result["indexed"], result["failed"]), (3, 0))
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM files WHERE path LIKE '%notes.md'")["n"], 1)
        self.assertFalse(any(e["event"] == "files.index_error" for e in self.audit.tail()))

    async def test_indexing_waits_on_a_low_battery(self):
        with mock.patch.object(power, "state", mock.AsyncMock(return_value=(True, 12, False))):
            self.assertIn("12%", await power.pause_reason(self.config))
        with mock.patch.object(power, "state", mock.AsyncMock(return_value=(True, 80, False))):
            self.assertEqual(await power.pause_reason(self.config), "")
        with mock.patch.object(power, "state", mock.AsyncMock(return_value=(False, 90, True))):
            self.assertIn("Low Power", await power.pause_reason(self.config))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for media")
    async def test_images_and_audio_are_embedded_as_media(self):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64",
                        "-frames:v", "1", str(self.docs / "red.png")], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=70",
                        str(self.docs / "tone.wav")], check=True)
        embedder, seen, _ = self.embedder(("embedding", "vision", "audio"))
        index = self.index(embedder)
        await index.index_once()
        self.assertTrue(any(isinstance(item, dict) and "image" in item for item in seen))
        parts = [r["part"] for r in self.store.query("SELECT part FROM file_chunks WHERE path LIKE '%tone.wav'")]
        self.assertEqual(parts, ["name", "sound at 0:00", "sound at 0:25", "sound at 0:50"])   # 30 s windows
        found = await index.search("image", limit=20)
        self.assertIn("image", {r["kind"] for r in found})
        self.assertEqual(len({r["path"] for r in found}), len(found))      # one result per file
        # Without vision support the image is indexed by its name only.
        text_only, seen, _ = self.embedder(("embedding",))
        self.store.execute("DELETE FROM files")
        self.store.execute("DELETE FROM file_chunks")
        await self.index(text_only).index_once()
        self.assertFalse(any(isinstance(item, dict) for item in seen))
        await embedder.aclose()
        await text_only.aclose()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is needed for media")
    async def test_a_video_without_sound_is_indexed_by_its_frames(self):
        # Screen recordings have no audio track; asking for their sound failed the whole file.
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=25:size=160x120:rate=5",
                        "-pix_fmt", "yuv420p", str(self.docs / "screen recording.mp4")], check=True)
        embedder, seen, _ = self.embedder(("embedding", "vision", "audio"))
        index = self.index(embedder)
        await index.index_once()
        row = self.store.one("SELECT error FROM files WHERE path LIKE '%screen recording.mp4'")
        self.assertFalse(row["error"])
        parts = [r["part"] for r in self.store.query("SELECT part FROM file_chunks WHERE path LIKE '%screen recording.mp4'")]
        self.assertTrue(any(part.startswith("frame at") for part in parts), parts)
        self.assertFalse(any(part.startswith("sound at") for part in parts), parts)
        # A frame a second (25 s), and the whole video as the mean of its frames.
        self.assertEqual(sum(part.startswith("frame at") for part in parts), 25)
        rows = self.store.query("SELECT part, embedding FROM file_chunks WHERE path LIKE '%screen recording.mp4' "
                                "AND modality = 'image'")
        frames = [unpack(r["embedding"]) for r in rows if r["part"].startswith("frame at")]
        whole = next(unpack(r["embedding"]) for r in rows if r["part"] == "the whole video")
        total = [sum(column) for column in zip(*frames)]
        norm = sum(x * x for x in total) ** 0.5
        self.assertTrue(all(abs(a - b / norm) < 2e-3 for a, b in zip(whole, total)))
        await embedder.aclose()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for media")
    async def test_on_battery_media_waits_for_the_charger_and_text_goes_on(self):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64",
                        "-frames:v", "1", str(self.docs / "red.png")], check=True)
        embedder, seen, _ = self.embedder(("embedding", "vision", "audio"))
        index = self.index(embedder)
        with mock.patch.object(power, "state", mock.AsyncMock(return_value=(True, 80, False))):
            await index.index_once()
            self.assertFalse(any(isinstance(item, dict) for item in seen))        # nothing for the GPU
            self.assertIsNone(self.store.one("SELECT 1 AS x FROM files WHERE path LIKE '%red.png'"))
            self.assertTrue(self.store.one("SELECT 1 AS x FROM files WHERE path LIKE '%notes.md'"))
            self.assertEqual(index.status()["waiting_for_power"], 1)
            scans = mock.AsyncMock(return_value=0)
            with mock.patch.object(index, "_scan", scans):
                await index.index_once()
            scans.assert_not_called()                     # no walk of every folder while media waits
        await index.index_once()                          # the charger is back
        self.assertTrue(any(isinstance(item, dict) and "image" in item for item in seen))
        self.assertEqual(index.status()["waiting_for_power"], 0)
        await embedder.aclose()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for media")
    async def test_a_silent_track_is_not_embedded_as_sound(self):
        # A video's empty audio track embedded as sound windows outranked real sound in retrieval.
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=160x120:rate=5",
                        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-shortest", "-pix_fmt", "yuv420p",
                        str(self.docs / "quiet clip.mp4")], check=True)
        embedder, seen, _ = self.embedder(("embedding", "vision", "audio"))
        await self.index(embedder).index_once()
        parts = [r["part"] for r in self.store.query("SELECT part FROM file_chunks WHERE path LIKE '%quiet clip.mp4'")]
        self.assertTrue(any(part.startswith("frame at") for part in parts), parts)
        self.assertFalse(any(part.startswith("sound at") for part in parts), parts)
        from coderunner.files.extract import audible
        self.assertFalse(audible(b"RIFF....not a wav") is False)          # unreadable sound is kept, not dropped
        await embedder.aclose()

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
        # "My documents" read as kind doc: the text invoice is still found, with a note saying so.
        other = await registry.invoke("files_search", {"query": "invoice", "kind": "doc"}, ctx)
        self.assertIn("invoice-2026.txt", other)
        self.assertIn("No doc files matched", other)
        # No query: what is there, not "no matching files".
        listed = await registry.invoke("files_search", {"query": ""}, ctx)
        self.assertIn("invoice-2026.txt", listed)
        self.assertIn("most recently changed", listed)


class FindCommandTests(unittest.TestCase):
    def test_find_prints_places_and_falls_back_without_the_daemon(self):
        import contextlib
        import io

        from coderunner import cli
        home = os.path.expanduser("~")
        found = [{"path": f"{home}/Documents/lease.pdf", "kind": "doc", "part": "page 2", "page": 2,
                  "snippet": "Monthly rent 1500 USD", "score": 3.1},
                 {"path": f"{home}/Desktop/memo.m4a", "kind": "audio", "part": "sound at 1:15", "at": 75.0,
                  "score": 2.0, "in_icloud": True}]
        out = io.StringIO()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with mock.patch.dict(os.environ, {"CODERUNNER_HOME": tmp}), \
                mock.patch.object(cli, "_daemon_up", return_value=False), \
                mock.patch.object(cli, "_find_here", mock.AsyncMock(return_value=found)) as here, \
                contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["find", "how", "much", "is", "the", "rent"]), 0)
        here.assert_awaited_once()
        self.assertEqual(here.await_args.args[1:], ("how much is the rent", None, 8))
        text = out.getvalue()
        self.assertIn(" 1. ~/Documents/lease.pdf  (doc, page 2)", text)
        self.assertIn("Monthly rent 1500 USD", text)
        self.assertIn(" 2. ~/Desktop/memo.m4a  (audio, at 1:15, in iCloud)", text)


class ExtractTests(unittest.TestCase):
    def test_figures_are_matched_by_keywords_only(self):
        self.assertTrue(wordy("Dentist on Friday at 3pm"))
        self.assertFalse(wordy("2026-01-04,1200.50,331.20\n2026-01-05,99.10,12.00"))

    def test_passages_overlap_and_know_their_line(self):
        text = "".join(f"line {n}\n" for n in range(1, 400))
        found = passages(text)
        self.assertEqual(found[0][0], 1)
        self.assertGreater(found[1][0], 150)
        self.assertIn(found[1][1][:20], found[0][1])                   # overlap


class SearchTests(unittest.IsolatedAsyncioTestCase):
    """Ranking over a corpus big enough to calibrate, with hand-made vectors."""

    def setUp(self):
        self.store = Store(":memory:")
        rng = random.Random(7)
        self.dims = 64

        def noise():
            vector = [rng.gauss(0, 1) for _ in range(self.dims)]
            norm = sum(x * x for x in vector) ** 0.5
            return [x / norm for x in vector]
        self.noise = noise

    def add(self, path, kind, part, text, vector, modality):
        self.store.execute("INSERT OR IGNORE INTO files(path, kind, size, mtime, indexed, chunks, error) "
                           "VALUES (?, ?, 1, 1, 1, 1, '')", (path, kind))
        self.store.execute("INSERT INTO file_chunks(path, part, text, embedding, code, modality, loc) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)", (path, part, text, pack(vector), code_bytes(vector),
                                                            modality, None))

    def topic(self, axis, strength):
        vector = self.noise()
        vector[axis] += strength
        norm = sum(x * x for x in vector) ** 0.5
        return [x / norm for x in vector]

    async def search(self, query_vector, query, kind=None):
        class Fixed:
            async def query(self, text):
                return query_vector
        vectors = VectorIndex(self.store)
        return await search.Searcher(self.store, vectors, Fixed()).search(query, kind)

    async def test_the_calibration_sample_is_the_same_on_every_load(self):
        from coderunner.files.vectors import SAMPLE
        for n in range(SAMPLE + 100):
            self.add(f"/docs/{n}.txt", "text", "from line 1", f"note {n}", self.noise(), "text")
        first, second = VectorIndex(self.store), VectorIndex(self.store)
        first.load()
        second.load()
        self.assertEqual(len(first.sample["text"]), SAMPLE)
        self.assertEqual([list(v) for v in first.sample["text"]], [list(v) for v in second.sample["text"]])

    async def test_pictures_rank_by_their_own_scale_and_weak_matches_are_cut(self):
        for n in range(80):
            self.add(f"/d/note{n}.txt", "text", "from line 1", f"note {n}", self.noise(), "text")
            self.add(f"/d/photo{n}.jpg", "image", "image", f"photo{n}.jpg", self.noise(), "image")
        # Pictures score low against a text query, so one that matches stands out only on the image scale.
        self.add("/d/beach.jpg", "image", "image", "beach.jpg", self.topic(0, 0.8), "image")
        self.add("/d/beach-notes.txt", "text", "from line 1", "trip", self.topic(0, 0.6), "text")
        query = self.topic(0, 3.0)
        found = await self.search(query, "beach trip photo")
        names = [pathlib.Path(r["path"]).name for r in found]
        self.assertEqual(names[0], "beach.jpg")                       # the photo, helped by "photo" in the query
        self.assertIn("beach-notes.txt", names)
        self.assertLessEqual(len(found), 4)                            # the noise is cut

    async def test_codes_and_numbers_lean_on_words(self):
        for n in range(80):
            self.add(f"/d/doc{n}.txt", "text", "from line 1", f"ordinary text {n}", self.noise(), "text")
        self.add("/d/order.txt", "text", "from line 1", "Order K7Q2LM shipped", self.noise(), "text")
        self.assertTrue(search.is_lookup("K7Q2LM".split()))
        self.assertTrue(search.is_lookup("HX-55821".split()))
        self.assertFalse(search.is_lookup("rent for 2026".split()))
        found = await self.search(self.noise(), "K7Q2LM")
        self.assertEqual(pathlib.Path(found[0]["path"]).name, "order.txt")
        self.assertIn("K7Q2LM", found[0]["snippet"])

    def test_kind_words(self):
        self.assertEqual(search.kind_hints("screenshot of the error"), {"image"})
        self.assertEqual(search.kind_hints("the song from the wedding video"), {"audio", "video"})
        self.assertEqual(search.terms("where is my boarding pass"), ["boarding", "pass"])

    async def test_first_pass_finds_the_closest_codes(self):
        target = self.topic(5, 4.0)
        for n in range(300):
            self.add(f"/d/f{n}.txt", "text", "from line 1", "x", self.noise(), "text")
        self.add("/d/target.txt", "text", "from line 1", "x", target, "text")
        vectors = VectorIndex(self.store)
        vectors.load()
        chosen = vectors.nearest(target, 20)
        target_id = self.store.one("SELECT id FROM file_chunks WHERE path = '/d/target.txt'")["id"]
        self.assertIn(target_id, chosen)
        self.assertLess(len(chosen), 60)


@unittest.skipUnless(SWIFT, "the macOS helper needs the Swift compiler")
class HelperTests(unittest.IsolatedAsyncioTestCase):
    """The native helper, built once into a temporary CodeRunner folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp())
        cls.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(cls.tmp / "crhome")})
        cls.env.start()
        cls.helper = native.build(HELPER_SOURCE, "files-helper")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.assertIsNotNone(self.helper)
        self.docs = self.tmp / f"Documents-{self._testMethodName}"
        self.docs.mkdir()
        self.config = Config({"files": {"roots": [str(self.docs)]}}, path=self.tmp / "config.json")
        self.store = Store(":memory:")
        self.index = FileIndex(self.config, self.store, AuditLog(self.tmp / "audit.jsonl"))

    async def asyncTearDown(self):
        await self.index.media.close()

    async def test_pdfs_are_read_by_page_and_pictures_by_their_text(self):
        (self.docs / "letter.pdf").write_bytes(make_pdf(["Dear Sam, welcome aboard", "Salary 120000 per year"]))
        subprocess.run(["sips", "-s", "format", "png", str(self.docs / "letter.pdf"), "--out",
                        str(self.docs / "screenshot.png")], capture_output=True, check=True)
        await self.index.index_once()
        found = await self.index.search("salary")
        self.assertEqual((pathlib.Path(found[0]["path"]).name, found[0]["page"]), ("letter.pdf", 2))
        found = await self.index.search("welcome aboard", kind="image")
        self.assertEqual(pathlib.Path(found[0]["path"]).name, "screenshot.png")      # read with OCR
        self.assertEqual(found[0]["part"], "text in the image")

    async def test_file_events_arrive_within_seconds(self):
        seen: set[str] = set()

        async def changed(paths):
            seen.update(paths)
        from coderunner.files.watch import Watcher
        task = asyncio.create_task(Watcher(self.index.media).run([self.docs.resolve()], changed))
        deadline = time.monotonic() + 15
        while not any(p.endswith("new.md") for p in seen) and time.monotonic() < deadline:
            (self.docs / "new.md").write_text(f"hello {time.monotonic()}")    # until the stream has started
            await asyncio.sleep(1.0)
        task.cancel()
        self.assertTrue(any(p.endswith("new.md") for p in seen), seen)


def make_pdf(pages: list[str]) -> bytes:
    """A minimal PDF with one line of Helvetica text per page."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b"",
                            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        kids.append(len(objects))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


if __name__ == "__main__":
    unittest.main()
