"""Mail, Messages, Notes and Calendar as entries in the file index."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from coderunner.agent.loop import RunContext
from coderunner.agent.tools import build_registry
from coderunner.core.audit import AuditLog
from coderunner.core.store import Store
from coderunner.files import Embedder, FileIndex
from coderunner.files.sources import APPLE_EPOCH, Sources, apple_seconds, attributed_text, is_source_path
from fakes import Svc
from test_files import fake_embed_server


def typedstream(text: str) -> bytes:
    """A message body as Messages stores it: an archived attributed string."""
    data = text.encode()
    length = bytes([len(data)]) if len(data) < 0x80 else b"\x81" + len(data).to_bytes(2, "little")
    return b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84\x08NSObject" \
        b"\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+" + length + data + b"\x86\x84\x02iI\x01"


class FakeApps:
    """Mail, Notes and Calendar as a fake JavaScript-for-Automation runner."""

    def __init__(self):
        self.mail = {"inbox": [{"id": 1, "subject": "Your code is 123456", "sender": "bank@example.com",
                                "to": ["me@example.com"], "date": 1_700_000_000_000,
                                "content": "Use code 123456 to sign in. Rent is due on the 5th."},
                               {"id": 2, "subject": "Lisbon trip itinerary", "sender": "ana@example.com",
                                "to": ["me@example.com"], "date": 1_700_100_000_000,
                                "content": "Pena Palace entry is at 10:30 on Thursday, bring the tickets."}],
                     "sent": []}
        self.notes = [{"id": "x-coredata://1", "name": "Garlic planting", "folder": "Garden", "modified": 1_700_200_000_000,
                       "body": "Plant garlic in late October before the first frost."}]
        self.events = [{"id": "EV-1", "title": "Dentist", "start": 1_700_300_000_000, "end": 1_700_303_600_000,
                        "location": "Main St", "calendar": "Home", "notes": "Cleaning appointment"}]
        self.calls: list[tuple[str, list[str]]] = []

    async def __call__(self, script: str, args: list[str], app: str, timeout: float):
        self.calls.append((app, args))
        if app == "Mail" and "box.messages.id()" in script:
            return [[m["id"], m["date"]] for m in self.mail.get(args[0], [])]
        if app == "Mail":
            wanted = {int(i) for i in args[2:]}
            return [{k: v for k, v in m.items() if k != "to"} for m in self.mail.get(args[0], []) if m["id"] in wanted]
        if app == "Notes" and "notes.id()" in script:
            return [[n["id"], n["name"], n["modified"]] for n in self.notes]
        if app == "Notes":
            return [n for n in self.notes if n["id"] in set(args[1:])]
        if app == "Calendar":
            return list(self.events)
        raise AssertionError(app)


def messages_db(path: pathlib.Path) -> None:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE handle(ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat(ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message(ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB, date INTEGER,
                             is_from_me INTEGER, handle_id INTEGER);
        CREATE TABLE chat_message_join(chat_id INTEGER, message_id INTEGER);
    """)
    db.execute("INSERT INTO handle VALUES (1, '+15551234567')")
    db.execute("INSERT INTO chat VALUES (7, '+15551234567', '')")
    day = datetime.datetime(2026, 11, 2, 9, 0).timestamp()
    rows = [(1, "Is the car ready?", None, int((day - APPLE_EPOCH) * 1e9), 1, 1),
            (2, None, typedstream("Yes, the AC repair is done, come by after 3"), int((day + 600 - APPLE_EPOCH) * 1e9), 0, 1)]
    db.executemany("INSERT INTO message VALUES (?, ?, ?, ?, ?, ?)", rows)
    db.executemany("INSERT INTO chat_message_join VALUES (7, ?)", [(1,), (2,)])
    db.commit()
    db.close()


class SourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.env = mock.patch.dict(os.environ, {"CODERUNNER_HOME": str(self.tmp / "home")})
        self.env.start()
        self.docs = self.tmp / "Documents"
        self.docs.mkdir()
        (self.docs / "notes.md").write_text("Dentist on Friday at 3pm.\n")
        # The policy, the index and the sources share one configuration, as in the daemon.
        self.svc = Svc({"files": {"roots": [str(self.docs)], "ocr": False, "cloud_file_mb": 0,
                                  "sources": ["mail", "messages", "notes", "calendar"]},
                        "connectors": {"enabled": ["mail", "notes", "calendar"]}})
        self.config = self.svc.config
        self.store = Store(":memory:")
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.apps = FakeApps()
        messages_db(self.tmp / "chat.db")

    async def asyncTearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def index(self):
        transport, seen = fake_embed_server(("embedding",), "sha256:aaaa1111bbbb2222")
        embedder = Embedder("http://ollama/v1", "embeddinggemma-2", transport=transport)
        index = FileIndex(self.config, self.store, self.audit, embedder)
        index.sources = Sources(self.config, self.store, self.audit, embedder.doc_prefix, runner=self.apps,
                                messages_db=str(self.tmp / "chat.db"))
        return index, embedder, seen

    def test_message_bodies_and_dates_are_decoded(self):
        self.assertEqual(attributed_text(typedstream("Yes, done")), "Yes, done")
        self.assertEqual(attributed_text(typedstream("x" * 300)), "x" * 300)
        self.assertEqual(attributed_text(None), "")
        self.assertEqual(apple_seconds(0), 0.0)
        self.assertAlmostEqual(apple_seconds(5 * 10 ** 17), APPLE_EPOCH + 5e8, places=3)   # nanoseconds
        self.assertAlmostEqual(apple_seconds(1), APPLE_EPOCH + 1, places=3)             # older rows: seconds
        self.assertEqual(is_source_path("mail://inbox/4"), "mail")
        self.assertEqual(is_source_path("/Users/me/notes.md"), None)

    async def test_entries_are_indexed_found_and_read_under_the_rules(self):
        index, embedder, seen = self.index()
        while (await index.index_once())["more"]:
            pass
        paths = {r["path"]: r for r in self.store.query("SELECT path, kind FROM files")}
        self.assertEqual(paths["mail://inbox/2"]["kind"], "mail")
        self.assertEqual(paths["note://x-coredata://1"]["kind"], "note")
        self.assertEqual(paths["event://EV-1"]["kind"], "event")
        self.assertEqual(paths["message://7/2026-11-02"]["kind"], "message")
        # Codes are redacted before anything is indexed; the rest of the mail is searchable by its words.
        text = index.sources.text("mail://inbox/1")
        self.assertNotIn("123456", text)
        self.assertIn("Rent is due", text)
        self.assertIn("Pena Palace", index.sources.text("mail://inbox/2"))
        self.assertIn("AC repair is done", index.sources.text("message://7/2026-11-02"))
        found = await index.search("pena palace entry")
        self.assertEqual(found[0]["path"], "mail://inbox/2")
        self.assertEqual(found[0]["kind"], "mail")
        found = await index.search("garlic planting", kind="note")
        self.assertEqual([r["path"] for r in found], ["note://x-coredata://1"])
        self.assertEqual(index.status()["sources"]["indexed"], {"mail": 2, "messages": 1, "notes": 1, "calendar": 1})
        # Reading through the tools: the task then holds that source and has read third-party text.
        svc = self.svc
        svc.files = index
        registry = build_registry(svc)
        ctx = RunContext(conversation="s1")
        result = await registry.invoke("files_search", {"query": "pena palace"}, ctx)
        self.assertIn("mail://inbox/2", result)
        self.assertTrue({"files", "mail"} <= ctx.private_sources)       # every source among the results
        self.assertTrue(ctx.content_tainted)
        text = await registry.invoke("file_read", {"path": "mail://inbox/2"}, ctx)
        self.assertIn("bring the tickets", text)
        # A source whose connector is not enabled cannot be read, and is not indexed again.
        self.config.set("connectors.enabled", ["notes", "calendar"])
        denied = await registry.invoke("file_read", {"path": "mail://inbox/2"}, RunContext(conversation="s2"))
        self.assertIn("not allowed", denied)
        self.assertNotIn("mail", index.sources.enabled())
        await embedder.aclose()

    async def test_sources_are_listed_at_once_on_a_machine_that_has_just_started(self):
        # The clock counts from boot: with "never listed" kept as time zero, nothing was listed for the first
        # quarter of an hour after a restart (and never in a container started for a test run).
        index, embedder, _ = self.index()
        try:
            with mock.patch("coderunner.files.sources.time.monotonic", return_value=30.0):
                items, _ = await index.sources.index_once()
                self.assertTrue(any(item.path.startswith("mail://") for item in items))
                again, _ = await index.sources.index_once()              # and then not again until it is due
                self.assertEqual(again, [])
        finally:
            await embedder.aclose()

    async def test_changes_and_removals_follow_the_apps(self):
        index, embedder, seen = self.index()
        while (await index.index_once())["more"]:
            pass
        self.apps.mail["inbox"].pop(0)                        # a mail deleted
        self.apps.notes[0]["modified"] += 1000                 # a note edited
        self.apps.notes[0]["body"] = "Plant garlic in November this year."
        self.apps.events[0]["title"] = "Dentist (moved)"
        index.sources._last.clear()                            # the next pass lists the apps again
        result = await index.index_once()
        self.assertEqual((result["entries"], result["removed"]), (2, 1))
        self.assertIsNone(self.store.one("SELECT 1 AS x FROM files WHERE path = 'mail://inbox/1'"))
        self.assertIn("November", index.sources.text("note://x-coredata://1"))
        self.assertIn("(moved)", index.sources.text("event://EV-1"))
        # A source taken away forgets its entries.
        index.sources.remove("calendar")
        self.assertEqual(index.sources.chosen(), ["mail", "messages", "notes"])
        await embedder.aclose()


if __name__ == "__main__":
    unittest.main()
