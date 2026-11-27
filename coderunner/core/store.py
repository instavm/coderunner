"""SQLite state shared by the daemon's components."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from typing import Any, Iterable

from . import paths

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY, kind TEXT, target TEXT, summary TEXT, detail TEXT,
        risk TEXT, origin TEXT, status TEXT, scope TEXT, created REAL,
        decided REAL, decided_by TEXT)""",
    """CREATE TABLE IF NOT EXISTS grants (
        id INTEGER PRIMARY KEY, kind TEXT, target TEXT, decision TEXT,
        expires REAL, created REAL, source TEXT)""",
    """CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY, text TEXT NOT NULL, tags TEXT DEFAULT '',
        created REAL, embedding TEXT, source TEXT DEFAULT '')""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        text, tags, content='memories', content_rowid='id')""",
    """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags); END""",
    """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, text, tags) VALUES ('delete', old.id, old.text, old.tags); END""",
    """CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY, name TEXT, spec TEXT, prompt TEXT, channel TEXT,
        chat_id TEXT, next_run REAL, last_run REAL, last_status TEXT,
        enabled INTEGER DEFAULT 1, created REAL)""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY, conversation TEXT, role TEXT, content TEXT, created REAL)""",
    "CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation, id)",
    # Tasks started from the app and what happened during each (steps, notes, approvals, the answer).
    """CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, conversation TEXT, prompt TEXT, status TEXT, reply TEXT, created REAL, finished REAL)""",
    "CREATE INDEX IF NOT EXISTS tasks_conversation ON tasks(conversation, created)",
    # A thread's short name in the app's list (server/titles.py): source is "model" or "heuristic".
    """CREATE TABLE IF NOT EXISTS conversation_titles (
        conversation TEXT PRIMARY KEY, title TEXT, source TEXT, created REAL)""",
    """CREATE TABLE IF NOT EXISTS task_events (
        seq INTEGER PRIMARY KEY, task TEXT, conversation TEXT, ts REAL, data TEXT)""",
    "CREATE INDEX IF NOT EXISTS task_events_conversation ON task_events(conversation, seq)",
    # A task's journal (agent/journal.py), so it goes on after the daemon restarts: what it was started with, its
    # conversation as the model last saw it (one row, replaced each step), and each tool call of its latest step,
    # written before the call runs and again with its result. Deleted when the task ends.
    """CREATE TABLE IF NOT EXISTS task_journal (
        task TEXT PRIMARY KEY, job TEXT, state TEXT, restarts INTEGER DEFAULT 0, stopped INTEGER DEFAULT 0,
        updated REAL)""",
    """CREATE TABLE IF NOT EXISTS task_calls (
        task TEXT, step INTEGER, pos INTEGER, name TEXT, result TEXT, flags TEXT, approval TEXT, at REAL,
        PRIMARY KEY (task, step, pos))""",
    """CREATE TABLE IF NOT EXISTS devices (
        id TEXT PRIMARY KEY, name TEXT, created REAL, last_seen REAL)""",
    "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)",
    """CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY, kind TEXT, size INTEGER, mtime REAL, indexed REAL, chunks INTEGER,
        error TEXT DEFAULT '')""",
    """CREATE TABLE IF NOT EXISTS file_chunks (
        id INTEGER PRIMARY KEY, path TEXT, part TEXT, text TEXT, embedding BLOB, code BLOB,
        modality TEXT DEFAULT '', loc REAL)""",
    "CREATE INDEX IF NOT EXISTS file_chunks_path ON file_chunks(path)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS file_chunks_fts USING fts5(
        text, content='file_chunks', content_rowid='id')""",
    """CREATE TRIGGER IF NOT EXISTS file_chunks_ai AFTER INSERT ON file_chunks BEGIN
        INSERT INTO file_chunks_fts(rowid, text) VALUES (new.id, new.text); END""",
    """CREATE TRIGGER IF NOT EXISTS file_chunks_ad AFTER DELETE ON file_chunks BEGIN
        INSERT INTO file_chunks_fts(file_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END""",
    # Each file's and folder's own name and kind, for the ranking by names (id: the name piece's id).
    "CREATE TABLE IF NOT EXISTS file_names_src (id INTEGER PRIMARY KEY, text TEXT)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS file_names USING fts5(
        text, content='file_names_src', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2')""",
    """CREATE TRIGGER IF NOT EXISTS file_names_ai AFTER INSERT ON file_names_src BEGIN
        INSERT INTO file_names(rowid, text) VALUES (new.id, new.text); END""",
    """CREATE TRIGGER IF NOT EXISTS file_names_ad AFTER DELETE ON file_names_src BEGIN
        INSERT INTO file_names(file_names, rowid, text) VALUES ('delete', old.id, old.text); END""",
    """CREATE TRIGGER IF NOT EXISTS file_chunks_names_ad AFTER DELETE ON file_chunks BEGIN
        DELETE FROM file_names_src WHERE id = old.id; END""",
    "CREATE INDEX IF NOT EXISTS files_digest ON files(digest)",
    "CREATE INDEX IF NOT EXISTS files_inode ON files(inode)",
    # The names still to be embedded, asked for on every indexing pass: without this the question read the
    # whole table of pieces, 13 s with 518,000 of them, and the daemon answered nothing meanwhile.
    "CREATE INDEX IF NOT EXISTS files_names_waiting ON file_chunks(id) "
    "WHERE modality = 'name' AND code IS NULL AND embedding IS NULL",
]
# Columns added after a table was first created; they fail harmlessly once present.
MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT ''",
    "ALTER TABLE schedules ADD COLUMN on_change INTEGER DEFAULT 0",
    "ALTER TABLE schedules ADD COLUMN last_digest TEXT DEFAULT ''",
    "ALTER TABLE schedules ADD COLUMN last_reply TEXT DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN code BLOB",
    "ALTER TABLE file_chunks ADD COLUMN modality TEXT DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN loc REAL",
    "ALTER TABLE files ADD COLUMN digest BLOB",            # the contents' fingerprint: a copy takes its twin's pieces
    "ALTER TABLE files ADD COLUMN device INTEGER",         # identity on its volume: a moved file keeps its pieces
    "ALTER TABLE files ADD COLUMN inode INTEGER",
    "ALTER TABLE files ADD COLUMN profile TEXT DEFAULT ''",  # what the pieces depend on (a folder: the kinds in it)
    "ALTER TABLE files ADD COLUMN holds TEXT DEFAULT ''",    # a folder: the kinds of file directly in it
]


class Store:
    def __init__(self, path=None):
        self.path = str(path or paths.db_path())
        self._lock = threading.RLock()
        if path is None:
            paths.ensure_dirs()
        if self.path != ":memory:":
            # Conversations, memories and approvals are private to the owner.
            os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
            os.chmod(self.path, 0o600)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            for statement in SCHEMA:
                if statement.startswith("CREATE INDEX IF NOT EXISTS files_"):
                    continue
                self._conn.execute(statement)
            for statement in MIGRATIONS:
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:  # duplicate column
                    pass
            for statement in SCHEMA:
                if statement.startswith("CREATE INDEX IF NOT EXISTS files_"):
                    self._conn.execute(statement)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    @contextlib.contextmanager
    def transaction(self):
        """Statements inside run as one transaction: many rows are written for the cost of one commit."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self
            except BaseException:
                if self._conn.in_transaction:      # SQLite rolls back by itself on some errors (a full disk)
                    self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    @contextlib.contextmanager
    def reader(self):
        """A read-only connection of its own for a long read, which WAL lets run beside the shared one: reading
        the file index's 227,000 vectors through the shared connection held its lock for 7 s, and every request
        to the daemon waited (`coderunner chat` concluded it was not running)."""
        if self.path == ":memory:":
            with self._lock:
                yield self._conn
            return
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM kv WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
