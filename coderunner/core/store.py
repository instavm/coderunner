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
]
# Columns added after a table was first created; they fail harmlessly once present.
MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN code BLOB",
    "ALTER TABLE file_chunks ADD COLUMN modality TEXT DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN loc REAL",
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
                self._conn.execute(statement)
            for statement in MIGRATIONS:
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:  # duplicate column
                    pass

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
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

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
