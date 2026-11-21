"""The owner's Mail, Messages, Notes and Calendar as entries in the file index.

Each item becomes an entry like a file, with an address instead of a path (mail://inbox/4120,
message://12/2026-11-02, note://x-coredata..., event://A1B2-...), a name piece and passages of its text, found
by the same search. Mail, Notes and Calendar are read through the connectors (JavaScript for Automation, dynamic
values only ever through argv), so only enabled connectors are indexed; Messages through its database
(~/Library/Messages/chat.db, which needs Full Disk Access), a day of one conversation per entry. Each source is
opt-in (`coderunner files sources add mail`). One-time codes and sign-in links in mail are redacted before they
are indexed, as the mail tools do.

What these entries hold is the owner's private data and third-party text: the tools that return them mark the
task as holding that source and as having read untrusted content, so the leak rules apply (ASSISTANT.md).
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..connectors.macos import CALENDAR_TIMEOUT, DEFAULT_TIMEOUT, PRELUDE, run_jxa
from ..connectors.redact import redact
from .extract import MAX_CHUNKS, Piece, passages, readable, wordy

SOURCES = ("mail", "messages", "notes", "calendar")
SCHEMES = {"mail": "mail", "messages": "message", "notes": "note", "calendar": "event"}
KINDS = {"mail": "mail", "messages": "message", "notes": "note", "calendar": "event"}   # the entry's kind
SOURCE_OF_KIND = {kind: source for source, kind in KINDS.items()}
CONNECTOR_SOURCES = ("mail", "notes", "calendar")        # read through an enabled connector
MAILBOXES = ("inbox", "sent")
MAX_TEXT = 20000            # characters of one item read
BULK = 40                   # items read in one script run
CALENDAR_DAYS = 365         # events this far back and ahead
LIST_TIMEOUT = 600.0        # listing a 100,000-message inbox or a year of events takes a while, once per interval
MESSAGES_DB = "~/Library/Messages/chat.db"
APPLE_EPOCH = 978307200     # 2001-01-01 in Unix seconds
Runner = Callable[[str, list[str], str, float], Awaitable[Any]]


def is_source_path(path: str) -> str | None:
    """The source a path belongs to ("mail" for mail://...), or None for a file."""
    for source, scheme in SCHEMES.items():
        if path.startswith(scheme + "://"):
            return source
    return None


@dataclass
class Item:
    """An entry the indexer writes like a file: what FileIndex._write reads of a Found, plus its pieces."""
    path: str
    kind: str
    mtime: float
    size: int
    pieces: list[Piece]
    own_name: str
    profile: str = ""
    holds: tuple = ()
    folder: bool = False
    remote: bool = False
    evicted: bool = False
    in_repo: bool = False
    repo: str | None = None
    screenshot: bool = False
    device: int = 0
    inode: int = 0


# ---- scripts (dynamic values only ever arrive through argv) ---------------

MAIL_IDS = PRELUDE + r"""
function run(argv) {
  const Mail = Application('Mail');
  const box = mailbox(Mail, argv[0]);
  const ids = box.messages.id();
  const dates = box.messages.dateReceived();
  const out = [];
  for (let i = 0; i < ids.length; i++) out.push([ids[i], dates[i] ? dates[i].getTime() : 0]);
  return JSON.stringify(out);
}
"""

MAIL_BULK = PRELUDE + r"""
function run(argv) {
  const Mail = Application('Mail');
  const box = mailbox(Mail, argv[0]);
  const max = Number(argv[1]);
  // One query for the batch (a query per message scanned a 100,000-message inbox each time), then one bulk
  // fetch per property.
  const found = box.messages.whose({_or: argv.slice(2).map(s => ({id: Number(s)}))});
  const ids = found.id();
  if (!ids.length) return JSON.stringify([]);
  const subjects = found.subject(), senders = found.sender(), dates = found.dateReceived();
  let contents = [];
  try { contents = found.content(); } catch (e) { if (fatal(e)) throw e; }
  const out = [];
  for (let i = 0; i < ids.length; i++) {
    out.push({id: ids[i], subject: subjects[i] || '', sender: senders[i] || '',
              date: dates[i] ? dates[i].getTime() : 0, content: (contents[i] || '').slice(0, max)});
  }
  return JSON.stringify(out);
}
"""

NOTES_LIST = PRELUDE + r"""
function run(argv) {
  const N = Application('Notes');
  const notes = N.notes;
  const ids = notes.id(), names = notes.name(), mods = notes.modificationDate();
  const out = [];
  for (let i = 0; i < ids.length; i++) out.push([ids[i], names[i] || '', mods[i] ? mods[i].getTime() : 0]);
  return JSON.stringify(out);
}
"""

NOTES_BULK = PRELUDE + r"""
function run(argv) {
  const N = Application('Notes');
  const max = Number(argv[0]);
  const out = [];
  for (const id of argv.slice(1)) {
    try {
      const note = N.notes.byId(id);
      let folder = '';
      try { folder = note.container().name(); } catch (e) { if (fatal(e)) throw e; }
      out.push({id: id, name: note.name() || '', folder: folder,
                modified: note.modificationDate() ? note.modificationDate().getTime() : 0,
                body: (note.plaintext() || '').slice(0, max)});
    } catch (e) { if (fatal(e)) throw e; }
  }
  return JSON.stringify(out);
}
"""

CALENDAR_RANGE = PRELUDE + r"""
function run(argv) {
  const start = new Date(Number(argv[0])), end = new Date(Number(argv[1])), max = Number(argv[2]);
  const Cal = Application('Calendar');
  const out = [];
  for (const cal of Cal.calendars()) {
    let name = '';
    try {
      name = cal.name();
      const evs = cal.events.whose({_and: [{startDate: {_lessThan: end}}, {endDate: {_greaterThan: start}}]});
      const titles = evs.summary();
      if (!titles.length) continue;
      const starts = evs.startDate(), ends = evs.endDate(), locs = evs.location(), ids = evs.uid();
      let notes = [];
      try { notes = evs.description(); } catch (e) { if (fatal(e)) throw e; }
      for (let i = 0; i < titles.length; i++) {
        out.push({id: ids[i], title: titles[i] || '', start: starts[i] ? starts[i].getTime() : 0,
                  end: ends[i] ? ends[i].getTime() : 0, location: locs[i] || '', calendar: name,
                  notes: (notes[i] || '').slice(0, max)});
      }
    } catch (e) {
      if (fatal(e)) throw e;
    }
  }
  return JSON.stringify(out);
}
"""


# ---- helpers --------------------------------------------------------------

def _when(ms: float) -> str:
    if not ms:
        return ""
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def attributed_text(blob: bytes | None) -> str:
    """The text of a message stored as an attributed string (typedstream): the NSString after its marker, with
    a one-byte length, or 0x81 and a two-byte one."""
    if not blob:
        return ""
    at = blob.find(b"NSString")
    if at < 0:
        return ""
    plus = blob.find(b"+", at)
    if plus < 0:
        return ""
    i = plus + 1
    if i >= len(blob):
        return ""
    length = blob[i]
    i += 1
    if length == 0x81 and i + 2 <= len(blob):
        length = int.from_bytes(blob[i:i + 2], "little")
        i += 2
    elif length == 0x82 and i + 4 <= len(blob):
        length = int.from_bytes(blob[i:i + 4], "little")
        i += 4
    return blob[i:i + length].decode("utf-8", "replace")


def apple_seconds(value: int | float | None) -> float:
    """Messages stores dates as nanoseconds (older rows: seconds) since 2001."""
    if not value:
        return 0.0
    value = float(value)
    if value > 1e12:
        value /= 1e9
    return value + APPLE_EPOCH


class Sources:
    def __init__(self, config, store, audit, doc_prefix: str = "", runner: Runner | None = None,
                 messages_db: str | None = None):
        self.config = config
        self.store = store
        self.audit = audit
        self.doc_prefix = doc_prefix
        self._run = runner or self._jxa
        self.messages_db = messages_db or os.path.expanduser(MESSAGES_DB)
        self._last: dict[str, float] = {}

    async def _jxa(self, script: str, args: list[str], app: str, timeout: float) -> Any:
        return await run_jxa(script, args, app=app, timeout=timeout)

    # ---- which ------------------------------------------------------------------

    def chosen(self) -> list[str]:
        return [s for s in (self.config.get("files.sources", []) or []) if s in SOURCES]

    def enabled(self) -> list[str]:
        """The chosen sources that can be read: a connector source needs its connector enabled."""
        connectors = set(self.config.get("connectors.enabled", []) or [])
        return [s for s in self.chosen() if s not in CONNECTOR_SOURCES or s in connectors]

    def add(self, source: str) -> list[str]:
        if source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}. Choose from: {', '.join(SOURCES)}")
        chosen = self.chosen()
        if source not in chosen:
            self.config.set("files.sources", chosen + [source])
            self.config.save()
        return self.chosen()

    def remove(self, source: str) -> list[str]:
        self.config.set("files.sources", [s for s in self.chosen() if s != source])
        self.config.save()
        return self.chosen()

    def indexed(self, source: str) -> dict[str, dict]:
        """The entries of a source already in the index, by address."""
        return {r["path"]: r for r in self.store.query(
            "SELECT path, mtime, size, profile FROM files WHERE path LIKE ?", (SCHEMES[source] + "://%",))}

    def status(self) -> dict:
        counts = {source: len(self.indexed(source)) for source in self.chosen()}
        return {"chosen": self.chosen(), "enabled": self.enabled(), "indexed": counts}

    def text(self, path: str) -> str:
        """An entry's text as indexed (its passages), for reading it without going back to the app."""
        rows = self.store.query("SELECT text FROM file_chunks WHERE path = ? AND modality = 'text' ORDER BY id", (path,))
        return "\n".join(r["text"] for r in rows)

    # ---- one pass -------------------------------------------------------------

    async def index_once(self, budget: int = BULK, every: float = 900.0) -> tuple[list[Item], list[str]]:
        """Entries to write (new or changed, newest first, at most `budget`) and addresses to drop, across the
        enabled sources; each source is listed again only every `every` seconds unless it has more to do."""
        items: list[Item] = []
        gone: list[str] = []
        now = time.monotonic()
        for source in self.enabled():
            # A source never listed is listed now: counted from zero, the clock (time since boot) kept every
            # source waiting for the first quarter of an hour after a restart.
            last = self._last.get(source)
            if last is not None and now - last < every and not self._more.get(source):
                continue
            try:
                found, dropped = await getattr(self, f"_{source}")(budget - len(items))
            except Exception as exc:
                self.audit.write("files.source_error", source=source, error=str(exc)[:300])
                self._last[source] = now
                self._more[source] = False
                continue
            items += found
            gone += dropped
            self._last[source] = now
            if len(items) >= budget:
                break
        return items, gone

    _more: dict[str, bool] = {}

    # ---- pieces ---------------------------------------------------------------

    def _prefix(self, title: str) -> str:
        return self.doc_prefix.format(title=title) if self.doc_prefix else ""

    def _item(self, path: str, kind: str, name_text: str, name_embed: str, body: str, mtime: float,
              profile: str = "", own_name: str = "") -> Item:
        pieces = [Piece("name", name_text, self._prefix(name_text) + name_embed, "name")]
        for line, chunk in passages(body)[:MAX_CHUNKS]:
            worth = readable(chunk) if wordy(chunk) else None
            pieces.append(Piece(f"from line {line}", chunk, self._prefix(name_text) + worth if worth else None,
                                "text", line))
        return Item(path, kind, mtime, len(body), pieces, own_name or name_text, profile)

    # ---- mail -------------------------------------------------------------------

    async def _mail(self, budget: int) -> tuple[list[Item], list[str]]:
        items: list[Item] = []
        gone: list[str] = []
        known = self.indexed("mail")
        more = False
        for mailbox in self.config.get("files.mail_mailboxes", list(MAILBOXES)) or list(MAILBOXES):
            listed = await self._run(MAIL_IDS, [mailbox], "Mail", LIST_TIMEOUT) or []
            current = {f"mail://{mailbox}/{int(i)}": float(ms or 0) / 1000 for i, ms in listed}
            gone += [p for p in known if p.startswith(f"mail://{mailbox}/") and p not in current]
            missing = sorted((p for p in current if p not in known), key=lambda p: -current[p])
            if len(missing) > budget - len(items):
                more = True
            for start in range(0, min(len(missing), budget - len(items)), BULK):
                ids = [p.rsplit("/", 1)[1] for p in missing[start:start + BULK]]
                for message in await self._run(MAIL_BULK, [mailbox, str(MAX_TEXT), *ids], "Mail", CALENDAR_TIMEOUT) or []:
                    subject = redact(message.get("subject") or "") or "(no subject)"
                    sender = message.get("sender") or ""
                    date = _when(message.get("date") or 0)
                    content = redact(message.get("content") or "")
                    body = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{content}"
                    items.append(self._item(f"mail://{mailbox}/{message['id']}", "mail",
                                            f"{subject} from {sender} email", f"{subject} (email from {sender}, {date})",
                                            body, float(message.get("date") or 0) / 1000 or time.time()))
            if len(items) >= budget:
                break
        self._more["mail"] = more
        return items, gone

    # ---- notes ------------------------------------------------------------------

    async def _notes(self, budget: int) -> tuple[list[Item], list[str]]:
        known = self.indexed("notes")
        listed = await self._run(NOTES_LIST, [], "Notes", LIST_TIMEOUT) or []
        current = {f"note://{i}": (name, float(ms or 0) / 1000) for i, name, ms in listed}
        gone = [p for p in known if p not in current]
        changed = sorted((p for p, (_, mtime) in current.items() if p not in known or known[p]["mtime"] != mtime),
                         key=lambda p: -current[p][1])
        self._more["notes"] = len(changed) > budget
        items: list[Item] = []
        for start in range(0, min(len(changed), budget), BULK):
            ids = [p[len("note://"):] for p in changed[start:start + BULK]]
            for note in await self._run(NOTES_BULK, [str(MAX_TEXT), *ids], "Notes", CALENDAR_TIMEOUT) or []:
                name = note.get("name") or "Untitled note"
                folder = note.get("folder") or ""
                body = note.get("body") or ""
                items.append(self._item(f"note://{note['id']}", "note", f"{name} {folder} note",
                                        f"{name} (note in {folder})" if folder else f"{name} (note)", body,
                                        float(note.get("modified") or 0) / 1000 or time.time()))
        return items, gone

    # ---- calendar -----------------------------------------------------------------

    async def _calendar(self, budget: int) -> tuple[list[Item], list[str]]:
        known = self.indexed("calendar")
        days = int(self.config.get("files.calendar_days", CALENDAR_DAYS) or CALENDAR_DAYS)
        now = time.time()
        start, end = int((now - days * 86400) * 1000), int((now + days * 86400) * 1000)
        events = await self._run(CALENDAR_RANGE, [str(start), str(end), str(MAX_TEXT)], "Calendar", LIST_TIMEOUT) or []
        current: dict[str, dict] = {}
        for event in events:
            if event.get("id"):
                current[f"event://{event['id']}"] = event
        gone = [p for p in known if p not in current]
        items: list[Item] = []
        changed = []
        for path, event in current.items():
            profile = hashlib.sha256(json.dumps([event.get(k) for k in ("title", "start", "end", "location", "notes",
                                                                       "calendar")], default=str).encode()).hexdigest()[:16]
            if path not in known or known[path]["profile"] != profile:
                changed.append((path, event, profile))
        changed.sort(key=lambda item: -abs(float(item[1].get("start") or 0) / 1000 - now))
        changed.reverse()           # nearest to today first
        self._more["calendar"] = len(changed) > budget
        for path, event, profile in changed[:budget]:
            title = event.get("title") or "(untitled event)"
            when = f"{_when(event.get('start') or 0)} to {_when(event.get('end') or 0)}"
            body = f"{title}\nWhen: {when}\nWhere: {event.get('location') or ''}\nCalendar: {event.get('calendar') or ''}"
            if event.get("notes"):
                body += f"\nNotes: {event['notes']}"
            items.append(self._item(path, "event", f"{title} {event.get('calendar') or ''} event",
                                    f"{title} ({when}, event in {event.get('calendar') or 'Calendar'})", body,
                                    float(event.get("start") or 0) / 1000 or now, profile))
        return items, gone

    # ---- messages -------------------------------------------------------------------

    def _read_messages(self, since_rowid: int, limit: int) -> tuple[list[dict], int]:
        db = sqlite3.connect(f"file:{self.messages_db}?mode=ro", uri=True, timeout=5)
        try:
            rows = db.execute(
                "SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id AS handle, c.ROWID AS chat, "
                "c.display_name, c.chat_identifier FROM message m LEFT JOIN handle h ON h.ROWID = m.handle_id "
                "LEFT JOIN chat_message_join j ON j.message_id = m.ROWID LEFT JOIN chat c ON c.ROWID = j.chat_id "
                "WHERE m.ROWID > ? ORDER BY m.ROWID LIMIT ?", (since_rowid, limit)).fetchall()
            found = []
            for rowid, text, body, date, mine, handle, chat, display, identifier in rows:
                words = text or attributed_text(body)
                found.append({"rowid": rowid, "text": words or "", "when": apple_seconds(date), "mine": bool(mine),
                              "handle": handle or "", "chat": chat or 0, "chat_name": display or identifier or ""})
            return found, (rows[-1][0] if rows else since_rowid)
        finally:
            db.close()

    def _day_messages(self, chat: int, day: str) -> list[dict]:
        begin = datetime.datetime.strptime(day, "%Y-%m-%d").timestamp()
        end = begin + 86400
        db = sqlite3.connect(f"file:{self.messages_db}?mode=ro", uri=True, timeout=5)
        try:
            rows = db.execute(
                "SELECT m.text, m.attributedBody, m.date, m.is_from_me, h.id FROM message m "
                "JOIN chat_message_join j ON j.message_id = m.ROWID LEFT JOIN handle h ON h.ROWID = m.handle_id "
                "WHERE j.chat_id = ? ORDER BY m.date", (chat,)).fetchall()
        finally:
            db.close()
        found = []
        for text, body, date, mine, handle in rows:
            when = apple_seconds(date)
            if begin <= when < end:
                words = text or attributed_text(body)
                if words:
                    found.append({"text": words, "when": when, "mine": bool(mine), "handle": handle or ""})
        return found

    async def _messages(self, budget: int) -> tuple[list[Item], list[str]]:
        cursor = int(self.store.get_kv("files.sources.messages.rowid", 0) or 0)
        rows, last = await asyncio.to_thread(self._read_messages, cursor, 2000)
        days: dict[tuple[int, str], str] = {}
        for row in rows:
            if row["chat"] and row["text"]:
                day = datetime.datetime.fromtimestamp(row["when"]).strftime("%Y-%m-%d")
                days[(row["chat"], day)] = row["chat_name"]
        items: list[Item] = []
        for (chat, day), chat_name in list(days.items())[:budget]:
            messages = await asyncio.to_thread(self._day_messages, chat, day)
            if not messages:
                continue
            lines = [f"{datetime.datetime.fromtimestamp(m['when']).strftime('%H:%M')} "
                     f"{'Me' if m['mine'] else m['handle'] or 'Them'}: {m['text']}" for m in messages]
            name = chat_name or messages[0]["handle"] or f"chat {chat}"
            items.append(self._item(f"message://{chat}/{day}", "message", f"{name} messages on {day}",
                                    f"messages with {name} on {day}", "\n".join(lines),
                                    max(m["when"] for m in messages)))
        if len(days) <= budget:
            self.store.set_kv("files.sources.messages.rowid", last)
            self._more["messages"] = len(rows) >= 2000
        else:
            # More days than fit: the rest come next pass from the same cursor (their documents are rebuilt whole).
            self._more["messages"] = True
            done_rowids = [r["rowid"] for r in rows if (r["chat"], datetime.datetime.fromtimestamp(r["when"])
                                                        .strftime("%Y-%m-%d")) in dict(list(days.items())[:budget])]
            if done_rowids and all(r["rowid"] <= max(done_rowids) for r in rows[:len(done_rowids)]):
                self.store.set_kv("files.sources.messages.rowid", max(done_rowids))
        return items, []
