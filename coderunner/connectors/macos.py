"""Calendar, Reminders, Notes, Mail and Contacts through JavaScript for Automation.

Every call runs `osascript -l JavaScript` with a fixed script whose `run(argv)`
handler receives all dynamic values as separate argv strings. Nothing the
model or a third party supplies is ever spliced into script source, so a title
like `"); do shell script "rm -rf ~"` stays an inert string. Scripts print
JSON.

Connectors are opt-in: load_connectors() only returns tools for names listed
in config `connectors.enabled`, and only on macOS. The first call to each app
triggers macOS's Automation prompt for the calling process; run
`coderunner connectors test <name>` from Terminal to grant it interactively.

Known limits:
- Calendar's scripting interface filters on each event's own start/end, so a
  recurring series that began before the requested range is missed (or shows
  its first occurrence). Large calendars can also be slow, hence the longer
  Calendar timeout.
- Mail ids are Mail's numeric message ids, looked up in the given mailbox.
- mail_draft saves a draft. mail_send exists only where the owner has switched sending on for Mail
  (`connectors.send`), and every message is approved one at a time with its recipients and full text shown.
- settings_set changes only the settings named in SETTINGS (appearance, sound volume, mute): reversible ones,
  nothing to do with security, privacy, accounts or sharing. phone_call dials a number in FaceTime, which asks
  the owner to confirm; the owner speaks, the assistant does not.
"""

from __future__ import annotations

import asyncio
import html
import json
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from .redact import redact

OSASCRIPT = "/usr/bin/osascript"
CONNECTORS = ("calendar", "reminders", "notes", "mail", "contacts", "settings", "phone")
SENDERS = ("mail",)             # connectors that can send for the owner once connectors.send lists them
APPS = {"calendar": "Calendar", "reminders": "Reminders", "notes": "Notes", "mail": "Mail", "contacts": "Contacts",
        "settings": "System Events", "phone": "FaceTime"}
DEFAULT_TIMEOUT = 30.0
CALENDAR_TIMEOUT = 60.0
MAX_RANGE_DAYS = 366

Runner = Callable[[list[str], float], Awaitable[tuple[int, str, str]]]


class ConnectorError(Exception):
    pass


# ---- running scripts -----------------------------------------------------

async def run_process(argv: list[str], timeout: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    except asyncio.CancelledError:
        proc.kill()   # don't leave osascript (and its Apple event) running after the run is aborted
        raise
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def osascript_argv(script: str, args: list[str]) -> list[str]:
    # "--" ends option parsing, so values starting with "-" still arrive as argv.
    return [OSASCRIPT, "-l", "JavaScript", "-e", script, "--", *args]


def map_error(stderr: str, app: str) -> ConnectorError:
    text = stderr.strip()
    match = re.search(r"\((-?\d+)\)\s*$", text)
    code = int(match.group(1)) if match else None
    message = re.sub(r"^.*?execution error:\s*", "", text, flags=re.S)
    message = re.sub(r"^(?:Error:\s*)+", "", message)
    message = re.sub(r"\s*\(-?\d+\)\s*$", "", message)
    if code in (-1743, -1744):
        return ConnectorError(f"Not authorized: allow CodeRunner's process to control {app} under System Settings → "
                              "Privacy & Security → Automation (Calendar/Reminders/Notes/Mail/Contacts).")
    if code == -1712:
        return ConnectorError(f"{app} did not respond in time (Apple event timed out). Try again, or ask for less "
                              "data at once.")
    if code == -1728:
        return ConnectorError(f"Not found in {app}: {message}")
    return ConnectorError(f"{app}: {message or 'unknown error'}")


async def run_jxa(script: str, args: list[str], *, app: str, timeout: float = DEFAULT_TIMEOUT,
                  runner: Runner | None = None) -> Any:
    try:
        code, out, err = await (runner or run_process)(osascript_argv(script, args), timeout)
    except asyncio.TimeoutError:
        raise ConnectorError(f"{app} did not respond within {timeout:.0f} seconds. Try again, or ask for less "
                             "data at once.") from None
    except FileNotFoundError:
        raise ConnectorError("osascript is not available; connectors need macOS.") from None
    except OSError as exc:
        raise ConnectorError(f"Could not run osascript: {exc}") from None
    if code != 0:
        raise map_error(err or out, app)
    out = out.strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise ConnectorError(f"Unexpected output from {app}: {out[:200]}") from None


# ---- JXA scripts (dynamic values only ever arrive through argv) -----------

PRELUDE = r"""
function iso(d) {
  if (!d) return null;
  const p = n => String(n).padStart(2, '0');
  const off = -d.getTimezoneOffset(), a = Math.abs(off);
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + 'T' + p(d.getHours()) + ':' +
    p(d.getMinutes()) + ':' + p(d.getSeconds()) + (off >= 0 ? '+' : '-') + p(Math.floor(a / 60)) + ':' + p(a % 60);
}
function fatal(e) { return e && [-1743, -1744, -1712].indexOf(e.errorNumber) >= 0; }
function one(collection, name, what) {
  const found = collection.whose({name: name})();
  if (!found.length) throw new Error('No ' + what + ' named ' + JSON.stringify(name));
  return found[0];
}
function mailbox(Mail, name) {
  const special = {inbox: 'inbox', sent: 'sentMailbox', drafts: 'draftsMailbox', junk: 'junkMailbox',
                   trash: 'trashMailbox', outbox: 'outbox'};
  const key = special[(name || 'inbox').toLowerCase()];
  if (key) return Mail[key];
  for (const account of Mail.accounts()) {
    const found = account.mailboxes.whose({name: name})();
    if (found.length) return found[0];
  }
  return one(Mail.mailboxes, name, 'mailbox');
}
"""

CALENDAR_EVENTS = PRELUDE + r"""
function run(argv) {
  const start = new Date(Number(argv[0])), end = new Date(Number(argv[1])), max = Number(argv[2]);
  const Cal = Application('Calendar');
  const out = [], skipped = [];
  for (const cal of Cal.calendars()) {
    let name = '';
    try {
      name = cal.name();
      // One bulk fetch per property per calendar instead of a round-trip per event.
      const evs = cal.events.whose({_and: [{startDate: {_lessThan: end}}, {endDate: {_greaterThan: start}}]});
      const titles = evs.summary();
      if (!titles.length) continue;
      const starts = evs.startDate(), ends = evs.endDate(), locs = evs.location(), allDay = evs.alldayEvent();
      const ids = evs.uid();
      for (let i = 0; i < titles.length; i++) {
        out.push({t: starts[i] ? starts[i].getTime() : 0, e: {id: ids[i], title: titles[i], start: iso(starts[i]),
          end: iso(ends[i]), all_day: allDay[i], location: locs[i] || '', calendar: name}});
      }
    } catch (e) {
      if (fatal(e)) throw e;
      skipped.push(name);
    }
  }
  out.sort((a, b) => a.t - b.t);
  return JSON.stringify({events: out.slice(0, max).map(x => x.e), total: out.length, skipped_calendars: skipped});
}
"""

CALENDAR_CREATE = PRELUDE + r"""
function run(argv) {
  const [title, startMs, endMs, calName, notes, location] = argv;
  const Cal = Application('Calendar');
  let cal;
  if (calName) {
    cal = one(Cal.calendars, calName, 'calendar');
  } else {
    const writable = Cal.calendars.whose({writable: true})();
    if (!writable.length) throw new Error('No writable calendar found');
    cal = writable[0];
  }
  const props = {summary: title, startDate: new Date(Number(startMs)), endDate: new Date(Number(endMs))};
  if (notes) props.description = notes;
  if (location) props.location = location;
  const ev = Cal.Event(props);
  cal.events.push(ev);
  return JSON.stringify({created: true, id: ev.uid(), calendar: cal.name()});
}
"""

REMINDERS_LIST = PRELUDE + r"""
function run(argv) {
  const [listName, includeCompleted, maxStr] = argv;
  const R = Application('Reminders');
  const lists = listName ? [one(R.lists, listName, 'reminders list')] : R.lists();
  const out = [];
  for (const list of lists) {
    const name = list.name();
    const rs = includeCompleted === '1' ? list.reminders : list.reminders.whose({completed: false});
    const titles = rs.name();
    if (!titles.length) continue;
    const ids = rs.id(), dues = rs.dueDate(), done = rs.completed(), bodies = rs.body(), prio = rs.priority();
    for (let i = 0; i < titles.length; i++) {
      out.push({t: dues[i] ? dues[i].getTime() : Infinity, r: {id: ids[i], title: titles[i], due: iso(dues[i]),
        completed: done[i], notes: bodies[i] || '', priority: prio[i], list: name}});
    }
  }
  out.sort((a, b) => a.t === b.t ? 0 : a.t < b.t ? -1 : 1);
  return JSON.stringify({reminders: out.slice(0, Number(maxStr)).map(x => x.r), total: out.length});
}
"""

REMINDERS_CREATE = PRELUDE + r"""
function run(argv) {
  const [title, dueMs, notes, listName] = argv;
  const R = Application('Reminders');
  const list = listName ? one(R.lists, listName, 'reminders list') : R.defaultList();
  const props = {name: title};
  if (dueMs) props.dueDate = new Date(Number(dueMs));
  if (notes) props.body = notes;
  const reminder = R.Reminder(props);
  list.reminders.push(reminder);
  return JSON.stringify({created: true, id: reminder.id(), list: list.name()});
}
"""

NOTES_SEARCH = PRELUDE + r"""
function run(argv) {
  const [query, limitStr] = argv;
  const N = Application('Notes');
  const hits = N.notes.whose({_or: [{name: {_contains: query}}, {plaintext: {_contains: query}}]});
  const ids = hits.id();
  if (!ids.length) return JSON.stringify({notes: [], total: 0});
  const names = hits.name(), mods = hits.modificationDate();
  const order = ids.map((_, i) => i).sort((a, b) => mods[b] - mods[a]).slice(0, Number(limitStr));
  const needle = query.toLowerCase();
  const notes = order.map(i => {
    let snippet = '';
    try {
      const text = N.notes.byId(ids[i]).plaintext();
      const at = Math.max(0, text.toLowerCase().indexOf(needle));
      snippet = text.slice(Math.max(0, at - 80), at + 160).replace(/\s+/g, ' ').trim();
    } catch (e) {
      if (fatal(e)) throw e;
    }
    return {id: ids[i], name: names[i], modified: iso(mods[i]), snippet: snippet};
  });
  return JSON.stringify({notes: notes, total: ids.length});
}
"""

NOTES_READ = PRELUDE + r"""
function run(argv) {
  const note = Application('Notes').notes.byId(argv[0]);
  const body = note.plaintext();
  let folder = '';
  try { folder = note.container().name(); } catch (e) { if (fatal(e)) throw e; }
  return JSON.stringify({id: note.id(), name: note.name(), folder: folder, modified: iso(note.modificationDate()),
                         body: body});
}
"""

NOTES_CREATE = PRELUDE + r"""
function run(argv) {
  const N = Application('Notes');
  const note = N.Note({body: argv[0]});
  N.defaultAccount.defaultFolder.notes.push(note);
  return JSON.stringify({created: true, id: note.id(), name: note.name()});
}
"""

MAIL_RECENT = PRELUDE + r"""
function run(argv) {
  const [boxName, limitStr, unreadOnly] = argv;
  const Mail = Application('Mail');
  const box = mailbox(Mail, boxName);
  const msgs = unreadOnly === '1' ? box.messages.whose({readStatus: false}) : box.messages;
  const ids = msgs.id();
  if (!ids.length) return JSON.stringify({messages: [], total: 0});
  const dates = msgs.dateReceived(), subjects = msgs.subject(), senders = msgs.sender(), read = msgs.readStatus();
  const order = ids.map((_, i) => i).sort((a, b) => (dates[b] || 0) - (dates[a] || 0)).slice(0, Number(limitStr));
  return JSON.stringify({total: ids.length, messages: order.map(i => ({id: ids[i], subject: subjects[i] || '',
    sender: senders[i] || '', date: iso(dates[i]), read: read[i]}))});
}
"""

MAIL_READ = PRELUDE + r"""
function run(argv) {
  const [idStr, boxName] = argv;
  const Mail = Application('Mail');
  const found = mailbox(Mail, boxName).messages.whose({id: Number(idStr)})();
  if (!found.length) throw new Error('No message with id ' + idStr + ' in ' + JSON.stringify(boxName));
  const m = found[0];
  return JSON.stringify({id: Number(idStr), subject: m.subject() || '', sender: m.sender() || '',
    to: m.toRecipients.address(), cc: m.ccRecipients.address(), date: iso(m.dateReceived()), read: m.readStatus(),
    content: m.content() || ''});
}
"""

# Creates and saves a draft. There is intentionally no call that sends.
MAIL_DRAFT = PRELUDE + r"""
function run(argv) {
  const [to, cc, subject, body] = argv;
  const Mail = Application('Mail');
  const msg = Mail.OutgoingMessage({subject: subject, content: body, visible: false});
  Mail.outgoingMessages.push(msg);
  for (const address of to.split('\n').filter(Boolean)) msg.toRecipients.push(Mail.Recipient({address: address}));
  for (const address of cc.split('\n').filter(Boolean)) msg.ccRecipients.push(Mail.CcRecipient({address: address}));
  msg.save();
  return JSON.stringify({drafted: true, sent: false, subject: subject});
}
"""

# Sends at once, from the owner's default account. Offered only where connectors.send lists mail, and each
# message is approved by the owner with its recipients and whole text shown.
MAIL_SEND = PRELUDE + r"""
function run(argv) {
  const [to, cc, subject, body] = argv;
  const Mail = Application('Mail');
  const msg = Mail.OutgoingMessage({subject: subject, content: body, visible: false});
  Mail.outgoingMessages.push(msg);
  for (const address of to.split('\n').filter(Boolean)) msg.toRecipients.push(Mail.Recipient({address: address}));
  for (const address of cc.split('\n').filter(Boolean)) msg.ccRecipients.push(Mail.CcRecipient({address: address}));
  msg.send();
  return JSON.stringify({sent: true, to: to.split('\n').filter(Boolean), subject: subject});
}
"""

SETTINGS_GET = r"""
function run(argv) {
  const app = Application.currentApplication();
  app.includeStandardAdditions = true;
  const sound = app.getVolumeSettings();
  return JSON.stringify({appearance: Application('System Events').appearancePreferences.darkMode() ? 'dark' : 'light',
                         volume: sound.outputVolume, muted: sound.outputMuted});
}
"""

# One setting at a time, and only these three: the name and value were checked before they got here.
SETTINGS_SET = r"""
function run(argv) {
  const [name, value] = argv;
  const app = Application.currentApplication();
  app.includeStandardAdditions = true;
  if (name === 'appearance') Application('System Events').appearancePreferences.darkMode = (value === 'dark');
  else if (name === 'volume') app.setVolume(null, {outputVolume: Number(value)});
  else if (name === 'muted') app.setVolume(null, {outputMuted: value === 'true'});
  else throw new Error('Not a setting the assistant may change: ' + name);
  // What the Mac says afterwards, not what was asked for.
  const sound = app.getVolumeSettings();
  return JSON.stringify({changed: name, now: {
    appearance: Application('System Events').appearancePreferences.darkMode() ? 'dark' : 'light',
    volume: sound.outputVolume, muted: sound.outputMuted}});
}
"""

# Hands the number to FaceTime, which shows its own "Call" confirmation. Nothing speaks for the owner.
PHONE_CALL = r"""
function run(argv) {
  const app = Application.currentApplication();
  app.includeStandardAdditions = true;
  app.openLocation('tel:' + argv[0]);
  return JSON.stringify({dialling: argv[0], note: 'FaceTime asks the owner to confirm the call; the owner speaks.'});
}
"""

CONTACTS_FIND = PRELUDE + r"""
function run(argv) {
  const [query, limitStr] = argv;
  const people = Application('Contacts').people.whose({name: {_contains: query}});
  const names = people.name();
  if (!names.length) return JSON.stringify({contacts: [], total: 0});
  const ids = people.id(), orgs = people.organization();
  const phones = people.phones.value(), phoneLabels = people.phones.label();
  const emails = people.emails.value(), emailLabels = people.emails.label();
  const contacts = [];
  for (let i = 0; i < Math.min(names.length, Number(limitStr)); i++) {
    contacts.push({id: ids[i], name: names[i], organization: orgs[i] || '', phones: phones[i] || [],
      phone_labels: phoneLabels[i] || [], emails: emails[i] || [], email_labels: emailLabels[i] || []});
  }
  return JSON.stringify({contacts: contacts, total: names.length});
}
"""

# Cheap reads used by `coderunner connectors test <name>` to trigger the Automation prompt.
PROBES = {
    "calendar": "function run(argv) { return JSON.stringify(Application('Calendar').calendars.name()) }",
    "reminders": "function run(argv) { return JSON.stringify(Application('Reminders').lists.name()) }",
    "notes": "function run(argv) { return JSON.stringify(Application('Notes').accounts.name()) }",
    "mail": "function run(argv) { return JSON.stringify(Application('Mail').accounts.name()) }",
    "contacts": "function run(argv) { return JSON.stringify(Application('Contacts').groups.name()) }",
    "settings": "function run(argv) { return JSON.stringify([Application('System Events').appearancePreferences.darkMode()]) }",
}


# ---- argument handling ---------------------------------------------------

def _text(args: dict, key: str, required: bool = False) -> str:
    value = args.get(key)
    if value is None or value == "":
        if required:
            raise ValueError(f"{key} is required")
        return ""
    value = value if isinstance(value, str) else str(value)
    if "\x00" in value:
        raise ValueError(f"{key} contains a NUL character")
    if required and not value.strip():
        raise ValueError(f"{key} is required")
    return value


def _flag(args: dict, key: str) -> str:
    value = args.get(key)
    if isinstance(value, str):
        value = value.strip().lower() in ("1", "true", "yes")
    return "1" if value else ""


def _int(args: dict, key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(args.get(key) if args.get(key) not in (None, "") else default)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    return max(low, min(high, value))


def _message_id(args: dict) -> str:
    value = str(args.get("id", "")).strip()
    if not value.isdigit() or int(value) <= 0:
        raise ValueError("id must be a positive integer from mail_recent")
    return str(int(value))


def parse_time(value: Any, key: str) -> datetime:
    """ISO 8601 date or date-time; values without an offset are local time."""
    text = str(value or "").strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{key} must be an ISO 8601 date or date-time, e.g. 2026-10-12T15:00") from None
    return moment.astimezone() if moment.tzinfo is None else moment


def _ms(moment: datetime) -> str:
    return str(int(moment.timestamp() * 1000))


def _when(value: Any) -> str:
    """The value as given, plus how it was interpreted, for approval prompts."""
    try:
        moment = parse_time(value, "time").astimezone()
    except ValueError:
        return str(value)
    return f"{value} ({moment.strftime('%a %d %b %Y %H:%M %Z').strip()})"


_ADDRESS_PART = r"[^\s@<>\"',;\x00-\x1f\x7f]+"
ADDRESS = re.compile(rf"{_ADDRESS_PART}@{_ADDRESS_PART}\.{_ADDRESS_PART}")


def _addresses(args: dict, key: str, required: bool = False) -> list[str]:
    raw = args.get(key) or ""
    items = raw if isinstance(raw, list) else re.split(r"[,;\s]+", str(raw))
    found = [str(item).strip() for item in items if str(item).strip()]
    if required and not found:
        raise ValueError(f"{key} needs at least one email address")
    for address in found:
        if not ADDRESS.fullmatch(address):
            raise ValueError(f"{key}: {address!r} is not a plain email address (use name@example.com)")
    return found


def _show(value: Any) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def note_html(title: str, body: str) -> str:
    """Plain text to the simple HTML Notes stores; the first line becomes the note's name."""
    parts = [f"<h1>{html.escape(title)}</h1>"]
    for line in body.splitlines():
        parts.append(f"<div>{html.escape(line)}</div>" if line.strip() else "<div><br></div>")
    return "".join(parts)


def _label(value: Any) -> str:
    return re.sub(r"^_\$!<(.*)>!\$_$", r"\1", str(value or ""))


# ---- tool specs ----------------------------------------------------------

def _params(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}


@dataclass
class ConnectorTool:
    name: str
    connector: str
    description: str
    parameters: dict
    script: str
    build: Callable[[dict], list[str]]
    write: bool = False
    untrusted: bool = False
    timeout: float = DEFAULT_TIMEOUT
    post: Callable[[Any], Any] | None = None
    summary: Callable[[dict], str] | None = None
    runner: Runner | None = field(default=None, repr=False)
    action: str = ""            # the policy kind when it is not connector.read or connector.write
    send: bool = False          # offered only where connectors.send lists its connector
    target: Callable[[dict], str] | None = None        # what the approval is about (recipients, a setting, a number)
    who: Callable[[dict], list[str]] | None = None     # the people it reaches, checked against what the owner gave

    def describe(self, args: dict) -> str:
        if self.summary is not None:
            return self.summary(args)
        return f"{self.name} {json.dumps(args, ensure_ascii=False, default=str)}"

    async def call(self, args: dict) -> Any:
        argv = self.build(args)
        result = await run_jxa(self.script, argv, app=APPS[self.connector], timeout=self.timeout,
                               runner=self.runner)
        return self.post(result) if self.post else result


def _calendar_events(args: dict) -> list[str]:
    start, end = parse_time(args.get("start"), "start"), parse_time(args.get("end"), "end")
    if end <= start:
        raise ValueError("end must be after start")
    if end - start > timedelta(days=MAX_RANGE_DAYS):
        raise ValueError(f"the range can be at most {MAX_RANGE_DAYS} days")
    return [_ms(start), _ms(end), str(_int(args, "max", 100, 1, 200))]


def _calendar_create(args: dict) -> list[str]:
    title = _text(args, "title", True)
    start, end = parse_time(args.get("start"), "start"), parse_time(args.get("end"), "end")
    if end <= start:
        raise ValueError("end must be after start")
    return [title, _ms(start), _ms(end), _text(args, "calendar"), _text(args, "notes"), _text(args, "location")]


def _describe_calendar_create(args: dict) -> str:
    lines = ["Create a Calendar event", f"Title: {args.get('title', '')}", f"Start: {_when(args.get('start'))}",
             f"End: {_when(args.get('end'))}", f"Calendar: {args.get('calendar') or '(first writable calendar)'}"]
    if args.get("location"):
        lines.append(f"Location: {args['location']}")
    if args.get("notes"):
        lines.append(f"Notes:\n{args['notes']}")
    return "\n".join(lines)


def _reminders_create(args: dict) -> list[str]:
    due = _ms(parse_time(args.get("due"), "due")) if args.get("due") else ""
    return [_text(args, "title", True), due, _text(args, "notes"), _text(args, "list")]


def _describe_reminders_create(args: dict) -> str:
    lines = ["Create a reminder", f"Title: {args.get('title', '')}"]
    if args.get("due"):
        lines.append(f"Due: {_when(args['due'])}")
    lines.append(f"List: {args.get('list') or '(default list)'}")
    if args.get("notes"):
        lines.append(f"Notes:\n{args['notes']}")
    return "\n".join(lines)


def _notes_create(args: dict) -> list[str]:
    return [note_html(_text(args, "title", True), _text(args, "body"))]


def _describe_notes_create(args: dict) -> str:
    return f"Create a note in Notes\nTitle: {args.get('title', '')}\nBody:\n{args.get('body', '')}"


def _mail_draft(args: dict) -> list[str]:
    to, cc = _addresses(args, "to", True), _addresses(args, "cc")
    return ["\n".join(to), "\n".join(cc), _text(args, "subject"), _text(args, "body")]


def _describe_mail_draft(args: dict) -> str:
    lines = ["Save a draft in Mail (it will NOT be sent)", f"To: {_show(args.get('to', ''))}"]
    if args.get("cc"):
        lines.append(f"Cc: {_show(args['cc'])}")
    lines += [f"Subject: {args.get('subject', '')}", f"Body:\n{args.get('body', '')}"]
    return "\n".join(lines)


def _describe_mail_send(args: dict) -> str:
    lines = ["Send this email from your Mail account now", f"To: {_show(args.get('to', ''))}"]
    if args.get("cc"):
        lines.append(f"Cc: {_show(args['cc'])}")
    lines += [f"Subject: {args.get('subject', '')}", f"Body:\n{args.get('body', '')}"]
    return "\n".join(lines)


# What settings_set may change, and to what. Nothing about security, privacy, accounts, sharing or networks.
SETTINGS = {"appearance": ("dark", "light"), "volume": range(0, 101), "muted": ("true", "false")}


def _setting(args: dict) -> list[str]:
    name = _text(args, "setting", True).lower()
    if name not in SETTINGS:
        raise ValueError(f"The assistant may change only: {', '.join(SETTINGS)}.")
    raw = args.get("value")
    value = str(raw).lower() if isinstance(raw, bool) else str(raw if raw is not None else "").strip().lower().rstrip("%")
    if name == "volume":
        if not value.isdigit() or int(value) not in SETTINGS["volume"]:
            raise ValueError("volume is a whole number from 0 to 100.")
        value = str(int(value))
    elif value not in SETTINGS[name]:
        raise ValueError(f"{name} is one of: {', '.join(SETTINGS[name])}.")
    return [name, value]


def _describe_setting(args: dict) -> str:
    name, value = _setting(args)
    return {"appearance": f"Change this Mac's appearance to {value}",
            "volume": f"Set this Mac's sound volume to {value}%",
            "muted": "Mute this Mac's sound" if value == "true" else "Turn this Mac's sound back on"}[name]


PHONE_NUMBER = re.compile(r"\+?\d{7,15}")


def _number(args: dict) -> list[str]:
    """A full phone number, digits only with an optional +. Short numbers (emergency and service codes among
    them) are refused: those are for a person to dial."""
    number = re.sub(r"[\s().-]", "", _text(args, "number", True))
    if not PHONE_NUMBER.fullmatch(number):
        raise ValueError("number must be a full phone number: 7 to 15 digits, with an optional leading +.")
    return [number]


def _describe_call(args: dict) -> str:
    who = f" ({args['name']})" if args.get("name") else ""
    return f"Call {_number(args)[0]}{who} from this Mac. FaceTime asks you to confirm, and you do the talking."


def _redact_recent(result: Any) -> Any:
    for message in (result or {}).get("messages", []):
        message["subject"] = redact(message.get("subject") or "")
    return result


def _redact_message(result: Any) -> Any:
    if isinstance(result, dict):
        result["subject"] = redact(result.get("subject") or "")
        result["content"] = redact(result.get("content") or "")
    return result


def _contacts(result: Any) -> Any:
    people = []
    for person in (result or {}).get("contacts", []):
        people.append({
            "name": person.get("name", ""), "organization": person.get("organization", ""),
            "phones": [{"label": _label(kind), "value": v} for v, kind in zip(person.get("phones", []),
                                                                        person.get("phone_labels", []))],
            "emails": [{"label": _label(kind), "value": v} for v, kind in zip(person.get("emails", []),
                                                                        person.get("email_labels", []))],
        })
    return {"contacts": people, "total": (result or {}).get("total", len(people))}


def all_tools(runner: Runner | None = None) -> list[ConnectorTool]:
    def tool(*args, **kwargs) -> ConnectorTool:
        return ConnectorTool(*args, runner=runner, **kwargs)

    return [
        tool("calendar_events", "calendar",
             "List Calendar events overlapping a time range. start/end are ISO 8601 (local time unless an offset "
             "is given; end is exclusive). Recurring events that started before the range may be missing.",
             _params({"start": STR, "end": STR, "max": INT}, ["start", "end"]), CALENDAR_EVENTS, _calendar_events,
             untrusted=True, timeout=CALENDAR_TIMEOUT),
        tool("calendar_create", "calendar",
             "Create a Calendar event. start/end are ISO 8601 local times.",
             _params({"title": STR, "start": STR, "end": STR, "calendar": STR, "notes": STR, "location": STR},
                     ["title", "start", "end"]), CALENDAR_CREATE, _calendar_create, write=True,
             timeout=CALENDAR_TIMEOUT, summary=_describe_calendar_create),
        tool("reminders_list", "reminders", "List reminders, optionally from one list. Incomplete only by default.",
             _params({"list": STR, "include_completed": BOOL}), REMINDERS_LIST,
             # Shared lists can contain items written by other people.
             lambda a: [_text(a, "list"), _flag(a, "include_completed"), "200"], untrusted=True),
        tool("reminders_create", "reminders",
             "Create a reminder. due is an optional ISO 8601 local time.",
             _params({"title": STR, "due": STR, "notes": STR, "list": STR}, ["title"]), REMINDERS_CREATE, _reminders_create,
             write=True, summary=_describe_reminders_create),
        tool("notes_search", "notes", "Search Notes by title or text. Returns id, name and a snippet.",
             _params({"query": STR, "limit": INT}, ["query"]), NOTES_SEARCH,
             lambda a: [_text(a, "query", True), str(_int(a, "limit", 10, 1, 50))], untrusted=True),
        tool("notes_read", "notes", "Read a note's full text by id (from notes_search).",
             _params({"id": STR}, ["id"]), NOTES_READ, lambda a: [_text(a, "id", True)], untrusted=True),
        tool("notes_create", "notes", "Create a note from a title and plain-text body.",
             _params({"title": STR, "body": STR}, ["title", "body"]), NOTES_CREATE, _notes_create, write=True,
             summary=_describe_notes_create),
        tool("mail_recent", "mail",
             "List recent Mail messages (id, subject, sender, date, read). mailbox: inbox (default), sent, drafts, "
             "junk, trash or a mailbox name. Codes and sign-in links are redacted.",
             _params({"limit": INT, "mailbox": STR, "unread_only": BOOL}), MAIL_RECENT,
             lambda a: [_text(a, "mailbox") or "inbox", str(_int(a, "limit", 20, 1, 50)), _flag(a, "unread_only")],
             untrusted=True, post=_redact_recent),
        tool("mail_read", "mail", "Read one Mail message by id (from mail_recent). Codes and sign-in links are "
             "redacted.", _params({"id": INT, "mailbox": STR}, ["id"]), MAIL_READ,
             lambda a: [_message_id(a), _text(a, "mailbox") or "inbox"],
             untrusted=True, post=_redact_message),
        tool("mail_draft", "mail",
             "Save an email as a draft in Mail for the owner to review and send. Never sends. Needs the owner's "
             "approval. to/cc are comma-separated plain email addresses.",
             _params({"to": STR, "subject": STR, "body": STR, "cc": STR}, ["to", "subject", "body"]), MAIL_DRAFT,
             _mail_draft, write=True, summary=_describe_mail_draft),
        tool("contacts_find", "contacts", "Find contacts by name. Returns names, phone numbers and emails.",
             _params({"name": STR}, ["name"]), CONTACTS_FIND, lambda a: [_text(a, "name", True), "25"],
             untrusted=True, post=_contacts),
        tool("mail_send", "mail",
             "Send an email from the owner's Mail account. The owner sees the recipients and the whole text and "
             "approves each message; use mail_draft when they only asked for a draft. to/cc are comma-separated "
             "plain email addresses.",
             _params({"to": STR, "subject": STR, "body": STR, "cc": STR}, ["to", "subject", "body"]), MAIL_SEND,
             _mail_draft, write=True, summary=_describe_mail_send, action="connector.send", send=True,
             target=lambda a: "mail:" + ",".join(_addresses(a, "to", True) + _addresses(a, "cc")),
             who=lambda a: _addresses(a, "to", True) + _addresses(a, "cc")),
        tool("settings_get", "settings", "Read this Mac's appearance (dark or light), sound volume and mute.",
             _params({}), SETTINGS_GET, lambda a: []),
        tool("settings_set", "settings",
             "Change one setting of this Mac, with the owner's approval each time: appearance (dark or light), "
             "volume (0 to 100) or muted (true or false). Nothing else can be changed.",
             _params({"setting": {"type": "string", "enum": list(SETTINGS)}, "value": STR}, ["setting", "value"]),
             SETTINGS_SET, _setting, write=True, summary=_describe_setting, action="system.setting",
             target=lambda a: _setting(a)[0]),
        tool("phone_call", "phone",
             "Dial a phone number for the owner from this Mac (FaceTime, through their iPhone). The owner approves, "
             "confirms in FaceTime and speaks; you cannot talk on the call. name is who it is, for the approval.",
             _params({"number": STR, "name": STR}, ["number"]), PHONE_CALL, _number, write=True,
             summary=_describe_call, action="phone.call", target=lambda a: _number(a)[0], who=_number),
    ]


def load_connectors(config, *, runner: Runner | None = None, system: str | None = None) -> list[ConnectorTool]:
    """Tools for the connectors enabled in config, on macOS only."""
    if (system or platform.system()) != "Darwin":
        return []
    enabled = set(config.get("connectors.enabled", []) or [])
    sending = set(config.get("connectors.send", []) or [])
    return [tool for tool in all_tools(runner)
            if tool.connector in enabled and (not tool.send or tool.connector in sending)]


async def check(name: str, *, runner: Runner | None = None) -> str:
    """Run a harmless read so macOS asks for (or reports) Automation access."""
    if name not in PROBES:
        raise ValueError(f"Unknown connector {name!r}. Choose from: {', '.join(CONNECTORS)}")
    items = await run_jxa(PROBES[name], [], app=APPS[name], runner=runner,
                          timeout=CALENDAR_TIMEOUT if name == "calendar" else DEFAULT_TIMEOUT)
    return f"{APPS[name]}: access works ({len(items or [])} found)."


async def notes_create(title: str, body: str, *, runner: Runner | None = None) -> dict:
    """Create a note in Apple Notes directly (used by owner-initiated quick actions)."""
    return await run_jxa(NOTES_CREATE, _notes_create({"title": title, "body": body}),
                         app="Notes", runner=runner)


async def mail_draft(to: str, subject: str, body: str, cc: str = "", *, runner: Runner | None = None) -> dict:
    """Save a draft in Apple Mail without sending (recipient may be blank for owner completion)."""
    to_list = _addresses({"to": to}, "to", False) if str(to or "").strip() else []
    cc_list = _addresses({"cc": cc}, "cc", False) if str(cc or "").strip() else []
    args = ["\n".join(to_list), "\n".join(cc_list), str(subject or "").strip(), str(body or "")]
    return await run_jxa(MAIL_DRAFT, args, app="Mail", runner=runner)
