import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import httpx

from coderunner.core.audit import AuditLog
from coderunner.channels import IncomingMessage, load_channels
from coderunner.channels.imessage import IMessageChannel, IMessageSendError
from coderunner.channels.signal import SignalChannel, SignalError
from coderunner.channels.base import NO_VOICE, PAIRED, split_text, utf16_len
from coderunner.channels.telegram import TelegramChannel
from coderunner.channels.whatsapp import WhatsAppChannel, handle_of
from coderunner.core.config import Config
from coderunner.safety.policy import Action, Policy
from coderunner.safety.secrets_store import SecretStore
from coderunner.core.store import Store

TOKEN = "123456:SECRET-bot_token-value"
OWNER = "+15550001111"
STRANGER = "+15559999999"

logging.getLogger("coderunner.channels").setLevel(logging.ERROR)  # expected errors are asserted via audit


def write_script(path: pathlib.Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return str(path)


async def wait_for(predicate, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.02)


class ChannelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        config = Config({}, path=self.dir / "config.json")
        store = Store(":memory:")
        audit = AuditLog(self.dir / "audit.jsonl")
        self.svc = SimpleNamespace(config=config, store=store, audit=audit, policy=Policy(config, store, audit),
                                   secrets=SecretStore(self.dir / "secrets.json"), voice=None)
        self.received: list[IncomingMessage] = []
        self.reply = "ok"
        self.channels = []

    async def asyncTearDown(self):
        for channel in self.channels:
            await asyncio.wait_for(channel.stop(), 10)

    def tearDown(self):
        self.svc.store.close()
        self.tmp.cleanup()

    async def on_message(self, message):
        self.received.append(message)
        return self.reply

    def events(self, name):
        return [e for e in self.svc.audit.tail(1000) if e["event"] == name]


class SplitTextTests(unittest.TestCase):
    def test_prefers_line_and_word_breaks_and_respects_limit(self):
        text = "\n".join(f"line {i} " + "word " * 20 for i in range(50))
        chunks = split_text(text, 300)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())
        self.assertTrue(all(c.startswith("line") for c in chunks))

    def test_hard_cut_and_utf16_measure(self):
        self.assertEqual(split_text("x" * 10, 4), ["xxxx", "xxxx", "xx"])
        emoji = "😀" * 5000
        chunks = split_text(emoji, 4096, utf16_len)
        self.assertTrue(all(utf16_len(c) <= 4096 for c in chunks))
        self.assertEqual("".join(chunks), emoji)
        self.assertEqual(split_text("   ", 10), [])


# ------------------------------------------------------------------ Telegram

class FakeBot:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.updates: list[list[dict]] = []
        self.failures: list[Exception] = []
        self.files: dict[str, bytes] = {}

    @staticmethod
    def ok(result):
        return httpx.Response(200, json={"ok": True, "result": result})

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith(f"/file/bot{TOKEN}/"):
            return httpx.Response(200, content=self.files[path.rsplit("/", 1)[-1]])
        method = path.rsplit("/", 1)[-1]
        is_json = request.headers.get("content-type", "").startswith("application/json")
        payload = json.loads(request.content) if is_json else {"multipart": True}
        self.calls.append((method, payload))
        if method == "getUpdates":
            if self.failures:
                raise self.failures.pop(0)
            if not self.updates:
                await asyncio.sleep(0.05)
                return self.ok([])
            return self.ok(self.updates.pop(0))
        if method == "getFile":
            return self.ok({"file_id": payload["file_id"], "file_path": f"media/{payload['file_id']}", "file_size": 10})
        if method == "sendMessage":
            return self.ok({"message_id": len(self.calls)})
        return self.ok(True)

    def sent(self, method="sendMessage"):
        return [payload for name, payload in self.calls if name == method]


def tg_message(update_id, user, text=None, chat_type="private", **extra):
    chat_id = user if chat_type == "private" else -100
    message = {"message_id": update_id, "from": {"id": user}, "chat": {"id": chat_id, "type": chat_type}, **extra}
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


class FakeVoice:
    def __init__(self, transcript="turn on the lights", speech_dir=None):
        self.transcript = transcript
        self.paths = []
        self.spoken = []
        self.speech_dir = speech_dir

    def available(self):
        return True

    async def transcribe(self, path):
        self.paths.append(str(path))
        assert os.path.exists(path)
        return self.transcript

    def can_speak(self):
        return self.speech_dir is not None

    async def speak(self, text):
        self.spoken.append(text)
        path = pathlib.Path(self.speech_dir) / "speech.ogg"
        path.write_bytes(b"OggS-speech")
        return str(path)

    def discard(self, path):
        os.unlink(path)


class TelegramTests(ChannelTestCase):
    def setUp(self):
        super().setUp()
        self.svc.secrets.set("telegram_bot_token", TOKEN)
        self.bot = FakeBot()

    def channel(self, owners=("42",), **settings):
        channel = TelegramChannel({"enabled": True, "owners": list(owners), **settings}, self.svc, self.on_message)
        channel.transport = httpx.MockTransport(self.bot.handler)
        channel.retry_delay = lambda attempt: 0.01
        self.channels.append(channel)
        return channel

    async def test_configured_needs_token_in_secret_store(self):
        self.assertTrue(self.channel().configured())
        self.svc.secrets.delete("telegram_bot_token")
        self.assertFalse(self.channel().configured())
        self.svc.config.set("channels", {"telegram": {"enabled": True}, "whatsapp": {"enabled": True}})
        self.assertEqual(load_channels(self.svc.config, self.svc, self.on_message), {})

    async def test_pairing_from_private_chat(self):
        channel = self.channel(owners=())
        code = channel.start_pairing()
        self.bot.updates.append([tg_message(1, 42, "/pair 000000")])
        await channel.poll_once()
        self.assertEqual(self.bot.sent(), [])
        self.bot.updates.append([tg_message(2, 42, f"/pair {code}")])
        await channel.poll_once()
        self.assertEqual(self.bot.sent()[-1]["text"], PAIRED)
        self.assertEqual(self.bot.sent()[-1]["chat_id"], "42")
        self.assertEqual(self.svc.config.get("channels.telegram.owners"), ["42"])
        self.assertIn('"42"', (self.dir / "config.json").read_text())
        self.assertEqual(self.received, [])

    async def test_non_owner_and_group_messages_are_ignored_silently(self):
        channel = self.channel()
        self.bot.updates.append([tg_message(1, 99, "do something for me"),
                                 tg_message(2, 42, "hello group", chat_type="group")])
        await channel.poll_once()
        await channel.drain()
        self.assertEqual(self.received, [])
        self.assertEqual(self.bot.sent(), [])
        ignored = self.events("channel.ignored")
        self.assertEqual([e["sender"] for e in ignored], ["99", "42"])
        self.assertTrue(all("text" not in e and "do something" not in json.dumps(e) for e in ignored))

    async def test_owner_message_reaches_agent_and_long_reply_is_chunked(self):
        self.reply = " ".join(f"line{i} 😀" for i in range(2500))
        channel = self.channel()
        self.bot.updates.append([tg_message(1, 42, "hello")])
        await channel.poll_once()
        await channel.drain()
        (message,) = self.received
        self.assertEqual((message.channel, message.chat_id, message.sender, message.text, message.sender_is_owner),
                         ("telegram", "42", "42", "hello", True))
        chunks = [p["text"] for p in self.bot.sent()]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(utf16_len(c) <= 4096 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), self.reply.split())
        self.assertTrue(all(p["link_preview_options"] == {"is_disabled": True} for p in self.bot.sent()))

    async def callback(self, channel, user, data, message_text="prompt"):
        self.bot.updates.append([{"update_id": 50, "callback_query": {
            "id": "cq1", "from": {"id": user}, "data": data,
            "message": {"message_id": 7, "chat": {"id": user}, "text": message_text}}}])
        await channel.poll_once()

    async def test_callback_allow_for_a_day_resolves_policy_approval(self):
        channel = self.channel()
        approval = await self.svc.policy.request(Action("egress.connect", "example.com:443", "Connect to example.com"),
                                                 origin="agent")
        await channel.notify_approval(approval)
        prompt = self.bot.sent()[-1]
        self.assertEqual(prompt["chat_id"], "42")
        buttons = prompt["reply_markup"]["inline_keyboard"][0]
        self.assertEqual([b["callback_data"] for b in buttons],
                         [f"a:{approval.id}:once", f"a:{approval.id}:day", f"d:{approval.id}"])
        self.assertTrue(all(len(b["callback_data"].encode()) <= 64 for b in buttons))
        waiter = asyncio.create_task(self.svc.policy.wait(approval.id, 5))
        await self.callback(channel, 42, f"a:{approval.id}:day", prompt["text"])
        self.assertTrue(await waiter)
        decided = self.svc.policy.get(approval.id)
        self.assertEqual((decided.status, decided.scope, decided.decided_by), ("approved", "day", "telegram:42"))
        self.assertEqual(self.bot.sent("answerCallbackQuery"), [{"callback_query_id": "cq1"}])
        self.assertEqual(self.bot.sent("editMessageReplyMarkup"),
                         [{"chat_id": 42, "message_id": 7, "reply_markup": {"inline_keyboard": []}}])
        self.assertTrue(self.bot.sent("editMessageText")[0]["text"].endswith("Approved (day)."))
        self.assertTrue(any(g["kind"] == "egress.connect" for g in self.svc.policy.grants()))

    async def test_long_approval_is_sent_in_full_with_buttons_last(self):
        channel = self.channel(owners=("42", "43"))
        code = "\n".join(f"print('step {i} 😀')" for i in range(900))
        action = Action("sandbox.exec", "s1", f"Run in the sandbox:\n{code}")
        approval = await self.svc.policy.request(action, origin="agent")
        await channel.notify_approval(approval)
        parts = [p for p in self.bot.sent() if p["chat_id"] == "42"]
        count = len(parts)
        self.assertGreater(count, 3)
        self.assertEqual(len(self.bot.sent()), 2 * count)  # every owner gets every part
        self.assertTrue(all(utf16_len(p["text"]) <= 4096 for p in parts))
        self.assertEqual(["reply_markup" in p for p in parts], [False] * (count - 1) + [True])
        self.assertTrue(all(p["text"].startswith(f"({i}/{count}) ") for i, p in enumerate(parts, 1)))
        body = "\n".join(p["text"].split(" ", 1)[1] for p in parts)
        self.assertTrue(all(f"print('step {i} 😀')" in body for i in range(900)))
        # Deciding on a full-length part never truncates it: the outcome goes out separately.
        await self.callback(channel, 42, f"a:{approval.id}:once", "x" * 4090)
        self.assertEqual(self.bot.sent("editMessageText"), [])
        self.assertEqual(self.bot.sent()[-1]["text"], "Approved (once).")

    async def test_pairing_keeps_config_edits_made_since_startup(self):
        path = self.dir / "config.json"
        path.write_text(json.dumps({"channels": {"telegram": {"enabled": True, "owners": []},
                                                 "imessage": {"enabled": True}},
                                    "browser": {"allow": ["example.com"]}}))
        # The daemon's copy predates those CLI edits and still lists an owner the CLI removed.
        self.svc.config.set("channels.telegram.owners", ["7"])
        self.svc.config.set("models.chat", "stale-model")
        channel = self.channel(owners=("7",))
        code = channel.start_pairing()
        self.bot.updates.append([tg_message(1, 42, f"/pair {code}")])
        await channel.poll_once()
        saved = json.loads(path.read_text())
        self.assertEqual(saved["channels"]["telegram"]["owners"], ["42"])
        self.assertTrue(saved["channels"]["imessage"]["enabled"])
        self.assertEqual(saved["browser"]["allow"], ["example.com"])
        self.assertNotEqual(saved["models"]["chat"], "stale-model")
        self.assertEqual(self.svc.config.get("channels.telegram.owners"), ["42"])
        self.assertEqual(channel.owners(), ["42"])

    async def test_high_risk_offers_once_or_deny_and_deny_works(self):
        channel = self.channel()
        approval = await self.svc.policy.request(Action("channel.send", "+15551230000", "Message Bob"), origin="agent")
        await channel.notify_approval(approval)
        buttons = self.bot.sent()[-1]["reply_markup"]["inline_keyboard"][0]
        self.assertEqual([b["text"] for b in buttons], ["Allow once", "Deny"])
        waiter = asyncio.create_task(self.svc.policy.wait(approval.id, 5))
        await self.callback(channel, 42, f"d:{approval.id}")
        self.assertFalse(await waiter)
        self.assertEqual(self.svc.policy.get(approval.id).status, "denied")
        # A second tap on a stale button is answered without changing anything.
        await self.callback(channel, 42, f"a:{approval.id}:once")
        self.assertEqual(self.svc.policy.get(approval.id).status, "denied")
        self.assertIn("Not applied", self.bot.sent("editMessageText")[-1]["text"])

    async def test_callback_from_non_owner_is_rejected(self):
        channel = self.channel()
        action = Action("egress.connect", "evil.example:443", "Connect")
        approval = await self.svc.policy.request(action, origin="agent")
        await self.callback(channel, 99, f"a:{approval.id}:always")
        self.assertEqual(self.svc.policy.get(approval.id).status, "pending")
        self.assertEqual(self.bot.sent("answerCallbackQuery"), [{"callback_query_id": "cq1", "text": "Not allowed."}])
        self.assertEqual(self.bot.sent("editMessageText"), [])
        self.assertEqual(self.events("channel.ignored")[-1]["sender"], "99")

    async def test_text_approval_reply_also_works(self):
        channel = self.channel()
        approval = await self.svc.policy.request(Action("browser.navigate", "example.org", "Open"), origin="agent")
        self.bot.updates.append([tg_message(1, 42, f"approve {approval.id} session")])
        await channel.poll_once()
        self.assertEqual(self.svc.policy.get(approval.id).status, "approved")
        self.assertTrue(self.bot.sent()[-1]["text"].startswith("Approved (session)"))
        self.assertEqual(self.received, [])

    async def test_offset_is_persisted_and_resumed(self):
        channel = self.channel()
        self.bot.updates.append([tg_message(10, 99, "a"), tg_message(11, 99, "b")])
        await channel.poll_once()
        await channel.poll_once()
        self.assertNotIn("offset", self.bot.sent("getUpdates")[0])
        self.assertEqual(self.bot.sent("getUpdates")[1]["offset"], 12)
        restarted = self.channel()
        await restarted.poll_once()
        self.assertEqual(self.bot.sent("getUpdates")[-1]["offset"], 12)
        self.assertEqual(self.bot.sent("getUpdates")[-1]["allowed_updates"], ["message", "callback_query"])

    async def test_network_errors_back_off_without_killing_the_loop(self):
        channel = self.channel()
        leaky = httpx.Request("POST", f"https://api.telegram.org/bot{TOKEN}/getUpdates")
        self.bot.failures = [httpx.ConnectError("connection refused"),
                             httpx.ConnectError(f"failed to reach {leaky.url}"),
                             httpx.ReadTimeout("timed out")]
        self.bot.updates.append([tg_message(20, 42, "still there?")])
        await channel.start()
        await wait_for(lambda: self.received)
        self.assertEqual(self.received[0].text, "still there?")
        self.assertTrue(all(not task.done() for task in channel._loops))
        self.assertGreaterEqual(len(self.events("channel.error")), 1)
        self.assertNotIn(TOKEN.split(":")[1], (self.dir / "audit.jsonl").read_text())
        await channel.stop()
        self.assertEqual(channel._loops, set())

    async def test_request_log_never_shows_the_token(self):
        channel = self.channel()
        with self.assertLogs("httpx", logging.INFO) as logs:
            await channel.poll_once()
        self.assertIn("/bot<redacted>/getUpdates", logs.output[0])
        self.assertNotIn(TOKEN, "\n".join(logs.output))

    async def test_voice_note_without_engine_gets_a_hint(self):
        channel = self.channel()
        self.bot.updates.append([tg_message(1, 42, voice={"file_id": "v1", "file_size": 10})])
        await channel.poll_once()
        await channel.drain()
        self.assertEqual(self.received, [])
        self.assertEqual(self.bot.sent()[-1]["text"], NO_VOICE)
        self.assertEqual(self.bot.sent("getFile"), [])

    async def test_voice_note_is_transcribed_and_photo_attached(self):
        self.svc.voice = FakeVoice()
        self.bot.files = {"v1": b"OggS-fake", "small": b"tiny", "big": b"\xff\xd8large-jpeg"}
        channel = self.channel()
        photo = [{"file_id": "small", "width": 90, "height": 90}, {"file_id": "big", "width": 1280, "height": 960}]
        self.bot.updates.append([tg_message(1, 42, voice={"file_id": "v1", "file_size": 9}),
                                 tg_message(2, 42, caption="what is this?", photo=photo)])
        await channel.poll_once()
        await channel.drain()
        voice, picture = self.received
        # The transcript is the text; audio_path stays unset so nothing transcribes it twice.
        self.assertEqual((voice.text, voice.audio_path), ("turn on the lights", None))
        self.assertFalse(os.path.exists(self.svc.voice.paths[0]))
        self.assertEqual(picture.text, "what is this?")
        self.assertEqual(picture.images, [("image/jpeg", base64.b64encode(b"\xff\xd8large-jpeg").decode())])
        self.assertEqual(self.bot.sent("sendVoice"), [])  # voice_replies is off by default

    async def test_voice_replies_only_answer_voice_notes(self):
        self.svc.voice = FakeVoice(speech_dir=self.dir)
        self.bot.files = {"v1": b"OggS-fake"}
        channel = self.channel(voice_replies=True)
        self.bot.updates.append([tg_message(1, 42, voice={"file_id": "v1"}), tg_message(2, 42, "and in text?")])
        await channel.poll_once()
        await channel.drain()
        self.assertEqual(len(self.bot.sent("sendVoice")), 1)
        self.assertEqual(self.svc.voice.spoken, ["ok"])
        self.assertFalse((self.dir / "speech.ogg").exists())
        self.assertEqual(len(self.bot.sent()), 2)  # both turns still get a text reply


# ------------------------------------------------------------------ iMessage

FAKE_IMSG = r'''
import json, pathlib, sys, time
base = pathlib.Path(__file__).parent
with open(base / "calls.jsonl", "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")
if sys.argv[1] == "watch":
    plan = json.loads((base / "watch.json").read_text())
    for row in plan.get("rows", []):
        print(json.dumps(row), flush=True)
    sys.stderr.write(plan.get("stderr", ""))
    if plan.get("hang"):
        time.sleep(60)
    sys.exit(plan.get("code", 0))
results = json.loads((base / "send.json").read_text())
result = results.pop(0) if results else {"stdout": "{\"status\":\"sent\"}"}
(base / "send.json").write_text(json.dumps(results))
print(result.get("stdout", ""))
sys.stderr.write(result.get("stderr", ""))
sys.exit(result.get("code", 0))
'''


def imsg_row(rowid, sender=OWNER, text="hi", **extra):
    return {"id": rowid, "chat_id": 1, "chat_identifier": sender, "chat_guid": f"iMessage;-;{sender}",
            "is_group": False, "sender": sender, "sender_name": "", "is_from_me": False, "text": text,
            "created_at": "2026-10-10T10:00:00Z", "attachments": [], **extra}


class IMessageTests(ChannelTestCase):
    def setUp(self):
        super().setUp()
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.imsg = write_script(self.bin / "imsg", FAKE_IMSG)
        self.plan(rows=[])
        self.send_results([])

    def plan(self, **plan):
        (self.bin / "watch.json").write_text(json.dumps(plan))

    def send_results(self, results):
        (self.bin / "send.json").write_text(json.dumps(results))

    def calls(self, command=None):
        path = self.bin / "calls.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [c for c in lines if command is None or c[0] == command]

    def channel(self, owners=(OWNER,), **settings):
        channel = IMessageChannel({"enabled": True, "owners": list(owners), "binary": self.imsg, **settings},
                                  self.svc, self.on_message)
        channel.retry_wait = 0
        channel.retry_delay = lambda attempt: 0.01
        self.channels.append(channel)
        return channel

    async def test_configured_requires_binary(self):
        self.assertTrue(self.channel().configured())
        self.assertFalse(self.channel(binary=str(self.dir / "missing")).configured())

    async def test_watch_filters_rows_and_resumes_from_cursor(self):
        self.plan(rows=[
            imsg_row(101, text="my own reply", is_from_me=True),
            imsg_row(102, text="Loved “hi”", is_reaction=True),
            imsg_row(103, sender=STRANGER, text="hi stranger"),
            imsg_row(104, text="group chatter", is_group=True, chat_id=9),
            imsg_row(105, text="what's on today?"),
        ])
        channel = self.channel()
        self.assertEqual(await channel.watch_once(), 5)
        await channel.drain()
        (message,) = self.received
        self.assertEqual((message.text, message.chat_id, message.sender), ("what's on today?", OWNER, OWNER))
        self.assertEqual(self.svc.store.get_kv("imessage:rowid"), 105)
        self.assertEqual(self.calls("watch")[0], ["watch", "--json", "--attachments"])
        self.assertEqual(self.calls("send"), [["send", f"--to={OWNER}", "--text=ok", "--json", "--service=imessage"]])
        self.assertEqual([e["sender"] for e in self.events("channel.ignored")], [STRANGER])
        self.plan(rows=[])
        await self.channel().watch_once()
        self.assertEqual(self.calls("watch")[-1], ["watch", "--json", "--since-rowid", "105", "--attachments"])

    async def test_audio_attachment_is_transcribed_in_place(self):
        self.svc.voice = FakeVoice("call mum")
        audio = self.dir / "Audio Message.caf"
        audio.write_bytes(b"caff")
        attachment = {"filename": audio.name, "mime_type": "audio/x-caf", "original_path": str(audio), "missing": False}
        self.plan(rows=[imsg_row(5, text="\ufffc", attachments=[attachment])])
        await self.channel().watch_once()
        await self.channels[-1].drain()
        self.assertEqual(self.received[0].text, "call mum")
        self.assertTrue(audio.exists())

    async def test_only_imessage_transport_counts_as_the_owner(self):
        self.plan(rows=[
            imsg_row(1, text="sms spoof", chat_guid=f"SMS;-;{OWNER}"),
            imsg_row(2, text="rcs spoof", chat_guid=f"RCS;-;{OWNER}"),
            imsg_row(3, text="ambiguous", chat_guid=f"any;-;{OWNER}"),
            imsg_row(4, text="service says sms", service="SMS"),
            imsg_row(5, text="verified by service", chat_guid=f"any;-;{OWNER}", service="iMessage"),
            imsg_row(6, text="verified by guid"),
        ])
        channel = self.channel()
        await channel.watch_once()
        await channel.drain()
        self.assertEqual([m.text for m in self.received], ["verified by service", "verified by guid"])
        rejected = [e for e in self.events("channel.ignored") if e.get("reason") == "not_imessage"]
        self.assertEqual([e["sender"] for e in rejected], [OWNER] * 4)
        self.assertTrue(all("spoof" not in json.dumps(e) for e in rejected))
        # A spoofed SMS can't pair either.
        stranger = self.channel(owners=())
        code = stranger.start_pairing()
        self.plan(rows=[imsg_row(7, sender=STRANGER, text=f"pair {code}", chat_guid=f"SMS;-;{STRANGER}")])
        await stranger.watch_once()
        self.assertEqual(stranger.owners(), [])
        # allow_sms opts in to the risk.
        self.received.clear()
        self.plan(rows=[imsg_row(8, text="sms accepted", chat_guid=f"SMS;-;{OWNER}")])
        relaxed = self.channel(allow_sms=True)
        await relaxed.watch_once()
        await relaxed.drain()
        self.assertEqual([m.text for m in self.received], ["sms accepted"])
        self.assertEqual(self.calls("send")[-1][-1], "--service=auto")

    async def test_long_approval_is_sent_in_full(self):
        summary = " ".join(f"word{i}" for i in range(1500))
        approval = await self.svc.policy.request(Action("channel.send", "signal:+1", summary), origin="agent")
        await self.channel().notify_approval(approval)
        texts = [c[2].removeprefix("--text=") for c in self.calls("send")]
        self.assertGreater(len(texts), 2)
        self.assertTrue(all(t.startswith(f"({i}/{len(texts)}) ") and len(t) <= 3000 for i, t in enumerate(texts, 1)))
        words = set(" ".join(t.split(" ", 1)[1] for t in texts).split())
        self.assertTrue(set(summary.split()) <= words)
        self.assertIn(f"approve {approval.id}", texts[-1])

    async def test_pairing_by_text(self):
        channel = self.channel(owners=())
        code = channel.start_pairing()
        self.plan(rows=[imsg_row(1, text=f"pair {code}")])
        await channel.watch_once()
        self.assertEqual(self.svc.config.get("channels.imessage.owners"), [OWNER])
        self.assertEqual(self.calls("send")[-1][2], f"--text={PAIRED}")

    async def test_send_passes_text_literally_as_one_argv_element(self):
        marker = self.dir / "pwned"
        nasty = f'He said "hi" & $(touch {marker}) `touch {marker}`; echo \'x\' > {marker}\n-v --help'
        channel = self.channel()
        await channel.send(OWNER, nasty)
        await channel.send(OWNER, "-v starts with a dash")
        self.assertEqual(self.calls("send")[0],
                         ["send", f"--to={OWNER}", f"--text={nasty}", "--json", "--service=imessage"])
        self.assertEqual(self.calls("send")[1][2], "--text=-v starts with a dash")
        self.assertFalse(marker.exists())
        await channel.send("chat:9", "to the group")  # groups keep the chat's own service
        self.assertEqual(self.calls("send")[2], ["send", "--chat-id=9", "--text=to the group", "--json"])

    async def test_only_not_started_failures_are_retried(self):
        channel = self.channel(service="imessage")
        self.send_results([{"code": 1, "stderr": "applescript send failed (not_started): Messages is not running."},
                           {"stdout": "{\"status\":\"sent\"}"}])
        await channel.send(OWNER, "hello")
        self.assertEqual(len(self.calls("send")), 2)
        self.assertIn("--service=imessage", self.calls("send")[0])
        self.send_results([{"code": 1, "stderr": "applescript send failed (may_have_completed): timed out"}])
        with self.assertRaises(IMessageSendError):
            await channel.send(OWNER, "again")
        self.assertEqual(len(self.calls("send")), 3)
        self.send_results([{"code": 1, "stderr": "Error: not authorized to send Apple events to Messages."}])
        with self.assertRaisesRegex(IMessageSendError, "Automation"):
            await channel.send(OWNER, "third")

    async def test_permission_error_is_surfaced(self):
        self.plan(rows=[], code=1, stderr="Error: unable to open database file\n")
        self.assertEqual(await self.channel().watch_once(), 0)
        (error,) = self.events("channel.error")
        self.assertIn("Full Disk Access", error["error"])

    async def test_stop_terminates_the_watch_process(self):
        self.plan(rows=[imsg_row(7, text="ping")], hang=True)
        channel = self.channel()
        await channel.start()
        await wait_for(lambda: self.received and channel._proc is not None)
        proc = channel._proc
        await asyncio.wait_for(channel.stop(), 10)
        self.assertIsNotNone(proc.returncode)


# ------------------------------------------------------------------ Signal

def sse(event_id, payload):
    return f"id:{event_id}\nevent:receive\ndata:{json.dumps(payload)}\n\n"


def envelope(number=OWNER, uuid="uuid-owner", ts=1000, text="hello", **data):
    return {"account": "+15550000000", "envelope": {
        "source": number or uuid, "sourceNumber": number, "sourceUuid": uuid, "sourceDevice": 1, "timestamp": ts,
        "dataMessage": {"timestamp": ts, "message": text, "attachments": [], **data}}}


class FakeSignal:
    def __init__(self):
        self.streams: list = []
        self.event_requests: list[httpx.Request] = []
        self.rpc: list[dict] = []
        self.rpc_error = None
        self.unreachable = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/events":
            self.event_requests.append(request)
            body = self.streams.pop(0) if self.streams else ""
            if isinstance(body, Exception):
                raise body
            if not body:
                await asyncio.sleep(0.05)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())
        payload = json.loads(request.content)
        self.rpc.append(payload)
        if self.rpc_error or payload["params"].get("recipient") == [self.unreachable]:
            error = self.rpc_error or {"code": -1, "message": "Unregistered user"}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "error": error})
        result = {"timestamp": 1, "results": [{"type": "SUCCESS"}]}
        if payload["method"] == "getAttachment":
            result = {"data": base64.b64encode(b"\x89PNG-image").decode()}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def sends(self):
        return [r["params"] for r in self.rpc if r["method"] == "send"]


class SignalTests(ChannelTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeSignal()

    def channel(self, owners=(OWNER,), **settings):
        channel = SignalChannel({"enabled": True, "url": "http://127.0.0.1:8080", "account": "+15550000000",
                                 "owners": list(owners), **settings}, self.svc, self.on_message)
        channel.transport = httpx.MockTransport(self.fake.handler)
        channel.retry_delay = lambda attempt: 0.01
        self.channels.append(channel)
        return channel

    async def test_configured_requires_url_and_account(self):
        self.assertTrue(self.channel().configured())
        self.assertFalse(SignalChannel({"enabled": True, "url": "http://127.0.0.1:8080"}, self.svc,
                                       self.on_message).configured())

    async def test_events_are_filtered_resumed_and_deduplicated(self):
        sync_only = {"account": "+15550000000", "envelope": {
            "sourceNumber": OWNER, "sourceUuid": "uuid-owner", "timestamp": 5,
            "syncMessage": {"sentMessage": {"message": "sent from my phone"}}}}
        owner = envelope(ts=2000, text="what's the weather?")
        self.fake.streams.append(":\n\n" + sse("1-1", envelope(STRANGER, "uuid-x", 1000, "hi"))
                                 + sse("1-2", sync_only)
                                 + sse("1-3", envelope("+15550000000", "uuid-self", 1500, "self"))
                                 + sse("1-4", {"jsonrpc": "2.0", "method": "receive", "params": owner})
                                 + sse("1-5", envelope(ts=2500, text="group", groupInfo={"groupId": "g1"})))
        channel = self.channel()
        self.assertEqual(await channel.listen_once(), 5)
        await channel.drain()
        (message,) = self.received
        self.assertEqual((message.text, message.chat_id, message.sender), ("what's the weather?", OWNER, OWNER))
        self.assertEqual(self.fake.sends(), [{"recipient": [OWNER], "message": "ok"}])
        self.assertEqual(self.svc.store.get_kv("signal:event_id"), "1-5")
        self.assertEqual([e["sender"] for e in self.events("channel.ignored")], [STRANGER])
        self.assertEqual(self.fake.event_requests[0].url.params["account"], "+15550000000")
        # The daemon restarted and replays its buffer under new ids: nothing runs twice.
        self.fake.streams.append(sse("2-1", owner))
        await channel.listen_once()
        await channel.drain()
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.fake.event_requests[1].headers["Last-Event-ID"], "1-5")

    async def test_pairing_and_uuid_only_owner(self):
        channel = self.channel(owners=())
        code = channel.start_pairing()
        self.fake.streams.append(sse("1-1", envelope(None, "uuid-private", 10, f"pair {code}")))
        await channel.listen_once()
        self.assertEqual(self.svc.config.get("channels.signal.owners"), ["uuid-private"])
        self.assertEqual(self.fake.sends()[-1], {"recipient": ["uuid-private"], "message": PAIRED})
        self.fake.streams.append(sse("1-2", envelope(None, "uuid-private", 11, "hi again")))
        await channel.listen_once()
        await channel.drain()
        self.assertEqual(self.received[0].sender, "uuid-private")

    async def test_long_reply_is_chunked_and_images_fetched(self):
        self.reply = "word " * 1000
        channel = self.channel()
        image = {"contentType": "image/png", "id": "att1", "size": 10}
        self.fake.streams.append(sse("1-1", envelope(text="look", attachments=[image])))
        await channel.listen_once()
        await channel.drain()
        self.assertEqual(self.received[0].images, [("image/png", base64.b64encode(b"\x89PNG-image").decode())])
        attachment_call = next(r for r in self.fake.rpc if r["method"] == "getAttachment")
        self.assertEqual(attachment_call["params"], {"id": "att1", "recipient": OWNER})
        chunks = [s["message"] for s in self.fake.sends()]
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c) <= 2000 for c in chunks))

    async def test_voice_note_attachment(self):
        channel = self.channel()
        note = {"contentType": "audio/aac", "id": "att2", "size": 10, "isVoiceNote": True}
        self.fake.streams.append(sse("1-1", envelope(text="", attachments=[note])))
        await channel.listen_once()
        await channel.drain()
        self.assertEqual(self.fake.sends()[-1]["message"], NO_VOICE)
        self.svc.voice = FakeVoice("remind me at five")
        self.fake.streams.append(sse("1-2", envelope(ts=3000, text="", attachments=[note])))
        await channel.listen_once()
        await channel.drain()
        self.assertEqual(self.received[0].text, "remind me at five")
        self.assertTrue(self.svc.voice.paths[0].endswith(".aac"))
        self.assertFalse(os.path.exists(self.svc.voice.paths[0]))

    async def test_long_approval_reaches_every_owner_in_full(self):
        self.fake.unreachable = "+15550002222"
        channel = self.channel(owners=("+15550002222", OWNER))
        summary = "Send to Bob:\n" + "\n".join(f"line {i} of the message" for i in range(300))
        approval = await self.svc.policy.request(Action("channel.send", "signal:+1555", summary), origin="agent")
        await channel.notify_approval(approval)
        parts = [s["message"] for s in self.fake.sends() if s["recipient"] == [OWNER]]
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(p.startswith(f"({i}/{len(parts)}) ") and len(p) <= 2000 for i, p in enumerate(parts, 1)))
        body = "\n".join(p.split(" ", 1)[1] for p in parts)
        self.assertTrue(all(f"line {i} of the message" in body for i in range(300)))
        self.assertIn(f"deny {approval.id}", parts[-1])
        self.assertEqual(self.events("channel.error")[-1]["stage"], "notify")

    async def test_rpc_errors_raise(self):
        self.fake.rpc_error = {"code": -1, "message": "Unregistered user"}
        with self.assertRaisesRegex(SignalError, "Unregistered user"):
            await self.channel().send(OWNER, "hello")

    async def test_stream_errors_back_off_and_recover(self):
        channel = self.channel()
        self.fake.streams = [httpx.ConnectError("refused"), httpx.ConnectError("refused"),
                             sse("1-1", envelope(text="back online"))]
        await channel.start()
        await wait_for(lambda: self.received)
        self.assertEqual(self.received[0].text, "back online")
        self.assertTrue(self.events("channel.error"))
        await channel.stop()


# ------------------------------------------------------------------ WhatsApp

FAKE_WACLI = r'''
import hashlib, hmac, json, pathlib, sys, time, urllib.request
base = pathlib.Path(__file__).parent
args = sys.argv[1:]
with open(base / "calls.jsonl", "a") as handle:
    handle.write(json.dumps(args) + "\n")
def option(name):
    return next(a.split("=", 1)[1] for a in args if a.startswith(f"--{name}="))
if "sync" in args:
    plan = json.loads((base / "plan.json").read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for payload in plan.get("payloads", []):
        body = json.dumps(payload).encode()
        signature = "sha256=" + hmac.new(option("webhook-secret").encode(), body, hashlib.sha256).hexdigest()
        request = urllib.request.Request(option("webhook"), data=body, headers={
            "Content-Type": "application/json", "X-Wacli-Signature": signature})
        opener.open(request, timeout=5).read()
    for event in plan.get("events", []):
        sys.stderr.write(json.dumps({"event": event, "ts": 1}) + "\n")
        sys.stderr.flush()
    if plan.get("follow"):
        time.sleep(60)
    sys.exit(0)
if "media" in args:
    pathlib.Path(option("output")).write_bytes(b"\xff\xd8jpeg-bytes")
    sys.exit(0)
print(json.dumps({"success": True, "data": {"sent": True, "id": "3EB0"}, "error": None}))
'''


def wa_payload(msg_id, sender="15550001111", text="hello", chat=None, **extra):
    return {"Chat": chat or f"{sender}@s.whatsapp.net", "ID": msg_id, "SenderJID": f"{sender}@s.whatsapp.net",
            "Timestamp": "2026-10-10T10:00:00Z", "FromMe": False, "Text": text, "Media": None, **extra}


class WhatsAppTests(ChannelTestCase):
    def setUp(self):
        super().setUp()
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.wacli = write_script(self.bin / "wacli", FAKE_WACLI)
        self.plan()

    def plan(self, **plan):
        (self.bin / "plan.json").write_text(json.dumps(plan))

    def calls(self, command):
        path = self.bin / "calls.jsonl"
        return [c for c in map(json.loads, path.read_text().splitlines()) if command in c] if path.exists() else []

    def channel(self, owners=(OWNER,), **settings):
        settings = {"enabled": True, "accept_tos_risk": True, "binary": self.wacli, "owners": list(owners), **settings}
        channel = WhatsAppChannel(settings, self.svc, self.on_message)
        channel.retry_delay = lambda attempt: 30
        channel.send_spacing = 0
        self.channels.append(channel)
        return channel

    async def test_requires_explicit_tos_risk_acceptance(self):
        self.assertTrue(self.channel().configured())
        for value in (None, False, "yes"):
            self.assertFalse(self.channel(accept_tos_risk=value).configured())
        self.assertFalse(self.channel(binary=str(self.dir / "missing")).configured())

    def test_handle_of(self):
        self.assertEqual(handle_of("15550001111:12@s.whatsapp.net"), "+15550001111")
        self.assertEqual(handle_of("12345@lid"), "12345@lid")
        self.assertEqual(handle_of("120363000@g.us"), "120363000@g.us")

    async def test_signed_webhook_events_reach_the_agent(self):
        owner = wa_payload("A1", text="hello from whatsapp")
        self.plan(follow=True, payloads=[
            wa_payload("A0", text="my own message", FromMe=True),
            wa_payload("B1", sender="15559999999", text="hi stranger"),
            wa_payload("G1", text="group chat", chat="120363000@g.us"),
            {"EventType": "receipt", "Chat": f"{OWNER[1:]}@s.whatsapp.net", "MessageIDs": ["X"], "Type": "read"},
            owner, owner,
        ])
        channel = self.channel()
        await channel.start()
        try:
            await wait_for(lambda: self.calls("send"))
            await channel.drain()
            (message,) = self.received
            self.assertEqual((message.text, message.sender, message.chat_id), ("hello from whatsapp", OWNER, OWNER))
            self.assertEqual([e["sender"] for e in self.events("channel.ignored")], ["+15559999999"])
            (sync,) = self.calls("sync")
            self.assertIn("--webhook-allow-private", sync)
            self.assertIn(f"--webhook={channel.webhook_url}", sync)
            self.assertTrue(channel.webhook_url.startswith("http://127.0.0.1:"))
            self.assertEqual(self.calls("send"), [["--json", "send", "text", f"--to={OWNER}", "--message=ok",
                                                   "--no-preview"]])
            # Forged requests without the per-run secret are rejected.
            async with httpx.AsyncClient(trust_env=False) as client:
                body = json.dumps(wa_payload("F1", text="forged")).encode()
                bad = "sha256=" + hmac.new(b"guess", body, hashlib.sha256).hexdigest()
                for headers in ({}, {"X-Wacli-Signature": bad}):
                    response = await client.post(channel.webhook_url, content=body, headers=headers)
                    self.assertEqual(response.status_code, 401)
                self.assertEqual((await client.get(channel.webhook_url)).status_code, 404)
            self.assertEqual(len(self.received), 1)
            proc = channel._proc
        finally:
            await asyncio.wait_for(channel.stop(), 10)
        self.assertIsNotNone(proc.returncode)

    async def test_send_text_literally(self):
        nasty = '-- "quoted" $(rm -rf ~) `id`\nsecond line'
        await self.channel(account="assistant").send(OWNER, nasty)
        (call,) = self.calls("send")
        self.assertEqual(call, ["--account=assistant", "--json", "send", "text", f"--to={OWNER}",
                                f"--message={nasty}", "--no-preview"])

    async def test_long_approval_is_sent_in_full_and_paced(self):
        channel = self.channel()
        channel.send_spacing = 0.2
        summary = " ".join(f"token{i}" for i in range(1500))
        approval = await self.svc.policy.request(Action("connector.write", "mail:send", summary), origin="agent")
        started = asyncio.get_running_loop().time()
        await channel.notify_approval(approval)
        texts = [next(a.removeprefix("--message=") for a in call if a.startswith("--message="))
                 for call in self.calls("send")]
        self.assertGreater(len(texts), 2)
        self.assertGreaterEqual(asyncio.get_running_loop().time() - started, 0.2 * (len(texts) - 1))
        self.assertTrue(all(t.startswith(f"({i}/{len(texts)}) ") for i, t in enumerate(texts, 1)))
        self.assertTrue(set(summary.split()) <= set(" ".join(texts).split()))

    async def test_image_message_downloads_media(self):
        channel = self.channel()
        media = {"Type": "image", "Caption": "what is this?", "MimeType": "image/jpeg"}
        await channel.handle_payload(wa_payload("M1", text="[Image]", Media=media))
        await channel.drain()
        (message,) = self.received
        self.assertEqual(message.text, "what is this?")
        self.assertEqual(message.images, [("image/jpeg", base64.b64encode(b"\xff\xd8jpeg-bytes").decode())])
        (download,) = self.calls("media")
        self.assertEqual(download[:4], ["--read-only", "media", "download", f"--chat={OWNER[1:]}@s.whatsapp.net"])
        await channel.stop()

    async def test_logged_out_stops_restarting(self):
        self.plan(events=["connected", "logged_out"])
        channel = self.channel()
        await channel.start()
        await wait_for(lambda: not channel._loops)
        self.assertEqual(len(self.calls("sync")), 1)
        self.assertIn("wacli auth", self.events("channel.error")[-1]["error"])
        await channel.stop()


if __name__ == "__main__":
    unittest.main()
