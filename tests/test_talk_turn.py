"""Talk: whose words a turn's message is (a transcript, never the voice model's answer), and how the turn stands in
the thread (the owner's words, then what was said back), kept so a reopened thread shows it the same way."""

import asyncio
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import httpx

from coderunner.agent import build_registry
from coderunner.core.config import Config
from coderunner.server import tasks as tasks_module
from coderunner.server.gateway import admin_app
from coderunner.voice.narrator import TaskSpeaker
from coderunner.voice.omni import pick_heard, reply_shaped
from fakes import Svc

ADMIN = {"Authorization": "Bearer admin-token-0123456789abcdefghijklmnopq"}
SAID = "check the quick commerce websites too"
ANSWERED = "Sure, I can check the quick commerce websites for you. Which ones are you looking at?"


class PickHeard(unittest.TestCase):
    def test_an_answer_is_not_the_owners_words(self):
        self.assertTrue(reply_shaped(ANSWERED))
        self.assertEqual(pick_heard(ANSWERED, SAID), SAID)
        # A transcript with an answer after it is not a transcript either.
        self.assertEqual(pick_heard("Check the other shop too. Sure, I'll check it.", "check the other shop too"), "check the other shop too")
        self.assertEqual(pick_heard("Hello! How can I help you today?", "hey"), "hey")

    def test_omni_keeps_the_names_dictation_mishears(self):
        self.assertEqual(pick_heard("Book a train ticket to Patna on IRCTC.", "Book a train ticket to Putna on I R C T C."),
                         "Book a train ticket to Patna on IRCTC.")
        self.assertEqual(pick_heard("Find cafes in Indiranagar.", "Find cafes in Toydon and Diranagar."),
                         "Find cafes in Indiranagar.")
        self.assertEqual(pick_heard("It's Bangalore, not Bangladesh.", "It's Bangladesh, not Bangladesh."),
                         "It's Bangalore, not Bangladesh.")

    def test_owner_words_that_only_look_like_a_reply_cost_nothing(self):
        for owner in ("Sorry, I meant Colaba.", "Okay, check the other shop too.", "Can you tell me the weather?",
                      "Let me know when it's done."):
            self.assertFalse(reply_shaped(owner), owner)
            self.assertEqual(pick_heard(owner, owner.lower().rstrip(".?")), owner)

    def test_with_one_side_missing(self):
        self.assertEqual(pick_heard("", SAID), SAID)
        self.assertEqual(pick_heard("Find cafes in Indiranagar.", ""), "Find cafes in Indiranagar.")
        self.assertEqual(pick_heard(ANSWERED, ""), "")          # nothing, rather than an answer as their words
        self.assertEqual(pick_heard("", ""), "")


class TalkTurn(unittest.IsolatedAsyncioTestCase):
    def speaker(self):
        daemon = mock.Mock()
        daemon.config = Config({"voice": {"omni_model": "lfm3-omni-2b"}, "models": {"chat": "qwen3.5-9b"}})
        daemon.tasks = types.SimpleNamespace(running={}, waiting=[])
        speaker = TaskSpeaker(daemon)
        speaker.turns.runtime = speaker.runtime = mock.Mock(ensure=mock.Mock(), ready=mock.Mock(return_value=True))
        return speaker

    @staticmethod
    def ready(text):
        future = asyncio.get_running_loop().create_future()
        future.set_result(text)
        return future

    async def test_the_reply_is_made_to_the_owners_words_not_to_omnis_answer(self):
        speaker = self.speaker()
        reply = mock.AsyncMock(return_value=("Sure, checking the other two shops too.", True))
        with mock.patch("coderunner.voice.narrator.hear", mock.AsyncMock(return_value=ANSWERED)), \
                mock.patch("coderunner.voice.narrator.talk_reply", reply):
            heard, say, do = await speaker.talk_turn(b"RIFF", "app:a", self.ready(SAID))
        self.assertEqual((heard, say, do), (SAID, "Sure, checking the other two shops too.", True))
        self.assertEqual(reply.call_args[0][1], SAID)
        self.assertFalse(speaker.slot.busy)

    async def test_dictation_hears_it_when_the_voice_model_fails(self):
        speaker = self.speaker()
        with mock.patch("coderunner.voice.narrator.hear", mock.AsyncMock(side_effect=httpx.ConnectError("down"))), \
                mock.patch("coderunner.voice.narrator.talk_reply", mock.AsyncMock(return_value=("On it.", True))):
            self.assertEqual(await speaker.talk_turn(b"RIFF", "app:a", self.ready(SAID)), (SAID, "On it.", True))

    async def test_an_answer_with_no_dictation_is_nothing_heard(self):
        speaker = self.speaker()
        reply = mock.AsyncMock()
        with mock.patch("coderunner.voice.narrator.hear", mock.AsyncMock(return_value=ANSWERED)), \
                mock.patch("coderunner.voice.narrator.talk_reply", reply):
            self.assertEqual(await speaker.talk_turn(b"RIFF", "app:a"), ("", "", False))
        reply.assert_not_called()


class FakeVoice:
    def __init__(self, text):
        self.text, self.light = text, []

    def ffmpeg(self):
        return "ffmpeg"

    async def transcribe(self, path, light=False):
        self.light.append(light)
        assert pathlib.Path(path).exists()
        return self.text


class TalkInTheThread(unittest.IsolatedAsyncioTestCase):
    """POST /api/voice/turn, with the voice model's hearing and replies stood in for."""

    def setUp(self):
        home = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"CODERUNNER_HOME": home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.svc = Svc({"voice": {"omni_model": "lfm3-omni-2b"}, "models": {"chat": "qwen3.5-9b"}})
        self.svc.registry = build_registry(self.svc)
        self.svc.voice = self.voice = FakeVoice(SAID)
        bus = tasks_module.EventBus(self.svc.store)
        daemon = types.SimpleNamespace(svc=self.svc, config=self.svc.config, bus=bus)
        daemon.tasks = self.tasks = tasks_module.Tasks(daemon, bus)
        self.tasks._launch = lambda job: None           # the work itself is not under test: it stays running
        speaker = TaskSpeaker(daemon)
        speaker.turns.runtime = speaker.runtime = mock.Mock(ensure=mock.Mock(), ready=mock.Mock(return_value=True))
        daemon.speaker = self.speaker = speaker
        self.bus, self.app = bus, admin_app(self.svc, daemon)

        async def to_wav(converter, source, target):
            pathlib.Path(target).write_bytes(pathlib.Path(source).read_bytes())
        for patch in (mock.patch("coderunner.voice.common.to_wav", to_wav),
                      mock.patch("coderunner.voice.narrator.synthesize",
                                 mock.AsyncMock(return_value=(b"\0\0", 24000)))):
            patch.start()
            self.addCleanup(patch.stop)

    async def turn(self, client, heard, say, do, conversation="app:t"):
        with mock.patch("coderunner.voice.narrator.hear", mock.AsyncMock(return_value=heard)), \
                mock.patch("coderunner.voice.narrator.talk_reply", mock.AsyncMock(return_value=(say, do))):
            response = await client.post(f"/api/voice/turn?conversation={conversation}", content=b"RIFF audio",
                                         headers=ADMIN)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def thread(self, conversation="app:t"):
        """The thread as a reopened app reads it (GET /api/conversations/<id>): its turns, in order."""
        shown = []
        for e in self.bus.since(0, conversation=conversation):
            if e["type"] == "started":
                shown.append(("owner", e["prompt"]))
            elif e["type"] == "steer":
                shown.append(("owner", e["text"]))
            elif e["type"] == "talk":
                if e.get("heard"):
                    shown.append(("owner", e["heard"]))
                if e.get("say"):
                    shown.append(("assistant", e["say"]))
        return shown

    async def test_a_talk_turn_is_the_owners_words_then_what_was_said_back(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1") as c:
            first = await self.turn(c, ANSWERED, "Sure, checking the quick commerce sites too.", True)
            self.assertEqual(first["heard"], SAID)
            self.assertEqual(self.tasks.get(first["task"]["id"])["prompt"], SAID)     # the planner's message
            self.assertEqual(self.voice.light, [True])         # beside the voice model: off the GPU only

            # Said while it works: the message to the task, then the reply under it.
            self.voice.text = "make it the other two shops"
            second = await self.turn(c, "Make it the other two shops.", "Got it, the other two shops.", True)
            self.assertTrue(second["task"]["steer"])

            # Words that send nothing (how is it going) are kept too, with their answer.
            self.voice.text = "how is it going"
            third = await self.turn(c, "How is it going?", "Still comparing prices on the second shop.", False)
            self.assertIsNone(third["task"])

            loaded = (await c.get("/api/conversations/app:t", headers=ADMIN)).json()
        self.assertEqual(self.thread(), [
            ("owner", SAID), ("assistant", "Sure, checking the quick commerce sites too."),
            ("owner", "Make it the other two shops."), ("assistant", "Got it, the other two shops."),
            ("owner", "How is it going?"), ("assistant", "Still comparing prices on the second shop."),
        ])
        talk = [e for e in loaded["events"] if e["type"] == "talk"]
        self.assertEqual([e["task"] for e in talk], [first["task"]["id"]] * 3)
        self.assertEqual(talk[1]["steer"], second["task"]["steer"])        # drawn under that message
        self.assertNotIn("heard", talk[0])                                  # the request already shows it
        self.assertNotIn(ANSWERED, str(loaded))

    async def test_the_line_is_kept_even_when_it_cannot_be_spoken(self):
        with mock.patch("coderunner.voice.narrator.synthesize", mock.AsyncMock(side_effect=RuntimeError("tts"))):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1") as c:
                result = await self.turn(c, SAID, "On it.", True)
        self.assertIsNone(result["say"])
        self.assertEqual(self.thread(), [("owner", SAID), ("assistant", "On it.")])

    async def test_a_new_thread_with_nothing_to_do_keeps_nothing(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1") as c:
            self.voice.text = "hello"
            result = await self.turn(c, "Hello.", "Hi! What can I do for you?", False, conversation="app:new")
        self.assertIsNone(result["task"])
        self.assertEqual(self.thread("app:new"), [])


if __name__ == "__main__":
    unittest.main()
