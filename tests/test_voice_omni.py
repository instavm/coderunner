"""LFM3-Omni voice sidecar: runtime command, catalog mmproj, phrasing filters, speech store."""

from __future__ import annotations

import asyncio
import json
import types
import unittest
from unittest import mock

from coderunner.core.config import Config
from coderunner.models import catalog, licenses
from coderunner.models.pool import ModelPool
from coderunner.models.runtimes import VoiceRuntime
from coderunner.voice.omni import OmniSlot, SpeechStore, VoiceBusy, VoiceTurns, clean_for_speech, looks_like_table
from coderunner.voice.narrator import TaskSpeaker
from coderunner.voice.tts import Speaker


class VoiceOmniCatalog(unittest.TestCase):
    def test_catalog_entry(self):
        entry = catalog.get("lfm3-omni-2b")
        self.assertIn("voice", entry["roles"])
        self.assertNotIn("chat", entry["roles"])
        art = entry["artifacts"]["gguf"]
        self.assertTrue(art.get("mmproj_required"))
        self.assertTrue(catalog.needs_mmproj(entry, "voice"))
        self.assertFalse(catalog.has_vision(entry, "voice"))

    def test_license_asks(self):
        entry = catalog.get("lfm3-omni-2b")
        cfg = Config({})
        verdict, reason, _ = licenses.check(entry, cfg)
        self.assertEqual(verdict, "ask")
        self.assertIn("LFM", reason)


class VoiceRuntimeCommand(unittest.TestCase):
    def test_command_flags(self):
        cfg = Config({"models": {"port_base": 8081}})
        rt = VoiceRuntime(cfg)
        self.assertEqual(rt.port, 8087)
        cmd = rt.command({"model": "/tmp/m.gguf", "mmproj": "/tmp/mm.gguf", "ctx": 8192, "alias": "lfm3-omni-2b"})
        self.assertIn("--mmproj", cmd)
        self.assertIn("--reasoning", cmd)
        self.assertIn("off", cmd)
        self.assertIn("--jinja", cmd)
        self.assertIn("-np", cmd)
        self.assertEqual(cmd[cmd.index("-np") + 1], "1")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8087")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")


class SpeechHelpers(unittest.TestCase):
    def test_clean_markdown(self):
        text = clean_for_speech("You paid **₹1,240** at [Sagar](https://example.com).")
        self.assertIn("₹1,240", text)
        self.assertIn("Sagar", text)
        self.assertNotIn("**", text)
        self.assertNotIn("http", text)

    def test_table_detect(self):
        table = "| Name | Price |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
        self.assertTrue(looks_like_table(table))
        self.assertFalse(looks_like_table("Just a sentence."))

    def test_speech_store_ephemeral(self):
        store = SpeechStore()
        sid = store.put(text="hi", pcm=b"\0\0", rate=24000, conversation="web:1")
        self.assertEqual(store.get(sid)["text"], "hi")
        self.assertEqual(store.pop(sid)["pcm"], b"\0\0")
        self.assertIsNone(store.get(sid))


class NarratorFilter(unittest.TestCase):
    def test_skips_when_speak_off(self):
        daemon = mock.Mock()
        daemon.config = Config({"voice": {"speak_tasks": False, "omni_model": "lfm3-omni-2b"}})
        speaker = TaskSpeaker(daemon)
        speaker._queue = mock.Mock()
        speaker.on_event({"type": "started", "prompt": "Book a table"})
        speaker._queue.put_nowait.assert_not_called()

    def test_queues_started_when_on(self):
        daemon = mock.Mock()
        daemon.config = Config({"voice": {"speak_tasks": True, "omni_model": "lfm3-omni-2b"}})
        speaker = TaskSpeaker(daemon)
        with mock.patch.object(speaker.runtime, "ready", return_value=True):
            speaker._queue = mock.Mock()
            speaker.on_event({"type": "started", "prompt": "Book a table", "conversation": "web:1"})
            speaker._queue.put_nowait.assert_called_once()

    def test_ignores_activity(self):
        daemon = mock.Mock()
        daemon.config = Config({"voice": {"speak_tasks": True, "omni_model": "lfm3-omni-2b"}})
        speaker = TaskSpeaker(daemon)
        with mock.patch.object(speaker.runtime, "ready", return_value=True):
            speaker._queue = mock.Mock()
            speaker.on_event({"type": "activity", "event": "browser.open"})
            speaker._queue.put_nowait.assert_not_called()


class ModelPoolOmni(unittest.TestCase):
    def test_exclusive_blocks_mid_task(self):
        cfg = Config({"voice": {"omni_model": "lfm3-omni-2b", "omni_exclusive": True},
                      "models": {"chat": "qwen3.5-9b", "remote": {}}})
        pool = ModelPool(cfg)
        self.assertTrue(pool.role_plan().omni_exclusive)
        self.assertFalse(pool.can_speak_during_task())

    def test_remote_allows_mid_task(self):
        cfg = Config({"voice": {"omni_model": "lfm3-omni-2b", "omni_exclusive": True},
                      "models": {"chat": "qwen3.5-9b", "remote": {"provider": "openai", "model": "x"}}})
        pool = ModelPool(cfg)
        self.assertTrue(pool.can_speak_during_task())


class FakeRuntime:
    def __init__(self, log, name):
        self.log, self.name, self.up = log, name, True

    def running(self):
        return self.up

    def ensure(self):
        self.log.append(f"{self.name}.ensure")

    def start(self):
        self.log.append(f"{self.name}.start")
        self.up = True

    def stop(self):
        self.log.append(f"{self.name}.stop")
        self.up = False


def fake_daemon(config, running=()):
    tasks = types.SimpleNamespace(running=dict.fromkeys(running), waiting=[])
    return types.SimpleNamespace(config=config, tasks=tasks, svc=types.SimpleNamespace(llm=None, turns=None))


TIGHT = {"voice": {"omni_model": "lfm3-omni-2b", "omni_exclusive": True},
         "models": {"chat": "qwen3.5-9b", "runtime": "llama.cpp", "remote": {}}}


class TakingTurns(unittest.IsolatedAsyncioTestCase):
    def turns(self, config, running=()):
        log = []
        voice, planner = FakeRuntime(log, "voice"), FakeRuntime(log, "planner")
        turns = VoiceTurns(fake_daemon(Config(config), running), voice)
        patch = mock.patch("coderunner.models.runtimes.choose_runtime", return_value=planner)
        patch.start()
        self.addCleanup(patch.stop)
        return turns, log

    async def test_side_by_side_only_starts_voice(self):
        turns, log = self.turns({"voice": {"omni_model": "lfm3-omni-2b", "omni_exclusive": False},
                                 "models": {"chat": "qwen3.5-9b"}})
        async with turns.use():
            pass
        self.assertEqual(log, ["voice.ensure"])

    async def test_tight_memory_unloads_the_assistant_first_and_loads_it_after(self):
        turns, log = self.turns(TIGHT)
        async with turns.use():
            self.assertEqual(log, ["planner.stop", "voice.ensure"])
        self.assertEqual(log, ["planner.stop", "voice.ensure", "voice.stop", "planner.start"])

    async def test_tight_memory_waits_while_the_assistant_works(self):
        turns, log = self.turns(TIGHT, running=["t1"])
        with self.assertRaises(VoiceBusy):
            async with turns.use():
                pass
        self.assertEqual(log, [])

    async def test_remote_chat_leaves_room(self):
        config = json.loads(json.dumps(TIGHT))
        config["models"]["remote"] = {"provider": "anthropic"}
        turns, log = self.turns(config)
        async with turns.use():
            pass
        self.assertEqual(log, ["voice.ensure"])

    async def test_a_message_ends_voice_and_gets_the_model_back(self):
        turns, log = self.turns(TIGHT)
        held, done = asyncio.Event(), asyncio.Event()

        async def talk():
            async with turns.use():
                held.set()
                await turns.yielding.wait()     # as the Talk proxy does
            done.set()

        session = asyncio.create_task(talk())
        await held.wait()
        await asyncio.wait_for(turns.make_way(), 2)
        self.assertTrue(done.is_set())
        self.assertEqual(log[-2:], ["voice.stop", "planner.start"])
        self.assertFalse(turns.yielding.is_set())
        await session

    async def test_make_way_is_free_when_voice_is_idle(self):
        turns, log = self.turns(TIGHT)
        await asyncio.wait_for(turns.make_way(), 1)
        self.assertEqual(log, [])


class SlotAndSpeaker(unittest.IsolatedAsyncioTestCase):
    async def test_busy_slot_says_so_in_plain_words(self):
        slot = OmniSlot()
        await slot.acquire("talk")
        with self.assertRaises(VoiceBusy) as caught:
            await slot.acquire("speaking:done")
        self.assertNotIn("speaking", str(caught.exception))
        self.assertTrue(slot.busy)          # the holder still has it

    def speaker(self, config=None):
        daemon = mock.Mock()
        daemon.config = Config(config or {"voice": {"speak_tasks": True, "omni_model": "lfm3-omni-2b"},
                                          "models": {"chat": "qwen3.5-9b"}})
        daemon.tasks = types.SimpleNamespace(running={}, waiting=[])
        speaker = TaskSpeaker(daemon)
        speaker.turns.runtime = speaker.runtime = mock.Mock(ensure=mock.Mock(), ready=mock.Mock(return_value=True))
        return speaker, daemon

    async def test_listen_is_not_announced_twice(self):
        speaker, daemon = self.speaker()
        with mock.patch("coderunner.voice.narrator.synthesize", mock.AsyncMock(return_value=(b"\0\0", 24000))):
            meta = await speaker.speak_text("**Done.** You paid 40.", conversation="web:1", kind="listen")
        self.assertEqual(meta["text"], "Done. You paid 40.")
        daemon.bus.notify_live.assert_not_called()
        self.assertFalse(speaker.slot.busy)

    async def test_final_reply_is_announced(self):
        speaker, daemon = self.speaker()
        with mock.patch("coderunner.voice.narrator.synthesize", mock.AsyncMock(return_value=(b"\0\0", 24000))):
            await speaker._handle({"type": "done", "conversation": "web:1", "reply": "All set."})
        daemon.bus.notify_live.assert_called_once()
        self.assertEqual(daemon.bus.notify_live.call_args[0][0]["utterance"]["kind"], "done")

    async def test_line_dropped_while_talk_holds_the_model(self):
        speaker, daemon = self.speaker()
        await speaker.slot.acquire("talk")
        synth = mock.AsyncMock(return_value=(b"\0\0", 24000))
        with mock.patch("coderunner.voice.narrator.synthesize", synth), \
                mock.patch("coderunner.voice.narrator.asyncio.sleep", mock.AsyncMock()):
            self.assertIsNone(await speaker._say("web:1", "done", text="Hi"))
        synth.assert_not_called()
        self.assertTrue(speaker.slot.busy)  # Talk keeps it

    def test_activity_events_are_ignored_cheaply(self):
        speaker, _ = self.speaker()
        speaker._queue = mock.Mock()
        speaker.on_event({"type": "activity", "event": "browser.open"})
        speaker.runtime.ready.assert_not_called()
        speaker._queue.put_nowait.assert_not_called()

    def test_channel_replies_do_not_use_lfm_when_it_takes_turns(self):
        cfg = Config(TIGHT)
        speaker = Speaker(lambda key, default=None: cfg.get(f"voice.{key}", default), config=cfg)
        with mock.patch("coderunner.models.runtimes.VoiceRuntime.ready", return_value=True):
            self.assertFalse(speaker._lfm_ready())
            cfg.set("voice.omni_exclusive", False)
            self.assertTrue(speaker._lfm_ready())


class VoiceFit(unittest.TestCase):
    def test_audio_projector_counts_toward_memory(self):
        from coderunner.models import fit, hardware
        entry = catalog.get("lfm3-omni-2b")
        hw = hardware.Hardware("darwin", "arm64", budget_gb=12, bandwidth_gbs=150)
        with_projector = fit.estimate(entry, hw, ctx=8192, runtime="voice")["need_gb"]
        art = dict(entry["artifacts"]["gguf"], mmproj_required=False)
        without = fit.estimate(dict(entry, artifacts={"gguf": art}), hw, ctx=8192, runtime="voice")["need_gb"]
        self.assertGreater(with_projector, without)


class VoiceCommandArgs(unittest.TestCase):
    def test_reasoning_off_once(self):
        rt = VoiceRuntime(Config({}))
        cmd = rt.command({"model": "m", "mmproj": "p", "alias": "a", "args": ["--reasoning", "off"]})
        self.assertEqual(cmd.count("--reasoning"), 1)


if __name__ == "__main__":
    unittest.main()
