"""LFM3-Omni voice sidecar: runtime command, catalog mmproj, phrasing filters, speech store."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from coderunner.core.config import Config
from coderunner.models import catalog, licenses
from coderunner.models.pool import ModelPool
from coderunner.models.runtimes import VoiceRuntime
from coderunner.voice.omni import SpeechStore, clean_for_speech, looks_like_table
from coderunner.voice.narrator import TaskSpeaker


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


if __name__ == "__main__":
    unittest.main()
