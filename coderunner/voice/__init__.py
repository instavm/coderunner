"""Local speech: what was said in voice messages, recordings and media files, and spoken replies.

Audio never leaves this computer. Modules:
  service    Voice: picks the engine, converts audio, sets engines up, speaks replies
  parakeet   Parakeet Ultra on Core ML (Neural Engine, Apple silicon, macOS 15+), through parakeet_speech.swift
  apple      Apple's on-device speech model (macOS 26+), through apple_speech.swift
  qwen       Qwen3-ASR on llama.cpp's official build, everywhere else and for languages the others lack
  installed  whisper-cli, parakeet-mlx and mlx_whisper when the owner installed them
  tts        say, piper and kokoro
  common     errors, timed transcript segments, audio conversion

Voice.segments(path) gives timed segments (for search); Voice.transcribe(path) the text. Config keys
(all optional) under `voice`: stt (auto|parakeet-coreml|apple|qwen3-asr|whisper-cli|parakeet-mlx|mlx_whisper),
language, asr_model (qwen3-asr-1.7b|qwen3-asr-0.6b), whisper_model, ffmpeg, whisper_cli, parakeet_mlx,
parakeet_model, mlx_whisper, mlx_whisper_model, tts (say|piper|kokoro), say_voice, piper,
piper_model, kokoro, kokoro_voice, timeout. Binary keys take a path when the tool is not on PATH.
"""

from ..core.process import find_binary, run, terminate
from .common import STT_HINT, TTS_HINT, Segment, UnsupportedLanguage, VoiceUnavailable
from .service import Voice

__all__ = ["STT_HINT", "TTS_HINT", "Segment", "UnsupportedLanguage", "Voice", "VoiceUnavailable",
           "find_binary", "run", "terminate"]
