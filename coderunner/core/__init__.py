"""Foundations every part of CodeRunner uses.

  config    the owner's settings (assistant/config.json)
  paths     where state lives (CODERUNNER_HOME) and the credentials
  store     the SQLite database
  audit     the append-only audit log
  llm       the OpenAI-compatible client for the local model runtime
  process   running local tools safely (no shell, killed on timeout)
  native    small Swift helpers built once on first use (macOS)
"""
