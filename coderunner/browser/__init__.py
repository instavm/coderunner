"""The owner's local browser, driven through a narrow, policy-checked API."""

from .broker import BrowserBroker, BrowserError

__all__ = ["BrowserBroker", "BrowserError"]
