"""The one error browser actions raise; its message is written for the agent."""


HEADLESS_HANDOFF = "The browser is headless, so the owner cannot take over; set browser.headless to false"


class BrowserError(Exception):
    """An action could not be done; the message is written for the agent."""
