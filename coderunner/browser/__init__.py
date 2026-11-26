"""The owner's local browser, driven through a narrow, policy-checked API.

  broker      the actions the agent may take (open, read, click, type...) and the checks around each
  page        the page as compact text with numbered elements, and what changed between two readings
  urls        which URLs and hosts may be opened (web schemes, no local networks)
  scripts     the fixed scripts the broker runs in an isolated world
  cdp         the DevTools protocol over a pipe to a browser the broker launched (launcher)
  extension   the same protocol through the CodeRunner extension in the owner's everyday Chrome
              (chrome_host is the native messaging host it talks to)
  walls       whether a page is a bot wall (Cloudflare, DataDome, ...) standing in for the site
  switch      a site that walls off the sandbox's browser goes on in the Mac's, with the owner's OK
  app         the same protocol through the CodeRunner app's own browser on the Mac (app/browser.js)
  cookies     the owner's sign-ins for the sites they name, from their own browser into the app's
"""

from .broker import BrowserBroker
from .errors import BrowserError

__all__ = ["BrowserBroker", "BrowserError"]
