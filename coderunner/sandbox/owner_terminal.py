"""Whether the owner is typing in the terminal on the assistant's computer.

The app's terminal reaches the VM through the gateway (/api/computer/terminal), which marks each keystroke here.
While the owner has typed in the last TYPING_S seconds, the assistant's next run_shell or run_python waits for them
to pause (at most MAX_WAIT_S), so its command does not land in the middle of what they are doing.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Awaitable, Callable

TYPING_S = 15.0
MAX_WAIT_S = 60.0
WAITING = "Waiting while you use the terminal"


class OwnerTerminal:
    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep):
        self.clock, self.sleep = clock, sleep
        self.last_input = -math.inf

    def typed(self) -> None:
        self.last_input = self.clock()

    def typing(self) -> bool:
        return self.clock() - self.last_input < TYPING_S

    async def wait_for_pause(self, on_wait: Callable[[], None] | None = None,
                             longest: float = MAX_WAIT_S) -> float:
        """Until the owner has not typed for TYPING_S seconds, or `longest` has passed; the seconds waited.
        on_wait is called once, when there is a wait."""
        if not self.typing():
            return 0.0
        if on_wait is not None:
            on_wait()
        start = self.clock()
        while self.typing():
            waited = self.clock() - start
            if waited >= longest:
                break
            left = TYPING_S - (self.clock() - self.last_input)
            await self.sleep(max(0.05, min(left, longest - waited, 1.0)))
        return self.clock() - start
