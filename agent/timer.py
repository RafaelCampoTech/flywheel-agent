"""Task timer with per-step budgets and emergency exit thresholds."""

import time

BUDGETS = {
    "plan":     30,
    "rag":      15,
    "codegen":  45,
    "execute":  60,
    "evaluate": 15,
    "repair":   45,
    "replan":   30,
}
EMERGENCY_EXIT_AT = 240
HARD_KILL_AT = 270


class Timer:
    def __init__(self):
        self._start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def should_attempt_repair(self, attempt: int) -> bool:
        return self.elapsed() + BUDGETS["repair"] < HARD_KILL_AT and attempt < 3

    def should_replan(self) -> bool:
        needed = BUDGETS["replan"] + BUDGETS["codegen"] + BUDGETS["execute"]
        return self.elapsed() + needed < EMERGENCY_EXIT_AT

    def at_emergency(self) -> bool:
        return self.elapsed() >= EMERGENCY_EXIT_AT

    def at_hard_kill(self) -> bool:
        return self.elapsed() >= HARD_KILL_AT
