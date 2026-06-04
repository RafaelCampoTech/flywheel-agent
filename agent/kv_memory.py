"""Cross-task key-value memory via ctx.memory (the flywheel JSON store)."""


def recall(ctx) -> dict:
    """Load persisted memory (wiped between tasks on the off-arm)."""
    return ctx.memory.read() or {}


def remember(ctx, key: str, value) -> None:
    """Persist a value for later tasks in this stream."""
    ctx.memory.write(key, value)
