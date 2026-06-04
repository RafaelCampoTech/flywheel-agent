"""Colorful structured logger for agent pipeline steps. All output to stderr."""

from __future__ import annotations

import sys
import textwrap
import time

_start_time: float = time.monotonic()

# ANSI color codes
_R = "\033[0m"       # reset
_BOLD = "\033[1m"
_DIM = "\033[2m"

_CYAN = "\033[36m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_WHITE = "\033[37m"
_B_CYAN = "\033[96m"
_B_GREEN = "\033[92m"
_B_YELLOW = "\033[93m"
_B_RED = "\033[91m"
_B_MAGENTA = "\033[95m"
_B_BLUE = "\033[94m"


def _ts() -> str:
    elapsed = time.monotonic() - _start_time
    return f"{_DIM}[{elapsed:6.1f}s]{_R}"


def _print(*parts: str) -> None:
    print(*parts, file=sys.stderr, flush=True)


def _divider(char: str = "─", width: int = 70) -> str:
    return f"{_DIM}{char * width}{_R}"


# ── Public API ────────────────────────────────────────────────────────────────

def task_start(instruction: str) -> None:
    _print()
    _print(_divider("═"))
    _print(f"{_ts()} {_BOLD}{_B_CYAN}🚀 TASK{_R}  {instruction[:200]}")
    _print(_divider("═"))


def step(n: int | str, label: str) -> None:
    _print(f"\n{_ts()} {_BOLD}{_CYAN}── STEP {n}: {label}{_R}")


def rag_result(
    apps: list[str],
    n_docs: int,
    sample_apis: list[str],
    method: str = "word_freq",
    top_scored: list[tuple[str, float]] | None = None,
) -> None:
    apps_str = ", ".join(apps) if apps else "none"
    if method in ("cosine", "chroma", "mcp"):
        method_tag = f"{_B_CYAN}{method}{_R}"
    else:
        method_tag = f"{_DIM}{method}{_R}"
    _print(f"{_ts()} {_B_YELLOW}📚 RAG{_R}  [{method_tag}]  apps={_BOLD}{apps_str}{_R}  docs={n_docs}")
    if top_scored:
        for api_id, score in top_scored[:8]:
            bar = "█" * int(score * 20)
            _print(f"         {_DIM}{score:.3f} {_YELLOW}{bar:<20}{_R} {api_id}")
    elif sample_apis:
        sample = ", ".join(sample_apis[:8])
        if len(sample_apis) > 8:
            sample += f"  (+{len(sample_apis) - 8} more)"
        _print(f"         {_DIM}apis: {sample}{_R}")


def memory_result(n_similar: int) -> None:
    if n_similar:
        _print(f"{_ts()} {_B_MAGENTA}🧠 MEMORY{_R}  {n_similar} similar solution(s) found")
    else:
        _print(f"{_ts()} {_MAGENTA}🧠 MEMORY{_R}  no similar solutions (scratch mode)")


def plan_result(task_plan: dict) -> None:
    obj = task_plan.get("objective", "")[:120]
    ttype = task_plan.get("task_type", "?")
    otype = task_plan.get("expected_output_type", "?")
    apps = ", ".join(task_plan.get("apps_involved", []))
    steps = task_plan.get("steps", [])
    _print(f"{_ts()} {_B_BLUE}📋 PLAN{_R}  {_BOLD}{ttype}{_R} → {otype}")
    _print(f"         {_DIM}objective: {obj}{_R}")
    _print(f"         {_DIM}apps: {apps or 'none'}{_R}")
    for s in steps[:6]:
        _print(f"         {_DIM}  {s.get('step_id','?')}. [{s.get('app','?')}] {s.get('description','')[:80]}{_R}")
    if len(steps) > 6:
        _print(f"         {_DIM}  ... ({len(steps) - 6} more steps){_R}")


def tokens_result(tokens: dict) -> None:
    apps = list(tokens.keys())
    if apps:
        _print(f"{_ts()} {_B_GREEN}🔑 TOKENS{_R}  bootstrapped: {', '.join(apps)}")
    else:
        _print(f"{_ts()} {_YELLOW}🔑 TOKENS{_R}  none bootstrapped")


def codegen_prompt(mode: str, system: str, user: str) -> None:
    """Log the prompts sent to the model for code generation."""
    _print(f"{_ts()} {_DIM}💬 CODEGEN PROMPT  mode={mode}{_R}")
    _print(f"         {_DIM}┌─ system ({len(system)} chars) ──────────────────────────────{_R}")
    for ln in system.strip().splitlines()[:6]:
        _print(f"         {_DIM}│ {ln[:100]}{_R}")
    if system.count("\n") > 6:
        _print(f"         {_DIM}│ ... ({system.count(chr(10)) - 6} more lines){_R}")
    _print(f"         {_DIM}├─ user ({len(user)} chars) ────────────────────────────────{_R}")
    for ln in user.strip().splitlines()[:10]:
        _print(f"         {_DIM}│ {ln[:100]}{_R}")
    if user.count("\n") > 10:
        _print(f"         {_DIM}│ ... ({user.count(chr(10)) - 10} more lines){_R}")
    _print(f"         {_DIM}└────────────────────────────────────────────────────{_R}")


def codegen_result(mode: str, code: str) -> None:
    lines = code.splitlines()
    _print(f"{_ts()} {_B_BLUE}💻 CODEGEN OUTPUT{_R}  mode={_BOLD}{mode}{_R}  {len(lines)} lines  {len(code)} chars")
    _print(f"         {_DIM}{_divider()}{_R}")
    for ln in lines[:25]:
        _print(f"         {_BLUE}{ln}{_R}")
    if len(lines) > 25:
        _print(f"         {_DIM}... ({len(lines) - 25} more lines){_R}")
    _print(f"         {_DIM}{_divider()}{_R}")


def exec_result(stdout: str) -> None:
    lines = stdout.strip().splitlines() if stdout else []
    _print(f"{_ts()} {_WHITE}▶ EXEC{_R}  {len(lines)} line(s) of output")
    if lines:
        _print(f"         {_DIM}{_divider()}{_R}")
        for ln in lines[:30]:
            _print(f"         {_WHITE}{ln}{_R}")
        if len(lines) > 30:
            _print(f"         {_DIM}... ({len(lines) - 30} more){_R}")
        _print(f"         {_DIM}{_divider()}{_R}")


def eval_pass(answer: str | None, expected_type: str) -> None:
    _print(f"{_ts()} {_B_GREEN}✅ EVAL PASS{_R}  type={expected_type}  answer={_BOLD}{answer!r}{_R}")


def eval_fail(reason: str, actual: str, expected_type: str) -> None:
    _print(f"{_ts()} {_B_RED}❌ EVAL FAIL{_R}  type={expected_type}  got={actual!r}")
    _print(f"         {_RED}reason: {reason}{_R}")


def repair_attempt(n: int, error: str) -> None:
    _print(f"\n{_ts()} {_B_MAGENTA}🔧 REPAIR {n}/3{_R}  {_RED}{error[:150]}{_R}")


def repair_success(n: int) -> None:
    _print(f"{_ts()} {_B_GREEN}🔧 REPAIR {n} → SUCCESS{_R}")


def repair_exhausted() -> None:
    _print(f"{_ts()} {_B_RED}🔧 REPAIR EXHAUSTED{_R}  all 3 attempts failed")


def replan() -> None:
    _print(f"\n{_ts()} {_B_YELLOW}♻️  REPLAN{_R}  generating new plan after repair exhaustion")


def finish(answer: str | None, elapsed: float) -> None:
    _print()
    _print(_divider("═"))
    label = _B_GREEN if answer else _B_YELLOW
    _print(f"{_ts()} {label}{'✅' if answer else '⚠️ '} COMPLETE{_R}  answer={_BOLD}{answer!r}{_R}  elapsed={elapsed:.1f}s")
    _print(_divider("═"))
    _print()


def emergency(reason: str) -> None:
    _print(f"\n{_ts()} {_B_RED}⚠️  EMERGENCY EXIT{_R}  {reason}")


def info(label: str, msg: str) -> None:
    _print(f"{_ts()} {_DIM}  {label}:{_R} {msg[:200]}")


def oracle_result(passed: bool) -> None:
    label = _B_GREEN if passed else _B_RED
    icon = "✅" if passed else "❌"
    verdict = "PASS — code promoted to memory" if passed else "FAIL — code discarded"
    _print(f"{_ts()} {label}{icon} ORACLE{_R}  {verdict}")


def error(label: str, msg: str) -> None:
    _print(f"{_ts()} {_RED}✗ {label}{_R}  {msg[:200]}")
