"""Time-budgeted repair loop: up to 3 attempts using generate_repair."""

from __future__ import annotations

import agent.logger as log
from agent.codegen import generate_repair
from agent.evaluation import evaluate, has_error
from agent.execution import make_preamble, run_code


def repair_loop(
    ctx,
    plan: dict,
    docs: str,
    tokens: dict,
    failed_code: str,
    first_error: str,
    timer,
    max_attempts: int = 3,
) -> tuple[str | None, str, dict | None]:
    """Run up to 3 repair attempts, time-gated by timer.

    Returns:
        (fixed_code, stdout, None)            on success
        (None, last_stdout, last_eval_result) on exhaustion
    """
    current_code = failed_code
    current_error = first_error
    last_eval: dict | None = None
    last_stdout = first_error

    for attempt in range(max_attempts):
        if not timer.should_attempt_repair(attempt):
            break

        log.repair_attempt(attempt + 1, current_error)
        ctx.reflect(f"repair attempt {attempt + 1}: {current_error[:200]}")

        fixed_code = generate_repair(
            ctx,
            plan,
            docs,
            failed_code=current_code,
            execution_error=current_error,
            evaluator_failure=last_eval,
        )
        log.codegen_result(f"repair-{attempt + 1}", fixed_code)

        preamble = make_preamble(tokens)
        stdout = run_code(ctx, preamble + fixed_code)
        last_stdout = stdout
        log.exec_result(stdout)

        if has_error(stdout):
            current_code = fixed_code
            current_error = stdout
            last_eval = None
            continue

        eval_result = evaluate(
            stdout,
            plan.get("task_type", "action"),
            plan.get("expected_output_type", "none"),
        )

        if eval_result["passed"]:
            log.repair_success(attempt + 1)
            return fixed_code, stdout, None

        log.eval_fail(eval_result["failure_reason"], eval_result["actual_output"],
                      plan.get("expected_output_type", "none"))
        last_eval = eval_result
        current_code = fixed_code
        current_error = ""

    log.repair_exhausted()
    return None, last_stdout, last_eval


def reflect(ctx, note: str) -> None:
    ctx.reflect(note)
