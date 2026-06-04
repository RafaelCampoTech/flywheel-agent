"""Syntactic output evaluator — checks format only, not world state."""

from __future__ import annotations

import ast


def extract_answer_from_stdout(stdout: str | None) -> str | None:
    """Pull the ANSWER= value from stdout, or return None.

    Normalizes Python list repr to comma-separated string so the oracle's
    .split(',') works correctly: ['a', 'b'] → 'a, b'
    """
    if not stdout or "ANSWER=" not in stdout:
        return None
    raw = stdout.split("ANSWER=", 1)[1].strip().splitlines()[0].strip()
    if not raw:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list):
                return ", ".join(str(x).strip() for x in items)
        except (ValueError, SyntaxError):
            pass
    return raw


def evaluate(
    stdout: str | None,
    task_type: str,
    expected_output_type: str = "none",
) -> dict:
    """Return {passed, failure_reason, expected_type, actual_output}.

    Uses extract_answer_from_stdout so actual_output always matches what
    complete_task will submit to the oracle.
    """
    stdout = stdout or ""

    if task_type == "action":
        # Only treat hard execution failures as errors; AppWorld API responses
        # often embed "error" fields in JSON output that get printed innocuously.
        is_error = "Traceback" in stdout or stdout.strip().startswith("Execution failed")
        if is_error:
            return {
                "passed": False,
                "failure_reason": f"execution error in stdout: {stdout[:300]}",
                "expected_type": "none",
                "actual_output": stdout[:300],
            }
        return {"passed": True, "failure_reason": "", "expected_type": "none", "actual_output": ""}

    # Response task — extract and normalize the answer the same way complete_task does.
    answer = extract_answer_from_stdout(stdout) or ""

    if not answer:
        return {
            "passed": False,
            "failure_reason": "ANSWER= marker not found or empty in stdout",
            "expected_type": expected_output_type,
            "actual_output": stdout[:300],
        }

    if expected_output_type == "number":
        try:
            float(answer.replace(",", ""))
            return {"passed": True, "failure_reason": "", "expected_type": "number", "actual_output": answer}
        except ValueError:
            return {
                "passed": False,
                "failure_reason": f"expected number, got: {answer!r}",
                "expected_type": "number",
                "actual_output": answer,
            }

    if expected_output_type == "yes_no":
        if answer.lower() not in ("yes", "no"):
            return {
                "passed": False,
                "failure_reason": f"expected 'yes' or 'no', got: {answer!r}",
                "expected_type": "yes_no",
                "actual_output": answer,
            }
        return {"passed": True, "failure_reason": "", "expected_type": "yes_no", "actual_output": answer}

    if expected_output_type == "list":
        items = [x.strip() for x in answer.split(",") if x.strip()]
        if len(items) < 2:
            return {
                "passed": False,
                "failure_reason": f"expected comma-separated list with ≥2 items, got {len(items)}: {answer!r}",
                "expected_type": "list",
                "actual_output": answer,
            }
        return {"passed": True, "failure_reason": "", "expected_type": "list", "actual_output": answer}

    return {"passed": True, "failure_reason": "", "expected_type": expected_output_type, "actual_output": answer}


def has_error(result) -> bool:
    """Check if a run_code result indicates a hard execution error."""
    if result is None:
        return True
    if isinstance(result, str):
        return "Traceback" in result or result.strip().startswith("Execution failed")
    if isinstance(result, dict):
        return bool(result.get("error"))
    return False


def is_complete(result: dict) -> bool:
    return bool(result and not result.get("error") and result.get("ok"))
