"""Syntactic output evaluator — checks format only, not world state."""

from __future__ import annotations


def evaluate(
    stdout: str | None,
    task_type: str,
    expected_output_type: str = "none",
) -> dict:
    """Return {passed, failure_reason, expected_type, actual_output}."""
    stdout = stdout or ""

    if task_type == "action":
        is_error = (
            "Traceback" in stdout
            or "Error:" in stdout
            or stdout.lower().startswith("error")
        )
        if is_error:
            return {
                "passed": False,
                "failure_reason": f"execution error in stdout: {stdout[:300]}",
                "expected_type": "none",
                "actual_output": stdout[:300],
            }
        return {"passed": True, "failure_reason": "", "expected_type": "none", "actual_output": ""}

    # Response task — must have ANSWER= marker
    if "ANSWER=" not in stdout:
        return {
            "passed": False,
            "failure_reason": "ANSWER= marker not found in stdout",
            "expected_type": expected_output_type,
            "actual_output": stdout[:300],
        }

    answer = stdout.split("ANSWER=", 1)[1].strip().splitlines()[0].strip()

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

    if not answer:
        return {
            "passed": False,
            "failure_reason": "answer is empty",
            "expected_type": expected_output_type,
            "actual_output": "",
        }

    return {"passed": True, "failure_reason": "", "expected_type": expected_output_type, "actual_output": answer}


def extract_answer_from_stdout(stdout: str | None) -> str | None:
    """Pull the ANSWER= value from stdout, or return None."""
    if not stdout or "ANSWER=" not in stdout:
        return None
    return stdout.split("ANSWER=", 1)[1].strip().splitlines()[0].strip() or None


def has_error(result) -> bool:
    """Backward-compat helper: check if a run_code result indicates an error."""
    if result is None:
        return True
    if isinstance(result, str):
        return "Traceback" in result or "Error:" in result
    if isinstance(result, dict):
        return bool(result.get("error"))
    return False


def is_complete(result: dict) -> bool:
    return bool(result and not result.get("error") and result.get("ok"))
