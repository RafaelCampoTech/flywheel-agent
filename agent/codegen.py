"""Three-mode code generation: Scratch, Adapt, Repair."""

from __future__ import annotations

import agent.logger as log
from flywheel.proxy import content_of

_SHARED_SYSTEM = """You are a Python code generator for AppWorld.

RULES — follow exactly:
- `apis` is already in scope. Never import it.
- `tokens` dict is pre-injected as a preamble. Use tokens[app_name] for authenticated calls.
  Example: apis.spotify.show_song_library(access_token=tokens["spotify"], page_index=0)
- Always paginate list APIs: loop page_index=0,1,2,... until a page returns fewer items than page_limit.
- For RESPONSE tasks: end with EXACTLY: print("ANSWER=" + str(ANSWER))
- For ACTION tasks: perform the action. No print needed.
- NEVER call complete_task inside the generated code.
- Output format must match: number as int/float, list as comma-separated string,
  yes/no as lowercase string, single value as exact text.
- App name keys for tokens: spotify, amazon, gmail, phone, venmo, splitwise,
  todoist, simple_note, file_system.
- Check API responses for errors before proceeding (check for "message" or "error" keys).
- Output raw Python only — no markdown fences.
"""


def _extract_code(raw: str) -> str:
    if "```python" in raw:
        raw = raw.split("```python", 1)[1]
        raw = raw.split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1]
        raw = raw.split("```", 1)[0]
    return raw.strip()


def generate_scratch(ctx, plan: dict, docs: str) -> str:
    """Generate code from scratch based on plan and docs."""
    steps_text = "\n".join(
        f"  {s['step_id']}. [{s['app']}] {s['description']}"
        for s in plan.get("steps", [])
    )
    user_content = (
        f"Task: {plan['objective']}\n"
        f"Task type: {plan['task_type']}\n"
        f"Expected output type: {plan.get('expected_output_type', 'none')}\n"
        f"Steps:\n{steps_text}\n\n"
        f"API docs:\n{docs[:8000]}\n\n"
        "Write the complete Python solution."
    )
    messages = [
        {"role": "system", "content": _SHARED_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt("scratch", _SHARED_SYSTEM, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result("scratch", code)
    return code


def generate_adapt(ctx, plan: dict, docs: str, similar_solution: dict) -> str:
    """Adapt a previously working solution to the current task."""
    user_content = (
        f"Task: {plan['objective']}\n"
        f"Task type: {plan['task_type']}\n"
        f"Expected output type: {plan.get('expected_output_type', 'none')}\n\n"
        f"Here is a previously working solution for a similar task:\n"
        f"--- SIMILAR TASK: {similar_solution['task_description']} ---\n"
        f"{similar_solution['working_code']}\n"
        f"--- END ---\n\n"
        "Modify only the parts that differ for the current task. "
        "Preserve pagination logic, token usage, and error handling.\n\n"
        f"API docs:\n{docs[:6000]}\n\n"
        "Write the complete Python solution."
    )
    messages = [
        {"role": "system", "content": _SHARED_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt("adapt", _SHARED_SYSTEM, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result("adapt", code)
    return code


def generate_repair(
    ctx,
    plan: dict,
    docs: str,
    failed_code: str,
    execution_error: str,
    evaluator_failure: dict | None,
) -> str:
    """Repair failed code given the execution error and evaluator feedback."""
    eval_msg = ""
    if evaluator_failure:
        eval_msg = (
            f"\nEvaluator failure: expected {evaluator_failure.get('expected_type')}, "
            f"got: {evaluator_failure.get('actual_output')!r}. "
            f"Reason: {evaluator_failure.get('failure_reason')}"
        )
    user_content = (
        f"Task: {plan['objective']}\n"
        f"Task type: {plan['task_type']}\n"
        f"Expected output type: {plan.get('expected_output_type', 'none')}\n\n"
        f"The following code failed:\n```python\n{failed_code}\n```\n\n"
        f"Execution error:\n{execution_error[:2000]}\n"
        f"{eval_msg}\n\n"
        f"API docs:\n{docs[:6000]}\n\n"
        "Identify the root cause. Fix only what is broken. "
        "Write the complete corrected Python solution."
    )
    messages = [
        {"role": "system", "content": _SHARED_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt("repair", _SHARED_SYSTEM, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result("repair", code)
    return code
