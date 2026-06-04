"""Structured task planner using JSON response_format."""

from __future__ import annotations

import json

from flywheel.proxy import content_of

_SYSTEM = """You are a task planner for AppWorld. Given a task instruction and API docs,
produce a JSON plan. Respond ONLY with valid JSON matching this exact schema:

{
  "objective": "<one-sentence description>",
  "task_type": "response" or "action",
  "apps_involved": ["<app_name>", ...],
  "expected_output_type": "number" | "list" | "yes_no" | "string" | "none",
  "steps": [
    {"step_id": 1, "description": "<what to do>", "app": "<app_name>", "depends_on": []}
  ]
}

task_type:
- "response": task asks a question; code must end with print("ANSWER=" + str(ANSWER))
- "action": task requests an action; no ANSWER print needed

expected_output_type (response tasks only):
- "number"  — a count, price, integer, or float
- "yes_no"  — a yes/no question
- "list"    — multiple items as a comma-separated string
- "string"  — a name, title, or single text value
- "none"    — action task

App names: spotify, amazon, gmail, phone, venmo, splitwise, todoist, simple_note, file_system
"""


def plan(
    ctx,
    task: str,
    docs: str,
    similar_solutions: list,
    failure_context: dict | None = None,
) -> dict:
    """Return a structured plan dict. Accepts failure_context for re-plan."""
    user_parts = [f"Task: {task}\n\nRelevant API docs:\n{docs[:6000]}"]

    if similar_solutions:
        examples = []
        for sol in similar_solutions[:2]:
            examples.append(
                f"Similar task: {sol['task_description']}\n"
                f"Apps used: {sol['apis_used']}\n"
                f"Plan: {json.dumps(sol.get('plan', {}), indent=2)}"
            )
        user_parts.append("Previously solved similar tasks:\n" + "\n\n".join(examples))

    if failure_context:
        user_parts.append(
            "REPLAN: A previous attempt failed after 3 repairs.\n"
            f"Failure: {json.dumps(failure_context)}\n"
            "Generate a new plan that avoids this failure. Be more explicit about "
            "output format requirements and API call ordering."
        )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    response = ctx.model(messages, response_format={"type": "json_object"})
    raw = content_of(response)

    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        result = {}

    result.setdefault("objective", task)
    result.setdefault("task_type", "action")
    result.setdefault("apps_involved", [])
    result.setdefault("expected_output_type", "none")
    result.setdefault("steps", [])
    return result
