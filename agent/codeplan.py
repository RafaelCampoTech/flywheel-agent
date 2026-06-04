"""Plan Python execution (run_code) from the high-level plan."""

from __future__ import annotations

import json
import re

from flywheel.proxy import content_of

_EVALUATE_SYSTEM = """You are a code plan validator for AppWorld.
Given a task plan, similar examples, and API docs, identify discrepancies.

Check for:
- Steps referencing APIs that don't exist or have wrong method/parameter names
- List APIs that need pagination but the plan doesn't mention it
- Wrong token usage patterns
- Output format mismatches for the expected_output_type

Return a concise bullet list of issues. If none, respond: "No discrepancies found."
"""

_BUILD_SYSTEM = """You are a code planning assistant for AppWorld Python tasks.

Given a high-level task plan, API docs, a discrepancy report, and similar solved examples,
produce a concrete pseudo-code plan.

The plan must specify:
- Exact API method signatures and parameter names (from the docs)
- Pagination loops (page_index=0,1,2,... until len(items) < page_limit)
- Token usage: tokens["app_name"] for each authenticated call
- Intermediate variable names
- For response tasks: how to format ANSWER for print("ANSWER=" + str(ANSWER))

Be concrete. This pseudo-code plan will guide Python code generation.
"""


def extract_apis_from_examples(examples: list) -> list[str]:
    """Extract 'app.method' patterns from similar solution code."""
    api_pattern = re.compile(r'apis\.(\w+)\.(\w+)')
    found: set[str] = set()
    for ex in examples:
        code = ex.get("working_code", "")
        for m in api_pattern.finditer(code):
            found.add(f"{m.group(1)}.{m.group(2)}")
    return sorted(found)


def retrieve_docs_for_apis(candidate_apis: list[str], existing_docs: str) -> str:
    """Return existing docs (hook for targeted retrieval)."""
    return existing_docs


def evaluate_code_plan(ctx, plan: dict, examples: list, docs: str) -> str:
    """LLM call: check plan against docs and examples, return discrepancy report."""
    examples_text = ""
    for ex in examples[:2]:
        examples_text += (
            f"\nSimilar task: {ex.get('task_description', '')}\n"
            f"Code:\n{ex.get('working_code', '')[:800]}\n"
        )

    user_content = (
        f"Task plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"API docs:\n{docs[:5000]}\n\n"
        + (f"Similar examples:{examples_text}\n\n" if examples_text else "")
        + "List discrepancies."
    )
    messages = [
        {"role": "system", "content": _EVALUATE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    response = ctx.model(messages)
    return content_of(response)


def build_code_plan(ctx, task_plan: dict, docs_str: str, examples: list) -> dict:
    """Generate a concrete pseudo-code plan.

    Returns dict: {code_plan: str, docs: str, examples: list}
    """
    candidate_apis = extract_apis_from_examples(examples)
    docs = retrieve_docs_for_apis(candidate_apis, docs_str)
    discrepancy_report = evaluate_code_plan(ctx, task_plan, examples, docs)

    steps_text = "\n".join(
        f"  {s['step_id']}. [{s['app']}] {s['description']}"
        for s in task_plan.get("steps", [])
    )
    examples_text = ""
    for ex in examples[:2]:
        examples_text += (
            f"\nSimilar task: {ex.get('task_description', '')}\n"
            f"Code:\n{ex.get('working_code', '')[:800]}\n"
        )

    user_content = (
        f"Task: {task_plan.get('objective', '')}\n"
        f"Task type: {task_plan.get('task_type', 'action')}\n"
        f"Expected output type: {task_plan.get('expected_output_type', 'none')}\n"
        f"Steps:\n{steps_text}\n\n"
        f"API docs:\n{docs[:6000]}\n\n"
        f"Discrepancy report:\n{discrepancy_report}\n\n"
        + (f"Similar examples:{examples_text}\n\n" if examples_text else "")
        + "Generate the concrete pseudo-code plan."
    )
    messages = [
        {"role": "system", "content": _BUILD_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    response = ctx.model(messages)
    code_plan = content_of(response)

    return {"code_plan": code_plan, "docs": docs, "examples": examples}
