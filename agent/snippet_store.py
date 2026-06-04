"""Extract reusable code patterns from prior working solutions for step-level codegen."""

from __future__ import annotations

import re


def get_snippets_for_plan(task_plan: dict, similar_solutions: list) -> list[str]:
    """Return per-app code snippets extracted from similar_solutions."""
    apps = set(task_plan.get("apps_involved") or [])
    for step in task_plan.get("steps", []):
        if step.get("app"):
            apps.add(step["app"])

    snippets: list[str] = []
    seen: set[str] = set()
    for sol in similar_solutions:
        code = sol.get("working_code", "").strip()
        if not code:
            continue
        for app in apps:
            chunk = _extract_app_calls(code, app)
            if chunk and chunk not in seen:
                seen.add(chunk)
                snippets.append(chunk)
    return snippets[:4]


def _extract_app_calls(code: str, app: str) -> str:
    """Extract lines that call a specific app's APIs plus surrounding context."""
    lines = code.splitlines()
    keep: list[str] = []
    added: set[int] = set()

    for i, line in enumerate(lines):
        if f"apis.{app}." in line:
            for j in range(max(0, i - 1), min(len(lines), i + 6)):
                if j not in added:
                    keep.append(lines[j])
                    added.add(j)

    return "\n".join(keep).strip()
