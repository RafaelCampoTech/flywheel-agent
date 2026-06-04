"""Step-by-step plan executor: one API call per step, exact docs via MCP or api_docs_dump."""

from __future__ import annotations

import json
import os
from pathlib import Path

import agent.logger as log
from agent.evaluation import has_error
from agent.execution import make_preamble, run_code
from flywheel.proxy import content_of

# ── API doc loading ─────────────────────────────────────────────────────────

_DUMP_ROOT = Path(os.environ.get("API_DOCS_DUMP", "api_docs_dump"))
if not _DUMP_ROOT.is_absolute():
    _DUMP_ROOT = Path(__file__).resolve().parent.parent / _DUMP_ROOT


def _api_docs_root() -> Path:
    root = Path(os.environ.get("API_DOCS_DUMP", "api_docs_dump"))
    return root if root.is_absolute() else Path(__file__).resolve().parent.parent / root


def _render_api_entry(app: str, api: str, entry: dict) -> str:
    lines = [
        f"{app}.{api}  [{entry.get('method', 'GET')} {entry.get('path', '')}]",
        entry.get("description", ""),
    ]
    for p in entry.get("parameters") or []:
        req = "required" if p.get("required") else "optional"
        constraints = ", ".join(p.get("constraints") or [])
        constraint_str = f" [{constraints}]" if constraints else ""
        lines.append(f"  {p['name']} ({p.get('type', '?')}, {req}){constraint_str}: {p.get('description', '')}")
    schema = (entry.get("response_schemas") or {}).get("success")
    if schema is not None:
        lines.append(f"  response: {json.dumps(schema, default=str)[:400]}")
    return "\n".join(lines)


_RERANK_SYS = """\
Select the 1-3 most relevant APIs for the given plan step from the candidates list.
Return ONLY valid JSON: {"selected": [<indices>], "reason": "<one line>"}
"""


def _keyword_candidates(app: str, query: str, top_k: int = 8) -> list[dict]:
    """Keyword-score all APIs in app's dump file, return top_k as candidate dicts."""
    path = _api_docs_root() / f"{app}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    q_words = set(query.lower().split())
    scored: list[tuple[int, str, dict]] = []
    for api_name, entry in data.items():
        blob = f"{api_name} {entry.get('description', '')}".lower()
        score = sum(1 for w in q_words if w in blob)
        if score:
            scored.append((score, api_name, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"app": app, "api": name, "entry": entry, "score": s}
        for s, name, entry in scored[:top_k]
    ]


def _rerank_apis(ctx, step: dict, candidates: list[dict]) -> list[dict]:
    """LLM rerank: pick the 1-3 most relevant APIs for this step."""
    if len(candidates) <= 2:
        return candidates
    user_content = json.dumps({
        "step": f"[{step.get('app')}] {step.get('description')}",
        "candidates": [
            {"index": i, "app": c["app"], "api": c["api"],
             "description": c["entry"].get("description", "")[:120]}
            for i, c in enumerate(candidates)
        ],
    })
    messages = [
        {"role": "system", "content": _RERANK_SYS},
        {"role": "user", "content": user_content},
    ]
    try:
        response = ctx.model(messages, response_format={"type": "json_object"})
        result = json.loads(content_of(response) or "{}")
        selected = result.get("selected") or []
        reason = result.get("reason", "")
        if selected:
            log.info("API rerank", f"{[candidates[i]['api'] for i in selected if i < len(candidates)]} — {reason[:80]}")
            return [candidates[i] for i in selected if 0 <= i < len(candidates)]
    except Exception:
        pass
    return candidates[:3]


def fetch_step_docs(ctx, step: dict) -> str:
    """Fetch exact API docs for a plan step.

    1. Keyword-search api_docs_dump for 8 candidates.
    2. LLM rerank → keep 1-3 best.
    3. Render full docs (with constraints) for the winners.
    Falls back to MCP search_apis if the dump has no matches.
    """
    app = step.get("app", "")
    description = step.get("description", "")
    query = f"{app} {description}"

    candidates = _keyword_candidates(app, query, top_k=8)
    if candidates:
        best = _rerank_apis(ctx, step, candidates)
        return "\n\n".join(
            _render_api_entry(c["app"], c["api"], c["entry"]) for c in best
        )

    # MCP fallback (graded run or missing dump file)
    try:
        result = ctx.mcp.call("search_apis", {"query": query})
        hits = (result.get("results") or []) if isinstance(result, dict) else []
        parts = []
        for hit in hits[:3]:
            if hit.get("app") == app:
                doc_result = ctx.mcp.call("api_doc", {"app": hit["app"], "api": hit["api"]})
                if isinstance(doc_result, dict) and doc_result.get("doc"):
                    parts.append(f"{hit['app']}.{hit['api']}:\n{str(doc_result['doc'])[:600]}")
        return "\n\n".join(parts)
    except Exception:
        return ""


# ── Code generation ──────────────────────────────────────────────────────────

_STEP_SYS = """\
You are a Python code generator for AppWorld. Generate code for ONE plan step.

RULES:
- `apis` is already in scope. Never import it.
- `tokens` dict is pre-injected. Use tokens[app_name] for authenticated calls.
- Use EXACT api names and parameter names from the api_definition.
- page_limit must respect the documented constraint (usually ≤ 20).
- Paginate list APIs: loop page_index=0,1,2,... until len(page) < page_limit.
- Store results in clearly named variables (e.g. liked_song_ids, song_details).
- Previous step variables are already defined — use them directly.
- Do NOT call complete_task. Do NOT import apis.
- Output raw Python only — no markdown fences.
"""

_REPAIR_SYS = """\
Fix ONLY the broken step code. Use the exact API definition.
Do not touch previous steps' code. Output only the corrected step code (raw Python, no fences).
"""

_FINAL_SYS = """\
You are a Python code generator for AppWorld. Write ONLY the final answer extraction.
Variables from all steps are already defined. Generate code that:
- Derives the final answer from existing variables
- For list answers: ends with  print("ANSWER=" + ", ".join(titles))
- For number answers: ends with  print("ANSWER=" + str(n))
- For yes/no answers: ends with  print("ANSWER=yes") or print("ANSWER=no")
- For string answers: ends with  print("ANSWER=" + str(value))
Output raw Python only — no markdown fences.
"""


def _extract_code(raw: str) -> str:
    if "```python" in raw:
        raw = raw.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    return raw.strip()


def _generate_step_code(
    ctx,
    step: dict,
    task_plan: dict,
    api_doc: str,
    accumulated_code: str,
    snippets: list[str],
) -> str:
    snippet_block = ""
    if snippets:
        examples = "\n---\n".join(s[:350] for s in snippets[:2])
        snippet_block = f"\nRelevant patterns from prior solved tasks:\n{examples}\n"

    context_note = ""
    if accumulated_code:
        # Show only variable assignments from previous steps (not full code)
        var_lines = [ln for ln in accumulated_code.splitlines() if ln and not ln.startswith(" ") and "=" in ln and not ln.strip().startswith("#")]
        context_note = f"\nVariables already defined by previous steps:\n" + "\n".join(var_lines[-20:]) + "\n"

    user_content = (
        f"Task: {task_plan.get('objective', '')}\n"
        f"Task type: {task_plan.get('task_type', 'action')}\n"
        f"Expected output type: {task_plan.get('expected_output_type', 'none')}\n\n"
        f"CURRENT STEP {step.get('step_id')}: [{step.get('app')}] {step.get('description')}\n\n"
        f"API documentation:\n{api_doc[:2500]}\n"
        + context_note
        + snippet_block
        + "\nWrite the Python code for this step only."
    )
    messages = [
        {"role": "system", "content": _STEP_SYS},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt(f"step-{step.get('step_id')}", _STEP_SYS, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result(f"step-{step.get('step_id')}", code)
    return code


def _repair_step_code(ctx, step: dict, api_doc: str, failed_code: str, error: str) -> str:
    user_content = (
        f"Step: [{step.get('app')}] {step.get('description')}\n\n"
        f"API definition:\n{api_doc[:2000]}\n\n"
        f"Failing code:\n```python\n{failed_code[:2000]}\n```\n\n"
        f"Error:\n{error[:600]}\n\n"
        "Fix only this step."
    )
    messages = [
        {"role": "system", "content": _REPAIR_SYS},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt(f"step-{step.get('step_id')}-repair", _REPAIR_SYS, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result(f"step-{step.get('step_id')}-repair", code)
    return code


def _generate_final_code(ctx, task_plan: dict, accumulated_code: str) -> str:
    var_lines = [ln for ln in accumulated_code.splitlines() if ln and not ln.startswith(" ") and "=" in ln and not ln.strip().startswith("#")]
    user_content = (
        f"Task: {task_plan.get('objective', '')}\n"
        f"Expected output type: {task_plan.get('expected_output_type', 'none')}\n\n"
        f"Variables computed by previous steps:\n" + "\n".join(var_lines[-30:]) + "\n\n"
        "Write the final answer extraction code."
    )
    messages = [
        {"role": "system", "content": _FINAL_SYS},
        {"role": "user", "content": user_content},
    ]
    log.codegen_prompt("finalize", _FINAL_SYS, user_content)
    response = ctx.model(messages)
    code = _extract_code(content_of(response))
    log.codegen_result("finalize", code)
    return code


# ── Main entry point ─────────────────────────────────────────────────────────

def execute_plan_stepwise(
    ctx,
    task_plan: dict,
    tokens: dict,
    snippets: list[str],
    timer,
) -> tuple[str, str]:
    """Execute the plan one step at a time, accumulating working code.

    Each step fetches exact API docs, generates targeted code, and repairs on
    error before adding to the accumulated script. Returns (final_code, stdout).
    """
    steps = task_plan.get("steps", [])
    if not steps:
        return "", ""

    accumulated_code = ""

    for step in steps:
        if timer.at_emergency():
            log.emergency(f"step {step.get('step_id')} aborted")
            break

        log.info(f"Step {step.get('step_id')}", f"[{step.get('app')}] {step.get('description', '')[:70]}")

        api_doc = fetch_step_docs(ctx, step)
        if not api_doc:
            log.error(f"Step {step.get('step_id')}", "no API docs found — skipping")
            continue

        step_code = _generate_step_code(ctx, step, task_plan, api_doc, accumulated_code, snippets)
        if not step_code:
            log.error(f"Step {step.get('step_id')}", "codegen returned empty — skipping")
            continue

        full = (accumulated_code + "\n\n" + step_code).strip() if accumulated_code else step_code

        # Execute with one repair attempt
        for attempt in range(2):
            preamble = make_preamble(tokens)
            stdout = run_code(ctx, preamble + full)

            if not has_error(stdout):
                accumulated_code = full
                log.info(f"Step {step.get('step_id')}", "✓ succeeded")
                break

            log.info(f"Step {step.get('step_id')}", f"attempt {attempt + 1} failed — {'repairing' if attempt == 0 else 'skipping'}")
            if attempt == 0 and not timer.at_emergency():
                step_code = _repair_step_code(ctx, step, api_doc, step_code, stdout)
                if not step_code:
                    break
                full = (accumulated_code + "\n\n" + step_code).strip() if accumulated_code else step_code
        else:
            log.error(f"Step {step.get('step_id')}", "failed after repair — skipping")

    if not accumulated_code:
        return "", ""

    # For response tasks: generate the final ANSWER= extraction
    if task_plan.get("task_type") == "response" and not timer.at_emergency():
        final_code = _generate_final_code(ctx, task_plan, accumulated_code)
        if final_code:
            full_final = accumulated_code + "\n\n" + final_code
            preamble = make_preamble(tokens)
            stdout = run_code(ctx, preamble + full_final)
            log.exec_result(stdout)
            return full_final, stdout

    # Action task — execute accumulated code for final stdout
    preamble = make_preamble(tokens)
    stdout = run_code(ctx, preamble + accumulated_code)
    log.exec_result(stdout)
    return accumulated_code, stdout
