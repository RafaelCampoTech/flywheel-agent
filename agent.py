"""Your agent. This is the only file you have to write.

The harness calls solve(ctx) once per task. A naive single-shot agent on this model scores ~0.
To clear the baseline you need all four capabilities to be REAL, because the grader checks the
trace and runs you with memory on AND wiped:

  - RAG        retrieve the right API docs per task (you can't fit 457 in context)
  - tools      act through the MCP tool surface (ctx.mcp), not hardcoded calls
  - memory     carry what you learn across tasks; the on/off gap is graded
  - self-loop  read tool errors, reflect, retry the fixed call

Act through ctx.mcp.call(name, args): on the graded run that's the MCP gateway, and it is what
the gate counts. (Locally, ctx.execute(code) runs the same AppWorld in-process for fast
iteration -- but write your solver against ctx.mcp so it runs identically when graded.)

Read docs/appworld.md first (especially THE LOGIN FLOW) and docs/proxy.md. Then build here.

ctx surface (see flywheel/ and the README):
  ctx.instruction          the task
  ctx.model(messages, tools=None, response_format=None)   the fixed model via the proxy
  ctx.mcp.call(name, args) the tool surface: search_apis, api_doc, call_api, run_code, complete_task
  ctx.retrieve(query)      your RAG hook (wire flywheel/ctx.py:retrieve to a real retriever)
  ctx.memory.read() / .write(k, v)   persisted across tasks (wiped between tasks on the off-arm)
  ctx.reflect(note)        record a self-correction
  ctx.trace(type, **kw)    append a JSONL event
"""



from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json


_JSON_RE = re.compile(r"\{[\s\S]*\}")
_MARKER = "__FW_CODEACT__"


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output; repair common JSON drift."""
    if not text:
        return {}
    text = text.strip()
    # First pass: strict JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Second: repair then parse
    try:
        fixed = repair_json(text)
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Last resort: parse the first {...} blob
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        fixed = repair_json(m.group(0))
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _content_of_model_response(data: Any) -> str:
    if not isinstance(data, dict) or data.get("error"):
        return ""
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    return msg.get("content") or ""


def _short_key(instruction: str) -> str:
    s = re.sub(r"\s+", " ", (instruction or "").strip().lower())
    s = re.sub(r"[^a-z0-9 ,._-]+", "", s)
    return s[:140] or "unknown"


def _memory_procedures(memory: Dict[str, Any]) -> Dict[str, Any]:
    procs = memory.get("procedures")
    return procs if isinstance(procs, dict) else {}


def _top_procedure_candidates(instruction: str, procedures: Dict[str, Any], k: int = 2) -> List[Dict[str, Any]]:
    """Very simple retrieval over stored procedures: substring overlap on signature key."""
    if not procedures:
        return []
    inst = (instruction or "").lower()
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for sig, proc in procedures.items():
        if not isinstance(sig, str) or not isinstance(proc, dict):
            continue
        s = sig.lower()
        score = 0
        for token in re.split(r"[^a-z0-9]+", s):
            if token and token in inst:
                score += 1
        if score > 0:
            scored.append((score, {"signature": sig, **proc}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def _model_plan(
    ctx,
    instruction: str,
    retrieval: Any,
    memory: Dict[str, Any],
    last_error: str,
    attempted_signatures: List[str],
) -> Dict[str, Any]:
    """Ask the model for a JSON plan: work_code + verify_code + kind (+ answer for question tasks)."""
    procedures = _memory_procedures(memory)
    candidates = _top_procedure_candidates(instruction, procedures, k=2)
    for c in candidates:
        # avoid re-trying the same stored procedure repeatedly
        if c.get("signature") in attempted_signatures:
            c["deprioritize"] = True

    sys = (
        "You are controlling AppWorld by writing Python code that runs with `apis` in scope.\n"
        "You must solve the instruction by interacting with AppWorld APIs.\n"
        "\n"
        "CRITICAL:\n"
        "- Do NOT call apis.supervisor.complete_task() in your code snippets.\n"
        "- Produce TWO snippets: work_code then verify_code.\n"
        "  - work_code: does the work only. MUST be idempotent (read-before-write; only apply missing changes).\n"
        "  - verify_code: reads back state to confirm the goal is met. It MUST print one line that starts with\n"
        f"    {_MARKER} followed by a JSON object: {{\"verified\": bool, \"answer\": str, \"notes\": str}}.\n"
        "- No traceback does NOT imply correctness; verification is mandatory.\n"
        "- Avoid guessing API names; use apis.api_docs.show_api_descriptions/show_api_doc when uncertain.\n"
        "- Login flow is commonly required: supervisor profile + passwords -> app.login -> access_token.\n"
        "- Many list APIs are paginated: loop page_index until empty/short.\n"
        "\n"
        "Output format: a single JSON object with keys:\n"
        "- kind: 'question' | 'action'\n"
        "- work_code: string\n"
        "- verify_code: string\n"
        "- answer: string (only meaningful for kind='question'; may be empty if verify_code computes it)\n"
    )

    user = {
        "instruction": instruction,
        "retrieval": retrieval,
        "proven_procedures": candidates,  # may be empty
        "last_error": last_error,
    }

    data = ctx.model(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    return _extract_json_obj(_content_of_model_response(data))


def _looks_like_traceback(out: Any) -> bool:
    if not isinstance(out, str):
        return False
    bad = ("Traceback (most recent call last)", "Exception:", "KeyError", "TypeError", "NameError")
    return any(b in out for b in bad)


def _marker_json_from_stdout(stdout: str) -> Dict[str, Any]:
    if not isinstance(stdout, str) or not stdout:
        return {}
    for line in reversed(stdout.splitlines()):
        if line.startswith(_MARKER):
            payload = line[len(_MARKER) :].strip()
            return _extract_json_obj(payload)
    return {}


def _run_snippet(ctx, code: str) -> Tuple[bool, str]:
    if not isinstance(code, str) or not code.strip():
        return False, "missing_code"
    out = ctx.run_code(code)
    if _looks_like_traceback(out):
        return False, str(out)[-2000:]
    return True, str(out or "")


def solve(ctx):
    instruction = ctx.instruction or ""

    # Memory (cross-task): store *proven procedures* to create a real on/off gap.
    memory = ctx.memory.read() or {}
    if not isinstance(memory, dict):
        memory = {}
    memory.setdefault("procedures", {})

    retrieval = ctx.retrieve(instruction)

    max_tries = 5
    last_error = ""
    attempted_signatures: List[str] = []
    for attempt in range(1, max_tries + 1):
        plan = _model_plan(ctx, instruction, retrieval, memory, last_error, attempted_signatures)
        kind = plan.get("kind", "question")
        work_code = plan.get("work_code", "")
        verify_code = plan.get("verify_code", "")

        ok_work, out_work = _run_snippet(ctx, work_code)
        if not ok_work:
            last_error = f"work_failed: {out_work}"
            ctx.reflect(f"attempt {attempt}: work snippet failed: {out_work[:350]}")
        else:
            ok_ver, out_ver = _run_snippet(ctx, verify_code)
            if not ok_ver:
                last_error = f"verify_failed: {out_ver}"
                ctx.reflect(f"attempt {attempt}: verify snippet failed: {out_ver[:350]}")
            else:
                verdict = _marker_json_from_stdout(out_ver)
                verified = bool(verdict.get("verified"))
                answer = (verdict.get("answer") if isinstance(verdict.get("answer"), str) else "") or ""
                notes = (verdict.get("notes") if isinstance(verdict.get("notes"), str) else "") or ""

                if verified:
                    # Submit explicitly via the gateway tool (trace-safe).
                    if kind == "action":
                        ctx.mcp.call("complete_task", {})
                    else:
                        # Prefer verify-produced answer; fall back to plan.answer.
                        if not answer:
                            answer = plan.get("answer", "") if isinstance(plan.get("answer"), str) else ""
                        ctx.mcp.call("complete_task", {"answer": answer})

                    # Persist a proven procedure (write only on verified success).
                    sig = _short_key(instruction)
                    proc = {
                        "kind": kind,
                        "work_code": work_code[:8000],
                        "verify_code": verify_code[:8000],
                        "notes": notes[:500],
                    }
                    if isinstance(memory.get("procedures"), dict):
                        memory["procedures"][sig] = proc
                    ctx.memory.write("procedures", memory.get("procedures"))
                    return

                last_error = f"verification_failed: {notes or 'state mismatch'}"
                ctx.reflect(f"attempt {attempt}: verification failed (no traceback). notes={notes[:350]!r}")

        # On repeated failure, broaden retrieval slightly instead of re-asking blindly.
        if attempt == 2:
            retrieval = ctx.retrieve(f"{instruction}\n\nFocus on login/token/pagination and exact api names.")
        if attempt == 3:
            retrieval = ctx.retrieve(
                "Readback verification idempotent write read-before-write "
                "supervisor show_profile show_account_passwords api_docs show_api_doc "
                "login access_token page_index page_limit"
            )
        if attempt == 4 and isinstance(memory.get("procedures"), dict):
            # After enough failures, try to bias toward reusing stored proven procedure shapes.
            attempted_signatures.extend(list(memory["procedures"].keys())[:3])

    # Failsafe: at least submit *something* so the run is well-formed.
    ctx.reflect(f"giving up after retries; last_error={last_error[:400]}")
    ctx.mcp.call("complete_task", {"answer": "<<not_solved>>"})
