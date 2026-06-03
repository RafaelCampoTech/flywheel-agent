"""Debug tool: natural-language question -> RAG -> API doc -> token -> execute -> print.

  set -a; source .env; set +a
  python tools/question_to_api.py "show my spotify song library page 0"

Pipeline:
  1. Cosine RAG (15 candidates) + LLM rerank -> keep best 1 API
  2. Load full endpoint definition from api_docs_dump/{app}.json
  3. Bootstrap access_token(s) for involved app(s) via AppWorld
  4. Model writes a single Python snippet that calls the best API
  5. Execute in AppWorld and print stdout / errors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.rag import RAG_RETRIEVE_K, cosine_similarity_search  # noqa: E402

MIN_RAG_SCORE = 0.14
# Always retrieve many, LLM-rerank, keep only the single best API for execution.
RAG_FINAL_TOP_K = 1
AUTH_APPS = {
    "spotify", "amazon", "gmail", "phone", "venmo", "splitwise",
    "todoist", "simple_note", "file_system",
}
SKIP_LOGIN = {"supervisor", "api_docs"}


def _api_docs_root() -> Path:
    root = Path(os.environ.get("API_DOCS_DUMP", "api_docs_dump"))
    if not root.is_absolute():
        root = _REPO / root
    return root


def _render_api_doc(doc: Dict[str, Any]) -> str:
    lines = [
        f"{doc.get('app_name')}.{doc.get('api_name')}  [{doc.get('method')} {doc.get('path')}]",
        doc.get("description", ""),
    ]
    for p in doc.get("parameters") or []:
        req = "required" if p.get("required") else "optional"
        lines.append(
            f"  - {p.get('name')} ({p.get('type', '?')}, {req}): {p.get('description', '')}"
        )
    schema = (doc.get("response_schemas") or {}).get("success")
    if schema is not None:
        lines.append("  response: " + json.dumps(schema)[:800])
    return "\n".join(lines)


def fetch_endpoint_definitions(rag_docs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map RAG hits to full definitions from api_docs_dump."""
    dump = _api_docs_root()
    out: List[Dict[str, Any]] = []
    for hit in rag_docs.get("results") or []:
        if not isinstance(hit, dict):
            continue
        app, api = hit.get("app", ""), hit.get("api", "")
        if not app or not api:
            continue
        entry: Dict[str, Any] = {
            "app": app,
            "api": api,
            "score": hit.get("cosine_similarity", hit.get("score")),
        }
        app_file = dump / f"{app}.json"
        if app_file.is_file():
            try:
                data = json.loads(app_file.read_text(encoding="utf-8"))
                if api in data and isinstance(data[api], dict):
                    doc = data[api]
                    entry["definition"] = doc
                    entry["definition_text"] = _render_api_doc(doc)
                    out.append(entry)
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        txt = dump / "apis" / f"{app}__{api}.txt"
        if txt.is_file():
            entry["definition_text"] = txt.read_text(encoding="utf-8")
            out.append(entry)
    return out


def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def _bootstrap_tokens(env, apps: List[str]) -> Dict[str, str]:
    apps_to_login = [a for a in apps if a in AUTH_APPS]
    if not apps_to_login:
        return {}
    apps_literal = json.dumps(apps_to_login)
    code = (
        "import json\n"
        "me = apis.supervisor.show_profile()\n"
        'pw = {p["account_name"]: p["password"] for p in apis.supervisor.show_account_passwords()}\n'
        "tokens = {}\n"
        f"for app_name in {apps_literal}:\n"
        "    if app_name not in pw:\n"
        "        continue\n"
        '    username = me["phone_number"] if app_name == "phone" else me["email"]\n'
        "    try:\n"
        "        res = getattr(apis, app_name).login(username=username, password=pw[app_name])\n"
        '        tokens[app_name] = res.get("access_token", "")\n'
        "    except Exception:\n"
        '        tokens[app_name] = ""\n'
        'print("__TOKENS__" + json.dumps(tokens))\n'
    )
    stdout = env.execute(code)
    tokens: Dict[str, str] = {}
    m = re.search(r"__TOKENS__(\{.*\})", stdout or "")
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                tokens = {k: v for k, v in parsed.items() if v}
        except json.JSONDecodeError:
            pass
    return tokens


_PROBE_SYS = (
    "Write ONE short Python snippet for AppWorld. Variable `apis` is ALREADY in scope.\n"
    "Never write import apis or from apis.\n"
    "Variable `access_tokens` is a dict app_name -> token string (already logged in).\n"
    "Call exactly the ONE API in api_definitions (already the LLM-reranked best match).\n"
    "Use EXACT api names and parameters from definition_text.\n"
    "Do NOT call apis.supervisor.complete_task().\n"
    "Paginate list APIs when the question implies 'all' items (page_index until empty).\n"
    "At the end: import json; print('__RESULT__' + json.dumps(result, default=str))\n"
    'Reply JSON: {"app":"...","api":"...","code":"..."}'
)


def _generate_call_code(
    question: str,
    api_definitions: List[Dict[str, Any]],
    access_tokens: Dict[str, str],
    *,
    model_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """Use the model to produce Python that calls the best API for this question."""
    user = {
        "question": question,
        "api_definitions": api_definitions,
        "access_tokens": {k: (v[:8] + "...") for k, v in access_tokens.items()},
    }
    messages = [
        {"role": "system", "content": _PROBE_SYS},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]

    if model_fn is not None:
        raw = model_fn(messages)
        parsed = _parse_json(raw)
        model_error = None if parsed.get("code") else "empty_model_response"
    else:
        from flywheel.proxy import chat, content_of

        key = os.environ.get("FLYWHEEL_KEY", "")
        url = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"
        if not key:
            return {"error": "FLYWHEEL_KEY not set", "code": ""}
        data, remaining = chat(
            url,
            key,
            messages,
            response_format={"type": "json_object"},
        )
        if remaining is not None:
            print(f"[question_to_api] tokens remaining: {remaining}")
        parsed = _parse_json(content_of(data))
        model_error = data.get("error") if isinstance(data, dict) else None

    code = parsed.get("code", "")
    code = re.sub(r"^\s*(import apis|from apis\b).*$", "", code, flags=re.MULTILINE).strip()
    tokens_literal = json.dumps(access_tokens)
    full = f"access_tokens = {tokens_literal}\n{code}"
    best = (api_definitions[0] if api_definitions else {}) or {}
    return {
        "app": parsed.get("app", "") or best.get("app", ""),
        "api": parsed.get("api", "") or best.get("api", ""),
        "code": full,
        "model_error": model_error,
    }


def rag_and_definitions(
    question: str,
    *,
    min_score: float = MIN_RAG_SCORE,
    retrieve_k: int = RAG_RETRIEVE_K,
    verbose: bool = False,
) -> Dict[str, Any]:
    """RAG + rerank + load API definition (no execute)."""
    rag = cosine_similarity_search(
        question,
        top_k=RAG_FINAL_TOP_K,
        min_score=min_score,
        retrieve_k=retrieve_k,
        use_llm_rerank=True,
        verbose=verbose,
    )
    best = (rag.get("results") or [None])[0]
    if not best:
        return {"question": question, "error": "no RAG hits", "rag": rag}
    rag["results"] = [best]
    api_definitions = fetch_endpoint_definitions(rag)
    return {
        "question": question,
        "rag": rag,
        "best_hit": best,
        "api_definitions": api_definitions,
        "app": best.get("app"),
        "api": best.get("api"),
    }


def question_to_api_call_ctx(
    ctx: Any,
    question: str,
    access_tokens: Dict[str, str],
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Probe one sub-question using the agent's ctx (same task env, shared tokens)."""
    def p(msg: str) -> None:
        if verbose:
            print(msg)

    p(f"\n=== question_to_api (ctx) ===\nQ: {question}\n")

    resolved = rag_and_definitions(question, verbose=verbose)
    if resolved.get("error"):
        return {**resolved, "ok": False, "access_tokens": access_tokens}

    api_definitions = resolved.get("api_definitions") or []
    best = resolved.get("best_hit") or {}

    def model_fn(messages: List[Dict[str, str]]) -> str:
        data = ctx.model(messages, response_format={"type": "json_object"})
        if isinstance(data, dict) and data.get("error"):
            return ""
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    gen = _generate_call_code(
        question, api_definitions, access_tokens, model_fn=model_fn
    )
    if gen.get("model_error"):
        return {
            **resolved,
            "error": gen["model_error"],
            "ok": False,
            "access_tokens": access_tokens,
        }

    code = gen.get("code", "")
    p(f"\n--- probe call ({gen.get('app')}.{gen.get('api')}) ---\n{code}")

    stdout = ctx.run_code(code)
    failed = any(
        x in (stdout or "")
        for x in ("Traceback (most recent call last)", "Exception:", "Execution failed")
    )
    result = _extract_result(stdout) if not failed else None

    p("\n--- probe stdout ---")
    p((stdout or "")[:2000])
    p("\n--- probe result ---")
    p(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000] if result else "(none)")

    return {
        **resolved,
        "access_tokens": access_tokens,
        "app": gen.get("app") or best.get("app"),
        "api": gen.get("api") or best.get("api"),
        "code": code,
        "stdout": stdout,
        "result": result,
        "ok": not failed,
        "error": (stdout or "")[-800:] if failed else None,
    }


def _extract_result(stdout: str) -> Any:
    if not stdout:
        return None
    m = re.search(r"__RESULT__(.+)", stdout, re.DOTALL)
    if not m:
        return stdout.strip()
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return m.group(1).strip()


def question_to_api_call(
    question: str,
    *,
    min_score: float = MIN_RAG_SCORE,
    retrieve_k: int = RAG_RETRIEVE_K,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Turn a natural-language question into an executed API call.

    Returns dict with keys: question, rag, api_definitions, access_tokens, code, stdout, result, error.
    """
    def p(msg: str) -> None:
        if verbose:
            print(msg)

    p(f"\n=== question_to_api ===\nQ: {question}\n")

    resolved = rag_and_definitions(
        question, min_score=min_score, retrieve_k=retrieve_k, verbose=verbose
    )
    rag = resolved.get("rag") or {}
    best = resolved.get("best_hit")
    if not best:
        err = resolved.get("error") or "no RAG hits (try lower min_score or rebuild index)"
        return {"question": question, "error": err, "rag": rag}

    p("\n--- RAG (retrieve + LLM rerank, best only) ---")
    p(f"  retrieve_k={rag.get('retrieve_k')}  reranked={rag.get('reranked')}  final_k={rag.get('final_k')}")
    p(
        f"  BEST: {best.get('app')}.{best.get('api')}  "
        f"cosine={best.get('cosine_similarity')}  llm_rank={best.get('llm_rank')}\n"
        f"  reason: {(best.get('rerank_reason') or '')[:200]}\n"
        f"  {best.get('description', '')[:120]}"
    )

    api_definitions = resolved.get("api_definitions") or []
    p("\n--- API definitions ---")
    for d in api_definitions:
        p(f"\n[{d.get('app')}.{d.get('api')}] score={d.get('score')}")
        p(d.get("definition_text", "")[:1200])

    # AppWorld + tokens (standalone CLI uses its own env)
    os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))
    from flywheel.appworld_env import AppWorldEnv, load_task_ids

    env = AppWorldEnv(load_task_ids("dev")[0], experiment_name="question_to_api")
    apps = [best["app"]] if best.get("app") not in SKIP_LOGIN else []
    access_tokens = _bootstrap_tokens(env, apps)
    p("\n--- access tokens ---")
    for app, tok in access_tokens.items():
        p(f"  {app}: {tok[:16]}..." if tok else f"  {app}: (failed)")
    if not access_tokens and any(a in AUTH_APPS for a in apps):
        p("  warning: no tokens obtained; authenticated calls may fail")

    gen = _generate_call_code(question, api_definitions, access_tokens)
    if gen.get("model_error"):
        env.close()
        return {"question": question, "error": gen["model_error"], "rag": rag, "api_definitions": api_definitions}

    code = gen.get("code", "")
    p(f"\n--- generated call ({gen.get('app')}.{gen.get('api')}) ---")
    p(code)

    stdout = env.execute(code)
    env.close()

    failed = any(
        x in (stdout or "")
        for x in ("Traceback (most recent call last)", "Exception:", "Execution failed")
    )
    result = _extract_result(stdout) if not failed else None

    p("\n--- execution stdout ---")
    p(stdout or "(empty)")

    p("\n--- parsed result ---")
    p(json.dumps(result, indent=2, ensure_ascii=False, default=str) if result is not None else "(none)")

    return {
        "question": question,
        "rag": rag,
        "best_hit": best,
        "api_definitions": api_definitions,
        "access_tokens": access_tokens,
        "app": gen.get("app") or best.get("app"),
        "api": gen.get("api") or best.get("api"),
        "code": code,
        "stdout": stdout,
        "result": result,
        "ok": not failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug: question -> RAG -> API definition -> token -> execute"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Natural language question (words joined if multiple)",
    )
    parser.add_argument("-q", "--query", help="Question (alternative to positional args)")
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RAG_RETRIEVE_K,
        help=f"Cosine candidates before LLM rerank (default {RAG_RETRIEVE_K})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_RAG_SCORE,
        help=f"Minimum cosine score (default {MIN_RAG_SCORE})",
    )
    parser.add_argument("--json", action="store_true", help="Print final payload as JSON only")
    args = parser.parse_args()

    q = args.query or " ".join(args.question).strip()
    if not q:
        parser.error("Provide a question: python tools/question_to_api.py \"your question\"")

    out = question_to_api_call(
        q, min_score=args.min_score, retrieve_k=args.retrieve_k, verbose=not args.json
    )
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
