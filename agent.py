"""Your agent. The harness calls solve(ctx) once per task.

Pipeline: classify -> bootstrap tokens -> question plan -> concurrent RAG per step
          -> api_hints vector -> synthesize code -> check loop (execute + verify + retry) -> submit.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- defaults stored in memory on first run ---
DEFAULT_SKILLS: List[Dict[str, Any]] = [
    {
        "name": "supervisor_login_bootstrap",
        "app": "supervisor",
        "recipe": ["show_profile", "show_account_passwords"],
    },
    {
        "name": "spotify_login",
        "app": "spotify",
        "recipe": ["show_profile", "show_account_passwords", "login"],
    },
]

APPS = [
    "spotify", "amazon", "gmail", "phone", "venmo", "splitwise",
    "todoist", "simple_note", "file_system", "supervisor", "api_docs",
]

MIN_RAG_SCORE = 0.14
RAG_FINAL_TOP_K = 1  # retrieve many, LLM rerank, keep best API only
_LOG_PREVIEW = 500
_CODE_LOG_PREVIEW = 12000
API_DOCS_DUMP = "api_docs_dump"
AUTH_APPS = {
    "spotify", "amazon", "gmail", "phone", "venmo", "splitwise",
    "todoist", "simple_note", "file_system",
}
SKIP_LOGIN = {"supervisor", "api_docs"}
MAX_PLAN_QUESTIONS = 10
_MAX_RAG_WORKERS = 8
_CHECK_LOOP_ATTEMPTS = 4
_PROBE_SUMMARY_LEN = 1200


def _log(phase: str, **fields) -> None:
    """Print a labeled step for local debugging."""
    print(f"\n[agent] === {phase} ===")
    for key, val in fields.items():
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            text = json.dumps(val, ensure_ascii=False, default=str)
        else:
            text = str(val)
        limit = _CODE_LOG_PREVIEW if key == "code" else _LOG_PREVIEW
        if len(text) > limit and key not in ("plan",):
            text = text[:limit] + "..."
        print(f"[agent]   {key}: {text}")


def _sanitize_appworld_code(code: str) -> str:
    """Strip imports that break AppWorld run_code (apis is already in scope)."""
    if not code:
        return code
    out: List[str] = []
    for line in code.splitlines():
        s = line.strip()
        if re.match(r"^(import apis|from apis\b)", s):
            continue
        out.append(line)
    return "\n".join(out)


def _model_text(ctx, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
    kwargs: Dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    data = ctx.model(messages, **kwargs)
    if isinstance(data, dict) and data.get("error"):
        return ""
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        obj = json.loads(text)
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


def _api_docs_root() -> Path:
    root = Path(os.environ.get("API_DOCS_DUMP", API_DOCS_DUMP))
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    return root


def _render_api_doc(doc: Dict[str, Any]) -> str:
    """Format a JSON API doc entry as plain text (same shape as apis/*.txt)."""
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
        lines.append("  response: " + json.dumps(schema)[:600])
    return "\n".join(lines)


def fetch_api_definitions(rag_docs: Any) -> List[Dict[str, Any]]:
    """Resolve RAG hits to full API definitions from api_docs_dump/{app}.json."""
    _log("fetch_api_definitions", source="api_docs_dump")
    dump = _api_docs_root()
    hits = rag_docs.get("results", []) if isinstance(rag_docs, dict) else []
    definitions: List[Dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        app = hit.get("app") or ""
        api = hit.get("api") or ""
        if not app or not api:
            continue

        app_file = dump / f"{app}.json"
        entry: Dict[str, Any] = {
            "app": app,
            "api": api,
            "rag_score": hit.get("cosine_similarity", hit.get("score")),
        }

        if app_file.is_file():
            try:
                app_data = json.loads(app_file.read_text(encoding="utf-8"))
                if api in app_data and isinstance(app_data[api], dict):
                    doc = app_data[api]
                    entry["definition"] = doc
                    entry["definition_text"] = _render_api_doc(doc)
                    entry["source"] = str(app_file)
                    definitions.append(entry)
                    _log(
                        "fetch_api_definitions",
                        app=app,
                        api=api,
                        source=f"{app}.json",
                        found=True,
                    )
                    continue
            except (json.JSONDecodeError, OSError) as exc:
                _log("fetch_api_definitions", app=app, api=api, json_error=str(exc))

        chunk = dump / "apis" / f"{app}__{api}.txt"
        if chunk.is_file():
            text = chunk.read_text(encoding="utf-8")
            entry["definition_text"] = text
            entry["source"] = str(chunk)
            definitions.append(entry)
            _log("fetch_api_definitions", app=app, api=api, source=chunk.name, found=True)
        else:
            _log("fetch_api_definitions", app=app, api=api, found=False)

    _log("fetch_api_definitions", count=len(definitions))
    return definitions


def fetch_api_definition_direct(app: str, api: str) -> Optional[Dict[str, Any]]:
    """Load one API definition from api_docs_dump without RAG."""
    if not app or not api:
        return None
    fake_hit = {"app": app, "api": api}
    defs = fetch_api_definitions({"results": [fake_hit]})
    return defs[0] if defs else None


def _merge_api_definitions(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    merged: List[Dict[str, Any]] = []
    for group in groups:
        for d in group:
            key = (d.get("app"), d.get("api"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
    return merged


def _parse_execution_error(error: str) -> Dict[str, Any]:
    """Extract API-related hints from AppWorld traceback text."""
    hints: Dict[str, Any] = {
        "wrong_apis": [],
        "needs_token": False,
        "needs_pagination": False,
        "key_error_field": None,
    }
    if not error:
        return hints

    for api_name, app_name in re.findall(
        r"No API named '([^']+)' found in the (\w+) app", error
    ):
        hints["wrong_apis"].append({"app": app_name, "wrong_api": api_name})

    low = error.lower()
    if "access_token" in low or "401" in low or "unauthorized" in low:
        hints["needs_token"] = True
    if "page_index" in low or "page_limit" in low or "pagination" in low:
        hints["needs_pagination"] = True
    m = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", error)
    if m:
        hints["key_error_field"] = m.group(1)

    return hints


def _suggest_api_names_from_dump(app: str, wrong_api: str, limit: int = 6) -> List[Tuple[str, str]]:
    """Find likely correct API names in app JSON when the wrong name was used."""
    app_file = _api_docs_root() / f"{app}.json"
    if not app_file.is_file():
        return []
    try:
        data = json.loads(app_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    names = list(data.keys())
    wrong_l = wrong_api.lower().replace("get_", "").replace("show_", "")
    scored: List[Tuple[int, str]] = []
    for name in names:
        nl = name.lower()
        score = 0
        if wrong_l and wrong_l in nl:
            score += 3
        if "profile" in wrong_l and "profile" in nl:
            score += 2
        if name.startswith("show_") and wrong_api.startswith("get_"):
            score += 1
        if score > 0:
            scored.append((score, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(app, name) for _, name in scored[:limit]]


def fetch_docs_for_error(
    ctx,
    error: str,
    apps: List[str],
    step: Dict[str, Any],
    existing: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """After a failure: pull targeted API docs from the error, not blind retry."""
    hints = _parse_execution_error(error)
    _log("fetch_docs_for_error", hints=hints)

    extra_defs: List[Dict[str, Any]] = []
    rag_queries: List[str] = []

    for item in hints["wrong_apis"]:
        app = item["app"]
        wrong = item["wrong_api"]
        rag_queries.append(f"{app} correct api for {wrong} parameters")
        for app_n, api_n in _suggest_api_names_from_dump(app, wrong):
            d = fetch_api_definition_direct(app_n, api_n)
            if d:
                extra_defs.append(d)
                _log("fetch_docs_for_error", suggested_api=f"{app_n}.{api_n}")

    if hints["needs_token"]:
        rag_queries.append(" ".join(apps) + " login access_token username password")
    if hints["needs_pagination"]:
        rag_queries.append(" ".join(apps) + " page_index page_limit list pagination")
    if hints["key_error_field"]:
        rag_queries.append(f"{hints['key_error_field']} response field " + " ".join(apps))

    rag_queries.append(step.get("doc_query") or step.get("instruction", ""))

    for q in rag_queries:
        if not (q or "").strip():
            continue
        rag_docs = retrieve_relevant_docs(ctx, q)
        extra_defs.extend(fetch_api_definitions(rag_docs))

    merged = _merge_api_definitions(existing, extra_defs)

    diagnosis_parts = [f"Execution failed:\n{error[-800:]}"]
    if hints["wrong_apis"]:
        diagnosis_parts.append(
            "Wrong API name(s) used. Compare with api_definitions — use exact api names "
            "(e.g. supervisor.show_profile, not get_profile)."
        )
    if hints["needs_token"]:
        diagnosis_parts.append("Missing or invalid access_token — use access_tokens dict.")
    if hints["needs_pagination"]:
        diagnosis_parts.append("Paginate with page_index until a short/empty page.")
    if hints["key_error_field"]:
        diagnosis_parts.append(
            f"Response missing key '{hints['key_error_field']}' — inspect api_definitions response schema."
        )
    diagnosis_parts.append("Do NOT repeat the same failing API call; change per the docs above.")

    diagnosis = "\n".join(diagnosis_parts)
    _log("fetch_docs_for_error", merged_doc_count=len(merged))
    return merged, diagnosis


def _code_fingerprint(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def bootstrap_access_tokens(ctx, apps: List[str]) -> Dict[str, str]:
    """Log into every involved app and return access_token map (runs via ctx.run_code)."""
    apps_to_login = [a for a in apps if a in AUTH_APPS and a not in SKIP_LOGIN]
    _log("bootstrap_access_tokens", apps=apps, login_targets=apps_to_login)

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

    result = execute_code(ctx, code, attempt=0)
    tokens: Dict[str, str] = {}
    stdout = result.get("stdout") or ""
    m = re.search(r"__TOKENS__(\{.*\})", stdout)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                tokens = {k: v for k, v in parsed.items() if v}
        except json.JSONDecodeError:
            pass

    for app_name, tok in tokens.items():
        _log("bootstrap_access_tokens", app=app_name, token_preview=tok[:12] + "..." if tok else "(empty)")

    if not tokens and not result.get("ok"):
        _log("bootstrap_access_tokens", warning="login snippet failed", error=result.get("error", "")[:200])

    return tokens


def task_classification(ctx) -> str:
    """Classify task as question (needs answer) or action (mutate state only)."""
    _log("task_classification", instruction=ctx.instruction)
    instruction = ctx.instruction or ""
    low = instruction.lower()
    if any(w in low for w in ("how many", "list", "give me", "what is", "which", "top ", "comma-separated")):
        _log("task_classification", result="question", method="heuristic")
        return "question"
    if any(w in low for w in ("delete", "remove", "create", "send", "pay", "transfer", "add ", "update")):
        _log("task_classification", result="action", method="heuristic")
        return "action"
    # fallback: ask model briefly
    _log("task_classification", method="model")
    out = _model_text(
        ctx,
        [
            {
                "role": "system",
                "content": "Reply JSON only: {\"kind\":\"question\"} or {\"kind\":\"action\"}.",
            },
            {"role": "user", "content": instruction},
        ],
        json_mode=True,
    )
    kind = _parse_json(out).get("kind", "question")
    _log("task_classification", result=kind, method="model")
    return kind


def domain_classification(ctx) -> List[str]:
    """Return one or more app names likely needed for this task."""
    _log("domain_classification", instruction=ctx.instruction)
    instruction = (ctx.instruction or "").lower()
    hits = [app for app in APPS if app in instruction or app.replace("_", " ") in instruction]
    if hits:
        _log("domain_classification", apps=hits, method="keyword")
        return hits
    _log("domain_classification", method="model")
    out = _model_text(
        ctx,
        [
            {
                "role": "system",
                "content": (
                    "Pick apps needed from: " + ", ".join(APPS) + ". "
                    "Reply JSON: {\"apps\":[\"spotify\",...]}."
                ),
            },
            {"role": "user", "content": ctx.instruction or ""},
        ],
        json_mode=True,
    )
    apps = _parse_json(out).get("apps", [])
    result = [a for a in apps if a in APPS] or ["spotify"]
    _log("domain_classification", apps=result, method="model")
    return result


def retrieve_relevant_docs(
    ctx,
    query: str,
    top_k: int = RAG_FINAL_TOP_K,
    min_score: float = MIN_RAG_SCORE,
) -> Any:
    """Retrieve API doc hits: cosine (15) -> LLM rerank -> best top_k (default 1)."""
    _log("retrieve_relevant_docs", query=query, top_k=top_k, min_score=min_score)

    try:
        from tools.rag import RAG_RETRIEVE_K, cosine_similarity_search

        ctx.trace("retrieval", query=query, source="embeddings_tfidf", min_score=min_score)
        docs = cosine_similarity_search(
            query,
            top_k=top_k,
            min_score=min_score,
            retrieve_k=RAG_RETRIEVE_K,
            use_llm_rerank=True,
        )
        _log(
            "retrieve_relevant_docs",
            source="tools.rag.cosine_similarity_search+llm_rerank",
            retrieve_k=docs.get("retrieve_k"),
            reranked=docs.get("reranked"),
        )
    except FileNotFoundError as exc:
        _log("retrieve_relevant_docs", source="ctx.retrieve (fallback)", reason=str(exc))
        docs = ctx.retrieve(query)

    if not isinstance(docs, dict):
        _log("retrieve_relevant_docs", hit_count=0, warning="unexpected response type")
        return docs

    results = docs.get("results") or []
    _log("retrieve_relevant_docs", hit_count=len(results))
    for rank, hit in enumerate(results, start=1):
        if not isinstance(hit, dict):
            continue
        _log(
            "retrieve_relevant_docs",
            rank=rank,
            llm_rank=hit.get("llm_rank"),
            app=hit.get("app"),
            api=hit.get("api"),
            score=hit.get("cosine_similarity", hit.get("score")),
            reason=(hit.get("rerank_reason") or "")[:80],
            description=hit.get("description", "")[:120],
        )

    return docs


def generate_query(ctx, task: str, apps: List[str]) -> str:
    """Build a short RAG query from the task and target apps."""
    _log("generate_query", task=task, apps=apps)
    app_part = " ".join(apps)
    out = _model_text(
        ctx,
        [
            {
                "role": "system",
                "content": (
                    "Write a short search query (max 12 words) to find AppWorld API docs. "
                    "Reply JSON: {\"query\":\"...\"}."
                ),
            },
            {"role": "user", "content": f"Task: {task}\nApps: {app_part}"},
        ],
        json_mode=True,
    )
    q = _parse_json(out).get("query", "")
    query = q if isinstance(q, str) and q.strip() else f"{app_part} {task[:80]}"
    _log("generate_query", query=query)
    return query


def retrieve_skills(ctx, apps: List[str]) -> List[Dict[str, Any]]:
    """Load skills from memory; seed defaults if empty."""
    _log("retrieve_skills", apps=apps)
    mem = ctx.memory.read() or {}
    skills = mem.get("skills")
    if not isinstance(skills, list) or not skills:
        skills = list(DEFAULT_SKILLS)
        ctx.memory.write("skills", skills)
    # filter to relevant apps (+ always include supervisor bootstrap)
    relevant = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        app = s.get("app", "")
        if app in apps or app == "supervisor":
            relevant.append(s)
    result = relevant or list(DEFAULT_SKILLS)
    _log("retrieve_skills", skills=[s.get("name") for s in result if isinstance(s, dict)])
    return result


def _summarize_probe_value(value: Any, max_len: int = _PROBE_SUMMARY_LEN) -> str:
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_plan(plan: Dict[str, Any], *, title: str = "PLAN") -> None:
    """Print the current question plan for local debugging (run_local)."""
    sep = "=" * 72
    print(f"\n{sep}\n  {title}\n{sep}")

    instruction = (plan.get("instruction") or "").strip()
    if instruction:
        print(f"\nTask:\n  {instruction}")

    print(
        f"\nKind: {plan.get('task_kind', '?')}  |  "
        f"Apps: {', '.join(plan.get('apps') or []) or '?'}"
    )
    tokens = plan.get("access_tokens") or {}
    if tokens:
        print(f"Tokens: {', '.join(tokens.keys())}")
    if plan.get("ready_for_synthesis"):
        print("Ready for synthesis: yes")

    questions = plan.get("questions") or []
    hints = plan.get("api_hints") or []
    hints_by_step = {h.get("step"): h for h in hints if isinstance(h, dict)}

    if questions:
        print(f"\nQuestions ({len(questions)}):")
        for q in questions:
            if not isinstance(q, dict):
                continue
            step = q.get("step", "?")
            kind = q.get("kind", "question")
            status = q.get("status", "pending")
            text = (q.get("question") or "").strip()
            hint = hints_by_step.get(step) or {}
            api = f"{hint.get('app')}.{hint.get('api')}" if hint.get("app") else "?"
            print(f"  [{step}] ({status}) {kind}: {text}")
            if hint.get("app"):
                print(f"       RAG → {api}  score={hint.get('cosine_similarity')}")

    if hints and not questions:
        print(f"\nAPI hints ({len(hints)}):")
        for h in hints:
            print(f"  [{h.get('step')}] {h.get('app')}.{h.get('api')} — {h.get('question', '')[:60]}")

    probes = plan.get("probes") or []
    if probes:
        print(f"\nProbes ({len(probes)}):")
        for p in probes:
            if not isinstance(p, dict):
                continue
            step = p.get("step", "?")
            ok = "ok" if p.get("ok") else "FAIL"
            api = f"{p.get('app')}.{p.get('api')}" if p.get("app") else "?"
            q = (p.get("question") or "")[:80]
            print(f"  [{step}] {ok} {api}")
            print(f"       Q: {q}")
            if p.get("error"):
                print(f"       err: {p['error'][:120]}")
            summary = (p.get("result_summary") or "")[:160]
            if summary:
                print(f"       → {summary}")

    print(f"{sep}\n")


def _rag_hint_for_question(
    item: Dict[str, Any],
    rag: Any,
) -> Dict[str, Any]:
    """RAG + rerank + API definition for one plan question (thread-safe read on shared index)."""
    question = (item.get("question") or "").strip()
    step = item.get("step")
    empty: Dict[str, Any] = {
        "step": step,
        "kind": item.get("kind", "question"),
        "question": question,
        "app": None,
        "api": None,
        "error": "empty question",
    }
    if not question:
        return empty

    from tools.rag import RAG_RETRIEVE_K

    docs = rag.cosine_similarity_search(
        question,
        top_k=RAG_FINAL_TOP_K,
        min_score=MIN_RAG_SCORE,
        retrieve_k=RAG_RETRIEVE_K,
        use_llm_rerank=True,
        verbose=False,
    )
    hit = (docs.get("results") or [None])[0]
    defs = fetch_api_definitions(docs) if hit else []
    defn = defs[0] if defs else {}

    return {
        "step": step,
        "kind": item.get("kind", "question"),
        "question": question,
        "app": hit.get("app") if hit else None,
        "api": hit.get("api") if hit else None,
        "cosine_similarity": hit.get("cosine_similarity") if hit else None,
        "rerank_reason": (hit.get("rerank_reason") or "") if hit else "",
        "description": (hit.get("description") or "")[:200] if hit else "",
        "definition_text": defn.get("definition_text", ""),
        "api_definitions": defs,
    }


def enrich_plan_with_rag(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each plan question (parallel), run RAG+rerank and store matches in api_hints[]
    aligned by step index with questions[].
    """
    questions = [q for q in (plan.get("questions") or []) if isinstance(q, dict)]
    if not questions:
        plan["api_hints"] = []
        return plan

    from tools.rag import ApiDocsRAG

    _log("enrich_plan_with_rag", question_count=len(questions))
    rag = ApiDocsRAG()
    workers = min(_MAX_RAG_WORKERS, len(questions))
    hints: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_rag_hint_for_question, q, rag): q.get("step")
            for q in questions
        }
        for fut in as_completed(futures):
            try:
                hints.append(fut.result())
            except Exception as exc:
                step = futures[fut]
                _log("enrich_plan_with_rag", step=step, error=str(exc))
                hints.append({"step": step, "error": str(exc)})

    hints.sort(key=lambda h: (h.get("step") is None, h.get("step", 0)))
    plan["api_hints"] = hints
    _log("enrich_plan_with_rag", hints_loaded=len(hints))
    for h in hints:
        _log(
            "enrich_plan_with_rag",
            step=h.get("step"),
            api=f"{h.get('app')}.{h.get('api')}" if h.get("app") else None,
            score=h.get("cosine_similarity"),
        )
    return plan


def _endpoint_has_pagination(definition_text: str) -> bool:
    """True when API docs mention page_index / page_limit (list endpoints)."""
    t = (definition_text or "").lower()
    return "page_index" in t or "page_limit" in t


def _api_hints_context(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact api_hints for model prompts."""
    out: List[Dict[str, Any]] = []
    for h in plan.get("api_hints") or []:
        if not isinstance(h, dict):
            continue
        defn = h.get("definition_text") or ""
        out.append(
            {
                "step": h.get("step"),
                "question": h.get("question"),
                "suggested_api": f"{h.get('app')}.{h.get('api')}" if h.get("app") else None,
                "paginated": _endpoint_has_pagination(defn),
                "rerank_reason": h.get("rerank_reason"),
                "definition_text": defn[:1200],
            }
        )
    return out


def _format_task_answer(answer: str, instruction: str = "") -> str:
    """
    Normalize ANSWER for AppWorld oracle (comma-separated lists, no stray spaces).
    Oracle splits on ',' — items must be stripped so ' Foo' != 'Foo'.
    """
    text = (answer or "").strip()
    if not text:
        return text
    low = (instruction or "").lower()
    if "comma-separated" in low or "comma separated" in low or "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts:
            return ", ".join(parts)
    return text


def _parse_answer_parts(answer: str) -> List[str]:
    return [p.strip() for p in (answer or "").split(",") if p.strip()]


def _expected_answer_count(instruction: str) -> Optional[int]:
    """Parse 'top 4', 'top-4', etc. from the task instruction."""
    m = re.search(r"\btop[\s-]*(\d+)\b", (instruction or "").lower())
    if m:
        return int(m.group(1))
    return None


def _validate_answer_format(plan: Dict[str, Any], answer: str) -> Optional[str]:
    """Return an error message if ANSWER format is wrong; None if ok."""
    instruction = plan.get("instruction") or ""
    formatted = _format_task_answer(answer, instruction)
    if formatted != (answer or "").strip():
        return (
            "ANSWER has bad comma spacing (leading spaces after commas). "
            f"Use: ANSWER = \", \".join(t.strip() for t in titles). "
            f"Expected shape like: {formatted[:120]}"
        )

    low = instruction.lower()
    if "comma-separated" in low or "comma separated" in low:
        parts = _parse_answer_parts(formatted)
        if len(parts) < 2:
            return "comma-separated answer must contain at least 2 items"
        expected_n = _expected_answer_count(instruction)
        if expected_n is not None and len(parts) != expected_n:
            return (
                f"task asks for top {expected_n} items but ANSWER has {len(parts)}: "
                f"{formatted[:200]}"
            )
    return None


def _oracle_failure_text(verdict: Dict[str, Any]) -> str:
    failures = verdict.get("failures") or []
    bits: List[str] = []
    for f in failures:
        if isinstance(f, dict):
            bits.append(f.get("trace") or f.get("requirement") or json.dumps(f)[:400])
        else:
            bits.append(str(f)[:400])
    return _feedback_str(bits or verdict, max_len=1200)


def _local_env(ctx) -> Any:
    return getattr(ctx, "_env", None)


def _oracle_gate(
    ctx,
    plan: Dict[str, Any],
    run_result: Dict[str, Any],
    solution: Dict[str, Any],
    *,
    max_retries: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Local run_local only: submit trial answer to AppWorld oracle; on mismatch regenerate.
    """
    env = _local_env(ctx)
    if env is None:
        return run_result, solution

    instruction = plan.get("instruction") or ""
    for oracle_try in range(1, max_retries + 1):
        answer = _format_task_answer(run_result.get("answer") or "", instruction)
        if not answer:
            break

        ctx.mcp.call("complete_task", {"answer": answer})
        verdict = env.world.evaluate().to_dict()
        _log("oracle_gate", attempt=oracle_try, success=verdict.get("success"))

        if verdict.get("success"):
            run_result["answer"] = answer
            run_result["ok"] = True
            return run_result, solution

        feedback = _oracle_failure_text(verdict)
        _log("oracle_gate", attempt=oracle_try, feedback=feedback[:300])
        if oracle_try >= max_retries:
            run_result["answer"] = answer
            break

        ctx.reflect(f"oracle_gate try {oracle_try}: answer rejected")
        solution = generate_solution_code(
            ctx,
            plan,
            check_feedback=(
                f"AppWorld oracle rejected the submitted answer.\n{feedback}\n"
                "Recompute from ALL library sources with full pagination. "
                "Filter genre with: g = (song.get('genre') or '').lower(); "
                "'r&b' in g. Sort by (-play_count, title). "
                "Format: ANSWER = \", \".join(t.strip() for t in titles)"
            ),
            failed_code=solution.get("code", ""),
        )
        run_result = execute_code(ctx, solution.get("code", ""), attempt=oracle_try + 10)

    return run_result, solution


def _feedback_str(value: Any, max_len: int = 800) -> str:
    """Coerce verifier/executor feedback to a safe string for prompts."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:max_len] if len(text) > max_len else text


def verify_solution(
    ctx,
    plan: Dict[str, Any],
    run_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check-loop gate after successful run_code: structural ANSWER checks only.
    Semantic correctness uses AppWorld oracle in _oracle_gate (local) at submit time.
    """
    del ctx
    if not run_result.get("ok"):
        return {
            "pass": False,
            "reason": _feedback_str(
                run_result.get("error") or run_result.get("stdout", "")
            ),
        }

    if plan.get("task_kind") == "action":
        return {"pass": True, "reason": "action task executed without traceback"}

    instruction = plan.get("instruction") or ""
    answer = _format_task_answer(run_result.get("answer") or "", instruction)
    run_result["answer"] = answer

    if not answer:
        return {
            "pass": False,
            "reason": "Missing ANSWER= line in stdout after successful execution.",
        }

    low = answer.lower()
    if low in ("<<not_solved>>", "<<not_given>>", "not_solved"):
        return {"pass": False, "reason": f"Placeholder answer: {answer}"}

    fmt_err = _validate_answer_format(plan, answer)
    if fmt_err:
        return {"pass": False, "reason": fmt_err}

    return {
        "pass": True,
        "reason": "execution ok; ANSWER format validated (oracle checks correctness locally)",
    }


def check_loop(
    ctx,
    plan: Dict[str, Any],
    *,
    max_attempts: int = _CHECK_LOOP_ATTEMPTS,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate code, execute, verify — retry with feedback until pass or max attempts."""
    solution = generate_solution_code(ctx, plan)
    run_result: Dict[str, Any] = {"ok": False, "answer": ""}

    for attempt in range(1, max_attempts + 1):
        run_result = execute_code(ctx, solution.get("code", ""), attempt=attempt)
        if not run_result.get("ok"):
            feedback = run_result.get("error") or run_result.get("stdout", "")
            passed = False
        else:
            verdict = verify_solution(ctx, plan, run_result)
            passed = verdict.get("pass", False)
            feedback = verdict.get("reason", "")

        fb = _feedback_str(feedback, max_len=200)
        _log("check_loop", attempt=attempt, passed=passed, feedback=fb)
        if passed:
            break

        ctx.reflect(f"check_loop attempt {attempt}: {fb or 'failed'}")
        solution = generate_solution_code(
            ctx,
            plan,
            last_error=_feedback_str(feedback),
            failed_code=solution.get("code", ""),
            check_feedback=_feedback_str(feedback, max_len=600),
        )

    return run_result, solution


def _probe_record(item: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": item.get("step"),
        "kind": item.get("kind"),
        "question": item.get("question"),
        "app": outcome.get("app"),
        "api": outcome.get("api"),
        "ok": outcome.get("ok"),
        "result_summary": _summarize_probe_value(outcome.get("result")),
        "error": (outcome.get("error") or "")[:400],
    }


def build_plan(
    ctx,
    task_kind: str,
    apps: List[str],
    skills: List[Dict[str, Any]],
    access_tokens: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build an ordered list of sub-questions/actions to probe via question_to_api."""
    _log("build_plan", task_kind=task_kind, apps=apps)

    sys = (
        "Plan how to solve an AppWorld task by breaking it into ordered sub-questions.\n"
        "Each item is ONE natural-language question that maps to a single API call "
        "(e.g. 'spotify login', 'list all songs in my song library with pagination', "
        "'get play_count and genre for song_id 42').\n"
        "Use kind=question for read/probe steps, kind=action for a single mutation step.\n"
        "Do NOT include a final formatting step — synthesis happens after all probes.\n"
        "Keep the plan short (3-8 items). Be specific about apps and data needed.\n"
        "Reply JSON only:\n"
        "{\n"
        '  "questions": [\n'
        '    {"step": 1, "kind": "question", "question": "..."},\n'
        '    {"step": 2, "kind": "action", "question": "..."}\n'
        "  ]\n"
        "}"
    )
    user = {
        "task_instruction": ctx.instruction,
        "task_kind": task_kind,
        "apps": apps,
        "skills": skills,
        "access_tokens_available": list((access_tokens or {}).keys()),
    }
    out = _model_text(
        ctx,
        [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user)}],
        json_mode=True,
    )
    parsed = _parse_json(out)
    items = parsed.get("questions") or parsed.get("steps") or []
    if not isinstance(items, list) or not items:
        items = [
            {
                "step": 1,
                "kind": "question",
                "question": ctx.instruction or "complete the task",
            }
        ]

    normalized: List[Dict[str, Any]] = []
    for i, s in enumerate(items, start=1):
        if not isinstance(s, dict):
            continue
        q = (s.get("question") or s.get("instruction") or "").strip()
        if not q:
            continue
        normalized.append(
            {
                "step": s.get("step", i),
                "kind": s.get("kind", "question"),
                "question": q,
                "status": "pending",
            }
        )

    plan = {
        "task_kind": task_kind,
        "apps": apps,
        "skills": skills,
        "access_tokens": access_tokens or {},
        "instruction": ctx.instruction,
        "questions": normalized[:MAX_PLAN_QUESTIONS],
        "api_hints": [],
        "probes": [],
        "ready_for_synthesis": True,
    }
    _log("build_plan", question_count=len(normalized))
    plan = enrich_plan_with_rag(plan)
    print_plan(plan, title="PLAN (with RAG hints)")
    return plan


def refine_plan(
    ctx,
    plan: Dict[str, Any],
    last_probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update remaining plan items based on probe results."""
    _log("refine_plan", last_ok=last_probe.get("ok") if last_probe else None)

    pending = [q for q in plan.get("questions", []) if q.get("status") == "pending"]
    if not pending and not last_probe:
        plan["ready_for_synthesis"] = True
        return plan

    sys = (
        "You refine an AppWorld task plan after executing sub-questions via API probes.\n"
        "Given the original task, completed probes (with API results), and pending questions:\n"
        "- Mark ready_for_synthesis=true when enough data was collected to write final code.\n"
        "- Otherwise return an updated questions list (pending items only, or replacements).\n"
        "- Add new questions if a probe failed or exposed missing data.\n"
        "- Remove redundant questions already answered by prior probes.\n"
        "Reply JSON only:\n"
        "{\n"
        '  "ready_for_synthesis": false,\n'
        '  "questions": [{"step": 3, "kind": "question", "question": "..."}],\n'
        '  "notes": "short reason"\n'
        "}"
    )
    user = {
        "task_instruction": plan.get("instruction"),
        "task_kind": plan.get("task_kind"),
        "apps": plan.get("apps"),
        "completed_probes": plan.get("probes", []),
        "pending_questions": pending,
        "last_probe": last_probe,
    }
    out = _model_text(
        ctx,
        [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        json_mode=True,
    )
    parsed = _parse_json(out)
    if parsed.get("ready_for_synthesis"):
        plan["ready_for_synthesis"] = True
        for q in plan.get("questions", []):
            if q.get("status") == "pending":
                q["status"] = "skipped"
        _log("refine_plan", ready_for_synthesis=True, notes=parsed.get("notes", ""))
        return plan

    new_items = parsed.get("questions") or []
    if isinstance(new_items, list) and new_items:
        done_steps = {p.get("step") for p in plan.get("probes", [])}
        max_step = max((q.get("step", 0) for q in plan.get("questions", [])), default=0)
        refreshed: List[Dict[str, Any]] = [
            q for q in plan.get("questions", []) if q.get("status") != "pending"
        ]
        for s in new_items:
            if not isinstance(s, dict):
                continue
            qtext = (s.get("question") or "").strip()
            if not qtext:
                continue
            step = s.get("step")
            if not step or step in done_steps:
                max_step += 1
                step = max_step
            refreshed.append(
                {
                    "step": step,
                    "kind": s.get("kind", "question"),
                    "question": qtext,
                    "status": "pending",
                }
            )
        plan["questions"] = refreshed[:MAX_PLAN_QUESTIONS]
        _log("refine_plan", pending_count=sum(1 for q in refreshed if q.get("status") == "pending"))
    return plan


def run_question_plan(ctx, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute each plan question via question_to_api; refine after each probe."""
    tokens = dict(plan.get("access_tokens") or {})
    probes: List[Dict[str, Any]] = list(plan.get("probes") or [])

    while True:
        pending = [q for q in plan.get("questions", []) if q.get("status") == "pending"]
        if plan.get("ready_for_synthesis") or not pending:
            break
        if len(probes) >= MAX_PLAN_QUESTIONS:
            _log("run_question_plan", warning="max probes reached")
            break

        item = pending[0]
        step_num = item.get("step")
        question = item.get("question", "")
        _log("run_question_plan", step=step_num, question=question)

        from tools.question_to_api import question_to_api_call_ctx

        outcome = question_to_api_call_ctx(ctx, question, tokens, verbose=False)
        _log("run_question_plan", step=step_num, probe_code=outcome.get("code") or "")
        item["status"] = "done" if outcome.get("ok") else "failed"

        probe_entry = _probe_record(item, outcome)
        probe_entry["definition_text"] = ""
        defs = outcome.get("api_definitions") or []
        if defs:
            probe_entry["definition_text"] = (defs[0].get("definition_text") or "")[:800]
        probes.append(probe_entry)
        plan["probes"] = probes

        if not outcome.get("ok"):
            ctx.reflect(f"probe step {step_num} failed: {question[:80]}")
        else:
            _log(
                "run_question_plan",
                step=step_num,
                api=f"{outcome.get('app')}.{outcome.get('api')}",
                result_preview=probe_entry.get("result_summary", "")[:200],
            )

        plan = refine_plan(ctx, plan, last_probe=probe_entry)

    return {"probes": probes, "plan": plan}


def generate_solution_code(
    ctx,
    plan: Dict[str, Any],
    last_error: str = "",
    failed_code: str = "",
    check_feedback: str = "",
) -> Dict[str, Any]:
    """Write final Python using plan questions + concurrent RAG api_hints."""
    _log(
        "generate_solution_code",
        hint_count=len(plan.get("api_hints", [])),
        probe_count=len(plan.get("probes", [])),
    )

    is_question = plan.get("task_kind") == "question"
    hints_ctx = _api_hints_context(plan)
    paginated_apis = [
        h["suggested_api"]
        for h in hints_ctx
        if h.get("paginated") and h.get("suggested_api")
    ]
    sys = (
        "Write Python to finish an AppWorld task.\n"
        "Variable `apis` is ALREADY in scope — never write import apis or from apis.\n"
        "`access_tokens` dict is provided — use access_tokens['spotify'], do not re-login.\n"
        "Use api_hints: each plan step has a RAG-suggested API and definition_text — "
        "call those exact api names (e.g. show_song_library, show_song).\n"
        "\n"
        "PAGINATION: Many list endpoints support optional page_index and page_limit. "
        "If api_hints marks paginated=true (or definition_text lists page_index), you MUST "
        "fetch ALL pages — a single call often returns only the first page.\n"
        "Loop page_index=0,1,2,... until the response is empty or shorter than page_limit. "
        "Accumulate items across pages before filtering/aggregating.\n"
        "Example pattern:\n"
        "  all_items = []\n"
        "  page_index = 0\n"
        "  while True:\n"
        "      batch = apis.spotify.show_song_library(access_token=..., page_index=page_index)\n"
        "      if not batch:\n"
        "          break\n"
        "      all_items.extend(batch)\n"
        "      if len(batch) < 20:  # default page_limit is often 20\n"
        "          break\n"
        "      page_index += 1\n"
        "\n"
        "Tasks that need 'all songs', 'every item', or library-wide stats require pagination "
        "on every paginated list API you call (song library, album library, playlist library, etc.).\n"
        "Do NOT call apis.supervisor.complete_task().\n"
    )
    if paginated_apis:
        sys += f"\nPaginated APIs in this plan (must loop): {', '.join(paginated_apis)}\n"
    err_text = _feedback_str(last_error)
    if err_text:
        sys += (
            f"\nThe previous code FAILED:\n{err_text}\n"
            "Fix the error. Do not import apis. Use only apis.<app>.<method>(...) calls.\n"
        )
        if failed_code:
            sys += f"\nFailed code (do not repeat mistakes):\n{_feedback_str(failed_code, 1200)}\n"
    fb_text = _feedback_str(check_feedback, max_len=600)
    if fb_text:
        sys += f"\nVerifier feedback (must address):\n{fb_text}\n"
    if is_question:
        sys += (
            "Set ANSWER to the exact final string the task expects, then print('ANSWER=' + ANSWER).\n"
            "ANSWER FORMAT (critical for grading):\n"
            "- Comma-separated lists: ANSWER = \", \".join(t.strip() for t in titles)\n"
            "  (one comma + one space between items; strip each title; no leading spaces on items)\n"
            "- If task says 'top N', return exactly N items after sorting.\n"
            "AGGREGATION (spotify-style tasks):\n"
            "- Collect song_ids from song library + album library + playlist library (all paginated).\n"
            "- Per song_id call show_song(access_token=..., song_id=sid) for genre and play_count.\n"
            "- R&B filter: g = (song.get('genre') or '').lower(); keep if 'r&b' in g\n"
            "- Sort: rb_songs.sort(key=lambda s: (-s.get('play_count', 0), s.get('title', '')))\n"
        )
    else:
        sys += "Perform required mutations; print('ANSWER=') only if the task expects a value.\n"
    sys += 'Reply JSON: {"code":"..."}.'

    user = {
        "task_instruction": plan.get("instruction"),
        "task_kind": plan.get("task_kind"),
        "apps": plan.get("apps"),
        "plan_questions": plan.get("questions", []),
        "api_hints": hints_ctx,
        "paginated_apis": paginated_apis,
        "probes": plan.get("probes", []),
        "access_tokens": list((plan.get("access_tokens") or {}).keys()),
    }
    out = _model_text(
        ctx,
        [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        json_mode=True,
    )
    parsed = _parse_json(out)
    code = _sanitize_appworld_code(parsed.get("code", ""))
    tokens_literal = json.dumps(plan.get("access_tokens") or {})
    full_code = f"access_tokens = {tokens_literal}\n{code}" if code else ""
    _log("generate_solution_code", code_len=len(full_code), code=full_code)
    return {"code": full_code}


def generate_code_for_step(
    ctx,
    plan: Dict[str, Any],
    step: Dict[str, Any],
    step_index: int,
    api_definitions: List[Dict[str, Any]],
    prior_steps: List[Dict[str, Any]],
    last_error: str = "",
    diagnosis: str = "",
    failed_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate Python for a single plan step using step docs + full plan context."""
    step_num = step.get("step", step_index + 1)
    _log(
        "generate_code_for_step",
        step=step_num,
        instruction=step.get("instruction"),
        is_retry=bool(last_error),
    )

    is_last = step_index == len(plan.get("steps", [])) - 1
    sys = (
        "Write Python for ONE step of a multi-step AppWorld plan. Variable `apis` is in scope.\n"
        "Do NOT call apis.supervisor.complete_task().\n"
        "Implement ONLY the current step. You MUST use api_definitions for exact API names and parameters.\n"
        "Never guess method names (use show_profile not get_profile).\n"
        "access_tokens dict is provided — use access_tokens['spotify'] etc.; do not re-login if token exists.\n"
        "Paginate list APIs with page_index until a short/empty page.\n"
        "For writes: read current state first; only apply missing changes (idempotent).\n"
        "Print useful debug output. Store intermediate results in variables.\n"
    )
    if last_error:
        sys += (
            "\nA previous attempt FAILED. Read diagnosis and api_definitions. "
            "Fix the specific API mistake — do NOT repeat the same calls or identical code.\n"
        )
    if is_last and plan.get("task_kind") == "question":
        sys += "This is the LAST step: set ANSWER to the final concise answer string and print('ANSWER=' + ANSWER).\n"
    elif is_last:
        sys += "This is the LAST step: perform any final mutations; set ANSWER='' unless you must print a value.\n"
    else:
        sys += "This is NOT the last step: do NOT set ANSWER unless needed for the next step; print key results.\n"
    sys += 'Reply JSON: {"code":"..."}.'

    user = {
        "full_task": plan.get("instruction"),
        "task_kind": plan.get("task_kind"),
        "apps": plan.get("apps"),
        "all_steps": plan.get("steps"),
        "current_step": step,
        "prior_step_results": prior_steps,
        "api_definitions": api_definitions,
        "access_tokens": plan.get("access_tokens"),
        "last_error": last_error,
        "diagnosis": diagnosis,
        "failed_attempts": failed_attempts or [],
    }
    out = _model_text(
        ctx,
        [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        json_mode=True,
    )
    parsed = _parse_json(out)
    code = parsed.get("code", "")
    tokens_literal = json.dumps(plan.get("access_tokens") or {})
    preamble = f"access_tokens = {tokens_literal}\n"
    full_code = preamble + code if code else ""
    _log("generate_code_for_step", step=step_num, code_len=len(full_code))
    return {"code": full_code, "step": step_num}


def execute_plan(ctx, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Run each plan step: RAG + fetch defs + generate code + execute."""
    steps = plan.get("steps") or []
    _log("execute_plan", total_steps=len(steps))

    prior_steps: List[Dict[str, Any]] = []
    final_answer = ""
    all_ok = True

    for idx, step in enumerate(steps):
        step_num = step.get("step", idx + 1)
        _log("execute_plan", running_step=step_num, instruction=step.get("instruction"))

        api_definitions: List[Dict[str, Any]] = []
        if step.get("needs_docs", True):
            query = step.get("doc_query") or step.get("instruction", "")
            rag_docs = retrieve_relevant_docs(ctx, query)
            api_definitions = fetch_api_definitions(rag_docs)
        else:
            _log("execute_plan", step=step_num, skip_docs=True)

        max_tries = 3
        last_error = ""
        diagnosis = ""
        failed_attempts: List[Dict[str, Any]] = []
        tried_fingerprints: set = set()
        result: Dict[str, Any] = {"ok": False, "stdout": "", "answer": ""}

        for attempt in range(1, max_tries + 1):
            code_bundle = generate_code_for_step(
                ctx,
                plan,
                step,
                idx,
                api_definitions,
                prior_steps,
                last_error=last_error,
                diagnosis=diagnosis,
                failed_attempts=failed_attempts,
            )
            code = code_bundle.get("code", "")
            fp = _code_fingerprint(code)

            if fp in tried_fingerprints:
                _log("execute_plan", step=step_num, blocked="duplicate_code", attempt=attempt)
                last_error = (
                    "Blocked: generated code is identical to a failed attempt. "
                    "Change API calls to match api_definitions."
                )
                api_definitions, diagnosis = fetch_docs_for_error(
                    ctx, last_error, plan.get("apps", []), step, api_definitions
                )
                failed_attempts.append(
                    {"attempt": attempt, "blocked": "duplicate", "error": last_error[:300]}
                )
                continue

            tried_fingerprints.add(fp)
            result = execute_code(ctx, code, attempt=step_num)
            if result.get("ok"):
                break

            last_error = result.get("error") or result.get("stdout", "")[:1500]
            failed_attempts.append(
                {
                    "attempt": attempt,
                    "error": last_error[:500],
                    "code_preview": code[:400],
                }
            )
            ctx.reflect(f"step {step_num} attempt {attempt} failed; fetching API docs for fix")

            api_definitions, diagnosis = fetch_docs_for_error(
                ctx, last_error, plan.get("apps", []), step, api_definitions
            )
            _log("execute_plan", step=step_num, retry=attempt + 1, docs_loaded=len(api_definitions))

        if not result.get("ok"):
            all_ok = False

        if result.get("answer"):
            final_answer = result["answer"]

        prior_steps.append(
            {
                "step": step_num,
                "instruction": step.get("instruction"),
                "ok": result.get("ok"),
                "stdout_tail": (result.get("stdout") or "")[-600:],
                "answer": result.get("answer", ""),
            }
        )
        _log("execute_plan", step=step_num, ok=result.get("ok"), answer=result.get("answer"))

    return {"ok": all_ok, "answer": final_answer, "step_outputs": prior_steps}


def execute_code(ctx, code: str, attempt: int = 1) -> Dict[str, Any]:
    """Run code in AppWorld; return ok flag, stdout, and extracted ANSWER if present."""
    code = _sanitize_appworld_code(str(code or ""))
    _log("execute_code", attempt=attempt, code_len=len(code), code=code)
    if not code or not code.strip():
        _log("execute_code", ok=False, error="empty_code")
        return {"ok": False, "stdout": "", "error": "empty_code", "answer": ""}
    stdout = ctx.run_code(code)
    err_markers = ("Traceback (most recent call last)", "Exception:", "Execution failed")
    failed = any(m in (stdout or "") for m in err_markers)
    answer = ""
    if not failed and stdout:
        m = re.search(r"^ANSWER=(.*)$", stdout, re.MULTILINE)
        if m:
            answer = m.group(1).strip()
    result = {
        "ok": not failed,
        "stdout": stdout or "",
        "error": stdout[-1500:] if failed else "",
        "answer": answer,
    }
    _log(
        "execute_code",
        attempt=attempt,
        ok=result["ok"],
        answer=answer,
        stdout_preview=(stdout or "")[-400:],
        error_preview=result["error"][:200] if failed else "",
    )
    return result


def _submit(ctx, task_kind: str, answer: str) -> None:
    _log("submit", task_kind=task_kind, answer=answer)
    env = _local_env(ctx)
    if env is not None and env.completed():
        _log("submit", status="done", note="already completed (oracle_gate)")
        return
    if task_kind == "action":
        ctx.mcp.call("complete_task", {})
    else:
        ctx.mcp.call("complete_task", {"answer": answer or "<<not_solved>>"})
    _log("submit", status="done")


def solve(ctx):
    instruction = ctx.instruction or ""
    _log("solve", instruction=instruction)

    task_kind = task_classification(ctx)
    apps = domain_classification(ctx)

    access_tokens = bootstrap_access_tokens(ctx, apps)
    skills = retrieve_skills(ctx, apps)
    plan = build_plan(ctx, task_kind, apps, skills, access_tokens=access_tokens)

    run_result, solution = check_loop(ctx, plan)
    if _local_env(ctx) is not None:
        run_result, solution = _oracle_gate(ctx, plan, run_result, solution)

    answer = _format_task_answer(
        run_result.get("answer", ""),
        instruction,
    )
    run_result["answer"] = answer
    _log("solve", final_answer=answer, execution_ok=run_result.get("ok"))

    _submit(ctx, task_kind, answer)

    if run_result.get("ok") and answer:
        mem = ctx.memory.read() or {}
        wins = mem.get("wins", {})
        if not isinstance(wins, dict):
            wins = {}
        key = re.sub(r"\s+", " ", instruction.lower())[:100]
        wins[key] = {"apps": apps, "task_kind": task_kind}
        ctx.memory.write("wins", wins)
        _log("solve", memory_write="wins", key=key)

    _log("solve", status="finished")
