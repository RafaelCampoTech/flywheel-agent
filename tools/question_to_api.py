"""Debug tool: natural-language question → API doc → token → execute → print.

  set -a; source .env; set +a
  python tools/question_to_api.py "show my spotify song library page 0"
  python tools/question_to_api.py "get top 4 r&b songs by play count from spotify"

Pipeline:
  1. Keyword search over api_docs_dump → top candidates
  2. Load full endpoint definitions
  3. Bootstrap access token for the app
  4. LLM writes a Python snippet that calls the API
  5. Execute in AppWorld and print result
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent.step_executor import _api_docs_root, _render_api_entry, _search_dump  # noqa: E402
from flywheel.proxy import chat, content_of  # noqa: E402

AUTH_APPS = {
    "spotify", "amazon", "gmail", "phone", "venmo", "splitwise",
    "todoist", "simple_note", "file_system",
}

_CALL_SYS = """\
Write ONE Python snippet for AppWorld. Rules:
- `apis` is already in scope. Never import it.
- `access_tokens` is a dict mapping app_name → token string.
- Use EXACT api names and parameter names from the api_definition provided.
- Respect page_limit constraints (usually ≤ 20).
- Paginate list APIs when the question implies 'all' items.
- At the end: import json; print('__RESULT__' + json.dumps(result, default=str))
- Do NOT call complete_task.
Output raw Python only — no markdown fences.
"""


def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """Search all apps in api_docs_dump for relevant APIs."""
    root = _api_docs_root()
    q_words = set(query.lower().split())
    scored: list[tuple[int, str, str, dict]] = []
    for app_file in sorted(root.glob("*.json")):
        app = app_file.stem
        try:
            data = json.loads(app_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for api_name, entry in data.items():
            blob = f"{app} {api_name} {entry.get('description', '')}".lower()
            score = sum(1 for w in q_words if w in blob)
            if score:
                scored.append((score, app, api_name, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"app": a, "api": n, "entry": e, "score": s} for s, a, n, e in scored[:top_k]]


def _bootstrap_token(env, app: str) -> str:
    if app not in AUTH_APPS:
        return ""
    code = (
        "import json\n"
        "me = apis.supervisor.show_profile()\n"
        "pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}\n"
        f"app_name = {json.dumps(app)}\n"
        "username = me['phone_number'] if app_name == 'phone' else me['email']\n"
        "res = getattr(apis, app_name).login(username=username, password=pw[app_name])\n"
        "print('__TOKEN__' + json.dumps(res.get('access_token', '')))\n"
    )
    stdout = env.execute(code)
    m = re.search(r"__TOKEN__(.+)", stdout or "")
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return ""


def _generate_call(question: str, api_defs: list[dict], access_tokens: dict) -> str:
    key = os.environ.get("FLYWHEEL_KEY", "")
    url = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"
    if not key:
        print("[question_to_api] FLYWHEEL_KEY not set", file=sys.stderr)
        return ""

    user_content = json.dumps({
        "question": question,
        "api_definitions": [
            {"app": d["app"], "api": d["api"], "doc": _render_api_entry(d["app"], d["api"], d["entry"])}
            for d in api_defs
        ],
        "access_tokens": {k: v[:8] + "..." for k, v in access_tokens.items()},
    }, ensure_ascii=False)

    data, remaining = chat(url, key, [
        {"role": "system", "content": _CALL_SYS},
        {"role": "user", "content": user_content},
    ])
    if remaining is not None:
        print(f"[tokens remaining: {remaining}]")
    return content_of(data)


def _extract_result(stdout: str):
    if not stdout:
        return None
    m = re.search(r"__RESULT__(.+)", stdout, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(1).strip()
    return stdout.strip()


def run(question: str, task_id: str | None = None) -> dict:
    print(f"\n=== question_to_api ===\nQ: {question}\n")

    hits = _keyword_search(question)
    if not hits:
        print("No API matches found.")
        return {"error": "no matches"}

    print("--- API matches ---")
    for h in hits:
        print(f"  {h['app']}.{h['api']}  score={h['score']}  {h['entry'].get('description', '')[:80]}")

    print("\n--- top API definition ---")
    best = hits[0]
    print(_render_api_entry(best["app"], best["api"], best["entry"]))

    os.environ.setdefault("APPWORLD_ROOT", "./aw")
    from flywheel.appworld_env import AppWorldEnv, load_task_ids
    tid = task_id or load_task_ids("dev")[0]
    env = AppWorldEnv(tid, experiment_name="question_to_api")

    apps = list({h["app"] for h in hits if h["app"] in AUTH_APPS})
    access_tokens = {a: _bootstrap_token(env, a) for a in apps}
    access_tokens = {k: v for k, v in access_tokens.items() if v}
    print(f"\n--- tokens ---  {list(access_tokens.keys())}")

    raw_code = _generate_call(question, hits[:3], access_tokens)
    if not raw_code:
        env.close()
        return {"error": "codegen failed"}

    # Strip any import statements (apis is in scope)
    code = re.sub(r"^\s*(import apis|from apis\b).*$", "", raw_code, flags=re.MULTILINE).strip()
    tokens_literal = json.dumps(access_tokens)
    full_code = f"access_tokens = {tokens_literal}\n{code}"

    print(f"\n--- generated code ---\n{full_code}\n")

    stdout = env.execute(full_code)
    env.close()

    failed = any(x in (stdout or "") for x in ("Traceback", "Exception:", "Execution failed"))
    result = _extract_result(stdout) if not failed else None

    print(f"--- stdout ---\n{stdout}\n")
    print(f"--- result ---\n{json.dumps(result, indent=2, default=str) if result is not None else '(none)'}\n")

    return {"question": question, "hits": hits, "code": full_code, "stdout": stdout, "result": result, "ok": not failed}


def main():
    ap = argparse.ArgumentParser(description="Debug: question → API → execute")
    ap.add_argument("question", nargs="*")
    ap.add_argument("-q", "--query")
    ap.add_argument("--task-id")
    args = ap.parse_args()
    q = args.query or " ".join(args.question).strip()
    if not q:
        ap.error("Provide a question")
    run(q, task_id=args.task_id)


if __name__ == "__main__":
    main()
