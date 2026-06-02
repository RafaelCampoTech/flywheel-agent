"""CODE_CALL: one model call that turns a validated plan into work_code + verify_code.

The plan has already been validated (all APIs and fields confirmed to exist).
The coder's job is purely mechanical: render the plan steps into correct Python.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from utils import extract_json, model_content

log = logging.getLogger("coder")

_VERIFY_MARKER = "__FW_CODEACT__"

CODE_SYSTEM = """\
You control AppWorld by writing two Python snippets that run with `apis` in scope.
You are given a VALIDATED plan (its APIs and fields are confirmed to exist) plus api_schemas.

## Your job
1. Render plan.collect / dedup / enrich / filter / rank / output (or plan.writes) into work_code
   using ONLY the api names and field names in the plan and api_schemas.
2. Write verify_code that re-reads state fresh and confirms plan.verify holds.

## Login flow (required for every authenticated call)
me = apis.supervisor.show_profile()
pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}
tok = apis.<app>.login(username=me['email'], password=pw['<app>'])['access_token']
# phone app ONLY: use me['phone_number'] instead of me['email']
# Obtain the token once per app; reuse it throughout the snippet.

## Code rules
- Do NOT call apis.supervisor.complete_task() — the harness submits after verification.
- work_code MUST be idempotent: read current state first, apply only the missing delta.
  Re-running work_code must not create duplicates or double-apply payments/follows/orders.
- Paginate every list API: loop page_index=0,1,2,... until len(page) < page_limit or page is empty.
- Per-song fields (genre, play_count, title) are NOT on library list items — get them via
  show_song(song_id=...). Per-album detail comes from show_album(album_id=...), etc.
- Compare user-supplied strings (genre names, labels) case-insensitively.
- verify_code MUST end with exactly one printed line:
    __FW_CODEACT__{"verified": <bool>, "answer": "<str>", "notes": "<str>"}
  "answer" is the concise response for question tasks; empty string for action tasks.
- A clean run (no traceback) does NOT mean correct — verify_code is mandatory.

## Output — single JSON object
{
  "work_code"  : "<python snippet>",
  "verify_code": "<python snippet that ends with the __FW_CODEACT__ line>"
}
"""


def code_call(
    ctx,
    plan: Dict[str, Any],
    api_schemas: List[Dict[str, Any]],
    last_error: str,
) -> Dict[str, Any]:
    payload = {
        "plan": plan,
        "api_schemas": api_schemas,
        "last_error": last_error,
    }
    data = ctx.model(
        [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    result = extract_json(model_content(data))
    log.info(
        "CODE work=%d chars verify=%d chars",
        len(result.get("work_code") or ""),
        len(result.get("verify_code") or ""),
    )
    return result


def parse_verdict(stdout: str) -> Dict[str, Any]:
    """Extract the __FW_CODEACT__ JSON verdict line from verify_code stdout."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_VERIFY_MARKER):
            return extract_json(line[len(_VERIFY_MARKER):].strip())
    return {}
