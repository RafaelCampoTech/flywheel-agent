"""PLAN_CALL: one model call that produces a structured plan (no code).

The plan is the portable, validated skill. Code is its disposable rendering.
Store plans, not code, in memory.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from utils import extract_json, model_content

log = logging.getLogger("planner")

PLAN_SYSTEM = """\
You produce a STRUCTURED PLAN (JSON) for an AppWorld task. You do NOT write code yet.

## Provided in the payload
- instruction    : the task to complete
- apps           : apps involved (already identified)
- api_schemas    : exact signatures + response field shapes for those apps' APIs
- proven_plans   : validated plans from similar past tasks — adapt one if it fits
- last_error     : validator or runtime error from the previous attempt (empty on first try)

## Method: reason BACKWARD from the output
1. State the goal: the exact end state (action) or output string (question).
2. List the fields the goal needs (e.g. title, play_count, genre).
3. For EACH field, find which api_schemas entry has it in its RESPONSE.
   - If a field is NOT in the list API's response items, you MUST add a detail API (enrich).
   - This is the most common mistake: filter/rank/output fields must be explicitly collected.
4. Identify the full universe to collect; note unions across sources that need dedup by id.

## How to read api_schemas
Each entry has:
  - app, api    : call as apis.<app>.<api>(...)
  - parameters  : list of {name, type, required}
  - response    : success response shape — a dict (single item) or list-of-one-item (paginated)
                  ONLY fields present here can be used downstream.

## Hard rules
- Use ONLY api names and field names that appear in api_schemas.
- Every field used in filter / rank / output MUST appear in enrich.fields
  OR be provably in a collect step's response items. Never assume a field is there.
- For unions ("across my song, album and playlist libraries") list ALL sources; dedup by id.
- User-supplied string values (genre, names) are matched case-insensitively in code.

## Output — single JSON object, no code
{
  "apps"   : ["supervisor", "..."],
  "kind"   : "question" | "action",
  "goal"   : "<one-line description of end state or expected output>",
  "collect": [
    {
      "list_api"  : "<api name for paginated list>",
      "id_field"  : "<field name on each list item that holds the item's id OR ids array>",
      "is_array"  : <true if id_field is an array of ids, false if a single id>
    }
  ],
  "dedup"  : "<dedup strategy, e.g. 'by song_id set', or empty string if not needed>",
  "enrich" : {
    "detail_api" : "<api that returns per-item detail>",
    "key"        : "<id field passed to detail_api>",
    "fields"     : ["<field1>", "<field2>"]
  } | null,
  "filter" : {"field": "<name>", "op": "equals_ci|gte|lte|gt|lt|contains_ci", "value": "<val>"} | null,
  "rank"   : {"field": "<name>", "order": "desc|asc", "take": <int>} | null,
  "output" : {"format": "comma_separated|single_value|count", "item_field": "<field>", "separator": ", "} | null,
  "writes" : [
    {
      "api"               : "<write api name>",
      "per"               : "<what we iterate over, e.g. 'artist' or 'product'>",
      "args_note"         : "<key args hint>",
      "idempotency_check" : "<read api to check before writing, or empty>"
    }
  ],
  "verify" : "<what to re-read to confirm the goal was reached>"
}
"""


def plan_call(
    ctx,
    instruction: str,
    apps: List[str],
    api_schemas: List[Dict[str, Any]],
    proven_plans: List[Dict[str, Any]],
    last_error: str,
) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "apps": apps,
        "api_schemas": api_schemas,
        "proven_plans": proven_plans,
        "last_error": last_error,
    }
    data = ctx.model(
        [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    plan = extract_json(model_content(data))
    log.info(
        "PLAN kind=%s goal=%r collect=%d writes=%d",
        plan.get("kind"), (plan.get("goal") or "")[:60],
        len(plan.get("collect") or []), len(plan.get("writes") or []),
    )
    return plan
