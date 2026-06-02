"""Deterministic plan validation against the loaded API schemas.

Catches the most common silent-failure class before any code runs:
  - referencing an API that doesn't exist
  - filtering/ranking/outputting on a field that isn't in the response of the API
    expected to produce it (e.g. filtering on 'genre' from show_song_library, which
    doesn't have genre — must enrich via show_song first)

No model calls, no world calls. Pure dict inspection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

log = logging.getLogger("validator")


def _response_fields(schema: Dict[str, Any]) -> Set[str]:
    """Extract available field names from a compact schema's response."""
    resp = schema.get("response")
    if isinstance(resp, dict):
        return set(resp.keys())
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        return set(resp[0].keys())
    return set()


def validate_plan(
    plan: Dict[str, Any],
    api_schemas: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Returns (is_valid, list_of_error_strings).
    Errors name the specific API or field that's missing and suggest the fix.
    """
    if not plan:
        return False, ["empty plan"]

    # Build lookup: api_name → compact schema (assumes names unique within task scope)
    api_map: Dict[str, Dict[str, Any]] = {s["api"]: s for s in api_schemas}

    errors: List[str] = []

    def _check_api(api_name: str, context: str) -> bool:
        if not api_name:
            return True  # nothing to check
        if api_name not in api_map:
            errors.append(
                f"{context}: api '{api_name}' not in loaded schemas. "
                f"Available: {sorted(api_map.keys())}"
            )
            return False
        return True

    # -----------------------------------------------------------------------
    # Collect — list APIs + id_field
    # -----------------------------------------------------------------------
    collected_fields: Set[str] = set()
    for item in plan.get("collect") or []:
        api_name = item.get("list_api", "")
        if _check_api(api_name, "collect.list_api"):
            resp = _response_fields(api_map[api_name])
            collected_fields.update(resp)
            # Verify the id_field exists on items
            id_field = item.get("id_field", "")
            if id_field and resp and id_field not in resp:
                errors.append(
                    f"collect.id_field '{id_field}' not in {api_name} response "
                    f"fields {resp}."
                )
        # Optional per-collect detail API
        if item.get("detail_api"):
            _check_api(item["detail_api"], "collect.detail_api")

    # -----------------------------------------------------------------------
    # Enrich — detail API whose response supplies filter/rank/output fields
    # -----------------------------------------------------------------------
    enrich_fields: Set[str] = set()
    enrich = plan.get("enrich")
    if enrich:
        detail_api = enrich.get("detail_api", "")
        if _check_api(detail_api, "enrich.detail_api") and detail_api:
            enrich_resp = _response_fields(api_map[detail_api])
            for field in enrich.get("fields") or []:
                if enrich_resp and field not in enrich_resp:
                    errors.append(
                        f"enrich.field '{field}' not in {detail_api} response "
                        f"{enrich_resp}. Check the api_schemas response shape."
                    )
                else:
                    enrich_fields.add(field)

    # All downstream-available fields
    all_fields = collected_fields | enrich_fields

    # -----------------------------------------------------------------------
    # Filter field must be available
    # -----------------------------------------------------------------------
    flt = plan.get("filter")
    if flt:
        field = flt.get("field", "")
        if field and all_fields and field not in all_fields:
            errors.append(
                f"filter.field '{field}' not in collected/enriched fields {all_fields}. "
                f"Add '{field}' to enrich.fields (fetched via a detail API that has it)."
            )

    # -----------------------------------------------------------------------
    # Rank field must be available
    # -----------------------------------------------------------------------
    rank = plan.get("rank")
    if rank:
        field = rank.get("field", "")
        if field and all_fields and field not in all_fields:
            errors.append(
                f"rank.field '{field}' not in collected/enriched fields {all_fields}. "
                f"Add '{field}' to enrich.fields."
            )

    # -----------------------------------------------------------------------
    # Output item_field must be available
    # -----------------------------------------------------------------------
    output = plan.get("output")
    if output:
        field = output.get("item_field", "")
        if field and all_fields and field not in all_fields:
            errors.append(
                f"output.item_field '{field}' not in collected/enriched fields {all_fields}."
            )

    # -----------------------------------------------------------------------
    # Writes — write APIs must exist
    # -----------------------------------------------------------------------
    for write in plan.get("writes") or []:
        _check_api(write.get("api", ""), "writes.api")
        check_api = write.get("idempotency_check", "")
        if check_api:
            _check_api(check_api, "writes.idempotency_check")

    is_valid = len(errors) == 0
    if not is_valid:
        log.warning("PLAN INVALID (%d errors):\n  %s", len(errors), "\n  ".join(errors))
    else:
        log.info("PLAN VALID")
    return is_valid, errors
