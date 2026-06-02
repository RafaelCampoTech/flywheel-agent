"""Schema loading, RAG candidates, doc grounding, world execution.

Schemas come from committed api_docs_dump/{app}.json files (fast, offline-safe).
Falls back to the api_doc MCP call for anything not found in those files.
Doc cache persists in ctx.memory across the task stream so schemas are fetched at most once.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("grounding")

_SCHEMA_DIR = Path(__file__).parent / "api_docs_dump"
_TRACEBACK_MARKERS = (
    "Traceback (most recent call last)",
    "Exception:",
    "KeyError:",
    "TypeError:",
    "NameError:",
    "AttributeError:",
    "ValueError:",
)

# In-process cache of parsed {app}.json files (avoids re-reading on every task)
_file_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schema loading from committed JSON files
# ---------------------------------------------------------------------------

def _load_app_file(app: str) -> Dict[str, Any]:
    if app not in _file_cache:
        path = _SCHEMA_DIR / f"{app}.json"
        _file_cache[app] = json.loads(path.read_text()) if path.exists() else {}
    return _file_cache[app]


def compact_schema(app: str, api: str) -> Optional[Dict[str, Any]]:
    """Return {app, api, description, parameters, response} from the local JSON file."""
    raw = _load_app_file(app).get(api)
    if not raw:
        return None
    params = [
        {
            "name": p["name"],
            "type": p.get("type", "any"),
            "required": p.get("required", False),
            "description": p.get("description", ""),
            "constraints": p.get("constraints") or [],
        }
        for p in (raw.get("parameters") or [])
    ]
    return {
        "app": app,
        "api": api,
        "description": raw.get("description", ""),
        "parameters": params,
        "response": raw.get("response_schemas", {}).get("success"),
    }


def list_app_apis(app: str) -> List[str]:
    """Return all API names available for an app from its JSON file."""
    return list(_load_app_file(app).keys())


# ---------------------------------------------------------------------------
# RAG candidates via search_apis
# ---------------------------------------------------------------------------

def rag_candidates(ctx, instruction: str, apps: List[str]) -> List[Tuple[str, str]]:
    """Run search_apis (RAG) to collect deduplicated (app, api) name pairs.

    Runs a broad query over the full instruction plus one targeted query per
    identified app. Supervisor fundamentals are always included.
    """
    seen: set = set()
    pairs: List[Tuple[str, str]] = []

    def _add(results: Any, limit: int) -> None:
        lst = results if isinstance(results, list) else (results or {}).get("results", [])
        added = 0
        for r in lst:
            if added >= limit:
                break
            a, api_name = r.get("app") or "", r.get("api") or ""
            if a and api_name and (a, api_name) not in seen:
                seen.add((a, api_name))
                pairs.append((a, api_name))
                added += 1

    # Supervisor fundamentals — always needed for login
    for api_name in ("show_profile", "show_account_passwords"):
        key = ("supervisor", api_name)
        seen.add(key)
        pairs.append(key)

    # Broad query
    _add(ctx.retrieve(instruction), limit=6)

    # Targeted per-app queries
    for app in apps:
        if app in ("supervisor", "api_docs"):
            continue
        _add(ctx.retrieve(f"{app} {instruction}"), limit=5)

    log.info("RAG candidates → %d (app, api) pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Ground docs — full pipeline: RAG → schema lookup → cache
# ---------------------------------------------------------------------------

def ground_docs(
    ctx,
    instruction: str,
    apps: List[str],
    doc_cache: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (api_schemas, updated_doc_cache).

    Priority:
      1. Memory doc_cache (already fetched in a prior task)
      2. Local JSON file (committed, offline-safe)
      3. api_doc MCP call (fallback for any gap)
    """
    pairs = rag_candidates(ctx, instruction, apps)
    schemas: List[Dict[str, Any]] = []
    cache = dict(doc_cache)
    new_entries = 0

    for app, api in pairs:
        key = f"{app}.{api}"

        if key in cache:
            schemas.append(cache[key])
            continue

        # Try local JSON file
        s = compact_schema(app, api)
        if s:
            cache[key] = s
            schemas.append(s)
            new_entries += 1
            continue

        # Fall back to api_doc MCP call
        result = ctx.mcp.call("api_doc", {"app": app, "api": api})
        if isinstance(result, dict) and result.get("doc"):
            s = {
                "app": app, "api": api,
                "description": str(result["doc"])[:800],
                "parameters": [], "response": None,
            }
            cache[key] = s
            schemas.append(s)
            new_entries += 1
        else:
            log.warning("Schema not found for %s.%s", app, api)

    log.info("SCHEMAS: %d loaded (%d new to cache)", len(schemas), new_entries)
    return schemas, cache


# ---------------------------------------------------------------------------
# World execution
# ---------------------------------------------------------------------------

def world_run(ctx, code: str) -> Tuple[bool, str]:
    """Execute a Python snippet via ctx.run_code, return (success, stdout)."""
    if not isinstance(code, str) or not code.strip():
        return False, "empty_code"
    out = str(ctx.run_code(code) or "")
    if any(m in out for m in _TRACEBACK_MARKERS):
        return False, out[-2000:]
    return True, out
