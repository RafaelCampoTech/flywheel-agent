"""Plan library: read, retrieve, store, reflect.

Memory structure (persisted in ctx.memory / memory.json):
  {
    "plans"    : { "<archetype_key>": {"plan": {...}, "uses": int, "sample": "..."} },
    "doc_cache": { "<app>.<api>": <compact_schema> },
    "notes"    : [ {"app": "...", "lesson": "...", "task": "..."} ]
  }

Rules:
  - Write a plan ONLY on verified success. Unverified plans poison the library.
  - Retrieve by token-overlap similarity (simple, fast, no model call).
  - Notes are failure reflections; they don't count as plans.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

log = logging.getLogger("memory_store")


def read_memory(ctx) -> Dict[str, Any]:
    mem = ctx.memory.read() or {}
    if not isinstance(mem, dict):
        mem = {}
    mem.setdefault("plans", {})
    mem.setdefault("doc_cache", {})
    mem.setdefault("notes", [])
    return mem


def _archetype_key(instruction: str) -> str:
    s = re.sub(r"\s+", " ", instruction.strip().lower())
    s = re.sub(r"[^a-z0-9 ,._-]+", "", s)
    return s[:140] or "unknown"


def retrieve_plans(instruction: str, memory: Dict[str, Any], k: int = 2) -> List[Dict[str, Any]]:
    """Return up to k stored plans most relevant to this instruction by token overlap."""
    plans = memory.get("plans") or {}
    inst = instruction.lower()
    scored: List = []
    for key, entry in plans.items():
        if not isinstance(entry, dict):
            continue
        score = sum(
            1 for tok in re.split(r"[^a-z0-9]+", key.lower()) if tok and tok in inst
        )
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


def retrieve_notes(apps: List[str], memory: Dict[str, Any]) -> List[str]:
    """Return failure lessons relevant to the apps in this task."""
    notes = memory.get("notes") or []
    app_set = set(a.lower() for a in apps)
    return [
        n["lesson"] for n in notes
        if isinstance(n, dict) and n.get("app", "").lower() in app_set
    ][:5]  # cap to avoid context bloat


def store_plan(ctx, instruction: str, plan: Dict[str, Any], memory: Dict[str, Any]) -> None:
    """Persist a verified plan to memory. Call ONLY after verified success."""
    key = _archetype_key(instruction)
    existing = memory["plans"].get(key, {})
    memory["plans"][key] = {
        "plan": plan,
        "uses": existing.get("uses", 0) + 1,
        "sample": instruction[:200],
    }
    ctx.memory.write("plans", memory["plans"])
    log.info("MEMORY stored plan key=%r uses=%d", key[:60], memory["plans"][key]["uses"])


def store_reflection(
    ctx, app: str, lesson: str, task: str, memory: Dict[str, Any]
) -> None:
    """Record a failure lesson (not a plan). Injected as context on future tasks."""
    notes = list(memory.get("notes") or [])
    notes.append({"app": app, "lesson": lesson, "task": task[:80]})
    if len(notes) > 60:
        notes = notes[-60:]
    memory["notes"] = notes
    ctx.memory.write("notes", notes)
