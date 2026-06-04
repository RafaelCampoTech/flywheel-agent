"""code_examples Chroma collection: successful solutions only."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.chroma_client import COLLECTION_CODE_EXAMPLES, get_collection
from memory.paths import pending_solution_path

_SUCCESS_FILTER = {"evaluation_result": "success"}


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def write_pending(
    task_desc: str,
    apis_used: list,
    plan: dict,
    code: str,
) -> str:
    """Record a solution awaiting oracle confirmation on the next task."""
    row_id = str(uuid.uuid4())
    payload = {
        "id": row_id,
        "task_description": task_desc,
        "apis_used": apis_used,
        "plan": plan,
        "working_code": code,
        "evaluation_result": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(pending_solution_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return row_id


def get_pending_id() -> str | None:
    path = pending_solution_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("id")
    except (OSError, json.JSONDecodeError):
        return None


def confirm_pending_success() -> bool:
    """Promote pending solution to code_examples if present. Returns True if promoted."""
    path = pending_solution_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    add_success(
        task_desc=data["task_description"],
        apis_used=data.get("apis_used", []),
        plan=data.get("plan", {}),
        code=data.get("working_code", ""),
        row_id=data.get("id"),
    )
    try:
        os.remove(path)
    except OSError:
        pass
    return True


def discard_pending() -> None:
    try:
        os.remove(pending_solution_path())
    except OSError:
        pass


def add_success(
    task_desc: str,
    apis_used: list,
    plan: dict,
    code: str,
    row_id: str | None = None,
) -> str:
    """Add a successful task to code_examples (embedding = task description)."""
    col = get_collection(COLLECTION_CODE_EXAMPLES)
    row_id = row_id or str(uuid.uuid4())
    col.add(
        documents=[task_desc],
        metadatas=[
            {
                "evaluation_result": "success",
                "apis_used": json.dumps(apis_used),
                "plan": json.dumps(plan),
                "working_code": code[:8000],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
        ids=[row_id],
    )
    return row_id


def search(
    task_description: str,
    top_k: int = 3,
    rag_apis: list[str] | None = None,
) -> list[dict]:
    """Return similar successful solutions (Chroma + optional API re-rank)."""
    col = get_collection(COLLECTION_CODE_EXAMPLES)
    if col.count() == 0:
        return []

    hits = col.query(
        query_texts=[task_description],
        n_results=min(top_k * 4, max(col.count(), 1)),
        where=_SUCCESS_FILTER,
    )
    rows = _hits_to_rows(hits)
    if not rows:
        return []

    task_words = _word_set(task_description)
    rag_api_set = set(rag_apis or [])

    scored: list[tuple[float, dict]] = []
    for row in rows:
        text_score = _jaccard(task_words, _word_set(row["task_description"]))
        api_score = _jaccard(rag_api_set, set(row["apis_used"])) if rag_api_set else 0.0
        chroma_score = row.get("_score", 0.0)
        total = 0.4 * chroma_score + 0.35 * text_score + 0.25 * api_score
        scored.append((total, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for score, row in scored[:top_k] if score > 0.0]


def _hits_to_rows(hits: dict) -> list[dict]:
    docs = hits.get("documents") or [[]]
    metas = hits.get("metadatas") or [[]]
    ids = hits.get("ids") or [[]]
    distances = hits.get("distances") or [[]]
    if not docs or not docs[0]:
        return []
    rows: list[dict] = []
    for i, task_desc in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        dist = distances[0][i] if distances and distances[0] else None
        score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
        rows.append(
            {
                "id": ids[0][i] if ids and ids[0] else "",
                "task_description": task_desc,
                "apis_used": json.loads(meta.get("apis_used") or "[]"),
                "plan": json.loads(meta.get("plan") or "{}"),
                "working_code": meta.get("working_code", ""),
                "evaluation_result": meta.get("evaluation_result", "success"),
                "_score": score,
            }
        )
    return rows
