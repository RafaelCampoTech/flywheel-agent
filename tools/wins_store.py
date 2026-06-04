"""Chroma-backed store for winning task solutions.

Each document = one winning solve() run, keyed by hashed instruction.
Embedding over the task instruction text enables semantic prior-plan retrieval.

Layout inside memory_dir:
    wins_chroma/    ← PersistentClient path (Chroma internals + SQLite)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

_COLLECTION = "wins"
_CHROMA_DIR = "wins_chroma"


class WinsStore:
    """Persist winning task solutions and retrieve them by semantic similarity."""

    def __init__(self, memory_dir: str) -> None:
        import chromadb

        persist_path = os.path.join(memory_dir, _CHROMA_DIR)
        os.makedirs(persist_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._migrate_from_json(memory_dir)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_win(
        self,
        instruction: str,
        apps: List[str],
        task_kind: str,
        questions: List[str],
        code: str,
        api_sequence: List[str],
    ) -> None:
        """Upsert a winning solution (latest version replaces previous for same instruction)."""
        self._col.upsert(
            ids=[_win_id(instruction)],
            documents=[instruction],
            metadatas=[{
                "apps": json.dumps(apps),
                "task_kind": task_kind,
                "questions": json.dumps(questions),
                "api_sequence": json.dumps(api_sequence),
                "code": (code or "")[:1500],
                "created_at": datetime.utcnow().isoformat(),
            }],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        instruction: str,
        apps: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return prior winning plans most similar to instruction with app overlap."""
        hits = self.retrieve(instruction, top_k=top_k * 4, apps=apps)
        return [
            {k: v for k, v in h.items() if k != "similarity"}
            for h in hits[:top_k]
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        apps: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search returning full metadata + similarity score.

        Args:
            query:  Natural-language task description to search against.
            top_k:  Maximum results to return.
            apps:   Optional app list; when given, only wins with at least one
                    overlapping app are returned.  Pass None for no filter.
        """
        count = self._col.count()
        if count == 0:
            return []
        fetch = min(max(top_k * 4, 10), count) if apps else min(top_k, count)
        results = self._col.query(
            query_texts=[query],
            n_results=fetch,
            include=["metadatas", "documents", "distances"],
        )
        metadatas  = (results.get("metadatas")  or [[]])[0]
        documents  = (results.get("documents")  or [[]])[0]
        distances  = (results.get("distances")  or [[]])[0]

        app_set = set(apps) if apps else None
        out: List[Dict[str, Any]] = []
        for meta, doc, dist in zip(metadatas, documents, distances):
            win_apps = json.loads(meta.get("apps", "[]"))
            if app_set and not (set(win_apps) & app_set):
                continue
            # Chroma cosine space: distance = 1 - cosine_similarity
            similarity = round(max(0.0, 1.0 - dist), 4)
            out.append({
                "description": doc,
                "apps": win_apps,
                "task_kind": meta.get("task_kind", "question"),
                "questions": json.loads(meta.get("questions", "[]")),
                "api_sequence": json.loads(meta.get("api_sequence", "[]")),
                "code": meta.get("code", ""),
                "created_at": meta.get("created_at", ""),
                "similarity": similarity,
            })
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate_from_json(self, memory_dir: str) -> None:
        """One-time import of legacy memory.json wins into this collection.

        Backfills code from skills.db by matching the skill name derived from
        each task_key — the same derivation used by solve() when it saves a skill.
        """
        if self._col.count() > 0:
            return
        json_path = os.path.join(memory_dir, "memory.json")
        if not os.path.isfile(json_path):
            return
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            wins = data.get("wins", {}) if isinstance(data, dict) else {}
            if not isinstance(wins, dict) or not wins:
                return

            skill_code = _load_skills_code(memory_dir)

            ids, documents, metadatas = [], [], []
            for task_key, win in wins.items():
                if not isinstance(win, dict):
                    continue
                apps = win.get("apps") or []
                code = _lookup_skill_code(task_key, apps, skill_code)
                ids.append(_win_id(task_key))
                documents.append(task_key)
                metadatas.append({
                    "apps": json.dumps(apps),
                    "task_kind": win.get("task_kind", "question"),
                    "questions": json.dumps(win.get("questions") or []),
                    "api_sequence": json.dumps(win.get("api_sequence") or []),
                    "code": code,
                    "created_at": datetime.utcnow().isoformat(),
                })
            if ids:
                self._col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            pass  # migration is best-effort; failures are non-fatal


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _win_id(instruction: str) -> str:
    """Stable document ID from the first 100 chars of the instruction."""
    key = instruction.strip().lower()[:100]
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def _load_skills_code(memory_dir: str) -> Dict[str, str]:
    """Return {skill_name: code} for all skills that have non-empty code."""
    db_path = os.path.join(memory_dir, "skills.db")
    if not os.path.isfile(db_path):
        return {}
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT name, code FROM skills WHERE length(code) > 0"
        ).fetchall()
        conn.close()
        return {name: code for name, code in rows}
    except Exception:
        return {}


def _lookup_skill_code(task_key: str, apps: List[str], skill_code: Dict[str, str]) -> str:
    """Derive the skill name from task_key+apps and return its code, or ''."""
    try:
        import re as _re
        app_str = "_".join(sorted(apps)[:2]) if apps else "general"
        words = [w for w in _re.findall(r"[a-z]+", task_key.lower()) if len(w) > 3][:3]
        suffix = "_".join(words) if words else "task"
        name = f"{app_str}_{suffix}"[:60]
        return skill_code.get(name, "")
    except Exception:
        return ""
