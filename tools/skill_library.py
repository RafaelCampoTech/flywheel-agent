"""Persistent SQLite skill library (GBrain-style).

Stores reusable AppWorld code patterns that survive across tasks.
Backed by FLYWHEEL_MEMORY_DIR/skills.db (no extra deps — sqlite3 is stdlib).
Retrieval uses keyword overlap + app matching; fast enough for a few hundred skills.

Usage:
    from tools.skill_library import SkillLibrary
    lib = SkillLibrary(ctx.memory.dir)
    skills = lib.search("spotify song library pagination", apps=["spotify"], top_k=4)
    lib.add_skill("spotify_song_union", ["spotify"], "Collect all song_ids ...", code, tags=["spotify","pagination"])
    lib.log_task(key, apps, kind, api_seq, answer, success=True)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

_DB_FILE = "skills.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    apps          TEXT NOT NULL DEFAULT '[]',
    description   TEXT NOT NULL,
    code          TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    last_used     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_key     TEXT NOT NULL,
    apps         TEXT NOT NULL DEFAULT '[]',
    task_kind    TEXT NOT NULL DEFAULT 'question',
    api_sequence TEXT NOT NULL DEFAULT '[]',
    answer       TEXT NOT NULL DEFAULT '',
    success      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    task_key      TEXT NOT NULL,
    replan_iter   INTEGER NOT NULL DEFAULT 1,
    attempt       INTEGER NOT NULL DEFAULT 1,
    phase         TEXT NOT NULL DEFAULT 'exec_verify',
    code_fp       TEXT NOT NULL DEFAULT '',
    stdout        TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    answer        TEXT NOT NULL DEFAULT '',
    exec_ok       INTEGER NOT NULL DEFAULT 0,
    verify_pass   INTEGER NOT NULL DEFAULT 0,
    verify_reason TEXT NOT NULL DEFAULT '',
    eval_pass     INTEGER,
    eval_critique TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_apps     ON skills(apps);
CREATE INDEX IF NOT EXISTS idx_task_log_apps   ON task_log(apps);
CREATE INDEX IF NOT EXISTS idx_task_log_kind   ON task_log(task_kind);
CREATE INDEX IF NOT EXISTS idx_task_log_succ   ON task_log(success);
CREATE INDEX IF NOT EXISTS idx_exec_logs_sess  ON execution_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_exec_logs_task  ON execution_logs(task_key);
"""

_DEFAULT_SKILLS: List[Dict[str, Any]] = [
    {
        "name": "supervisor_bootstrap",
        "apps": ["supervisor"],
        "description": "Get supervisor profile (email, phone_number) and all app passwords dict. Always run before any app login.",
        "code": (
            "me = apis.supervisor.show_profile()\n"
            "pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}\n"
            "# me['email'], me['phone_number'] | pw['spotify'], pw['amazon'], pw['gmail'], ..."
        ),
        "tags": ["login", "bootstrap", "supervisor", "password"],
    },
    {
        "name": "all_app_logins",
        "apps": [],
        "description": (
            "Exact login code for every AppWorld app. Use when access_tokens is missing a key "
            "(e.g. KeyError: 'venmo'). All email-based except phone which uses phone_number."
        ),
        "code": (
            "me = apis.supervisor.show_profile()\n"
            "pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}\n"
            "\n"
            "# --- email-based logins ---\n"
            "spotify_tok     = apis.spotify.login(username=me['email'], password=pw['spotify'])['access_token']\n"
            "amazon_tok      = apis.amazon.login(username=me['email'], password=pw['amazon'])['access_token']\n"
            "gmail_tok       = apis.gmail.login(username=me['email'], password=pw['gmail'])['access_token']\n"
            "venmo_tok       = apis.venmo.login(username=me['email'], password=pw['venmo'])['access_token']\n"
            "splitwise_tok   = apis.splitwise.login(username=me['email'], password=pw['splitwise'])['access_token']\n"
            "todoist_tok     = apis.todoist.login(username=me['email'], password=pw['todoist'])['access_token']\n"
            "simple_note_tok = apis.simple_note.login(username=me['email'], password=pw['simple_note'])['access_token']\n"
            "file_system_tok = apis.file_system.login(username=me['email'], password=pw['file_system'])['access_token']\n"
            "\n"
            "# --- phone uses phone_number, NOT email ---\n"
            "phone_tok = apis.phone.login(username=me['phone_number'], password=pw['phone'])['access_token']\n"
        ),
        "tags": [
            "login", "access_token", "all_apps",
            "spotify", "amazon", "gmail", "venmo", "splitwise",
            "todoist", "simple_note", "file_system", "phone",
        ],
    },
    {
        "name": "paginated_list_all",
        "apps": [],
        "description": "Fetch ALL pages from a paginated list API using page_index + page_limit loop.",
        "code": (
            "all_items = []\n"
            "page_index = 0\n"
            "while True:\n"
            "    batch = apis.APP.METHOD(access_token=tok, page_index=page_index, page_limit=20)\n"
            "    if not batch:\n"
            "        break\n"
            "    all_items.extend(batch)\n"
            "    if len(batch) < 20:\n"
            "        break\n"
            "    page_index += 1"
        ),
        "tags": ["pagination", "page_index", "list", "all"],
    },
    {
        "name": "spotify_song_union",
        "apps": ["spotify"],
        "description": "Collect all unique song_ids from song library + album library + playlist library (union, dedup).",
        "code": (
            "song_ids = set()\n"
            "# song library\n"
            "pi = 0\n"
            "while True:\n"
            "    b = apis.spotify.show_song_library(access_token=tok, page_index=pi, page_limit=20)\n"
            "    if not b: break\n"
            "    song_ids.update(s['song_id'] for s in b)\n"
            "    if len(b) < 20: break\n"
            "    pi += 1\n"
            "# album library\n"
            "pi = 0\n"
            "while True:\n"
            "    b = apis.spotify.show_album_library(access_token=tok, page_index=pi, page_limit=20)\n"
            "    if not b: break\n"
            "    for album in b:\n"
            "        song_ids.update(apis.spotify.show_album(access_token=tok, album_id=album['album_id']).get('song_ids', []))\n"
            "    if len(b) < 20: break\n"
            "    pi += 1\n"
            "# playlist library\n"
            "pi = 0\n"
            "while True:\n"
            "    b = apis.spotify.show_playlist_library(access_token=tok, page_index=pi, page_limit=20)\n"
            "    if not b: break\n"
            "    for pl in b:\n"
            "        song_ids.update(apis.spotify.show_playlist(access_token=tok, playlist_id=pl['playlist_id']).get('song_ids', []))\n"
            "    if len(b) < 20: break\n"
            "    pi += 1"
        ),
        "tags": ["spotify", "song_id", "union", "album", "playlist", "pagination"],
    },
    {
        "name": "spotify_enrich_songs",
        "apps": ["spotify"],
        "description": "Fetch full song details (title, genre, play_count) for a set of song_ids via show_song.",
        "code": (
            "songs = []\n"
            "for sid in song_ids:\n"
            "    s = apis.spotify.show_song(access_token=tok, song_id=sid)\n"
            "    songs.append(s)\n"
            "# fields: title, genre, play_count, artist, album_id, song_id\n"
            "# filter R&B: g = (s.get('genre') or '').lower(); keep if 'r&b' in g\n"
            "# sort by play_count desc: songs.sort(key=lambda s: -s.get('play_count', 0))"
        ),
        "tags": ["spotify", "show_song", "genre", "play_count", "enrich"],
    },
]


class SkillLibrary:
    """Persistent SQLite skill library backed by FLYWHEEL_MEMORY_DIR/skills.db."""

    def __init__(self, memory_dir: str) -> None:
        os.makedirs(memory_dir, exist_ok=True)
        self._path = os.path.join(memory_dir, _DB_FILE)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        if count > 0:
            return
        now = _now()
        for d in _DEFAULT_SKILLS:
            self._conn.execute(
                "INSERT OR IGNORE INTO skills (name, apps, description, code, tags, success_count, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (d["name"], json.dumps(d["apps"]), d["description"], d["code"],
                 json.dumps(d["tags"]), now, now),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_skill(
        self,
        name: str,
        apps: List[str],
        description: str,
        code: str,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Upsert a skill. Increments success_count on update."""
        now = _now()
        tags = tags or []
        existing = self._conn.execute(
            "SELECT id, success_count FROM skills WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE skills SET apps=?, description=?, code=?, tags=?, "
                "success_count=?, last_used=? WHERE id=?",
                (json.dumps(apps), description, code, json.dumps(tags),
                 existing[1] + 1, now, existing[0]),
            )
        else:
            self._conn.execute(
                "INSERT INTO skills (name, apps, description, code, tags, success_count, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (name, json.dumps(apps), description, code, json.dumps(tags), now, now),
            )
        self._conn.commit()

    def log_task(
        self,
        task_key: str,
        apps: List[str],
        task_kind: str,
        api_sequence: List[str],
        answer: str,
        success: bool,
    ) -> None:
        """Record a task outcome."""
        self._conn.execute(
            "INSERT INTO task_log (task_key, apps, task_kind, api_sequence, answer, success, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_key, json.dumps(apps), task_kind, json.dumps(api_sequence),
             answer or "", int(success), _now()),
        )
        self._conn.commit()

    def log_execution(
        self,
        session_id: str,
        task_key: str,
        replan_iter: int,
        attempt: int,
        phase: str,
        code_fp: str = "",
        stdout: str = "",
        error: str = "",
        answer: str = "",
        exec_ok: bool = False,
        verify_pass: bool = False,
        verify_reason: str = "",
        eval_pass: Optional[bool] = None,
        eval_critique: Optional[str] = None,
    ) -> None:
        """Persist one execution attempt with its verify/evaluate result.

        phase='exec_verify' → logged per check_loop attempt (attempt 1..N).
        phase='evaluate'    → logged once per replan iteration after evaluate_solution
                              (attempt=0, exec_ok/eval_pass/eval_critique populated).
        stdout and error are truncated to their last 4000/2000 chars to keep the DB lean
        while preserving the most useful tail (tracebacks, API responses, ANSWER= lines).
        """
        self._conn.execute(
            "INSERT INTO execution_logs "
            "(session_id, task_key, replan_iter, attempt, phase, code_fp, "
            "stdout, error, answer, exec_ok, verify_pass, verify_reason, "
            "eval_pass, eval_critique, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                task_key,
                replan_iter,
                attempt,
                phase,
                code_fp or "",
                (stdout or "")[-4000:],
                (error or "")[-2000:],
                answer or "",
                int(exec_ok),
                int(verify_pass),
                verify_reason or "",
                int(eval_pass) if eval_pass is not None else None,
                eval_critique,
                _now(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, apps: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
        """Return relevant skills by keyword overlap + app match. O(n_skills).

        Index text = name + description + tags + API method names extracted from code.
        API method names (show_song_library, page_index, access_token …) are the best
        signal: a task instruction that mentions "song library" or "pagination" will
        directly overlap with the method names in a working skill's code.
        """
        q_words = set(re.findall(r"\w+", query.lower()))
        app_set = set(apps)

        rows = self._conn.execute(
            "SELECT name, apps, description, code, tags, success_count "
            "FROM skills ORDER BY success_count DESC, last_used DESC LIMIT 200"
        ).fetchall()

        scored: List[tuple] = []
        for name, apps_json, desc, code, tags_json, cnt in rows:
            skill_apps = set(json.loads(apps_json or "[]"))
            skill_tags = set(json.loads(tags_json or "[]"))

            # Extract API method names from code (apis.app.method_name → "method_name")
            code_methods = set(re.findall(r"apis\.\w+\.(\w+)", code or ""))
            # Also extract bare identifiers likely to be domain terms
            code_words = set(re.findall(r"\b([a-z][a-z_]{3,})\b", (code or "").lower()))

            index_text = " ".join(
                [name, desc]
                + list(skill_tags)
                + list(code_methods)
                + list(code_words)
            ).lower()
            s_words = set(re.findall(r"\w+", index_text))

            overlap = len(q_words & s_words)
            score = overlap * 2
            if skill_apps & app_set:
                score += 6
            if not skill_apps:  # general (pagination, login) — always useful
                score += 2
            score += min(cnt, 10)  # boost frequently successful skills

            if score <= 2:
                continue
            scored.append((score, {
                "name": name,
                "apps": json.loads(apps_json),
                "description": desc,
                "code": code,
                "success_count": cnt,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def search_multi(self, queries: List[str], apps: List[str], top_k: int = 6) -> List[Dict[str, Any]]:
        """Search with multiple queries and merge by best score per skill (deduped)."""
        seen: Dict[str, int] = {}  # name -> best score index
        merged: List[Dict[str, Any]] = []
        for q in queries:
            if not q or not q.strip():
                continue
            for s in self.search(q, apps, top_k=top_k):
                if s["name"] not in seen:
                    seen[s["name"]] = len(merged)
                    merged.append(s)
        return merged[:top_k]

    def similar_tasks(self, apps: List[str], task_kind: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return recent successful tasks with overlapping apps."""
        rows = self._conn.execute(
            "SELECT task_key, apps, api_sequence, answer FROM task_log "
            "WHERE success=1 AND task_kind=? ORDER BY created_at DESC LIMIT 100",
            (task_kind,),
        ).fetchall()

        app_set = set(apps)
        scored: List[tuple] = []
        for task_key, apps_json, api_seq, answer in rows:
            task_apps = set(json.loads(apps_json or "[]"))
            overlap = len(task_apps & app_set)
            if overlap:
                scored.append((overlap, {
                    "task_key": task_key,
                    "apps": list(task_apps),
                    "api_sequence": json.loads(api_seq or "[]"),
                    "answer_preview": (answer or "")[:80],
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def close(self) -> None:
        self._conn.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat()


def skill_name_for(instruction: str, apps: List[str]) -> str:
    """Derive a short stable skill name from a task instruction."""
    app_str = "_".join(sorted(apps)[:2]) if apps else "general"
    words = [w for w in re.findall(r"[a-z]+", instruction.lower()) if len(w) > 3][:3]
    suffix = "_".join(words) if words else "task"
    return f"{app_str}_{suffix}"[:60]
