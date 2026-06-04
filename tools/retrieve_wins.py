"""CLI: search the Chroma wins store by task description.

  python tools/retrieve_wins.py "top 4 most played R&B songs on spotify"
  python tools/retrieve_wins.py "pay joseph on venmo" --apps venmo,phone
  python tools/retrieve_wins.py "spotify songs" --top-k 5 --no-code
  python tools/retrieve_wins.py --list          # dump all stored wins
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.wins_store import WinsStore  # noqa: E402

_SEP   = "─" * 72
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_YELLOW = "\033[33m"


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _score_bar(sim: float, width: int = 20) -> str:
    filled = round(sim * width)
    bar = "█" * filled + "░" * (width - filled)
    if sim >= 0.8:
        color = _GREEN
    elif sim >= 0.5:
        color = _YELLOW
    else:
        color = _DIM
    return _color(f"[{bar}]", color)


def _print_win(idx: int, hit: dict, show_code: bool) -> None:
    sim = hit.get("similarity", 0.0)
    print(_color(_SEP, _DIM))
    header = f"  Result {idx}  similarity={sim:.3f}  {_score_bar(sim)}"
    if hit.get("created_at"):
        header += _color(f"  {hit['created_at'][:10]}", _DIM)
    print(_color(header, _BOLD))
    print()

    desc = hit.get("description") or ""
    print(f"  {_color('Description :', _CYAN)} {desc}")
    print(f"  {_color('Apps        :', _CYAN)} {', '.join(hit.get('apps') or [])}")
    print(f"  {_color('Task kind   :', _CYAN)} {hit.get('task_kind', '?')}")

    questions = hit.get("questions") or []
    if questions:
        print(f"  {_color('Plan        :', _CYAN)}")
        for i, q in enumerate(questions, 1):
            print(f"    {_color(str(i) + '.', _DIM)} {q}")

    api_seq = hit.get("api_sequence") or []
    if api_seq:
        print(f"  {_color('API sequence:', _CYAN)} {', '.join(api_seq)}")

    code = (hit.get("code") or "").strip()
    if show_code and code:
        print(f"  {_color('Code        :', _CYAN)}")
        for line in code.splitlines():
            print(f"    {_color(line, _DIM)}")

    print()


def _list_all(store: WinsStore) -> None:
    count = store._col.count()
    print(f"Wins store contains {_color(str(count), _BOLD)} documents.\n")
    if count == 0:
        return
    raw = store._col.get(include=["metadatas", "documents"])
    docs      = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        apps = json.loads(meta.get("apps", "[]"))
        kind = meta.get("task_kind", "?")
        date = (meta.get("created_at") or "")[:10]
        print(f"  {i:>3}.  {_color(date, _DIM)}  [{kind:8s}]  {', '.join(apps):<20s}  {doc[:80]}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Search the Chroma wins store by task description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query", nargs="?", default="",
                    help="Natural-language task description to search for.")
    ap.add_argument("--apps", default="",
                    help="Comma-separated app names to filter by (e.g. spotify,phone).")
    ap.add_argument("--top-k", type=int, default=3, dest="top_k",
                    help="Number of results to return (default: 3).")
    ap.add_argument("--no-code", action="store_true",
                    help="Suppress code in output.")
    ap.add_argument("--list", action="store_true",
                    help="List all stored wins (ignores query).")
    ap.add_argument("--memory-dir", default=None,
                    help="Path to memory dir (default: memory/ inside repo).")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit raw JSON instead of pretty output.")
    args = ap.parse_args()

    memory_dir = args.memory_dir or str(_REPO_ROOT / "memory")
    store = WinsStore(memory_dir)

    if args.list:
        _list_all(store)
        return

    query = args.query.strip()
    if not query:
        ap.print_help()
        sys.exit(0)

    apps: List[str] = [a.strip() for a in args.apps.split(",") if a.strip()]

    hits = store.retrieve(query, top_k=args.top_k, apps=apps or None)

    if args.as_json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return

    label = f'"{query}"'
    if apps:
        label += f"  apps={apps}"
    print(f"\n{_color('Searching for:', _BOLD)} {label}")
    print(f"{_color('Found:', _BOLD)} {len(hits)} result(s)\n")

    if not hits:
        print("  No matching wins found.")
        return

    for i, hit in enumerate(hits, 1):
        _print_win(i, hit, show_code=not args.no_code)

    print(_color(_SEP, _DIM))


if __name__ == "__main__":
    main()
