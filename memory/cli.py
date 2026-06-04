"""CLI: index and search Chroma collections.

  python -m memory.cli index-api-docs [--force]
  python -m memory.cli search-api-docs "task description" [--top-k 20]
  python -m memory.cli search-code-examples "task description" [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import sys

from memory import code_store, index
from memory.api_retriever import LocalApiRetriever
from memory.paths import chroma_path, memory_dir


def cmd_index_api_docs(force: bool) -> int:
    n = index.index_api_docs(force=force)
    print(f"api_docs collection: {n} document(s) in {chroma_path()}")
    return 0


def cmd_search_api_docs(query: str, top_k: int, show_docs: bool) -> int:
    retriever = LocalApiRetriever(top_k=top_k, bootstrap=True)
    hits = retriever.search(query)
    if not hits:
        print("(no matches in api_docs — run index-api-docs first)")
        return 0
    for i, hit in enumerate(hits, 1):
        api_id = hit.get("api_id", f"{hit['app']}.{hit['api']}")
        score = hit.get("score", 0.0)
        desc = (hit.get("description") or "")[:120]
        print(f"\n--- {i}. {api_id} (score={score:.4f}) ---")
        if desc:
            print(desc)
        if show_docs:
            doc = retriever.get_doc(hit["app"], hit["api"])
            content = (doc.get("content") or "").strip()
            if content:
                preview = content[:600] + ("..." if len(content) > 600 else "")
                print(preview)
    return 0


def cmd_search_code_examples(query: str, top_k: int) -> int:
    hits = code_store.search(query, top_k=top_k)
    if not hits:
        print("(no matches in code_examples)")
        return 0
    for i, row in enumerate(hits, 1):
        print(f"\n--- match {i} (id={row.get('id', '')}) ---")
        print(f"task: {row['task_description'][:200]}")
        print(f"apis: {row.get('apis_used', [])}")
        print(f"code preview: {(row.get('working_code') or '')[:400]}")
    if "--json" in sys.argv:
        print(json.dumps(hits, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flywheel Chroma memory utilities")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Override FLYWHEEL_MEMORY_DIR for this command",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index-api-docs", help="Index api_docs_dump into api_docs")
    p_index.add_argument(
        "--force",
        action="store_true",
        help="Delete existing api_docs entries and re-index",
    )

    p_api = sub.add_parser(
        "search-api-docs",
        help="Semantic search over indexed API docs (local Chroma)",
    )
    p_api.add_argument("query", help="Task description to search for")
    p_api.add_argument("--top-k", type=int, default=20)
    p_api.add_argument(
        "--docs",
        action="store_true",
        help="Print a short preview of each API doc body from api_docs_dump",
    )

    p_search = sub.add_parser(
        "search-code-examples",
        help="Semantic search over successful solutions",
    )
    p_search.add_argument("query", help="Task description to search for")
    p_search.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    if args.memory_dir:
        import os

        os.environ["FLYWHEEL_MEMORY_DIR"] = args.memory_dir

    print(f"memory dir: {memory_dir()}", file=sys.stderr)

    if args.command == "index-api-docs":
        return cmd_index_api_docs(args.force)
    if args.command == "search-api-docs":
        return cmd_search_api_docs(args.query, args.top_k, args.docs)
    if args.command == "search-code-examples":
        return cmd_search_code_examples(args.query, args.top_k)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
