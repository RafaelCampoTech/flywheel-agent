"""API doc retrieval: MCP in graded mode, Chroma + dump in local mode."""

from __future__ import annotations

import os

from memory.api_retriever import api_retriever_for_ctx


def retrieve(ctx, task: str, top_k: int = 20) -> dict:
    """Return relevant API docs for the task.

    Graded run (FLYWHEEL_MCP_URL): search_apis + api_doc via MCP.
    Local run: Chroma collection api_docs + api_docs_dump text files.

    Returns:
        {
            "docs":          [{"api": "app.name", "content": str, "score": float}],
            "apps_involved": [str],
            "api_names":     ["app.api_name", ...],
            "method":        "mcp" | "chroma"
        }
    """
    retriever = api_retriever_for_ctx(ctx, top_k=top_k)
    method = "mcp" if os.environ.get("FLYWHEEL_MCP_URL") else "chroma"

    hits = retriever.search(task)
    docs: list[dict] = []
    api_names: list[str] = []
    seen_apps: set[str] = set()

    for hit in hits:
        app = hit["app"]
        api = hit["api"]
        api_id = hit.get("api_id", f"{app}.{api}")
        full = retriever.get_doc(app, api)
        content = full.get("content", "")
        if not content:
            continue
        score = float(hit.get("score", 0.0))
        docs.append({"api": api_id, "content": content, "score": round(score, 4)})
        api_names.append(api_id)
        seen_apps.add(app)

    return {
        "docs": docs,
        "apps_involved": sorted(seen_apps),
        "api_names": api_names,
        "method": method,
    }


def format_docs(rag_result: dict) -> str:
    """Format RAG result into a string for prompt injection."""
    parts: list[str] = []
    for d in rag_result.get("docs", []):
        parts.append(f"--- {d['api']} ---\n{d['content']}")
    return "\n\n".join(parts) if parts else "(no docs retrieved)"
