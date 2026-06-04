"""API doc retrieval: MCP in graded mode, Chroma + dump in local mode."""

from __future__ import annotations

import os
from typing import Any, Protocol

from memory.chroma_client import COLLECTION_API_DOCS, get_collection
from memory.dump_loader import load_api_docs_dump, load_description, load_doc_text


class MCPClient(Protocol):
    def call(self, name: str, args: dict | None = None) -> Any: ...


class MCPApiRetriever:
    """Used in graded mode. Delegates to search_apis and api_doc MCP tools."""

    def __init__(self, mcp: MCPClient, top_k: int = 20):
        self._mcp = mcp
        self._top_k = top_k

    def search(self, query: str) -> list[dict]:
        raw = self._mcp.call("search_apis", {"query": query})
        if isinstance(raw, dict):
            items = raw.get("results", raw.get("hits", []))
        else:
            items = raw or []
        out: list[dict] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            app = item.get("app", "")
            api = item.get("api", "")
            if not app or not api:
                continue
            key = f"{app}.{api}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "app": app,
                    "api": api,
                    "api_id": key,
                    "description": item.get("description", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )
            if len(out) >= self._top_k:
                break
        return out

    def get_doc(self, app: str, api: str) -> dict:
        raw = self._mcp.call("api_doc", {"app": app, "api": api})
        if isinstance(raw, dict):
            doc = raw.get("doc", raw)
        else:
            doc = raw
        content = _format_mcp_doc(app, api, doc)
        return {"app": app, "api": api, "api_id": f"{app}.{api}", "content": content}


class LocalApiRetriever:
    """Used in local mode. Queries ChromaDB api_docs; full text from dump."""

    def __init__(self, top_k: int = 20, bootstrap: bool = True):
        self._top_k = top_k
        self.collection = get_collection(COLLECTION_API_DOCS)
        if bootstrap:
            self._bootstrap_if_empty()

    def search(self, query: str) -> list[dict]:
        if self.collection.count() == 0:
            self._bootstrap_if_empty()
        if self.collection.count() == 0:
            return []
        hits = self.collection.query(query_texts=[query], n_results=self._top_k)
        return self._format_hits(hits)

    def get_doc(self, app: str, api: str) -> dict:
        content = load_doc_text(app, api)
        return {"app": app, "api": api, "api_id": f"{app}.{api}", "content": content}

    def _bootstrap_if_empty(self) -> None:
        if self.collection.count() > 0:
            return
        from memory.index import index_api_docs

        index_api_docs(force=False)

    @staticmethod
    def _format_hits(hits: dict) -> list[dict]:
        metas = hits.get("metadatas") or [[]]
        distances = hits.get("distances") or [[]]
        if not metas or not metas[0]:
            return []
        results: list[dict] = []
        for i, meta in enumerate(metas[0]):
            app = meta.get("app", "")
            api = meta.get("api", "")
            if not app or not api:
                continue
            dist = distances[0][i] if distances and distances[0] else None
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            results.append(
                {
                    "app": app,
                    "api": api,
                    "api_id": f"{app}.{api}",
                    "description": meta.get("description", ""),
                    "score": round(score, 4),
                }
            )
        return results


def _format_mcp_doc(app: str, api: str, doc: Any) -> str:
    if doc is None:
        return ""
    if isinstance(doc, str):
        return doc
    if not isinstance(doc, dict):
        return str(doc)
    lines = [
        f"{doc.get('app_name', app)}.{doc.get('api_name', api)}  "
        f"[{doc.get('method', '?')} {doc.get('path', '')}]",
        doc.get("description", ""),
    ]
    for p in doc.get("parameters", []):
        req = "required" if p.get("required") else "optional"
        lines.append(
            f"  - {p['name']} ({p.get('type', '?')}, {req}): {p.get('description', '')}"
        )
    schema = (doc.get("response_schemas") or {}).get("success")
    if schema is not None:
        import json

        lines.append("  response: " + json.dumps(schema)[:600])
    return "\n".join(lines)


def api_retriever_for_ctx(ctx, top_k: int = 20):
    """Pick MCP vs local retriever from environment / ctx backend."""
    if os.environ.get("FLYWHEEL_MCP_URL"):
        return MCPApiRetriever(ctx.mcp, top_k=top_k)
    return LocalApiRetriever(top_k=top_k)
