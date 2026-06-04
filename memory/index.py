"""Index api_docs_dump into the read-only api_docs Chroma collection."""

from __future__ import annotations

from memory.chroma_client import COLLECTION_API_DOCS, get_collection
from memory.dump_loader import load_api_docs_dump, load_description


def index_api_docs(*, force: bool = False) -> int:
    """Index all APIs from api_docs_dump into collection api_docs. Returns count added."""
    col = get_collection(COLLECTION_API_DOCS)
    if col.count() > 0 and not force:
        return col.count()

    if force and col.count() > 0:
        ids = col.get()["ids"]
        if ids:
            col.delete(ids=ids)

    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for app, api_name, _text in load_api_docs_dump():
        doc_id = f"{app}.{api_name}"
        desc = load_description(app, api_name)
        docs.append(desc)
        metadatas.append({"app": app, "api": api_name, "description": desc[:500]})
        ids.append(doc_id)

    if not ids:
        return 0

    batch = 128
    for i in range(0, len(ids), batch):
        col.add(
            documents=docs[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )
    return len(ids)
