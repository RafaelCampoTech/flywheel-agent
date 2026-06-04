"""ChromaDB persistent client and collection names."""

from __future__ import annotations

import chromadb

from memory.paths import chroma_path

COLLECTION_API_DOCS = "api_docs"
COLLECTION_CODE_EXAMPLES = "code_examples"

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=chroma_path())
    return _client


def get_collection(name: str):
    return get_client().get_or_create_collection(name)
