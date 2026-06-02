"""TF-IDF retriever over the 457 pre-computed API doc vectors.

Usage:
    from retriever import load_retriever
    retrieve = load_retriever()
    docs = retrieve("spotify login token")

Build the index first (once):
    python tools/build_index.py
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)

_INDEX_DIR = Path(__file__).parent / "api_docs_dump"
_TOP_K = 8


def load_retriever(index_dir: str | None = None, top_k: int = _TOP_K):
    """Return a retrieve(query) -> list[dict] callable backed by TF-IDF cosine similarity."""
    idx_dir = Path(index_dir or _INDEX_DIR)
    pkl_path = idx_dir / "tfidf_index.pkl"
    meta_path = idx_dir / "index_meta.json"

    if not pkl_path.exists() or not meta_path.exists():
        log.warning("RAG index missing at %s — building now (runs once).", idx_dir)
        import subprocess, sys
        build_script = Path(__file__).parent / "tools" / "build_index.py"
        result = subprocess.run([sys.executable, str(build_script)], check=False)
        if result.returncode != 0 or not pkl_path.exists():
            raise FileNotFoundError(
                "RAG index missing and auto-build failed. "
                "Run manually: python tools/build_index.py"
            )

    with open(pkl_path, "rb") as fh:
        index = pickle.load(fh)
    vectorizer = index["vectorizer"]
    doc_matrix = index["doc_matrix"]

    with open(meta_path) as fh:
        meta: List[Dict[str, Any]] = json.load(fh)

    log.info("RAG index loaded: %d docs, vocab=%d", len(meta), len(vectorizer.vocabulary_))

    def retrieve(query: str) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, doc_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "app": meta[i]["app"],
                "api": meta[i]["api"],
                "score": round(float(scores[i]), 4),
                "doc": meta[i].get("text", ""),
            }
            for i in top_idx
        ]

    return retrieve
