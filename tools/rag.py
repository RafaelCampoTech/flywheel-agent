"""Cosine-similarity RAG over API docs + optional LLM rerank.

  python tools/rag.py -q "spotify login access_token"

Retrieves RAG_RETRIEVE_K (default 15) by cosine similarity, reranks with the LLM,
returns RAG_FINAL_K (default 3) best matches.

Requires: python tools/index_generation.py   # builds embeddings/tfidf_index.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.index_generation import (
    INDEX_PKL,
    INDEX_VERSION,
    apis_dir,
    index_path,
    repo_root,
)

# Initial vector retrieval count, then LLM keeps the best RAG_FINAL_K.
RAG_RETRIEVE_K = int(os.environ.get("RAG_RETRIEVE_K", "15"))
RAG_FINAL_K = int(os.environ.get("RAG_FINAL_K", "3"))

__all__ = [
    "ApiDocsRAG",
    "cosine_similarity_search",
    "llm_rerank",
    "retrieve",
    "make_retriever",
    "RAG_RETRIEVE_K",
    "RAG_FINAL_K",
]


def _parse_doc_id(doc_id: str) -> Tuple[str, str]:
    if "__" in doc_id:
        app, api = doc_id.split("__", 1)
        return app, api
    return doc_id, ""


def _parse_json_obj(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def llm_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    final_k: int = RAG_FINAL_K,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Rerank API doc candidates with the fixed model; return top final_k."""
    if not candidates:
        return []
    if len(candidates) <= final_k:
        return candidates[:final_k]

    from flywheel.proxy import chat, content_of

    key = os.environ.get("FLYWHEEL_KEY", "")
    url = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"
    if not key:
        if verbose:
            print("[rag] FLYWHEEL_KEY missing; skipping LLM rerank")
        return candidates[:final_k]

    compact = [
        {
            "index": i,
            "app": c.get("app"),
            "api": c.get("api"),
            "cosine_score": c.get("cosine_similarity", c.get("score")),
            "description": (c.get("description") or "")[:200],
        }
        for i, c in enumerate(candidates)
    ]

    sys_prompt = (
        "You rerank AppWorld API documentation for a user task.\n"
        f"Pick the {final_k} APIs that are MOST useful to answer the query.\n"
        "Prefer: correct app, correct operation (read vs write), login/token if auth needed, "
        "pagination APIs for list tasks.\n"
        "Reply JSON only:\n"
        '{"ranked":[{"app":"spotify","api":"show_song_library","reason":"..."}, ...]}\n'
        f"Return exactly up to {final_k} items, best first. Use only apps/apis from the candidate list."
    )
    user = {"query": query, "candidates": compact}

    data, _ = chat(
        url,
        key,
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    parsed = _parse_json_obj(content_of(data))
    ranked = parsed.get("ranked") or parsed.get("results") or []
    if not isinstance(ranked, list):
        return candidates[:final_k]

  # Map (app, api) -> candidate
    by_key = {(c.get("app"), c.get("api")): c for c in candidates}
    out: List[Dict[str, Any]] = []
    seen: set = set()

    for item in ranked:
        if not isinstance(item, dict):
            continue
        app = item.get("app", "")
        api = item.get("api", "")
        key_pair = (app, api)
        if key_pair in seen or key_pair not in by_key:
            continue
        seen.add(key_pair)
        hit = dict(by_key[key_pair])
        hit["rerank_reason"] = item.get("reason", "")
        hit["llm_rank"] = len(out) + 1
        out.append(hit)
        if len(out) >= final_k:
            break

    # Fill remaining slots from cosine order if LLM returned too few
    if len(out) < final_k:
        for c in candidates:
            key_pair = (c.get("app"), c.get("api"))
            if key_pair in seen:
                continue
            out.append(c)
            seen.add(key_pair)
            if len(out) >= final_k:
                break

    if verbose:
        print(f"[rag] LLM rerank: {len(candidates)} -> {len(out)}")
        for h in out:
            print(
                f"  {h.get('llm_rank')}. {h.get('app')}.{h.get('api')} "
                f"cosine={h.get('cosine_similarity')} reason={h.get('rerank_reason', '')[:60]}"
            )

    return out


class ApiDocsRAG:
    """Load TF-IDF vectors from embeddings/ and rank docs by cosine similarity to a query."""

    def __init__(
        self,
        index_file: Optional[Path] = None,
        apis_root: Optional[Path] = None,
        root: Optional[Path] = None,
    ):
        self._root = root or repo_root()
        self._index_file = index_file or index_path(self._root)
        self._apis_root = apis_root or apis_dir(self._root)

        if not self._index_file.is_file():
            raise FileNotFoundError(
                f"Missing index {self._index_file}. Run: python tools/index_generation.py"
            )

        with open(self._index_file, "rb") as f:
            data = pickle.load(f)

        if data.get("version") != INDEX_VERSION:
            raise ValueError(
                f"Unsupported index version {data.get('version')}; rebuild with index_generation."
            )

        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        self._doc_ids: List[str] = data["doc_ids"]
        self._rel_paths: List[str] = data.get("rel_paths", [])

    def _cosine_candidates(
        self,
        query: str,
        retrieve_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).ravel()
        ranked = scores.argsort()[::-1]

        results: List[Dict[str, Any]] = []
        for i in ranked:
            score = float(scores[i])
            if score < min_score:
                continue
            doc_id = self._doc_ids[i]
            app, api = _parse_doc_id(doc_id)
            chunk_path = self._apis_root / f"{doc_id}.txt"
            text = chunk_path.read_text(encoding="utf-8") if chunk_path.is_file() else ""
            results.append(
                {
                    "app": app,
                    "api": api,
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "cosine_similarity": round(score, 4),
                    "description": text.split("\n", 1)[0] if text else "",
                    "text": text[:1200],
                    "path": self._rel_paths[i] if i < len(self._rel_paths) else str(chunk_path),
                }
            )
            if len(results) >= retrieve_k:
                break
        return results

    def cosine_similarity_search(
        self,
        query: str,
        top_k: int = RAG_FINAL_K,
        min_score: float = 0.0,
        *,
        retrieve_k: int = RAG_RETRIEVE_K,
        use_llm_rerank: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve retrieve_k by cosine, optionally LLM-rerank, return top_k."""
        q = (query or "").strip()
        if not q:
            return {"query": query, "results": []}

        candidates = self._cosine_candidates(q, retrieve_k=retrieve_k, min_score=min_score)

        if use_llm_rerank and len(candidates) > top_k:
            final = llm_rerank(q, candidates, final_k=top_k, verbose=verbose)
        else:
            final = candidates[:top_k]

        return {
            "query": q,
            "min_score": min_score,
            "retrieve_k": retrieve_k,
            "final_k": top_k,
            "reranked": use_llm_rerank,
            "results": final,
        }

    search = cosine_similarity_search


def cosine_similarity_search(
    query: str,
    top_k: int = RAG_FINAL_K,
    min_score: float = 0.0,
    *,
    retrieve_k: int = RAG_RETRIEVE_K,
    use_llm_rerank: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """One-shot search (loads index each call)."""
    return ApiDocsRAG().cosine_similarity_search(
        query,
        top_k=top_k,
        min_score=min_score,
        retrieve_k=retrieve_k,
        use_llm_rerank=use_llm_rerank,
        verbose=verbose,
    )


def retrieve(
    query: str,
    top_k: int = RAG_FINAL_K,
    min_score: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for cosine_similarity_search (Ctx retriever hook)."""
    return cosine_similarity_search(query, top_k=top_k, min_score=min_score, **kwargs)


def make_retriever(
    top_k: int = RAG_FINAL_K,
    min_score: float = 0.0,
    **kwargs: Any,
) -> Callable[[str], Dict[str, Any]]:
    """Return a function for Ctx(retriever=...) that reuses one loaded index."""
    rag = ApiDocsRAG()

    def _fn(query: str) -> Dict[str, Any]:
        return rag.cosine_similarity_search(
            query, top_k=top_k, min_score=min_score, **kwargs
        )

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG search with LLM rerank over API docs.")
    parser.add_argument("-q", "--query", required=True, help="Input text to match against docs.")
    parser.add_argument("--top-k", type=int, default=RAG_FINAL_K, help="Final results after rerank.")
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RAG_RETRIEVE_K,
        help="Initial cosine candidates before rerank.",
    )
    parser.add_argument("--min-score", type=float, default=0.14, help="Minimum cosine score.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip LLM rerank.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print rerank details.")
    args = parser.parse_args()
    out = cosine_similarity_search(
        args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        retrieve_k=args.retrieve_k,
        use_llm_rerank=not args.no_rerank,
        verbose=args.verbose,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
