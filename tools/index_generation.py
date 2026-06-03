"""Build and query a local TF-IDF index over api_docs_dump/apis/*.txt (no database).

  python tools/index_generation.py              # build index
  python tools/index_generation.py --query "spotify login access_token"

Writes under embeddings/ (commit this folder before push):
  tfidf_index.pkl   vectorizer + sparse matrix + doc metadata (pickle)
  index_meta.json   human-readable summary (doc count, paths)

Search (cosine similarity) lives in tools/rag.py:
  from tools.rag import cosine_similarity_search
  hits = cosine_similarity_search("spotify song library pagination", top_k=8)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

INDEX_VERSION = 1
DEFAULT_APIS_DIR = "api_docs_dump/apis"
DEFAULT_EMBEDDINGS_DIR = "embeddings"
LEGACY_EMBEDDINGS_DIR = "api_docs_dump"  # older builds wrote here
INDEX_PKL = "tfidf_index.pkl"
INDEX_META = "index_meta.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def apis_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / DEFAULT_APIS_DIR


def embeddings_dir(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / DEFAULT_EMBEDDINGS_DIR


def index_path(root: Optional[Path] = None) -> Path:
    """Prefer embeddings/; fall back to legacy api_docs_dump/tfidf_index.pkl."""
    root = root or repo_root()
    primary = embeddings_dir(root) / INDEX_PKL
    if primary.is_file():
        return primary
    legacy = root / LEGACY_EMBEDDINGS_DIR / INDEX_PKL
    if legacy.is_file():
        return legacy
    return primary


def list_doc_paths(apis_root: Path) -> List[Path]:
    if not apis_root.is_dir():
        return []
    return sorted(apis_root.glob("*.txt"))


def load_corpus(paths: List[Path]) -> Tuple[List[str], List[str], List[str]]:
    """Return (doc_ids, texts, rel_paths). doc_id is stem: app__api_name."""
    doc_ids: List[str] = []
    texts: List[str] = []
    rel_paths: List[str] = []
    root = repo_root()
    for p in paths:
        doc_ids.append(p.stem)
        texts.append(p.read_text(encoding="utf-8"))
        try:
            rel_paths.append(str(p.relative_to(root)))
        except ValueError:
            rel_paths.append(str(p))
    return doc_ids, texts, rel_paths


def build_index(
    apis_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Index all .txt API chunks; save pickle + meta JSON."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    root = repo_root()
    apis_root = apis_root or apis_dir(root)
    out_dir = out_dir or embeddings_dir(root)

    paths = list_doc_paths(apis_root)
    if not paths:
        raise SystemExit(
            f"No API docs in {apis_root}. Run: python tools/dump_api_docs.py"
        )

    doc_ids, texts, rel_paths = load_corpus(paths)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)

    payload = {
        "version": INDEX_VERSION,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_ids": doc_ids,
        "rel_paths": rel_paths,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / INDEX_PKL
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)

    meta = {
        "version": INDEX_VERSION,
        "num_docs": len(doc_ids),
        "apis_dir": str(apis_root.relative_to(root) if apis_root.is_relative_to(root) else apis_root),
        "index_file": str(pkl.relative_to(root) if pkl.is_relative_to(root) else pkl),
        "vectorizer": "sklearn.TfidfVectorizer(english, 1-2 grams)",
    }
    meta_path = out_dir / INDEX_META
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Indexed {len(doc_ids)} documents")
    print(f"  apis:  {apis_root}")
    print(f"  index: {pkl}")
    print(f"  meta:  {meta_path}")
    return pkl


def _cli_search(query: str, top_k: int) -> None:
    from tools.rag import cosine_similarity_search

    out = cosine_similarity_search(query, top_k=top_k)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the API docs TF-IDF index.")
    parser.add_argument(
        "--query", "-q", help="Test retrieval with this query instead of rebuilding."
    )
    parser.add_argument("--top-k", type=int, default=8, help="Results for --query (default 8).")
    parser.add_argument(
        "--apis-dir",
        default=None,
        help=f"Override APIs folder (default: {DEFAULT_APIS_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Override output folder (default: {DEFAULT_EMBEDDINGS_DIR}/).",
    )
    args = parser.parse_args()

    root = repo_root()
    if args.apis_dir:
        a_dir = Path(args.apis_dir)
        if not a_dir.is_absolute():
            a_dir = root / a_dir
    else:
        a_dir = apis_dir(root)

    if args.out_dir:
        o_dir = Path(args.out_dir)
        if not o_dir.is_absolute():
            o_dir = root / o_dir
    else:
        o_dir = embeddings_dir(root)

    if args.query:
        # Allow query without rebuilding even if index already exists
        if not index_path(root).is_file():
            build_index(a_dir, o_dir)
        _cli_search(args.query, args.top_k)
        return

    build_index(a_dir, o_dir)


if __name__ == "__main__":
    # Ensure repo root is importable when run as a script
    sys.path.insert(0, str(repo_root()))
    main()
