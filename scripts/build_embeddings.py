"""Generate and store cosine-similarity embeddings for all API docs.

Usage (run once from the project root):
    python scripts/build_embeddings.py

Outputs:
    api_docs_dump/embeddings.npz  — {ids: [...], embeddings: float32 (N, 384)}

The model used is all-MiniLM-L6-v2, loaded from the locally cached copy in
api_docs_dump/model/ so no network access is needed.

After running this, agent/rag.py will automatically switch from word-frequency
scoring to cosine similarity retrieval.
"""

import os
import sys
import time

import numpy as np

# Allow running from project root or scripts/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

_APIS_DIR = os.path.join(_ROOT, "api_docs_dump", "apis")
_MODEL_CACHE = os.path.join(_ROOT, "api_docs_dump", "model")
_OUT = os.path.join(_ROOT, "api_docs_dump", "embeddings.npz")
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_docs() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for fname in sorted(os.listdir(_APIS_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(_APIS_DIR, fname)
        api_id = fname[:-4].replace("__", ".", 1)  # e.g. spotify__login -> spotify.login
        try:
            text = open(path, encoding="utf-8").read().strip()
        except OSError:
            continue
        if text:
            ids.append(api_id)
            texts.append(text)
    return ids, texts


def main():
    print(f"Loading docs from {_APIS_DIR} ...")
    ids, texts = load_docs()
    print(f"  {len(ids)} API docs found")

    print(f"Loading model from cache: {_MODEL_CACHE}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_MODEL_NAME, cache_folder=_MODEL_CACHE)
    print(f"  Model loaded: {_MODEL_NAME}  dim={model.get_sentence_embedding_dimension()}")

    print("Encoding docs (batch_size=64) ...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → dot product == cosine similarity
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  shape={embeddings.shape}  dtype={embeddings.dtype}")

    np.savez_compressed(_OUT, ids=np.array(ids), embeddings=embeddings.astype(np.float32))
    size_mb = os.path.getsize(_OUT) / 1024 / 1024
    print(f"\nSaved → {_OUT}  ({size_mb:.1f} MB)")
    print("RAG will now use cosine similarity automatically.")


if __name__ == "__main__":
    main()
