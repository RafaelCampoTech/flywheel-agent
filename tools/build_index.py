"""Pre-compute TF-IDF index for all 457 API doc .txt files.

Run once locally:
    python tools/build_index.py

Writes:
    api_docs_dump/tfidf_index.pkl   TfidfVectorizer + sparse doc matrix
    api_docs_dump/index_meta.json   list of {app, api, file, text}
"""
import json
import pickle
import sys
from pathlib import Path

APIS_DIR = Path("api_docs_dump/apis")
OUT_DIR = Path("api_docs_dump")


def main():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        sys.exit("pip install scikit-learn")

    files = sorted(APIS_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {APIS_DIR}. Run tools/dump_api_docs.py first.")

    print(f"Found {len(files)} API docs in {APIS_DIR}")

    docs = []
    texts = []
    for f in files:
        stem = f.stem  # e.g. "spotify__login"
        parts = stem.split("__", 1)
        app = parts[0]
        api = parts[1] if len(parts) > 1 else stem
        text = f.read_text().strip()
        docs.append({"app": app, "api": api, "file": f.name, "text": text})
        texts.append(f"{app} {api.replace('_', ' ')}\n{text}")

    print("Fitting TF-IDF vectorizer (ngram 1-2, sublinear_tf)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    doc_matrix = vectorizer.fit_transform(texts)

    OUT_DIR.mkdir(exist_ok=True)
    pkl_path = OUT_DIR / "tfidf_index.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"vectorizer": vectorizer, "doc_matrix": doc_matrix}, fh)

    meta_path = OUT_DIR / "index_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(docs, fh, indent=2)

    print(f"\nDone.")
    print(f"  tfidf_index.pkl  vocab={len(vectorizer.vocabulary_)}  docs={doc_matrix.shape[0]}")
    print(f"  index_meta.json  {len(docs)} entries")
    print(f"\nCommit api_docs_dump/tfidf_index.pkl and index_meta.json for offline sandbox use.")

