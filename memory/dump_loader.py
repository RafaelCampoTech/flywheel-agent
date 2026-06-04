"""Load API documentation from api_docs_dump/ (tools/dump_api_docs.py output)."""

from __future__ import annotations

import json
import os
from typing import Iterator

from memory.paths import api_docs_dump_dir


def _apis_dir() -> str:
    return os.path.join(api_docs_dump_dir(), "apis")


def load_doc_text(app: str, api: str) -> str:
    """Read the flat text chunk for one API."""
    fname = f"{app}__{api}.txt"
    path = os.path.join(_apis_dir(), fname)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def load_api_docs_dump() -> Iterator[tuple[str, str, str]]:
    """Yield (app, api_name, document_text) for every API in the dump."""
    apis_dir = _apis_dir()
    if not os.path.isdir(apis_dir):
        return
    for fname in sorted(os.listdir(apis_dir)):
        if not fname.endswith(".txt") or "__" not in fname:
            continue
        app, rest = fname[:-4].split("__", 1)
        api_name = rest
        path = os.path.join(apis_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        if text.strip():
            yield app, api_name, text


def load_description(app: str, api: str) -> str:
    """Short text for indexing: first line / description from JSON if present."""
    json_path = os.path.join(api_docs_dump_dir(), f"{app}.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                doc_map = json.load(f)
            doc = doc_map.get(api)
            if isinstance(doc, dict):
                return doc.get("description") or doc.get("api_name", api)
        except (OSError, json.JSONDecodeError):
            pass
    text = load_doc_text(app, api)
    if not text:
        return f"{app}.{api}"
    first = text.split("\n", 1)[0]
    return first if len(first) < 500 else text[:500]
