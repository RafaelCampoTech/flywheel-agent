"""Paths under FLYWHEEL_MEMORY_DIR for Chroma and pending-solution metadata."""

from __future__ import annotations

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MEMORY_DIR = os.path.join(_REPO_ROOT, ".memory")
_DEFAULT_DUMP_DIR = os.path.join(_REPO_ROOT, "api_docs_dump")


def memory_dir() -> str:
    mem = os.environ.get("FLYWHEEL_MEMORY_DIR", _DEFAULT_MEMORY_DIR)
    os.makedirs(mem, exist_ok=True)
    return mem


def chroma_path() -> str:
    path = os.path.join(memory_dir(), "chroma_db")
    os.makedirs(path, exist_ok=True)
    return path


def api_docs_dump_dir() -> str:
    return os.environ.get("API_DOCS_DUMP_DIR", _DEFAULT_DUMP_DIR)


def pending_solution_path() -> str:
    return os.path.join(memory_dir(), "pending_solution.json")
