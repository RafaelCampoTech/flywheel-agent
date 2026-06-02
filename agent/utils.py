"""Shared JSON parsing helpers used across modules."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from json_repair import repair_json

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating drift."""
    if not text:
        return {}
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        obj = json.loads(repair_json(text))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        obj = json.loads(repair_json(m.group(0)))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def model_content(data: Any) -> str:
    """Extract text content from an OpenAI-style response dict."""
    if not isinstance(data, dict) or data.get("error"):
        return ""
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
