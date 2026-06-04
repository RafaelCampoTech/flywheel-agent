"""MCP tool wrappers. Every world interaction goes through ctx.mcp.call()."""

from __future__ import annotations

def make_preamble(tokens: dict) -> str:
    return f"tokens = {repr(tokens)}\n"


def run_code(ctx, code: str) -> str:
    """Execute Python code via MCP run_code. Returns stdout string."""
    result = ctx.mcp.call("run_code", {"code": code})
    if isinstance(result, dict):
        return result.get("stdout") or result.get("error") or ""
    return str(result) if result else ""


def complete_task(ctx, answer: str | None = None) -> None:
    """Submit the final answer. Must be called exactly once per task."""
    if answer is not None and str(answer).strip() not in ("", "None"):
        ctx.mcp.call("complete_task", {"answer": str(answer)})
    else:
        ctx.mcp.call("complete_task", {})


def call_api(ctx, app: str, api: str, arguments: dict | None = None) -> dict:
    return ctx.mcp.call("call_api", {"app": app, "api": api, "arguments": arguments or {}})


def fetch_api_doc(ctx, app: str, api: str) -> dict:
    return ctx.mcp.call("api_doc", {"app": app, "api": api})
