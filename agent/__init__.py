"""Agent modules — composed from agent.py via solve(ctx)."""

# Re-export entry points so `from agent import solve` works
from agent._solve import solve, solve_production_task, solve_test_task  # noqa: F401

from agent.bootstrap import bootstrap
from agent.codegen import generate_adapt, generate_repair, generate_scratch
from agent.evaluation import evaluate, extract_answer_from_stdout, has_error, is_complete
from agent.execution import call_api, complete_task, fetch_api_doc, run_code
from agent.kv_memory import recall, remember
from agent.planner import plan
from agent.rag import format_docs, retrieve
from agent.repair import repair_loop
from agent.timer import Timer

__all__ = [
    "bootstrap",
    "evaluate",
    "evaluate",
    "extract_answer_from_stdout",
    "format_docs",
    "generate_adapt",
    "generate_repair",
    "generate_scratch",
    "has_error",
    "is_complete",
    "call_api",
    "complete_task",
    "fetch_api_doc",
    "run_code",
    "recall",
    "remember",
    "plan",
    "repair_loop",
    "retrieve",
    "Timer",
]
