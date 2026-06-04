"""Thin shim — delegates to agent/_solve.py so both import paths work.

The harness may do either:
  from agent import solve          (picks agent/ package → agent/__init__.py)
  from agent import solve          (picks agent.py if package not found)

Either way solve() is available.
"""

from agent._solve import solve, solve_production_task, solve_test_task  # noqa: F401

__all__ = ["solve", "solve_test_task", "solve_production_task"]
