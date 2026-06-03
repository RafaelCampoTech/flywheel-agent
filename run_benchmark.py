"""Single-shot benchmark: run every available local task with one attempt and no oracle gate.

Purpose
-------
Measure raw single-attempt accuracy and populate the skill DB with working solutions as fast
as possible. Unlike run_local.py, this:
  - Targets all local tasks by default (not a capped --n)
  - Uses exactly 1 check_loop attempt and 0 replan iterations per task
  - Disables the oracle gate so the agent never reads oracle feedback during solve()
  - Still evaluates each task with world.evaluate() after submission (for metrics only)
  - Shares memory across tasks (memory ON by default) so each success immediately enriches
    the skill DB and benefits later tasks in the same run

Usage
-----
  python run_benchmark.py                    # all local tasks, memory ON
  python run_benchmark.py --n 20             # first 20 tasks
  python run_benchmark.py --memory-off       # isolated memory per task (clean baseline)
  python run_benchmark.py --memory-dir /tmp/m  # custom memory directory
  python run_benchmark.py --verbose          # print oracle failure details per task
"""

import os

# Set BEFORE importing agent so module-level constants (_CHECK_LOOP_ATTEMPTS,
# MAX_REPLAN_ITERATIONS) are initialised with benchmark values.
os.environ["FLYWHEEL_CHECK_ATTEMPTS"] = "1"
os.environ["FLYWHEEL_REPLAN_ITERATIONS"] = "0"

import argparse
import json
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flywheel.appworld_env import AppWorldEnv, load_task_ids  # noqa: E402
from flywheel.ctx import Ctx  # noqa: E402
from agent import solve  # noqa: E402

PROXY_URL = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"


def _api_docs_retriever():
    try:
        from tools.rag import RAG_RETRIEVE_K, make_retriever
        return make_retriever(top_k=1, retrieve_k=RAG_RETRIEVE_K, use_llm_rerank=True)
    except FileNotFoundError:
        return None


def _all_task_ids(n=None):
    local = os.path.join(os.path.dirname(__file__), "substrate", "splits", "practice.txt")
    if os.path.exists(local):
        ids = [line.strip() for line in open(local) if line.strip()]
    else:
        ids = load_task_ids("dev")
    return ids if n is None else ids[:n]


def _read_trace(trace_path):
    events = []
    try:
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return events


def run_one(tid, key, memory_dir, verbose=False):
    trace_fd, trace_path = tempfile.mkstemp(suffix=".jsonl", prefix=f"trace_{tid}_")
    os.close(trace_fd)

    env = AppWorldEnv(tid, experiment_name="run_benchmark")
    ctx = Ctx(
        instruction=env.instruction,
        proxy_url=PROXY_URL,
        key=key,
        memory_dir=memory_dir,
        trace_file=trace_path,
        env=env,
        retriever=_api_docs_retriever(),
    )
    # Disable oracle gate: the agent gets no oracle feedback during solve().
    # The oracle is still called below (after solve) for external metrics only.
    ctx._no_oracle_gate = True

    try:
        solve(ctx)
    except NotImplementedError:
        env.close()
        os.unlink(trace_path)
        return None, None, []
    except Exception:
        traceback.print_exc()

    verdict = env.world.evaluate().to_dict()
    events = _read_trace(trace_path)
    env.close()
    os.unlink(trace_path)
    return bool(verdict.get("success")), verdict, events


def main():
    ap = argparse.ArgumentParser(
        description="Single-shot benchmark: 1 attempt, no oracle gate, all local tasks."
    )
    ap.add_argument("--n", type=int, default=None,
                    help="Limit to first N tasks (default: all available)")
    ap.add_argument("--memory-off", action="store_true",
                    help="Wipe memory between tasks — measures baseline with no skill reuse")
    ap.add_argument("--memory-dir", default=None,
                    help="Override memory directory (default: memory/ inside the repo)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print oracle failure details per task")
    a = ap.parse_args()

    key = os.environ.get("FLYWHEEL_KEY", "")
    if not key:
        print("warning: FLYWHEEL_KEY not set; ctx.model calls will fail.", file=sys.stderr)
    os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))

    repo_memory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
    if a.memory_off:
        mem_root = tempfile.mkdtemp(prefix="fw_bench_off_")
        _cleanup_tmp = mem_root
    else:
        mem_root = a.memory_dir or repo_memory
        os.makedirs(mem_root, exist_ok=True)
        _cleanup_tmp = None

    ids = _all_task_ids(a.n)
    print(
        f"benchmark  tasks={len(ids)}  attempts=1  replan=0  oracle_gate=OFF"
        f"  memory={'OFF (isolated)' if a.memory_off else mem_root}"
    )
    print(f"tasks: {ids}\n")

    results = []
    for tid in ids:
        if a.memory_off:
            task_mem = tempfile.mkdtemp(prefix=f"fw_bench_{tid}_")
        else:
            task_mem = mem_root

        ok, verdict, events = run_one(tid, key, task_mem, verbose=a.verbose)

        if a.memory_off:
            shutil.rmtree(task_mem, ignore_errors=True)

        if ok is None:
            print("agent.py is still the skeleton (NotImplementedError). Implement solve(ctx).")
            return

        results.append((tid, ok))
        status = "PASS" if ok else "FAIL"
        model_calls = sum(1 for e in events if e.get("type") == "model")
        print(f"  {tid:14s}  {status}  (model={model_calls})")

        if not ok or a.verbose:
            failures = (verdict or {}).get("failures") or []
            for f in failures[:3]:
                if isinstance(f, dict):
                    msg = (f.get("requirement") or f.get("trace") or json.dumps(f))[:200]
                else:
                    msg = str(f)[:200]
                print(f"    - {msg}")

    if _cleanup_tmp:
        shutil.rmtree(_cleanup_tmp, ignore_errors=True)

    passed = sum(1 for _, ok in results if ok)
    arm = "memory off (baseline)" if a.memory_off else "memory on -> commit memory/ to ship skills"
    print(f"\nTGC: {passed}/{len(results)}  ({arm})")


if __name__ == "__main__":
    main()
