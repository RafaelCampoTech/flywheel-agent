"""Run YOUR solve(ctx) on N real AppWorld dev tasks locally and grade each on AppWorld's own
deterministic oracle. This is the loop you tune against before you submit.

  python run_local.py --n 5                 # 5 dev tasks, memory persists (the on-arm)
  python run_local.py --n 5 --memory-off    # wipe memory between tasks (the off-arm)
  python run_local.py --n 5 --verbose       # also print oracle.why, tool_calls, agent_log

Run both and compare TGC: the gap IS your memory grade. If --memory-off matches --n with the
same score, your memory isn't doing anything yet. Per task it prints pass/fail; at the end, the
TGC (task goal completion = tasks passed / tasks run).

The four feedback items per task (--verbose shows all):
  passed      — PASS or FAIL (and in how many of the k repetitions)
  oracle.why  — the exact assertion that broke: left (produced) vs right (expected)
  tool_calls  — run_code / call_api calls and any errors they raised
  agent_log   — reflect events: the agent's self-corrections step by step

Task ids come from substrate/splits/practice.txt if present, else load_task_ids('dev').
Set APPWORLD_ROOT (see .env.example); FLYWHEEL_KEY must be set for ctx.model to work.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flywheel.appworld_env import AppWorldEnv, load_task_ids  # noqa: E402
from flywheel.ctx import Ctx  # noqa: E402
from agent import solve  # noqa: E402

PROXY_URL = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"
SOLVE_TIMEOUT_SECONDS = 300  # must match agent.py


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _api_docs_retriever():
    """TF-IDF over api_docs_dump/apis if index exists; else default search_apis."""
    try:
        from tools.rag import RAG_RETRIEVE_K, make_retriever
        return make_retriever(top_k=1, retrieve_k=RAG_RETRIEVE_K, use_llm_rerank=True)
    except FileNotFoundError:
        return None


def task_ids(n):
    local = os.path.join(os.path.dirname(__file__), "substrate", "splits", "practice.txt")
    if os.path.exists(local):
        ids = [l.strip() for l in open(local) if l.strip()]
    else:
        ids = load_task_ids("dev")
    return ids[:n]


def _read_trace(trace_path):
    """Parse a JSONL trace file into a list of event dicts."""
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


def _print_oracle_why(verdict):
    """Print oracle.why: what assertion broke, left vs right."""
    failures = verdict.get("failures") or []
    if not failures:
        print("    oracle.why: (no failures recorded)")
        return
    print("    oracle.why:")
    for f in failures[:5]:
        if isinstance(f, dict):
            req = f.get("requirement") or f.get("trace") or json.dumps(f)[:200]
            print(f"      {req}")
        else:
            print(f"      {str(f)[:200]}")


def _print_tool_calls(events):
    """Print run_code / call_api calls and errors from trace."""
    tool_events = [e for e in events if e.get("type") in ("execute", "tool")]
    if not tool_events:
        print("    tool_calls: (none recorded)")
        return
    print(f"    tool_calls ({len(tool_events)} calls):")
    for e in tool_events[:10]:
        t = e.get("type")
        if t == "execute":
            print("      run_code")
        elif t == "tool":
            print(f"      {e.get('name', 'tool')}")


def _print_agent_log(events):
    """Print reflect events: the agent's self-correction reasoning."""
    reflects = [e for e in events if e.get("type") == "reflect"]
    if not reflects:
        print("    agent_log: (no reflect events)")
        return
    print(f"    agent_log ({len(reflects)} reflect steps):")
    for e in reflects[:8]:
        note = (e.get("note") or "")[:120]
        print(f"      → {note}")


def run_one(tid, key, memory_dir, verbose=False):
    trace_fd, trace_path = tempfile.mkstemp(suffix=".jsonl", prefix=f"trace_{tid}_")
    os.close(trace_fd)

    env = AppWorldEnv(tid, experiment_name="run_local")
    ctx = Ctx(
        instruction=env.instruction,
        proxy_url=PROXY_URL,
        key=key,
        memory_dir=memory_dir,
        trace_file=trace_path,  # always capture trace for feedback loop
        env=env,
        retriever=_api_docs_retriever(),
    )
    t0 = time.time()
    try:
        solve(ctx)
    except NotImplementedError:
        env.close()
        os.unlink(trace_path)
        return None, None, [], 0.0
    except Exception:
        traceback.print_exc()
    elapsed = time.time() - t0

    verdict = env.world.evaluate().to_dict()
    events = _read_trace(trace_path)
    env.close()
    os.unlink(trace_path)
    return bool(verdict.get("success")), verdict, events, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--memory-off", action="store_true",
                    help="run with isolated tmpdir memory (off-arm baseline; does not touch memory/)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print oracle.why, tool_calls, and agent_log per task")
    ap.add_argument("--memory-dir", default=None,
                    help="override memory directory (default: memory/ inside the repo)")
    a = ap.parse_args()

    key = os.environ.get("FLYWHEEL_KEY", "")
    if not key:
        print("warning: FLYWHEEL_KEY not set; ctx.model calls will fail.", file=sys.stderr)
    os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))

    # Default: write to memory/ in the repo so skills persist and can be committed.
    # --memory-off: isolated tmpdir, discarded after run (no effect on committed memory).
    repo_memory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
    if a.memory_off:
        mem_root = tempfile.mkdtemp(prefix="fw_mem_off_")
        _cleanup_tmp = mem_root
    else:
        mem_root = a.memory_dir or repo_memory
        os.makedirs(mem_root, exist_ok=True)
        _cleanup_tmp = None

    ids = task_ids(a.n)
    print(f"running {len(ids)} dev tasks  memory={'OFF (tmpdir)' if a.memory_off else mem_root}")
    print(f"tasks: {ids}\n")

    results = []
    total_start = time.time()
    for tid in ids:
        if a.memory_off:
            # Fresh tmpdir per task for true off-arm isolation
            task_mem = tempfile.mkdtemp(prefix=f"fw_mem_off_{tid}_")
        else:
            task_mem = mem_root

        ok, verdict, events, elapsed = run_one(tid, key, task_mem, verbose=a.verbose)

        if a.memory_off:
            shutil.rmtree(task_mem, ignore_errors=True)

        if ok is None:
            print("agent.py is still the skeleton (NotImplementedError). Implement solve(ctx), then rerun.")
            return

        results.append((tid, ok, elapsed))
        status = "PASS" if ok else "FAIL"
        model_calls = sum(1 for e in events if e.get("type") == "model")
        reflects = sum(1 for e in events if e.get("type") == "reflect")
        timeout_warn = "  ⚠ near timeout" if elapsed > 240 else ""
        print(f"  {tid:14s}  {status}  {elapsed:6.1f}s  (model={model_calls} reflect={reflects}){timeout_warn}")

        if not ok or a.verbose:
            _print_oracle_why(verdict)
            if a.verbose:
                _print_tool_calls(events)
                _print_agent_log(events)

    if _cleanup_tmp:
        shutil.rmtree(_cleanup_tmp, ignore_errors=True)

    passed = sum(1 for _, ok, _ in results if ok)
    total_elapsed = time.time() - total_start
    times = [t for _, _, t in results]
    avg = sum(times) / len(times) if times else 0.0
    slowest = max(times) if times else 0.0
    arm = "memory off" if a.memory_off else f"memory on  →  commit memory/ to ship skills"
    print(f"\nTGC: {passed}/{len(results)}  ({arm})")
    print(f"Time: total={_fmt(total_elapsed)}  avg={avg:.1f}s/task  slowest={slowest:.1f}s  (timeout={SOLVE_TIMEOUT_SECONDS}s)")


if __name__ == "__main__":
    main()
