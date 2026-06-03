"""Run YOUR solve(ctx) on N real AppWorld dev tasks locally and grade each on AppWorld's own
deterministic oracle. This is the loop you tune against before you submit.

  python run_local.py --n 5                 # 5 dev tasks, memory persists across them (the on-arm)
  python run_local.py --n 5 --memory-off    # wipe memory between tasks (the off-arm)

Run both and compare TGC: the gap IS your memory grade. If --memory-off matches --n with the
same score, your memory isn't doing anything yet. Per task it prints pass/fail; at the end, the
TGC (task goal completion = tasks passed / tasks run).

Task ids come from substrate/splits/practice.txt if present, else load_task_ids('dev').
Set APPWORLD_ROOT (see .env.example); FLYWHEEL_KEY must be set for ctx.model to work.
"""
import argparse
import json
import os
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
    """TF-IDF over api_docs_dump/apis if index exists; else default search_apis."""
    try:
        from tools.rag import RAG_RETRIEVE_K, make_retriever

        return make_retriever(
            top_k=1,
            retrieve_k=RAG_RETRIEVE_K,
            use_llm_rerank=True,
        )
    except FileNotFoundError:
        return None


def task_ids(n):
    local = os.path.join(os.path.dirname(__file__), "substrate", "splits", "practice.txt")
    if os.path.exists(local):
        ids = [l.strip() for l in open(local) if l.strip()]
    else:
        ids = load_task_ids("dev")
    return ids[:n]


def run_one(tid, key, memory_dir):
    env = AppWorldEnv(tid, experiment_name="run_local")
    ctx = Ctx(
        instruction=env.instruction,
        proxy_url=PROXY_URL,
        key=key,
        memory_dir=memory_dir,
        trace_file=os.environ.get("FLYWHEEL_TRACE_FILE"),
        env=env,
        retriever=_api_docs_retriever(),
    )
    try:
        solve(ctx)
    except NotImplementedError:
        env.close()
        return None, None  # skeleton not implemented yet
    except Exception:
        traceback.print_exc()
    verdict = env.world.evaluate().to_dict()
    env.close()
    return bool(verdict.get("success")), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--memory-off", action="store_true",
                    help="wipe memory between tasks (the off-arm; self-check your memory gap)")
    a = ap.parse_args()

    key = os.environ.get("FLYWHEEL_KEY", "")
    if not key:
        print("warning: FLYWHEEL_KEY not set; ctx.model calls will fail.", file=sys.stderr)
    os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))

    mem_root = tempfile.mkdtemp(prefix="fw_mem_")
    ids = task_ids(a.n)
    print(f"running {len(ids)} dev tasks (memory {'OFF' if a.memory_off else 'ON'}): {ids}\n")

    results = []
    for tid in ids:
        if a.memory_off:
            shutil.rmtree(mem_root, ignore_errors=True)
        os.makedirs(mem_root, exist_ok=True)
        ok, verdict = run_one(tid, key, mem_root)
        if ok is None:
            print("agent.py is still the skeleton (NotImplementedError). Implement solve(ctx), then rerun.")
            return
        results.append((tid, ok))
        print(f"  {tid:14s} {'PASS' if ok else 'FAIL'}")
        if not ok:  # oracle verdict so the loop is learnable: what failed and why
            print(f"    oracle: {json.dumps(verdict, default=str)}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\nTGC: {passed}/{len(results)}  ({'memory off' if a.memory_off else 'memory on'})")


if __name__ == "__main__":
    main()
