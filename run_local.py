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
import logging
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


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy third-party loggers
    for noisy in ("sentence_transformers", "transformers", "torch", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_retriever():
    try:
        from retriever import load_retriever
        r = load_retriever()
        logging.getLogger("run_local").info("Local RAG retriever ready")
        return r
    except FileNotFoundError:
        logging.getLogger("run_local").warning(
            "RAG index not found — run 'python tools/build_index.py' to build it. "
            "Falling back to search_apis."
        )
        return None
    except ImportError as e:
        logging.getLogger("run_local").warning("RAG unavailable (%s), falling back to search_apis.", e)
        return None


def task_ids(n):
    local = os.path.join(os.path.dirname(__file__), "substrate", "splits", "practice.txt")
    if os.path.exists(local):
        ids = [l.strip() for l in open(local) if l.strip()]
    else:
        ids = load_task_ids("dev")
    return ids[:n]


def run_one(tid, key, memory_dir, retriever=None):
    log = logging.getLogger("run_local")
    env = AppWorldEnv(tid, experiment_name="run_local")
    log.info("Task %s: %s", tid, env.instruction[:120])
    ctx = Ctx(
        instruction=env.instruction,
        proxy_url=PROXY_URL,
        key=key,
        memory_dir=memory_dir,
        trace_file=os.environ.get("FLYWHEEL_TRACE_FILE"),
        env=env,
        retriever=retriever,
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
    _setup_logging()
    log = logging.getLogger("run_local")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--memory-off", action="store_true",
                    help="wipe memory between tasks (the off-arm; self-check your memory gap)")
    a = ap.parse_args()

    key = os.environ.get("FLYWHEEL_KEY", "")
    if not key:
        log.warning("FLYWHEEL_KEY not set; ctx.model calls will fail.")
    os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))

    retriever = _load_retriever()

    mem_root = tempfile.mkdtemp(prefix="fw_mem_")
    ids = task_ids(a.n)
    log.info("Running %d dev tasks (memory %s): %s", len(ids), "OFF" if a.memory_off else "ON", ids)
    print()

    results = []
    for tid in ids:
        if a.memory_off:
            shutil.rmtree(mem_root, ignore_errors=True)
        os.makedirs(mem_root, exist_ok=True)
        ok, verdict = run_one(tid, key, mem_root, retriever=retriever)
        if ok is None:
            print("agent.py not implemented yet (NotImplementedError). Implement solve(ctx) and rerun.")
            return
        results.append((tid, ok))
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {tid:14s}  {status}")
        if not ok:
            print(f"    oracle: {json.dumps(verdict, default=str)}")
        print()

    passed = sum(1 for _, ok in results if ok)
    print(f"TGC: {passed}/{len(results)}  ({'memory off' if a.memory_off else 'memory on'})")


if __name__ == "__main__":
    main()
