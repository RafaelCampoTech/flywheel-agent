"""Agent pipeline. Imported by agent/__init__.py so `from agent import solve` works."""

import agent.logger as log
from agent.bootstrap import bootstrap
from agent.codeplan import build_code_plan
from agent.codegen import generate_adapt, generate_oracle_repair, generate_repair, generate_scratch
from agent.evaluation import evaluate, extract_answer_from_stdout
from agent.execution import complete_task, make_preamble, run_code
from agent.kv_memory import remember
from memory.code_store import confirm_pending_success
from memory.code_store import discard_pending
from memory.code_store import get_pending_id
from memory.code_store import search as mem_search
from memory.code_store import write_pending
from agent.planner import plan
from agent.rag import format_docs, retrieve
from agent.repair import repair_loop
from agent.timer import Timer


def solve_test_task(ctx):
    solve(ctx)


def solve_production_task(ctx):
    solve(ctx)


def solve(ctx):
    timer = Timer()
    instruction = ctx.instruction
    best_stdout: str = ""

    log.task_start(instruction)

    # --- Step 2: Oracle check for previous task (local only) ---
    # Pending solutions are only written on local runs; graded runs have no pending file.
    if getattr(ctx, "_env", None) is not None:
        log.step(2, "Oracle check")
        try:
            if get_pending_id():
                confirm_pending_success()
        except Exception:
            pass

    # --- Step 3: RAG ---
    log.step(3, "RAG retrieval")
    try:
        rag_result = retrieve(ctx, instruction)
    except Exception as e:
        log.error("RAG", str(e))
        rag_result = {"docs": [], "apps_involved": [], "api_names": []}
    docs_str = format_docs(rag_result)
    apps_from_rag: list[str] = rag_result.get("apps_involved", [])
    api_names: list[str] = rag_result.get("api_names", [])
    top_scored = [(d["api"], d.get("score", 0.0)) for d in rag_result.get("docs", [])[:8]]
    log.rag_result(
        apps_from_rag,
        len(rag_result.get("docs", [])),
        api_names,
        method=rag_result.get("method", "chroma"),
        top_scored=top_scored if rag_result.get("method") in ("cosine", "chroma") else None,
    )

    if timer.at_emergency():
        log.emergency("after RAG")
        complete_task(ctx, None)
        return

    # --- Step 4: Memory similarity search ---
    log.step(4, "Memory search")
    try:
        similar_solutions = mem_search(instruction, top_k=3, rag_apis=api_names)
        similar_solutions = [s for s in similar_solutions if s.get("working_code", "").strip()]
    except Exception as e:
        log.error("Memory search", str(e))
        similar_solutions = []
    log.memory_result(len(similar_solutions))

    # --- Step 5: Structured plan ---
    log.step(5, "Planning")
    try:
        task_plan = plan(ctx, instruction, docs_str, similar_solutions, failure_context=None)
    except Exception as e:
        log.error("Planner", str(e))
        task_plan = {
            "objective": instruction,
            "task_type": "action",
            "apps_involved": apps_from_rag,
            "expected_output_type": "none",
            "steps": [],
        }
    log.plan_result(task_plan)

    apps_involved: list[str] = task_plan.get("apps_involved") or apps_from_rag
    task_type: str = task_plan.get("task_type", "action")
    expected_type: str = task_plan.get("expected_output_type", "none")

    if timer.at_emergency():
        log.emergency("after planning")
        complete_task(ctx, None)
        return

    # --- Step 6: Bootstrap tokens ---
    log.step(6, "Token bootstrap")
    try:
        tokens = bootstrap(ctx, apps_involved)
    except Exception as e:
        log.error("Bootstrap", str(e))
        tokens = {}
    log.tokens_result(tokens)

    # --- Step 7: Code generation ---
    # Code planning (2 LLM calls) only runs on the scratch path — adapt already has a template.
    log.step(7, "Code generation")
    code_plan_str = ""
    try:
        if similar_solutions:
            code = generate_adapt(ctx, task_plan, docs_str, similar_solutions[0])
        else:
            log.step("6.5", "Code planning")
            try:
                cp_result = build_code_plan(ctx, task_plan, docs_str, similar_solutions)
                code_plan_str = cp_result["code_plan"]
            except Exception as e:
                log.error("Code planning", str(e))
            if timer.at_emergency():
                log.emergency("after code planning")
                complete_task(ctx, None)
                return
            code = generate_scratch(ctx, task_plan, docs_str, code_plan=code_plan_str)
    except Exception as e:
        log.error("Codegen", str(e))
        code = f"# codegen failed: {e}\nprint('ANSWER=error')"

    # --- Step 8: Execute ---
    log.step(8, "Execute")
    preamble = make_preamble(tokens)
    try:
        stdout = run_code(ctx, preamble + code)
    except Exception as e:
        log.error("Execute", str(e))
        stdout = f"Error: {e}"
    best_stdout = stdout
    log.exec_result(stdout)

    if timer.at_emergency():
        log.emergency("after first execute")
        answer = extract_answer_from_stdout(best_stdout)
        log.finish(answer, timer.elapsed())
        complete_task(ctx, answer)
        _persist(instruction, task_plan, code, api_names)
        return

    # --- Step 9: Evaluate ---
    log.step(9, "Evaluation")
    eval_result = evaluate(stdout, task_type, expected_type)
    if eval_result["passed"]:
        log.eval_pass(eval_result["actual_output"], expected_type)
    else:
        log.eval_fail(eval_result["failure_reason"], eval_result["actual_output"], expected_type)

    # --- Step 10: Repair loop ---
    # Cap at 2 attempts in test mode to reserve budget for oracle repair (Step 13).
    _is_local = getattr(ctx, "_env", None) is not None
    if not eval_result["passed"]:
        log.step(10, "Repair loop")
        fixed_code, repair_stdout, last_failure = repair_loop(
            ctx,
            task_plan,
            docs_str,
            tokens,
            failed_code=code,
            first_error=stdout,
            timer=timer,
            max_attempts=2 if _is_local else 3,
        )

        if fixed_code is not None:
            best_stdout = repair_stdout
            code = fixed_code
        else:
            # --- Step 11: Re-plan once ---
            if last_failure and timer.should_replan():
                log.step(11, "Re-plan")
                log.replan()
                try:
                    task_plan = plan(
                        ctx,
                        instruction,
                        docs_str,
                        similar_solutions,
                        failure_context=last_failure,
                    )
                    apps_involved = task_plan.get("apps_involved") or apps_from_rag
                    task_type = task_plan.get("task_type", task_type)
                    expected_type = task_plan.get("expected_output_type", expected_type)
                    tokens = bootstrap(ctx, apps_involved)
                    log.plan_result(task_plan)
                    log.tokens_result(tokens)
                except Exception as e:
                    log.error("Re-plan", str(e))

                if timer.at_emergency():
                    log.emergency("after re-plan")
                    answer = extract_answer_from_stdout(best_stdout)
                    log.finish(answer, timer.elapsed())
                    complete_task(ctx, answer)
                    _persist(instruction, task_plan, code, api_names)
                    return

                try:
                    code = generate_scratch(ctx, task_plan, docs_str)
                    preamble = make_preamble(tokens)
                    stdout = run_code(ctx, preamble + code)
                    best_stdout = stdout
                    log.exec_result(stdout)
                except Exception as e:
                    log.error("Replan execute", str(e))
                    stdout = f"Error: {e}"

                eval2 = evaluate(stdout, task_type, expected_type)
                if eval2["passed"]:
                    log.eval_pass(eval2["actual_output"], expected_type)
                else:
                    log.eval_fail(eval2["failure_reason"], eval2["actual_output"], expected_type)
                    if not timer.at_emergency():
                        _, repair_stdout2, _ = repair_loop(
                            ctx,
                            task_plan,
                            docs_str,
                            tokens,
                            failed_code=code,
                            first_error=stdout,
                            timer=timer,
                        )
                        if repair_stdout2:
                            best_stdout = repair_stdout2

    # --- Step 12: Finish ---
    log.step(12, "complete_task")
    if timer.at_hard_kill():
        log.emergency("hard kill — submitting empty answer")
        complete_task(ctx, None)
        log.finish(None, timer.elapsed())
    else:
        answer = extract_answer_from_stdout(best_stdout)
        complete_task(ctx, answer)
        log.finish(answer, timer.elapsed())

    # --- Step 13: Oracle verify + repair loop (local only) ---
    # Up to 3 oracle checks; on each failure feed the oracle diff to generate_repair,
    # re-run, and re-submit. Only persist code that the oracle confirms correct.
    if getattr(ctx, "_env", None) is not None:
        log.step(13, "Oracle verify + persist")
        _oracle_repair_loop(
            ctx, task_plan, docs_str, tokens, code, instruction, api_names
        )

    try:
        remember(ctx, "last_plan", str(task_plan)[:2000])
    except Exception:
        pass


def _persist(instruction: str, task_plan: dict, code: str, api_names: list) -> None:
    if not code or not code.strip():
        return
    try:
        write_pending(
            task_desc=instruction,
            apis_used=api_names,
            plan=task_plan,
            code=code,
        )
    except Exception:
        pass


def _oracle_failure_summary(verdict: dict) -> str:
    """Extract the expected-vs-actual diff from an oracle verdict."""
    failures = verdict.get("failures", [])
    if not failures:
        return "Oracle failed."
    parts = []
    for f in failures[:2]:
        trace = f.get("trace", "")
        if "Original values:" in trace:
            parts.append(trace[trace.index("Original values:"):trace.index("Original values:") + 500])
        elif "AssertionError" in trace:
            parts.append(trace[trace.index("AssertionError"):trace.index("AssertionError") + 400])
        elif trace:
            parts.append(trace[:300])
    return "\n".join(parts) or "Oracle failed."


def _oracle_repair_loop(
    ctx, task_plan: dict, docs_str: str, tokens: dict,
    code: str, instruction: str, api_names: list,
) -> None:
    """Oracle-guided repair: up to 3 checks, 2 repair attempts. Persists only on oracle pass."""
    for attempt in range(3):
        _persist(instruction, task_plan, code, api_names)
        try:
            verdict = ctx.evaluate_full()
        except Exception as e:
            log.error("Oracle verify", str(e))
            discard_pending()
            return

        passed = verdict.get("success", False)
        log.oracle_result(passed)

        if passed:
            confirm_pending_success()
            return

        discard_pending()

        if attempt >= 2:
            return

        # Oracle-guided repair: lightweight prompt with expected-vs-actual diff only.
        # Intentionally avoids full docs to stay within the token budget.
        oracle_error = _oracle_failure_summary(verdict)
        log.info("Oracle repair", f"attempt {attempt + 1}/2 — {oracle_error[:120]}")
        try:
            code = generate_oracle_repair(ctx, task_plan, code, oracle_error)
            if not code.strip():
                return
            preamble = make_preamble(tokens)
            stdout = run_code(ctx, preamble + code)
            new_answer = extract_answer_from_stdout(stdout)
            log.exec_result(stdout)
            complete_task(ctx, new_answer)
        except Exception as e:
            log.error("Oracle repair", str(e))
            return
