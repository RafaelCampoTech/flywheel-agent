# AGENTS.md — Flywheel Agent Specification

## Purpose

This document describes the full architecture of a task-solving agent for the Flywheel challenge.
The agent operates inside AppWorld: a simulated digital environment with 9 apps (Spotify, Amazon,
Gmail, Phone, Venmo, Splitwise, Todoist, SimpleNote, FileSystem) exposed as 457 Python APIs.

The agent is implemented in **pure Python with no framework**. All LLM calls go through the
Flywheel proxy (OpenAI-compatible). All world interactions go through 5 MCP tools exposed at
`FLYWHEEL_MCP_URL`.

---

## Environment Variables

```
FLYWHEEL_PROXY_URL       # OpenAI-compatible LLM endpoint
FLYWHEEL_PROXY_TOKEN     # Bearer token for LLM calls
FLYWHEEL_MCP_URL         # JSON-RPC endpoint for the 5 MCP tools
FLYWHEEL_MEMORY_DIR      # Persistent directory across tasks (survives between sessions)
FLYWHEEL_TASK_INSTRUCTION # The current task string
FLYWHEEL_MAX_STEPS       # Max steps per task (50)
```

---

## The 5 MCP Tools

These are the only way the agent touches the world. All calls go through JSON-RPC to
`FLYWHEEL_MCP_URL`.

```python
mcp("search_apis", {"query": str})
# Returns: list of matching API descriptions from the 457 docs

mcp("api_doc", {"app": str, "api": str})
# Returns: full parameter and response schema for one API

mcp("call_api", {"app": str, "api": str, "arguments": dict})
# Executes a single precise API call

mcp("run_code", {"code": str})
# Executes arbitrary Python. `apis` object is already in scope.
# Returns: stdout string, or traceback string on error

mcp("complete_task", {"answer": str})   # response tasks
mcp("complete_task", {})                # action tasks
# CRITICAL: this is the ONLY signal the oracle reads.
# Printing to stdout does NOT count. Forgetting this = score 0.
```

---

## Project Structure

```
agent/
├── main.py                  # Entry point. Reads env vars, runs the pipeline.
├── mcp_client.py            # Thin wrapper around JSON-RPC calls to FLYWHEEL_MCP_URL
├── llm_client.py            # Thin wrapper around the Flywheel LLM proxy
├── pipeline/
│   ├── planner.py           # Step 1: produce structured plan from task + context
│   ├── rag.py               # Step 2: API doc retrieval + code example retrieval
│   ├── codegen.py           # Step 3: Scratch / Adapt / Repair code generation
│   ├── executor.py          # Step 4: run_code + token bootstrap
│   ├── evaluator.py         # Step 5: syntactic output validation
│   └── repairer.py          # Step 6: diagnose + fix failed code
├── memory/
│   ├── store.py             # Read/write to FLYWHEEL_MEMORY_DIR (SQLite + embeddings)
│   └── retriever.py         # Embedding search + re-rank over stored solutions
├── bootstrap.py             # Login flow for all apps, returns access token dict
└── timer.py                 # Global task timer + budget enforcement
```

---

## Execution Pipeline

Every task runs this exact sequence. Steps are time-budgeted (see Timer section).

```
1. Timer starts
2. Check oracle result for previous task → persist to memory if success
3. Memory recall → fetch similar solved tasks
4. RAG retrieval → fetch relevant API docs
5. Planner → produce structured plan
6. Code generation (Scratch / Adapt)
7. Token bootstrap
8. Execute via run_code
9. Evaluator → syntactic check
10. If fail → Repair loop (max 3 attempts, time-gated)
11. If still fail → Re-plan with evaluator failure message → restart from step 6
12. complete_task (mandatory, always reached)
13. Write candidate solution to pending memory (awaiting oracle confirmation)
```

---

## Step 1 — Timer (`timer.py`)

The task hard limit is **5 minutes (300 seconds)**. The timer is a cross-cutting concern
that every step checks before proceeding.

```python
BUDGETS = {
    "plan":     30,   # seconds
    "rag":      15,
    "codegen":  45,
    "execute":  60,
    "evaluate": 15,
    "repair":   45,   # per attempt
    "replan":   30,
}

EMERGENCY_EXIT_AT  = 240  # seconds — force complete_task with best answer
HARD_KILL_AT       = 270  # seconds — complete_task with empty answer
```

Rules:

- Before every step, check `elapsed < EMERGENCY_EXIT_AT`. If not, skip to `complete_task`.
- Before each repair attempt, check `elapsed + BUDGET["repair"] < 270`. If not, skip to `complete_task`.
- The repair loop condition is: `attempt < 3 AND elapsed < 240`.
- The re-plan condition is: `repairs exhausted AND elapsed < 210` (need time for full cycle after).

---

## Step 2 — Memory (`memory/`)

### Store schema (SQLite in `FLYWHEEL_MEMORY_DIR/memory.db`)

```sql
CREATE TABLE solutions (
    id              TEXT PRIMARY KEY,
    task_description TEXT,
    embedding       BLOB,       -- float32 numpy array, serialized
    apis_used       TEXT,       -- JSON array of "app.api_name" strings
    plan            TEXT,       -- JSON plan object
    working_code    TEXT,
    evaluation_result TEXT,     -- "success" | "pending"
    created_at      TEXT
);
```

### Retrieval (`memory/retriever.py`)

1. Embed the current task description using the LLM proxy (or a local sentence transformer).
2. Cosine similarity search over all `evaluation_result = "success"` rows.
3. Re-rank top-10 results by overlap between their `apis_used` and the APIs retrieved by RAG.
4. Return top-3 solutions.

### Persistence timing

- After `complete_task` is called: write the solution with `evaluation_result = "pending"`.
- At the **start of the next task**: call the Flywheel API to check the previous task's oracle
  result. If success: update to `"success"`. If fail: delete the row.

> If the oracle result is only available as an aggregate (not per-task), persist optimistically
> with `evaluation_result = "success"` immediately after `complete_task`.

---

## Step 3 — RAG (`pipeline/rag.py`)

### API doc retrieval

Use a **two-stage retrieval**, not a flat vector search over all 457 docs:

1. From the task description, identify which apps are likely involved
   (use keyword matching first, then `mcp("search_apis", {"query": task_description})`).
2. For each identified app, call `mcp("api_doc", ...)` for every relevant API in that app.
3. Return the full doc text for each retrieved API.

Rationale: flat semantic search over 457 docs is noisy. App-first filtering is cheaper and
more precise.

### Code example retrieval

Delegate to `memory/retriever.py`. Returns up to 3 previously solved similar tasks with their
working code, plan, and apis_used.

---

## Step 4 — Planner (`pipeline/planner.py`)

### Input

```python
{
    "task": str,                     # FLYWHEEL_TASK_INSTRUCTION
    "retrieved_api_docs": [str],     # from RAG
    "similar_solutions": [...],      # from memory retriever
    "failure_context": dict | None   # only on re-plan; evaluator failure message
}
```

### Output (structured JSON, enforced via `response_format`)

```json
{
    "objective": "...",
    "task_type": "response | action",
    "apps_involved": ["spotify", "venmo"],
    "steps": [
        {
            "step_id": 1,
            "description": "...",
            "app": "spotify",
            "depends_on": []
        },
        {
            "step_id": 2,
            "description": "...",
            "app": "venmo",
            "depends_on": [1]
        }
    ]
}
```

### Re-plan

When called after repair loop exhaustion, inject the evaluator failure message:

```
System: You previously generated a plan that failed after 3 repair attempts.
Failure message: {failure_context}
Generate a new plan that avoids this failure. Be more explicit about output
format requirements and API call ordering.
```

---

## Step 5 — Token Bootstrap (`bootstrap.py`)

This runs **at execution time**, not at plan time. Tokens are scoped to the current execution.

```python
def bootstrap_access_tokens() -> dict:
    # 1. Get supervisor profile
    profile = mcp("call_api", {"app": "supervisor", "api": "show_profile", "arguments": {}})
    email = profile["email"]
    phone = profile["phone_number"]

    # 2. Get all passwords
    passwords_list = mcp("call_api", {"app": "supervisor", "api": "show_account_passwords", "arguments": {}})
    passwords = {p["account_name"]: p["password"] for p in passwords_list}

    # 3. Login per app — CRITICAL: phone app uses phone_number, not email
    tokens = {}
    tokens["spotify"]    = login("spotify",    username=email,  password=passwords["spotify"])
    tokens["gmail"]      = login("gmail",      username=email,  password=passwords["gmail"])
    tokens["amazon"]     = login("amazon",     username=email,  password=passwords["amazon"])
    tokens["venmo"]      = login("venmo",      username=email,  password=passwords["venmo"])
    tokens["splitwise"]  = login("splitwise",  username=email,  password=passwords["splitwise"])
    tokens["todoist"]    = login("todoist",    username=email,  password=passwords["todoist"])
    tokens["simplenote"] = login("simplenote", username=email,  password=passwords["simplenote"])
    tokens["phone"]      = login("phone",      username=phone,  password=passwords["phone"])
    tokens["filesystem"] = login("filesystem", username=email,  password=passwords["filesystem"])
    tokens["amazon"]     = login("amazon",     username=email,  password=passwords["amazon"])

    return tokens
```

Only bootstrap tokens for apps that the plan's `apps_involved` list includes, to save time.

---

## Step 6 — Code Generation (`pipeline/codegen.py`)

Three scenarios, selected deterministically:

| Scenario | Condition |
|----------|-----------|
| **Adapt** | Memory retriever returned at least 1 similar solution with `evaluation_result = "success"` |
| **Scratch** | No similar solution found |
| **Repair** | Previous execution failed (has error + evaluator failure message) |

### Shared system prompt (all scenarios)

```
You are a Python code generator for AppWorld.

RULES — follow exactly:
- `apis` is already in scope. Never import it.
- Access tokens are provided as a dict called `tokens`. Use tokens[app] for every authenticated call.
- Always paginate: loop page_index=0,1,2,... until a page returns empty.
- For response tasks: end with exactly print("ANSWER=" + str(ANSWER))
- For action tasks: perform the action. No print needed.
- Never call complete_task inside the generated code. The executor handles this.
- Use run_code for bulk operations ("follow ALL", "like ALL") — one loop, not repeated single calls.
- Dependency order matters: complete step N before starting step N+1.
- Output format must match exactly: number as int/float, list as comma-separated string,
  yes/no as lowercase string.
```

### Adapt prompt addition

```
Here is a previously working solution for a similar task:
{working_code}

Modify only the parts that differ for the current task. Preserve pagination logic,
token usage, and error handling from the original.
```

### Repair prompt addition

```
The following code failed:
{failed_code}

Execution error: {execution_error}
Evaluator failure: {evaluator_failure_message}

Identify the root cause. Fix only what is broken. Preserve working sections.
```

---

## Step 7 — Executor (`pipeline/executor.py`)

```python
def execute(code: str, tokens: dict) -> dict:
    # Inject tokens into code preamble
    preamble = f"tokens = {repr(tokens)}\n"
    full_code = preamble + code

    result = mcp("run_code", {"code": full_code})

    return {
        "stdout": result,
        "is_error": "Traceback" in result or "Error" in result
    }
```

---

## Step 8 — Evaluator (`pipeline/evaluator.py`)

The evaluator is **syntactic only** — it checks output format, not world state.

### Response tasks

```python
def evaluate_response(stdout: str, expected_type: str) -> dict:
    # expected_type comes from the plan: "number" | "list" | "yes_no" | "string"
    
    if "ANSWER=" not in stdout:
        return {
            "passed": False,
            "expected_type": expected_type,
            "actual_output": stdout,
            "failure_reason": "ANSWER= marker not found in stdout"
        }

    answer = stdout.split("ANSWER=")[1].strip()

    if expected_type == "number":
        try:
            float(answer)
            return {"passed": True}
        except ValueError:
            return {
                "passed": False,
                "expected_type": "number",
                "actual_output": answer,
                "failure_reason": f"output is not a number: {answer}"
            }

    if expected_type == "yes_no":
        if answer.lower() not in ("yes", "no"):
            return {
                "passed": False,
                "expected_type": "yes_no",
                "actual_output": answer,
                "failure_reason": f"output must be 'yes' or 'no', got: {answer}"
            }
        return {"passed": True}

    # list and string: just check it's non-empty
    if not answer:
        return {
            "passed": False,
            "expected_type": expected_type,
            "actual_output": answer,
            "failure_reason": "output is empty"
        }

    return {"passed": True}
```

### Action tasks

For action tasks the evaluator checks that `stdout` contains no error markers. World state
verification is not performed (syntactic only). If `is_error` is False, the evaluator passes.

### Evaluator failure message format

Always a structured dict passed to the repairer and planner:

```json
{
    "expected_type": "number",
    "actual_output": "['Song A', 'Song B']",
    "failure_reason": "output is a list, expected a single integer"
}
```

---

## Step 9 — Repair Loop (`pipeline/repairer.py`)

```python
MAX_REPAIRS = 3

def repair_loop(plan, code, tokens, timer):
    attempt = 0
    last_failure = None

    while attempt < MAX_REPAIRS:
        if not timer.should_attempt_repair(attempt):
            break  # time budget exceeded

        fixed_code = codegen.repair(
            failed_code=code,
            execution_error=last_execution_error,
            evaluator_failure=last_failure,
            plan=plan,
            api_docs=retrieved_docs
        )

        result = executor.execute(fixed_code, tokens)

        if result["is_error"]:
            last_execution_error = result["stdout"]
            attempt += 1
            continue

        eval_result = evaluator.evaluate(result["stdout"], plan["task_type"])

        if eval_result["passed"]:
            return fixed_code, result["stdout"]  # success

        last_failure = eval_result
        last_execution_error = None
        attempt += 1

    return None, last_failure  # exhausted
```

If the repair loop returns `None`: pass `last_failure` to the planner as `failure_context`
and restart the full pipeline from the planning step (once only — no infinite re-plan loops).

---

## Step 10 — complete_task (mandatory)

```python
def finish(task_type: str, stdout: str):
    if task_type == "response":
        answer = stdout.split("ANSWER=")[1].strip()
        mcp("complete_task", {"answer": answer})
    else:
        mcp("complete_task", {})
```

**This must be called at the end of every task, no exceptions.**

If the timer hits `EMERGENCY_EXIT_AT` (240s): call `finish` with whatever `stdout` is available.
If the timer hits `HARD_KILL_AT` (270s): call `mcp("complete_task", {})` immediately.

---

## Common Pitfalls (baked in as rules)

| Pitfall | Rule |
|---------|------|
| Forgetting `complete_task` | `finish()` is always the last call in `main.py`, wrapped in `finally` |
| Silent pagination miss | All list/search calls loop until empty page |
| Wrong login field for phone app | `bootstrap.py` hardcodes `username=phone_number` for phone |
| Token used after generation | Tokens generated in executor, not planner |
| Printing answer instead of submitting | Code generation prompt forbids `complete_task` inside generated code |
| Repair loop running out of time | `timer.should_attempt_repair()` checked before every attempt |
| Re-plan loop infinite | Re-plan is triggered at most once per task |
| Venmo search returns email not user_id | Codegen prompt notes: transactions use `receiver_email` |
| Song metadata not on library item | Must call `show_song()` separately for `play_count`, `genre`, `title` |

---

## LLM Call conventions (`llm_client.py`)

```python
def call_llm(system: str, user: str, response_format: str = "text") -> str:
    # Always use response_format={"type": "json_object"} for structured outputs
    # Model is fixed: gemini-3-flash-preview (set by proxy, not by caller)
    # Temperature: 0 (deterministic)
    # Max tokens: 4096
    pass
```

All planner and evaluator calls must use `response_format={"type": "json_object"}`.
Code generation calls use plain text (the output is Python, not JSON).

---

## Memory Login Recipe Cache

Beyond task solutions, the memory store caches **login recipes** as first-class entries:

```json
{
    "type": "login_recipe",
    "app": "spotify",
    "login_field": "email",
    "token_path": "access_token",
    "notes": "standard login, no quirks"
}
```

This is written after the first successful bootstrap and read at the start of every subsequent
task. If a login fails in bootstrap, the recipe is invalidated and re-derived.