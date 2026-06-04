"""Cross-task memory: read procedures from prior tasks, write what worked."""


def recall(ctx) -> dict:
    """Load persisted memory (wiped between tasks on the off-arm)."""
    return ctx.memory.read()


def remember(ctx, key: str, value) -> None:
    """Persist a recipe or fact for later tasks in this stream."""
    ctx.memory.write(key, value)


def find_matching_skill(task):

    embedding = embed(task)

    return vector_search(
        index="skills",
        embedding=embedding,
        top_k=5
    )


def store_successful_solution(
    task,
    plan,
    code,
    apis_used
):

    embedding = embed(task)

    record = {
        "task_description": task,
        "embedding": embedding,
        "plan": plan,
        "code": code,
        "apis_used": apis_used
    }

    save_solution_record(
        record
    )


def save_solution_record(record):

    database.insert(record)