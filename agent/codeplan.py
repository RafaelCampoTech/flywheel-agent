"""Plan Python execution (run_code) from the high-level plan."""

def build_code_plan(enriched_plan):

    examples = enriched_plan.examples

    candidate_apis = extract_apis_from_examples(
        examples
    )

    docs = retrieve_docs_for_apis(
        candidate_apis
    )

    discrepancy_report = evaluate_code_plan(
        enriched_plan.plan,
        examples,
        docs
    )

    code_plan = LLM(
        plan=enriched_plan.plan,
        examples=examples,
        docs=docs,
        discrepancy_report=discrepancy_report
    )

    return {
        "code_plan": code_plan,
        "docs": docs,
        "examples": examples
    }


def evaluate_code_plan(
    plan,
    examples,
    docs
):

    return LLM(
        plan=plan,
        examples=examples,
        docs=docs
    )