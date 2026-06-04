def attempt_solution(
    ctx,
    task,
    code_plan
):

    tokens = bootstrap_access_tokens()

    skill = find_matching_skill(task)

    failed_code = None
    diagnosis = None

    for attempt in range(3):

        generation_context = {

            "skill": skill,
            "failed_code": failed_code,
            "diagnosis": diagnosis,
            "code_plan": code_plan,
            "tokens": tokens
        }

        code = generate_solution_code(
            generation_context
        )

        execution_result = execute_solution(
            ctx,
            code
        )

        if execution_result.success:

            evaluation = evaluate_solution(
                ctx,
                execution_result
            )

            if check_success(
                evaluation
            ):

                return SuccessResult(
                    code=code,
                    answer=extract_answer(
                        execution_result
                    ),
                    apis_used=extract_used_apis(
                        code
                    )
                )

        diagnosis = diagnose_failure(
            code,
            execution_result.error,
            code_plan
        )

        failed_code = code

    return FailureResult(
        feedback=diagnosis
    )