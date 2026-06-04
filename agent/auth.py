def bootstrap_access_tokens():
    pass

def login_all_apps():
    pass

def validate_tokens(tokens):
    pass


def bootstrap_access_tokens():

    code = generate_login_script()

    result = run_code(code)

    return result.tokens


def validate_tokens(tokens):

    for app, token in tokens.items():

        verify_token(
            app,
            token
        )

 