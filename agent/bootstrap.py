"""Bootstrap access tokens for all apps the plan requires.

Phone is the only app that uses phone_number as username; all others use email.
Real AppWorld app names use underscores: simple_note, file_system.
"""

import agent.logger as log

# Normalise LLM-emitted aliases to real AppWorld app names
APP_NAMES: dict[str, str] = {
    "spotify":     "spotify",
    "amazon":      "amazon",
    "gmail":       "gmail",
    "phone":       "phone",
    "venmo":       "venmo",
    "splitwise":   "splitwise",
    "todoist":     "todoist",
    "simple_note": "simple_note",
    "simplenote":  "simple_note",
    "file_system": "file_system",
    "filesystem":  "file_system",
}


def bootstrap(ctx, apps_involved: list[str]) -> dict[str, str]:
    """Login to each app in apps_involved via MCP. Returns {app_name: access_token}."""
    # Step 1: supervisor profile (email + phone_number)
    profile = ctx.mcp.call(
        "call_api",
        {"app": "supervisor", "api": "show_profile", "arguments": {}},
    )
    if isinstance(profile, dict) and "result" in profile:
        profile = profile["result"]
    profile = profile or {}
    email = profile.get("email", "")
    phone_number = profile.get("phone_number", "")

    # Step 2: all account passwords
    pw_raw = ctx.mcp.call(
        "call_api",
        {"app": "supervisor", "api": "show_account_passwords", "arguments": {}},
    )
    if isinstance(pw_raw, dict) and "result" in pw_raw:
        pw_raw = pw_raw["result"]
    passwords: dict[str, str] = {
        p["account_name"]: p["password"]
        for p in (pw_raw or [])
        if "account_name" in p and "password" in p
    }

    tokens: dict[str, str] = {}

    for short_name in apps_involved:
        app = APP_NAMES.get(short_name, short_name)
        password = passwords.get(app) or passwords.get(short_name, "")
        if not password:
            continue
        username = phone_number if app == "phone" else email
        try:
            result = ctx.mcp.call(
                "call_api",
                {"app": app, "api": "login", "arguments": {"username": username, "password": password}},
            )
            if isinstance(result, dict) and "result" in result:
                result = result["result"]
            token = result.get("access_token", "") if isinstance(result, dict) else ""
            if token:
                tokens[short_name] = token
                if app != short_name:
                    tokens[app] = token
                log.info("bootstrap", f"{app} ✓ token acquired")
            else:
                log.error("bootstrap", f"{app}: no token returned")
        except Exception as e:
            log.error("bootstrap", f"{app}: {e}")

    return tokens
