"""Thin wrapper over a single AppWorld task. AppWorld is the substrate: 9 simulated apps
(spotify, amazon, gmail, phone, venmo, splitwise, todoist, simple_note, file_system) plus
supervisor and api_docs, exposed as the `apis` object inside an interactive Python sandbox.

You act by writing Python and running it with execute(code): the `apis` object is in scope,
state (logins, data) persists across calls within a task. Grading is AppWorld's own
deterministic state oracle, read here as evaluate()['success']. See docs/appworld.md, and
read the login flow there before you write a solver: it's the #1 thing that trips people up.
"""
from appworld import AppWorld, load_task_ids

__all__ = ["AppWorldEnv", "load_task_ids"]


class AppWorldEnv:
    """One open task. `world` is the live AppWorld instance; `task` exposes instruction,
    supervisor (.first_name/.last_name/.email/.phone_number) and app_descriptions."""

    def __init__(self, task_id, experiment_name="flywheel"):
        self.world = AppWorld(task_id=task_id, experiment_name=experiment_name)
        self.task = self.world.task
        self.task_id = task_id

    @property
    def instruction(self):
        return self.task.instruction

    def execute(self, code):
        """Run a Python snippet in the task sandbox. Returns stdout / result / traceback as a
        string. Call apis.<app>.<method>(...); print() what you want to read back."""
        return self.world.execute(code)

    def completed(self):
        return self.world.task_completed()

    def evaluate(self):
        """Deterministic oracle. True iff the task's goal state was reached. The oracle only
        sees answers submitted via apis.supervisor.complete_task, so you must call it."""
        return bool(self.world.evaluate().to_dict().get("success"))

    def evaluate_full(self):
        """Return the full oracle verdict dict (success, failures, traces)."""
        return self.world.evaluate().to_dict()

    def close(self):
        try:
            self.world.close()
        except Exception:
            pass
