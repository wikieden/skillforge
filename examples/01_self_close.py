"""Self-closing loop demo: the lab fixes its own policy on the second try.

No GPU, no network — `ScriptedAuthor` + `MockEnvironment` stand in for a real LLM
author and real hardware. Run:

    uv run python skillforge/examples/01_self_close.py
"""

from __future__ import annotations

from bodylink import MockBody
from mindloop import SubTask

from skillforge import (
    Goal,
    Lab,
    MockEnvironment,
    Policy,
    SceneObject,
    ScriptedAuthor,
    Task,
)


def _place(loc: tuple[float, float, float]) -> SubTask:
    return SubTask(id="place-cube", verb="place", target_node_ids=(1,), params={"location": loc})


def main() -> None:
    task = Task(
        name="move-cube",
        goal=Goal("cube", (1.0, 0.0, 0.0), 0.05),
        initial_scene=(SceneObject(1, "cube", (0.0, 0.0, 0.0)),),
        max_steps=4,
    )
    # The scripted "coding agent": attempt 1 undershoots, attempt 2 reaches the goal.
    script = [
        Policy(subtasks=(_place((0.5, 0.0, 0.0)),), version=0, notes="undershoots the target"),
        Policy(subtasks=(_place((1.0, 0.0, 0.0)),), version=1, notes="reaches the target"),
    ]

    result = Lab(MockEnvironment(MockBody(), task.initial_scene), ScriptedAuthor(script)).run(task)

    print(f"task {task.name!r}: solved={result.solved}  pass@{result.metrics.pass_k}")
    print("attempt tree:")
    for attempt in result.tree:
        mark = "OK " if attempt.success else "FAIL"
        parent = "root" if attempt.parent_id is None else f"#{attempt.parent_id}"
        print(
            f"  #{attempt.id} (parent {parent}) [{mark}] "
            f"reward={attempt.reward:+.3f}  v{attempt.policy.version}  {attempt.policy.notes!r}"
        )
    metrics = result.metrics
    print(
        f"metrics: MRU={metrics.mru:.2f}  MTU={metrics.mtu:.0f} tok/robot-s  "
        f"attempts={metrics.attempts}"
    )


if __name__ == "__main__":
    main()
