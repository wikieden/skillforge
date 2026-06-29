"""The walking-skeleton test: the EN-PI-R-E loop closes on its own.

Attempt 1 uses a policy that undershoots the target (fails); the ScriptedAuthor
then returns a corrected policy and attempt 2 succeeds — with no LLM and no
network. Asserts the loop solved it, recorded both attempts with correct lineage,
and produced sane metrics.
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


def _task() -> Task:
    return Task(
        name="move-cube",
        goal=Goal("cube", (1.0, 0.0, 0.0), 0.05),
        initial_scene=(SceneObject(1, "cube", (0.0, 0.0, 0.0)),),
        max_steps=4,
    )


def _bad() -> Policy:
    return Policy(subtasks=(_place((0.5, 0.0, 0.0)),), version=0, notes="undershoots")


def _good() -> Policy:
    return Policy(subtasks=(_place((1.0, 0.0, 0.0)),), version=1, notes="reaches goal")


def test_loop_self_closes() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    result = Lab(env, ScriptedAuthor([_bad(), _good()])).run(task)

    assert result.solved
    assert len(result.tree) == 2
    attempts = list(result.tree)
    assert attempts[0].success is False
    assert attempts[1].success is True
    assert [a.id for a in result.tree.lineage(1)] == [0, 1]
    assert attempts[1].parent_id == 0


def test_failed_attempt_is_recorded() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    result = Lab(env, ScriptedAuthor([_bad(), _good()])).run(task)

    first = result.tree.get(0)
    assert not first.success
    assert first.reward < 0.0  # value is -distance; the cube fell short of the goal
    assert first.log.steps[0].subtask.verb == "place"


def test_metrics_wired() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    metrics = Lab(env, ScriptedAuthor([_bad(), _good()])).run(task).metrics

    assert metrics.solved
    assert metrics.pass_k == 2
    assert 0.0 < metrics.mru < 1.0
    assert metrics.mtu > 0.0


def test_unsolved_within_budget() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    # an author that only ever proposes the failing policy
    result = Lab(env, ScriptedAuthor([_bad()]), max_attempts=3).run(task)

    assert not result.solved
    assert len(result.tree) == 3
    assert result.metrics.pass_k == 0
