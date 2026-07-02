"""The acceptance gate: strict-improvement unit + regression rejection + held-out val.

The discipline ported from microsoft/SkillOpt — accept a candidate only if it
strictly improves a held-out score — is what stops skillforge's loop from drifting
backwards. These tests pin the gate verdict, that a regressing attempt is rejected
(not appended as the new current), and that the held-out path actually scores on
val tasks rather than the train reward.
"""

from __future__ import annotations

import pytest
from bodylink import MockBody
from mindloop import SubTask

from skillforge import (
    GateAction,
    Goal,
    Lab,
    MockEnvironment,
    Policy,
    SceneObject,
    ScriptedAuthor,
    Task,
    evaluate_gate,
    evolve,
    mock_env_factory,
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
    return Policy(subtasks=(_place((0.5, 0.0, 0.0)),), version=0)  # train reward -0.5


def _good() -> Policy:
    return Policy(subtasks=(_place((1.0, 0.0, 0.0)),), version=1)  # train reward 0.0 (at goal)


def _worse() -> Policy:
    return Policy(subtasks=(_place((0.2, 0.0, 0.0)),), version=2)  # train reward -0.8


def test_gate_verdicts() -> None:
    assert evaluate_gate(0.0, -0.5, -0.5).action is GateAction.ACCEPT_NEW_BEST
    assert evaluate_gate(-0.2, -0.5, -0.1).action is GateAction.ACCEPT  # beats current, not best
    assert evaluate_gate(-0.5, -0.5, -0.1).action is GateAction.REJECT  # tie → reject
    assert evaluate_gate(-0.9, -0.5, -0.1).action is GateAction.REJECT
    assert evaluate_gate(0.0, -0.5, -0.5).accepted is True
    assert evaluate_gate(-0.5, -0.5, -0.5).accepted is False


def test_regression_is_rejected() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    # author proposes bad, then only-worse — the good policy never comes
    result = Lab(env, ScriptedAuthor([_bad(), _worse()]), max_attempts=3).run(task)

    attempts = list(result.tree)
    assert attempts[0].accepted is True  # first candidate beats -inf
    assert attempts[1].accepted is False  # worse than current → vetoed
    assert attempts[1].reward < attempts[0].reward  # it genuinely regressed
    # rejected attempts hang off the accepted node and never advance the lineage
    assert attempts[1].parent_id == 0
    assert attempts[2].parent_id == 0
    assert not result.solved


def test_holdout_val_path_is_used() -> None:
    train = _task()
    body = MockBody()
    env = MockEnvironment(body, train.initial_scene)
    # val goal differs from train goal, so the held-out score != the train reward
    val = Task(
        name="move-cube-val",
        goal=Goal("cube", (2.0, 0.0, 0.0), 0.05),
        initial_scene=(SceneObject(1, "cube", (0.0, 0.0, 0.0)),),
        max_steps=4,
    )
    result = Lab(env, ScriptedAuthor([_bad(), _good()])).run(
        train, val_tasks=[val], env_for=mock_env_factory(body)
    )

    assert result.solved  # the good policy passes the TRAIN goal
    good = result.tree.get(1)
    # holdout_score is scored on the VAL goal (good places cube at (1,0,0); val goal
    # is (2,0,0) → distance 1.0 → -1.0), NOT the train reward (0.0) — proves the
    # held-out env_for path ran.
    assert good.holdout_score == -1.0
    assert good.log.reward.value == 0.0


def test_val_tasks_require_env_for() -> None:
    task = _task()
    env = MockEnvironment(MockBody(), task.initial_scene)
    with pytest.raises(ValueError, match="env_for"):
        evolve(env, ScriptedAuthor([_good()]), task, val_tasks=[task])
