"""R — run one budgeted episode and log everything.

reset → plan → dispatch each step → score. The full `RolloutLog` (plan, statuses,
reward, success, accounted robot-busy time) is the audit record the Evolution
module appends to the attempt tree. The loop stops early on the first FAILED step.
"""

from __future__ import annotations

from dataclasses import dataclass

from mindloop import StatusUpdate, SubTask, TaskStatus

from .environment import STEP_ROBOT_MS, Environment
from .policy import Policy
from .reward import Reward
from .task import Goal


@dataclass(frozen=True, slots=True)
class StepLog:
    """One dispatched step and the body's terminal status for it."""

    subtask: SubTask
    status: StatusUpdate


@dataclass(frozen=True, slots=True)
class RolloutLog:
    """The complete record of one episode."""

    episode: str
    steps: tuple[StepLog, ...]
    reward: Reward
    success: bool
    robot_busy_ms: float


def run_episode(
    env: Environment,
    policy: Policy,
    goal: Goal,
    *,
    episode: str,
    max_steps: int = 8,
) -> RolloutLog:
    """Reset, run `policy` against `env` under a step budget, and score `goal`."""
    obs = env.reset()
    plan = policy.plan(obs, goal)
    steps: list[StepLog] = []
    for subtask in plan[:max_steps]:
        status = env.step(subtask)
        steps.append(StepLog(subtask, status))
        if status.status is TaskStatus.FAILED:
            break
    reward = env.reward(goal)
    return RolloutLog(
        episode=episode,
        steps=tuple(steps),
        reward=reward,
        success=reward.success,
        robot_busy_ms=len(steps) * STEP_ROBOT_MS,
    )
