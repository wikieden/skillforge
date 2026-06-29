"""E — the evolution loop: author → rollout → score → append → repeat, until solved.

Cross-episode skill improvement: each round the author proposes an improved policy
from the history, a rollout scores it on the (mock) hardware, and the attempt joins
the tree as a child of the previous one. The loop stops at the first success or the
attempt budget. (mindloop's P4 `Replanner` is the *intra*-episode analog — recover
one failed step; this is the *inter*-episode one — compound many trials into a skill.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .author import PolicyAuthor
from .environment import Environment
from .rollout import run_episode
from .task import Task
from .tree import Attempt, AttemptTree

_log = logging.getLogger("skillforge.evolution")


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """The tree built by a run, plus whether the task was solved."""

    tree: AttemptTree
    solved: bool
    attempts: int


def evolve(
    env: Environment,
    author: PolicyAuthor,
    task: Task,
    *,
    max_attempts: int = 8,
) -> EvolutionResult:
    """Run the EN-PI-R-E loop until `task` is solved or `max_attempts` is hit."""
    tree = AttemptTree()
    history: list[Attempt] = []
    prev_tokens = 0
    parent_id: int | None = None
    for i in range(max_attempts):
        policy = author.author(task, tuple(history))
        spent = author.cost_tokens()
        tokens = spent - prev_tokens
        prev_tokens = spent
        log = run_episode(
            env, policy, task.goal, episode=f"{task.name}#{i}", max_steps=task.max_steps
        )
        attempt = Attempt(id=i, parent_id=parent_id, policy=policy, log=log, author_tokens=tokens)
        tree.add(attempt)
        history.append(attempt)
        _log.info("attempt %d: success=%s reward=%.4f", i, log.success, log.reward.value)
        if log.success:
            return EvolutionResult(tree=tree, solved=True, attempts=len(tree))
        parent_id = i
    best = tree.best()
    return EvolutionResult(tree=tree, solved=bool(best and best.success), attempts=len(tree))
