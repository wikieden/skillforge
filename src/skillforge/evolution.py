"""E — the evolution loop: author → rollout → score → **gate** → append → repeat.

Cross-episode skill improvement: each round the author proposes a policy from the
history, a rollout scores it, and an **acceptance gate** (ported discipline from
microsoft/SkillOpt — see `gate.py`) keeps it only if its held-out score strictly
beats the current accepted policy. Accepted attempts advance the lineage; rejected
ones are still recorded (append-only audit) but do not move the frontier. The loop
stops when an accepted policy succeeds, or at the attempt budget.

Held-out scoring: pass `val_tasks` (+ an `env_for` factory) to vet each candidate
on scenes it was *not* optimized on — the non-leaky split that makes the gate
meaningful. With no `val_tasks` the gate falls back to the train reward (still
accept-only-if-improves, but it cannot measure generalization).

(mindloop's P4 `Replanner` is the *intra*-episode analog — recover one failed step;
this is the *inter*-episode one — compound many trials into a skill.)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .author import PolicyAuthor
from .environment import EnvFactory, Environment
from .gate import evaluate_gate
from .policy import Policy
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


def _score_holdout(env_for: EnvFactory, policy: Policy, tasks: Sequence[Task]) -> float:
    """Mean rollout reward of `policy` over `tasks`, each in its own fresh env."""
    scores = [
        run_episode(
            env_for(t), policy, t.goal, episode=f"val:{t.name}", max_steps=t.max_steps
        ).reward.value
        for t in tasks
    ]
    return sum(scores) / len(scores)


def evolve(
    env: Environment,
    author: PolicyAuthor,
    task: Task,
    *,
    val_tasks: Sequence[Task] = (),
    env_for: EnvFactory | None = None,
    max_attempts: int = 8,
) -> EvolutionResult:
    """Run the gated EN-PI-R-E loop until an accepted policy solves `task`.

    `env` is the training environment. To vet candidates on a non-leaky held-out
    set, pass `val_tasks` plus an `env_for` factory that builds an environment for
    each; otherwise the gate scores on the train reward. Stops early when an
    *accepted* attempt's training rollout succeeds, else runs to `max_attempts`.
    """
    if val_tasks and env_for is None:
        raise ValueError("val_tasks require env_for to build a holdout environment per task")
    tree = AttemptTree()
    history: list[Attempt] = []
    prev_tokens = 0
    parent_id: int | None = None
    current_score = float("-inf")
    best_score = float("-inf")
    for i in range(max_attempts):
        policy = author.author(task, tuple(history))
        spent = author.cost_tokens()
        tokens = spent - prev_tokens
        prev_tokens = spent
        log = run_episode(
            env, policy, task.goal, episode=f"{task.name}#{i}", max_steps=task.max_steps
        )
        cand_score = (
            _score_holdout(env_for, policy, val_tasks)  # type: ignore[arg-type]  # env_for set when val_tasks
            if val_tasks
            else log.reward.value
        )
        gate = evaluate_gate(cand_score, current_score, best_score)
        attempt = Attempt(
            id=i,
            parent_id=parent_id,
            policy=policy,
            log=log,
            author_tokens=tokens,
            holdout_score=cand_score,
            accepted=gate.accepted,
        )
        tree.add(attempt)
        history.append(attempt)
        _log.info(
            "attempt %d: success=%s reward=%.4f holdout=%.4f gate=%s",
            i,
            log.success,
            log.reward.value,
            cand_score,
            gate.action.value,
        )
        if gate.accepted:
            current_score = cand_score
            best_score = max(best_score, cand_score)
            parent_id = i
            if log.success:
                return EvolutionResult(tree=tree, solved=True, attempts=len(tree))
    best = tree.best()
    return EvolutionResult(tree=tree, solved=bool(best and best.success), attempts=len(tree))
