"""The Lab — the closed EN-PI-R-E loop as one call.

Wires an `Environment` (EN) + a `PolicyAuthor` (PI) through `run_episode` (R) and
`evolve` (E), then scores the run. `Lab.run(task)` returns the attempt tree, the
metrics, and whether the task was solved — the whole self-driving loop at fleet=1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import metrics as _metrics
from .author import PolicyAuthor
from .environment import EnvFactory, Environment
from .evolution import evolve
from .metrics import LabMetrics
from .task import Task
from .tree import AttemptTree


@dataclass(frozen=True, slots=True)
class LabResult:
    """The outcome of a lab run: the tree, its metrics, and solved/not."""

    tree: AttemptTree
    metrics: LabMetrics
    solved: bool


class Lab:
    """Drives the EN-PI-R-E loop for a single task on a single environment."""

    def __init__(self, env: Environment, author: PolicyAuthor, *, max_attempts: int = 8) -> None:
        self._env = env
        self._author = author
        self._max_attempts = max_attempts

    def run(
        self,
        task: Task,
        *,
        val_tasks: Sequence[Task] = (),
        env_for: EnvFactory | None = None,
    ) -> LabResult:
        """Run the closed loop on `task` and return the scored result.

        Pass `val_tasks` + an `env_for` factory to gate candidates on a held-out
        set (see `evolution.evolve`); omit them for the train-reward gate.
        """
        result = evolve(
            self._env,
            self._author,
            task,
            val_tasks=val_tasks,
            env_for=env_for,
            max_attempts=self._max_attempts,
        )
        metrics = _metrics.compute(list(result.tree))
        return LabResult(tree=result.tree, metrics=metrics, solved=result.solved)
