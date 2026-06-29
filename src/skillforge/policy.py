"""The policy artifact — a declarative plan the author edits between attempts.

In ENPIRE the coding agent rewrites ``policy.py``; here a `Policy` is an
immutable, inspectable plan (a tuple of `SubTask`s) plus provenance (`version`,
`notes`). Keeping it declarative and serializable lets the attempt tree store it
and a diff between versions be read at a glance. v2 may unify this with mindloop's
P2 `SkillCard`; the skeleton stays minimal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mindloop import SubTask

if TYPE_CHECKING:
    from tempomem import Detection

    from .task import Goal


@dataclass(frozen=True, slots=True)
class Policy:
    """An immutable verb plan with provenance.

    `subtasks` is the ordered plan; `version`/`notes` track the author's lineage.
    """

    subtasks: tuple[SubTask, ...]
    version: int = 0
    notes: str = ""

    def plan(self, observation: Sequence[Detection], goal: Goal) -> list[SubTask]:
        """Return this episode's verb plan.

        v1 is observation-independent (the mock world is deterministic); the
        signature carries `observation` + `goal` so a real, reactive policy can
        replan from what it sees without a contract change.
        """
        return list(self.subtasks)
