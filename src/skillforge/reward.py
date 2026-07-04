"""Reward bricks — score a goal over a list of Detections (world frame, meters).

These are the EN module's verification primitives: no scoring lived in the stack
before (mindloop's eval.py did citation-compliance only). A real Environment
scores the same way over `worldsense` multi-camera Detections; `MockEnvironment`
scores over its own world. Returns-over-raises: a missing object yields a failed
`Reward`, not an exception.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .task import Goal, Vec3

if TYPE_CHECKING:
    from tempomem import Detection


@dataclass(frozen=True, slots=True)
class Reward:
    """A scored outcome: `success` (goal met) + a scalar `value` (higher is
    better; here ``-distance`` to the target, so ``0.0`` exactly at the goal).
    """

    success: bool
    value: float


@dataclass(frozen=True, slots=True)
class Weights:
    """D-G composite-reward weights over the three outcome categories.

    The ordering ``task > handle > honesty`` is the ratified invariant
    (TASK-FAMILIES D-G): the loop must never learn to prefer graceful declining
    over actually completing the task. Defaults are the ratified 0.6/0.25/0.15.
    """

    task: float = 0.6
    handle: float = 0.25
    honesty: float = 0.15

    def __post_init__(self) -> None:
        if not (self.task > self.handle > self.honesty > 0.0):
            raise ValueError(
                f"D-G requires w_task > w_handle > w_honesty > 0, "
                f"got {self.task}/{self.handle}/{self.honesty}"
            )


def composite_score(
    task_rate: float,
    handle_rate: float,
    honesty_rate: float,
    *,
    weights: Weights | None = None,
) -> float:
    """The D-G ENPIRE training signal: weighted sum of the three outcome rates.

    Rates are in [0, 1]; the result is in [0, 1] when weights sum to 1. This is
    the shaping signal only — go/no-go (`meets`) gates on task success alone.
    """
    w = weights if weights is not None else Weights()
    return w.task * task_rate + w.handle * handle_rate + w.honesty * honesty_rate


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def distance_to_target(dets: Sequence[Detection], label: str, target_xyz: Vec3) -> float:
    """Smallest distance (meters) from a `label` detection to `target_xyz`.

    Returns ``inf`` when no detection carries `label` (object not present).
    """
    best = math.inf
    for d in dets:
        if d.label == label:
            best = min(best, _dist(d.center_xyz, target_xyz))
    return best


def object_at_pose(
    dets: Sequence[Detection], label: str, target_xyz: Vec3, tolerance_m: float
) -> bool:
    """True if some `label` detection is within `tolerance_m` of `target_xyz`."""
    return distance_to_target(dets, label, target_xyz) <= tolerance_m


def score(dets: Sequence[Detection], goal: Goal) -> Reward:
    """Score `goal` against the current detections: binary success + a shaped
    scalar (``-distance``; a large negative when the object is absent).
    """
    dist = distance_to_target(dets, goal.label, goal.target_xyz)
    value = -dist if math.isfinite(dist) else -1.0e9
    return Reward(success=dist <= goal.tolerance_m, value=value)
