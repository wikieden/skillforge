"""Task — the unit of work the lab learns to solve.

A `Task` pairs a `Goal` (the success criterion, world frame, meters) with the
scene to reset to and a per-episode step budget. Stdlib-only value types
(`frozen=True, slots=True`, per tempomem/spec/ENGINEERING.md).
"""

from __future__ import annotations

from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Goal:
    """A success criterion: an object labelled `label` within `tolerance_m`
    (meters) of `target_xyz` (world frame).
    """

    label: str
    target_xyz: Vec3
    tolerance_m: float = 0.05


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One movable object in the mock world: a tempomem scene-graph `node_id`, a
    `label`, and its world position `xyz` in meters.
    """

    node_id: int
    label: str
    xyz: Vec3


@dataclass(frozen=True, slots=True)
class Task:
    """A self-contained learning task.

    `initial_scene` is what `Environment.reset()` restores; `goal` is scored after
    each episode; `max_steps` budgets one rollout.
    """

    name: str
    goal: Goal
    initial_scene: tuple[SceneObject, ...]
    max_steps: int = 8
