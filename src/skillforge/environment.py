"""EN — the Environment seam: reset + act + observe + reward.

The brain's `ExecutorAdapter` is stateless (dispatch only); a closed learning
loop also needs to reset the world and score outcomes. `Environment` adds exactly
those two missing primitives. `MockEnvironment` is the deterministic, no-hardware
reference: it owns a mock world, drives a `bodylink.MockBody` for actions, and
scores from its own state — so bodylink stays unchanged. A real robot fills the
same protocol (reset = a homing routine; observe = `worldsense` detections).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from mindloop import ExecutorAdapter, StatusUpdate, SubTask, TaskStatus
from tempomem import Detection

from .reward import Reward, score
from .task import Goal, SceneObject, Task, Vec3

_log = logging.getLogger("skillforge.environment")

# Nominal robot-busy time charged per dispatched step (deterministic — keeps MRU
# reproducible in tests; a real env measures wall time instead).
STEP_ROBOT_MS = 100.0


@runtime_checkable
class Environment(Protocol):
    """A resettable, scoreable world the rollout drives a policy through."""

    def reset(self) -> list[Detection]: ...
    def step(self, subtask: SubTask) -> StatusUpdate: ...
    def observe(self) -> list[Detection]: ...
    def reward(self, goal: Goal) -> Reward: ...


def _unit_feature() -> np.ndarray:
    """A valid (non-empty, finite-norm) placeholder feature for mock detections."""
    return np.ones(1, dtype=np.float32)


def _det(label: str, xyz: Vec3) -> Detection:
    lo = (xyz[0] - 0.01, xyz[1] - 0.01, xyz[2] - 0.01)
    hi = (xyz[0] + 0.01, xyz[1] + 0.01, xyz[2] + 0.01)
    return Detection(label=label, feature=_unit_feature(), center_xyz=xyz, bbox_min=lo, bbox_max=hi)


def _drain(stream: Iterable[StatusUpdate]) -> StatusUpdate:
    """Consume a body's status stream down to its terminal update."""
    last: StatusUpdate | None = None
    for update in stream:
        last = update
    if last is None:
        raise RuntimeError("body produced no status update")
    return last


class MockEnvironment:
    """A deterministic mock world over a `bodylink.MockBody`.

    Holds objects by node id; a ``place(target=<node>, location=<xyz>)`` SubTask
    moves that object. `step` dispatches to the body, then applies the action's
    world effect on SUCCESS. `reset` restores the initial scene exactly — v1 is
    deterministic (randomized reset is a later EN extension; determinism first).
    """

    def __init__(self, body: ExecutorAdapter, initial_scene: Sequence[SceneObject]) -> None:
        self._body = body
        self._initial = tuple(initial_scene)
        self._world: dict[int, SceneObject] = {}
        self.reset()

    def reset(self) -> list[Detection]:
        """Restore the initial scene and return the opening observation."""
        self._world = {o.node_id: o for o in self._initial}
        return self.observe()

    def observe(self) -> list[Detection]:
        """Current world state as detections (the reward seam's input)."""
        return [_det(o.label, o.xyz) for o in self._world.values()]

    def step(self, subtask: SubTask) -> StatusUpdate:
        """Dispatch one SubTask to the body, then apply its world effect on success."""
        status = _drain(self._body.dispatch(subtask))
        if status.status is TaskStatus.SUCCESS:
            self._apply(subtask)
        return status

    def reward(self, goal: Goal) -> Reward:
        """Score `goal` against the current world state."""
        return score(self.observe(), goal)

    def _apply(self, subtask: SubTask) -> None:
        if subtask.verb != "place":
            return
        loc: Any = subtask.params.get("location")
        if loc is None:
            return
        xyz: Vec3 = (float(loc[0]), float(loc[1]), float(loc[2]))
        for nid in subtask.target_node_ids:
            obj = self._world.get(nid)
            if obj is not None:
                self._world[nid] = SceneObject(obj.node_id, obj.label, xyz)
                _log.debug("placed node %d (%s) at %s", nid, obj.label, xyz)


# An environment builder: given a Task, produce a fresh Environment for its scene.
# Lets the evolution loop score a policy on held-out (validation) tasks, each in
# its own world, rather than only on the one scene a single env is bound to.
EnvFactory = Callable[[Task], Environment]


def mock_env_factory(body: ExecutorAdapter) -> EnvFactory:
    """Build a `MockEnvironment` for any task's `initial_scene`, reusing one body.

    `MockBody` is stateless, so the same body safely drives every task's env.
    """

    def make(task: Task) -> Environment:
        return MockEnvironment(body, task.initial_scene)

    return make
