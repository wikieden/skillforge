"""Reward-brick unit tests — scoring over Detections, network-free."""

from __future__ import annotations

import numpy as np
from tempomem import Detection

from skillforge import Goal, distance_to_target, object_at_pose, score


def _det(label: str, xyz: tuple[float, float, float]) -> Detection:
    lo = (xyz[0] - 0.01, xyz[1] - 0.01, xyz[2] - 0.01)
    hi = (xyz[0] + 0.01, xyz[1] + 0.01, xyz[2] + 0.01)
    return Detection(
        label=label, feature=np.ones(1, dtype=np.float32), center_xyz=xyz, bbox_min=lo, bbox_max=hi
    )


def test_distance_and_pose() -> None:
    dets = [_det("cube", (1.0, 0.0, 0.0)), _det("ball", (0.0, 0.0, 0.0))]
    assert distance_to_target(dets, "cube", (1.0, 0.0, 0.0)) == 0.0
    assert object_at_pose(dets, "cube", (1.0, 0.0, 0.0), 0.05)
    assert not object_at_pose(dets, "cube", (0.0, 0.0, 0.0), 0.05)


def test_missing_object_is_failure() -> None:
    reward = score([], Goal("cube", (0.0, 0.0, 0.0), 0.05))
    assert not reward.success
    assert reward.value < 0.0


def test_success_at_goal() -> None:
    dets = [_det("cube", (0.0, 0.0, 0.0))]
    reward = score(dets, Goal("cube", (0.0, 0.0, 0.0), 0.05))
    assert reward.success
    assert reward.value == 0.0
