"""AttemptTree unit tests — lineage, best/frontier selection, pruning."""

from __future__ import annotations

import pytest

from skillforge import Attempt, AttemptTree, Policy, Reward, RolloutLog


def _log(success: bool, value: float, episode: str = "t") -> RolloutLog:
    return RolloutLog(
        episode=episode,
        steps=(),
        reward=Reward(success, value),
        success=success,
        robot_busy_ms=100.0,
    )


def _attempt(att_id: int, parent: int | None, success: bool, value: float) -> Attempt:
    return Attempt(
        id=att_id,
        parent_id=parent,
        policy=Policy(subtasks=()),
        log=_log(success, value),
        author_tokens=1000,
    )


def test_lineage_and_best() -> None:
    tree = AttemptTree()
    tree.add(_attempt(0, None, False, -0.5))
    tree.add(_attempt(1, 0, True, 0.0))

    assert len(tree) == 2
    assert [a.id for a in tree.lineage(1)] == [0, 1]
    best = tree.best()
    assert best is not None and best.id == 1


def test_add_rejects_unknown_parent() -> None:
    tree = AttemptTree()
    with pytest.raises(ValueError, match="unknown parent"):
        tree.add(_attempt(5, 99, False, -1.0))


def test_frontier_prefers_best_pending_leaf() -> None:
    tree = AttemptTree()
    tree.add(_attempt(0, None, False, -0.9))
    tree.add(_attempt(1, 0, False, -0.2))  # better-scoring failure → expand this next
    frontier = tree.frontier()
    assert frontier is not None and frontier.id == 1


def test_prune_drops_failed_leaf() -> None:
    tree = AttemptTree()
    tree.add(_attempt(0, None, False, -1.0))
    removed = tree.prune(lambda a: not a.success)
    assert removed == [0]
    assert len(tree) == 0
