"""Metrics — pass@k and the ENPIRE utilization pair (MRU / MTU).

mindloop's eval.py scored citation compliance only. A self-driving lab also needs
outcome + budget metrics: did it solve within k tries, and how well were the two
scarce resources used — robot time (MRU, Mean Robot Utilization) and agent tokens
(MTU, here token throughput per robot-second). At fleet=1 these are simple but
wired, so the fleet generalization later is a change of inputs, not of contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .tree import Attempt

# Nominal author "thinking" time charged per 1k tokens (deterministic; feeds MRU).
THINK_MS_PER_KTOK = 50.0


@dataclass(frozen=True, slots=True)
class LabMetrics:
    """The scored summary of one lab run."""

    attempts: int
    solved: bool
    pass_k: int  # the k at which it first solved (0 if never)
    mru: float  # mean robot utilization in [0, 1]
    mtu: float  # token throughput: author tokens per robot-second


def pass_at_k(attempts: Sequence[Attempt], k: int) -> bool:
    """True if any of the first `k` attempts succeeded."""
    return any(a.success for a in list(attempts)[:k])


def _first_success_k(attempts: Sequence[Attempt]) -> int:
    for i, attempt in enumerate(attempts, start=1):
        if attempt.success:
            return i
    return 0


def compute(attempts: Sequence[Attempt]) -> LabMetrics:
    """Score a run's attempts into a `LabMetrics`."""
    atts = list(attempts)
    robot_ms = sum(a.log.robot_busy_ms for a in atts)
    tokens = sum(a.author_tokens for a in atts)
    think_ms = tokens / 1000.0 * THINK_MS_PER_KTOK
    wall_ms = robot_ms + think_ms
    robot_s = robot_ms / 1000.0
    k = _first_success_k(atts)
    return LabMetrics(
        attempts=len(atts),
        solved=k > 0,
        pass_k=k,
        mru=robot_ms / wall_ms if wall_ms else 0.0,
        mtu=tokens / robot_s if robot_s else 0.0,
    )
