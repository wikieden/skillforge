"""The acceptance gate — keep a candidate policy only if it strictly improves.

Ported *discipline* (not code) from microsoft/SkillOpt's `evaluate_gate`: a
candidate is accepted **only when its held-out score is strictly greater** than
the current accepted score (ties are rejected). This is the rigor skillforge's
Evolution loop lacked — it previously appended every attempt unconditionally and
stopped at the first success, so a regressing author could never be vetoed.

Pure, stdlib-only, frozen result — the loop owns the side effects (see
`evolution.evolve`); this module is just the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateAction(str, Enum):
    """The gate's verdict on a candidate."""

    ACCEPT = "accept"
    ACCEPT_NEW_BEST = "accept_new_best"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class GateResult:
    """A gate verdict plus the three scores it compared (for the audit trail)."""

    action: GateAction
    cand_score: float
    current_score: float
    best_score: float

    @property
    def accepted(self) -> bool:
        """True unless the candidate was rejected."""
        return self.action is not GateAction.REJECT


def evaluate_gate(cand_score: float, current_score: float, best_score: float) -> GateResult:
    """Accept `cand_score` only if it **strictly** beats `current_score`.

    Mirrors SkillOpt's gate: strict ``>`` (ties rejected); a candidate that also
    beats `best_score` is flagged ``ACCEPT_NEW_BEST``. Scores are "higher is
    better" (skillforge uses ``Reward.value`` = ``-distance``, so ``0.0`` at goal).
    A rejected candidate leaves the current policy unchanged.
    """
    if cand_score > current_score:
        if cand_score > best_score:
            return GateResult(GateAction.ACCEPT_NEW_BEST, cand_score, current_score, best_score)
        return GateResult(GateAction.ACCEPT, cand_score, current_score, best_score)
    return GateResult(GateAction.REJECT, cand_score, current_score, best_score)
