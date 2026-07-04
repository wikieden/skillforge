"""Crystallize — a gate-accepted winning policy becomes an Episteme card (step 6).

The last arc of the data/training closed loop (FLOWS diagram 5): evolve() found
a policy that solved the task through the SkillOpt acceptance gate; this turns it
into a `mindloop.SkillCard` the runtime brain can *reuse* via the existing P2
skill library — learn → crystallize → plan-from-card, no new runtime machinery.

Trust model (D-L consistent): learned cards carry ``provenance="learned"`` and
**never** any ``safety`` annotations. `Phylax.vet_card` only refuses untrusted
cards that carry safety annotations — so a learned card is honored as a
*procedure* while remaining structurally unable to loosen a single safety rule
(the universal `Policy` invariants bind regardless of what a card says).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mindloop import SkillCard, SkillStep

if TYPE_CHECKING:
    from .evolution import EvolutionResult
    from .task import Task
    from .tree import Attempt

__all__ = ["crystallize"]


def _winning_attempt(result: EvolutionResult) -> Attempt | None:
    """The gate-accepted attempt whose training rollout actually succeeded."""
    for attempt in reversed(tuple(result.tree)):
        if attempt.accepted and attempt.log.success:
            return attempt
    return None


def crystallize(result: EvolutionResult, task: Task) -> SkillCard | None:
    """Turn a solved run's winning policy into a learned `SkillCard`.

    Returns None when nothing won (unsolved run) — expected absence, not an
    error. Node ids are resolved back to scene labels so the card stays
    scene-independent: the P2 compiler re-binds mentions to node ids in
    whatever home it is instantiated in (B9).
    """
    if not result.solved:
        return None
    attempt = _winning_attempt(result)
    if attempt is None:
        return None
    labels = {o.node_id: o.label for o in task.initial_scene}
    steps = tuple(
        SkillStep(
            verb=st.verb,
            target=labels.get(next(iter(st.target_node_ids), -1), ""),
            params=dict(st.params),
        )
        for st in attempt.policy.subtasks
    )
    return SkillCard(
        id=f"learned/{task.name}",
        category=task.name,
        steps=steps,
        provenance="learned",  # untrusted by default; carries no safety keys
    )
