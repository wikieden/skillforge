"""Step 6 slice 3 — crystallize: a gate-accepted winning policy becomes an
Episteme SkillCard (provenance='learned'), closing the learn->reuse loop."""

from __future__ import annotations

from bodylink import MockBody
from mindloop import Phylax, SkillCard, SubTask
from mindloop import Policy as PhylaxPolicy

from skillforge import Goal, SceneObject, Task
from skillforge.author import ScriptedAuthor
from skillforge.crystallize import crystallize
from skillforge.environment import MockEnvironment
from skillforge.evolution import evolve
from skillforge.policy import Policy

_TASK = Task(
    name="put-mug-on-shelf",
    goal=Goal(label="mug", target_xyz=(2.0, 1.0, 1.2), tolerance_m=0.05),
    initial_scene=(SceneObject(node_id=1, label="mug", xyz=(0.5, 0.5, 0.8)),),
    max_steps=4,
)

_WIN = Policy(
    subtasks=(
        SubTask(id="w0", verb="place", target_node_ids=(1,), params={"location": (2.0, 1.0, 1.2)}),
    ),
    version=1,
)
_LOSE = Policy(
    subtasks=(
        SubTask(id="l0", verb="place", target_node_ids=(1,), params={"location": (9.0, 9.0, 9.0)}),
    ),
    version=0,
)


def _solved():
    env = MockEnvironment(MockBody(), _TASK.initial_scene)
    return evolve(env, ScriptedAuthor([_LOSE, _WIN]), _TASK, max_attempts=3)


def test_crystallize_solved_run_yields_learned_card() -> None:
    result = _solved()
    assert result.solved
    card = crystallize(result, _TASK)
    assert isinstance(card, SkillCard)
    assert card.provenance == "learned"
    assert card.category == _TASK.name
    assert card.steps and card.steps[0].verb == "place"
    assert card.steps[0].target == "mug"  # node id resolved back to a label
    assert not card.safety  # a learned card NEVER carries safety annotations


def test_learned_card_passes_vet_but_cannot_loosen_safety() -> None:
    # D-L consistency: vet_card only refuses untrusted cards WITH safety
    # annotations — a learned card has none, so it is honored as procedure
    # while remaining unable to smuggle safety loosening.
    card = crystallize(_solved(), _TASK)
    gate = Phylax(PhylaxPolicy())  # trusts only "builtin"
    assert card is not None
    assert gate.vet_card(card)


def test_unsolved_run_crystallizes_nothing() -> None:
    env = MockEnvironment(MockBody(), _TASK.initial_scene)
    result = evolve(env, ScriptedAuthor([_LOSE]), _TASK, max_attempts=2)
    assert not result.solved
    assert crystallize(result, _TASK) is None


def test_card_roundtrips_through_from_dict() -> None:
    card = crystallize(_solved(), _TASK)
    assert card is not None
    clone = SkillCard.from_dict(
        {
            "id": card.id,
            "category": card.category,
            "steps": [
                {"verb": s.verb, "target": s.target, "params": dict(s.params)} for s in card.steps
            ],
            "provenance": card.provenance,
        }
    )
    assert clone.steps == card.steps
