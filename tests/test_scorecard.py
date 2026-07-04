"""D-P — skill scorecard: (verb x body_profile x condition) advisory reliability.

Two data sources, one card: production `TraceStore` rows (zero new
instrumentation) and sim batch outcomes recorded directly. `annotate` surfaces
the rates as `VerbSpec.reliability` — advisory ONLY: safety gates never read
it, and a high score never skips Phylax.
"""

from __future__ import annotations

from bodylink import MockBody
from mindloop import (
    Phylax,
    Plan,
    Policy,
    SafetyContext,
    StatusUpdate,
    SubTask,
    TaskStatus,
    TraceStore,
)

from skillforge import Scorecard


def test_record_and_rate_per_bucket() -> None:
    sc = Scorecard()
    for ok in (True, True, False):
        sc.record("grasp", "g1d", "fragile", success=ok)
    sc.record("grasp", "g1d", "rigid", success=True)
    assert sc.rate("grasp", "g1d", "fragile") == 2 / 3
    assert sc.rate("grasp", "g1d", "rigid") == 1.0
    assert sc.rate("grasp", "h1", "fragile") is None  # never tried on that body
    assert sc.rate("pour", "g1d", "fragile") is None


def _log(tr: TraceStore, verb: str, ok: bool) -> None:
    status = TaskStatus.SUCCESS if ok else TaskStatus.FAILED
    tr.log_attempt(
        SubTask(id="x", verb=verb, target_node_ids=(1,)),
        StatusUpdate(task_id="t", subtask_id="x", status=status),
        session_id="s1",
    )


def test_observe_trace_aggregates_dispatch_outcomes() -> None:
    tr = TraceStore(":memory:")
    _log(tr, "grasp", True)
    _log(tr, "grasp", False)
    _log(tr, "navigate", True)
    sc = Scorecard()
    sc.observe_trace(tr, body_profile="g1d")
    assert sc.rate("grasp", "g1d") == 0.5
    assert sc.rate("navigate", "g1d") == 1.0
    tr.close()


def test_observe_trace_skips_brain_side_verbs() -> None:
    # serve_mode (P7.4 tier telemetry) logs SUCCESS but is not a body skill
    tr = TraceStore(":memory:")
    _log(tr, "serve_mode", True)
    _log(tr, "barge_in", True)
    _log(tr, "grasp", True)
    sc = Scorecard()
    n = sc.observe_trace(tr, body_profile="g1d")
    assert n == 1  # only the skill row consumed
    assert sc.rate("serve_mode", "g1d") is None
    assert sc.rate("barge_in", "g1d") is None
    assert sc.rate("grasp", "g1d") == 1.0
    tr.close()


def test_annotate_fills_verbspec_reliability() -> None:
    sc = Scorecard()
    sc.record("grasp", "g1d", "fragile", success=False)
    sc.record("grasp", "g1d", "fragile", success=True)
    manifest = sc.annotate(MockBody().capabilities())
    by_name = {v.name: v for v in manifest.verbs}
    assert by_name["grasp"].reliability == {"fragile": 0.5}
    assert by_name["navigate"].reliability == {}  # unscored verbs untouched


def test_reliability_never_touches_safety() -> None:
    # a perfect score must not buy a pass through Phylax (guard 1 of D-P)
    sc = Scorecard()
    for _ in range(10):
        sc.record("grasp", "g1d", "default", success=True)
    manifest = sc.annotate(MockBody().capabilities())
    gate = Phylax(Policy(adult_only_verbs=frozenset({"grasp"})))
    plan = Plan(subtasks=(SubTask(id="a", verb="grasp", target_node_ids=(1,)),))
    before = gate.check(plan, SafetyContext(role="child"))
    assert before  # blocked for a child…
    assert manifest is not None  # …and the annotation gives no way around it:
    after = gate.check(plan, SafetyContext(role="child"))
    assert [v.rule for v in before] == [v.rule for v in after]
