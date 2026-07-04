"""D-G composite reward shaping — reward = w_task·task + w_handle·handled + w_honesty·honesty.

Ratified defaults 0.6 / 0.25 / 0.15 with the strict ordering w_task > w_handle >
w_honesty (TASK-FAMILIES D-G): the loop must never learn to prefer graceful
declining over actually completing the task. `meets()` stays gated on task
success only — composite is the ENPIRE training signal, not the go/no-go.
"""

import pytest
from mindloop import EvalReport

from skillforge import Weights, composite_score
from skillforge.scenario import IntegrationReport, ScenarioResult


def test_default_weights_are_ratified() -> None:
    w = Weights()
    assert (w.task, w.handle, w.honesty) == (0.6, 0.25, 0.15)


def test_weights_ordering_enforced() -> None:
    # w_task > w_handle > w_honesty is the D-G invariant, not a suggestion
    with pytest.raises(ValueError):
        Weights(task=0.2, handle=0.5, honesty=0.3)
    with pytest.raises(ValueError):
        Weights(task=0.5, handle=0.2, honesty=0.3)


def test_composite_score_math() -> None:
    assert composite_score(1.0, 0.0, 1.0) == pytest.approx(0.75)
    assert composite_score(0.0, 1.0, 0.0) == pytest.approx(0.25)


def _scenario_result(report: EvalReport, ok: int, total: int) -> ScenarioResult:
    return ScenarioResult(
        name="s",
        report=report,
        success_ok=ok,
        success_total=total,
        latencies_ms=[],
        trace_rows=0,
    )


def test_integration_report_composite_score() -> None:
    rep = EvalReport()
    rep.honesty_ok, rep.honesty_total = 1, 2  # 0.5
    rep.handle_ok, rep.handle_total = 1, 1  # 1.0
    ir = IntegrationReport(results=[_scenario_result(rep, ok=1, total=2)])  # task 0.5
    # 0.6*0.5 + 0.25*1.0 + 0.15*0.5 = 0.625
    assert ir.composite_score() == pytest.approx(0.625)


def test_meets_ignores_composite_gates_task_only() -> None:
    # handled-success alone must never flip the go/no-go (D-G: truthful
    # not-found is handled_success=1, task_success=0 — tests honesty, not
    # competence)
    rep = EvalReport()
    rep.handle_ok, rep.handle_total = 5, 5
    ir = IntegrationReport(results=[_scenario_result(rep, ok=0, total=1)])
    assert ir.composite_score() > 0.0
    assert not ir.meets()
