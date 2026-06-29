"""Full-stack integration harness — baseline (network-free) production go/no-go."""

from __future__ import annotations

from mindloop import Command

from skillforge import (
    Scenario,
    ScenarioRunner,
    TurnSpec,
    household_suite,
)


def test_baseline_suite_meets_production_bar() -> None:
    report = ScenarioRunner().run(household_suite())
    # accuracy + safety + replan + success all clear the production thresholds
    assert report.planning_accuracy == 1.0
    assert report.policy_compliance == 1.0
    assert report.replan_success == 1.0
    assert report.success_rate == 1.0
    assert report.meets()
    # traceability: dispatches are auditable (question-only scenarios trace nothing)
    assert sum(r.trace_rows for r in report.results) > 0
    fetch = next(r for r in report.results if r.name == "question_and_fetch")
    assert fetch.trace_rows > 0  # the fetch task left a trace
    # latency is recorded (observational, not asserted exact)
    assert report.p_latency(95) >= 0.0


def test_safety_scenario_blocks_child_and_clears_adult() -> None:
    sc = next(s for s in household_suite() if s.name == "safety_block")
    report = ScenarioRunner().run([sc])
    r = report.results[0]
    # both policy turns scored correctly (child blocked, adult cleared)
    assert r.report.policy_ok == r.report.policy_total == 2


def test_replan_scenario_recovers() -> None:
    sc = next(s for s in household_suite() if s.name == "replan_recover")
    report = ScenarioRunner().run([sc])
    assert report.success_rate == 1.0  # the once-failing grasp recovered via replan
    assert report.results[0].trace_rows >= 2  # failed attempt + recovered attempt traced


def test_report_flags_a_regression() -> None:
    # a scenario whose expectation the system does NOT meet -> meets() False
    bad = Scenario(
        name="wrong",
        objects=(("apple", (1.0, 0.0, 0.9)),),
        plans={"go": (Command("grasp", ("apple",)),)},
        turns=(TurnSpec("go", expect_verbs=("navigate", "grasp")),),  # plan is just grasp
    )
    report = ScenarioRunner().run([bad])
    assert report.planning_accuracy == 0.0
    assert not report.meets()
