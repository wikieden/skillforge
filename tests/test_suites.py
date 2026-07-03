"""4a slice C — the six family suites run GO against the §7e thresholds, and
the D-F coverage gate + L1 invariants actually bite."""

from __future__ import annotations

from skillforge import ScenarioRunner
from skillforge.suites import all_suites


def test_every_suite_meets_its_bar_with_required_coverage() -> None:
    runner = ScenarioRunner()
    for suite in all_suites():
        report = runner.run(list(suite.scenarios))
        combined = report.combined
        assert not combined.failures, (suite.name, combined.failures)
        assert report.meets(required=suite.required_metrics), (
            suite.name,
            {
                m: (
                    getattr(combined, f"{m}_ok", "?"),
                    getattr(combined, f"{m}_total", "?"),
                )
                for m in sorted(suite.required_metrics)
            },
        )


def test_all_suites_perfect_scores_baseline() -> None:
    runner = ScenarioRunner()
    scenarios = [s for suite in all_suites() for s in suite.scenarios]
    report = runner.run(scenarios)
    c = report.combined
    assert c.planning_accuracy == 1.0
    assert c.policy_compliance == 1.0
    assert c.citation_accuracy == 1.0
    assert c.allergy_block_rate == 1.0
    assert c.interrupt_accuracy == 1.0
    assert c.clarification_precision == 1.0
    assert c.cleaning_coverage == 1.0
    assert report.success_rate == 1.0


def test_coverage_gate_fails_a_suite_missing_its_metric() -> None:
    # run the tour scenarios but demand a metric they never exercise
    runner = ScenarioRunner()
    from skillforge.suites import tour_suite

    report = runner.run(list(tour_suite().scenarios))
    assert not report.meets(required=frozenset({"allergy"}))  # D-F: no vacuous pass
