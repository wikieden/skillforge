"""4b — parametric templates (D-Q): generator-as-oracle, seed determinism,
generated breadth passes the same go/no-go pipeline as the golden suites."""

from __future__ import annotations

from skillforge import ScenarioRunner
from skillforge.templates import find_template, privacy_template, tidy_template


def test_same_seed_same_scenario() -> None:
    a, b = tidy_template(4, seed=7), tidy_template(4, seed=7)
    assert a == b  # frozen dataclasses: full structural equality
    assert tidy_template(4, seed=8) != a


def test_generated_breadth_goes_green() -> None:
    scenarios = (
        [tidy_template(n, seed) for n in (2, 4, 6) for seed in (1, 2, 3)]
        + [privacy_template(seed) for seed in range(5)]
        + [find_template(seed) for seed in range(5)]
    )
    report = ScenarioRunner().run(scenarios)
    c = report.combined
    assert not c.failures
    assert report.meets(required=frozenset({"planning", "policy", "citation", "localization"}))
    assert c.planning_accuracy == 1.0
    assert c.localization_accuracy == 1.0


def test_seed_in_name_for_replay() -> None:
    assert "seed=42" in find_template(42).name
