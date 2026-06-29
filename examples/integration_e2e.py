"""Production e2e integration demo — run the household suite, print the go/no-go.

Baseline (network-free, deterministic) by default:

    uv run python skillforge/examples/integration_e2e.py

The real-LLM validation pass swaps a `FailoverVerbalizer`→host reasoner into a
custom `brain_factory` and runs the same suite on the GPU host (P6/host).
"""

from __future__ import annotations

from skillforge import ScenarioRunner, household_suite


def main() -> None:
    report = ScenarioRunner().run(household_suite())

    print("\n=== Chronotope full-stack integration (baseline) ===\n")
    for r in report.results:
        rep = r.report
        print(
            f"  {r.name:<20} "
            f"plan {rep.planning_ok}/{rep.planning_total}  "
            f"policy {rep.policy_ok}/{rep.policy_total}  "
            f"replan {rep.replan_ok}/{rep.replan_total}  "
            f"success {r.success_ok}/{r.success_total}  "
            f"trace={r.trace_rows}"
        )

    print("\n  --- aggregate (production bar) ---")
    print(f"  planning accuracy : {report.planning_accuracy:.0%}")
    print(f"  policy compliance : {report.policy_compliance:.0%}")
    print(f"  replan success    : {report.replan_success:.0%}")
    print(f"  task success rate : {report.success_rate:.0%}")
    print(f"  p95 latency       : {report.p_latency(95):.1f} ms")
    verdict = "GO ✅" if report.meets() else "NO-GO ❌"
    print(f"\n  verdict: {verdict}\n")


if __name__ == "__main__":
    main()
