"""E2E phase 3 — the family suites through a REAL physics body (Genesis).

Runs every *physically meaningful* baseline scenario through
`SimDeviceAdapter` + `GenesisSimBackend` instead of `MockBody`: the same
brain spine (route → plan → compile → Phylax → dispatch → trace), but each
gated verb now steps a Genesis scene (locomotion approaches the target box,
manipulation settles physics). Scenarios whose expectations are *scripted
body behavior* (fail-once replan drills, barge/clarify fixtures) are skipped
here by design — those test the loop, not the body, and stay on MockBody.

Needs the sim extra:  uv sync --extra sim
Run:                  uv run python skillforge/examples/sim_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# SimDeviceAdapter is bodylink's reference *example* device package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bodylink" / "examples"))
from sim_device import SimDeviceAdapter  # noqa: E402

from mindloop import (  # noqa: E402
    Brain,
    Phylax,
    Policy,
    Replanner,
    ScriptedIntentRouter,
    ScriptedPlanner,
    ScriptedReasoner,
    load_menu,
)
from bodylink import SceneObject, g1d_manifest, make_sim_backend  # noqa: E402

from skillforge import ScenarioRunner, all_suites, household_suite  # noqa: E402
from skillforge.scenario import Scenario  # noqa: E402


def _physical(sc: Scenario) -> bool:
    """Scenarios whose outcomes a physics body can honestly realize."""
    return not sc.fail and not sc.fail_once and not sc.barge_first_step and not sc.clarify_default


def _sim_factory(sc: Scenario, backend) -> "callable":
    """baseline_brain's wiring, with the Genesis-backed device as the body."""

    def factory(mem, trace):
        region_labels = frozenset(label for label, _, _ in sc.regions)
        policy = Phylax(
            Policy(
                adult_only_verbs=frozenset(sc.adult_only),
                adult_only_verb_params=dict(sc.adult_only_verb_params),
                service_permitted_regions=dict(sc.service_permitted_regions),
                known_regions=region_labels if sc.guest_blocked_regions else frozenset(),
                guest_blocked_regions=sc.guest_blocked_regions,
                menu=load_menu("builtin") if sc.menu else None,
            )
        )
        return Brain(
            mem,
            ScriptedReasoner(dict(sc.answers)),
            router=ScriptedIntentRouter(dict(sc.intents), default="task"),  # type: ignore[arg-type]
            planner=ScriptedPlanner(dict(sc.plans)),
            executor=SimDeviceAdapter(g1d_manifest(), backend=backend),
            policy=policy,
            replanner=Replanner(),
            trace=trace,
        )

    return factory


def main() -> int:
    scenarios = [
        sc
        for suite_scs in ([*household_suite()], *([*s.scenarios] for s in all_suites()))
        for sc in suite_scs
        if _physical(sc)
    ]
    print(f"sim e2e: {len(scenarios)} physical scenarios through Genesis (cpu)")

    ok = 0
    t0 = time.perf_counter()
    reports = []
    for sc in scenarios:
        # objects were inserted first in _build_memory -> node ids 1..N in order
        objs = [SceneObject(name, pos) for name, pos in sc.objects]
        node_map = {i + 1: name for i, (name, _) in enumerate(sc.objects)}
        backend = make_sim_backend("genesis", objects=objs, node_map=node_map,
                                   compute_backend="cpu", settle_steps=5)
        try:
            report = ScenarioRunner(brain_factory=_sim_factory(sc, backend)).run([sc])
        finally:
            backend.close()  # gs.destroy() so the next scenario can init fresh
        reports.append((sc.name, report))
        passed = report.meets()
        ok += int(passed)
        print(f"  {'GO ' if passed else 'NO-GO'}  {sc.name}"
              f"  (success={report.success_rate:.0%}, p95={report.p_latency():.0f}ms)")

    dt = time.perf_counter() - t0
    print(f"\n{ok}/{len(scenarios)} scenarios GO through the physics body in {dt:.1f}s")
    return 0 if ok == len(scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
