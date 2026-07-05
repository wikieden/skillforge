"""T6-12 — the nightly skill-scorecard batch (D-P source 1), end to end.

    sim batch (executor seam) -> Scorecard rates -> annotate(manifest)
        -> the planner context discloses the weak skill (D-P advisory)

Network-free by default: a deterministic flaky body plays the sim. Swap in
`SimDeviceAdapter(load_manifest("g1d"), backend=make_sim_backend("genesis", ...))`
for the real physics nightly — same seam, same scorecard.

Run:  uv run python skillforge/examples/scorecard_nightly.py
"""

from __future__ import annotations

from bodylink import MockBody
from mindloop import ErrorCode, StatusUpdate, SubTask, TaskStatus

from skillforge import Scorecard, run_sim_batch

VERBS = ["navigate", "grasp", "place", "wipe", "vacuum"]


class _FlakyGrasp(MockBody):
    """Deterministic stand-in for a physics sim: every 3rd grasp slips."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def _execute(self, subtask: SubTask) -> StatusUpdate:
        if subtask.verb == "grasp":
            self._n += 1
            if self._n % 3 == 0:
                return StatusUpdate(
                    task_id="",
                    subtask_id=subtask.id,
                    status=TaskStatus.FAILED,
                    error_code=ErrorCode.GRASP_FAILED,
                    touched_node_ids=subtask.target_node_ids,
                )
        return super()._execute(subtask)


def main() -> int:
    body = _FlakyGrasp()
    sc = Scorecard()
    profile = body.capabilities().body_profile.profile_id
    print(f"1) sim batch: {len(VERBS)} verbs x 9 episodes on {profile} …")
    n = run_sim_batch(body, VERBS, scorecard=sc, episodes=9, condition="fragile")
    print(f"   {n} dispatches recorded")

    print("2) scorecard rates (verb @ fragile):")
    for v in VERBS:
        print(f"   {v:<10} {sc.rate(v, 'g1d', 'fragile'):.2f}")

    print("3) annotate the manifest -> D-P advisory reliability:")
    manifest = sc.annotate(body.capabilities())
    weak = {
        v.name: v.reliability
        for v in manifest.verbs
        if v.reliability.get("fragile", 1.0) < 0.8
    }
    for name, rel in weak.items():
        print(f"   {name}: {rel}  (below the 0.8 disclosure bar -> planner context annotates it)")

    ok = weak.get("grasp", {}).get("fragile") == 6 / 9
    print(f"\nnightly scorecard {'GO' if ok else 'NO-GO'} — weak skill surfaced, safety untouched")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
