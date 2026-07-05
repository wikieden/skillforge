"""T6-12 — sim batch eval: D-P's source 1 (pre-deploy skill scorecard feed).

`run_sim_batch` dispatches each verb N times through the SAME executor seam
production uses (SubTask -> StatusUpdate) and records terminal outcomes into a
`Scorecard`. Deterministic: outcomes come from the injected body, no RNG here.
"""

from __future__ import annotations

from bodylink import MockBody
from mindloop import ErrorCode, StatusUpdate, SubTask, TaskStatus

from skillforge import Scorecard, run_sim_batch


def test_clean_body_scores_perfect_rates() -> None:
    sc = Scorecard()
    n = run_sim_batch(MockBody(), ["navigate", "grasp"], scorecard=sc, episodes=3)
    assert n == 6
    assert sc.rate("navigate", "g1d") == 1.0  # body_profile read off the manifest
    assert sc.rate("grasp", "g1d") == 1.0


class _FlakyGrasp(MockBody):
    """Deterministic flake: every 3rd grasp fails (a real sim's role)."""

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


def test_flaky_verb_rate_lands_between_zero_and_one() -> None:
    sc = Scorecard()
    run_sim_batch(_FlakyGrasp(), ["grasp"], scorecard=sc, episodes=6, condition="rigid")
    assert sc.rate("grasp", "g1d", "rigid") == 4 / 6
    assert sc.rate("grasp", "g1d") is None  # other buckets untouched


def test_batch_feeds_manifest_annotation() -> None:
    sc = Scorecard()
    body = _FlakyGrasp()
    run_sim_batch(body, ["grasp", "navigate"], scorecard=sc, episodes=6, condition="fragile")
    manifest = sc.annotate(body.capabilities())
    by_name = {v.name: v for v in manifest.verbs}
    assert by_name["grasp"].reliability == {"fragile": 4 / 6}
    assert by_name["navigate"].reliability == {"fragile": 1.0}
