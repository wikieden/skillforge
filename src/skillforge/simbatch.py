"""Sim batch eval — D-P source 1: pre-deploy skill rates from the sim (T6-12).

Dispatch each verb N times through the SAME executor seam production uses
(`SubTask` -> `StatusUpdate` stream) — a `SimDeviceAdapter` over Genesis for
the real nightly, `MockBody`/scripted flakes in CI — and record every terminal
outcome into a `Scorecard`. Deterministic by construction: this module has no
randomness; variation comes from the injected body/sim. The scorecard's
`annotate` then surfaces the rates as advisory `VerbSpec.reliability` (never
read by Phylax — the D-P guard), and the exploration that fills blind spots
happens HERE, in sim, never on a real body in a user's home.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mindloop import SubTask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mindloop import ExecutorAdapter

    from .scorecard import Scorecard

__all__ = ["run_sim_batch"]


def run_sim_batch(
    executor: ExecutorAdapter,
    verbs: Sequence[str],
    *,
    scorecard: Scorecard,
    body_profile: str | None = None,
    condition: str = "default",
    episodes: int = 5,
    node_ids: tuple[int, ...] = (1,),
) -> int:
    """Run ``episodes`` dispatches of each verb; record outcomes into `scorecard`.

    ``body_profile`` defaults to the executor manifest's profile id, so sim
    rates land in the same (verb x body x condition) buckets production traces
    feed. Returns the number of dispatches recorded. Only terminal
    success/failed outcomes count (aborted/paused runs are not competence
    signals — same rule as `Scorecard.observe_trace`).
    """
    body = body_profile or executor.capabilities().body_profile.profile_id
    n = 0
    for verb in verbs:
        for i in range(episodes):
            sub = SubTask(id=f"simbatch-{verb}-{i:03d}", verb=verb, target_node_ids=node_ids)
            last = None
            for update in executor.dispatch(sub):
                last = update
            if last is None or last.status.value not in ("success", "failed"):
                continue
            scorecard.record(verb, body, condition, success=last.status.value == "success")
            n += 1
    return n
