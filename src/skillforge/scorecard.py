"""D-P skill scorecard — (verb x body_profile x condition) advisory reliability.

The L2 skill (VLA) eval track: per-bucket success rates from two sources —
production `TraceStore` rows (zero new instrumentation, `observe_trace`) and
sim batch outcomes (`record`, driven by the same SubTask→StatusUpdate seam
against Genesis). `annotate` surfaces the rates onto `VerbSpec.reliability`
so the planner context can disclose weak skills (mindloop side).

Guards (ratified): scores bias planning *preferences* only — Phylax never
reads them (a high score never skips the gate, a low score never exempts);
exploration to fill blind spots happens in sim only, never on a real body in
a user's home. Promotion of a new skill version reuses `gate.py`'s held-out
strict-improvement discipline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindloop import CapabilityManifest, TraceStore

__all__ = ["Scorecard"]

# trace verbs that are not body skills (brain-side decisions) — never scored
_NON_SKILL_VERBS: frozenset[str] = frozenset({"barge_in", "ask_human", "serve_mode"})


class Scorecard:
    """Success counters keyed ``(verb, body_profile, condition bucket)``."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, str], list[int]] = {}

    def record(
        self, verb: str, body_profile: str, condition: str = "default", *, success: bool
    ) -> None:
        """Record one skill outcome (sim batch eval or live telemetry)."""
        ok, total = self._counts.setdefault((verb, body_profile, condition), [0, 0])
        self._counts[(verb, body_profile, condition)] = [ok + int(success), total + 1]

    def rate(self, verb: str, body_profile: str, condition: str = "default") -> float | None:
        """Success rate for the bucket, or None when never tried (expected absence).

        A never-tried bucket stays None rather than 0.0 — the no-self-fulfilling-
        avoidance guard: absence of data must not read as incompetence.
        """
        entry = self._counts.get((verb, body_profile, condition))
        if entry is None or entry[1] == 0:
            return None
        return entry[0] / entry[1]

    def observe_trace(
        self, trace: TraceStore, *, body_profile: str, condition: str = "default"
    ) -> int:
        """Aggregate a production trace's dispatch outcomes (source 2 of D-P).

        Counts terminal success/failed rows for skill verbs; brain-side rows
        (barge_in / ask_human / serve_mode) and skips are not competence
        signals. Returns the number of rows consumed.
        """
        n = 0
        for r in trace.episodes(limit=1_000_000):
            if r.verb in _NON_SKILL_VERBS or r.status not in ("success", "failed"):
                continue
            self.record(r.verb, body_profile, condition, success=r.status == "success")
            n += 1
        return n

    def annotate(self, manifest: CapabilityManifest) -> CapabilityManifest:
        """A copy of `manifest` with each scored verb's `VerbSpec.reliability` filled.

        Advisory only: the manifest's verb *set* and params are untouched, and
        nothing here is read by Phylax. Unscored verbs keep an empty mapping.
        """
        body = manifest.body_profile.profile_id
        verbs = []
        for v in manifest.verbs:
            buckets = {
                cond: self.rate(v.name, body, cond)
                for (verb, b, cond) in self._counts
                if verb == v.name and b == body
            }
            buckets = {c: r for c, r in buckets.items() if r is not None}
            verbs.append(replace(v, reliability=buckets) if buckets else v)
        return replace(manifest, verbs=tuple(verbs))
