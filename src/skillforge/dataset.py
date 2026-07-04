"""Trace → dataset curation (step 6, FLOWS diagram 5).

The runtime already emits a `TraceStore` episode for every decision — that
exhaust IS the training corpus. This module turns it into labeled `Case`s under
the ratified data constraints (TASK-FAMILIES §7c):

* **Home-scoped (constraint O/P)**: every case is stamped with a ``home_id`` and
  `merge` refuses to mix homes — no cross-home leakage into a training set.
* **PII scrubbed by default**: `barge_in` / `ask_human` rows carry raw human
  speech in their detail; `curate` redacts the quoted utterance but keeps the
  classified label (the *decision* is the training signal, the *speech* is the
  PII). Everything else in a trace row is already typed (verb, node ids, status,
  error code) and carries no personal text.

Stdlib-only, read-only over the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindloop import TraceStore

__all__ = ["Case", "curate", "merge"]

# Verbs whose trace detail embeds raw human speech (see Brain._trace_barge /
# _trace_ask): the quoted tail is redacted, the classification label survives.
_SPEECH_VERBS: frozenset[str] = frozenset({"barge_in", "ask_human"})
_QUOTED_RE = re.compile(r"['\"].*['\"]", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Case:
    """One labeled training case derived from a trace row."""

    home_id: str
    kind: str  # trace row kind: attempt / failure
    verb: str
    subtask_id: str
    node_ids: tuple[int, ...]
    status: str
    error_code: str | None
    detail: str
    ts: float


def _scrub_detail(verb: str, detail: str) -> str:
    """Redact quoted human speech from speech-bearing rows (constraint P)."""
    if verb not in _SPEECH_VERBS:
        return detail
    return _QUOTED_RE.sub("[redacted]", detail)


def curate(trace: TraceStore, *, home_id: str, scrub: bool = True) -> tuple[Case, ...]:
    """Read every episode of `trace` into home-stamped, scrubbed `Case`s.

    A trace comes from one home's runtime, so the caller names it once; the
    stamp is what `merge` enforces isolation on. Returns oldest-first.
    """
    rows = trace.episodes(limit=1_000_000)  # most recent first
    return tuple(
        Case(
            home_id=home_id,
            kind=r.kind,
            verb=r.verb,
            subtask_id=r.subtask_id,
            node_ids=tuple(r.node_ids),
            status=r.status,
            error_code=r.error_code,
            detail=_scrub_detail(r.verb, r.detail) if scrub else r.detail,
            ts=r.ts,
        )
        for r in reversed(rows)
    )


def merge(*datasets: tuple[Case, ...]) -> tuple[Case, ...]:
    """Concatenate datasets from the SAME home; anything else is refused.

    Cross-home merging is a privacy boundary (constraint O), not a convenience
    trade-off — train per home, never on a neighbours' layout and belongings.
    """
    homes = {c.home_id for ds in datasets for c in ds}
    if len(homes) > 1:
        raise ValueError(f"cross-home merge refused: {sorted(homes)}")
    return tuple(c for ds in datasets for c in ds)
