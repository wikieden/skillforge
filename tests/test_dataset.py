"""Step 6 slice 1 — trace→dataset curation (FLOWS diagram 5, constraints O/P):
home-scoped cases, human speech scrubbed, no cross-home merging."""

from __future__ import annotations

import pytest
from mindloop import StatusUpdate, SubTask, TaskStatus, TraceStore

from skillforge.dataset import Case, curate, merge


def _trace_with(*entries: tuple[str, str, str]) -> TraceStore:
    """Build a TraceStore with (verb, message, status) attempt rows."""
    trace = TraceStore(":memory:")
    for i, (verb, message, status) in enumerate(entries):
        sub = SubTask(id=f"s{i}", verb=verb, target_node_ids=(1,))
        st = StatusUpdate(
            "",
            sub.id,
            TaskStatus.SUCCESS if status == "ok" else TaskStatus.FAILED,
            progress=1.0,
            message=message,
        )
        trace.log_attempt(sub, st, session_id="s1")
    return trace


def test_curate_maps_episodes_to_home_stamped_cases() -> None:
    trace = _trace_with(("navigate", "mock navigate ok", "ok"), ("grasp", "mock grasp ok", "ok"))
    cases = curate(trace, home_id="home-a")
    assert len(cases) == 2
    assert all(isinstance(c, Case) and c.home_id == "home-a" for c in cases)
    verbs = {c.verb for c in cases}
    assert verbs == {"navigate", "grasp"}
    trace.close()


def test_scrub_redacts_human_speech_keeps_kind() -> None:
    # barge_in / ask_human rows carry raw utterances = PII (constraint P)
    trace = _trace_with(
        ("barge_in", "barge_in:supersede 'bring my medication to the bedroom'", "ok"),
        ("ask_human", "ask_human -> 'my name is Alice, use the tray'", "ok"),
        ("navigate", "mock navigate ok", "ok"),
    )
    cases = curate(trace, home_id="home-a")  # scrub defaults ON
    by_verb = {c.verb: c for c in cases}
    assert "medication" not in by_verb["barge_in"].detail
    assert "supersede" in by_verb["barge_in"].detail  # the label survives
    assert "Alice" not in by_verb["ask_human"].detail
    assert by_verb["navigate"].detail == "mock navigate ok"  # non-speech untouched
    trace.close()


def test_scrub_off_keeps_raw_detail() -> None:
    trace = _trace_with(("barge_in", "barge_in:abort 'stop'", "ok"))
    (case,) = curate(trace, home_id="home-a", scrub=False)
    assert "'stop'" in case.detail
    trace.close()


def test_merge_same_home_concatenates() -> None:
    t1 = _trace_with(("navigate", "ok1", "ok"))
    t2 = _trace_with(("grasp", "ok2", "ok"))
    a = curate(t1, home_id="home-a")
    b = curate(t2, home_id="home-a")
    assert len(merge(a, b)) == 2
    t1.close()
    t2.close()


def test_merge_cross_home_refused() -> None:
    t1 = _trace_with(("navigate", "ok", "ok"))
    t2 = _trace_with(("grasp", "ok", "ok"))
    a = curate(t1, home_id="home-a")
    b = curate(t2, home_id="home-b")
    with pytest.raises(ValueError, match="cross-home"):
        merge(a, b)
    t1.close()
    t2.close()
