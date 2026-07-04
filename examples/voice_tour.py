"""Step 7 — a voice-narrated home tour with a voice barge-in (network-free).

The whole voice path through existing seams, no audio stack:

    MockListener (ASR)  ─►  VoiceLoop.turn  ─►  Brain.converse (tour plan)
    guest speaks mid-tour  ─►  ListenerInterruptSource (P7.3)  ─►  barge-in
        (chitchat: answered inline via the Speaker, tour resumes)
    every narration line  ─►  MockSpeaker (TTS)

Swap the mocks for voicelink's `EndpointListener`/`EndpointSpeaker` and the
same loop runs against a real Qwen3-ASR/TTS endpoint; the streaming mic/VAD
fill of ``audio_source`` is the deferred hardware piece.

Run:  uv run python skillforge/examples/voice_tour.py
"""

from __future__ import annotations

from collections import deque

import numpy as np
from bodylink import MockBody
from mindloop import (
    Brain,
    Command,
    Conversation,
    ListenerInterruptSource,
    MockListener,
    MockSpeaker,
    Phylax,
    Policy,
    ScriptedIntentRouter,
    ScriptedPlanner,
    ScriptedReasoner,
    TraceStore,
    VoiceLoop,
)
from tempomem import Detection, TempoMem

TOUR = "show our guest around"
BARGE = "what a lovely kettle!"


def _mem() -> TempoMem:
    mem = TempoMem.open(":memory:", embedding_dim=4)
    mem.define_region("kitchen", (0.0, 0.0, 0.0), (3.0, 3.0, 2.5))
    mem.define_region("living room", (3.0, 0.0, 0.0), (6.0, 3.0, 2.5))
    v = np.ones(4, dtype="float32") / 2.0
    mem.add_detections([Detection("kettle", v, (1.0, 1.0, 0.9), (0.9, 0.9, 0.8), (1.1, 1.1, 1.0))])
    mem.commit()
    return mem


def main() -> int:
    mem = _mem()
    speaker = MockSpeaker()
    # the guest's mid-tour remark arrives as ONE pending audio chunk
    barge_audio = deque([b"<guest-audio>"])
    interrupts = ListenerInterruptSource(
        MockListener([BARGE]),
        audio_source=lambda: barge_audio.popleft() if barge_audio else None,
    )
    brain = Brain(
        mem,
        ScriptedReasoner({BARGE: "Thank you — it's the kettle [#1] we use every morning."}),
        router=ScriptedIntentRouter({BARGE: "chat"}, default="task"),
        planner=ScriptedPlanner(
            {
                TOUR: (
                    Command("navigate", ("kitchen",)),
                    Command("present", ("kitchen",)),
                    Command("navigate", ("living room",)),
                    Command("present", ("living room",)),
                    Command("speak", ("Here we are — that's the whole home!",)),
                )
            }
        ),
        executor=MockBody(),
        policy=Phylax(Policy()),
        trace=TraceStore(":memory:"),
        interrupts=interrupts,
        speaker=speaker,
    )
    loop = VoiceLoop(brain, MockListener([TOUR]), speaker)

    print("1) guest asks for a tour (voice) …")
    result = loop.turn(b"<mic>", Conversation(), person="guest")
    steps = [(s.subtask_id, s.status.value) for s in result.statuses]
    ok = bool(steps) and all(st == "success" for _, st in steps)
    print(f"   dispatched {len(steps)} step(s), all success={ok}")

    print("2) what the robot said (narration + inline barge answer):")
    for line in speaker.said:
        print(f"   🗣  {line}")

    barged = any("kettle" in s for s in speaker.said)
    print(f"\nvoice tour {'GO' if ok and barged else 'NO-GO'} — "
          f"barge answered inline={barged}, tour completed={ok}")
    return 0 if ok and barged else 1


if __name__ == "__main__":
    raise SystemExit(main())
