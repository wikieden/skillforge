"""Step 6 — the data/training closed loop, end to end (FLOWS diagram 5).

    evolve (SkillOpt gate)  →  crystallize (learned SkillCard)
        →  SkillLibrary  →  a FRESH Brain plans FROM the learned card  →  SUCCESS
        →  curate (home-scoped, scrubbed dataset from the run's trace)

Deterministic by default (ScriptedAuthor). Pass ``--llm`` to author with the
real host model instead (Cosmos-Reason2-8B via vLLM, temperature=0 + seed —
the same pinned setup as real_llm_e2e.py).

Run:  uv run python skillforge/examples/learning_loop.py [--llm]
"""

from __future__ import annotations

import os
import sys

import numpy as np
from bodylink import MockBody
from mindloop import (
    Brain,
    CosmosReasonVerbalizer,
    ScriptedIntentRouter,
    ScriptedReasoner,
    SkillCardPlanner,
    SkillLibrary,
    SubTask,
    TraceStore,
)
from tempomem import Detection, TempoMem

from skillforge import Goal, SceneObject, Task, crystallize, curate
from skillforge.author import LLMPolicyAuthor, ScriptedAuthor
from skillforge.environment import MockEnvironment
from skillforge.evolution import evolve
from skillforge.policy import Policy

HOST = os.environ.get("REAL_LLM_HOST", "172.16.1.12")

TASK = Task(
    name="put-mug-on-shelf",
    goal=Goal(label="mug", target_xyz=(2.0, 1.0, 1.2), tolerance_m=0.05),
    initial_scene=(SceneObject(node_id=1, label="mug", xyz=(0.5, 0.5, 0.8)),),
    max_steps=4,
)


def _scripted_author() -> ScriptedAuthor:
    lose = Policy(
        subtasks=(
            SubTask(
                id="l0", verb="place", target_node_ids=(1,), params={"location": (9.0, 9.0, 9.0)}
            ),
        ),
        version=0,
    )
    win = Policy(
        subtasks=(
            SubTask(
                id="w0", verb="place", target_node_ids=(1,), params={"location": (2.0, 1.0, 1.2)}
            ),
        ),
        version=1,
    )
    return ScriptedAuthor([lose, win])


def _llm_author() -> LLMPolicyAuthor:
    vb = CosmosReasonVerbalizer(
        base_url=f"http://{HOST}:8002/v1",
        model="nvidia/Cosmos-Reason2-8B",
        api_key="local",
        max_tokens=4096,
        temperature=0.0,
        extra_body={"seed": 42},
    )
    return LLMPolicyAuthor(vb)


def main() -> int:
    use_llm = "--llm" in sys.argv
    author = _llm_author() if use_llm else _scripted_author()
    print(f"1) evolve with {'the HOST LLM' if use_llm else 'a scripted'} author …")
    env = MockEnvironment(MockBody(), TASK.initial_scene)
    result = evolve(env, author, TASK, max_attempts=4)
    print(
        f"   solved={result.solved} in {result.attempts} attempts "
        f"(author spent ~{author.cost_tokens()} tokens)"
    )
    if not result.solved:
        print("   nothing to crystallize — stopping honestly.")
        return 1

    print("2) crystallize the winner into a learned SkillCard …")
    card = crystallize(result, TASK)
    assert card is not None
    print(f"   {card.id}  provenance={card.provenance}  steps={[s.verb for s in card.steps]}")

    print("3) a FRESH brain plans from the learned card and executes …")
    mem = TempoMem.open(":memory:", embedding_dim=4)
    v = np.ones(4, dtype="float32") / 2.0
    mem.add_detections([Detection("mug", v, (0.5, 0.5, 0.8), (0.4, 0.4, 0.7), (0.6, 0.6, 0.9))])
    mem.commit()
    trace = TraceStore(":memory:")
    body = MockBody()
    brain = Brain(
        mem,
        ScriptedReasoner(),
        router=ScriptedIntentRouter({}, default="task"),
        planner=SkillCardPlanner(SkillLibrary([card]), body.capabilities()),
        executor=body,
        trace=trace,
    )
    r = brain.handle("put the mug on the shelf")
    ok = bool(r.statuses) and all(st.status.value == "success" for st in r.statuses)
    print(f"   dispatched {len(r.statuses)} step(s), all success={ok}")

    print("4) curate the run's trace into a home-scoped dataset …")
    cases = curate(trace, home_id="home-demo")
    print(f"   {len(cases)} case(s), home_id=home-demo, speech scrubbed by default")
    trace.close()

    print("\nlearn → crystallize → reuse → curate: closed." if ok else "\nreuse failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
