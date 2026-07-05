"""T6-12 — LLM-rollout scale-up: the ENPIRE loop over a BATCH of tasks.

K seeded place-object tasks run through evolve(); the report is the research
track's health check: solve rate, mean attempts, author cost. Deterministic by
default (a scripted lose->win author per task); pass ``--llm`` to author every
task with the real host model (Cosmos-Reason2-8B, temp=0 + seed — the same
pinned setup as learning_loop.py).

Run:  uv run python skillforge/examples/rollout_scale.py [--llm] [K]
"""

from __future__ import annotations

import os
import sys

from bodylink import MockBody
from mindloop import CosmosReasonVerbalizer, SubTask

from skillforge import Goal, MockEnvironment, SceneObject, Task, evolve
from skillforge.author import LLMPolicyAuthor, ScriptedAuthor
from skillforge.policy import Policy

HOST = os.environ.get("REAL_LLM_HOST", "172.16.1.12")


def _tasks(k: int) -> list[Task]:
    """K seeded variants: same skill (place the mug), different targets."""
    out = []
    for i in range(k):
        target = (1.0 + 0.5 * (i % 4), 0.5 + 0.25 * (i // 4), 1.0)
        out.append(
            Task(
                name=f"place-mug-{i:02d}",
                goal=Goal(label="mug", target_xyz=target, tolerance_m=0.05),
                initial_scene=(SceneObject(node_id=1, label="mug", xyz=(0.4, 0.4, 0.8)),),
                max_steps=4,
            )
        )
    return out


def _scripted_author(task: Task) -> ScriptedAuthor:
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
                id="w0",
                verb="place",
                target_node_ids=(1,),
                params={"location": task.goal.target_xyz},
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
    args = [a for a in sys.argv[1:] if a != "--llm"]
    k = int(args[0]) if args else (4 if use_llm else 12)
    tasks = _tasks(k)
    print(f"rollout scale-up: {k} tasks, author={'HOST LLM' if use_llm else 'scripted'} …")

    solved = 0
    first_try = 0
    attempts_total = 0
    tokens = 0
    for task in tasks:
        author = _llm_author() if use_llm else _scripted_author(task)
        env = MockEnvironment(MockBody(), task.initial_scene)
        result = evolve(env, author, task, max_attempts=4)
        solved += int(result.solved)
        first_try += int(result.solved and result.attempts == 1)
        attempts_total += result.attempts
        tokens += author.cost_tokens()
        print(f"  {task.name}: solved={result.solved} attempts={result.attempts}")

    rate = solved / k
    print(
        f"\nsolve rate {solved}/{k} = {rate:.2f} · pass@1 {first_try}/{k} "
        f"· mean attempts {attempts_total / k:.2f} · author ~{tokens} tokens"
    )
    print(f"rollout scale-up {'GO' if rate >= 0.9 else 'NO-GO'}")
    return 0 if rate >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
