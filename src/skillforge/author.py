"""PI — the policy author: read the attempt history, return an improved Policy.

This is the coding-agent seam (the PI module). `ScriptedAuthor` is the
deterministic test double — it mirrors `mindloop.ScriptedReasoner`: canned
policies returned in order, so a test can script attempt-1 (fails) → attempt-2
(succeeds) with no network. `LLMPolicyAuthor` is the real one (step 6): it hands
the task + attempt history to a BYO `Verbalizer` (the same seam the brain's
reasoner uses — host Cosmos/Qwen at runtime, a scripted stub in tests) and
parses the reply back into a `Policy`. Both fill the same protocol.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mindloop import SubTask

from .policy import Policy
from .task import Task

if TYPE_CHECKING:
    from tempomem import Verbalizer

    from .tree import Attempt

_log = logging.getLogger("skillforge.author")


@runtime_checkable
class PolicyAuthor(Protocol):
    """Proposes the next policy from the task + the attempts so far."""

    def author(self, task: Task, history: Sequence[Attempt]) -> Policy: ...

    def cost_tokens(self) -> int:
        """Cumulative tokens this author has spent (feeds the MTU metric)."""
        ...


class ScriptedAuthor:
    """A deterministic `PolicyAuthor` returning canned policies in order.

    Call ``n`` returns ``policies[min(n, last)]`` and charges `tokens_per_call` —
    enough to drive a self-closing test and to make the token-utilization metric
    non-trivial, with no LLM.
    """

    def __init__(self, policies: Sequence[Policy], *, tokens_per_call: int = 1000) -> None:
        if not policies:
            raise ValueError("ScriptedAuthor needs at least one policy")
        self._policies = tuple(policies)
        self._tokens_per_call = tokens_per_call
        self._spent = 0

    def author(self, task: Task, history: Sequence[Attempt]) -> Policy:
        idx = min(len(history), len(self._policies) - 1)
        self._spent += self._tokens_per_call
        return self._policies[idx]

    def cost_tokens(self) -> int:
        return self._spent


_AUTHOR_PROMPT = (
    "You are a robot skill author. Improve the plan so the goal is met.\n"
    "GOAL: move the object labelled {label!r} (node #{node_id}) to "
    "[{tx:.2f}, {ty:.2f}, {tz:.2f}] within {tol:.2f} m (world frame, meters).\n"
    "SCENE: {scene}\n"
    "HISTORY (older first):\n{history}\n"
    "Reply ONLY with plan lines, one per line, in the exact form:\n"
    "place <node_id> <x> <y> <z>\n"
)


class LLMPolicyAuthor:
    """A `PolicyAuthor` over a BYO `Verbalizer` (step 6 — the real PI module).

    Each call sends goal + scene + per-attempt feedback (plan, reward, success)
    and parses ``place <node_id> <x> <y> <z>`` lines into a `Policy`. Malformed
    lines are skipped (logged), never fatal — the SkillOpt gate downstream vetoes
    a bad policy by score, not by parse errors. Token cost is estimated at
    ~4 chars/token over prompt + reply (feeds the MTU metric).
    """

    def __init__(self, verbalizer: Verbalizer) -> None:
        self._vb = verbalizer
        self._spent = 0
        self._version = 0

    def author(self, task: Task, history: Sequence[Attempt]) -> Policy:
        goal = task.goal
        node_id = next((o.node_id for o in task.initial_scene if o.label == goal.label), 0)
        scene = "; ".join(
            f"node #{o.node_id} {o.label!r} at [{o.xyz[0]:.2f}, {o.xyz[1]:.2f}, {o.xyz[2]:.2f}]"
            for o in task.initial_scene
        )
        lines = []
        for a in history:
            plan = (
                " | ".join(
                    f"{s.verb} {','.join(map(str, s.target_node_ids))} -> {s.params.get('location')}"
                    for s in a.policy.subtasks
                )
                or "(empty)"
            )
            lines.append(
                f"attempt {a.id}: plan [{plan}] reward={a.log.reward.value:.4f} "
                f"success={a.log.success}"
            )
        prompt = _AUTHOR_PROMPT.format(
            label=goal.label,
            node_id=node_id,
            tx=goal.target_xyz[0],
            ty=goal.target_xyz[1],
            tz=goal.target_xyz[2],
            tol=goal.tolerance_m,
            scene=scene,
            history="\n".join(lines) or "(none yet)",
        )
        reply = self._vb.complete(prompt)
        self._spent += (len(prompt) + len(reply)) // 4
        self._version += 1
        return Policy(
            subtasks=_parse_place_lines(reply),
            version=self._version,
            notes=f"llm author, {len(history)} prior attempts",
        )

    def cost_tokens(self) -> int:
        return self._spent


def _parse_place_lines(text: str) -> tuple[SubTask, ...]:
    """Parse ``place <node_id> <x> <y> <z>`` lines; skip anything malformed."""
    subtasks: list[SubTask] = []
    for i, raw in enumerate(text.splitlines()):
        parts = raw.strip().lstrip("-*0123456789. ").split()
        if len(parts) != 5 or parts[0].lower() != "place":
            continue
        try:
            nid = int(parts[1])
            loc = (float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            _log.debug("author line skipped (unparseable): %r", raw)
            continue
        subtasks.append(
            SubTask(
                id=f"llm-{i:02d}",
                verb="place",
                target_node_ids=(nid,),
                params={"location": loc},
            )
        )
    return tuple(subtasks)
