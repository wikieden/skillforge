"""PI — the policy author: read the attempt history, return an improved Policy.

This is the coding-agent seam (the PI module). `ScriptedAuthor` is the
deterministic test double — it mirrors `mindloop.ScriptedReasoner`: canned
policies returned in order, so a test can script attempt-1 (fails) → attempt-2
(succeeds) with no network. A real author wraps Codex / Claude Code or an
OpenAI-compatible endpoint behind the `[llm]` extra; it fills the same protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .policy import Policy
from .task import Task

if TYPE_CHECKING:
    from .tree import Attempt


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
