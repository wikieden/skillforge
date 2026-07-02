"""E — the attempt tree: every policy version + its rollout result, with lineage.

ENPIRE tracks ideas as a git tree of attempts; this is the in-memory analog. Each
node is one `(policy, RolloutLog)`; an edge is "derived from". v1 lineage is a
chain (one author edit per round); branching arrives with fleets (multiple authors
expanding the same frontier).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .policy import Policy
from .rollout import RolloutLog


@dataclass(frozen=True, slots=True)
class Attempt:
    """One node in the attempt tree.

    `holdout_score` is the candidate's score on the held-out set and `accepted`
    is the gate's verdict (see `evolution.evolve`); both default so attempts built
    without a gate (e.g. in unit tests) stay valid.
    """

    id: int
    parent_id: int | None
    policy: Policy
    log: RolloutLog
    author_tokens: int
    holdout_score: float = 0.0
    accepted: bool = True

    @property
    def success(self) -> bool:
        return self.log.success

    @property
    def reward(self) -> float:
        return self.log.reward.value


class AttemptTree:
    """An append-only tree of attempts, keyed by id, ordered by insertion."""

    def __init__(self) -> None:
        self._nodes: dict[int, Attempt] = {}
        self._order: list[int] = []

    def add(self, attempt: Attempt) -> None:
        """Append `attempt`; rejects a duplicate id or an unknown parent."""
        if attempt.id in self._nodes:
            raise ValueError(f"attempt id {attempt.id} already in tree")
        if attempt.parent_id is not None and attempt.parent_id not in self._nodes:
            raise ValueError(f"unknown parent {attempt.parent_id}")
        self._nodes[attempt.id] = attempt
        self._order.append(attempt.id)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Attempt]:
        return (self._nodes[i] for i in self._order)

    def get(self, attempt_id: int) -> Attempt:
        return self._nodes[attempt_id]

    def children(self, attempt_id: int) -> list[Attempt]:
        return [n for n in self if n.parent_id == attempt_id]

    def lineage(self, attempt_id: int) -> list[Attempt]:
        """The root→`attempt_id` chain, oldest first."""
        chain: list[Attempt] = []
        cur: int | None = attempt_id
        while cur is not None:
            node = self._nodes[cur]
            chain.append(node)
            cur = node.parent_id
        chain.reverse()
        return chain

    def best(self) -> Attempt | None:
        """The highest-value attempt (a success outranks any failure)."""
        if not self._nodes:
            return None
        return max(self, key=lambda n: (n.success, n.reward))

    def frontier(self) -> Attempt | None:
        """The node to expand next: the best non-success leaf, else the best leaf."""
        leaves = [n for n in self if not self.children(n.id)]
        pending = [n for n in leaves if not n.success] or leaves
        if not pending:
            return None
        return max(pending, key=lambda n: n.reward)

    def prune(self, predicate: Callable[[Attempt], bool]) -> list[int]:
        """Drop childless attempts matching `predicate` (e.g. hard failures).

        Returns the removed ids. Only leaves are pruned, so lineage stays intact.
        """
        removed: list[int] = []
        for node in list(self):
            if predicate(node) and not self.children(node.id):
                del self._nodes[node.id]
                self._order.remove(node.id)
                removed.append(node.id)
        return removed
