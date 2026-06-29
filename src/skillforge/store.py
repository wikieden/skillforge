"""Lab-local persistence of the attempt tree (a SQLite sidecar).

A compact, dependency-free record — id, parent, success, reward, tokens, and the
verb plan — enough to reload lineage + outcomes for inspection across runs. Full
object round-trip and migration to the tempomem **L4 Trace store** land at
DEV-PLAN P4; this keeps the numpy-only core untouched meanwhile. Stdlib `sqlite3`
only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .tree import Attempt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempt (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER,
    success INTEGER NOT NULL,
    reward REAL NOT NULL,
    author_tokens INTEGER NOT NULL,
    verbs TEXT NOT NULL,
    episode TEXT NOT NULL
);
"""

_Row = tuple[int, int | None, int, float, int, str, str]


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    """A persisted attempt row, reloaded for inspection."""

    id: int
    parent_id: int | None
    success: bool
    reward: float
    author_tokens: int
    verbs: tuple[str, ...]
    episode: str


def _row(attempt: Attempt) -> _Row:
    verbs = "|".join(step.subtask.verb for step in attempt.log.steps)
    return (
        attempt.id,
        attempt.parent_id,
        int(attempt.success),
        attempt.reward,
        attempt.author_tokens,
        verbs,
        attempt.log.episode,
    )


class TreeStore:
    """Saves/loads an attempt tree as compact rows in a SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)

    def save(self, attempts: Iterable[Attempt]) -> int:
        """Overwrite the store with `attempts`; returns the number written."""
        rows = [_row(a) for a in attempts]
        with sqlite3.connect(self._path) as conn:
            conn.executescript(_SCHEMA)
            conn.execute("DELETE FROM attempt")
            conn.executemany(
                "INSERT INTO attempt"
                " (id, parent_id, success, reward, author_tokens, verbs, episode)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def load(self) -> list[AttemptSummary]:
        """Reload all rows, ordered by id."""
        with sqlite3.connect(self._path) as conn:
            conn.executescript(_SCHEMA)
            rows = conn.execute(
                "SELECT id, parent_id, success, reward, author_tokens, verbs, episode"
                " FROM attempt ORDER BY id"
            ).fetchall()
        return [
            AttemptSummary(
                id=r[0],
                parent_id=r[1],
                success=bool(r[2]),
                reward=r[3],
                author_tokens=r[4],
                verbs=tuple(v for v in r[5].split("|") if v),
                episode=r[6],
            )
            for r in rows
        ]
