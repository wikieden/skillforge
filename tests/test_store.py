"""TreeStore round-trip — persist an attempt tree and reload its summaries."""

from __future__ import annotations

from pathlib import Path

from bodylink import MockBody
from mindloop import SubTask

from skillforge import (
    Goal,
    Lab,
    MockEnvironment,
    Policy,
    SceneObject,
    ScriptedAuthor,
    Task,
    TreeStore,
)


def _place(loc: tuple[float, float, float]) -> SubTask:
    return SubTask(id="p", verb="place", target_node_ids=(1,), params={"location": loc})


def test_store_roundtrip(tmp_path: Path) -> None:
    task = Task(
        name="move-cube",
        goal=Goal("cube", (1.0, 0.0, 0.0), 0.05),
        initial_scene=(SceneObject(1, "cube", (0.0, 0.0, 0.0)),),
        max_steps=4,
    )
    bad = Policy(subtasks=(_place((0.5, 0.0, 0.0)),))
    good = Policy(subtasks=(_place((1.0, 0.0, 0.0)),))
    result = Lab(MockEnvironment(MockBody(), task.initial_scene), ScriptedAuthor([bad, good])).run(
        task
    )

    store = TreeStore(tmp_path / "tree.sqlite")
    assert store.save(result.tree) == 2

    rows = store.load()
    assert [r.id for r in rows] == [0, 1]
    assert rows[0].parent_id is None
    assert rows[1].parent_id == 0
    assert rows[0].success is False
    assert rows[1].success is True
    assert rows[0].verbs == ("place",)
    assert rows[0].episode == "move-cube#0"
