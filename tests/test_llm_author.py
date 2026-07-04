"""Step 6 slice 2 — LLMPolicyAuthor: a real PolicyAuthor over a BYO Verbalizer
(network-free here via a scripted stub; the host LLM fills the same seam)."""

from __future__ import annotations

from bodylink import MockBody as Body

from skillforge import Goal, SceneObject, Task
from skillforge.author import LLMPolicyAuthor
from skillforge.environment import MockEnvironment
from skillforge.evolution import evolve


class _StubVerbalizer:
    """Scripted Verbalizer: canned replies in order; records prompts."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._n = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self._replies[min(self._n, len(self._replies) - 1)]
        self._n += 1
        return reply


_TASK = Task(
    name="put-mug-on-shelf",
    goal=Goal(label="mug", target_xyz=(2.0, 1.0, 1.2), tolerance_m=0.05),
    initial_scene=(SceneObject(node_id=1, label="mug", xyz=(0.5, 0.5, 0.8)),),
    max_steps=4,
)


def test_author_parses_place_lines_into_policy() -> None:
    vb = _StubVerbalizer(["place 1 2.0 1.0 1.2"])
    author = LLMPolicyAuthor(vb)
    policy = author.author(_TASK, ())
    assert len(policy.subtasks) == 1
    st = policy.subtasks[0]
    assert st.verb == "place"
    assert st.target_node_ids == (1,)
    assert st.params["location"] == (2.0, 1.0, 1.2)
    assert author.cost_tokens() > 0


def test_malformed_lines_skipped_not_fatal() -> None:
    vb = _StubVerbalizer(["nonsense line\nplace 1 2.0 1.0 1.2\nplace x y z w"])
    policy = LLMPolicyAuthor(vb).author(_TASK, ())
    assert len(policy.subtasks) == 1  # only the valid line survives


def test_prompt_carries_goal_scene_and_history() -> None:
    vb = _StubVerbalizer(["place 1 0.0 0.0 0.0", "place 1 2.0 1.0 1.2"])
    author = LLMPolicyAuthor(vb)
    p1 = author.author(_TASK, ())
    # simulate one failed attempt in history via a real rollout
    env = MockEnvironment(Body(), _TASK.initial_scene)
    from skillforge.rollout import run_episode
    from skillforge.tree import Attempt

    log = run_episode(env, p1, _TASK.goal, episode="t#0", max_steps=4)
    att = Attempt(id=0, parent_id=None, policy=p1, log=log, author_tokens=1)
    author.author(_TASK, (att,))
    prompt2 = vb.prompts[-1]
    assert "mug" in prompt2 and "2.0" in prompt2  # goal + scene
    assert "reward" in prompt2.lower()  # history feedback present


def test_evolve_with_llm_author_solves_when_model_improves() -> None:
    # attempt 1 places wrong, attempt 2 places at goal -> gate accepts, solved
    vb = _StubVerbalizer(["place 1 9.0 9.0 9.0", "place 1 2.0 1.0 1.2"])
    author = LLMPolicyAuthor(vb)
    env = MockEnvironment(Body(), _TASK.initial_scene)
    result = evolve(env, author, _TASK, max_attempts=3)
    assert result.solved
