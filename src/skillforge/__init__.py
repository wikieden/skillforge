"""skillforge — an ENPIRE-style autonomous self-learning lab for the Chronotope stack.

The closed loop **EN → PI → R → E**: an Environment that resets + scores (EN), a
PolicyAuthor that rewrites the policy from experience (PI), a budgeted Rollout (R),
and an Evolution loop that compounds attempts into a skill (E). Network-free at the
core (`ScriptedAuthor` + `bodylink.MockBody` + `MockEnvironment`); a real robot + an
LLM author fill the same protocols, with no change to the loop. See
docs/en/architecture.md.
"""

from __future__ import annotations

from .author import LLMPolicyAuthor, PolicyAuthor, ScriptedAuthor
from .crystallize import crystallize
from .dataset import Case, curate, merge
from .environment import (
    STEP_ROBOT_MS,
    EnvFactory,
    Environment,
    MockEnvironment,
    mock_env_factory,
)
from .evolution import EvolutionResult, evolve
from .gate import GateAction, GateResult, evaluate_gate
from .lab import Lab, LabResult
from .metrics import LabMetrics, compute, pass_at_k
from .policy import Policy
from .reward import Reward, distance_to_target, object_at_pose, score
from .rollout import RolloutLog, StepLog, run_episode
from .scenario import (
    IntegrationReport,
    Scenario,
    ScenarioResult,
    ScenarioRunner,
    TurnSpec,
    baseline_brain,
    household_suite,
)
from .store import AttemptSummary, TreeStore
from .suites import Suite, all_suites
from .task import Goal, SceneObject, Task, Vec3
from .templates import find_template, privacy_template, tidy_template
from .tree import Attempt, AttemptTree

__version__ = "0.1.0a0"

__all__ = [
    "STEP_ROBOT_MS",
    "Attempt",
    "AttemptSummary",
    "AttemptTree",
    "Case",
    "EnvFactory",
    "Environment",
    "EvolutionResult",
    "GateAction",
    "GateResult",
    "Goal",
    "IntegrationReport",
    "LLMPolicyAuthor",
    "Lab",
    "LabMetrics",
    "LabResult",
    "MockEnvironment",
    "Policy",
    "PolicyAuthor",
    "Reward",
    "RolloutLog",
    "Scenario",
    "ScenarioResult",
    "ScenarioRunner",
    "SceneObject",
    "ScriptedAuthor",
    "StepLog",
    "Suite",
    "Task",
    "TreeStore",
    "TurnSpec",
    "Vec3",
    "__version__",
    "all_suites",
    "baseline_brain",
    "compute",
    "crystallize",
    "curate",
    "distance_to_target",
    "evaluate_gate",
    "evolve",
    "find_template",
    "household_suite",
    "merge",
    "mock_env_factory",
    "object_at_pose",
    "pass_at_k",
    "privacy_template",
    "run_episode",
    "score",
    "tidy_template",
]
