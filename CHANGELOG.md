# Changelog

All notable changes to `skillforge` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Real-LLM validation pass** (`examples/real_llm_e2e.py`, INTEGRATION-E2E
  phase 2 / build step 5) — the full brain driven by the production reasoner
  (Cosmos-Reason2-8B on the RTX PRO host via vLLM): real intent routing, real
  planning -> compile -> Phylax, real cited answers; `temperature=0` + fixed
  seed (reproducible). 4/4 scenarios GO (intent, canonical fetch plan, child
  safety block, guest region privacy, citations), turn p95 ~0.25 s.


- **Sim-body e2e pass** (`examples/sim_e2e.py`, INTEGRATION-E2E phase 3) — every
  physically-meaningful baseline scenario (21) runs through `SimDeviceAdapter` +
  `GenesisSimBackend` (real physics stepping, CPU) instead of `MockBody`:
  same brain spine, gated verbs realized in a Genesis scene. 21/21 GO in ~48 s;
  scripted-body drills (fail-once/barge/clarify fixtures) stay on MockBody by design.


- **Acceptance gate for the Evolution loop** (`gate.py`) — discipline ported (not
  code) from microsoft/SkillOpt: `evaluate_gate` accepts a candidate policy only
  when its held-out score **strictly improves** on the current accepted score
  (ties rejected) → `GateResult{accept | accept_new_best | reject}`, a pure
  stdlib function over `Reward.value`. `evolve()` now scores each candidate, vetoes
  regressions (rejected attempts are still recorded for the audit trail but do not
  advance the lineage), and accepts `val_tasks` + an `env_for` `EnvFactory`
  (`mock_env_factory`) to gate on a **non-leaky held-out split** rather than the
  task it optimizes on; `Attempt` gains `holdout_score` + `accepted`. Closes the
  "scripted-to-succeed / no validation gate" gap from the architecture review.
- **Full-stack integration scenario harness** (`scenario.py`) — the production
  e2e capstone. `ScenarioRunner.run(household_suite())` drives a production-wired
  `mindloop.Brain` (Phylax safety + bounded Replanner + TraceStore + world-state +
  scene-context) through household `Scenario`s and scores the owner's production
  bar: planning accuracy, policy compliance, replan success, task **success rate**,
  p95 **latency**, plus an auditable **trace** per scenario. `IntegrationReport.meets()`
  is one go/no-go. **Baseline mode is fully deterministic + network-free** (Scripted
  reasoner/planner + a fail-once body) — the CI gate; the same harness takes a
  custom `brain_factory` for the real-LLM (FailoverVerbalizer→host) + Genesis-sim
  validation pass. `household_suite()` covers question / fetch / replan-recover /
  safety-block / multi-turn-dialogue. Runnable: `examples/integration_e2e.py`.
  See docs/en/INTEGRATION-E2E.md.

## [0.1.0a0] — 2026-06-18

### Added

- Initial **walking skeleton**: the closed EN-PI-R-E loop as the 5th Chronotope
  member — the thinnest complete self-learning slice (1 task, 1 agent, `MockBody`),
  network-free and GPU-free.
- **EN** — `Environment` protocol (`reset` / `step` / `observe` / `reward`) +
  deterministic `MockEnvironment` over `bodylink.MockBody`; reward bricks
  (`object_at_pose`, `distance_to_target`, `score`) scoring over `tempomem.Detection`s.
- **PI** — `PolicyAuthor` protocol + `ScriptedAuthor` test double; the `Policy` artifact.
- **R** — `run_episode` budgeted rollout + `RolloutLog`.
- **E** — `AttemptTree` (lineage / best / frontier / prune) + the `evolve` loop.
- Metrics — `pass_at_k`, MRU, MTU (`compute` → `LabMetrics`).
- `TreeStore` — a SQLite sidecar for the attempt tree (graduates to the tempomem L4
  Trace store at mindloop DEV-PLAN P4).
- `Lab.run(task)` one-call entry; `examples/01_self_close.py` demo.
- Registered as a workspace member in the root `pyproject.toml`.

[Unreleased]: https://github.com/wikieden/mindloop/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://github.com/wikieden/mindloop/releases/tag/v0.1.0a0
