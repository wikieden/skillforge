# Changelog

All notable changes to `skillforge` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
