# skillforge architecture — the EN-PI-R-E loop

> 🌐 **English** · [中文](../zh/architecture.md)

`skillforge` ports the four modules of NVIDIA GEAR's
[ENPIRE](https://research.nvidia.com/labs/gear/enpire/) — **EN**vironment,
**P**olicy **I**mprovement, **R**ollout, **E**volution — into the Chronotope stack
as a single, closed self-learning loop. This is the **walking skeleton**: the
thinnest complete slice (1 task, 1 author, `MockBody`), network-free and GPU-free.

## The loop

```
                 ┌──────────────────────────────────────────────┐
   Task ───────► │  PI  author.author(task, history) → Policy    │
                 │   │                                            │
                 │   ▼                                            │
                 │  R   run_episode(env, policy) → RolloutLog     │
                 │   │     env.reset → step·step·… → env.reward   │
                 │   ▼                                            │
                 │  E   tree.add(Attempt)  ──┐                    │
                 │       solved? ── no ───────┘ (loop)            │
                 └───────────────────────────────────────────────┘
                        EN = the env the rollout drives
```

`Lab.run(task)` runs it until the task is solved or the attempt budget is spent,
then scores the run.

## Module map

| ENPIRE | Module(s) | Responsibility |
|---|---|---|
| **EN** | `environment.py`, `reward.py` | `Environment` protocol — `reset` + `step` + `observe` + `reward`. `MockEnvironment` owns a deterministic mock world and wraps `bodylink.MockBody`. Reward bricks score over `tempomem.Detection`s. |
| **PI** | `author.py`, `policy.py` | `PolicyAuthor` proposes an improved `Policy` from the attempt history. `ScriptedAuthor` is the network-free double; a real LLM author fills the same protocol. |
| **R** | `rollout.py` | `run_episode` resets, runs the policy's verb plan under a step budget via the executor seam, and scores the goal → `RolloutLog`. |
| **E** | `evolution.py`, `tree.py` | `AttemptTree` records every `(policy, RolloutLog)` with lineage; `evolve` is the round loop. |
| — | `metrics.py` | `pass_at_k` + the utilization pair MRU (robot time) / MTU (author tokens). |
| — | `store.py` | A SQLite sidecar persisting the attempt tree for cross-run inspection. |
| — | `lab.py` | `Lab.run(task)` — the whole loop as one call. |

## Why these are the gaps it fills

The brain's `ExecutorAdapter` (`mindloop.execute`) is **stateless** — it dispatches a
`SubTask` and streams a `StatusUpdate`, nothing more. A closed learning loop needs two
primitives the stack never had: **reset** (return the world to a known state) and
**reward** (score the outcome). `Environment` adds exactly those, and `MockEnvironment`
supplies them deterministically — so `bodylink` stays unchanged and tests need no
hardware. Outcome scoring (`pass@k`) and the budget metrics (MRU/MTU) are likewise
net-new; `mindloop.eval` previously scored only citation compliance.

## Reserved seams (real hardware, same protocols)

- A real `Environment`: `reset` = a robot homing routine; `reward` = `worldsense`
  multi-camera `Detection`s through the same `reward.py` bricks.
- A real `PolicyAuthor`: Codex / Claude Code via subprocess, or an OpenAI-compatible
  endpoint (mirrors `mindloop.CosmosReasonVerbalizer` — point `base_url` at the host).
- `store.py` graduates to the tempomem **L4 Trace store** at mindloop DEV-PLAN P4.

## Relationship to mindloop

A parallel **research track**, not a rewrite. mindloop's `Replanner` (DEV-PLAN P4)
repairs one failed step *within* an episode; `skillforge`'s `Evolution` improves the
policy *across* episodes. Different altitude, complementary. `skillforge` consumes the
other four members as libraries and does not block on the mindloop phases.
