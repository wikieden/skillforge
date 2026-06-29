# skillforge

> 🌐 **English** · [中文](README.zh.md)

**An ENPIRE-style autonomous self-learning lab for the Chronotope stack.**

`skillforge` is the 5th Chronotope member: a meta-orchestrator that drives memory
(`tempomem`), eyes (`worldsense`), brain (`mindloop`), and body (`bodylink`) in a
closed loop so a body can **acquire a skill from its own trials** — no human in the
loop. It ports the four modules of NVIDIA GEAR's [ENPIRE](https://research.nvidia.com/labs/gear/enpire/)
into this stack:

```
EN  Environment        reset the world + score the outcome   (environment.py + reward.py)
PI  Policy Improvement  rewrite the policy from experience    (author.py + policy.py)
R   Rollout            run a budgeted trial, log everything   (rollout.py)
E   Evolution          compound attempts into a skill         (evolution.py + tree.py)
```

`Lab.run(task)` turns the crank: an author proposes a policy, a rollout scores it on
the (mock) hardware, the attempt joins a lineage tree, and the loop repeats until the
task is solved or the budget runs out.

## Status — walking skeleton (v0.1.0a0)

The thinnest **complete** vertical slice: 1 task, 1 coding-agent, `MockBody`. It runs
**network-free and GPU-free** — `ScriptedAuthor` + `MockEnvironment` stand in for a
real LLM author and real hardware. Fleet parallelism, branching hypothesis trees, and
the full MRU/MTU economics come after the loop proves it turns on its own.

## Run the demo

```bash
uv sync
uv run python skillforge/examples/01_self_close.py   # prints the attempt tree + pass-rate
uv run pytest skillforge/tests                        # the loop self-closes
```

## How it stays real but light

- The reward bricks (`object_at_pose`, `distance_to_target`) score over real
  `tempomem.Detection`s — the same type a real `worldsense` camera produces.
- `MockEnvironment` *wraps* `bodylink.MockBody` (the executor seam is unchanged) and
  adds the two primitives the stack lacked: **reset** and **reward**.
- Swap in a real `Environment` (robot reset + camera reward) and a real `PolicyAuthor`
  (Codex / Claude Code / an OpenAI-compatible endpoint) — the loop code does not change.

## Where this sits

`skillforge` is a **research track**, parallel to the [mindloop DEV-PLAN](../mindloop/docs/en/devplan/DEV-PLAN.md).
mindloop's `Replanner` repairs one failed step *within* an episode; `skillforge`'s
`Evolution` improves the policy *across* episodes — different altitude, complementary.

See [docs/en/architecture.md](docs/en/architecture.md) for the module map.

> The member name `skillforge` is a working name, pending owner ratification under the
> Chronotope naming convention.
