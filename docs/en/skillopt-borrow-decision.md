# Decision — Borrow SkillOpt's discipline, do not depend on the package

> 🌐 **English** · [中文](../zh/skillopt-borrow-decision.md)
>
> Decision record. Date: 2026-06-30. Status: **decided — BORROW** (the held-out
> gate has shipped; see [`gate.py`](../../src/skillforge/gate.py) + `CHANGELOG`).

## Decision

skillforge **borrows the *discipline*** of [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)
(arXiv 2605.23904, MIT) — porting its ideas as our own small, owned code — and does
**NOT depend on** the `skillopt` PyPI package.

## Why (the crux: a category mismatch)

The two projects optimize categorically different artifacts:

| | trainable artifact | scoring |
|---|---|---|
| **SkillOpt** | a single **natural-language Markdown string** (the "prompt weights" of a frozen LLM), edited by four string ops (`append`/`insert_after`/`replace`/`delete`) | QA correctness `{id, hard, soft}` |
| **skillforge** | a typed `Policy = tuple[SubTask, ...]` (verb + node_ids + params) for a robot body | geometric `Reward(success, value=-distance)` over tempomem Detections |

SkillOpt's edit ops are substring/line surgery on prose; they cannot express a
structured `Policy` mutation (reorder / retarget `SubTask`s). Depending would force a
**lossy `Policy` ↔ Markdown round-trip every step**, discarding the typed-dataclass
guarantee and inviting parse failures mid-loop. The package code (`skillopt/optimizer/skill.py`)
is **zero reuse** for a `SubTask` tuple.

Depending also **breaks two Chronotope contracts**, verified against SkillOpt source:

- **Network-free deterministic tests** — its `ReflACTTrainer` calls a remote optimizer
  LLM (`chat_optimizer`) every step; keeping skillforge's offline `ScriptedAuthor`/
  `MockBody` tests would mean mocking an alpha (v0.1.0) API each step.
- **Independent publish** — it drags `openai` / `azure-identity` / `azure-core` /
  `httpx` into the one member that already imports all others, and exposes **no public
  optimizer API** (`skillopt/__init__.py` exports types only; the real entrypoint is a
  CLI over YAML).

The *valuable* part — the loop discipline — is **policy-representation-agnostic**, so we
keep it as ~40–200 lines of pure Python we own (~1–2 days) instead of weeks of
throwaway impedance-matching glue.

## What we borrowed (and shipped)

| SkillOpt | skillforge | status |
|---|---|---|
| `evaluate_gate`: accept only if `cand_score > current_score` on held-out (ties rejected, pure frozen result) | `gate.py` `evaluate_gate` → `GateResult{accept \| accept_new_best \| reject}` over `Reward.value` | **shipped** |
| train / valid_seen / valid_unseen splits | `evolve(..., val_tasks, env_for)` scores candidates on a non-leaky held-out set; `mock_env_factory` builds a per-task env | **shipped** |
| best_skill.md = argmax on validation | `tree.best()` fed non-degenerate candidates by the gate; `Attempt` gains `holdout_score` + `accepted` | **shipped** |
| four bounded edit ops (string surgery) | a structured **`PolicyEdit`** algebra (insert/delete/replace a `SubTask` by index) — genuinely new, zero SkillOpt reuse | **deferred** |
| edit-budget / textual learning-rate; rejected-edit buffer | features of the future real LLM `PolicyAuthor` behind the `[llm]` extra (no teeth with `ScriptedAuthor`) | **deferred** |

The gate closes the "scripted-to-succeed / no validation gate" gap flagged in the
architecture review: a regressing candidate is now vetoed instead of appended.

## What would flip this to DEPEND (revisit deliberately, not by drift)

1. skillforge's artifact **migrates from a typed `Policy` to a natural-language SkillCard**
   (the `policy.py` note about unifying with mindloop's P2 SkillCard). The category
   mismatch then collapses and SkillOpt's text-edit + LLM-target loop becomes a near-fit.
2. SkillOpt ships a **clean, stdlib-only public optimizer API** that does not require a
   CLI/YAML and does not hard-call a remote LLM in deterministic mode.
3. skillforge gains a **real LLM `PolicyAuthor`** behind `[llm]` **and** a many-task corpus
   — at that scale the epoch / rejected-buffer / textual-LR machinery starts paying off.
4. `skillopt` becomes a transitive dependency anyway via another member (e.g. mindloop
   adopting it for SkillCard optimization), making the marginal cost near-zero.

Keep the borrowed gate / split / edit code modular enough to swap if a flip condition fires.

## References

- microsoft/SkillOpt — README, `skillopt/evaluation/gate.py`, `skillopt/types.py`,
  `skillopt/engine/trainer.py`, `docs/guide/new-benchmark.md`, `pyproject.toml` (MIT,
  Python ≥3.10, Dev Status 3 — Alpha, v0.1.0). Paper: arXiv 2605.23904.
- skillforge: [`gate.py`](../../src/skillforge/gate.py), [`evolution.py`](../../src/skillforge/evolution.py),
  [architecture.md](architecture.md).
