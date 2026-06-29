# skillforge 架构 — EN-PI-R-E 闭环

> 🌐 [English](../en/architecture.md) · **中文**

`skillforge` 把 NVIDIA GEAR
[ENPIRE](https://research.nvidia.com/labs/gear/enpire/) 的四模块——**EN**vironment、
**P**olicy **I**mprovement、**R**ollout、**E**volution——移植进 Chronotope 栈，成为一条
封闭的自学习闭环。当前是**走路骨架**：最薄的完整纵切（1 任务、1 author、`MockBody`），
无网络、无 GPU。

## 闭环

```
                 ┌──────────────────────────────────────────────┐
   Task ───────► │  PI  author.author(task, history) → Policy    │
                 │   │                                            │
                 │   ▼                                            │
                 │  R   run_episode(env, policy) → RolloutLog     │
                 │   │     env.reset → step·step·… → env.reward   │
                 │   ▼                                            │
                 │  E   tree.add(Attempt)  ──┐                    │
                 │       solved? ── no ───────┘ （循环）          │
                 └───────────────────────────────────────────────┘
                        EN = rollout 驱动的那个 env
```

`Lab.run(task)` 转动闭环，直到解决或预算耗尽，再给整轮评分。

## 模块图

| ENPIRE | 模块 | 职责 |
|---|---|---|
| **EN** | `environment.py`、`reward.py` | `Environment` 协议——`reset` + `step` + `observe` + `reward`。`MockEnvironment` 持有确定性 mock 世界并包裹 `bodylink.MockBody`。reward 砖块在 `tempomem.Detection` 上评分。 |
| **PI** | `author.py`、`policy.py` | `PolicyAuthor` 据尝试历史提出改进的 `Policy`。`ScriptedAuthor` 是无网络替身；真 LLM author 填同一协议。 |
| **R** | `rollout.py` | `run_episode` 复位、按步预算经 executor seam 跑策略的 verb 计划、给目标评分 → `RolloutLog`。 |
| **E** | `evolution.py`、`tree.py` | `AttemptTree` 记录每个 `(policy, RolloutLog)` 及谱系；`evolve` 是回合循环。 |
| — | `metrics.py` | `pass_at_k` + 利用率对 MRU（机器人时间）/ MTU（author token）。 |
| — | `store.py` | SQLite sidecar，跨运行持久化尝试树。 |
| — | `lab.py` | `Lab.run(task)`——整条闭环一次调用。 |

## 它补的是哪些缺口

脑的 `ExecutorAdapter`（`mindloop.execute`）是**无状态**的——派发一个 `SubTask`、流回
`StatusUpdate`，仅此。封闭学习闭环还需本栈从未有过的两个原语：**reset**（把世界复位到已知
态）与 **reward**（给结果评分）。`Environment` 正好补这俩，`MockEnvironment` 确定性地提供
——于是 `bodylink` 不动、测试无需硬件。结果评分（`pass@k`）与预算指标（MRU/MTU）同为净新增；
`mindloop.eval` 此前只评引用合规。

## 预留 seam（真硬件，同协议）

- 真 `Environment`：`reset` = 机器人归位例程；`reward` = `worldsense` 多相机 `Detection`
  经同一套 `reward.py` 砖块。
- 真 `PolicyAuthor`：Codex / Claude Code 子进程，或 OpenAI 兼容端点（仿
  `mindloop.CosmosReasonVerbalizer`——把 `base_url` 指向主机）。
- `store.py` 在 mindloop DEV-PLAN P4 时毕业到 tempomem **L4 Trace store**。

## 与 mindloop 的关系

并行的**研究轨**，非重写。mindloop 的 `Replanner`（DEV-PLAN P4）修复回合**内**一个失败步；
`skillforge` 的 `Evolution` 在回合**间**改进策略。不同层级、互补。`skillforge` 把其余四个
成员当库消费，不阻塞 mindloop 的各阶段。
