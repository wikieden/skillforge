# skillforge

> 🌐 [English](README.md) · **中文**

**Chronotope 栈的 ENPIRE 式自驱自学习实验室。**

`skillforge` 是 Chronotope 第 5 个成员：一个元编排器，驱动记忆（`tempomem`）、眼
（`worldsense`）、脑（`mindloop`）、身（`bodylink`）闭环运转，让机器人**从自己的试验中
习得技能**——全程无人干预。它把 NVIDIA GEAR
[ENPIRE](https://research.nvidia.com/labs/gear/enpire/) 的四模块移植进本栈：

```
EN  Environment  环境    复位世界 + 给结果评分        (environment.py + reward.py)
PI  Policy Improvement  据经验重写策略               (author.py + policy.py)
R   Rollout      回合    跑预算内试验，全程记录        (rollout.py)
E   Evolution    进化    把多次尝试复利成技能          (evolution.py + tree.py)
```

`Lab.run(task)` 转动闭环：author 提策略 → rollout 在（仿真）硬件上评分 → 尝试入谱系树
→ 循环，直到解决或预算耗尽。

## 状态 — 走路骨架（v0.1.0a0）

最薄的**完整**纵切：1 任务、1 编码 agent、`MockBody`。**无网络、无 GPU**——
`ScriptedAuthor` + `MockEnvironment` 顶替真 LLM author 与真硬件。舰队并行、分支假设树、
完整 MRU/MTU 经济学待闭环证明能自转后再加。

## 跑演示

```bash
uv sync
uv run python skillforge/examples/01_self_close.py   # 打印尝试树 + 通过率
uv run pytest skillforge/tests                        # 闭环自闭合
```

## 怎么做到既真又轻

- reward 砖块（`object_at_pose`、`distance_to_target`）在真 `tempomem.Detection` 上评分
  ——跟真 `worldsense` 相机产出的是同一类型。
- `MockEnvironment` **包裹** `bodylink.MockBody`（executor seam 不动），补上本栈缺的两个
  原语：**reset** 与 **reward**。
- 换上真 `Environment`（机器人复位 + 相机奖励）和真 `PolicyAuthor`（Codex / Claude Code /
  OpenAI 兼容端点）——闭环代码零改动。

## 定位

`skillforge` 是**研究轨**，与 [mindloop DEV-PLAN](../mindloop/docs/zh/devplan/DEV-PLAN.md)
并行。mindloop 的 `Replanner` 修复回合**内**一个失败步；`skillforge` 的 `Evolution` 在回合
**间**改进策略——不同层级，互补。

模块图见 [docs/zh/architecture.md](docs/zh/architecture.md)。

> 成员名 `skillforge` 为工作名，待 owner 按 Chronotope 命名约定批准。
