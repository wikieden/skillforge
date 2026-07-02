# 决策 —— 借鉴 SkillOpt 的纪律,不依赖其包

> 🌐 [English](../en/skillopt-borrow-decision.md) · **中文**
>
> 决策记录。日期：2026-06-30。状态：**已定 —— 借鉴(BORROW)**(held-out 门已落地，
> 见 [`gate.py`](../../src/skillforge/gate.py) + `CHANGELOG`）。

## 决策

skillforge **借鉴** [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)
（arXiv 2605.23904，MIT）**的「纪律」**——把它的想法移植成我们自己的少量自有代码——
而**不依赖** `skillopt` PyPI 包。

## 为什么(核心：品类错配)

两个项目优化的是**品类不同**的可训练物：

| | 可训练物 | 评分 |
|---|---|---|
| **SkillOpt** | 单个**自然语言 Markdown 字符串**（冻结 LLM 的"prompt 权重"），由 4 个字符串算子（`append`/`insert_after`/`replace`/`delete`）编辑 | QA 正确性 `{id, hard, soft}` |
| **skillforge** | 类型化 `Policy = tuple[SubTask, ...]`（verb + node_ids + params），给机器人本体 | 几何 `Reward(success, value=-distance)` over tempomem Detections |

SkillOpt 的编辑算子是对散文的子串/行手术，**表达不了结构化 `Policy` 变更**（重排 / 改
`SubTask` target）。依赖会**每步强制 `Policy` ↔ Markdown 有损往返**，丢掉类型保证、引入
解析失败。它的包代码（`skillopt/optimizer/skill.py`）对 `SubTask` 元组**零复用**。

依赖还**打破两条 Chronotope 契约**（已对照 SkillOpt 源码核实）：

- **网络无关确定性测试** —— 它的 `ReflACTTrainer` 每步调远程 optimizer LLM
  （`chat_optimizer`）；要保住 skillforge 的离线 `ScriptedAuthor`/`MockBody` 测试，就得
  每步 mock 一个 alpha（v0.1.0）API。
- **独立发布** —— 它把 `openai` / `azure-identity` / `azure-core` / `httpx` 拖进那个
  已经 import 所有其他成员的成员，且**无公开 optimizer API**（`skillopt/__init__.py` 只
  导出 types；真入口是 CLI over YAML）。

**值钱的部分**——循环纪律——是**与策略表示无关**的，所以我们把它当 ~40–200 行自有纯
Python 留下（~1–2 天），而不是几周的丢弃式胶水。

## 借鉴了什么（且已落地）

| SkillOpt | skillforge | 状态 |
|---|---|---|
| `evaluate_gate`：held-out 上 `cand > current` 才接受（平局拒，纯 frozen 结果） | `gate.py` `evaluate_gate` → `GateResult{accept \| accept_new_best \| reject}` over `Reward.value` | **已落地** |
| train / valid_seen / valid_unseen 拆分 | `evolve(..., val_tasks, env_for)` 在非泄漏 held-out 集上打分；`mock_env_factory` 按 task 建 env | **已落地** |
| best_skill.md = 验证集 argmax | `tree.best()` 由门喂非退化候选；`Attempt` 加 `holdout_score` + `accepted` | **已落地** |
| 4 个有界编辑算子（字符串手术） | 结构化 **`PolicyEdit`** 代数（按 index 增删改 `SubTask`）—— 全新，零 SkillOpt 复用 | **推迟** |
| edit-budget / 文本学习率；rejected-edit buffer | 未来真 LLM `PolicyAuthor`（`[llm]` extra）的特性（`ScriptedAuthor` 下无牙） | **推迟** |

门闭合了架构 review 标的 **"脚本必胜 / 无验证门"** 缺口：回归候选现在被否决，而非无条件追加。

## 什么会翻转成「依赖」（刻意重审，而非漂移）

1. skillforge 的 artifact **从类型 `Policy` 迁成自然语言 SkillCard**（`policy.py` 关于与
   mindloop P2 SkillCard 统一的注记）。品类错配随即坍缩，SkillOpt 的文本编辑 + LLM-target
   循环几乎完美契合。
2. SkillOpt 出**干净、纯 stdlib 的公开 optimizer API**，不需 CLI/YAML、确定模式不硬调远程 LLM。
3. skillforge 上**真 LLM `PolicyAuthor`**（`[llm]`）**且**有多任务语料 —— 那个规模下
   epoch / rejected-buffer / 文本学习率机器才开始兑现。
4. `skillopt` 经由别的成员（如 mindloop 用它做 SkillCard 优化）反正成了传递依赖，边际成本≈0。

把借鉴来的门 / 拆分 / 编辑代码保持足够模块化，以便翻转条件触发时可换。

## 参考

- microsoft/SkillOpt —— README、`skillopt/evaluation/gate.py`、`skillopt/types.py`、
  `skillopt/engine/trainer.py`、`docs/guide/new-benchmark.md`、`pyproject.toml`（MIT，
  Python ≥3.10，Dev Status 3 — Alpha，v0.1.0）。论文：arXiv 2605.23904。
- skillforge：[`gate.py`](../../src/skillforge/gate.py)、[`evolution.py`](../../src/skillforge/evolution.py)、
  [architecture.md](architecture.md)。
