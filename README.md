# 国标麻将 Agent

面向 Botzone `Chinese-Standard-Mahjong` 的单局国标麻将智能体工程。项目提供
依赖标准库即可运行的单局环境、启发式 Bot、Botzone 入口、复式评测，以及基于
PyTorch 的 Hybrid Transformer、行为克隆和 PPO 训练流水线。

## 当前功能

- 34 种普通牌统一编码、发牌、摸牌、弃牌和单局结算
- 吃、碰、明杠、暗杠、补杠、胡牌和合法动作检查
- Botzone request 历史重放与 JSON 标准输入输出入口
- 随机策略、向听数/有效牌启发式策略、模型策略
- 结构化盘面特征与合法候选动作编码
- Hybrid Transformer 候选动作评分器、价值头和辅助预测头
- 启发式对局数据生成、行为克隆和轻量 PPO fine-tune
- 固定牌墙、座次轮换的复式评测与 Botzone zip 导出

> **重要限制：** 仓库中的纯 Python 规则 fallback 只用于开发与测试，可识别
> 标准牌型和七对，但未实现完整国标番种表。正式参赛前必须安装或内置课程提供的
> `PyMahjongGB/MahjongGB`，并用官方黄金案例验证至少 8 番的胡牌判定。

## 代码库结构

```text
Mahjong/
├── configs/                 # 模型、训练和评测配置
├── mahjong_agent/
│   ├── engine/              # 单局状态机、动作和牌编码
│   ├── rules/               # 官方规则组件 facade 与 fallback
│   ├── botzone/             # Botzone 协议解析和状态重放
│   ├── features/            # Hybrid observation 与动作编码
│   ├── policies/            # 随机、启发式和模型策略
│   ├── models/              # Hybrid Transformer
│   ├── training/            # rollout、数据集、checkpoint、PPO
│   └── evaluation/          # 单局与复式评测
├── scripts/                 # 数据、训练、评测、导出脚本
├── tests/                   # 标准库 unittest 测试
├── third_party/PyMahjongGB/ # 官方规则组件预留目录
├── botzone_entry.py         # Botzone 提交入口
└── Makefile
```

## 实现技术

### 状态与动作

普通牌使用 `0-33` 编码。环境输出当前玩家的私有手牌与全部公开信息，不暴露其他
玩家手牌。每一步由引擎生成合法的结构化动作：

```text
PASS / PLAY / CHI / PENG / GANG / BUGANG / HU
```

模型只对当前合法动作候选评分，因此非法动作不会进入采样或 `argmax`。

### 特征与模型

`mahjong_agent.features.encode_observation` 编码自家手牌、可见牌、四家弃牌河、
副露、座位、阶段和剩余牌数。候选动作单独编码后与盘面状态融合。

`HybridTransformer` 使用结构化状态 tokens、双向 Transformer、动作评分头、
价值头与辅助预测头。默认模型约为轻量级配置，并仅使用 PyTorch 1.4 已存在的
基础 API。

### 训练

1. `HeuristicPolicy` 自对弈生成 JSONL 行为克隆数据。
2. BC 使用 masked candidate-action cross entropy 预训练。
3. PPO 让模型玩家对战启发式与随机策略，并使用终局得分更新。

`train_ppo.py` 是可运行的单机 PPO 基线，并在 `torchrun` 设置 `WORLD_SIZE` 时
启用 `DistributedDataParallel`。trajectory 与 checkpoint 模块独立，便于后续
迁移到 IMPALA/V-trace。

## 环境安装

训练推荐 Python 3.10+、PyTorch 2.x：

```bash
make install-train
```

Botzone 启发式部署入口只依赖 Python 标准库：

```bash
make install-deploy
```

课程要求的 Python 3.6 / PyTorch 1.4 部署环境应额外执行 `make verify-deploy`。
checkpoint 使用旧 zip 格式保存，但仍需在实际旧版环境验证模型结构。

## 常用命令

```bash
make test
make generate-bc-data
make train-bc
make train-ppo
make evaluate
make export-bot
make verify-deploy
```

等价的细粒度命令：

```bash
python scripts/generate_bc_data.py --games 1000 --output artifacts/bc_data.jsonl
python scripts/train_bc.py --data artifacts/bc_data.jsonl --output artifacts/bc_model.pt
python scripts/train_ppo.py --checkpoint artifacts/bc_model.pt --output artifacts/ppo_model.pt
torchrun --nproc_per_node=2 scripts/train_ppo.py --checkpoint artifacts/bc_model.pt
python scripts/evaluate.py --games 100 --duplicate
python scripts/export_bot.py --model artifacts/ppo_model.pt
```

在当前提供的 conda 环境中：

```bash
conda run -n d2l make test
conda run -n d2l python scripts/generate_bc_data.py --games 2
```

## Botzone 提交

`botzone_entry.py` 从标准输入读取 Botzone JSON，重放 `requests` 并输出
`{"response": "..."}`。默认使用无需 PyTorch 的启发式策略。

```bash
python scripts/export_bot.py --output artifacts/botzone_submission.zip
```

正式上传前必须：

1. 内置并接通官方 `PyMahjongGB` 完整算番上下文转换。
2. 使用 Botzone 官方交互样例测试吃碰杠、抢杠胡和花牌行为。
3. 在 Botzone 限制环境验证提交包大小、Python 版本和单步延迟。

## 评测指标

普通评测输出平均得分、和牌率、非法动作数和环境吞吐。复式评测复用固定牌墙，
轮换模型座位以降低运气方差。

```bash
python scripts/evaluate.py --games 100 --duplicate
```

## 已知限制与后续工作

- 完整国标番种、花牌、抢杠胡、响应优先级和 Botzone 所有协议细节需与官方组件对齐。
- 当前 PPO 是清晰可运行的基线，尚未实现多 GPU DDP actor/learner 编排。
- belief teacher-student、对手池持久化、IMPALA/V-trace 和 top-k search 尚未实现。
- 仓库不包含预训练 `.pt`；需运行 BC/PPO 脚本生成后再提交。

研究背景、模型选择和扩展路线见 `deep-research-report.md`。
