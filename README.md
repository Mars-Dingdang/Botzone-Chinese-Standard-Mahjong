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

### 规则、状态与动作

普通牌统一使用 `0-33` 编码。`mahjong_agent/engine` 提供轻量单局状态机，
`mahjong_agent/rules` 优先调用 `PyMahjongGB==1.3.0` 做官方算番，并保留纯 Python
fallback 供测试。`mahjong_agent/botzone` 将 Botzone request 历史重放为当前玩家可见状态。

在线 observation 只包含自家手牌和公开信息，不暴露其他玩家手牌或暗杠内容。合法动作由
环境生成，模型只对当前候选集合评分：

```text
PASS / PLAY / CHI / PENG / GANG / BUGANG / HU
```

动作 mask 在采样和 `argmax` 前将 padding/非法候选 logit 置为大负数，因此训练与部署
不会主动选择非法动作。v2 真实牌谱预处理恢复摸牌与三家响应决策，覆盖
`PASS/PLAY/CHI/PENG/GANG/ANGANG/BUGANG/HU`；吃碰与紧随其后的弃牌合并为联合候选标签。
与官方预处理一致，只有一个合法动作的强制状态会被过滤，避免大量强制 `PASS` 虚高准确率。

### 状态表示与启发式特征

`mahjong_agent/features/encoder.py` 输出 394 维公开状态向量和每个候选动作的 8 维编码。
状态向量包含：

- 自家 34 种牌计数、全部可见牌计数
- 四家弃牌河和四家公开副露
- 34 维有效牌 mask、向听数、有效牌数量
- 圈风、当前行动玩家、决策阶段、最后弃牌、牌墙估计
- `DRAW/PLAY/CHI/PENG/GANG/BUGANG/HU` 公共事件计数

向听数和有效牌由规则后端计算。启发式策略优先胡牌，并用以下经验评分比较弃牌候选：

```text
heuristic_score = -10 * shanten + remaining_useful_tiles - honor_penalty
```

吃碰额外施加 `-1` 惩罚，杠获得固定正偏置。该启发式同时作为 BC 数据生成器和 PPO
混合对手，但它不是最终训练 objective。

### 模型结构

`mahjong_agent/models/hybrid_transformer.py` 实现候选动作评分式 actor-critic：

1. 394 维状态经过 MLP 映射为 4 个 `d_model=192` 状态 token。
2. 4 层、6 头双向 Transformer 编码状态 token。
3. 每个合法候选动作由独立 MLP 编码，与全局状态拼接后输出一个 policy logit。
4. 共享状态同时进入标量 value head 和 3 维 auxiliary head。

默认模型约 `2.82M` 参数。候选动作评分比固定 235 类动作头更容易支持可变合法动作集合。
当前 auxiliary head 仅保留接口，尚未接入真实辅助标签。

### 真实牌谱预处理与数据管线

`Chinese-Standard-Mahjong/SL/data/data.txt` 包含 98,209 局 Botzone 牌谱。数据管线分两层：

1. `scripts/preprocess_official_data.py` 按完整 `Match` 边界切分，使用多进程重放牌局，
   重建各家私有手牌与公共事件，输出 zstd 压缩 Parquet。
2. `scripts/build_tensor_cache.py` 将 Parquet 一次性转换为固定形状 FP16 tensor shards，
   预先 padding 候选动作和 mask，避免每个 epoch 重复执行 `to_pylist` 和 Python collation。

全量结果为 4,368,883 条训练决策、231,518 条验证决策，解析失败数为 0。当前容器实际
CPU 配额为 8 核；32 worker 的 Parquet 预处理用时约 2,268 秒。Parquet 保留作归档，
tensor cache 用于高吞吐训练。

### Behavior Cloning Objective

真实玩家动作在合法候选集合中的索引为 `y`，模型输出 masked logits `z`。BC 使用候选动作
交叉熵：

```text
L_BC = -log softmax(z)_y
accuracy = mean(argmax(z) == y)
```

优化器为 AdamW，默认学习率 `3e-4`，梯度范数裁剪为 `1.0`，CUDA 上使用 FP16 AMP。
双卡使用 DDP；Parquet/tensor shards 按样本数贪心均衡，并将两个 rank 限制为严格相同
batch 数，避免 epoch 末尾 collective 错位。

经验上采用每卡 batch `4096`。同等全局样本数基准中，双卡约 8 秒、单卡 batch `8192`
约 11 秒，因此保留双卡。最新 5-epoch BC 结果：

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.975 | 0.297 | 1.805 | 0.353 |
| 2 | 1.710 | 0.403 | 1.605 | 0.446 |
| 3 | 1.512 | 0.486 | 1.417 | 0.524 |
| 4 | 1.351 | 0.546 | 1.285 | 0.571 |
| 5 | 1.226 | 0.587 | 1.173 | 0.603 |

### PPO Objective 与 Reward

`scripts/train_ppo.py` 从最佳 BC checkpoint 开始 fine-tune。每局轮换 learner 座位，其他
座位混合使用启发式策略和随机策略；每个 DDP rank 独立采样，梯度通过 DDP 同步。

当前 reward 以真实番数结算后的终局得分为主，并使用 GAE 分配信用：

```text
R_T = tanh(score / 64)
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

可选 potential shaping 默认系数为 `0.02`，使用
`Phi(s) = -shanten + 0.1 * remaining_useful_tiles` 的势能差；终局得分仍是主奖励。

PPO 使用 clipped surrogate objective、价值回归与 entropy bonus：

```text
ratio_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))
L_policy = -mean(min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t))
L_value = mean((V(s_t) - R)^2)
L_total = L_policy + 0.5 * L_value - 0.01 * entropy(pi)
```

默认 `eps=0.2`、`gamma=0.99`、`lambda=0.95`，每批最多执行 4 个 PPO epoch；approximate
KL 超过 `0.02` 时提前停止，同时报告 clip fraction 和 explained variance。PPO 当前属于轻量
on-policy baseline；轻量模拟器与完整官方国标规则仍有差距，因此 PPO 指标需要结合官方
裁判或 Botzone 对局继续验证。

### 训练指标与评估

BC 报告交叉熵 loss、候选动作 top-1 accuracy、样本数和 optimizer steps。PPO 报告总 loss、
policy loss、value loss 和策略 entropy。

`mahjong_agent/evaluation/evaluator.py` 支持普通评测和复式评测。复式评测复用固定牌墙并轮换
模型座位，以降低牌运方差；主要指标包括平均得分、和牌率、非法动作数和环境吞吐。最新
BC checkpoint 对启发式策略的 40 局复式评估为：模型平均分 `1.8`，启发式平均分 `-0.6`。

### 经验策略与工程选择

- 使用“真实牌谱 BC 冷启动，再 PPO 自对弈 fine-tune”，避免纯 PPO 从随机策略开始。
- 使用候选动作评分与合法动作 mask，适配可变动作空间并消除非法动作采样。
- Parquet 用于可复现归档，固定形状 tensor cache 用于训练吞吐。
- 双卡 DDP 每卡 batch `4096`；模型较小，双卡相对单卡只快约 20-30%，但仍是当前实测更快配置。
- 保存 latest 与 best BC checkpoint；PPO 每 10 个 update 保存一次 checkpoint。
- 当前不使用 belief teacher-student、完整 opponent pool、IMPALA/V-trace 或搜索，这些是后续研究方向。

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
- 当前 PPO 支持 GAE、KL early stop 和单机双卡 DDP 梯度同步，但 rollout 仍在各 rank 内串行采样，尚未实现独立 actor/learner 编排。
- belief teacher-student、对手池持久化、IMPALA/V-trace 和 top-k search 尚未实现。
- 训练 checkpoint 位于 `artifacts/runs/<run-id>/`；提交前应选择复式评估最优模型并验证部署兼容性。

研究背景、模型选择和扩展路线见 `deep-research-report.md`。

## 双卡真实牌谱训练与评估

推荐使用独立 conda 环境。`PyMahjongGB==1.3.0` 用于官方国标算番；训练使用
PyTorch DDP，以下命令适用于单机双卡 RTX 4090：

```bash
conda env create -f environment.train.yml
conda activate mahjong-train

# 使用多核 CPU 将 98,209 局官方牌谱转换为压缩 Parquet 分片
python scripts/preprocess_official_full_actions.py \
  --input Chinese-Standard-Mahjong/SL/data/data.txt \
  --output-dir artifacts/official_bc_full_v2 --workers 8

# 构建一次性固定形状 tensor cache
python scripts/build_tensor_cache.py --input-dir artifacts/official_bc_full_v2 \
  --output-dir artifacts/official_bc_full_v2_tensors --workers 8 --max-actions 64

# 双卡从连续 tensor cache 训练；每卡 batch 4096
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_bc.py --data artifacts/official_bc_full_v2_tensors \
  --output artifacts/bc_model.pt --epochs 5 --batch-size 4096

# 双卡 PPO；模型座位轮换并混合启发式/随机对手
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_ppo.py --checkpoint artifacts/bc_model.best.pt \
  --output artifacts/ppo_model.pt --updates 100 --games-per-update 8

# 固定牌墙、座次轮换的复式评估
python scripts/evaluate.py --model artifacts/ppo_model.pt --games 100 --duplicate

# 自动执行全量预处理、BC、评估、PPO 和最终评估
setsid -f bash -c 'exec env PATH=/root/LLM_HW2/.venv/bin:$PATH \
  CUDA_VISIBLE_DEVICES=0,1 PREPROCESS_WORKERS=8 CACHE_WORKERS=8 bash scripts/run_training_pipeline.sh \
  > artifacts/training-launch.log 2>&1'
```

v2 预处理按完整 `Match` 分块并行并输出 zstd Parquet；旧的弃牌-only 缓存继续保留且不会与
v2 路径混用。环境已统一收集三家响应并按 `HU > PENG/GANG > CHI` 裁决，且使用
`PyMahjongGB` 真实番数结算；抢杠胡等边界仍需继续与官方裁判做差分测试。

## 数据加速优化

这次 v2 的预处理加速做了四件事：

1. 预处理阶段只保存紧凑原始状态和动作标签，`encode_observation()` 与 `encode_action()` 挪到 `build_tensor_cache.py` 批量执行。
2. `claim` 阶段加入便宜的胡牌剪枝，明显不可能成和的状态不再反复调用完整番数计算。
3. `shanten` 和 `useful_tiles` 在 worker 内做 LRU 缓存，避免同一状态重复计算。
4. 默认 `chunk-matches=500`，减少 Parquet 分片数量和写盘次数。

这类优化主要提升 CPU 侧吞吐，GPU 对预处理本身帮助不大。

推荐的后台启动方式是：

```bash
nohup bash -lc 'exec env PATH=/root/LLM_HW2/.venv/bin:$PATH CUDA_VISIBLE_DEVICES=0,1 PREPROCESS_WORKERS=8 CACHE_WORKERS=8 bash scripts/run_training_pipeline.sh' \
  > artifacts/training-v2-launch.log 2>&1 &
echo $! > artifacts/training-v2.pid
```

训练进度、日志路径和后续工作记录在 `MEMORY.md`。


python scripts/evaluate.py --model artifacts/runs/20260609-173116/ppo_model.pt --games 40 --duplicate