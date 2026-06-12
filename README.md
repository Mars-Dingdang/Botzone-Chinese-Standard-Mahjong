# 国标麻将 Agent

面向 Botzone `Chinese-Standard-Mahjong` 的单局国标麻将智能体。仓库包含规则环境、
Botzone 协议适配、启发式 baseline、Feature V2 Token Transformer、真实牌谱行为克隆、PPO
微调、固定牌墙复式评测和 Botzone 导出工具。

## 当前状态

- 默认训练使用公开信息 Feature V2 token 表示；旧 Feature V1 checkpoint 仅支持只读评测与导出。
- BC 同时报告 macro/Top-3 accuracy、NLL、分类 precision/recall 和辅助任务指标。
- 最终模型必须通过相同牌墙、座位和对手的配对复式评测选择。PPO 相对 BC 的
  `score_delta_95_ci` 下界必须不低于 `0`，否则自动回退 BC。
- Botzone 提交默认加载用户存储空间中的 `data/model.pt`；加载失败时回退启发式策略。
- 仓库中的 `artifacts/botzone_model.pt` 是现有 PPO update 60 的可上传候选，
  尚未经过 BC-vs-PPO 400 局选择，不能视为最终最优模型。

> 本地环境已覆盖抢补杠、末张与补牌限制，但仍不是完整 Botzone 裁判：花牌替换、
> 多家同时和牌、完整官方协议边界仍需在 Botzone 官方调试环境验证。

## 安装与测试

训练环境推荐 Python 3.10+、PyTorch 2.x：

```bash
make install-train
make test
```

部署目标是 Botzone Python 3.6 / PyTorch 1.8。仓库代码避免依赖较新的 Python 语法，
但模型结构、包大小、内存和单步延迟仍需在 Botzone 验证。

## 代码结构

```text
mahjong_agent/
├── botzone/       # Botzone 协议解析和状态重放
├── engine/        # 单局状态机、动作与牌编码
├── evaluation/    # 普通评测和固定牌墙复式评测
├── features/      # V1 扁平特征与 V2 公开信息 token 编码
├── models/        # V1 Hybrid Transformer 与 V2 Token Transformer
├── policies/      # 随机、启发式、模型策略
├── rules/         # PyMahjongGB facade 与开发 fallback
└── training/      # 数据集、checkpoint、PPO
scripts/           # 数据处理、训练、评测、选择与导出
tests/             # 单元测试
botzone_entry.py   # Botzone JSON 入口
```

## 数据处理

完整流水线默认从 `Chinese-Standard-Mahjong/SL/data/data.txt` 生成 V2 Parquet 和 tensor
cache。公开 observation 与终局/对手手牌辅助标签分开存储，Actor 输入不包含特权信息。

```bash
python scripts/preprocess_official_full_actions.py \
  --output-dir artifacts/official_bc_v4 --workers 8

python scripts/build_tensor_cache.py \
  --input-dir artifacts/official_bc_v4 \
  --output-dir artifacts/official_bc_v4_tensors \
  --workers 8 --max-actions 64
```

## Behavior Cloning

BC 对合法候选动作分类，并联合训练和牌、放铳、最终得分、番数区间和 belief 辅助头。
默认使用温和动作类别权重、`0.03` label smoothing 和 `belief_mode=aux`。

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_bc.py \
  --data artifacts/official_bc_v4_tensors \
  --output artifacts/runs/example/bc_model.pt \
  --epochs 50 --patience 8 --batch-size 512 \
  --metrics-jsonl artifacts/runs/example/logs/bc_metrics.jsonl
```

继续训练时使用 latest checkpoint；脚本会恢复 optimizer、scheduler、最佳指标和
early-stopping 状态：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_bc.py \
  --data artifacts/official_bc_v4_tensors \
  --output artifacts/runs/example/bc_model.pt \
  --resume artifacts/runs/example/bc_model.pt \
  --epochs 50 --batch-size 512
```

双卡 RTX 4090 的已验证默认值为每卡 batch `1536`（全局 batch `3072`）。流水线从 `configs/train/bc.yaml` 读取该值，`BC_BATCH_SIZE` 环境变量可覆盖。训练器会在送入 GPU 前按当前 batch 的最大有效 token 和合法动作数裁剪 padding，并在日志中打印实际 per-rank/global batch size。

`--belief-mode` 支持 `none`、`aux` 和 `actor`。`actor` 仅将 stop-gradient belief
embedding 输入 Actor。每个 epoch 会保存独立 checkpoint，流水线使用固定牌墙得分选择
最佳 BC；得分相同时依次比较 macro accuracy 和整体 accuracy。V1/V2 checkpoint 或
cache 混用会直接报错。

单独按固定牌墙选择 BC checkpoint：

```bash
python scripts/select_best_bc.py \
  --checkpoint-glob 'artifacts/runs/example/bc_model.epoch-*.pt' \
  --output artifacts/runs/example/bc_model.best.pt \
  --report artifacts/runs/example/evaluations/bc_selection.json \
  --results-dir artifacts/runs/example/evaluations/bc_checkpoints \
  --games 400 --seed 2026 \
  --wall-manifest artifacts/runs/example/evaluations/walls.json
```

## PPO

PPO 从最佳 BC checkpoint 开始。当前实现使用：

- 终局真实得分 `tanh(score / 64)`；
- 有单步/单局上限的牌效率、8 番可行性、放铳风险和可选流局听牌势能；
- GAE、clipped PPO、value loss 和 entropy bonus；
- 相对初始 BC 策略的 KL 正则，降低灾难性遗忘；
- 当前/最佳、历史、BC/启发式和少量随机策略组成的混合对手池；
- 原始终局分、Reward 分项、对手采样比例和周期 checkpoint。

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py \
  --checkpoint artifacts/runs/example/bc_model.best.pt \
  --output artifacts/runs/example/ppo_model.pt \
  --updates 100 --games-per-update 8 --save-every 10 \
  --bc-kl-coef 0.01 --minibatch-size 1024 \
  --metrics-jsonl artifacts/runs/example/logs/ppo_metrics.jsonl
```

PPO checkpoint 会保存为 `ppo_model.update-0010.pt` 等文件。PPO loss 下降并不能证明
水平提升；只有复式评测不劣于 BC 时才应采用 PPO。

## 可靠评测与模型选择

复式评测将待测模型轮换到四个座位并复用持久化牌墙。输出平均分、标准差、95% CI、
和牌/自摸/放铳/流局听牌率、番数、动作分布、非法动作数和推理延迟。

```bash
python scripts/evaluate.py \
  --model artifacts/runs/example/bc_model.best.pt \
  --policy-name bc --games 400 --seed 2026 --duplicate --progress \
  --save-wall-manifest artifacts/runs/example/evaluations/walls.json \
  --output-json artifacts/runs/example/evaluations/bc_eval.json
```

统一评测 BC 和所有 PPO checkpoint。只有相对 BC 的配对 `score_delta` 95% CI 下界
不低于 `0` 时才选择 PPO：

```bash
python scripts/select_best_ppo.py \
  --bc artifacts/runs/example/bc_model.best.pt \
  --ppo-glob 'artifacts/runs/example/ppo_model.update-*.pt' \
  --output artifacts/runs/example/final_model.pt \
  --report artifacts/runs/example/evaluations/model_selection.json \
  --games 400 --seed 2026 \
  --wall-manifest artifacts/runs/example/evaluations/walls.json \
  --results-dir artifacts/runs/example/evaluations/checkpoints
```

## Screen 训练流水线

流水线使用 detached `screen` 管理后台任务，并将日志、指标、配置快照和 evaluation
结果写入独立运行目录：

```bash
bash scripts/run_training_pipeline.sh start --from data
bash scripts/run_training_pipeline.sh start --from bc --run-dir artifacts/runs/example
bash scripts/run_training_pipeline.sh start --from rl --run-dir artifacts/runs/example
bash scripts/run_training_pipeline.sh start --from eval --run-dir artifacts/runs/example

bash scripts/run_training_pipeline.sh status
bash scripts/run_training_pipeline.sh attach
bash scripts/run_training_pipeline.sh resume
bash scripts/run_training_pipeline.sh stop
```

阶段顺序为数据处理、BC、BC 评测、PPO、统一评测与模型选择。`resume` 使用
`artifacts/latest_run.txt` 和现有 checkpoint 恢复。常用覆盖参数包括：

```bash
GPUS=2 BC_EPOCHS=50 PPO_UPDATES=100 EVAL_GAMES=400 \
  bash scripts/run_training_pipeline.sh start --from data
```

运行目录包含 `run_manifest.json`、`configs/`、`logs/`、`evaluations/`、BC/PPO
checkpoint 和 `final_model.pt`。长任务使用 `tqdm`；非交互日志仍会保留阶段和指标输出。

## Botzone 上传

Botzone 的 Python 多文件源码 zip 根目录必须包含 `__main__.py`；模型等数据文件不应
打进源码 zip，应通过账户的用户存储空间上传。参考：
[Bot 文档](https://wiki.botzone.org.cn/index.php?title=Bot)、
[国标麻将协议](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)。

部署入口使用独立的严格合法性层：模型只能在已验证候选中选择，输出前还会再次校验。
无法严格证明合法的响应动作回退为 `PASS`，自摸阶段则回退为合法弃牌。特别地，官方
`MahjongGB` 不可用或算番异常时不会输出 `HU`，避免错和导致 `-30`。

生成两个上传文件：

```bash
python scripts/export_bot.py \
  --model artifacts/runs/example/final_model.pt \
  --output artifacts/botzone_submission.zip \
  --storage-model artifacts/botzone_model.pt

python scripts/export_bot.py \
  --model artifacts/runs/20260611-210636/bc_model.best.pt \
  --output artifacts/botzone_submission_bc.zip \
  --storage-model artifacts/botzone_model_bc.pt
```

上传步骤：

1. 在 Botzone 账户菜单的“管理存储空间”上传 `artifacts/botzone_model.pt`，目标路径为
   `data/model.pt`。
2. 为 `Chinese-Standard-Mahjong` 创建 Python 3.6 Bot，将
   `artifacts/botzone_submission.zip` 作为 Python 源码上传。
3. 使用 Botzone 调试功能测试 request 类型 `0-9`，重点检查吃、碰、明杠、暗杠、
   补杠、抢杠胡和花牌。
4. 确认每步无非法动作、无超时，并至少进行若干练习对局后再提交正式版本。

本地检查导出包、模型加载、JSON 交互和启动延迟：

```bash
python scripts/verify_deploy.py \
  --archive artifacts/botzone_submission.zip \
  --model artifacts/botzone_model.pt
```

Botzone 对局因非法动作结束时，可下载完整日志并定位首个非法响应：

```bash
python scripts/audit_botzone_log.py path/to/botzone-log.json --all
python scripts/audit_botzone_log.py path/to/botzone-log.json --player 2
```

审计器会重放每名玩家收到的 requests，并报告请求、响应、严格校验失败原因和状态摘要。
Botzone 不支持动作发送后 redo；本项目的“redo”发生在输出前，即拒绝模型的非法提案并
选择安全动作。

## 课程提交

课程最终压缩包按要求命名为 `学号+姓名.zip`，建议结构：

```text
学号+姓名/
├── report.pdf                    # 3-8 页研究报告
├── code/                         # 完整代码、README、requirements
├── model/final_model.pt          # 最终训练模型
├── botzone/botzone_submission.zip
├── botzone/botzone_model.pt
└── BOTZONE.md                    # Bot 名称、链接、版本和上传说明
```

提交前重新运行：

```bash
make test
python scripts/verify_deploy.py
git status --short
```

## 常见问题

**BC 训练 5 epochs、约 62.82% 是否够用？**

不能仅凭该指标判断。当前验证准确率仍在上升，可以继续训练至 early stopping；最终应以
同牌墙复式评测得分选择 checkpoint。更高的人类动作准确率通常意味着更强的行为对齐，
但也可能复现牌谱中的次优决策。

**为什么 RL 看起来比启发式差？**

先确认字段含义：`average_score_a` 是传入 `--model` 的策略。随后用至少 400 局、相同
seed 的 BC/PPO 复式评测比较。RL 退化通常来自稀疏奖励、rollout 太少、策略漂移和模拟器
与真实规则不一致；当前实现使用 BC KL 正则和保守 checkpoint 选择降低这些风险。

**下一步优化方向是什么？**

下一阶段应优先实现 Oracle Critic、扩大并行 rollout，并在固定牌墙评测下比较经验规则
重排序、determinization rollout 与 ISMCTS。正式扩大训练前仍需补充 Botzone 协议黄金测试。
