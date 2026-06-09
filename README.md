# 国标麻将 Agent

面向 Botzone `Chinese-Standard-Mahjong` 的单局国标麻将智能体。仓库包含规则环境、
Botzone 协议适配、启发式 baseline、Hybrid Transformer、真实牌谱行为克隆、PPO
微调、固定牌墙复式评测和 Botzone 导出工具。

## 当前状态

- 真实牌谱 BC 第 5 轮验证 top-1 accuracy 为约 `62.82%`，且仍在上升。
- 该准确率只衡量模型是否复现牌谱中的人类动作，不等价于实战胜率或平均得分。
- 已有 PPO 日志中的 `average_score_a=0.15` 是传入的 RL 模型，
  `average_score_b=-0.05` 是启发式对手；40 局样本不足以判断 RL 是否提升。
- 最终模型必须通过相同 seed、相同牌墙、相同对手的复式评测选择。默认评测 400 局。
- Botzone 提交默认加载用户存储空间中的 `data/model.pt`；加载失败时回退启发式策略。
- 仓库中的 `artifacts/botzone_model.pt` 是现有 PPO update 60 的可上传候选，
  尚未经过 BC-vs-PPO 400 局选择，不能视为最终最优模型。

> 当前本地环境仍不是完整 Botzone 裁判：花牌、抢杠胡、完整 81 番种及部分协议边界
> 尚需在 Botzone 官方调试环境验证。正式提交前必须进行线上测试。

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
├── features/      # 394 维状态与候选动作编码
├── models/        # Hybrid Transformer actor-critic
├── policies/      # 随机、启发式、模型策略
├── rules/         # PyMahjongGB facade 与开发 fallback
└── training/      # 数据集、checkpoint、PPO
scripts/           # 数据处理、训练、评测、选择与导出
tests/             # 单元测试
botzone_entry.py   # Botzone JSON 入口
```

## 数据处理

完整流水线默认从 `Chinese-Standard-Mahjong/SL/data/data.txt` 生成 Parquet 和 tensor
cache。Parquet 用于归档，固定形状 tensor shard 用于训练吞吐。

```bash
python scripts/preprocess_official_full_actions.py \
  --output-dir artifacts/official_bc_full_v2 --workers 8

python scripts/build_tensor_cache.py \
  --input-dir artifacts/official_bc_full_v2 \
  --output-dir artifacts/official_bc_full_v2_tensors \
  --workers 8 --max-actions 64
```

## Behavior Cloning

BC 对每个状态的合法候选动作进行交叉熵分类。训练默认最多 15 epochs；验证准确率连续
3 轮没有提升时 early stop，并在停滞时降低学习率。输出同时包含整体准确率和
`PASS/PLAY/CHI/PENG/GANG/BUGANG/HU` 分类准确率。

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_bc.py \
  --data artifacts/official_bc_full_v2_tensors \
  --output artifacts/runs/example/bc_model.pt \
  --epochs 15 --patience 3 --batch-size 4096
```

继续训练时使用 latest checkpoint；脚本会恢复 optimizer、scheduler、最佳准确率和
early-stopping 状态：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_bc.py \
  --data artifacts/official_bc_full_v2_tensors \
  --output artifacts/runs/example/bc_model.pt \
  --resume artifacts/runs/example/bc_model.pt \
  --epochs 20 --batch-size 4096
```

不要单纯追求动作准确率。应重点检查低频鸣牌和胡牌准确率，并用复式实战得分选择模型。

## PPO

PPO 从最佳 BC checkpoint 开始。当前实现使用：

- 终局真实得分 `tanh(score / 64)`；
- 包含终局回落到零的 potential-based shaping；
- GAE、clipped PPO、value loss 和 entropy bonus；
- 相对初始 BC 策略的 KL 正则，降低灾难性遗忘；
- PPO 更新期间关闭 dropout，避免随机 mask 造成虚假的新旧策略 KL；
- minibatch 更新、target-KL early stop 和周期 checkpoint。

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py \
  --checkpoint artifacts/runs/example/bc_model.best.pt \
  --output artifacts/runs/example/ppo_model.pt \
  --updates 100 --games-per-update 8 --save-every 10 \
  --bc-kl-coef 0.01 --minibatch-size 1024
```

PPO checkpoint 会保存为 `ppo_model.update-0010.pt` 等文件。PPO loss 下降并不能证明
水平提升；只有复式评测不劣于 BC 时才应采用 PPO。

## 可靠评测与模型选择

复式评测将待测模型放在四个座位，复用固定牌墙来降低运气方差。输出明确标记
`policy_a` 和 `policy_b`，并报告平均分差及 95% 置信区间。

```bash
python scripts/evaluate.py \
  --model artifacts/runs/example/bc_model.best.pt \
  --policy-name bc --games 400 --seed 2026 --duplicate
```

统一评测 BC 和所有 PPO checkpoint，并选择满足“平均分和置信区间均不劣于 BC”的模型：

```bash
python scripts/select_best_ppo.py \
  --bc artifacts/runs/example/bc_model.best.pt \
  --ppo-glob 'artifacts/runs/example/ppo_model.update-*.pt' \
  --output artifacts/runs/example/final_model.pt \
  --report artifacts/runs/example/model_selection.json \
  --games 400 --seed 2026
```

完整训练流水线：

```bash
bash scripts/run_training_pipeline.sh
```

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
  --model artifacts/runs/20260609-173116/bc_model.best.pt \
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

优先补齐官方规则与 Botzone 协议黄金测试，然后增加对手池、更多并行 rollout、完整番种
特征和辅助任务。只有环境与评测可信后，才值得投入更复杂的 RL 或搜索方法。
