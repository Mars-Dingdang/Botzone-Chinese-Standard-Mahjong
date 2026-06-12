# 国标麻将 Agent 与 Suphx 的逐章技术对比报告

## 摘要

本文按照论文 *Suphx: Mastering Mahjong with Deep Reinforcement Learning*
的章节结构，对本项目的国标麻将 Agent 进行系统性技术对比。

Suphx 面向天凤平台的四人日式立直麻将，其核心训练路线是：

1. 使用顶尖人类牌谱分别训练弃牌、立直、吃、碰、杠模型；
2. 使用大规模分布式自对弈强化学习提升弃牌模型；
3. 使用 Global Reward Prediction 解决多局比赛中的全局奖励归因；
4. 使用 Oracle Guiding，以完整隐藏信息加速部分可观测策略训练；
5. 使用 Parametric Monte-Carlo Policy Adaptation 针对当前起手牌进行运行时策略适应。

本项目面向 Botzone 单局国标麻将，当前路线是：

1. 将公开观测和合法候选动作编码为 token；
2. 使用统一 Token Transformer 为每个合法候选动作打分；
3. 使用约 9.8 万局官方牌谱进行行为克隆，并联合训练终局结果、番数和对手手牌
   belief 辅助任务；
4. 从 BC checkpoint 初始化 PPO，使用终局得分、公开信息势能塑形、BC KL 正则和
   混合对手池进行自对弈；
5. 使用固定牌墙、座位轮换和置信区间进行模型选择；
6. 使用严格合法动作层导出并部署至 Botzone。

本项目已经实现了 Suphx 中“监督预训练、自对弈强化学习、全局结果辅助预测、隐藏信息
辅助监督”的部分思想，但尚未实现 Suphx 意义上的 Global Reward Prediction、
Oracle Guiding 和 Run-time Policy Adaptation。当前项目在合法候选动作建模、统一策略
模型、可重复复式评测和部署安全性上具有自己的工程特点，但训练规模、规则完整性、
数据质量和实验验证仍明显弱于 Suphx。

---

## 1. Introduction：问题定义与研究目标

### 1.1 Suphx 的问题定义

Suphx 研究四人日式立直麻将。论文指出三项主要困难：

1. **复杂的跨局奖励**：一场比赛由多局组成，最终奖励由累计点数和排名决定。单局负分
   不一定表示策略差，例如领先玩家可能主动采取保守策略以保持总排名。
2. **大量隐藏信息**：对手手牌、牌墙和死墙不可见，一个信息集对应大量可能的真实状态。
3. **不规则决策流程**：吃、碰、杠、和牌会中断正常摸打顺序，使常规博弈树搜索困难。

Suphx 的目标不是单纯最大化单局和牌率，而是提升长期比赛排名，并最终在 Tenhou
真实玩家环境中达到超越顶尖人类玩家的稳定段位。

### 1.2 本项目的问题定义

本项目研究 Botzone `Chinese-Standard-Mahjong` 单局国标麻将。它与 Suphx 共享以下
困难：

- 四人部分可观测博弈；
- 对手暗手牌不可见；
- 吃、碰、杠、补杠、抢杠和会改变正常行动顺序；
- 动作合法集合随阶段变化；
- 终局反馈稀疏；
- 需要同时平衡牌效率、成番能力和放铳风险。

但本项目与 Suphx 的优化目标存在根本区别：

| 维度 | Suphx | 本项目 |
|---|---|---|
| 规则 | 日式立直麻将 | 国标麻将 |
| 对局单位 | 多局组成完整比赛 | 当前环境和数据主要按单局处理 |
| 核心终局目标 | 完整比赛排名与稳定段位 | 单局国标得分 |
| 起和条件 | 至少一个役 | 至少 8 番 |
| 特殊机制 | 立直、宝牌、死墙、连庄 | 81 类国标番种、花牌、8 番门槛 |
| 部署平台 | Tenhou | Botzone |

因此，Suphx 的 Global Reward Prediction 解决的是“完整比赛奖励如何归因到每一局”，
而本项目目前没有跨局累计分数和最终排名，无法直接复现这项技术。

### 1.3 本项目当前研究定位

本项目更准确的定位是：

> 一个以公开信息 Token Transformer、全动作行为克隆、PPO 自对弈和严格 Botzone
> 部署为核心的单局国标麻将 Agent。

与 Suphx 相比，本项目当前更强调统一动作建模、训练与部署合法性、低成本可复现训练和
固定牌墙评测；Suphx 更强调大规模分布式训练、跨局奖励归因和隐藏信息利用。

---

## 2. Overview：决策流程、特征与模型结构

## 2.1 Decision Flow：决策流程

### Suphx

Suphx 将策略拆分为五个独立学习模型：

- Discard model；
- Riichi model；
- Chow model；
- Pong model；
- Kong model。

和牌由额外的规则模型决定。不同阶段依次调用对应模型，多个鸣牌动作同时可用时，比较
模型输出的置信度。

这种设计的优点是每个模型只解决较简单的分类问题，也能针对各动作类型使用不同训练数据
规模。缺点是多个独立模型可能产生概率不可比、重复计算和共享表示不足的问题。

### 本项目

本项目使用一个统一策略模型处理全部动作：

- `PASS`
- `PLAY`
- `CHI`
- `PENG`
- `GANG`
- `BUGANG`
- `HU`

环境或部署合法性层先生成合法候选动作，模型仅对这些候选动作评分，然后选择最高分动作。
吃、碰后的弃牌被包含在候选动作定义中，因此模型可以直接比较：

```text
PASS
PENG W5 -> discard W1
PENG W5 -> discard B9
CHI W4-W5-W6 -> discard J2
```

相关实现：

- `mahjong_agent/engine/env.py`
- `mahjong_agent/botzone/legality.py`
- `mahjong_agent/policies/model.py`
- `mahjong_agent/engine/actions.py`

### 对比结论

| 项目 | Suphx | 本项目 |
|---|---|---|
| 策略模型数量 | 五个独立模型 | 一个统一候选动作评分器 |
| 和牌决策 | 规则模型 | 合法性规则生成候选，策略模型可以在合法候选中选择 |
| 动作冲突处理 | 比较独立模型置信度 | 所有合法动作统一比较 |
| 共享状态表示 | 模型间不共享 | 全动作共享 Transformer 主干 |
| 类别不平衡风险 | 通过拆模型缓解 | 通过动作权重、指标和合法候选集合缓解 |

本项目的统一候选动作设计更适合国标麻将，因为吃碰动作常常必须与后续弃牌联合考虑。
但当前统一模型也更容易受到大量 `PASS` 和 `PLAY` 样本支配。后续可保留候选动作评分器，
同时增加动作家族辅助头，结合两种设计的优点。

## 2.2 Features：状态表示

### Suphx

Suphx 将状态编码为大量 `34 x 1` 通道，输入深度卷积网络。论文中的特征包括：

- 自己的暗手牌；
- 各家副露；
- 宝牌；
- 按顺序排列的弃牌；
- 四家累计分数；
- 剩余牌数；
- 场次、庄家、连庄、本场棒和立直棒；
- 面向不同弃牌的 100 多组 look-ahead features。

Suphx 的 look-ahead features 通过简化 DFS 估计：打出某张牌后，通过替换若干牌形成不同
和牌形状的概率与得分。该设计将麻将规则计算结果显式提供给策略网络。

### 本项目

本项目 Feature V2 只将线上可见公开信息输入 Actor，编码为最多 256 个 token：

- 全局阶段、当前行动者、最后弃牌、场风和剩余牌；
- 自己的暗手牌 token；
- 四家的有序弃牌 token，以及弃牌是否被副露消耗；
- 四家的副露类型、来源和牌张 token；
- 最近 64 个公开事件 token；
- 34 种牌的不可见数量估计 token；
- 各家的估计暗手牌数量和剩余摸牌数量。

类别字段通过 embedding 编码，包括：

- token 类型；
- 牌 ID；
- 相对座位；
- 动作类型。

相关实现：

- `mahjong_agent/features/token_encoder.py`
- `mahjong_agent/features/encoder.py`

### 相同点

- 都只把正常玩家可见信息输入最终部署策略；
- 都显式编码手牌、副露、弃牌顺序、场况和剩余牌信息；
- 都试图将复杂麻将状态转换为适合深度网络处理的结构化输入；
- 都使用相对稳定的规则信息辅助策略判断。

### 不同点与影响

| 维度 | Suphx | 本项目 |
|---|---|---|
| 表示形式 | 34 列多通道张量 | 异质 token 序列 |
| 主干适配 | 深层 CNN | Transformer |
| look-ahead | 100+ 显式搜索特征 | 没有完整 look-ahead；仅有简单向听、有效牌势能用于 PPO reward |
| 对局累计分数 | 包含 | 当前单局环境不包含 |
| 对手 belief | Oracle 阶段直接使用完美特征 | 使用辅助头预测对手牌数量 |
| 花牌 | 日麻不使用国标花牌机制 | Botzone 协议记录花牌，但训练环境和 V2 Actor 未完整编码 |

本项目 token 表示更自然地保留事件顺序和异质关系，但当前缺少国标麻将最关键的显式规则
特征，例如：

- 8 番可达性；
- 国标番种路线；
- 每个候选弃牌的有效牌与成番变化；
- 每个候选弃牌的放铳概率和预期损失；
- 花牌数量；
- 暗杠隐藏信息的正确表示。

## 2.3 Model Structures：模型结构

### Suphx

Suphx 使用深层残差 CNN。论文描述的模型大致包含：

- 34 列输入；
- 256 通道卷积；
- 大量重复残差卷积块；
- 弃牌模型输出 34 类；
- 立直、吃、碰、杠模型输出二分类结果；
- 不使用 pooling，避免丢失每种牌对应列的语义。

### 本项目

Feature V2 模型为 Token Transformer：

```text
公开状态 token
  -> 线性投影 + kind/tile/player/action embedding
  -> 4 层 Transformer Encoder
  -> masked mean pooling
  -> 全局 state embedding

候选动作 token
  -> 线性投影 + embedding
  -> mean pooling
  -> action embedding

[state embedding, action embedding]
  -> MLP
  -> candidate logit
```

默认参数：

- `d_model = 192`
- `layers = 4`
- `heads = 6`
- `dropout = 0.1`

此外模型包含：

- value head；
- win / deal-in / final score / 8-fan outcome head；
- fan bucket head；
- 三家对手手牌计数 belief head；
- 可选 belief-to-actor adapter。

相关实现：

- `mahjong_agent/models/token_transformer.py`
- `configs/model/base.yaml`

### 对比结论

本项目模型参数规模和训练成本远低于 Suphx，适合课程项目与 Botzone 部署。统一候选动作
评分器也比 Suphx 的固定输出分类器更自然地适配动态合法动作集合。

但当前结构存在两个主要瓶颈：

1. 状态先被压缩成单个 mean-pooled 向量，再与动作交互，候选动作无法直接关注相关状态
   token；
2. 吃碰后的弃牌目前主要以连续数值字段编码，没有充分利用 categorical tile embedding。

更合理的下一代结构是让候选动作通过 cross-attention 直接读取状态 token，或将候选动作
token 与状态 token 联合送入 Transformer。

---

## 3. Learning Algorithm：学习算法

## 3.1 Supervised Learning / Behavior Cloning

### Suphx

Suphx 从顶尖人类玩家日志中提取 `(state, action)`，分别监督训练五个模型。论文报告：

| 模型 | 训练样本 | 测试准确率 |
|---|---:|---:|
| Discard | 15M | 76.7% |
| Riichi | 5M | 85.7% |
| Chow | 10M | 95.0% |
| Pong | 10M | 91.9% |
| Kong | 4M | 94.0% |

这是标准 Behavior Cloning：训练本身属于监督学习，但策略部署后会进入由自身动作产生的
状态分布，因此仍存在 distribution shift。

### 本项目

本项目从官方 Botzone 数据中的 98,209 局牌谱提取所有决策点，构建公开 observation、
合法候选动作和人类目标动作，并训练统一动作评分器。

BC 目标包括：

```text
policy loss
+ outcome binary losses
+ final score MSE
+ fan bucket classification
+ opponent-hand belief classification
+ belief tile-count constraint
```

训练机制包括：

- 合法候选动作上的交叉熵；
- 默认 `0.03` label smoothing；
- 对 `GANG / BUGANG / HU` 进行温和加权；
- AdamW；
- gradient clipping；
- mixed precision；
- DistributedDataParallel；
- ReduceLROnPlateau；
- early stopping；
- 按动作类型报告 accuracy、precision 和 recall；
- 使用固定牌墙实战分数选择最佳 BC checkpoint。

相关实现：

- `scripts/preprocess_official_full_actions.py`
- `scripts/build_tensor_cache.py`
- `scripts/train_bc.py`
- `mahjong_agent/training/dataset.py`
- `scripts/select_best_bc.py`

### 相同点

- 都用人类牌谱完成策略冷启动；
- 都将牌谱动作作为监督标签；
- 都对不同动作阶段建模；
- 都将 SL 模型作为后续 RL 的初始化策略。

### 差异与评价

| 维度 | Suphx | 本项目 |
|---|---|---|
| 数据来源 | 顶尖人类玩家 | 官方 Botzone 对局，未按玩家水平筛选 |
| 数据规模 | 各模型 4M-15M 样本 | 98,209 局；当前仓库旧 artifact 约 257 万训练决策 |
| 模型组织 | 分动作独立训练 | 全动作统一训练 |
| 辅助任务 | 论文 SL 部分主要为动作分类 | outcome、fan、belief 多任务 |
| 模型选择 | 测试准确率与后续大规模对战 | macro accuracy 加固定牌墙复式得分 |

本项目的多任务 BC 比 Suphx 论文公开的 SL 目标更丰富，但牌谱质量控制较弱，而且当前
预处理存在必须先修复的数据可信度问题：

- 对手暗杠牌可能作为公开副露泄漏给 Actor；
- 目标动作不在生成合法集合时会被强行追加，掩盖状态重放错误；
- 解析异常后没有丢弃整局；
- 补杠后预处理状态未完整更新；
- Ignore 子句中的稀有正动作没有全部保存；
- 最终番数标签被无条件应用到所有玩家。

在这些问题修复前，扩大模型或增加数据增强可能放大错误监督。

## 3.2 Distributed Reinforcement Learning with Entropy Regularization

### Suphx

Suphx 使用分布式异步 Policy Gradient：

- 多个 CPU Mahjong simulator 生成自对弈轨迹；
- GPU inference engine 批量推理；
- experience replay buffer 存储轨迹；
- parameter server 使用多 GPU 更新最新策略；
- 使用 importance sampling 修正旧策略生成的轨迹；
- 动态调整 entropy 系数，使策略熵接近目标值。

Suphx 的 RL 主要提升 discard model，其他模型大多继承 SL 权重，以降低训练成本。

### 本项目

本项目从最佳 BC checkpoint 初始化 PPO，并联合更新统一策略模型的所有动作：

- clipped PPO objective；
- GAE；
- value loss；
- entropy bonus；
- target-KL early stopping；
- 相对 BC reference policy 的 KL 正则；
- 多轮 minibatch 更新；
- DDP 多 GPU 参数同步；
- 当前模型、BC、历史模型、启发式和随机策略组成的混合对手池。

默认对手比例设计为：

```text
当前/参考模型 40%
历史模型      30%
BC/启发式     25%
随机策略       5%
```

相关实现：

- `scripts/train_ppo.py`
- `mahjong_agent/training/ppo.py`
- `mahjong_agent/training/rollout.py`
- `configs/train/ppo.yaml`

### 对比结论

| 维度 | Suphx | 本项目 |
|---|---|---|
| RL 算法 | 异步 Policy Gradient + importance sampling | PPO + GAE |
| 采样系统 | 大规模 actor/inference/parameter-server 架构 | 单机或 DDP，每个进程顺序 rollout |
| Replay buffer | 有 | 无 |
| 策略陈旧修正 | importance sampling | PPO old-policy ratio，仅限当前 batch |
| 熵控制 | 动态目标熵 | 固定 entropy coefficient |
| RL 更新范围 | 主要更新弃牌模型 | 更新统一全动作模型 |
| 对手 | 自对弈系统 | 显式混合对手池 |
| 训练规模 | 约 150 万至 250 万局，44 GPU 级别 | 默认 100 updates，每 update 少量对局 |

本项目 PPO 更容易实现和调试，但训练规模与吞吐远低于 Suphx。当前每个 update 默认只生成
少量完整对局，样本相关性和方差都较高。若继续扩大 RL，优先方向不是更换损失公式，而是：

- 批量或并行环境；
- actor-learner 解耦；
- 更高 rollout 吞吐；
- 动态熵或 KL 控制；
- 更可靠的环境规则；
- 按对手类型分别评测。

## 3.3 Global Reward Prediction

### Suphx

Suphx 的 Global Reward Prediction 解决多局比赛中的奖励归因问题。

其 reward predictor 是两层 GRU 加两层全连接网络，输入当前局和之前各局的信息，预测
完整比赛结束后的最终 reward：

```text
round 1 info -> GRU
round 2 info -> GRU
...
round k info -> GRU -> predicted final game reward
```

第 `k` 局的训练奖励为：

\[
r_k = \Phi(x_1,\ldots,x_k)-\Phi(x_1,\ldots,x_{k-1})
\]

这样可以区分“为了保持总排名而战术性输掉一局”和“真正的错误策略”。

### 本项目

本项目具有三个相关但不等价的机制：

1. **终局 outcome 辅助头**：从每个决策状态预测最终和牌、放铳、得分和是否达到 8 番；
2. **value head**：PPO 中预测当前轨迹的 return；
3. **公开信息势能塑形**：根据向听、有效牌、简单番可行性和风险估计构造潜势差。

相关实现：

- `mahjong_agent/models/token_transformer.py`
- `scripts/train_bc.py`
- `mahjong_agent/training/reward.py`
- `scripts/train_ppo.py`

### 状态判定

**本项目未实现 Suphx 意义上的 Global Reward Prediction。**

原因不是仅缺一个 GRU，而是任务定义不同：

- Suphx 输入跨多个 round 的历史，并预测完整 game 最终排名 reward；
- 本项目当前环境只模拟单局，没有跨局累计分数和比赛排名；
- 本项目 outcome head 预测的是单局终局结果；
- 本项目 potential shaping 是手工设计的公开状态势能，不是学习得到的跨局 reward
  predictor。

### 适用于本项目的改造方向

对于当前单局国标任务，更合适的“Global Reward Prediction”变体是：

- 使用完整终局得分分布，而不是单一 MSE；
- 预测和牌概率、放铳概率、流局听牌概率和条件番数；
- 预测各候选动作的长期结果，而不只预测当前状态；
- 使用 privileged full-state predictor 校正运气因素；
- 如果未来扩展为多局比赛，再加入跨局 GRU 或 Transformer。

## 3.4 Oracle Guiding

### Suphx

Suphx 的 oracle agent 在训练时可以看到：

- 自己手牌；
- 所有公开信息；
- 三家对手暗手牌；
- 牌墙中的牌。

Suphx 先使用完整信息训练一个强 oracle policy，然后逐渐降低 perfect feature 的保留概率，
最终完全移除隐藏信息，使同一个策略从 oracle agent 平滑过渡为 normal agent。隐藏特征
dropout 从 1 逐渐衰减到 0。

论文还指出，简单知识蒸馏效果不佳，因为 normal agent 无法完全模仿拥有额外信息的
oracle agent。

### 本项目

本项目在自对弈和牌谱预处理中保存三家对手真实手牌计数，并训练 belief head：

```text
public observation
  -> shared Transformer
  -> predict each opponent's count of each tile
```

`belief_mode=aux` 时，belief 只作为辅助监督，Actor 不读取真实对手手牌。

`belief_mode=actor` 时，Actor 读取模型自己预测出的 belief 概率，并通过
`belief_adapter` 加到 state embedding；该概率使用 stop-gradient，仍不暴露真实隐藏信息。

相关实现：

- `mahjong_agent/models/token_transformer.py`
- `scripts/preprocess_official_full_actions.py`
- `scripts/build_tensor_cache.py`
- `scripts/train_bc.py`
- `scripts/train_ppo.py`
- `mahjong_agent/training/rollout.py`

### 状态判定

**本项目实现了 privileged auxiliary supervision，但未实现完整 Oracle Guiding。**

| Oracle Guiding 要素 | Suphx | 本项目 |
|---|---|---|
| 对手真实手牌作为训练特权信息 | 有 | 有，作为 belief label |
| 真实牌墙作为训练特权信息 | 有 | 环境可访问，但未作为模型训练目标 |
| Oracle policy 直接读取完整状态 | 有 | 无 |
| Oracle policy 先独立训练 | 有 | 无 |
| Perfect feature dropout 逐渐降为零 | 有 | 无 |
| Oracle 到 normal policy 连续迁移 | 有 | 无 |
| Oracle critic | 论文未来方向 | 当前未实现 |

本项目的设计更接近“隐藏状态重建辅助任务”，优点是部署安全、实现稳定；缺点是对策略的
指导较弱。下一步最推荐实现 asymmetric actor-critic：

- Actor 只读取公开观测；
- Critic 在训练时读取公开观测、四家真实手牌和牌墙计数；
- 使用 privileged critic 降低 advantage 方差；
- 部署时仅导出 Actor。

这比直接将完整隐藏信息输入 Actor 再逐渐 dropout 更容易保证线上合法性，也更适合当前
PPO 代码结构。

## 3.5 Parametric Monte-Carlo Policy Adaptation

### Suphx

Suphx 的 pMCPA 在每局开始获得起手牌后：

1. 固定自己的起手牌；
2. 随机采样对手手牌和牌墙；
3. 使用离线策略完成多次 rollout；
4. 根据 rollout 对策略参数做临时梯度更新；
5. 使用该局专属的适应后策略进行决策；
6. 下一局重新从离线策略开始。

该方法不是常规 MCTS，而是通过有限采样直接调整网络参数，使更新后的策略泛化到未访问
状态。

### 本项目

本项目部署阶段：

- 从 Botzone 公开历史重建状态；
- 生成严格合法动作；
- 使用静态离线模型进行一次前向推理；
- 使用 argmax 选择动作；
- 非法或无法证明合法的动作由安全层替换。

相关实现：

- `botzone_entry.py`
- `mahjong_agent/botzone/protocol.py`
- `mahjong_agent/botzone/legality.py`
- `mahjong_agent/policies/model.py`

### 状态判定

**本项目未实现任何运行时策略适应。**

考虑 Botzone 的时延和运行环境限制，直接实现 Suphx pMCPA 成本较高。更现实的渐进路线：

1. 对 top-k 候选弃牌进行小规模确定化 rollout；
2. 使用 belief head 采样对手手牌；
3. 使用固定模型评价，不在运行时更新参数；
4. 仅在推理预算允许时加入轻量 candidate reranking；
5. 最后再研究临时 adapter 参数更新。

---

## 4. Reward、辅助任务与训练信号

### Suphx

Suphx 的主要训练信号包括：

- 顶尖人类动作标签；
- 自对弈 Policy Gradient；
- 动态 entropy regularization；
- Global Reward Prediction 生成的跨局 reward；
- Oracle 完整状态；
- look-ahead features 表达可能和牌形状与分数。

### 本项目

本项目 PPO 的主终局奖励为：

\[
R_T = \tanh(\text{score}/64)
\]

公开状态势能包括：

- `efficiency`：向听数与有效牌剩余量；
- `fan_feasibility`：当前为简单近似值；
- `deal_in_risk`：基于可见牌的简化风险；
- `draw_tenpai`：流局听牌信号。

势能差设置单步和单局上限，并在终局加入真实得分。PPO 还加入：

- BC reference KL；
- value loss；
- entropy bonus；
- outcome、fan、belief 辅助损失。

### 对比评价

本项目奖励函数可审计、实现简单，但其 `fan_feasibility` 和 `deal_in_risk` 目前只是非常粗糙
的启发式，不足以表达国标麻将策略：

- `fan_feasibility = 1 / (shanten + 2)` 本质仍主要衡量普通成牌距离；
- 简单可见张比例不能可靠估计放铳概率；
- 当前向听计算并非完整国标 8 番向听；
- 没有候选动作级的得分与风险预测；
- 没有利用完整隐藏状态区分“策略优秀”和“发牌运气好”。

本项目需要借鉴 Suphx 的核心不是照搬 GRU，而是将“最终目标”更有效地分配到中间状态。

---

## 5. Offline Evaluation：离线实验与消融

### Suphx

Suphx 的离线实验明确比较：

- `SL`
- `SL-weak`
- `RL-basic`
- `RL-1 = RL-basic + Global Reward Prediction`
- `RL-2 = RL-1 + Oracle Guiding`

为了降低起手牌随机性，Suphx 对每个 Agent 使用相同的大规模随机对局条件，并进行了约
一百万局评测。RL 训练本身约使用 150 万局，消耗 44 张 GPU 两天。

此外，Suphx 独立评估 pMCPA：固定起手牌生成训练和测试 rollout，比较适应前后策略。

### 本项目

本项目提供固定牌墙复式评测：

- 保存并复用牌墙；
- 待测模型轮换四个座位；
- 对每个牌墙计算平均表现；
- 输出平均分、标准差和 95% 置信区间；
- 输出和牌、自摸、放铳、流局听牌、番数、动作分布和推理时延；
- 使用相同牌墙比较 BC 和 PPO；
- 只有 PPO 相对 BC 的配对分数差 95% CI 下界不低于 0 时才采用 PPO。

相关实现：

- `mahjong_agent/evaluation/evaluator.py`
- `scripts/evaluate.py`
- `scripts/select_best_bc.py`
- `scripts/select_best_ppo.py`

### 对比评价

固定牌墙和座位轮换是本项目很重要的工程优势，能够在有限计算预算下显著降低方差。
但当前实验规模和证据不足：

- 流水线建议 400 局，远少于 Suphx 的百万局评测；
- 仓库现有 `bc_eval.json` 只有 40 局；
- 尚未保存针对 GRP、belief、reward shaping、对手池等组件的系统消融报告；
- 当前评测对手主要是启发式策略，无法证明对公开竞赛强模型的泛化能力；
- 本地环境仍缺少完整花牌和部分 Botzone 裁判边界。

建议建立如下消融矩阵：

| 实验 | 唯一变化 | 主要观察指标 |
|---|---|---|
| BC-base | 仅 policy loss | 固定牌墙得分、PLAY accuracy |
| BC-outcome | 加 outcome heads | 得分、放铳率、Brier score |
| BC-belief | 加 belief auxiliary | 得分、belief CE、放铳率 |
| BC-belief-actor | belief 输入 Actor | 得分、延迟、泛化 |
| PPO-terminal | 仅终局得分 | 相对 BC score delta |
| PPO-shaping | 加公开势能 | score delta、动作分布 |
| PPO-pool | 加历史对手池 | 分对手得分 |
| PPO-oracle-critic | 加特权 Critic | explained variance、score delta |

---

## 6. Online Evaluation 与部署

### Suphx

Suphx 在 Tenhou expert room 与真实玩家对局 5,000 多场，达到：

- record rank 10 dan；
- stable rank 8.74；
- 较低放铳率和第四名率；
- 超越 Bakuuchi、NAGA 和顶尖人类玩家整体表现。

pMCPA 因实时计算成本没有部署到正式 Tenhou 测试版本。

### 本项目

本项目已建立 Botzone 部署链：

- Botzone 协议状态重放；
- 严格合法动作生成；
- 模型仅从合法动作中选择；
- 输出前二次校验；
- 无法证明合法的响应动作回退为 `PASS`；
- 无法证明合法的自摸动作回退为启发式弃牌；
- 模型 checkpoint 与源码 zip 分离；
- 提供导出和本地部署验证脚本。

相关实现：

- `botzone_entry.py`
- `mahjong_agent/botzone/protocol.py`
- `mahjong_agent/botzone/legality.py`
- `scripts/export_bot.py`
- `scripts/verify_deploy.py`

### 对比评价

本项目在部署安全性方面比 Suphx 论文公开内容描述得更具体，但目前没有与 Suphx
Tenhou 结果同等级的线上统计证据。必须明确区分：

- “可以上传 Botzone”；
- “本地规则测试通过”；
- “在 Botzone 对强对手长期取得高分”。

当前项目只充分覆盖前两项的一部分，尚不能声称达到 Suphx 级别的实战验证。

---

## 7. 技术组件状态总表

| Suphx 组件 | 本项目对应实现 | 状态 | 结论 |
|---|---|---|---|
| 顶尖人类牌谱 SL | 官方 Botzone 牌谱 BC | 部分实现 | 有 BC，但未按玩家水平筛选 |
| 分动作模型 | 统一候选动作评分器 | 替代实现 | 更统一，但需处理类别不平衡 |
| 深层残差 CNN | Token Transformer | 替代实现 | 更适合事件序列，动作交互仍弱 |
| Look-ahead features | 向听、有效牌和势能 | 弱对应 | 缺少候选动作级成番/得分搜索 |
| 分布式 Policy Gradient | DDP PPO | 部分实现 | 算法稳定但采样吞吐低 |
| 动态 entropy 控制 | 固定 entropy bonus | 未实现 | 建议加入 target entropy |
| Replay buffer + stale correction | 当前 batch PPO ratio | 未实现 | 无异步 replay |
| Global Reward Prediction | outcome/value heads | 弱对应 | 不是 Suphx 的跨局 GRP |
| Oracle Guiding | belief privileged labels | 部分思想 | 没有 oracle policy 和 feature dropout |
| Oracle Critic | 无 | 未实现 | 推荐作为下一阶段重点 |
| pMCPA | 静态模型推理 | 未实现 | 可先做 top-k belief rollout |
| 百万局离线评测 | 固定牌墙复式评测 | 方法部分实现 | 统计规模不足 |
| 大规模线上评测 | Botzone 导出与验证 | 部署实现，实战证据不足 | 需要长期线上结果 |

---

## 8. 本项目相对 Suphx 的技术优势

尽管整体成熟度与训练规模不及 Suphx，本项目并非简单缩小版 Suphx，具有以下独立特点：

### 8.1 统一合法候选动作评分

模型不依赖固定 235 类或多个独立模型，而是对当前合法动作集合评分。该设计自然支持：

- 动态合法集合；
- 吃碰动作与后续弃牌联合决策；
- 规则引擎生成动作后模型排序；
- 同一个策略接口用于 BC 和 PPO。

### 8.2 公开信息 Actor 与 privileged labels 分离

本项目明确区分：

- Actor 可见的公开 observation；
- 训练时可用的对手手牌标签；
- 部署时不允许访问的隐藏信息。

该分离方式为未来 Oracle Critic 和蒸馏提供了清晰接口。

### 8.3 固定牌墙复式模型选择

本项目不仅比较平均得分，还使用相同牌墙、座位轮换和配对置信区间选择 BC/PPO
checkpoint。在有限预算下，这比独立随机对局更适合识别微小策略变化。

### 8.4 部署合法性安全层

模型输出不会直接发送至 Botzone，而是经过严格合法性验证和回退。这能够减少训练环境
误差、模型错误和第三方算番异常带来的非法动作风险。

---

## 9. 本项目相对 Suphx 的主要差距

### 9.1 数据质量与专家水平

Suphx 使用顶尖人类牌谱；本项目使用官方牌谱但没有玩家水平筛选，且预处理仍存在状态
重放与标签可信度问题。当前最高优先级应是建立严格数据审计，而不是增加模型容量。

### 9.2 训练规模

Suphx 使用数十 GPU、百万级自对弈和百万局评测。本项目当前 PPO rollout 数量和评测规模
都不足以稳定学习复杂防守与成番策略。

### 9.3 缺少真正的规则前瞻

Suphx 显式计算 look-ahead features。本项目 Transformer 主要依靠数据自行学习国标番型
关系，但训练数据和模型规模不足以保证学会复杂 8 番规划。

### 9.4 特权信息利用较弱

本项目只使用 belief 辅助监督，没有让完整隐藏状态直接改善 value estimation 或 policy
learning。Oracle Critic 是最直接且风险可控的改进方向。

### 9.5 规则与环境完整性

当前本地环境没有完整模拟花牌替换、多家同时和牌和全部官方协议边界。环境误差会同时
污染 PPO reward、合法动作和评测结果。

### 9.6 缺少系统消融与线上证明

本项目代码包含多个有潜力的组件，但没有像 Suphx 一样逐项证明：

- RL 是否优于 SL；
- outcome heads 是否有效；
- belief 是否有效；
- reward shaping 是否有效；
- opponent pool 是否有效；
- PPO 是否真正优于最佳 BC。

---

## 10. 面向国标麻将的 Suphx 式升级路线

### Phase 0：先修复可信训练闭环

1. 修复牌谱状态重放和隐藏信息泄漏；
2. 目标动作不在合法集合时丢弃并审计，而不是强行追加；
3. 解析异常时丢弃整局；
4. 补全补杠、暗杠、抢杠和花牌状态；
5. 增加数据集统计和规则一致性测试；
6. 使用至少 400 局固定牌墙评测，最终候选使用更大规模。

### Phase 1：实现国标 Look-ahead Features

对每个候选 `PLAY / CHI / PENG` 动作离线计算并作为辅助标签或输入：

- 动作后向听数；
- 有效牌集合和剩余有效牌数；
- 8 番可达性；
- 番种 family multi-label；
- 最大可达番与期望番；
- 公开信息下的放铳风险；
- 候选动作后的预期终局得分。

优先将这些量作为辅助预测目标，避免昂贵规则计算影响线上延迟。

### Phase 2：实现 Asymmetric Oracle Critic

```text
Actor:
  public observation -> policy logits

Oracle Critic during training:
  public observation
  + opponent true hands
  + wall counts
  -> value / return distribution
```

训练时 Oracle Critic 提供低方差 advantage，部署时完全移除。该路线最接近 Suphx
Oracle Guiding 的目标，同时不需要让 Actor 曾经依赖隐藏信息。

### Phase 3：改进策略网络中的 state-action 交互

将当前：

```text
mean_pool(state) + mean_pool(action) -> MLP
```

升级为：

```text
action query -> cross-attention over state tokens -> candidate score
```

并增加：

- `[CLS]` 或 attention pooling；
- 动作家族辅助 head；
- post-claim discard tile embedding；
- 候选动作级 outcome/risk head。

### Phase 4：扩展分布式自对弈

- 并行环境 rollout；
- actor-learner 解耦；
- 动态 entropy 或 KL 目标；
- 更强历史对手池；
- 定期将实战失败状态加入离线训练集；
- 对 BC、历史 PPO、启发式和外部模型分别评测。

### Phase 5：轻量运行时适应

不直接复制 pMCPA，而是逐步实现：

1. belief-conditioned hidden-state sampling；
2. top-k 候选动作 rollout；
3. candidate reranking；
4. 小型 adapter 的局内临时更新；
5. 严格限制推理时间和内存。

---

## 11. Suphx Conclusion and Discussions：论文未来方向对照

Suphx 在论文结论中提出了三个后续研究方向。这些方向也可以用于检查本项目当前设计是否
走在合理路径上。

### 11.1 使用完整信息衡量对局难度

Suphx 指出，单纯奖励胜利会混淆策略水平与发牌运气：好起手牌下获胜不应得到过高奖励，
困难牌局中的优秀表现则应获得更有区分力的反馈。论文建议使用完整信息衡量 round/game
difficulty，从而改善 reward predictor。

本项目当前终局奖励只依赖玩家最终得分：

\[
R_T=\tanh(\text{score}/64)
\]

它没有区分：

- 起手牌本来就很强；
- 对手起手牌和牌墙分布极其有利；
- 策略在劣势牌局中有效减少损失；
- 策略错误但依靠运气获胜。

适合本项目的实现是训练一个 privileged expected-score baseline：

```text
完整起手牌 + 对手手牌 + 牌墙统计
  -> predicted expected terminal score

advantage target
  = actual terminal score - predicted expected terminal score
```

该信号可以作为辅助目标或 Oracle Critic 的组成部分，但不应直接输入部署 Actor。

### 11.2 Oracle Distillation 与 Oracle Critic

Suphx 认为 perfect feature dropout 不是利用隐藏信息的唯一方法，还提出：

- 同时训练 oracle agent 和 normal agent；
- 使用 oracle 向 normal agent 蒸馏；
- 约束两个策略之间的距离；
- 使用 oracle critic 提供状态级即时反馈。

本项目当前的 belief auxiliary head 已经为这些方向提供了部分基础设施，但还没有：

- oracle policy；
- oracle-normal policy KL；
- privileged value network；
- oracle 给候选动作提供的排序或软标签。

结合当前 PPO 结构，Oracle Critic 比完整 Oracle Actor 蒸馏更适合作为第一步，因为它：

- 不改变线上 Actor 输入；
- 可以直接替换或增强当前 value head；
- 能降低 GAE 方差；
- 不要求正常策略模仿不可实现的完美信息动作。

### 11.3 持续 Run-time Policy Adaptation

Suphx 指出，pMCPA 不应只在起手牌发完时执行，也可以随着每次公开弃牌出现而持续适应。
由于新公开信息逐步减少隐藏状态空间，每一步只需要少量增量采样。

本项目当前完全没有运行时适应。考虑 Botzone 时限，更可行的方式不是每步更新整个
Transformer，而是：

- 缓存公开历史编码；
- 更新 belief posterior；
- 对少量候选动作进行增量 rollout；
- 只更新轻量 adapter 或候选动作偏置；
- 超过时间预算时立即回退静态策略。

### 11.4 对本项目的总体启示

Suphx 第 6 节表明，其核心思想不是某个固定网络结构，而是：

> 使用训练时可获得但部署时不可见的信息，改善奖励归因、价值估计和策略适应，同时保证
> 最终线上策略遵守信息约束。

本项目当前已经遵守“公开信息 Actor”原则，下一步应优先增强训练信号，而不是将隐藏信息
直接加入线上特征。

---

## 12. Appendix 对照：规则、排名与评测目标

### 12.1 规则差异

Suphx 附录 A 描述日式立直麻将规则。本项目国标规则与其存在以下关键差异：

| 规则维度 | Suphx / 日麻 | 本项目 / 国标 |
|---|---|---|
| 基础牌 | 136 张，34 种普通牌 | 核心环境为 136 张普通牌；Botzone 协议还涉及花牌 |
| 和牌门槛 | 至少一个役 | 至少 8 番 |
| 得分结构 | 符、番、宝牌、立直等 | 国标番种与番值累计 |
| 立直 | 核心动作，需要独立模型 | 不存在 |
| 宝牌与死墙 | 核心公开/隐藏信息 | 不存在对应日麻机制 |
| 花牌 | 通常不使用 | 国标和 Botzone 协议需要处理 |
| 比赛结构 | 多局累计分与排名 | 当前项目主要优化单局得分 |

这些差异直接决定了不能机械复制 Suphx：

- Riichi model 对本项目无意义；
- Suphx look-ahead 的役与分数计算必须替换为国标番型搜索；
- Suphx Global Reward Prediction 必须先有多局比赛环境才可原样实现；
- 本项目需要比 Suphx 更关注“完整牌形能否达到 8 番”。

### 12.2 排名系统与 Stable Rank

Suphx 附录 B 和 C 使用 Tenhou 排名规则与 stable rank 衡量长期水平。Stable rank 重点
惩罚第四名，因此会推动策略学习防守和避免大败。

本项目当前评测主要使用单局平均得分和配对 score delta，没有对应 stable rank。这使得
二者的策略风格目标不同：

- Suphx 会特别重视避免第四名和保持整场领先；
- 本项目主要最大化单局期望得分；
- 本项目的放铳率只是评测指标，不直接等价于排名惩罚；
- 当前环境无法学习“领先时保守、落后时追求高番”的跨局策略。

如果未来希望学习类似 Suphx 的全局攻防切换，需要新增完整比赛环境，并将以下信息加入
公开状态：

- 四家累计分数；
- 当前局次；
- 剩余局数；
- 当前排名；
- 达到不同最终排名所需的分数差。

### 12.3 评测可比性边界

Suphx 的 Tenhou stable rank、和牌率和放铳率不能直接与本项目单局国标结果横向比较。
合理的比较对象是技术方法和实验设计，而不是绝对数值：

- 是否使用专家监督预训练；
- RL 是否在相同随机条件下显著优于 SL；
- 隐藏信息辅助是否带来可重复提升；
- 是否有足够对局和置信区间；
- 是否有真实线上长期评测。

---

## 13. 最终结论

本项目与 Suphx 的共同主线是：

```text
人类牌谱监督预训练
-> 自对弈强化学习
-> 利用隐藏信息和终局结果改善策略
-> 线上平台部署与评测
```

但当前项目只完成了这一主线的工程化初版：

- Behavior Cloning：**已实现，但数据质量需要优先修复**；
- 自对弈强化学习：**已实现 PPO 基线，但规模和吞吐较低**；
- Global Reward Prediction：**只有单局 outcome/value 辅助头，不是完整 GRP**；
- Oracle Guiding：**只有 belief 特权辅助监督，不是完整 Oracle Guiding**；
- Run-time Policy Adaptation：**未实现**；
- 可靠评测：**方法已实现，但实际评测规模和消融证据不足**；
- 在线验证：**具备 Botzone 部署链，但缺少长期高水平实战结果**。

相对于继续堆叠 Transformer 层数，当前最有价值的升级顺序是：

1. 修复牌谱预处理、规则一致性和隐藏信息泄漏；
2. 建立完整消融实验与大规模固定牌墙评测；
3. 加入国标 8 番 look-ahead 辅助目标；
4. 实现 privileged Oracle Critic；
5. 提升自对弈吞吐和对手强度；
6. 最后研究轻量运行时适应。

只有在这些环节形成可信闭环后，本项目才能从“包含 Suphx 思想的国标麻将 Agent”
进一步发展为“经实验验证的强国标麻将系统”。

---

## 参考资料

1. Junjie Li et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning*,
   arXiv:2003.13590v2, 2020.
2. 本仓库 `2003.13590v2.pdf`。
3. 本仓库 `README.md`、`TODO20260609.md` 与对应源码实现。
