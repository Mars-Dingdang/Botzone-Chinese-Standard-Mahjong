# Training Memory

Updated: 2026-06-09 (Asia/Shanghai), optimization implementation pass

## Current Status

- Active v2 pipeline PID: `202706`
- Active v2 run: `artifacts/runs/20260609-162316`
- Current v2 stage: full-action preprocessing to `artifacts/official_bc_full_v2`
- Monitor: `tail -f artifacts/training-v2-launch.log` and `ps -p "$(cat artifacts/training-v2.pid)" -o pid,etime,stat,cmd`
- Legacy v1 pipeline PID `161313` has completed
- Active run: `artifacts/runs/20260609-153548`
- Legacy v1 stage completed 100 PPO updates; final duplicate result was model `-5.4` versus heuristic `1.8`
- Runtime: `/root/LLM_HW2/.venv`, PyTorch `2.6.0+cu124`, `PyMahjongGB==1.3.0`
- Hardware: 2 x RTX 4090 24 GB; container CPU quota is 8 cores despite exposing 112 CPUs

Monitor with:

```bash
tail -f artifacts/training-launch.log
watch -n 1 nvidia-smi
ps -p "$(cat artifacts/training.pid)" -o pid,etime,stat,cmd
```

## Data Pipeline

- Source: `Chinese-Standard-Mahjong/SL/data/data.txt`
- Parsed matches: `98,209`
- Train decisions: `4,368,883`
- Validation decisions: `231,518`
- Parse failures: `0`
- Existing v1 labels cover discard decisions; new v2 preprocessor covers all action families
- v2 validation over 250 matches produced 13,767 non-trivial decisions with 0 failures and all PASS/PLAY/CHI/PENG/GANG/ANGANG/BUGANG/HU families; candidate count range was 2-38
- v2 filters states with only one legal action, matching the bundled official preprocessor and preventing forced-PASS accuracy inflation

Processing stages:

1. `scripts/preprocess_official_data.py`: reconstructs hands/public events by complete Match, runs multi-process preprocessing, writes zstd Parquet.
2. `scripts/build_tensor_cache.py`: converts archival Parquet into fixed-shape FP16 tensor shards with padded actions and masks.

Artifacts:

- Parquet archive: `artifacts/official_bc`, about `439 MB`
- Training tensor cache: `artifacts/official_bc_tensors`, about `4.5 GB`, 786 tensor shards
- Full Parquet preprocessing: about `2,268 seconds`, roughly `43 matches/s`

## Model and Features

- `HybridTransformer`: candidate-action scorer, about `2.82M` parameters
- State: 394 public/private-observable features
- Action: 8 features per legal candidate
- Backbone: state MLP -> 4 state tokens -> 4-layer, 6-head Transformer, `d_model=192`
- Heads: candidate policy score, scalar value, 3-dimensional auxiliary placeholder
- Features include hand/visible counts, rivers, melds, prevalent wind, event counts, wall estimate, shanten and useful tiles

## Training Objectives

Behavior cloning:

```text
L_BC = -log softmax(masked_logits)[expert_action]
metric = top-1 candidate-action accuracy
```

- AdamW, learning rate `3e-4`, AMP FP16, gradient clipping `1.0`
- DDP uses balanced tensor shards and equal steps per rank
- Empirical batch: `4096` per GPU

PPO v2 implementation:

```text
R = tanh(score / 64)
A = GAE(gamma=0.99, lambda=0.95)
L = L_clipped_policy + 0.5 * MSE(V, R) - 0.01 * entropy
clip epsilon = 0.2
```

- Starts from best BC checkpoint
- Learner seat rotates; opponents mix heuristic and random policies
- Four PPO epochs per collected batch, AdamW learning rate `1e-4`
- Uses terminal reward plus low-weight shanten/useful-tile potential shaping (`0.02` by default)
- Uses normalized advantages, approximate-KL early stop, clip fraction and explained variance metrics

## Completed Results

BC 5-epoch progression:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.975 | 0.297 | 1.805 | 0.353 |
| 2 | 1.710 | 0.403 | 1.605 | 0.446 |
| 3 | 1.512 | 0.486 | 1.417 | 0.524 |
| 4 | 1.351 | 0.546 | 1.285 | 0.571 |
| 5 | 1.226 | 0.587 | 1.173 | 0.603 |

BC duplicate evaluation over 40 games:

```text
model average score:     1.8
heuristic average score: -0.6
```

Checkpoints:

- `artifacts/runs/20260609-153548/bc_model.best.pt`
- `artifacts/runs/20260609-153548/bc_model.pt`
- `artifacts/runs/20260609-153548/ppo_model.pt` saved every 10 updates

## Performance and Fixes

- Replaced single-core JSONL preprocessing with multi-process compressed Parquet: about 5x faster and over 20x smaller at matched scale.
- Added tensor cache to remove per-row Parquet `to_pylist` and Python collation from every epoch.
- Equal-global-sample benchmark: two GPUs, batch 4096 each, about 8 seconds; one GPU, batch 8192, about 11 seconds.
- Fixed DDP NCCL timeout caused by unequal rank batch counts. Shards are greedily balanced and ranks are capped to equal steps.
- Fixed PPO rollout DDP forward hang by disabling per-forward buffer broadcasts; two-GPU GAE/KL smoke test passed.
- A 200-step two-GPU BC regression test and all 10 unit tests passed.

## Known Limitations and Next Work

- Future pipelines default to `artifacts/official_bc_full_v2` and `artifacts/official_bc_full_v2_tensors`; full-scale v2 preprocessing has not run yet.
- Environment now collects all three claim responses, resolves priority, passes official fan context and scores by real fan count.
- Robbing a kong and remaining official-rule edge cases still need differential tests against the bundled official environment.
- PPO rollout is CPU/environment limited and GPU utilization is expected to be low.
- Auxiliary head has no supervised targets yet.
- Belief teacher-student, fan-template prediction, persistent opponent pool, actor-learner/V-trace and search remain future work.
- Final PPO checkpoint still needs duplicate evaluation against BC, heuristic and historical PPO checkpoints.
