# Training Memory

Updated: 2026-06-12 (Asia/Shanghai), active v4 training pass

## Current Status

- Active full-v3 run: `artifacts/runs/20260611-210636`
- Current stage: BC training in detached screen `mahjong-20260611-210636`
- Launch command: `bash scripts/run_training_pipeline.sh start --from bc --run-dir artifacts/runs/20260611-210636`
- Runtime: PyTorch environment with 2 x RTX 4090 24 GB
- Effective BC configuration: batch `1536` per GPU, global batch `3072`, 15 epochs, patience 3
- Steady-state BC memory is about 19.5 GiB per GPU with about 95% sampled utilization

Monitor with:

```bash
bash scripts/run_training_pipeline.sh status --run-dir artifacts/runs/20260611-210636
tail -f artifacts/runs/20260611-210636/logs/pipeline.log
watch -n 1 nvidia-smi
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
- Validated long-run batch on 2 x RTX 4090: `1536` per GPU (`3072` global)

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
- The pipeline previously ignored `configs/train/bc.yaml` and hard-coded batch 4096 per GPU, causing 24 GiB OOM even after the YAML was changed. It now resolves YAML first and permits `BC_BATCH_SIZE` override.
- Batch-local padding trimming reduces the state sequence from fixed 256 tokens to the current batch maximum and trims padded action candidates before GPU transfer.
- Batch 2048 per GPU passed smoke tests but reached about 23.2 GiB, so the long run uses safer batch 1536 at about 19.5 GiB.
- BC and PPO DDP use unused-parameter detection because aux-mode leaves actor-only/value-only parameters outside some losses.
- Fixed DDP NCCL timeout caused by unequal rank batch counts. Shards are greedily balanced and ranks are capped to equal steps.
- Fixed PPO rollout DDP forward hang by disabling per-forward buffer broadcasts; two-GPU GAE/KL smoke test passed.
- A 200-step two-GPU BC regression test and all 10 unit tests passed.

## Known Limitations and Next Work

- The active pipeline uses `artifacts/official_bc_v4` and `artifacts/official_bc_v4_tensors`; legacy `official_bc_full_*` paths are migrated by the pipeline.
- Environment now collects all three claim responses, resolves priority, passes official fan context and scores by real fan count.
- Robbing a kong and remaining official-rule edge cases still need differential tests against the bundled official environment.
- PPO rollout is CPU/environment limited and GPU utilization is expected to be low.
- Auxiliary head has no supervised targets yet.
- Belief teacher-student, fan-template prediction, persistent opponent pool, actor-learner/V-trace and search remain future work.
- The active v4 pipeline still needs to finish BC selection, PPO training, and duplicate model selection.
