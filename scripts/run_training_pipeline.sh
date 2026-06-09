#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${RUN_DIR:-artifacts/runs/$(date +%Y%m%d-%H%M%S)}"
DATA_DIR="${DATA_DIR:-artifacts/official_bc_full_v2}"
TENSOR_DIR="${TENSOR_DIR:-artifacts/official_bc_full_v2_tensors}"
mkdir -p "$RUN_DIR"
echo "run_dir=$RUN_DIR"
printf '%s\n' "$RUN_DIR" > artifacts/latest_run.txt
if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
  if find "$DATA_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "clearing incomplete preprocessed data: $DATA_DIR"
    rm -rf "$DATA_DIR"/*
  fi
  python -u scripts/preprocess_official_full_actions.py --output-dir "$DATA_DIR" --workers "${PREPROCESS_WORKERS:-8}"
else
  echo "using existing preprocessed data: $DATA_DIR"
fi
if [[ ! -f "$TENSOR_DIR/tensor_metadata.json" ]]; then
  if [[ -d "$TENSOR_DIR" ]] && find "$TENSOR_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "clearing incomplete tensor cache: $TENSOR_DIR"
    rm -rf "$TENSOR_DIR"/*
  fi
  python -u scripts/build_tensor_cache.py --input-dir "$DATA_DIR" --output-dir "$TENSOR_DIR" --workers "${CACHE_WORKERS:-8}" --max-actions 64
else
  echo "using existing tensor cache: $TENSOR_DIR"
fi
torchrun --standalone --nproc_per_node=2 scripts/train_bc.py --data "$TENSOR_DIR" --output "$RUN_DIR/bc_model.pt" --epochs "${BC_EPOCHS:-5}" --batch-size "${BC_BATCH_SIZE:-4096}"
python scripts/evaluate.py --model "$RUN_DIR/bc_model.best.pt" --games "${EVAL_GAMES:-40}" --duplicate | tee "$RUN_DIR/bc_eval.json"
torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py --checkpoint "$RUN_DIR/bc_model.best.pt" --output "$RUN_DIR/ppo_model.pt" --updates "${PPO_UPDATES:-100}" --games-per-update "${PPO_GAMES_PER_UPDATE:-8}" --save-every 10
python scripts/evaluate.py --model "$RUN_DIR/ppo_model.pt" --games "${EVAL_GAMES:-40}" --duplicate | tee "$RUN_DIR/ppo_eval.json"
