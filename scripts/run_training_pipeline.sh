#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_training_pipeline.sh start [--from data|bc|rl|eval] [--run-dir PATH] [--bc-run-dir PATH]
  bash scripts/run_training_pipeline.sh resume [--run-dir PATH]
  bash scripts/run_training_pipeline.sh status [--run-dir PATH]
  bash scripts/run_training_pipeline.sh attach [--run-dir PATH]
  bash scripts/run_training_pipeline.sh stop [--run-dir PATH]
EOF
}

command="${1:-start}"; shift || true
from_stage="data"
run_dir="${RUN_DIR:-}"
bc_run_dir="${BC_RUN_DIR:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) from_stage="$2"; shift 2 ;;
    --run-dir) run_dir="$2"; shift 2 ;;
    --bc-run-dir) bc_run_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ "$command" != "start" && -z "$run_dir" && -f artifacts/latest_run.txt ]]; then
  run_dir="$(cat artifacts/latest_run.txt)"
fi
if [[ "$command" == "start" && -z "${RUN_DIR:-}" && -z "$run_dir" ]]; then
  run_dir="artifacts/runs/$(date +%Y%m%d-%H%M%S)"
fi
run_dir="${run_dir:-artifacts/runs/$(date +%Y%m%d-%H%M%S)}"
session="mahjong-$(basename "$run_dir")"

case "$command" in
  status)
    echo "run_dir=$run_dir session=$session"
    screen -ls | grep -F "$session" || true
    [[ -f "$run_dir/stage.txt" ]] && echo "stage=$(cat "$run_dir/stage.txt")"
    [[ -f "$run_dir/logs/ppo_metrics.jsonl" ]] && tail -1 "$run_dir/logs/ppo_metrics.jsonl"
    exit 0 ;;
  attach) exec screen -r "$session" ;;
  stop) screen -S "$session" -X quit; exit 0 ;;
  resume)
    [[ -d "$run_dir" ]] || { echo "run directory not found: $run_dir" >&2; exit 1; }
    from_stage="$(cat "$run_dir/stage.txt" 2>/dev/null || echo data)"
    [[ "$from_stage" == "complete" ]] && from_stage="eval"
    ;;
  start) ;;
  _run) ;;
  *) usage; exit 2 ;;
esac

if [[ "$command" != "_run" ]]; then
  command -v screen >/dev/null || { echo "screen is required" >&2; exit 1; }
  mkdir -p "$run_dir"
  printf '%s\n' "$run_dir" > artifacts/latest_run.txt
  echo "starting screen session=$session run_dir=$run_dir from=$from_stage"
  launch=(bash "$0" _run --from "$from_stage" --run-dir "$run_dir")
  [[ -n "$bc_run_dir" ]] && launch+=(--bc-run-dir "$bc_run_dir")
  exec screen -dmS "$session" "${launch[@]}"
fi

mkdir -p "$run_dir/logs" "$run_dir/evaluations" "$run_dir/configs"
printf '%s\n' "$run_dir" > artifacts/latest_run.txt
exec >> "$run_dir/logs/pipeline.log" 2>&1

DATA_DIR="${DATA_DIR:-artifacts/official_bc_v4}"
TENSOR_DIR="${TENSOR_DIR:-artifacts/official_bc_v4_tensors}"
BC_CONFIG="${BC_CONFIG:-configs/train/bc.yaml}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("batch_size", 256))' "$BC_CONFIG")}"
BC_EPOCHS="${BC_EPOCHS:-$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("epochs", 50))' "$BC_CONFIG")}"
BC_PATIENCE="${BC_PATIENCE:-$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("patience", 8))' "$BC_CONFIG")}"
if [[ "$DATA_DIR" == "artifacts/official_bc_full_v3" || "$DATA_DIR" == "artifacts/official_bc_full_v4" || "$DATA_DIR" == "artifacts/official_bc_full_v5" ]]; then
  echo "replacing obsolete DATA_DIR=$DATA_DIR with artifacts/official_bc_v4"
  DATA_DIR="artifacts/official_bc_v4"
fi
if [[ "$TENSOR_DIR" == "artifacts/official_bc_full_v3_tensors" || "$TENSOR_DIR" == "artifacts/official_bc_full_v4_tensors" || "$TENSOR_DIR" == "artifacts/official_bc_full_v5_tensors" ]]; then
  echo "replacing obsolete TENSOR_DIR=$TENSOR_DIR with artifacts/official_bc_v4_tensors"
  TENSOR_DIR="artifacts/official_bc_v4_tensors"
fi
GPUS="${GPUS:-2}"
EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_SEED="${EVAL_SEED:-2026}"
WALL_MANIFEST="$run_dir/evaluations/walls.json"

if [[ -n "$bc_run_dir" ]]; then
  [[ -d "$bc_run_dir" ]] || { echo "BC run directory not found: $bc_run_dir" >&2; exit 1; }
  bc_source="$bc_run_dir/bc_model.best.pt"
  [[ -f "$bc_source" ]] || bc_source="$bc_run_dir/bc_model.pt"
  [[ -f "$bc_source" ]] || {
    echo "BC checkpoint not found in $bc_run_dir (expected bc_model.best.pt or bc_model.pt)" >&2
    exit 1
  }
  if [[ ! -f "$run_dir/bc_model.best.pt" ]]; then
    cp "$bc_source" "$run_dir/bc_model.best.pt"
    [[ -f "$bc_source.json" ]] && cp "$bc_source.json" "$run_dir/bc_model.best.pt.json"
  fi
  printf '%s\n' "$bc_run_dir" > "$run_dir/bc_source_run.txt"
fi

if [[ ! -f "$run_dir/run_manifest.json" ]]; then
python - "$run_dir/run_manifest.json" "$from_stage" "$EVAL_SEED" "$DATA_DIR" "$TENSOR_DIR" "$bc_run_dir" <<'PY'
import json, os, sys, time
with open(sys.argv[1], "w") as handle:
    json.dump({"created_at": time.time(), "from_stage": sys.argv[2],
               "eval_seed": int(sys.argv[3]), "data_dir": sys.argv[4],
               "tensor_dir": sys.argv[5], "bc_source_run": sys.argv[6] or None,
               "environment": {
                   key: value for key, value in os.environ.items()
                   if key.startswith(("BC_", "PPO_", "EVAL_", "PREPROCESS_", "CACHE_"))
               }}, handle, indent=2, sort_keys=True)
PY
fi
cp configs/train/bc.yaml configs/train/ppo.yaml configs/model/base.yaml configs/eval/default.yaml "$run_dir/configs/" 2>/dev/null || true

stage_index() { case "$1" in data) echo 0;; bc) echo 1;; rl) echo 2;; eval) echo 3;; *) return 1;; esac; }
start_index="$(stage_index "$from_stage")"
run_stage() { [[ "$start_index" -le "$(stage_index "$1")" ]]; }

if run_stage data; then
  echo data > "$run_dir/stage.txt"
  if [[ ! -f "$DATA_DIR/metadata.json" ]] || ! grep -q '"version": 4' "$DATA_DIR/metadata.json"; then
    python -u scripts/preprocess_official_full_actions.py \
      --output-dir "$DATA_DIR" --workers "${PREPROCESS_WORKERS:-8}" \
      2>&1 | tee "$run_dir/logs/preprocess.log"
  fi
  if [[ ! -f "$TENSOR_DIR/tensor_metadata.json" ]] || ! grep -q '"cache_version": 2' "$TENSOR_DIR/tensor_metadata.json"; then
    python -u scripts/build_tensor_cache.py --input-dir "$DATA_DIR" \
      --output-dir "$TENSOR_DIR" --workers "${CACHE_WORKERS:-8}" --max-actions 64 \
      2>&1 | tee "$run_dir/logs/tensor_cache.log"
  fi
fi

if run_stage bc; then
  echo bc > "$run_dir/stage.txt"
  bc_resume=()
  [[ -f "$run_dir/bc_model.pt" ]] && bc_resume=(--resume "$run_dir/bc_model.pt")
  torchrun --standalone --nproc_per_node="$GPUS" scripts/train_bc.py \
    --data "$TENSOR_DIR" --output "$run_dir/bc_model.pt" \
    --config "$BC_CONFIG" --model-config "${MODEL_CONFIG:-configs/model/base.yaml}" \
    --epochs "$BC_EPOCHS" --patience "$BC_PATIENCE" \
    --batch-size "$BC_BATCH_SIZE" --metrics-jsonl "$run_dir/logs/bc_metrics.jsonl" \
    --seed "$EVAL_SEED" \
    "${bc_resume[@]}" 2>&1 | tee "$run_dir/logs/bc.log"
  python scripts/select_best_bc.py --checkpoint-glob "$run_dir/bc_model.epoch-*.pt" \
    --output "$run_dir/bc_model.best.pt" \
    --report "$run_dir/evaluations/bc_selection.json" \
    --results-dir "$run_dir/evaluations/bc_checkpoints" \
    --games "$EVAL_GAMES" --seed "$EVAL_SEED" --wall-manifest "$WALL_MANIFEST" \
    2>&1 | tee "$run_dir/logs/bc_selection.log"
  python scripts/evaluate.py --model "$run_dir/bc_model.best.pt" --policy-name bc \
    --games "$EVAL_GAMES" --seed "$EVAL_SEED" --duplicate --progress \
    --save-wall-manifest "$WALL_MANIFEST" \
    --output-json "$run_dir/evaluations/bc_eval.json" \
    2>&1 | tee "$run_dir/logs/bc_eval.log"
fi

if run_stage rl; then
  echo rl > "$run_dir/stage.txt"
  ppo_resume=()
  ppo_overrides=()
  [[ -n "${PPO_GAMES_PER_UPDATE:-}" ]] && ppo_overrides+=(--games-per-update "$PPO_GAMES_PER_UPDATE")
  [[ -n "${PPO_ROLLOUT_ENVS:-}" ]] && ppo_overrides+=(--rollout-envs "$PPO_ROLLOUT_ENVS")
  [[ -n "${PPO_LEAGUE_GAMES:-}" ]] && ppo_overrides+=(--league-games "$PPO_LEAGUE_GAMES")
  [[ -n "${PPO_LR:-}" ]] && ppo_overrides+=(--lr "$PPO_LR")
  [[ -n "${PPO_TARGET_KL:-}" ]] && ppo_overrides+=(--target-kl "$PPO_TARGET_KL")
  [[ -n "${PPO_BC_KL_COEF:-}" ]] && ppo_overrides+=(--bc-kl-coef "$PPO_BC_KL_COEF")
  [[ -n "${PPO_REWARD_MODE:-}" ]] && ppo_overrides+=(--reward-mode "$PPO_REWARD_MODE")
  [[ -n "${PPO_BELIEF_MODE:-}" ]] && ppo_overrides+=(--belief-mode "$PPO_BELIEF_MODE")
  if [[ -f "$run_dir/ppo_model.pt" ]]; then
    ppo_resume=(--resume "$run_dir/ppo_model.pt")
  else
    shopt -s nullglob
    ppo_checkpoints=("$run_dir"/ppo_model.update-*.pt)
    shopt -u nullglob
    if [[ "${#ppo_checkpoints[@]}" -gt 0 ]]; then
      latest_index=$((${#ppo_checkpoints[@]} - 1))
      ppo_resume=(--resume "${ppo_checkpoints[$latest_index]}")
    fi
  fi
  ppo_args=(
    --checkpoint "$run_dir/bc_model.best.pt" --output "$run_dir/ppo_model.pt"
    --config "${PPO_CONFIG:-configs/train/ppo.yaml}"
    --updates "${PPO_UPDATES:-100}"
    --save-every "${PPO_SAVE_EVERY:-10}" --metrics-jsonl "$run_dir/logs/ppo_metrics.jsonl"
    --seed "$EVAL_SEED"
  )
  [[ "${#ppo_overrides[@]}" -gt 0 ]] && ppo_args+=("${ppo_overrides[@]}")
  [[ "${#ppo_resume[@]}" -gt 0 ]] && ppo_args+=("${ppo_resume[@]}")
  torchrun --standalone --nproc_per_node="$GPUS" scripts/train_ppo.py "${ppo_args[@]}" \
    2>&1 | tee "$run_dir/logs/ppo.log"
fi

echo eval > "$run_dir/stage.txt"
python scripts/select_best_ppo.py --bc "$run_dir/bc_model.best.pt" \
  --ppo-glob "$run_dir/ppo_model.update-*.pt" --output "$run_dir/final_model.pt" \
  --report "$run_dir/evaluations/model_selection.json" --games "$EVAL_GAMES" \
  --seed "$EVAL_SEED" --wall-manifest "$WALL_MANIFEST" \
  --results-dir "$run_dir/evaluations/checkpoints" \
  2>&1 | tee "$run_dir/logs/model_selection.log"
echo complete > "$run_dir/stage.txt"
