#!/usr/bin/env bash
set -euo pipefail

# Submit fixed-split, LOSO, baseline, ablation, or training-size experiments.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  HERE="${NISM_ROOT:?NISM_ROOT is required}"
  EXPERIMENT="${NISM_EXPERIMENT:-standard}"
  PROTOCOL="${NISM_PROTOCOL:-loso}"
  DATASET="${NISM_DATASET:-pimforce}"
  PYTHON_BIN="${NISM_PYTHON:-python}"
  NPROC="${NISM_TORCHRUN_NPROC_PER_NODE:-1}"
  PRECOMPUTED_ROOT="${NISM_PRECOMPUTED_ROOT:-}"
  ABLATION="${NISM_ABLATION:-full}"
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/${PROTOCOL}_${DATASET}_${ABLATION}}"
  if [[ "$EXPERIMENT" == "figure3" ]]; then
    FIGURE3_COUNTS=(17 14 11 8 5 2)
    FIGURE3_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
    TRAIN_USER_COUNT="${FIGURE3_COUNTS[$FIGURE3_INDEX]:?Invalid Figure 3 array index}"
    FIGURE3_MODEL="nism"
    if [[ "$SLURM_ARRAY_TASK_ID" -ge 6 ]]; then
      FIGURE3_MODEL="pimforce_dense"
    fi
    PROTOCOL="fixed"
    DATASET="pimforce"
    ABLATION="full"
    PRECOMPUTED_ROOT="$PRECOMPUTED_ROOT/train_users_${TRAIN_USER_COUNT}"
    OUTPUT_DIR="$OUTPUT_DIR/train_users_${TRAIN_USER_COUNT}/$FIGURE3_MODEL"
  elif [[ "$EXPERIMENT" == "table2" ]]; then
    TABLE2_ABLATIONS=(full no_context_modulation no_sparse_feedforward no_emg_scaling)
    ABLATION="${TABLE2_ABLATIONS[$SLURM_ARRAY_TASK_ID]:?Invalid Table 2 array index}"
    PROTOCOL="fixed"
    DATASET="pimforce"
    OUTPUT_DIR="$OUTPUT_DIR/$ABLATION"
  elif [[ "$EXPERIMENT" == "table1" ]]; then
    TABLE1_MODELS=(nism ca_t pimforce_dense nism ca_t)
    TABLE1_DATASETS=(pimforce pimforce pimforce self_collected self_collected)
    TABLE1_MODEL="${TABLE1_MODELS[$SLURM_ARRAY_TASK_ID]:?Invalid Table 1 array index}"
    DATASET="${TABLE1_DATASETS[$SLURM_ARRAY_TASK_ID]:?Invalid Table 1 array index}"
    PROTOCOL="fixed"
    ABLATION="full"
    OUTPUT_DIR="$OUTPUT_DIR/$DATASET/$TABLE1_MODEL"
  elif [[ "$EXPERIMENT" == "ca_t_loso" ]]; then
    PROTOCOL="loso"
    DATASET="pimforce"
    ABLATION="full"
    CA_T_TEST_USER="User$((SLURM_ARRAY_TASK_ID + 1))"
    OUTPUT_DIR="$OUTPUT_DIR/$CA_T_TEST_USER"
  elif [[ "$EXPERIMENT" == "pimforce_dense_loso" ]]; then
    PROTOCOL="loso"
    DATASET="pimforce"
    ABLATION="full"
    DENSE_TEST_USER="User$((SLURM_ARRAY_TASK_ID + 1))"
    OUTPUT_DIR="$OUTPUT_DIR/$DENSE_TEST_USER"
  fi
  LOG_JSONL="$HERE/logs/nism_${PROTOCOL}_${DATASET}_${ABLATION}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}.jsonl"

  mkdir -p "$HERE/logs" "$OUTPUT_DIR"
  export PYTHONHASHSEED="${NISM_SEED:-2026}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

  if [[ "$EXPERIMENT" == "ca_t_loso" ]]; then
    exec "$PYTHON_BIN" "$HERE/train_baseline.py" \
      --model ca_t \
      --protocol loso \
      --dataset pimforce \
      --test-user "$CA_T_TEST_USER" \
      --precomputed-root "$PRECOMPUTED_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --num-epochs "${NISM_EPOCHS:-100}" \
      --batch-size "${NISM_BATCH_SIZE:-256}" \
      --learning-rate "${NISM_LEARNING_RATE:-1e-3}" \
      --weight-decay "${NISM_WEIGHT_DECAY:-1e-5}" \
      --lr-scheduler-patience "${NISM_LR_SCHEDULER_PATIENCE:-30}" \
      --lr-scheduler-factor "${NISM_LR_SCHEDULER_FACTOR:-0.5}" \
      --early-stopping-patience "${NISM_EARLY_STOPPING_PATIENCE:-50}" \
      --num-workers "${NISM_NUM_WORKERS:-0}" \
      --device "${NISM_DEVICE:-cuda}" \
      --seed "${NISM_SEED:-2026}"
  fi

  if [[ "$EXPERIMENT" == "pimforce_dense_loso" ]]; then
    exec "$PYTHON_BIN" "$HERE/train_pimforce_dense.py" \
      --protocol loso \
      --test-user "$DENSE_TEST_USER" \
      --data-root "${NISM_PIMFORCE_DENSE_ROOT:?NISM_PIMFORCE_DENSE_ROOT is required}" \
      --output-dir "$OUTPUT_DIR" \
      --num-epochs "${NISM_DENSE_EPOCHS:-100}" \
      --batch-size "${NISM_DENSE_BATCH_SIZE:-64}" \
      --learning-rate "${NISM_DENSE_LEARNING_RATE:-1e-4}" \
      --scheduler-patience "${NISM_DENSE_SCHEDULER_PATIENCE:-3}" \
      --scheduler-factor "${NISM_DENSE_SCHEDULER_FACTOR:-0.2}" \
      --early-stopping-patience "${NISM_DENSE_EARLY_STOPPING_PATIENCE:-15}" \
      --num-workers "${NISM_NUM_WORKERS:-0}" \
      --device "${NISM_DEVICE:-cuda}" \
      --seed "${NISM_DENSE_SEED:-123}"
  fi

  if [[ "$EXPERIMENT" == "table1" && "${TABLE1_MODEL:-}" == "pimforce_dense" ]] || \
     [[ "$EXPERIMENT" == "figure3" && "${FIGURE3_MODEL:-}" == "pimforce_dense" ]]; then
    exec "$PYTHON_BIN" "$HERE/train_pimforce_dense.py" \
      --protocol fixed \
      --train-user-count "${TRAIN_USER_COUNT:-17}" \
      --data-root "${NISM_PIMFORCE_DENSE_ROOT:?NISM_PIMFORCE_DENSE_ROOT is required}" \
      --output-dir "$OUTPUT_DIR" \
      --num-epochs "${NISM_DENSE_EPOCHS:-100}" \
      --batch-size "${NISM_DENSE_BATCH_SIZE:-64}" \
      --learning-rate "${NISM_DENSE_LEARNING_RATE:-1e-4}" \
      --scheduler-patience "${NISM_DENSE_SCHEDULER_PATIENCE:-3}" \
      --scheduler-factor "${NISM_DENSE_SCHEDULER_FACTOR:-0.2}" \
      --num-workers "${NISM_NUM_WORKERS:-0}" \
      --device "${NISM_DEVICE:-cuda}" \
      --seed "${NISM_DENSE_SEED:-123}"
  fi

  if [[ "$EXPERIMENT" == "table1" && "${TABLE1_MODEL:-}" == "ca_t" ]]; then
    exec "$PYTHON_BIN" "$HERE/train_baseline.py" \
      --model ca_t \
      --protocol fixed \
      --dataset "$DATASET" \
      --precomputed-root "$PRECOMPUTED_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --num-epochs "${NISM_EPOCHS:-100}" \
      --batch-size "${NISM_BATCH_SIZE:-256}" \
      --learning-rate "${NISM_LEARNING_RATE:-1e-3}" \
      --weight-decay "${NISM_WEIGHT_DECAY:-1e-5}" \
      --lr-scheduler-patience "${NISM_LR_SCHEDULER_PATIENCE:-30}" \
      --lr-scheduler-factor "${NISM_LR_SCHEDULER_FACTOR:-0.5}" \
      --early-stopping-patience "${NISM_EARLY_STOPPING_PATIENCE:-50}" \
      --num-workers "${NISM_NUM_WORKERS:-0}" \
      --device "${NISM_DEVICE:-cuda}" \
      --seed "${NISM_SEED:-2026}"
  fi

  ARGS=(
    --protocol "$PROTOCOL"
    --dataset "$DATASET"
    --ablation "$ABLATION"
    --model-name "${NISM_MODEL_NAME:-NiSM}"
    --precomputed-root "$PRECOMPUTED_ROOT"
    --output-dir "$OUTPUT_DIR"
    --log-jsonl "$LOG_JSONL"
    --num-epochs "${NISM_EPOCHS:-100}"
    --batch-size "${NISM_BATCH_SIZE:-256}"
    --learning-rate "${NISM_LEARNING_RATE:-1e-4}"
    --weight-decay "${NISM_WEIGHT_DECAY:-1e-5}"
    --device "${NISM_DEVICE:-cuda}"
    --seed "${NISM_SEED:-2026}"
    --num-workers "${NISM_NUM_WORKERS:-0}"
    --window-length "${NISM_WINDOW_LENGTH:-30}"
    --hop-length 2
    --num-frames "${NISM_WINDOW_LENGTH:-30}"
    --target-mode last
    --output-mode point
    --emg-input-mode feature10
    --intent-label-mode sensor_context
    --lambda-ctx "${NISM_LAMBDA_CTX:-0.05}"
    --lambda-shape "${NISM_LAMBDA_SHAPE:-0.1}"
    --lambda-peak 2
    --lambda-synergy-sparse 0.0001
    --lambda-mask-sparse 0.0001
    --optimizer adam
    --lr-scheduler reduce_on_plateau
    --lr-scheduler-patience 3
    --lr-scheduler-factor 0.5
    --val-user-count 1
    --validate-precomputed-metadata
    --deterministic
  )
  if [[ "$PROTOCOL" == "loso" ]]; then
    ARGS+=(--test-user "User$((SLURM_ARRAY_TASK_ID + 1))")
  fi

  if [[ "$NPROC" -gt 1 ]]; then
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$HERE/train.py" "${ARGS[@]}"
  fi
  exec "$PYTHON_BIN" "$HERE/train.py" "${ARGS[@]}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_DIR="$HERE/logs"
PROTOCOL="${NISM_PROTOCOL:-loso}"
DATASET="${NISM_DATASET:-pimforce}"
ABLATION="${NISM_ABLATION:-full}"
EXPERIMENT="${NISM_EXPERIMENT:-standard}"
if [[ "$EXPERIMENT" != "pimforce_dense_loso" ]]; then
  : "${NISM_PRECOMPUTED_ROOT:?Set NISM_PRECOMPUTED_ROOT to the precomputed dataset}"
fi
if [[ "$EXPERIMENT" == "table1" || "$EXPERIMENT" == "figure3" || "$EXPERIMENT" == "pimforce_dense_loso" ]]; then
  : "${NISM_PIMFORCE_DENSE_ROOT:?Set NISM_PIMFORCE_DENSE_ROOT to the prepared PiMForce Dense arrays}"
fi
ARRAY_DEFAULT="0-0"
if [[ "$EXPERIMENT" == "figure3" ]]; then
  PROTOCOL="fixed"
  DATASET="pimforce"
  ABLATION="full"
  ARRAY_DEFAULT="0-11"
elif [[ "$EXPERIMENT" == "table2" ]]; then
  PROTOCOL="fixed"
  DATASET="pimforce"
  ARRAY_DEFAULT="0-3"
elif [[ "$EXPERIMENT" == "table1" ]]; then
  PROTOCOL="fixed"
  ARRAY_DEFAULT="0-4"
elif [[ "$EXPERIMENT" == "ca_t_loso" ]]; then
  PROTOCOL="loso"
  DATASET="pimforce"
  ABLATION="full"
  ARRAY_DEFAULT="0-20"
elif [[ "$EXPERIMENT" == "pimforce_dense_loso" ]]; then
  PROTOCOL="loso"
  DATASET="pimforce"
  ABLATION="full"
  ARRAY_DEFAULT="0-20"
elif [[ "$PROTOCOL" == "loso" ]]; then
  ARRAY_DEFAULT="0-20"
fi
if [[ "$EXPERIMENT" == "figure3" ]]; then
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/figure3}"
elif [[ "$EXPERIMENT" == "table2" ]]; then
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/table2}"
elif [[ "$EXPERIMENT" == "table1" ]]; then
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/table1}"
elif [[ "$EXPERIMENT" == "ca_t_loso" ]]; then
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/ca_t_loso}"
elif [[ "$EXPERIMENT" == "pimforce_dense_loso" ]]; then
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/pimforce_dense_loso}"
else
  OUTPUT_DIR="${NISM_OUTPUT_DIR:-$HERE/outputs/${PROTOCOL}_${DATASET}_${ABLATION}}"
fi
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export NISM_ROOT="$HERE"
export NISM_OUTPUT_DIR="$OUTPUT_DIR"
export NISM_TORCHRUN_NPROC_PER_NODE="${NISM_TORCHRUN_NPROC_PER_NODE:-1}"

JOB_ID=$(sbatch --parsable \
  -J "${NISM_JOB_NAME:-nism_${PROTOCOL}_${DATASET}_${ABLATION}}" \
  -p "${NISM_PARTITION:-gpu}" \
  --array="${NISM_ARRAY:-$ARRAY_DEFAULT}" \
  -N 1 -n 1 \
  --cpus-per-task="${NISM_CPUS:-4}" \
  --mem="${NISM_MEM:-16G}" \
  -t "${NISM_TIME:-8:00:00}" \
  --gres="${NISM_GRES:-gpu:1}" \
  --chdir="$HERE" \
  -o "$LOG_DIR/nism_${PROTOCOL}_${DATASET}_${ABLATION}_%A_%a.out" \
  -e "$LOG_DIR/nism_${PROTOCOL}_${DATASET}_${ABLATION}_%A_%a.err" \
  "$HERE/run_experiments.sh")

echo "Submitted NiSM protocol=$PROTOCOL dataset=$DATASET ablation=$ABLATION job: $JOB_ID"
echo "Outputs: $OUTPUT_DIR"
