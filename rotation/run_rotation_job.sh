#!/usr/bin/env bash
# Slurm job body (submitted by submit_rotation.sh). Do not sbatch this directly.
set -euo pipefail
HERE="${NISM_ROOT:?NISM_ROOT is required}"
cd "$HERE"
if [[ -n "${NISM_CONDA_ENV:-}" ]]; then
  CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$NISM_CONDA_ENV"
fi
PYTHON_BIN="${NISM_PYTHON:-python}"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=2026 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONWARNINGS=ignore::UserWarning
export NISM_PRELOAD="${NISM_PRELOAD:-1}"       # keep the whole cache in RAM (~0.3 GB)
export NISM_BEST_VAL="${NISM_BEST_VAL:-1}"     # keep the checkpoint with the lowest validation MSE
export NISM_VAL_METRIC="${NISM_VAL_METRIC:-mse}"   # select on final-frame validation MSE
export NISM_BN_LASTSTEP="${NISM_BN_LASTSTEP:-1}"   # CBN running stats from the final window step only
export NISM_TARE="${NISM_TARE:-1}"   # zero the no-contact pressure offset before log1p
echo "job ${SLURM_JOB_ID:-local}  python $(command -v "$PYTHON_BIN")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
exec "$PYTHON_BIN" -u "$HERE/rotation_job.py" \
  ${NISM_CACHE_ROOT:+--cache-root "$NISM_CACHE_ROOT"} ${NISM_RAW_ROOT:+--raw-root "$NISM_RAW_ROOT"} \
  ${NISM_RECORDINGS_ROOT:+--recordings-root "$NISM_RECORDINGS_ROOT"} ${NISM_ID_MAP:+--id-map "$NISM_ID_MAP"} --work-dir "${NISM_WORK_DIR:?}" --epochs "${NISM_EPOCHS:-100}" --gpus "${NISM_GPUS:-1}" --workers "${NISM_WORKERS:-0}" --split-mode "${NISM_SPLIT:-within}"
