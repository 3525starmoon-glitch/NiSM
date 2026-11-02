#!/usr/bin/env bash
# Submit the simulated ring-rotation experiment (paper Appendix A.4.5) as one Slurm job.
#   NISM_RECORDINGS_ROOT=/path/to/raw_recordings NISM_ID_MAP=/path/to/id_map.json bash submit_rotation.sh
#   (or NISM_CACHE_ROOT=/path/to/existing_cache)
# Optional: NISM_PARTITION, NISM_GRES (default gpu:1), NISM_CONDA_ENV, NISM_WORK_DIR, NISM_EPOCHS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$HERE/run_rotation_job.sh"

export NISM_ROOT="$HERE"
[[ -n "${NISM_CACHE_ROOT:-}${NISM_RAW_ROOT:-}${NISM_RECORDINGS_ROOT:-}" ]] || { echo "[ERR] set NISM_RECORDINGS_ROOT and NISM_ID_MAP (or NISM_CACHE_ROOT)"; exit 1; }
export NISM_CACHE_ROOT="${NISM_CACHE_ROOT:-}" NISM_RAW_ROOT="${NISM_RAW_ROOT:-}" NISM_RECORDINGS_ROOT="${NISM_RECORDINGS_ROOT:-}" NISM_ID_MAP="${NISM_ID_MAP:-}"
export NISM_WORK_DIR="${NISM_WORK_DIR:-$HERE/rotation_exp}"
export NISM_CONDA_ENV="${NISM_CONDA_ENV:-}" NISM_PYTHON="${NISM_PYTHON:-python}" NISM_EPOCHS="${NISM_EPOCHS:-100}"
LOGDIR="$HERE/logs"; mkdir -p "$LOGDIR" "$NISM_WORK_DIR"

PARTITION_ARGS=()
[[ -n "${NISM_PARTITION:-}" ]] && PARTITION_ARGS=(-p "$NISM_PARTITION")
JOB_ID=$(sbatch --parsable -J nism_rotation "${PARTITION_ARGS[@]}" -N 1 -n 1 \
  --cpus-per-task="${NISM_CPUS:-8}" --mem="${NISM_MEM:-64G}" -t "${NISM_TIME:-12:00:00}" \
  --gres="${NISM_GRES:-gpu:1}" --chdir="$HERE" --export=ALL \
  -o "$LOGDIR/nism_rotation_%j.out" -e "$LOGDIR/nism_rotation_%j.err" \
  "$HERE/run_rotation_job.sh")
echo "Submitted job $JOB_ID"
echo "Watch progress:  tail -f $LOGDIR/nism_rotation_${JOB_ID}.out"
