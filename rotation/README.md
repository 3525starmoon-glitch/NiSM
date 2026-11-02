# Simulated ring rotation

This folder is self-contained (it has its own copy of the `nism` package) and contains the code of the ring-rotation simulation. Compared with the main code, this experiment uses:

- **Split.** All 20 self-collected participants; each trial is split in time into 60% training, 10% checkpoint selection, and 30% testing (`create_cache.py --split-mode within`).
- **Labels.** Pressure is zeroed by each participant's no-contact offset before the log1p compression.
- **EMG features.** The 10 EMG features are averaged over 100-sample sub-windows.
- **Training.** NiSM is trained for 100 epochs with batch size 256 on one GPU. The checkpoint with the lowest final-frame validation MSE is kept (`NISM_BEST_VAL=1`, `NISM_VAL_METRIC=mse`). The conditional batch-normalization running statistics are updated from the final window step (`NISM_BN_LASTSTEP=1`).
- **Rotation.** A mounting offset `R_off` is applied to the IMU inputs of the trained model at test time: `a -> R_off^T a` and `R -> R R_off`. Fixed offsets of 10/20/30/45 degrees about random axes and random per-trial offsets of up to 15/30 degrees are evaluated, each averaged over three random draws.

## Run

The features of this experiment are built from the raw recordings together with a mapping from anonymous IDs to raw folder names.

Single machine (one GPU):

```bash
cd rotation
NISM_ROOT="$PWD" \
NISM_RECORDINGS_ROOT=/path/to/raw_recordings \
NISM_ID_MAP=/path/to/id_map.json \
NISM_WORK_DIR="$PWD/rotation_exp" \
  bash run_rotation_job.sh
```

Slurm:

```bash
cd rotation
NISM_RECORDINGS_ROOT=/path/to/raw_recordings NISM_ID_MAP=/path/to/id_map.json bash submit_rotation.sh
```

`rotation_job.py` converts the raw recordings, builds the feature cache with this split, labels, and EMG features, trains NiSM, and evaluates all rotation settings. If such a cache already exists, pass it with `NISM_CACHE_ROOT` to skip the first two steps. Each step is skipped if its output already exists, so a resubmitted job resumes. The summary is printed at the end of the log, and per-user results are written to `rotation_exp/rotation_result*.json`.

Requirements: Python >= 3.9, PyTorch with CUDA, numpy, pandas >= 2.0, scipy.

## Regenerating the features

`convert_selfcollected.py` converts the raw recordings into per-trial CSV files, and `create_cache.py --split-mode within` builds the feature cache from them. The conversion takes the raw recordings and a JSON file mapping the anonymous IDs to the raw folder names (`--id-map`). Use `--tare` for the offset-zeroed labels.
