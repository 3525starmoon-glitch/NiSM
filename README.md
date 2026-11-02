# NiSM: Neuro-inspired Structured Model

Code for **"Trading Sensing for Structure: Sparse IMU-EMG Fingertip Force Estimation via Neuro-inspired Structured Modeling"** (NeurIPS 2026).

NiSM estimates five-finger contact pressure from a thumb-mounted IMU (30-frame window, 12 channels: 3D acceleration and a 3x3 rotation matrix) and a single-channel wrist EMG sensor (10-dimensional feature vector). An IMU-derived grasp-context feature conditions the kinematic pathway (conditional batch normalization, sparse feature-to-finger projection, finger-wise context mask), and an EMG-conditioned pathway provides finger-wise effort scaling.

```text
NiSM/
|-- nism/                      NiSM model, losses, training loop, cache loader
|-- baselines/                 CA-T and PiMForce (dense) reference models
|-- splits/                    user splits
|-- rotation/                  ring-rotation simulation (self-contained)
|-- train.py                   train NiSM
|-- train_baseline.py          train CA-T
|-- train_pimforce_dense.py    train PiMForce (dense)
|-- run_experiments.sh         Slurm launcher
|-- requirements.txt / environment.yml
`-- LICENSE
```

## Installation

Reference software: Python 3.9 and PyTorch 2.5.1.

```bash
conda env create -f environment.yml
conda activate nism
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

Hardware used in the paper: NiSM and CA-T ran on an Intel Xeon Gold 6138 CPU (2.00 GHz) with one NVIDIA GeForce RTX 4090; PiMForce (dense) and the rotation simulation ran on one NVIDIA A800 (80 GB). The Slurm launcher requests one generic GPU (`--gres=gpu:1`); set `NISM_GRES` and `NISM_PARTITION` for your cluster.

## Data

The training scripts read windowed feature caches. One cache root holds the split folders:

```text
cache/
|-- fixed_pimforce/          PiMForce fixed split
|-- fixed_self_collected/    self-collected fixed split (the NiSM_data/data folder)
`-- fold_User{N}/            PiMForce leave-one-subject-out (LOSO) folds
```

Each split folder contains `metadata.json` (split, window settings, normalization statistics of the training users), `{train,val,test}_index.json`, and `{train,val,test}_chunk_XXXX.npz` files with

| Key | Shape | Description |
| --- | --- | --- |
| `x_imu` | (N, T, 12) | thumb IMU: 3-axis acceleration + 3x3 rotation matrix |
| `emg_features` | (N, 10) | MAV, RMS, ZCR, WL, variance, amplitude sum, std, mean PSD, peak-frequency index, total spectral energy |
| `emg_sequence` | (N, T, 1) | single-channel EMG per frame (CA-T input) |
| `force` | (N, T, 5) | five-finger pressure; the last frame is the target |
| `hand_pose` | (N, 63) | hand-pose keypoints, training-time supervision |
| `trial_index`, `user_id` | (N,) | trial record in `metadata.json` and participant ID |

T is 30 by default (10 or 20 are also supported), with a stride of 2 frames. Pass the cache root, or a split folder itself, as `--precomputed-root`.

- **PiMForce** (public): 21 participants, 22 tasks. Fixed split: 17 training users (Session 1, half of the training windows) and User4, User9, User15, User20 for testing (all three sessions). `splits/` records the user splits.
- **Self-collected dataset**: 20 healthy right-handed participants (User1-User20, anonymous IDs), 20 gestures and grasp actions, one session per participant; User1-16 train, User17-20 test. It is available upon request for non-commercial research under CC BY-NC 4.0 terms, as the cache folder `NiSM_data/data`. Its hand-pose keypoints are also available upon request.

The PiMForce Dense reference uses the original dense modalities and has a separate memory-mapped input directory:

```text
dense_root/
  metadata.npz       # user_id, session_id, and optional sample_id arrays
  emg_dense.npy      # float32 [N, 8, raw_window_samples], raw_window_samples >= 480
  pose_voxels.npy    # float32 [N, 21, 48, 48, 48]
  pressure.npy       # float32 [N, 5]
```

All four arrays are row-aligned. One row represents one retrospective 30-frame window: `emg_dense` contains the eight synchronized EMG channels for that window, `pose_voxels` its aligned 3D hand-posture representation, and `pressure` the final-frame five-finger target. EMG and pressure min-max statistics are fitted on the training rows and saved in each checkpoint.

## Training

### NiSM

```bash
python train.py \
  --precomputed-root /path/to/cache \
  --output-dir ./outputs/nism_pimforce \
  --protocol fixed \
  --dataset pimforce \
  --device cuda
```

Use `--dataset self_collected` for the self-collected dataset, `--protocol loso --test-user UserN` for one LOSO fold, `--ablation {full,no_context_modulation,no_sparse_feedforward,no_emg_scaling}` for the model variants, and `--window-length T --num-frames T` for a cache with T-frame windows.

NiSM uses Adam with learning rate `1e-4`, weight decay `1e-5`, batch size 256, and 100 epochs. `ReduceLROnPlateau` (factor 0.5, patience 3) monitors training loss for the fixed split and validation loss for LOSO. The fixed split uses the final-epoch checkpoint; LOSO uses the checkpoint with the lowest validation loss on the closest non-test user. During training, the peak term compares the per-finger maxima of the 30-frame prediction and ground-truth pressure sequences; inference returns the final-frame pressure vector.

### CA-T

```bash
python train_baseline.py \
  --model ca_t \
  --dataset pimforce \
  --precomputed-root /path/to/cache \
  --output-dir ./outputs/ca_t_pimforce \
  --device cuda
```

CA-T uses learning rate `1e-3`, factor-0.5 decay every 30 epochs, batch size 256, and 100 epochs; LOSO (`--protocol loso --test-user UserN`) uses early-stopping patience 50 with the validation user. `--fine-tune-from CHECKPOINT` loads a CA-T checkpoint and trains with learning rate `1e-4` and a 100-epoch decay interval.

### PiMForce (dense)

```bash
python train_pimforce_dense.py \
  --data-root /path/to/dense_root \
  --output-dir ./outputs/pimforce_dense \
  --protocol fixed \
  --device cuda
```

Use `--protocol loso --test-user UserN` for one LOSO fold and `--train-user-count {17,14,11,8,5,2}` for the nested training subsets in `splits/pimforce_figure3.json`.

### Slurm

```bash
NISM_PROTOCOL=fixed \
NISM_DATASET=pimforce \
NISM_PRECOMPUTED_ROOT=/path/to/cache \
  bash run_experiments.sh
```

The ring-rotation simulation is described in [`rotation/README.md`](rotation/README.md).

## Outputs

Each training run writes `summaries/summary_<run_id>.json` to its output directory with NRMSE, MAE, R², and PCC on the test windows (`test_metrics`), for each test user (`per_user_metrics`), and as the mean and standard deviation across test users (`across_user_metrics`). The test predictions are saved in `predictions/`.

## License

The code is released under the [MIT License](LICENSE). The self-collected dataset is available upon request under CC BY-NC 4.0 terms.

## Citation

If you use this code, please cite our NeurIPS 2026 paper. The BibTeX entry will be added once the proceedings are published.
