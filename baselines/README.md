# Baselines

## CA-T

The fixed-split runs use the NiSM user splits, 30-frame input windows, 12 IMU channels and one synchronized EMG channel, normalization statistics of the training users, and the same NRMSE, MAE, R2, and PCC as NiSM.

CA-T defaults to Adam with learning rate `1e-3`, factor-0.5 decay every 30 epochs, batch size 256, and 100 epochs. Passing `--fine-tune-from CHECKPOINT` loads a CA-T checkpoint and switches to learning rate `1e-4` with a 100-epoch decay interval. Fixed-split training runs for 100 epochs and evaluates the final-epoch checkpoint. LOSO uses early-stopping patience 50 with the validation user.

CA-T additionally uses the normalized 63-dimensional hand-pose target during training.

CA-T has 4.54M parameters.

CA-T also supports the 21-fold PiMForce LOSO protocol. Every fold uses one held-out test user and a separate closest-ID validation user. The minimum-validation-loss checkpoint is evaluated on the test user. Run it through `NISM_EXPERIMENT=ca_t_loso bash run_experiments.sh` or `train_baseline.py --model ca_t --protocol loso --test-user UserN`.

## PiMForce Dense

PiMForce Dense runs on the public PiMForce dataset and takes 8-channel EMG and 21-channel 3D hand-posture heatmaps. Its posture branch (a 3D ResNet with a [3, 4, 6, 3] block layout) and EMG branch each produce a 512-dimensional embedding. A three-layer MLP maps their 1024-dimensional concatenation to the five-finger structured pressure output. With four force levels, the model has 66.39M parameters.

The paper settings are Adam, learning rate `1e-4`, batch size 64, MAE loss, at most 100 epochs, factor-0.2 plateau scheduling, and early-stopping patience 15 with the LOSO validation user. Fixed-split runs train for 100 epochs, schedule on training MAE, and evaluate the final-epoch checkpoint. LOSO uses the closest non-test user for validation and selects the minimum-validation-RMSE checkpoint.

Run fixed-split and LOSO training through `train_pimforce_dense.py`.
