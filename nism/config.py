from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NiSMConfig:
    imu_input_dim: int = 12
    num_fingers: int = 5
    d_imu: int = 48
    d_ctx: int = 48
    d_emg: int = 48
    imu_lstm_hidden: int = 225
    d_hidden: int = 128
    dropout: float = 0.2
    intent_label_mode: str = "sensor_context"
    sensor_context_proto_logits_mode: str = "none"
    use_sensor_context_bit_head: bool = False
    use_context_classifier: bool = True
    num_context_classes: int = 22
    hand_shape_dim: int = 63
    emg_input_mode: str = "feature10"
    output_mode: str = "point"
    force_sequence_len: int = 30
    emg_frame_window: int = 30
    emg_frame_hop: int = 1
    # Fixed fields of the NiSM constructor.
    sensor_context_hidden: int = 128
    ablation: str = "full"


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 100
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    log_every_batches: int = 200
    log_jsonl: str | None = None
    sparsity_warmup_epochs: int = 0
    sparsity_ramp_epochs: int = 0
    val_every_epochs: int = 1
    lr_scheduler: str = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    val_score_loss_weight: float = 0.0
