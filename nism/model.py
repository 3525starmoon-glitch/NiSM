from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NiSMConfig
from .encoders import EMGFeatureEncoder, HandShapeLatentEncoder, IMUEncoder
from .modules import AuxiliaryContextClassifier, ContextConditionedModulation, ContextMaskHead, DenseFingerProjection, EffortScalingHead, SensorContextEncoder, SparseFingerProjection


@dataclass
class NiSMOutput:
    p_hat: torch.Tensor
    p_window: torch.Tensor | None
    z_imu: torch.Tensor
    z_ctx: torch.Tensor
    h_ctx: torch.Tensor
    p_base: torch.Tensor
    m_ctx: torch.Tensor
    z_emg: torch.Tensor
    s_emg: torch.Tensor
    synergy_matrix: torch.Tensor
    context_logits: torch.Tensor
    hand_shape_hat: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return self.__dict__.copy()


class NiSM(nn.Module):
    """Structured point model with optional component ablations."""

    def __init__(self, config: NiSMConfig | None = None):
        super().__init__()
        self.config = config or NiSMConfig()
        cfg = self.config
        fixed = {
            "d_imu": 48,
            "d_ctx": 48,
            "d_emg": 48,
            "imu_lstm_hidden": 225,
            "hand_shape_dim": 63,
            "intent_label_mode": "sensor_context",
            "sensor_context_proto_logits_mode": "none",
            "emg_input_mode": "feature10",
            "output_mode": "point",
        }
        for name, expected in fixed.items():
            if getattr(cfg, name) != expected:
                raise ValueError(f"NiSM requires {name}={expected!r}")
        if cfg.use_sensor_context_bit_head:
            raise ValueError("Bit supervision is not supported")
        if cfg.ablation not in {"full", "no_context_modulation", "no_sparse_feedforward", "no_emg_scaling"}:
            raise ValueError(f"Unsupported ablation: {cfg.ablation}")

        self.imu_encoder = IMUEncoder(
            cfg.imu_input_dim,
            cfg.d_imu,
            hidden_size=cfg.imu_lstm_hidden,
            dropout=cfg.dropout,
        )
        self.hand_shape_encoder = HandShapeLatentEncoder(cfg.d_imu, cfg.d_ctx, cfg.dropout)
        self.emg_encoder = EMGFeatureEncoder(10, cfg.d_emg, dropout=cfg.dropout)
        self.sensor_context_encoder = SensorContextEncoder(cfg.d_imu, cfg.d_ctx, cfg.sensor_context_hidden, cfg.dropout)
        self.context_modulation = ContextConditionedModulation(cfg.d_imu, cfg.d_ctx)
        self.coordination = (
            DenseFingerProjection(cfg.d_imu, cfg.num_fingers)
            if cfg.ablation == "no_sparse_feedforward"
            else SparseFingerProjection(cfg.d_imu, cfg.num_fingers)
        )
        self.context_mask_head = ContextMaskHead(cfg.d_ctx, cfg.num_fingers)
        self.effort_scaling_head = EffortScalingHead(cfg.d_emg, cfg.num_fingers, cfg.dropout)
        self.context_classifier = AuxiliaryContextClassifier(cfg.d_ctx, cfg.num_context_classes)
        self.hand_shape_head = nn.Linear(cfg.d_ctx, cfg.hand_shape_dim)

    @property
    def synergy_matrix(self) -> torch.Tensor:
        return self.coordination.weight

    def forward(
        self,
        x_imu: torch.Tensor,
        emg_features: torch.Tensor | None = None,
        context_override: torch.Tensor | None = None,
        return_pressure_window: bool = False,
        return_dict: bool = False,
    ) -> NiSMOutput | dict[str, torch.Tensor]:
        if emg_features is None:
            raise ValueError("NiSM requires precomputed 10-dimensional EMG features")
        z_emg = self.emg_encoder(emg_features)
        s_emg = self.effort_scaling_head(z_emg)
        if self.config.ablation == "no_emg_scaling":
            s_emg = torch.ones_like(s_emg)

        def structured_prediction(z_imu_step: torch.Tensor, override: torch.Tensor | None = None):
            hand_shape_step = self.hand_shape_encoder(z_imu_step)
            inferred_context = self.sensor_context_encoder(hand_shape_step)
            if override is None:
                z_ctx_step = inferred_context
            else:
                if override.shape != inferred_context.shape:
                    raise ValueError(
                        f"context_override must have shape {tuple(inferred_context.shape)}, "
                        f"got {tuple(override.shape)}"
                    )
                z_ctx_step = override.to(device=inferred_context.device, dtype=inferred_context.dtype)
            h_ctx_step = (
                z_imu_step
                if self.config.ablation == "no_context_modulation"
                else self.context_modulation(z_imu_step, z_ctx_step)
            )
            p_base_step = self.coordination(h_ctx_step)
            m_ctx_step = self.context_mask_head(z_ctx_step)
            p_step = torch.relu(p_base_step * m_ctx_step * s_emg)
            return p_step, hand_shape_step, z_ctx_step, h_ctx_step, p_base_step, m_ctx_step

        z_imu_sequence = self.imu_encoder.forward_sequence(x_imu)
        p_window = None
        if return_pressure_window:
            if context_override is not None:
                raise ValueError("context_override is only supported for point inference")
            step_outputs = [structured_prediction(z_imu_sequence[:, index]) for index in range(x_imu.shape[1])]
            p_window = torch.stack([values[0] for values in step_outputs], dim=1)
            p_hat, hand_shape, z_ctx, h_ctx, p_base, m_ctx = step_outputs[-1]
        else:
            p_hat, hand_shape, z_ctx, h_ctx, p_base, m_ctx = structured_prediction(
                z_imu_sequence[:, -1], context_override
            )
        z_imu = z_imu_sequence[:, -1]
        hand_shape_hat = self.hand_shape_head(hand_shape)
        context_logits = self.context_classifier(z_ctx)
        output = NiSMOutput(
            p_hat, p_window, z_imu, z_ctx, h_ctx, p_base, m_ctx,
            z_emg, s_emg, self.coordination.weight, context_logits, hand_shape_hat,
        )
        return output.as_dict() if return_dict else output
