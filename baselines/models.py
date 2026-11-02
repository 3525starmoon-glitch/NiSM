"""CA-T baseline architecture used in Table 1."""

from __future__ import annotations

import math

import torch
from torch import nn


BASELINE_NAMES = ("ca_t",)
PAPER_PARAMETER_LABELS = {"ca_t": "4.54M"}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 64):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class TimeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.input_projection = nn.Linear(d_model, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x)
        return self.encoder(self.position(x))


class CrossAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(
            self.query(query),
            self.key(key),
            self.value(key),
            need_weights=False,
        )
        return self.output(output)


class CAT(nn.Module):
    """Cross-attention model with pressure and hand-pose supervision."""

    def __init__(self):
        super().__init__()
        self.imu_encoder = TimeEncoder(12)
        self.emg_encoder = nn.LSTM(
            1,
            128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.position = PositionalEncoding(128)
        self.imu_to_emg = CrossAttention()
        self.emg_to_imu = CrossAttention()
        self.fusion = nn.Linear(256, 128)
        pressure_layer = nn.TransformerDecoderLayer(128, 4, batch_first=True)
        pose_layer = nn.TransformerDecoderLayer(128, 4, batch_first=True)
        self.pressure_decoder = nn.TransformerDecoder(pressure_layer, num_layers=2)
        self.pose_decoder = nn.TransformerDecoder(pose_layer, num_layers=2)
        self.pressure_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.ReLU(),
                )
                for _ in range(5)
            ]
        )
        self.pose_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 63),
        )
        self.imu_pressure_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.ReLU(),
                )
                for _ in range(5)
            ]
        )
        self.emg_pressure_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.ReLU(),
                )
                for _ in range(5)
            ]
        )
        self.imu_pose_head = nn.Linear(128, 63)
        self.emg_pose_head = nn.Linear(128, 63)
        self.query_projection = nn.Linear(128, 128)
        self.pose_enhancement = nn.Linear(128 + 5 * 3, 128)
        self.cross_pressure_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.ReLU(),
        )

    def forward(
        self,
        x_imu: torch.Tensor,
        emg_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        imu = self.imu_encoder(x_imu)
        emg, _ = self.emg_encoder(emg_sequence)
        emg = self.position(emg)
        imu_cross = self.imu_to_emg(imu, emg)
        emg_cross = self.emg_to_imu(emg, imu)
        fused = self.fusion(torch.cat([imu_cross, emg_cross], dim=-1))
        pose_features = self.pose_decoder(self.query_projection(fused), fused)
        pose_sequence = self.pose_head(pose_features)
        pose_joints = pose_sequence.view(
            pose_sequence.shape[0],
            pose_sequence.shape[1],
            21,
            3,
        )
        fingertip_pose = pose_joints[:, :, [4, 8, 12, 16, 20]].reshape(
            pose_sequence.shape[0],
            pose_sequence.shape[1],
            15,
        )
        pressure_input = self.pose_enhancement(
            torch.cat([fused, fingertip_pose], dim=-1)
        )
        pressure_features = self.pressure_decoder(pressure_input, fused)
        final_pressure_features = pressure_features[:, -1]
        return {
            "pressure": torch.cat(
                [head(final_pressure_features) for head in self.pressure_heads],
                dim=-1,
            ),
            "pose": pose_sequence[:, -1],
            "pressure_imu": torch.cat(
                [head(imu[:, -1]) for head in self.imu_pressure_heads],
                dim=-1,
            ),
            "pressure_emg": torch.cat(
                [head(emg[:, -1]) for head in self.emg_pressure_heads],
                dim=-1,
            ),
            "pose_imu": self.imu_pose_head(imu[:, -1]),
            "pose_emg": self.emg_pose_head(emg[:, -1]),
            "pressure_cross": self.cross_pressure_head(imu_cross[:, -1]),
        }


def validate_paper_parameter_count(model: nn.Module) -> None:
    count = sum(parameter.numel() for parameter in model.parameters())
    actual = f"{count / 1_000_000:.2f}M"
    expected = PAPER_PARAMETER_LABELS["ca_t"]
    if actual != expected:
        raise RuntimeError(
            "CA-T parameter count drifted from the paper: "
            f"expected {expected}, got {actual} ({count:,})"
        )


def build_baseline(name: str, sequence_length: int = 30) -> nn.Module:
    if name != "ca_t":
        raise ValueError(f"Only the CA-T baseline is available, got {name!r}")
    if sequence_length != 30:
        raise ValueError("CA-T requires a 30-frame sequence")
    model = CAT()
    validate_paper_parameter_count(model)
    return model
