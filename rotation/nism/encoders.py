import torch
from torch import nn


EMG_FEATURE_NAMES = (
    "MAV",
    "RMS",
    "ZCR",
    "WL",
    "Variance",
    "Amplitude_Sum",
    "Std",
    "Mean_PSD",
    "Peak_Frequency_Index",
    "Total_Spectral_Energy",
)


class IMUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        d_imu: int = 48,
        hidden_size: int = 225,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(2 * hidden_size, d_imu)
        self.norm = nn.LayerNorm(d_imu)

    def forward_sequence(self, x_imu: torch.Tensor) -> torch.Tensor:
        if x_imu.ndim != 3 or x_imu.shape[1] not in {10, 20, 30} or x_imu.shape[-1] != 12:
            raise ValueError(f"Expected IMU [B,T,12] with T in {{10,20,30}}, got {tuple(x_imu.shape)}")
        sequence, _ = self.lstm(x_imu)
        return self.norm(self.projection(sequence))

    def forward(self, x_imu: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x_imu)[:, -1]


class HandShapeLatentEncoder(nn.Module):
    """Independent IMU-derived hand-shape representation used by ContextEncoder."""

    def __init__(self, d_imu: int = 48, d_shape: int = 48, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_imu, d_shape),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_shape, d_shape),
            nn.Tanh(),
        )

    def forward(self, z_imu: torch.Tensor) -> torch.Tensor:
        return self.net(z_imu)


class EMGFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int = 10, d_emg: int = 48, d_hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_emg),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            features = features[:, -1]
        if features.ndim != 2 or features.shape[-1] != 10:
            raise ValueError(f"Expected 10-dimensional EMG features, got {tuple(features.shape)}")
        return self.net(features)
