import torch
from torch import nn


class SensorContextEncoder(nn.Module):
    """Continuous context inferred exclusively from the IMU hand-shape latent."""

    def __init__(self, d_imu: int = 48, d_ctx: int = 48, d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_imu),
            nn.Linear(d_imu, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_ctx),
        )

    def forward(self, z_imu: torch.Tensor) -> torch.Tensor:
        return self.net(z_imu)


class ContextConditionedModulation(nn.Module):
    """Conditional BatchNorm with context-predicted gamma and beta."""

    def __init__(self, d_imu: int = 48, d_ctx: int = 48):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_imu)
        self.gamma = nn.Linear(d_ctx, d_imu)
        self.beta = nn.Linear(d_ctx, d_imu)

    def forward(self, z_imu: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        if z_imu.ndim != 2 or z_ctx.ndim != 2:
            raise ValueError(
                f"Conditional BatchNorm expects [B,D] features, got {tuple(z_imu.shape)} and {tuple(z_ctx.shape)}"
            )
        normalized = self.norm(z_imu)
        return normalized * self.gamma(z_ctx) + self.beta(z_ctx)


class SparseFingerProjection(nn.Module):
    def __init__(self, d_in: int = 48, num_fingers: int = 5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, num_fingers))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight


class DenseFingerProjection(nn.Module):
    """Unstructured dense replacement used by the feedforward ablation."""

    def __init__(self, d_in: int = 48, num_fingers: int = 5):
        super().__init__()
        self.linear = nn.Linear(d_in, num_fingers)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class ContextMaskHead(nn.Module):
    def __init__(self, d_ctx: int = 48, num_fingers: int = 5):
        super().__init__()
        self.linear = nn.Linear(d_ctx, num_fingers)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(z_ctx))


class EffortScalingHead(nn.Module):
    def __init__(self, d_emg: int = 48, num_fingers: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emg, d_emg),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emg, num_fingers),
            nn.Sigmoid(),
        )

    def forward(self, z_emg: torch.Tensor) -> torch.Tensor:
        return self.net(z_emg)


class AuxiliaryContextClassifier(nn.Module):
    def __init__(self, d_ctx: int = 48, num_classes: int = 22):
        super().__init__()
        self.linear = nn.Linear(d_ctx, num_classes)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.linear(z_ctx)
