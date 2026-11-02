"""PiMForce Dense reference model used for the PiMForce comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use stored statistics for a final singleton batch without dropping it."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training and values.shape[0] == 1:
            return F.batch_norm(
                values,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(values)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = downsample

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self.relu(self.bn1(self.conv1(values)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(values)
        return self.relu(output + residual)


class PoseResNet(nn.Module):
    """3D ResNet posture branch with a [3, 4, 6, 3] block layout."""

    def __init__(self, input_channels: int = 21) -> None:
        super().__init__()
        self.input_channels = 64
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 3)
        self.layer2 = self._layer(128, 4, stride=2)
        self.layer3 = self._layer(256, 6, stride=2)
        self.layer4 = self._layer(512, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Classifier of the posture backbone; the fusion path uses the 512-D
        # feature before it.
        self.fc = nn.Linear(512, 400)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.input_channels != channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.input_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(channels),
            )
        layers: list[nn.Module] = [
            BasicBlock3D(self.input_channels, channels, stride, downsample)
        ]
        self.input_channels = channels
        layers.extend(BasicBlock3D(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        values = self.relu(self.bn1(self.conv1(values)))
        values = self.maxpool(values)
        values = self.layer4(self.layer3(self.layer2(self.layer1(values))))
        return self.pool(values).flatten(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(values))


class DenseEMGNet(nn.Module):
    """Eight-channel STFT encoder-decoder producing a 512-D embedding."""

    def __init__(
        self,
        force_levels: int = 4,
        output_frames: int = 32,
        stft_frequencies: int = 64,
        stft_window: int = 256,
        stft_hop: int = 32,
    ) -> None:
        super().__init__()
        if output_frames % 4:
            raise ValueError("output_frames must be divisible by four")
        self.channels = 8
        self.output_frames = output_frames
        self.frequencies = stft_frequencies
        self.window = stft_window
        self.hop = stft_hop
        self.register_buffer("hann_window", torch.hann_window(stft_window), persistent=False)
        self.encoder = nn.Sequential(
            self._encoder_block(8, 32),
            self._encoder_block(32, 128),
            self._encoder_block(128, 256),
        )
        self.decoder = nn.Sequential(
            self._decoder_block(256, 256, (output_frames // 4, 1)),
            self._decoder_block(256, 128, (output_frames // 2, 1)),
            self._decoder_block(128, 32, (output_frames, 1)),
        )
        self.linear = nn.Linear(32, force_levels * 5)
        self.output = nn.Linear(32 * output_frames, 512)

    @staticmethod
    def _encoder_block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 4), stride=(2, 4)),
        )

    @staticmethod
    def _decoder_block(
        input_channels: int, output_channels: int, size: tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Upsample(size=size, mode="bilinear", align_corners=False),
        )

    def _spectrogram(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError("emg_dense must have shape [batch, 8, samples]")
        minimum = self.window + 7 * self.hop  # at least 8 STFT frames for the encoder pooling
        if values.shape[-1] < minimum:
            raise ValueError(f"EMG windows must contain at least {minimum} samples")
        batch_size = values.shape[0]
        spectrum = torch.stft(
            values.reshape(batch_size * self.channels, -1),
            n_fft=self.window,
            hop_length=self.hop,
            win_length=self.window,
            window=self.hann_window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        spectrum = spectrum.reshape(
            batch_size, self.channels, spectrum.shape[-2], spectrum.shape[-1]
        )
        return spectrum.transpose(2, 3)[..., : self.frequencies]

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        values = self.decoder(self.encoder(self._spectrogram(values)))
        return self.output(values.flatten(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.decoder(self.encoder(self._spectrogram(values)))
        features = features.squeeze(-1).transpose(1, 2)
        return self.linear(features).transpose(1, 2)


class PiMForceDense(nn.Module):
    """Dense multi-channel EMG and 3D hand-posture pressure estimator."""

    def __init__(
        self,
        force_levels: int = 4,
        output_frames: int = 32,
        stft_frequencies: int = 64,
        stft_window: int = 256,
        stft_hop: int = 32,
    ) -> None:
        super().__init__()
        self.force_levels = force_levels
        self.output_frames = output_frames
        self.pose_encoder = PoseResNet(input_channels=21)
        self.emg_encoder = DenseEMGNet(
            force_levels=force_levels,
            output_frames=output_frames,
            stft_frequencies=stft_frequencies,
            stft_window=stft_window,
            stft_hop=stft_hop,
        )
        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            SafeBatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            SafeBatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, force_levels * 5 * output_frames),
            SafeBatchNorm1d(force_levels * 5 * output_frames),
            nn.ReLU(inplace=True),
        )
        self.register_buffer(
            "level_values",
            torch.linspace(0.0, 1.0, force_levels).reshape(1, force_levels, 1, 1),
            persistent=False,
        )

    def forward(
        self, pose_voxels: torch.Tensor, emg_dense: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pose = self.pose_encoder.forward_features(pose_voxels)
        emg = self.emg_encoder.forward_features(emg_dense)
        logits = self.fusion(torch.cat([pose, emg], dim=-1)).reshape(
            pose.shape[0], self.force_levels, 5, self.output_frames
        )
        probabilities = torch.softmax(logits, dim=1)
        sequence = (probabilities * self.level_values).sum(dim=1).transpose(1, 2)
        return {
            "pressure": sequence[:, -1],
            "pressure_sequence": sequence,
            "force_logits": logits,
        }


def build_pimforce_dense() -> PiMForceDense:
    model = PiMForceDense()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if round(parameter_count / 1_000_000, 2) != 66.39:
        raise RuntimeError(
            f"PiMForce Dense must round to the paper's 66.39M parameters, got {parameter_count:,}"
        )
    return model
