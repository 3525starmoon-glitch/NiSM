from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import NiSMOutput


@dataclass
class NiSMLossWeights:
    lambda_force: float = 1.0
    lambda_peak: float = 2.0
    lambda_synergy_sparse: float = 1e-4
    lambda_mask_sparse: float = 1e-4
    lambda_ctx: float = 0.0
    lambda_shape: float = 0.1


class NiSMLoss(nn.Module):
    def __init__(self, weights: NiSMLossWeights | None = None, peak_threshold: float = 0.7, finger_weights=None):
        super().__init__()
        del finger_weights
        self.weights = weights or NiSMLossWeights()
        self.peak_threshold = 0.7 if peak_threshold is None else float(peak_threshold)
        if self.peak_threshold != 0.7:
            raise ValueError("Peak threshold is fixed at 0.7")

    def set_sparsity_scale(self, scale: float) -> None:
        del scale

    def forward(
        self,
        output: NiSMOutput,
        target: torch.Tensor,
        context_label: torch.Tensor | None = None,
        hand_shape_target: torch.Tensor | None = None,
        **unused,
    ) -> dict[str, torch.Tensor]:
        del unused
        prediction = output.p_hat
        if target.ndim != 3 or target.shape[1] not in {10, 20, 30} or target.shape[2] != 5:
            raise ValueError(
                f"Peak loss requires [B,T,5] with T in {{10,20,30}}, got {tuple(target.shape)}"
            )
        final_target = target[:, -1]
        error2 = (prediction - final_target).square()
        loss_force = error2.mean()
        if output.p_window is None:
            raise ValueError("Peak loss requires the training-time pressure sequence from NiSM")
        if output.p_window.shape != target.shape:
            raise ValueError(
                f"Predicted/target pressure-window mismatch: {tuple(output.p_window.shape)} vs {tuple(target.shape)}"
            )
        pred_peak = output.p_window.amax(dim=1)
        true_peak = target.amax(dim=1)
        peak_mask = true_peak > self.peak_threshold
        peak_error = (pred_peak - true_peak).square()
        loss_peak = peak_error[peak_mask].sum() if peak_mask.any() else peak_error.new_tensor(0.0)
        loss_sparse = output.synergy_matrix.abs().sum()
        loss_mask_sparse = output.m_ctx.abs().mean()
        loss_ctx = (
            F.cross_entropy(output.context_logits, context_label.long())
            if context_label is not None
            else output.context_logits.new_tensor(0.0)
        )
        if hand_shape_target is None:
            loss_shape = output.hand_shape_hat.new_tensor(0.0)
        else:
            hand_shape_target = hand_shape_target.reshape(hand_shape_target.shape[0], -1)
            if hand_shape_target.shape != output.hand_shape_hat.shape:
                raise ValueError(
                    f"hand-shape prediction/target mismatch: {tuple(output.hand_shape_hat.shape)} vs "
                    f"{tuple(hand_shape_target.shape)}"
                )
            loss_shape = F.mse_loss(output.hand_shape_hat, hand_shape_target)
        total = (
            self.weights.lambda_force * loss_force
            + self.weights.lambda_peak * loss_peak
            + self.weights.lambda_synergy_sparse * loss_sparse
            + self.weights.lambda_mask_sparse * loss_mask_sparse
            + self.weights.lambda_ctx * loss_ctx
            + self.weights.lambda_shape * loss_shape
        )
        return {
            "loss": total,
            "loss_force": loss_force,
            "loss_peak": loss_peak,
            "loss_synergy_sparse": loss_sparse,
            "loss_mask_sparse": loss_mask_sparse,
            "loss_ctx": loss_ctx,
            "loss_shape": loss_shape,
        }
