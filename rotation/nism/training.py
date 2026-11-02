from __future__ import annotations

import os

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .losses import NiSMLoss
from .model import NiSM


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_metrics(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed():
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(keys, tensor.tolist()))


def save_model_checkpoint(path: str | Path, model: nn.Module, epoch: int, monitor: str, monitor_value: float, extra_payload: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = unwrap_model(model)
    torch.save({
        "model_state_dict": core.state_dict(),
        "model_config": core.config.__dict__,
        "epoch": epoch,
        "monitor": monitor,
        "monitor_value": monitor_value,
        **(extra_payload or {}),
    }, path)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if torch.is_floating_point(value):
                value = torch.nan_to_num(value)
        result[key] = value
    return result


def forward_batch(model: nn.Module, batch: dict[str, Any]):
    return model(
        batch["x_imu"],
        emg_features=batch.get("emg_features"),
        return_pressure_window=True,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: NiSMLoss, device: torch.device, grad_clip: float) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if "hand_shape_target" not in batch:
            raise ValueError("Training requires hand_shape_target for L_shape")
        if "context_label" not in batch:
            raise ValueError("Training requires context_label for auxiliary context supervision")
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(
            forward_batch(model, batch),
            batch["force"],
            context_label=batch.get("context_label"),
            hand_shape_target=batch.get("hand_shape_target"),
        )
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        steps += 1
    return reduce_metrics({key: value / max(steps, 1) for key, value in totals.items()}, device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: NiSMLoss, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    sum_squared_error = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = forward_batch(model, batch)
        losses = criterion(
            output,
            batch["force"],
            context_label=batch.get("context_label"),
            hand_shape_target=batch.get("hand_shape_target"),
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        target = batch["force"][:, -1]
        sum_squared_error += float((output.p_hat - target).square().sum())
        count += target.numel()
        steps += 1
    result = {key: value / max(steps, 1) for key, value in totals.items()}
    result["mse"] = sum_squared_error / max(count, 1)
    return reduce_metrics(result, device)


def fit(
    model: NiSM,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    training_config: TrainingConfig,
    criterion: NiSMLoss,
    best_checkpoint_path: str | Path,
    monitor: str = "val_loss",
    monitor_mode: str = "min",
    checkpoint_metadata: dict[str, Any] | None = None,
    checkpoint_selection: str = "best_validation",
    **unused,
) -> list[dict[str, float]]:
    del unused
    if checkpoint_selection not in {"best_validation", "final_epoch"}:
        raise ValueError(f"Unsupported checkpoint selection: {checkpoint_selection}")
    if checkpoint_selection == "best_validation" and (monitor != "val_loss" or monitor_mode != "min" or val_loader is None):
        raise ValueError("LOSO checkpoint selection requires minimum validation loss")
    cfg = training_config
    device = torch.device(cfg.device)
    model.to(device)
    if distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
        threshold=1e-3,
        threshold_mode="abs",
    )
    history: list[dict[str, float]] = []
    best = math.inf
    rank = dist.get_rank() if distributed() else 0
    for epoch in range(1, cfg.num_epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip_norm)
        val_metrics = evaluate(model, val_loader, criterion, device) if val_loader is not None else {}
        val_key = "mse" if os.environ.get("NISM_VAL_METRIC", "loss") == "mse" else "loss"
        # NISM_VAL_METRIC=mse: select and schedule on the final-frame validation MSE
        schedule_loss = val_metrics[val_key] if checkpoint_selection == "best_validation" else train_metrics["loss"]
        if scheduler is not None:
            scheduler.step(schedule_loss)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        improved = checkpoint_selection == "best_validation" and schedule_loss < best
        if improved or (checkpoint_selection == "final_epoch" and epoch == cfg.num_epochs):
            best = schedule_loss
            if rank == 0:
                save_model_checkpoint(best_checkpoint_path, model, epoch, monitor, schedule_loss, checkpoint_metadata)
        history.append(row)
        if rank == 0:
            val_text = f" val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f}" if val_metrics else ""
            print(f"[NiSM] epoch={epoch}/{cfg.num_epochs} train_loss={train_metrics['loss']:.6f}{val_text}", flush=True)
            if cfg.log_jsonl:
                path = Path(cfg.log_jsonl)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
    return history
