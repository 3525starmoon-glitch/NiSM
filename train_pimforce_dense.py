"""Train the PiMForce Dense reference with fixed-split or PiMForce LOSO."""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baselines import build_pimforce_dense
from baselines.pimforce_dense_data import (
    DenseArrayStore,
    DenseNormalizer,
    PiMForceDenseDataset,
    build_split,
    select_indices,
)
from train import json_safe, metrics


MODEL_LABEL = "PiMForce (Dense)"
PAPER_PARAMETER_LABEL = "66.39M"
PAPER_CONFIG_PATH = Path(__file__).resolve().parent / "baselines" / "pimforce_dense_config.json"
PAPER_CONFIG = json.loads(PAPER_CONFIG_PATH.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], default="fixed")
    parser.add_argument("--test-user")
    parser.add_argument("--train-user-count", type=int, choices=[17, 14, 11, 8, 5, 2], default=17)
    parser.add_argument("--num-epochs", type=int, default=int(PAPER_CONFIG["max_epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(PAPER_CONFIG["batch_size"]))
    parser.add_argument("--learning-rate", type=float, default=float(PAPER_CONFIG["learning_rate"]))
    parser.add_argument("--scheduler-factor", type=float, default=float(PAPER_CONFIG["scheduler_factor"]))
    parser.add_argument("--scheduler-patience", type=int, default=int(PAPER_CONFIG["scheduler_patience"]))
    parser.add_argument("--early-stopping-patience", type=int, default=int(PAPER_CONFIG["early_stopping_patience"]))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def setup(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(
    dataset: PiMForceDenseDataset,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=partial(seed_worker, base_seed=seed),
        generator=generator,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if torch.is_tensor(value)
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch["pose_voxels"], batch["emg_dense"])
        loss = F.l1_loss(output["pressure"], batch["pressure"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PiMForce Dense training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(batch["pressure"].shape[0])
        total += float(loss.detach()) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Training loader is empty")
    return total / samples


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: DenseNormalizer,
    inverse_pressure: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    users: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        prediction = model(batch["pose_voxels"], batch["emg_dense"])["pressure"]
        target = batch["pressure"]
        if inverse_pressure:
            prediction = normalizer.pressure.inverse(prediction)
            target = normalizer.pressure.inverse(target)
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        users.append(batch["user_id"].cpu())
    if not predictions:
        raise ValueError("Evaluation loader is empty")
    return (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(users).numpy(),
    )


def normalized_rmse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: DenseNormalizer,
) -> float:
    prediction, target, _ = predict(model, loader, device, normalizer, inverse_pressure=False)
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def evaluation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Table metrics plus the additional Dense metrics listed in the appendix."""
    from scipy.stats import spearmanr

    result = metrics(prediction, target)
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    result["rmse"] = float(np.sqrt(np.mean(np.square(error))))
    nonzero = np.abs(target) > 1e-8
    result["mape"] = (
        float(np.mean(np.abs(error[nonzero] / target[nonzero])))
        if np.any(nonzero)
        else 0.0
    )
    correlations = []
    for finger in range(target.shape[1]):
        coefficient = float(spearmanr(prediction[:, finger], target[:, finger]).statistic)
        if np.isfinite(coefficient):
            correlations.append(coefficient)
    result["spearman"] = float(np.mean(correlations)) if correlations else 0.0
    return result


def checkpoint_payload(
    model: torch.nn.Module,
    normalizer: DenseNormalizer,
    args: argparse.Namespace,
    split_metadata: dict[str, object],
    epoch: int,
    monitor: str,
    monitor_value: float,
    checkpoint_selection: str,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "model": "pimforce_dense",
        "model_label": MODEL_LABEL,
        "dataset": "pimforce",
        "evaluation_protocol": args.protocol,
        "test_user": args.test_user,
        "train_user_count": len(split_metadata["train_users"]),
        "epoch": epoch,
        "monitor": monitor,
        "monitor_value": monitor_value,
        "checkpoint_selection": checkpoint_selection,
        "normalizer": normalizer.state_dict(),
        "protocol": split_metadata,
        "training_configuration": {
            "config": str(PAPER_CONFIG_PATH.name),
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "loss": "MAE",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "early_stopping_patience": (
                args.early_stopping_patience if args.protocol == "loso" else None
            ),
        },
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    if args.protocol == "loso" and args.train_user_count != 17:
        raise ValueError("--train-user-count is a fixed-split Figure 3 option")
    if args.protocol == "fixed" and args.test_user is not None:
        raise ValueError("Fixed PiMForce Dense always tests User4, User9, User15, and User20")
    setup(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    repository_root = Path(__file__).resolve().parent
    store = DenseArrayStore(args.data_root)
    split = build_split(
        store,
        repository_root / "splits",
        args.protocol,
        args.test_user,
        args.train_user_count,
        args.seed,
    )
    split_metadata = split.metadata()
    normalizer = DenseNormalizer.fit(store, split.train_indices)
    train_set = PiMForceDenseDataset(store, split.train_indices, normalizer)
    train_loader = make_loader(
        train_set, args.batch_size, args.num_workers, args.seed, shuffle=True
    )
    validation_set = (
        PiMForceDenseDataset(store, split.validation_indices, normalizer)
        if split.validation_indices.size
        else None
    )
    validation_loader = (
        make_loader(
            validation_set,
            args.batch_size,
            args.num_workers,
            args.seed + 10_000,
            shuffle=False,
        )
        if validation_set is not None
        else None
    )

    model = build_pimforce_dense().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    checkpoint_selection = "best_validation" if args.protocol == "loso" else "final_epoch"
    run_id = (
        f"{args.test_user}_pimforce_dense"
        if args.protocol == "loso"
        else f"fixed_pimforce_dense_{len(split.train_users)}users"
    )
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{run_id}.pt"
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        row: dict[str, float | int] = {"epoch": epoch, "train_loss": train_loss}
        if validation_loader is None:
            monitor_name = "train_mae"
            monitor_value = train_loss
        else:
            monitor_name = "validation_rmse"
            monitor_value = normalized_rmse(model, validation_loader, device, normalizer)
            row["validation_rmse"] = monitor_value
        scheduler.step(monitor_value)
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        if args.protocol == "loso":
            if monitor_value < best_validation:
                best_validation = monitor_value
                stale_epochs = 0
                torch.save(
                    checkpoint_payload(
                        model,
                        normalizer,
                        args,
                        split_metadata,
                        epoch,
                        monitor_name,
                        monitor_value,
                        checkpoint_selection,
                    ),
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(
                    f"[{MODEL_LABEL}] early_stop=True patience={args.early_stopping_patience}",
                    flush=True,
                )
                break
        elif epoch == args.num_epochs:
            torch.save(
                checkpoint_payload(
                    model,
                    normalizer,
                    args,
                    split_metadata,
                    epoch,
                    monitor_name,
                    monitor_value,
                    checkpoint_selection,
                ),
                checkpoint_path,
            )
        validation_text = (
            f" validation_rmse={monitor_value:.6f}" if validation_loader is not None else ""
        )
        print(
            f"[{MODEL_LABEL}] epoch={epoch}/{args.num_epochs} train_mae={train_loss:.6f}"
            f"{validation_text} lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )

    if not checkpoint_path.is_file():
        raise RuntimeError("Training did not create a checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    # Test rows and loader.
    test_indices = select_indices(store, split.test_users, [1, 2, 3])
    test_set = PiMForceDenseDataset(store, test_indices, normalizer)
    test_loader = make_loader(
        test_set, args.batch_size, args.num_workers, args.seed + 20_000, shuffle=False
    )
    y_pred, y_true, user_ids = predict(
        model, test_loader, device, normalizer, inverse_pressure=True
    )
    per_user = {
        f"User{int(user)}": evaluation_metrics(
            y_pred[user_ids == user], y_true[user_ids == user]
        )
        for user in sorted(np.unique(user_ids))
    }
    metric_names = ("nrmse", "mae", "r2", "pcc")
    across_users = {
        name: {
            "mean": float(np.mean([value[name] for value in per_user.values()])),
            "std": (
                float(np.std([value[name] for value in per_user.values()], ddof=1))
                if len(per_user) > 1
                else 0.0
            ),
        }
        for name in metric_names
    }
    predictions_dir = args.output_dir / "predictions"
    summaries_dir = args.output_dir / "summaries"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_dir / f"predictions_{run_id}.npz",
        y_pred=y_pred,
        y_true=y_true,
        user_id=user_ids,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "model": MODEL_LABEL,
        "dataset": "pimforce",
        "ablation": "full",
        "evaluation_protocol": args.protocol,
        "train_user_count": len(split.train_users),
        "parameter_count": parameter_count,
        "paper_parameter_label": PAPER_PARAMETER_LABEL,
        "test_user": args.test_user,
        "run_id": run_id,
        "seed": args.seed,
        "protocol": split_metadata,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set) if validation_set is not None else 0,
        "test_samples": len(test_set),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_epoch": checkpoint["epoch"],
        "training_configuration": checkpoint["training_configuration"],
        "test_metrics": evaluation_metrics(y_pred, y_true),
        "per_user_metrics": per_user,
        "across_user_metrics": across_users,
        "history": history,
    }
    summary_path = summaries_dir / f"summary_{run_id}.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"summary": str(summary_path), "test_metrics": summary["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
