"""Train and evaluate the CA-T baseline with fixed-split or LOSO protocols."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines import BASELINE_NAMES, build_baseline
from nism.precomputed import (
    ChunkLevelSampler,
    load_precomputed_fixed_datasets,
    load_precomputed_loso_datasets,
)
from train import json_safe, make_loader, metrics, validate_fixed_protocol, validate_protocol


MODEL_LABELS = {"ca_t": "CA-T"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASELINE_NAMES, required=True)
    parser.add_argument("--dataset", choices=["pimforce", "self_collected"], required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], default="fixed")
    parser.add_argument("--test-user")
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler-patience", type=int)
    parser.add_argument("--lr-scheduler-factor", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--fine-tune-from",
        type=Path,
        help="CA-T checkpoint used for the paper's 1e-4 fine-tuning configuration",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window-length", type=int, choices=[30], default=30)
    parser.add_argument("--hop-length", type=int, choices=[2], default=2)
    parser.add_argument("--num-frames", type=int, choices=[30], default=30)
    parser.add_argument("--target-mode", choices=["last"], default="last")
    return parser.parse_args()


def apply_paper_training_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "num_epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "scheduler": "step",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 30,
        "early_stopping_patience": 50,
    }
    if args.fine_tune_from is not None:
        defaults.update({
            "learning_rate": 1e-4,
            "scheduler": "step",
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 100,
        })
    for name in (
        "num_epochs", "batch_size", "learning_rate", "lr_scheduler_factor",
        "lr_scheduler_patience", "early_stopping_patience",
    ):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    args.scheduler = defaults["scheduler"]
    return args


def setup(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.nan_to_num(value.to(device)) if torch.is_floating_point(value) else value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def ca_t_loss(output: dict[str, torch.Tensor], target: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    pressure_loss = sum(F.smooth_l1_loss(output["pressure"][:, finger], target[:, finger]) for finger in range(5))
    return (
        F.mse_loss(output["pose"], pose)
        + 0.3 * F.mse_loss(output["pose_imu"], pose)
        + 0.3 * F.mse_loss(output["pose_emg"], pose)
        + 3.0 * pressure_loss
        + 0.6 * F.smooth_l1_loss(output["pressure_imu"], target)
        + 0.6 * F.smooth_l1_loss(output["pressure_emg"], target)
        + 0.5 * F.smooth_l1_loss(output["pressure_cross"], target)
    )


def forward_prediction(model: torch.nn.Module, batch: dict[str, torch.Tensor]):
    output = model(batch["x_imu"], batch["emg_sequence"])
    return output["pressure"], output


def batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "emg_sequence" not in batch:
        raise KeyError("CA-T training requires emg_sequence; recreate the cache")
    _, output = forward_prediction(model, batch)
    target = batch["force"][:, -1]
    if "hand_shape_target" not in batch:
        raise KeyError("CA-T training and validation require hand_shape_target")
    return ca_t_loss(output, target, batch["hand_shape_target"])


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    steps = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        loss = batch_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CA-T loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        loss = batch_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CA-T validation loss")
        total += float(loss.detach())
        steps += 1
    if steps == 0:
        raise ValueError("Validation loader is empty")
    return total / steps


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    predictions, targets, users = [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        prediction, _ = forward_prediction(model, batch)
        predictions.append(prediction.cpu())
        targets.append(batch["force"][:, -1].cpu())
        users.append(batch["user_id"].cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(users).numpy(),
    )


def main() -> None:
    args = apply_paper_training_defaults(parse_args())
    if args.protocol == "loso" and not args.test_user:
        raise ValueError("--test-user is required for LOSO")
    if args.protocol == "loso" and args.dataset != "pimforce":
        raise ValueError("The LOSO protocol is defined for PiMForce only")
    setup(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    if args.protocol == "loso":
        train_set, val_set, test_set, metadata = load_precomputed_loso_datasets(
            args.precomputed_root, args.test_user
        )
        protocol = validate_protocol(metadata, args)
        if val_set is None:
            raise ValueError("LOSO requires an independent validation user in every fold")
        if len(train_set) != int(metadata["used_training_sample_count"]):
            raise ValueError(
                f"train index contains {len(train_set)} samples but metadata "
                f"used_training_sample_count={metadata['used_training_sample_count']}"
            )
        run_id = f"{args.test_user}_{args.model}"
        checkpoint_selection = "best_validation"
    else:
        train_set, val_set, test_set, metadata = load_precomputed_fixed_datasets(
            args.precomputed_root, args.dataset
        )
        if val_set is not None:
            raise ValueError("Table 1 fixed split must not contain a validation set")
        protocol = validate_fixed_protocol(metadata, argparse.Namespace(dataset=args.dataset))
        expected_users = 17 if args.dataset == "pimforce" else 16
        if len(protocol["train_users"]) != expected_users:
            raise ValueError(
                f"Table 1 {args.dataset} requires {expected_users} training users, "
                f"got {len(protocol['train_users'])}"
            )
        run_id = f"fixed_{args.dataset}_{args.model}"
        checkpoint_selection = "final_epoch"
    if tuple(metadata.get("emg_raw_shape", (30, 1))) != (30, 1):
        raise ValueError("CA-T requires a 30-frame single-channel EMG sequence")

    train_loader = make_loader(
        train_set,
        args.batch_size,
        args.num_workers,
        args.seed,
        sampler=ChunkLevelSampler(train_set, True, True, args.seed),
        drop_last=False,
    )
    val_loader = (
        make_loader(val_set, args.batch_size, args.num_workers, args.seed + 10_000)
        if val_set is not None
        else None
    )
    model = build_baseline(args.model, sequence_length=30).to(device)
    if args.fine_tune_from is not None:
        payload = torch.load(args.fine_tune_from, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_scheduler_patience,
            gamma=args.lr_scheduler_factor,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            threshold=1e-3,
            threshold_mode="abs",
        )

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{run_id}.pt"
    best_validation = float("inf")
    best_monitor = float("inf")
    epochs_without_improvement = 0

    history = []
    for epoch in range(1, args.num_epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = (
            validation_loss(model, val_loader, device)
            if val_loader is not None
            else None
        )
        monitor_loss = val_loss if val_loss is not None else loss
        if args.scheduler == "step":
            scheduler.step()
        else:
            scheduler.step(monitor_loss)
        row = {"epoch": epoch, "train_loss": loss, "lr": optimizer.param_groups[0]["lr"]}
        if val_loss is not None:
            row["val_loss"] = val_loss
        history.append(row)
        improved = monitor_loss < best_monitor - args.early_stopping_min_delta
        if improved:
            best_monitor = float(monitor_loss)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        validation_improved = val_loss is not None and val_loss < best_validation
        if validation_improved or args.protocol == "fixed":
            if validation_improved:
                best_validation = float(val_loss)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "dataset": args.dataset,
                    "evaluation_protocol": args.protocol,
                    "test_user": args.test_user,
                    "epoch": epoch,
                    "monitor": "val_loss" if args.protocol == "loso" else "train_loss",
                    "monitor_value": val_loss if args.protocol == "loso" else loss,
                    "checkpoint_selection": checkpoint_selection,
                    "training_configuration": {
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "num_epochs": args.num_epochs,
                        "scheduler": args.scheduler,
                        "scheduler_factor": args.lr_scheduler_factor,
                        "scheduler_interval_or_patience": args.lr_scheduler_patience,
                        "early_stopping_patience": args.early_stopping_patience,
                        "fine_tune_from": str(args.fine_tune_from) if args.fine_tune_from else None,
                    },
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        val_text = f" val_loss={val_loss:.6f}" if val_loss is not None else ""
        print(
            f"[{MODEL_LABELS[args.model]}] epoch={epoch}/{args.num_epochs} "
            f"train_loss={loss:.6f}{val_text}",
            flush=True,
        )
        if (
            args.protocol == "loso"
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"[{MODEL_LABELS[args.model]}] early_stop=True "
                f"patience={args.early_stopping_patience}",
                flush=True,
            )
            break

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)

    test_loader = make_loader(
        test_set, args.batch_size, args.num_workers, args.seed + 20_000
    )
    y_pred, y_true, user_ids = predict(model, test_loader, device)
    per_user = {
        f"User{int(user)}": metrics(y_pred[user_ids == user], y_true[user_ids == user])
        for user in sorted(np.unique(user_ids))
    }
    metric_names = ("nrmse", "mae", "r2", "pcc")
    across_users = {
        name: {
            "mean": float(np.mean([value[name] for value in per_user.values()])),
            "std": float(np.std([value[name] for value in per_user.values()], ddof=1))
            if len(per_user) > 1 else 0.0,
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
    summary = {
        "model": MODEL_LABELS[args.model],
        "dataset": args.dataset,
        "ablation": "full",
        "evaluation_protocol": args.protocol,
        "train_user_count": len(protocol["train_users"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_user": args.test_user,
        "run_id": run_id,
        "seed": args.seed,
        "protocol": protocol,
        "train_windows": len(train_set),
        "val_windows": len(val_set) if val_set is not None else 0,
        "test_windows": len(test_set),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_epoch": payload.get("epoch"),
        "training_configuration": payload.get("training_configuration"),
        "test_metrics": metrics(y_pred, y_true),
        "per_user_metrics": per_user,
        "across_user_metrics": across_users,
        "history": history,
    }
    summary_path = summaries_dir / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "test_metrics": summary["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
