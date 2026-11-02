"""Train and evaluate NiSM with fixed-split or LOSO protocols."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from nism import NiSM, NiSMConfig, NiSMLoss, NiSMLossWeights, TrainingConfig
from nism.encoders import EMG_FEATURE_NAMES
from nism.precomputed import ChunkLevelSampler, load_precomputed_fixed_datasets, load_precomputed_loso_datasets
from nism.training import evaluate, fit, unwrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["loso", "fixed"], default="loso")
    parser.add_argument("--dataset", choices=["pimforce", "self_collected"], default="pimforce")
    parser.add_argument(
        "--ablation",
        choices=["full", "no_context_modulation", "no_sparse_feedforward", "no_emg_scaling"],
        default="full",
    )
    parser.add_argument("--model-name", default="NiSM")
    parser.add_argument("--test-user")
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-length", type=int, choices=[10, 20, 30], default=30)
    parser.add_argument("--hop-length", type=int, choices=[2], default=2)
    parser.add_argument("--num-frames", type=int, choices=[10, 20, 30], default=30)
    parser.add_argument("--target-mode", choices=["last"], default="last")
    parser.add_argument("--output-mode", choices=["point"], default="point")
    parser.add_argument("--emg-input-mode", choices=["feature10"], default="feature10")
    parser.add_argument("--intent-label-mode", choices=["sensor_context"], default="sensor_context")
    parser.add_argument("--lambda-ctx", type=float, default=0.05)
    parser.add_argument("--lambda-shape", type=float, default=0.1)
    parser.add_argument("--lambda-peak", type=float, default=2.0)
    parser.add_argument("--lambda-synergy-sparse", type=float, default=1e-4)
    parser.add_argument("--lambda-mask-sparse", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["reduce_on_plateau"], default="reduce_on_plateau")
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--val-user-count", type=int, choices=[1], default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-precomputed-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def user_id(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"User name has no numeric ID: {value!r}")
    return int(match.group())


def validate_protocol(metadata: dict, args: argparse.Namespace) -> dict:
    problems: list[str] = []
    if metadata.get("protocol") != "loso":
        problems.append(f"metadata protocol must be 'loso', got {metadata.get('protocol')!r}")
    if metadata.get("dataset") != args.dataset:
        problems.append(
            f"metadata dataset={metadata.get('dataset')!r}, requested={args.dataset!r}"
        )
    if metadata.get("checkpoint_selection") != "best_validation":
        problems.append("LOSO checkpoint_selection must be best_validation")
    test_user = str(metadata.get("test_user", ""))
    train_users = {str(x) for x in metadata.get("train_users", [])}
    val_users = {str(x) for x in metadata.get("val_users", [])}
    if test_user != args.test_user:
        problems.append(f"metadata test_user={test_user!r}, requested={args.test_user!r}")
    if args.test_user in train_users or args.test_user in val_users:
        problems.append("held-out test user appears in train/validation users")
    if train_users & val_users:
        problems.append("train and validation users overlap")
    candidates = train_users | val_users
    if candidates:
        expected = min(candidates, key=lambda x: (abs(user_id(x) - user_id(args.test_user)), user_id(x)))
        if val_users != {expected}:
            problems.append(f"validation user must be closest non-test user {expected}, got {sorted(val_users)}")

    record_sets: dict[str, set[str]] = {}
    split_sessions: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        records = metadata.get(f"{split}_trial_records", [])
        keys = set()
        users = set()
        sessions = set()
        for record in records if isinstance(records, list) else []:
            if isinstance(record, dict):
                users.add(str(record.get("user", "")))
                sessions.add(user_id(record.get("session", "")))
                keys.add(str(record.get("force_path") or (record.get("user"), record.get("session"), record.get("trial_id"))))
            else:
                keys.add(str(record))
        record_sets[split] = keys
        split_sessions[split] = sessions
        if split == "test" and users and users != {args.test_user}:
            problems.append(f"test records contain users other than {args.test_user}: {sorted(users)}")
        if split != "test" and args.test_user in users:
            problems.append(f"held-out user appears in {split} records")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if record_sets[left] & record_sets[right]:
            problems.append(f"{left}/{right} trial records overlap")
    if split_sessions.get("train") != {1}:
        problems.append(f"PiMForce training users must use only Session 1, got {sorted(split_sessions.get('train', set()))}")
    if split_sessions.get("val") != {1}:
        problems.append(f"PiMForce validation user must use only Session 1, got {sorted(split_sessions.get('val', set()))}")
    if split_sessions.get("test") != {1, 2, 3}:
        problems.append(f"PiMForce test user must use Sessions 1,2,3, got {sorted(split_sessions.get('test', set()))}")

    source_count = metadata.get("full_training_sample_count")
    used_count = metadata.get("used_training_sample_count")
    training_fraction = None
    if source_count is not None and used_count is not None and float(source_count) > 0:
        training_fraction = float(used_count) / float(source_count)
    else:
        problems.append("full_training_sample_count and used_training_sample_count are required")
    half_tolerance = 1.0 / max(float(source_count or 1), 1.0)
    if training_fraction is None or abs(float(training_fraction) - 0.5) > half_tolerance:
        problems.append(f"PiMForce training_fraction must be 0.5, got {training_fraction!r}")

    normalization = metadata.get("normalization")
    required_norm = {"fit_users", "applied", "imu_mean", "imu_std", "emg_feature_mean", "emg_feature_std", "emg_raw_mean", "emg_raw_std", "force_min", "force_max", "hand_shape_mean", "hand_shape_std"}
    if not isinstance(normalization, dict):
        problems.append("normalization metadata is missing")
    else:
        missing_norm = required_norm - normalization.keys()
        if missing_norm:
            problems.append(f"normalization metadata is missing keys {sorted(missing_norm)}")
        hand_shape_dim = int(metadata.get("hand_shape_dim", 0))
        if hand_shape_dim <= 0:
            problems.append("hand_shape_dim must be a positive integer")
        else:
            for key in ("hand_shape_mean", "hand_shape_std"):
                if key in normalization and len(normalization[key]) != hand_shape_dim:
                    problems.append(
                        f"normalization[{key}] must contain {hand_shape_dim} values, "
                        f"got {len(normalization[key])}"
                    )
        fit_users = {str(x) for x in normalization.get("fit_users", [])}
        if fit_users != train_users:
            problems.append(f"normalization must be fitted only on training users; fit_users={sorted(fit_users)}")

    window = metadata.get("window_config", {})
    expected_window = {
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "num_frames": args.num_frames,
        "target_mode": args.target_mode,
    }
    for key, expected in expected_window.items():
        if str(window.get(key)) != str(expected):
            problems.append(f"window_config[{key}]={window.get(key)!r}, expected={expected!r}")
    if problems:
        raise ValueError("Invalid LOSO metadata: " + "; ".join(problems))
    return {
        "protocol": "loso",
        "dataset": args.dataset,
        "test_user": args.test_user,
        "train_users": sorted(train_users, key=user_id),
        "val_users": sorted(val_users, key=user_id),
        "validation_rule": "closest_non_test_numeric_user_id_lower_id_tiebreak",
        "checkpoint_selection": "best_validation",
        "test_user_excluded_from_model_selection": True,
        "training_fraction": float(training_fraction) if training_fraction is not None else None,
        "sessions": {key: sorted(value) for key, value in split_sessions.items()},
        "normalization_applied_in_cache": bool(normalization.get("applied", False)) if isinstance(normalization, dict) else None,
    }


def validate_fixed_protocol(metadata: dict, args: argparse.Namespace) -> dict:
    problems: list[str] = []
    expected_test_ids = {4, 9, 15, 20} if args.dataset == "pimforce" else set()
    train_records = metadata.get("train_trial_records", [])
    val_records = metadata.get("val_trial_records", [])
    test_records = metadata.get("test_trial_records", [])
    train_users = {user_id(record.get("user")) for record in train_records if isinstance(record, dict)}
    val_users = {user_id(record.get("user")) for record in val_records if isinstance(record, dict)}
    test_users = {user_id(record.get("user")) for record in test_records if isinstance(record, dict)}
    if args.dataset == "self_collected":
        all_ids = train_users | val_users | test_users
        expected_test_ids = set(sorted(all_ids)[-4:])
    if test_users != expected_test_ids:
        problems.append(f"fixed {args.dataset} test users must be {sorted(expected_test_ids)}, got {sorted(test_users)}")
    allowed_train_counts = {17, 14, 11, 8, 5, 2} if args.dataset == "pimforce" else {16}
    if len(train_users) not in allowed_train_counts:
        problems.append(
            f"fixed {args.dataset} requires a training-user count in "
            f"{sorted(allowed_train_counts, reverse=True)}, got {len(train_users)}"
        )
    if val_users:
        problems.append("fixed split must not create a validation split; training uses the final epoch checkpoint")
    if (train_users | val_users) & test_users:
        problems.append("fixed-split test users overlap training/validation users")
    if args.dataset == "pimforce":
        train_sessions = {user_id(record.get("session")) for record in train_records if isinstance(record, dict)}
        test_sessions = {user_id(record.get("session")) for record in test_records if isinstance(record, dict)}
        if train_sessions != {1}:
            problems.append(f"PiMForce fixed training must use only Session 1, got {sorted(train_sessions)}")
        if test_sessions != {1, 2, 3}:
            problems.append(f"PiMForce fixed testing must use Sessions 1,2,3, got {sorted(test_sessions)}")
        full_count = metadata.get("full_training_sample_count")
        used_count = metadata.get("used_training_sample_count")
        if full_count is None or used_count is None or float(full_count) <= 0 or abs(float(used_count) / float(full_count) - 0.5) > 1.0 / float(full_count):
            problems.append("PiMForce fixed split must record and use exactly 50% of training samples")
    normalization = metadata.get("normalization")
    required_norm = {"fit_users", "applied", "imu_mean", "imu_std", "emg_feature_mean", "emg_feature_std", "emg_raw_mean", "emg_raw_std", "force_min", "force_max", "hand_shape_mean", "hand_shape_std"}
    if not isinstance(normalization, dict) or required_norm - normalization.keys():
        problems.append("fixed split requires complete training-only normalization metadata")
    else:
        hand_shape_dim = int(metadata.get("hand_shape_dim", 0))
        if hand_shape_dim <= 0:
            problems.append("hand_shape_dim must be a positive integer")
        else:
            for key in ("hand_shape_mean", "hand_shape_std"):
                if key in normalization and len(normalization[key]) != hand_shape_dim:
                    problems.append(
                        f"normalization[{key}] must contain {hand_shape_dim} values, "
                        f"got {len(normalization[key])}"
                    )
        fit_users = {user_id(value) for value in normalization.get("fit_users", [])}
        if fit_users != train_users:
            problems.append(f"normalization fit users must equal fixed training users, got {sorted(fit_users)}")
        if fit_users & test_users:
            problems.append("fixed test users participated in normalization")
    if metadata.get("checkpoint_selection") != "final_epoch":
        problems.append("fixed split checkpoint_selection must be final_epoch")
    if problems:
        raise ValueError("Invalid fixed-split metadata: " + "; ".join(problems))
    return {
        "protocol": "fixed",
        "dataset": args.dataset,
        "train_users": sorted(train_users),
        "validation_users": sorted(val_users),
        "test_users": sorted(test_users),
        "checkpoint_selection": "final_epoch",
        "test_users_excluded_from_model_selection": True,
    }


def setup_runtime(seed: int, deterministic: bool) -> tuple[bool, int, int, int]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return distributed, rank, local_rank, world_size


def make_loader(dataset, batch_size: int, workers: int, seed: int, sampler=None, drop_last: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )


@torch.no_grad()
def predictions(model: NiSM, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pred_parts, true_parts, user_parts = [], [], []
    for batch in loader:
        x_imu = batch["x_imu"].to(device).float()
        emg_feat = batch.get("emg_features")
        if emg_feat is not None:
            emg_feat = emg_feat.to(device).float()
        output = model(x_imu, emg_features=emg_feat)
        pred_parts.append(output.p_hat.cpu())
        true_parts.append(batch["force"][:, -1].float().cpu())
        user_parts.append(batch["user_id"].long().cpu())
    return torch.cat(pred_parts).numpy(), torch.cat(true_parts).numpy(), torch.cat(user_parts).numpy()


def metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    pred = np.nan_to_num(y_pred).reshape(-1, 5).astype(np.float64)
    true = np.nan_to_num(y_true).reshape(-1, 5).astype(np.float64)
    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    finger_rmse = np.sqrt(np.mean(error**2, axis=0))
    finger_range = np.max(true, axis=0) - np.min(true, axis=0)
    nrmse = float(np.mean(finger_rmse / np.maximum(finger_range, 1e-8)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    pcc_values = []
    for finger in range(5):
        if np.std(pred[:, finger]) > 0 and np.std(true[:, finger]) > 0:
            pcc_values.append(float(np.corrcoef(pred[:, finger], true[:, finger])[0, 1]))
    return {"mse": mse, "rmse": rmse, "nrmse": nrmse, "mae": mae, "r2": r2, "pcc": float(np.mean(pcc_values)) if pcc_values else 0.0}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    if args.window_length != args.num_frames:
        raise ValueError("--window-length and --num-frames must be identical")
    if args.protocol == "loso" and not args.test_user:
        raise ValueError("--test-user is required for LOSO")
    distributed, rank, local_rank, world_size = setup_runtime(args.seed, args.deterministic)
    device = torch.device(f"cuda:{local_rank}" if args.device.startswith("cuda") else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.protocol == "loso":
        train_set, val_set, test_set, metadata = load_precomputed_loso_datasets(args.precomputed_root, args.test_user)
        protocol = validate_protocol(metadata, args) if args.validate_precomputed_metadata else {"validation": "disabled"}
        run_id = str(args.test_user)
        checkpoint_selection = "best_validation"
    else:
        train_set, val_set, test_set, metadata = load_precomputed_fixed_datasets(args.precomputed_root, args.dataset)
        protocol = validate_fixed_protocol(metadata, args) if args.validate_precomputed_metadata else {"validation": "disabled"}
        run_id = f"fixed_{args.dataset}"
        checkpoint_selection = "final_epoch"
    hand_shape_dim = int(metadata.get("hand_shape_dim", 0))
    if hand_shape_dim != 63:
        raise ValueError(
            f"metadata hand_shape_dim must be 63 (21 joints x 3 coordinates), got {hand_shape_dim}"
        )
    if tuple(metadata.get("emg_feature_names", ())) != EMG_FEATURE_NAMES:
        raise ValueError(
            "Cache EMG feature order does not match the paper-defined 10 features; "
            "rebuild the cache with the paper-defined feature order"
        )
    if args.protocol == "loso" and args.validate_precomputed_metadata and len(train_set) != int(metadata["used_training_sample_count"]):
        raise ValueError(
            f"train index contains {len(train_set)} samples but metadata used_training_sample_count="
            f"{metadata['used_training_sample_count']}"
        )
    if args.protocol == "fixed" and args.dataset == "pimforce" and args.validate_precomputed_metadata:
        if len(train_set) != int(metadata["used_training_sample_count"]):
            raise ValueError(
                f"fixed train index contains {len(train_set)} samples but metadata used_training_sample_count="
                f"{metadata['used_training_sample_count']}"
            )
    if args.protocol == "loso" and val_set is None:
        raise ValueError("LOSO requires a validation user in every fold")

    if args.batch_size % world_size != 0:
        raise ValueError(f"Global batch size {args.batch_size} must be divisible by world size {world_size}")
    per_rank_batch_size = args.batch_size // world_size
    train_sampler = ChunkLevelSampler(train_set, True, True, args.seed, world_size, rank) if distributed else ChunkLevelSampler(train_set, True, True, args.seed)
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed and val_set is not None else None
    train_loader = make_loader(train_set, per_rank_batch_size, args.num_workers, args.seed + rank, train_sampler, drop_last=False)
    val_loader = make_loader(val_set, per_rank_batch_size, args.num_workers, args.seed + 10_000 + rank, val_sampler) if val_set is not None else None

    model = NiSM(NiSMConfig(
        d_imu=48,
        d_ctx=48,
        d_emg=48,
        imu_lstm_hidden=225,
        intent_label_mode="sensor_context",
        sensor_context_proto_logits_mode="none",
        use_sensor_context_bit_head=False,
        use_context_classifier=True,
        num_context_classes=int(metadata.get("num_context_classes", 22 if args.dataset == "pimforce" else 20)),
        hand_shape_dim=hand_shape_dim,
        emg_input_mode="feature10",
        output_mode="point",
        force_sequence_len=args.num_frames,
        emg_frame_window=args.num_frames,
        emg_frame_hop=2,
        ablation=args.ablation,
    ))
    criterion = NiSMLoss(
        NiSMLossWeights(
            lambda_force=1.0,
            lambda_peak=args.lambda_peak,
            lambda_synergy_sparse=0.0 if args.ablation == "no_sparse_feedforward" else args.lambda_synergy_sparse,
            lambda_mask_sparse=args.lambda_mask_sparse,
            lambda_ctx=args.lambda_ctx,
            lambda_shape=args.lambda_shape,
        ),
        peak_threshold=0.7,
    )
    training = TrainingConfig(
        learning_rate=args.learning_rate,
        optimizer="adam",
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=str(device),
        log_jsonl=str(args.log_jsonl) if args.log_jsonl else None,
        lr_scheduler="reduce_on_plateau",
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
    )
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"checkpoint_{run_id}.pt"
    history = fit(
        model,
        train_loader,
        val_loader,
        training,
        criterion,
        checkpoint_path,
        monitor="val_loss" if args.protocol == "loso" else "train_loss",
        monitor_mode="min",
        checkpoint_metadata={"test_user": args.test_user, "protocol": protocol, "seed": args.seed},
        checkpoint_selection=checkpoint_selection,
    )

    if distributed:
        dist.barrier()
        model = unwrap_model(model)
        dist.destroy_process_group()
    if rank != 0:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    test_loader = make_loader(test_set, args.batch_size, args.num_workers, args.seed + 20_000)
    test_loss = evaluate(model, test_loader, criterion, device)
    y_pred, y_true, test_user_ids = predictions(model, test_loader, device)
    test_metrics = metrics(y_pred, y_true)
    per_user_metrics = {
        f"User{int(user)}": metrics(y_pred[test_user_ids == user], y_true[test_user_ids == user])
        for user in sorted(np.unique(test_user_ids))
    }
    metric_names = ("nrmse", "mae", "r2", "pcc")
    across_user_metrics = {
        name: {
            "mean": float(np.mean([values[name] for values in per_user_metrics.values()])),
            "std": float(np.std([values[name] for values in per_user_metrics.values()], ddof=1)) if len(per_user_metrics) > 1 else 0.0,
        }
        for name in metric_names
    }

    predictions_dir = args.output_dir / "predictions"
    summaries_dir = args.output_dir / "summaries"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_dir / f"predictions_{run_id}.npz", y_pred=y_pred, y_true=y_true, user_id=test_user_ids)
    summary = {
        "model": args.model_name,
        "dataset": args.dataset,
        "ablation": args.ablation,
        "evaluation_protocol": args.protocol,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_user_count": len(protocol.get("train_users", [])),
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
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "per_user_metrics": per_user_metrics,
        "across_user_metrics": across_user_metrics,
        "history": history,
    }
    summary_path = summaries_dir / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
