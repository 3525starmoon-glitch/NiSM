"""Evaluate a trained NiSM checkpoint without auxiliary training labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nism import NiSM, NiSMConfig
from nism.precomputed import load_precomputed_fixed_datasets, load_precomputed_loso_datasets
from train import json_safe, make_loader, metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], required=True)
    parser.add_argument("--dataset", choices=["pimforce", "self_collected"], required=True)
    parser.add_argument("--test-user", help="Required for LOSO, for example User1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "loso" and not args.test_user:
        raise ValueError("--test-user is required for LOSO evaluation")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    if args.protocol == "loso":
        _, _, test_set, metadata = load_precomputed_loso_datasets(args.precomputed_root, args.test_user)
        run_id = args.test_user
    else:
        _, _, test_set, metadata = load_precomputed_fixed_datasets(args.precomputed_root, args.dataset)
        run_id = f"fixed_{args.dataset}"

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload or "model_config" not in payload:
        raise ValueError("Checkpoint must contain model_state_dict and model_config")
    model = NiSM(NiSMConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)

    loader: DataLoader = make_loader(test_set, args.batch_size, args.num_workers, seed=2026)
    y_pred, y_true, user_ids = predictions(model, loader, device)
    per_user = {
        f"User{int(user)}": metrics(y_pred[user_ids == user], y_true[user_ids == user])
        for user in sorted(np.unique(user_ids))
    }
    metric_names = ("nrmse", "mae", "r2", "pcc")
    across_users = {
        name: {
            "mean": float(np.mean([values[name] for values in per_user.values()])),
            "std": float(np.std([values[name] for values in per_user.values()], ddof=1))
            if len(per_user) > 1 else 0.0,
        }
        for name in metric_names
    }
    train_users = metadata.get("train_users", [])
    result = {
        "model": "NiSM",
        "dataset": args.dataset,
        "ablation": model.config.ablation,
        "evaluation_protocol": args.protocol,
        "train_user_count": len(train_users),
        "test_user": args.test_user,
        "run_id": run_id,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "test_windows": len(test_set),
        "test_metrics": metrics(y_pred, y_true),
        "per_user_metrics": per_user,
        "across_user_metrics": across_users,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(args.output), "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
