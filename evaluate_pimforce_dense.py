"""Evaluate a saved PiMForce Dense checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines import build_pimforce_dense
from baselines.pimforce_dense_data import (
    DenseArrayStore,
    DenseNormalizer,
    PiMForceDenseDataset,
    build_split,
    select_indices,
)
from train import json_safe
from train_pimforce_dense import MODEL_LABEL, evaluation_metrics, make_loader, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], required=True)
    parser.add_argument("--test-user")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "pimforce_dense":
        raise ValueError("Checkpoint is not a PiMForce Dense checkpoint")
    if checkpoint.get("evaluation_protocol") != args.protocol:
        raise ValueError("Requested protocol does not match checkpoint protocol")
    if args.protocol == "loso" and checkpoint.get("test_user") != args.test_user:
        raise ValueError("Requested LOSO user does not match checkpoint test user")
    expected_selection = "best_validation" if args.protocol == "loso" else "final_epoch"
    if checkpoint.get("checkpoint_selection") != expected_selection:
        raise ValueError(
            f"Checkpoint selection must be {expected_selection!r} for {args.protocol}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    store = DenseArrayStore(args.data_root)
    train_user_count = int(checkpoint.get("train_user_count", 17))
    split = build_split(
        store,
        Path(__file__).resolve().parent / "splits",
        args.protocol,
        args.test_user,
        train_user_count,
        int(checkpoint.get("seed", 123)),
    )
    if split.metadata() != checkpoint.get("protocol"):
        raise ValueError("Current split manifest does not match the checkpoint split")
    normalizer = DenseNormalizer.from_state_dict(checkpoint["normalizer"])
    test_indices = select_indices(store, split.test_users, [1, 2, 3])
    test_set = PiMForceDenseDataset(store, test_indices, normalizer)
    test_loader = make_loader(
        test_set,
        args.batch_size,
        args.num_workers,
        int(checkpoint.get("seed", 123)) + 20_000,
        shuffle=False,
    )
    model = build_pimforce_dense()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    y_pred, y_true, user_ids = predict(
        model, test_loader, device, normalizer, inverse_pressure=True
    )
    per_user = {
        f"User{int(user)}": evaluation_metrics(
            y_pred[user_ids == user], y_true[user_ids == user]
        )
        for user in sorted(np.unique(user_ids))
    }
    result = {
        "model": MODEL_LABEL,
        "dataset": "pimforce",
        "ablation": "full",
        "evaluation_protocol": args.protocol,
        "train_user_count": len(split.train_users),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_user": args.test_user,
        "protocol": split.metadata(),
        "checkpoint": args.checkpoint.name,
        "checkpoint_selection": checkpoint["checkpoint_selection"],
        "checkpoint_epoch": checkpoint["epoch"],
        "training_configuration": checkpoint["training_configuration"],
        "test_metrics": evaluation_metrics(y_pred, y_true),
        "per_user_metrics": per_user,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"summary": str(args.output), "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
