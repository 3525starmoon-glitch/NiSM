"""Evaluate a CA-T checkpoint on a fixed or LOSO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines import build_baseline
from nism.precomputed import load_precomputed_fixed_datasets, load_precomputed_loso_datasets
from train import json_safe, make_loader, metrics, validate_fixed_protocol, validate_protocol
from train_baseline import MODEL_LABELS, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["pimforce", "self_collected"], required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], default="fixed")
    parser.add_argument("--test-user")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-length", type=int, choices=[30], default=30)
    parser.add_argument("--hop-length", type=int, choices=[2], default=2)
    parser.add_argument("--num-frames", type=int, choices=[30], default=30)
    parser.add_argument("--target-mode", choices=["last"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "loso" and not args.test_user:
        raise ValueError("--test-user is required for LOSO")
    if args.protocol == "loso" and args.dataset != "pimforce":
        raise ValueError("The LOSO protocol is defined for PiMForce only")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(payload.get("model", ""))
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported or missing CA-T model in checkpoint: {model_name!r}")
    if payload.get("dataset") != args.dataset:
        raise ValueError(
            f"Checkpoint dataset={payload.get('dataset')!r}, requested={args.dataset!r}"
        )
    checkpoint_protocol = str(payload.get("evaluation_protocol", "fixed"))
    if checkpoint_protocol != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={checkpoint_protocol!r}, requested={args.protocol!r}"
        )
    if args.protocol == "loso" and payload.get("test_user") != args.test_user:
        raise ValueError(
            f"Checkpoint test_user={payload.get('test_user')!r}, requested={args.test_user!r}"
        )
    expected_selection = "best_validation" if args.protocol == "loso" else "final_epoch"
    if payload.get("checkpoint_selection") != expected_selection:
        raise ValueError(
            f"Checkpoint selection={payload.get('checkpoint_selection')!r}, "
            f"expected={expected_selection!r}"
        )

    if args.protocol == "loso":
        _, val_set, test_set, metadata = load_precomputed_loso_datasets(
            args.precomputed_root, args.test_user
        )
        protocol = validate_protocol(metadata, args)
        if val_set is None:
            raise ValueError("LOSO requires an independent validation user in every fold")
    else:
        _, _, test_set, metadata = load_precomputed_fixed_datasets(
            args.precomputed_root, args.dataset
        )
        protocol = validate_fixed_protocol(metadata, args)
    sequence_length = int(metadata["window_config"]["num_frames"])
    model = build_baseline(model_name, sequence_length=sequence_length)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    loader = make_loader(test_set, args.batch_size, args.num_workers, seed=int(payload.get("seed", 2026)))
    y_pred, y_true, user_ids = predict(model, loader, device)
    per_user = {
        f"User{int(user)}": metrics(y_pred[user_ids == user], y_true[user_ids == user])
        for user in sorted(np.unique(user_ids))
    }
    result = {
        "model": MODEL_LABELS[model_name],
        "dataset": args.dataset,
        "ablation": "full",
        "evaluation_protocol": args.protocol,
        "train_user_count": len(protocol.get("train_users", [])),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_user": args.test_user,
        "protocol": protocol,
        "checkpoint": args.checkpoint.name,
        "checkpoint_selection": payload.get("checkpoint_selection"),
        "checkpoint_epoch": payload.get("epoch"),
        "training_configuration": payload.get("training_configuration"),
        "test_metrics": metrics(y_pred, y_true),
        "per_user_metrics": per_user,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(args.output), "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
