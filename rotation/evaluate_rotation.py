"""Test-time rotational-offset sweep for a trained NiSM checkpoint (no retraining).

Simulates a ring that is mounted with an angular offset relative to the placement seen in
training. The 12-D IMU input is [acc_xyz (sensor frame), R (3x3 sensor->world, row-major)].
A mounting offset R_off changes the measurements to

    acc' = R_off^T @ acc          R' = R @ R_off

which is applied to the raw (un-normalized) test windows before the training normalization.

Two settings are reported:
  * fixed offset: one rotation of angle theta about a given axis (x / y / z) or a random axis,
    applied to every test window (a constant mis-mounting);
  * per-trial offset ("re-donning" proxy): each test trial gets its own random axis and an
    angle drawn uniformly from [0, theta], emulating a different mounting at every donning.

Metrics follow evaluate_checkpoint.py: per-user NRMSE / MAE / R2 / PCC, then mean +/- std
across held-out users (same aggregation as Table 1).

Example:
  python evaluate_rotation.py --checkpoint <ckpt.pt> --precomputed-root <cache> \
      --protocol fixed --dataset pimforce --output rotation_pimforce.json --device cuda
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nism import NiSM, NiSMConfig
from nism.precomputed import load_precomputed_fixed_datasets, load_precomputed_loso_datasets
from train import json_safe, make_loader, metrics, predictions


def axis_angle_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    t = math.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def random_axis(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


class RotatedIMU(Dataset):
    """Wrap a precomputed test set and apply a mounting offset to x_imu."""

    def __init__(self, base, mean, std, normalization_applied: bool, offset_fn):
        self.base = base
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-8)
        self.offset_fn = offset_fn  # (index, trial_index) -> 3x3 np array or None

    def _trial(self, index):
        # trial_index is popped by the base dataset, so read it from the raw chunk
        b = self.base
        if getattr(b, "trial_index_all", None) is not None:  # preloaded dataset
            return int(b.trial_index_all[index])
        cid = bisect.bisect_right(b.cumulative, index)
        prev = 0 if cid == 0 else b.cumulative[cid - 1]
        data = b._chunk(cid)
        return int(data["trial_index"][index - prev]) if "trial_index" in data else index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        r_off = self.offset_fn(index, self._trial(index))
        item = self.base[index]
        if r_off is None:
            return item
        x = item["x_imu"].float()
        raw = x * self.std + self.mean            # back to physical units
        r_off_t = torch.as_tensor(r_off, dtype=torch.float32)
        acc = raw[:, :3] @ r_off_t                 # (R_off^T acc)^T = acc^T R_off
        rot = raw[:, 3:12].reshape(-1, 3, 3) @ r_off_t
        raw = torch.cat([acc, rot.reshape(-1, 9)], dim=1)
        item = dict(item)
        item["x_imu"] = ((raw - self.mean) / self.std).to(item["x_imu"].dtype)
        return item


def summarize(y_pred, y_true, user_ids):
    per_user = {f"User{int(u)}": metrics(y_pred[user_ids == u], y_true[user_ids == u]) for u in sorted(np.unique(user_ids))}
    out = {}
    for name in ("nrmse", "mae", "r2", "pcc"):
        vals = [v[name] for v in per_user.values()]
        out[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return out, per_user


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--precomputed-root", type=Path, required=True)
    p.add_argument("--protocol", choices=["fixed", "loso"], default="fixed")
    p.add_argument("--dataset", choices=["pimforce", "self_collected"], required=True)
    p.add_argument("--test-user")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--angles", type=float, nargs="+", default=[0, 10, 20, 30, 45])
    p.add_argument("--axes", nargs="+", default=["x", "y", "z", "random"])
    p.add_argument("--redon-max-angles", type=float, nargs="+", default=[15, 30])
    p.add_argument("--seeds", type=int, default=3, help="repeats for random-axis / per-trial settings")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.protocol == "loso":
        _, _, test_set, metadata = load_precomputed_loso_datasets(args.precomputed_root, args.test_user)
    else:
        _, _, test_set, metadata = load_precomputed_fixed_datasets(args.precomputed_root, args.dataset)
    norm = metadata["normalization"]
    mean, std = norm["imu_mean"], norm["imu_std"]

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NiSM(NiSMConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)

    def run(offset_fn):
        ds = RotatedIMU(test_set, mean, std, bool(norm.get("applied", False)), offset_fn)
        loader = make_loader(ds, args.batch_size, args.num_workers, seed=2026)
        y_pred, y_true, users = predictions(model, loader, device)
        return summarize(y_pred, y_true, users)

    results = {"checkpoint": str(args.checkpoint), "dataset": args.dataset, "fixed_offset": {}, "per_trial_offset": {}}
    base, _ = run(lambda i, it: None)
    results["clean"] = base
    print("clean", json.dumps(base))

    unit = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    for axis in args.axes:
        for angle in args.angles:
            if angle == 0:
                continue
            reps = []
            for s in range(args.seeds if axis == "random" else 1):
                ax = random_axis(np.random.default_rng(1000 + s)) if axis == "random" else unit[axis]
                r = axis_angle_matrix(ax, angle)
                summary, _ = run(lambda i, it, r=r: r)
                reps.append(summary)
            agg = {k: float(np.mean([rr[k]["mean"] for rr in reps])) for k in ("nrmse", "mae", "r2", "pcc")}
            results["fixed_offset"][f"{axis}_{angle:g}"] = {"mean_over_seeds": agg, "runs": reps}
            print("fixed", axis, angle, json.dumps(agg))

    for max_angle in args.redon_max_angles:
        reps = []
        for s in range(args.seeds):
            cache = {}
            def fn(i, trial, s=s, max_angle=max_angle, cache=cache):
                if trial not in cache:
                    rng = np.random.default_rng(10_000 * (s + 1) + trial)
                    cache[trial] = axis_angle_matrix(random_axis(rng), rng.uniform(0, max_angle))
                return cache[trial]
            summary, _ = run(fn)
            reps.append(summary)
        agg = {k: float(np.mean([rr[k]["mean"] for rr in reps])) for k in ("nrmse", "mae", "r2", "pcc")}
        results["per_trial_offset"][f"max{max_angle:g}"] = {"mean_over_seeds": agg, "runs": reps}
        print("per-trial", max_angle, json.dumps(agg))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
