"""Create fixed-split or LOSO NumPy caches from synchronized raw trials."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nism.encoders import EMG_FEATURE_NAMES


ACC_COLUMNS = [
    "thumb_proximal_mid_acc_x",
    "thumb_proximal_mid_acc_y",
    "thumb_proximal_mid_acc_z",
]
ROT_COLUMNS = [f"rot_thumb_pip_r{i}c{j}" for i in range(1, 4) for j in range(1, 4)]
FORCE_COLUMNS = [f"force_{i}" for i in range(1, 6)]
EMG_COLUMN = "Sensor0_sEMG"
HAND_SHAPE_DIM = 63
FIGURE3_TRAIN_USER_COUNTS = (17, 14, 11, 8, 5, 2)
FIGURE3_PIMFORCE_USER_ORDER = (
    "User16", "User14", "User8", "User2", "User3", "User1", "User12",
    "User19", "User11", "User18", "User17", "User5", "User7", "User21",
    "User10", "User6", "User13",
)


@dataclass(frozen=True)
class Trial:
    user: str
    session: str
    trial_id: str
    source_path: Path
    source_format: str
    emg_channel: int = 0
    force_path: Path | None = None
    emg_path: Path | None = None
    acc_path: Path | None = None
    rot_path: Path | None = None
    pose_path: Path | None = None
    emg_feat_path: Path | None = None


@dataclass
class LoadedTrial:
    force_t: np.ndarray
    force: np.ndarray
    emg_t: np.ndarray
    emg: np.ndarray
    imu_t: np.ndarray
    imu: np.ndarray
    pose_t: np.ndarray
    pose: np.ndarray
    emg_feat_t: np.ndarray | None = None
    emg_feat: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["fixed", "loso"], required=True)
    parser.add_argument("--dataset", choices=["pimforce", "self_collected"], default="pimforce")
    parser.add_argument("--test-user")
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--window-length", type=int, choices=[10, 20, 30], default=30)
    parser.add_argument("--stride", type=int, choices=[2], default=2)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pimforce-emg-channel",
        type=int,
        default=0,
        help="Zero-based EMG channel selected from each PiMForce data_{task_id}.npy file",
    )
    parser.add_argument("--train-user-count", type=int)
    parser.add_argument("--train-users", help="Comma-separated explicit training-user IDs")
    parser.add_argument(
        "--figure3-all",
        action="store_true",
        help="Create nested PiMForce fixed caches for 17, 14, 11, 8, 5, and 2 training users",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=["users", "within"],
        default="users",
        help="users: held-out test users (paper). within: every trial of every user is split in time; "
             "the first (1 - test fraction) trains, the rest tests (no window crosses the cut)",
    )
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="within split: segment just before the test segment used for checkpoint selection")
    return parser.parse_args()


def numeric_id(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise ValueError(f"No numeric ID in {value!r}")
    return int(match.group())


def pose_path_for(force_path: Path, suffix: str) -> Path | None:
    candidates = (
        force_path.parent / f"hand_pose_{suffix}",
        force_path.parent / f"pose_{suffix}",
        force_path.parent / f"keypoints_{suffix}",
    )
    return next((path for path in candidates if path.is_file()), None)


def discover_trials(root: Path, dataset: str, pimforce_emg_channel: int = 0) -> list[Trial]:
    if dataset == "pimforce":
        if pimforce_emg_channel < 0:
            raise ValueError("--pimforce-emg-channel must be non-negative")
        trials = []
        for data_path in sorted(root.glob("User*/Session*/data_*.npy")):
            relative = data_path.relative_to(root)
            trial_id = data_path.stem.split("data_", 1)[1]
            trials.append(
                Trial(
                    relative.parts[0],
                    relative.parts[1],
                    trial_id,
                    data_path,
                    "pimforce_npy",
                    pimforce_emg_channel,
                )
            )
        if trials:
            return trials

    trials = []
    for force_path in sorted(root.glob("User*/Session*/force_*.csv")):
        suffix = force_path.name.split("force_", 1)[1]
        trial_id = Path(suffix).stem
        emg_path = force_path.parent / f"emg_{suffix}"
        acc_path = force_path.parent / f"proximal_acc_{suffix}"
        rot_path = force_path.parent / f"rotation_{suffix}"
        pose_path = pose_path_for(force_path, suffix)
        if not all(path.is_file() for path in (emg_path, acc_path, rot_path)):
            continue
        relative = force_path.relative_to(root)
        trials.append(
            Trial(
                relative.parts[0],
                relative.parts[1],
                trial_id,
                force_path,
                "synchronized_csv",
                0,
                force_path,
                emg_path,
                acc_path,
                rot_path,
                pose_path,
                emg_feat_path if (emg_feat_path := force_path.parent / f"emg_feat_{suffix}").is_file() else None,
            )
        )
    if not trials:
        raise FileNotFoundError(f"No complete trials found below {root}")
    return trials


def read_columns(path: Path, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["timestamp", *columns])
    values = frame[columns].to_numpy(dtype=np.float32)
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    finite = np.isfinite(timestamps) & np.isfinite(values).all(axis=1)
    return timestamps[finite], values[finite]


def read_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError(f"Pose file has no timestamp column: {path}")
    value_columns = [column for column in frame.columns if column != "timestamp"]
    if len(value_columns) != HAND_SHAPE_DIM:
        raise ValueError(f"Expected 63 pose coordinates in {path}, found {len(value_columns)}")
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    values = frame[value_columns].to_numpy(dtype=np.float32)
    finite = np.isfinite(timestamps) & np.isfinite(values).all(axis=1)
    return timestamps[finite], values[finite]


def load_numpy_mapping(path: Path) -> dict[str, object]:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        try:
            mapping = {key: raw[key] for key in raw.files}
        finally:
            raw.close()
        return mapping
    if isinstance(raw, np.ndarray) and raw.dtype.names:
        return {name: raw[name] for name in raw.dtype.names}
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        value = raw.reshape(()).item()
        if isinstance(value, dict):
            return value
    if isinstance(raw, dict):
        return raw
    raise ValueError(
        f"{path} must contain a dictionary-like NumPy payload with named synchronized modalities"
    )


def mapping_value(mapping: dict[str, object], aliases: tuple[str, ...], required: bool = True):
    lookup = {str(key).lower(): key for key in mapping}
    for alias in aliases:
        key = lookup.get(alias.lower())
        if key is not None:
            return mapping[key]
    if required:
        raise KeyError(f"Required PiMForce field is missing; expected one of {aliases}")
    return None


def time_major(values, feature_dims: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim == 2 and array.shape[0] in feature_dims and array.shape[1] not in feature_dims:
        array = array.T
    if array.ndim < 2:
        raise ValueError(f"PiMForce {name} must be a time-major array, got {array.shape}")
    return array


def modality_timestamps(
    mapping: dict[str, object],
    aliases: tuple[str, ...],
    common,
    length: int,
    name: str,
) -> np.ndarray | None:
    values = mapping_value(mapping, aliases, required=False)
    if values is None:
        values = common
    if values is None:
        return None
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(timestamps) != length:
        raise ValueError(
            f"PiMForce {name} timestamps contain {len(timestamps)} rows but the modality contains {length}"
        )
    return timestamps


def load_pimforce_trial(trial: Trial) -> LoadedTrial:
    mapping = load_numpy_mapping(trial.source_path)
    nested = mapping_value(mapping, ("data", "modalities"), required=False)
    if isinstance(nested, dict):
        mapping = {**mapping, **nested}

    imu_raw = mapping_value(mapping, ("imu", "thumb_imu"), required=False)
    if imu_raw is not None:
        imu = time_major(imu_raw, (12,), "IMU")
        imu = imu.reshape(imu.shape[0], -1)
        if imu.shape[1] < 12:
            raise ValueError(f"PiMForce IMU requires at least 12 channels, got {imu.shape}")
        imu = imu[:, :12]
    else:
        acc = time_major(
            mapping_value(mapping, ("acc", "acceleration", "imu_acc", "thumb_acc")),
            (3,),
            "acceleration",
        )
        rotation = time_major(
            mapping_value(mapping, ("rotation", "rot", "orientation", "rotation_matrix")),
            (3, 9),
            "rotation",
        )
        rotation = rotation.reshape(rotation.shape[0], -1)
        if acc.shape[1] != 3 or rotation.shape[1] != 9 or len(acc) != len(rotation):
            raise ValueError(
                f"PiMForce acceleration/rotation must be [T,3] and [T,9], got {acc.shape} and {rotation.shape}"
            )
        imu = np.concatenate([acc, rotation], axis=1)

    emg_all = time_major(
        mapping_value(mapping, ("emg", "semg", "s_emg", "muscle")),
        tuple(range(1, 33)),
        "EMG",
    )
    if trial.emg_channel >= emg_all.shape[1]:
        raise ValueError(
            f"Requested PiMForce EMG channel {trial.emg_channel}, but {trial.source_path} has "
            f"only {emg_all.shape[1]} channels"
        )
    emg = emg_all[:, trial.emg_channel]

    force = time_major(
        mapping_value(mapping, ("force", "pressure", "finger_pressure", "contact_pressure")),
        tuple(range(5, 129)),
        "pressure",
    )
    force = force.reshape(force.shape[0], -1)
    if force.shape[1] < 5:
        raise ValueError(f"PiMForce pressure requires at least five finger channels, got {force.shape}")
    force = force[:, :5]

    pose = time_major(
        mapping_value(mapping, ("hand_pose", "pose", "pose3d", "joints", "hand_joints")),
        (21, 63),
        "3D hand pose",
    ).reshape(-1, HAND_SHAPE_DIM)

    common_t = mapping_value(mapping, ("timestamp", "timestamps", "time"), required=False)
    force_t = modality_timestamps(
        mapping, ("force_timestamp", "force_timestamps", "pressure_timestamp", "pressure_timestamps"),
        common_t, len(force), "pressure",
    )
    emg_t = modality_timestamps(
        mapping, ("emg_timestamp", "emg_timestamps", "semg_timestamp", "semg_timestamps"),
        common_t, len(emg), "EMG",
    )
    imu_t = modality_timestamps(
        mapping, ("imu_timestamp", "imu_timestamps"), common_t, len(imu), "IMU",
    )
    pose_t = modality_timestamps(
        mapping, ("pose_timestamp", "pose_timestamps", "hand_pose_timestamp", "hand_pose_timestamps"),
        common_t, len(pose), "3D hand pose",
    )
    timestamps = (force_t, emg_t, imu_t, pose_t)
    if all(value is None for value in timestamps):
        lengths = {len(force), len(emg), len(imu), len(pose)}
        if len(lengths) != 1:
            raise ValueError(
                f"PiMForce modalities with different lengths require modality timestamps in {trial.source_path}"
            )
        shared = np.arange(len(force), dtype=np.float64)
        force_t = emg_t = imu_t = pose_t = shared
    elif any(value is None for value in timestamps):
        raise ValueError(f"All PiMForce modalities must provide timestamps when one modality does: {trial.source_path}")

    assert force_t is not None and emg_t is not None and imu_t is not None and pose_t is not None
    return LoadedTrial(force_t, force, emg_t, emg, imu_t, imu, pose_t, pose)


def load_trial(trial: Trial) -> LoadedTrial:
    if trial.source_format == "pimforce_npy":
        return load_pimforce_trial(trial)
    if not all(path is not None for path in (trial.force_path, trial.emg_path, trial.acc_path, trial.rot_path)):
        raise ValueError(f"Incomplete synchronized CSV trial: {trial.source_path}")
    assert trial.force_path is not None and trial.emg_path is not None
    assert trial.acc_path is not None and trial.rot_path is not None
    force_t, force = read_columns(trial.force_path, FORCE_COLUMNS)
    emg_t, emg = read_columns(trial.emg_path, [EMG_COLUMN])
    acc_t, acc = read_columns(trial.acc_path, ACC_COLUMNS)
    rot_t, rot = read_columns(trial.rot_path, ROT_COLUMNS)
    if trial.pose_path is None:
        pose_t, pose = np.empty(0, dtype=np.float64), np.empty((0, HAND_SHAPE_DIM), dtype=np.float32)
    else:
        pose_t, pose = read_pose(trial.pose_path)
    if not np.array_equal(acc_t, rot_t):
        common_start = max(acc_t[0], rot_t[0])
        common_stop = min(acc_t[-1], rot_t[-1])
        target_t = acc_t[(acc_t >= common_start) & (acc_t <= common_stop)]
        rot = previous_rows(rot_t, rot, target_t)
        acc = previous_rows(acc_t, acc, target_t)
        imu_t = target_t
    else:
        imu_t = acc_t
    loaded = LoadedTrial(force_t, force, emg_t, emg[:, 0], imu_t, np.concatenate([acc, rot], axis=1), pose_t, pose)
    if trial.emg_feat_path is not None:
        frame = pd.read_csv(trial.emg_feat_path)
        loaded.emg_feat_t = frame["timestamp"].to_numpy(dtype=np.float64)
        loaded.emg_feat = frame[list(EMG_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    return loaded


def nearest_rows(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_t, target_t, side="left").clip(0, len(source_t) - 1)
    left = (right - 1).clip(0, len(source_t) - 1)
    choose_left = np.abs(target_t - source_t[left]) <= np.abs(source_t[right] - target_t)
    indices = np.where(choose_left, left, right)
    return values[indices]


def previous_rows(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Select the latest sample at or before each target timestamp."""
    indices = (np.searchsorted(source_t, target_t, side="right") - 1).clip(0, len(source_t) - 1)
    return values[indices]


def emg_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 3:
        raise ValueError("An EMG window must contain at least three samples")
    absolute = np.abs(x)
    diff = np.diff(x)
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2 / x.size
    zero_crossing_rate = np.count_nonzero(
        (x[:-1] * x[1:] < 0) & (np.abs(diff) > 1e-8)
    ) / max(x.size - 1, 1)
    peak_frequency_index = int(np.argmax(spectrum))
    return np.asarray([
        absolute.mean(),
        np.sqrt(np.mean(x * x)),
        zero_crossing_rate,
        np.abs(diff).sum(),
        x.var(),
        absolute.sum(),
        x.std(),
        spectrum.mean(),
        peak_frequency_index,
        spectrum.sum(),
    ], dtype=np.float32)


def window_starts(length: int, frames: int, stride: int) -> range:
    return range(0, max(length - frames + 1, 0), stride)


def make_window(loaded: LoadedTrial, start: int, frames: int, include_pose: bool = True) -> dict[str, np.ndarray]:
    stop = start + frames
    force_t = loaded.force_t[start:stop]
    if len(force_t) != frames:
        raise ValueError("Incomplete force window")
    imu = previous_rows(loaded.imu_t, loaded.imu, force_t).astype(np.float32)
    emg_mask = (loaded.emg_t >= force_t[0]) & (loaded.emg_t <= force_t[-1])
    if emg_mask.sum() < 3:
        center = int(np.searchsorted(loaded.emg_t, force_t[-1]))
        lo, hi = max(center - 3, 0), min(center + 3, len(loaded.emg))
        emg = loaded.emg[lo:hi]
    else:
        emg = loaded.emg[emg_mask]
    result = {
        "x_imu": imu,
        "emg_sequence": previous_rows(
            loaded.emg_t, loaded.emg[:, None], force_t
        ).astype(np.float32),
        # frame-level features averaged over the window when available,
        # otherwise the 10 features computed once on the raw EMG samples inside the window
        "emg_features": (previous_rows(loaded.emg_feat_t, loaded.emg_feat, force_t).mean(axis=0).astype(np.float32)
                         if loaded.emg_feat is not None else emg_features(emg)),
        "force": loaded.force[start:stop].astype(np.float32),
    }
    if include_pose:
        if not len(loaded.pose_t):
            raise ValueError("Training/validation window has no 63D hand-pose supervision")
        result["hand_pose"] = nearest_rows(loaded.pose_t, loaded.pose, force_t[-1:])[0].astype(np.float32)
    return result


def record_payload(trial: Trial, root: Path, include_action: bool = True) -> dict[str, object]:
    payload = {
        "user": trial.user,
        "session": trial.session,
        "trial_id": trial.trial_id,
        "force_path": str(trial.source_path.relative_to(root)),
        "source_format": trial.source_format,
    }
    if include_action:
        payload["action_label"] = trial.trial_id
    return payload


def closest_user(users: list[str], test_user: str) -> str:
    return min(users, key=lambda user: (abs(numeric_id(user) - numeric_id(test_user)), numeric_id(user)))


def split_trials(trials: list[Trial], protocol: str, dataset: str, test_user: str | None) -> tuple[list[Trial], list[Trial], list[Trial], str]:
    users = sorted({trial.user for trial in trials}, key=numeric_id)
    if protocol == "fixed":
        test_ids = {4, 9, 15, 20} if dataset == "pimforce" else {numeric_id(user) for user in users[-4:]}
        train = [trial for trial in trials if numeric_id(trial.user) not in test_ids]
        test = [trial for trial in trials if numeric_id(trial.user) in test_ids]
        val: list[Trial] = []
        split_name = f"fixed_{dataset}"
    else:
        if not test_user:
            raise ValueError("--test-user is required for LOSO cache creation")
        remaining = [user for user in users if user != test_user]
        val_user = closest_user(remaining, test_user)
        train = [trial for trial in trials if trial.user not in {test_user, val_user}]
        val = [trial for trial in trials if trial.user == val_user]
        test = [trial for trial in trials if trial.user == test_user]
        split_name = f"fold_{test_user}"
    if dataset == "pimforce":
        train = [trial for trial in train if numeric_id(trial.session) == 1]
        val = [trial for trial in val if numeric_id(trial.session) == 1]
    if not train or not test or (protocol == "loso" and not val):
        raise ValueError(f"Empty required split for {split_name}")
    return train, val, test, split_name


def descriptors(trials: list[Trial], frames: int, stride: int) -> list[tuple[int, int]]:
    result = []
    for index, trial in enumerate(trials):
        if trial.source_format == "pimforce_npy":
            force_rows = len(load_trial(trial).force_t)
        else:
            assert trial.force_path is not None
            with trial.force_path.open(encoding="utf-8") as handle:
                force_rows = sum(1 for _ in handle) - 1
        result.extend((index, start) for start in window_starts(force_rows, frames, stride))
    return result


def half_descriptors(items: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(items))
    count = len(items) // 2
    return [items[int(index)] for index in np.sort(order[:count])]


def normalization(trials: list[Trial], items: list[tuple[int, int]], frames: int) -> dict[str, object]:
    sums = {
        "imu": np.zeros(12), "imu2": np.zeros(12),
        "emg": np.zeros(10), "emg2": np.zeros(10),
        "emg_raw": np.zeros(1), "emg_raw2": np.zeros(1),
        "pose": np.zeros(63), "pose2": np.zeros(63),
    }
    counts = {"imu": 0, "emg": 0, "emg_raw": 0, "pose": 0}
    force_min = np.full(5, np.inf)
    force_max = np.full(5, -np.inf)
    loaded_index = -1
    loaded = None
    for count, (trial_index, start) in enumerate(items, 1):
        if count % 5000 == 0 or count == len(items):
            print(f"[cache] normalization stats: {count}/{len(items)} windows", flush=True)
        if trial_index != loaded_index:
            loaded = load_trial(trials[trial_index])
            loaded_index = trial_index
        assert loaded is not None
        item = make_window(loaded, start, frames, include_pose=True)
        imu = item["x_imu"].astype(np.float64)
        emg = item["emg_features"].astype(np.float64)
        emg_raw = item["emg_sequence"].astype(np.float64)
        pose = item["hand_pose"].astype(np.float64)
        force = item["force"].astype(np.float64)
        sums["imu"] += imu.sum(axis=0); sums["imu2"] += np.square(imu).sum(axis=0); counts["imu"] += len(imu)
        sums["emg"] += emg; sums["emg2"] += np.square(emg); counts["emg"] += 1
        sums["emg_raw"] += emg_raw.sum(axis=0); sums["emg_raw2"] += np.square(emg_raw).sum(axis=0); counts["emg_raw"] += len(emg_raw)
        sums["pose"] += pose; sums["pose2"] += np.square(pose); counts["pose"] += 1
        force_min = np.minimum(force_min, force.min(axis=0)); force_max = np.maximum(force_max, force.max(axis=0))
    def mean_std(name: str) -> tuple[list[float], list[float]]:
        mu = sums[name] / max(counts[name], 1)
        variance = sums[f"{name}2"] / max(counts[name], 1) - np.square(mu)
        return mu.astype(np.float32).tolist(), np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32).tolist()
    imu_mean, imu_std = mean_std("imu")
    emg_mean, emg_std = mean_std("emg")
    emg_raw_mean, emg_raw_std = mean_std("emg_raw")
    pose_mean, pose_std = mean_std("pose")
    return {
        "applied": False,
        "imu_mean": imu_mean, "imu_std": imu_std,
        "emg_feature_mean": emg_mean, "emg_feature_std": emg_std,
        "emg_raw_mean": emg_raw_mean, "emg_raw_std": emg_raw_std,
        "force_min": force_min.astype(np.float32).tolist(), "force_max": force_max.astype(np.float32).tolist(),
        "hand_shape_mean": pose_mean, "hand_shape_std": pose_std,
        "fit_users": sorted({trial.user for trial in trials}, key=numeric_id),
    }


def write_split(split_dir: Path, split: str, trials: list[Trial], items: list[tuple[int, int]], frames: int, chunk_size: int) -> None:
    chunks = []
    buffers: dict[str, list[np.ndarray]] = {}
    chunk_id = 0
    loaded_index = -1
    loaded = None
    def flush() -> None:
        nonlocal chunk_id
        if not buffers:
            return
        arrays = {key: np.stack(values) for key, values in buffers.items()}
        filename = f"{split}_chunk_{chunk_id:04d}.npz"
        np.savez_compressed(split_dir / filename, **arrays)
        chunks.append({"file": filename, "num_samples": len(next(iter(buffers.values())))})
        buffers.clear(); chunk_id += 1
    for count, (trial_index, start) in enumerate(items, 1):
        if count % 5000 == 0 or count == len(items):
            print(f"[cache] writing {split}: {count}/{len(items)} windows", flush=True)
        if trial_index != loaded_index:
            loaded = load_trial(trials[trial_index]); loaded_index = trial_index
        assert loaded is not None
        window = make_window(loaded, start, frames, include_pose=split != "test")
        window["trial_index"] = np.asarray(trial_index, dtype=np.int64)
        window["user_id"] = np.asarray(numeric_id(trials[trial_index].user), dtype=np.int64)
        for key, value in window.items():
            buffers.setdefault(key, []).append(np.asarray(value))
        if len(next(iter(buffers.values()))) >= chunk_size:
            flush()
    flush()
    index = {"split": split, "chunks": chunks, "num_samples": len(items)}
    (split_dir / f"{split}_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def trial_rows(trial: Trial) -> int:
    if trial.source_format == "pimforce_npy":
        return len(load_trial(trial).force_t)
    assert trial.force_path is not None
    with trial.force_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def build_within_cache(args: argparse.Namespace, trials: list[Trial]) -> Path:
    """Within-user temporal split: all users in train and test, disjoint time segments per trial."""
    trials = sorted(trials, key=lambda t: (numeric_id(t.user), numeric_id(t.session), numeric_id(t.trial_id)))
    missing_pose = [t.source_path for t in trials if t.pose_path is None]
    if missing_pose:
        raise FileNotFoundError(f"Training requires 63D hand pose; missing for {len(missing_pose)} trials")
    frames, stride = args.window_length, args.stride
    train_items, val_items, test_items = [], [], []
    for index, trial in enumerate(trials):
        rows = trial_rows(trial)
        cut = int(round(rows * (1.0 - args.test_fraction)))
        vcut = int(round(rows * (1.0 - args.test_fraction - args.val_fraction)))
        for start in window_starts(rows, frames, stride):
            if start + frames <= vcut:
                train_items.append((index, start))
            elif start >= vcut and start + frames <= cut:
                val_items.append((index, start))
            elif start >= cut:
                test_items.append((index, start))
    split_dir = args.output_root / f"fixed_{args.dataset}"
    if split_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Cache already exists: {split_dir}; pass --overwrite to replace files")
    split_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_chunk_*.npz", "*_index.json", "metadata.json"):
        for old_path in split_dir.glob(pattern):
            old_path.unlink()
    print(f"[cache] within-user split: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test windows, "
          f"{len(trials)} trials, val fraction {args.val_fraction}, test fraction {args.test_fraction}", flush=True)
    stats = normalization(trials, train_items, frames)
    write_split(split_dir, "train", trials, train_items, frames, args.chunk_size)
    if val_items:
        write_split(split_dir, "val", trials, val_items, frames, args.chunk_size)
    write_split(split_dir, "test", trials, test_items, frames, args.chunk_size)
    users = sorted({t.user for t in trials}, key=numeric_id)
    metadata = {
        "dataset": args.dataset,
        "protocol": "fixed",
        "split_mode": "within_user_temporal",
        "test_fraction": args.test_fraction,
        "val_fraction": args.val_fraction,
        "test_user": None,
        "train_users": users,
        "train_user_count": len(users),
        "val_users": [],
        "hand_shape_dim": HAND_SHAPE_DIM,
        "num_context_classes": len({t.trial_id for t in trials}),
        "emg_feature_names": list(EMG_FEATURE_NAMES),
        "emg_raw_shape": [frames, 1],
        "selected_emg_channel": 0,
        "cache_format": "numpy_npz_chunks_v1",
        "window_config": {"window_length": frames, "hop_length": stride, "num_frames": frames, "target_mode": "last"},
        "alignment": {"retrospective": True, "uses_future_frames": False, "target": "final_frame"},
        "normalization": stats,
        "full_training_sample_count": len(train_items),
        "used_training_sample_count": len(train_items),
        "checkpoint_selection": "best_validation" if val_items else "final_epoch",
        "train_trial_records": [record_payload(t, args.data_root, True) for t in trials],
        "val_trial_records": [record_payload(t, args.data_root, True) for t in trials] if val_items else [],
        "test_trial_records": [record_payload(t, args.data_root, False) for t in trials],
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split_dir


def build_cache(args: argparse.Namespace, trials: list[Trial], test_user: str | None) -> Path:
    if getattr(args, "split_mode", "users") == "within":
        return build_within_cache(args, trials)
    train_trials, val_trials, test_trials, name = split_trials(trials, args.protocol, args.dataset, test_user)
    available_train_users = sorted({trial.user for trial in train_trials}, key=numeric_id)
    if args.train_users:
        selected_train_users = {value.strip() for value in args.train_users.split(",") if value.strip()}
        unknown = selected_train_users - set(available_train_users)
        if unknown:
            raise ValueError(f"Requested training users are unavailable: {sorted(unknown)}")
    elif args.train_user_count is not None:
        if not 1 <= args.train_user_count <= len(available_train_users):
            raise ValueError(f"--train-user-count must be between 1 and {len(available_train_users)}")
        if args.protocol == "fixed" and args.dataset == "pimforce":
            unavailable = set(FIGURE3_PIMFORCE_USER_ORDER) - set(available_train_users)
            if unavailable:
                raise ValueError(f"Figure 3 user order contains unavailable users: {sorted(unavailable)}")
            selected_train_users = set(FIGURE3_PIMFORCE_USER_ORDER[:args.train_user_count])
        else:
            generator = np.random.default_rng(args.seed)
            order = generator.permutation(available_train_users).tolist()
            selected_train_users = set(order[:args.train_user_count])
    else:
        selected_train_users = set(available_train_users)
    train_trials = [trial for trial in train_trials if trial.user in selected_train_users]
    if not train_trials:
        raise ValueError("Training-user selection produced an empty training split")
    missing_pose = []
    for trial in [*train_trials, *val_trials]:
        if trial.source_format == "pimforce_npy":
            if not len(load_trial(trial).pose_t):
                missing_pose.append(trial.source_path)
        elif trial.pose_path is None:
            missing_pose.append(trial.source_path)
    if missing_pose:
        examples = ", ".join(str(path) for path in missing_pose[:3])
        raise FileNotFoundError(
            f"Training/validation requires 63D hand pose; {len(missing_pose)} trials are missing it. Examples: {examples}"
        )
    split_dir = args.output_root / name
    if split_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Cache already exists: {split_dir}; pass --overwrite to replace files")
    split_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pattern in ("*_chunk_*.npz", "*_chunk_*.pt", "*_index.json", "*_index.pt", "metadata.pt"):
            for old_path in split_dir.glob(pattern):
                old_path.unlink()
    full_train = descriptors(train_trials, args.window_length, args.stride)
    used_train = half_descriptors(full_train, args.seed) if args.dataset == "pimforce" else full_train
    val_items = descriptors(val_trials, args.window_length, args.stride)
    test_items = descriptors(test_trials, args.window_length, args.stride)
    stats = normalization(train_trials, used_train, args.window_length)
    write_split(split_dir, "train", train_trials, used_train, args.window_length, args.chunk_size)
    if val_trials:
        write_split(split_dir, "val", val_trials, val_items, args.window_length, args.chunk_size)
    write_split(split_dir, "test", test_trials, test_items, args.window_length, args.chunk_size)
    all_train_labels = sorted({trial.trial_id for trial in train_trials})
    metadata = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "test_user": test_user,
        "train_users": sorted({trial.user for trial in train_trials}, key=numeric_id),
        "train_user_count": len({trial.user for trial in train_trials}),
        "val_users": sorted({trial.user for trial in val_trials}, key=numeric_id),
        "hand_shape_dim": HAND_SHAPE_DIM,
        "num_context_classes": len(all_train_labels),
        "emg_feature_names": list(EMG_FEATURE_NAMES),
        "emg_raw_shape": [args.window_length, 1],
        "selected_emg_channel": args.pimforce_emg_channel if args.dataset == "pimforce" else 0,
        "cache_format": "numpy_npz_chunks_v1",
        "window_config": {"window_length": args.window_length, "hop_length": args.stride, "num_frames": args.window_length, "target_mode": "last"},
        "alignment": {"retrospective": True, "uses_future_frames": False, "target": "final_frame"},
        "normalization": stats,
        "full_training_sample_count": len(full_train),
        "used_training_sample_count": len(used_train),
        "checkpoint_selection": "final_epoch" if args.protocol == "fixed" else "best_validation",
        "train_trial_records": [record_payload(trial, args.data_root, True) for trial in train_trials],
        "val_trial_records": [record_payload(trial, args.data_root, True) for trial in val_trials],
        "test_trial_records": [record_payload(trial, args.data_root, False) for trial in test_trials],
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {key: metadata[key] for key in ("dataset", "protocol", "test_user", "train_users", "val_users", "train_trial_records", "val_trial_records", "test_trial_records")}
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (split_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("split", "user", "session", "trial_id", "force_path", "source_format", "action_label")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for record in metadata[f"{split}_trial_records"]:
                writer.writerow({"split": split, **record})
    return split_dir


def main() -> None:
    args = parse_args()
    trials = discover_trials(args.data_root, args.dataset, args.pimforce_emg_channel)
    users = sorted({trial.user for trial in trials}, key=numeric_id)
    if args.figure3_all:
        if args.protocol != "fixed" or args.dataset != "pimforce":
            raise ValueError("--figure3-all requires --protocol fixed --dataset pimforce")
        if args.train_user_count is not None or args.train_users:
            raise ValueError("--figure3-all cannot be combined with --train-user-count or --train-users")
        for count in FIGURE3_TRAIN_USER_COUNTS:
            count_args = argparse.Namespace(**vars(args))
            count_args.figure3_all = False
            count_args.train_user_count = count
            count_args.output_root = args.output_root / f"train_users_{count}"
            path = build_cache(count_args, trials, None)
            print(f"Created Figure 3 cache ({count} users): {path}", flush=True)
        return
    if args.protocol == "loso" and args.all_users:
        targets = users
    else:
        targets = [args.test_user]
    for test_user in targets:
        path = build_cache(args, trials, test_user)
        print(f"Created cache: {path}", flush=True)


if __name__ == "__main__":
    main()
