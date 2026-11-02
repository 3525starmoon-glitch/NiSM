"""Memory-mapped arrays, train-only scaling, and split rules for PiMForce Dense."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


FIELDS = ("emg_dense", "pose_voxels", "pressure")
EPSILON = 1e-8


def numeric_user(value: str | int) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"User identifier has no numeric component: {value!r}")
    return int(match.group())


class DenseArrayStore:
    """Read one row-aligned NumPy array per dense modality."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        metadata_path = self.root / "metadata.npz"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Dense metadata: {metadata_path}")
        with np.load(metadata_path) as metadata:
            self.user_ids = np.asarray(metadata["user_id"], dtype=np.int64)
            self.session_ids = np.asarray(metadata["session_id"], dtype=np.int64)
            self.sample_ids = np.asarray(
                metadata["sample_id"]
                if "sample_id" in metadata.files
                else np.arange(self.user_ids.size),
                dtype=np.int64,
            )
        if self.user_ids.ndim != 1 or not (
            self.user_ids.shape == self.session_ids.shape == self.sample_ids.shape
        ):
            raise ValueError("metadata user_id, session_id, and sample_id must be aligned 1-D arrays")
        self.arrays: dict[str, np.ndarray] = {}
        for field in FIELDS:
            path = self.root / f"{field}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Dense array: {path}")
            array = np.load(path, mmap_mode="r")
            if len(array) != len(self.user_ids):
                raise ValueError(f"{path.name} has {len(array)} rows; expected {len(self.user_ids)}")
            self.arrays[field] = array
        self.validate()

    def validate(self) -> None:
        emg = self.arrays["emg_dense"]
        pose = self.arrays["pose_voxels"]
        pressure = self.arrays["pressure"]
        if emg.ndim != 3 or emg.shape[1] != 8 or emg.shape[2] < 480:
            raise ValueError("emg_dense.npy must have shape [N, 8, samples] with samples >= 480")
        if pose.ndim != 5 or pose.shape[1:] != (21, 48, 48, 48):
            raise ValueError("pose_voxels.npy must have shape [N, 21, 48, 48, 48]")
        if pressure.ndim != 2 or pressure.shape[1] != 5:
            raise ValueError("pressure.npy must have shape [N, 5]")
        users = set(map(int, np.unique(self.user_ids)))
        if not set(range(1, 22)).issubset(users):
            raise ValueError("PiMForce Dense data must contain User1 through User21")
        sessions = set(map(int, np.unique(self.session_ids)))
        if not {1, 2, 3}.issubset(sessions):
            raise ValueError("PiMForce Dense data must contain Session1 through Session3")


def select_indices(
    store: DenseArrayStore,
    users: Sequence[int],
    sessions: Sequence[int],
    fraction: float = 1.0,
    seed: int = 123,
) -> NDArray[np.int64]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    indices = np.flatnonzero(
        np.isin(store.user_ids, np.asarray(users, dtype=np.int64))
        & np.isin(store.session_ids, np.asarray(sessions, dtype=np.int64))
    ).astype(np.int64)
    if indices.size == 0:
        raise ValueError(f"No rows found for users={list(users)} sessions={list(sessions)}")
    if fraction < 1:
        generator = np.random.default_rng(seed)
        count = max(1, int(round(indices.size * fraction)))
        indices = np.sort(generator.choice(indices, size=count, replace=False))
    return indices


@dataclass(frozen=True)
class DenseSplit:
    protocol: str
    train_users: list[int]
    validation_users: list[int]
    test_users: list[int]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]

    def metadata(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "train_users": [f"User{user}" for user in self.train_users],
            "validation_users": [f"User{user}" for user in self.validation_users],
            "test_users": [f"User{user}" for user in self.test_users],
            "training_sessions": [1],
            "validation_sessions": [1] if self.validation_users else [],
            "test_sessions": [1, 2, 3],
            "training_sample_fraction": 0.5,
            "normalization_fit_users": [f"User{user}" for user in self.train_users],
            "validation_rule": (
                "closest_non_test_numeric_user_id_lower_id_tiebreak"
                if self.protocol == "loso"
                else "none"
            ),
        }


def build_split(
    store: DenseArrayStore,
    manifest_dir: Path,
    protocol: str,
    test_user: str | None,
    train_user_count: int,
    seed: int,
) -> DenseSplit:
    if protocol == "fixed":
        manifest_path = manifest_dir / (
            "pimforce_figure3.json" if train_user_count != 17 else "pimforce_fixed.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        test_users = [numeric_user(user) for user in manifest["test_users"]]
        if test_users != [4, 9, 15, 20]:
            raise ValueError("PiMForce fixed test users must be User4, User9, User15, and User20")
        if train_user_count == 17:
            train_users = [numeric_user(user) for user in manifest["train_users"]]
        else:
            allowed = [17, 14, 11, 8, 5, 2]
            if train_user_count not in allowed:
                raise ValueError(f"Figure 3 train-user count must be one of {allowed}")
            order = [numeric_user(user) for user in manifest["nested_training_user_order"]]
            train_users = order[:train_user_count]
        validation_users: list[int] = []
    elif protocol == "loso":
        if test_user is None:
            raise ValueError("--test-user is required for LOSO")
        test_id = numeric_user(test_user)
        if test_id not in range(1, 22):
            raise ValueError("PiMForce LOSO test user must be User1 through User21")
        test_users = [test_id]
        candidates = [user for user in range(1, 22) if user != test_id]
        validation_users = [min(candidates, key=lambda user: (abs(user - test_id), user))]
        train_users = [
            user for user in candidates if user not in validation_users
        ]
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    if set(train_users) & set(validation_users + test_users):
        raise ValueError("Dense train, validation, and test users must be disjoint")
    train_indices = select_indices(store, train_users, [1], fraction=0.5, seed=seed)
    validation_indices = (
        select_indices(store, validation_users, [1])
        if validation_users
        else np.empty(0, dtype=np.int64)
    )
    return DenseSplit(
        protocol=protocol,
        train_users=train_users,
        validation_users=validation_users,
        test_users=test_users,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


@dataclass(frozen=True)
class MinMaxScale:
    offset: NDArray[np.float32]
    scale: NDArray[np.float32]

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        offset = torch.as_tensor(self.offset, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.scale, device=value.device, dtype=value.dtype)
        return (value - offset) / scale.clamp_min(EPSILON)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        offset = torch.as_tensor(self.offset, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.scale, device=value.device, dtype=value.dtype)
        return value * scale.clamp_min(EPSILON) + offset


class DenseNormalizer:
    """Min-max statistics fitted exclusively on the selected training rows."""

    def __init__(self, emg: MinMaxScale, pressure: MinMaxScale) -> None:
        self.emg = emg
        self.pressure = pressure

    @classmethod
    def fit(cls, store: DenseArrayStore, indices: NDArray[np.int64]) -> "DenseNormalizer":
        if indices.size == 0:
            raise ValueError("Cannot fit normalization on an empty training split")
        emg_min = np.full((8, 1), np.inf, dtype=np.float64)
        emg_max = np.full((8, 1), -np.inf, dtype=np.float64)
        pressure_min = np.full(5, np.inf, dtype=np.float64)
        pressure_max = np.full(5, -np.inf, dtype=np.float64)
        for start in range(0, indices.size, 256):
            rows = indices[start : start + 256]
            emg = np.asarray(store.arrays["emg_dense"][rows], dtype=np.float64)
            pressure = np.asarray(store.arrays["pressure"][rows], dtype=np.float64)
            emg_min = np.minimum(emg_min, emg.min(axis=(0, 2)).reshape(8, 1))
            emg_max = np.maximum(emg_max, emg.max(axis=(0, 2)).reshape(8, 1))
            pressure_min = np.minimum(pressure_min, pressure.min(axis=0))
            pressure_max = np.maximum(pressure_max, pressure.max(axis=0))
        return cls(
            MinMaxScale(
                emg_min.astype(np.float32),
                np.maximum(emg_max - emg_min, EPSILON).astype(np.float32),
            ),
            MinMaxScale(
                pressure_min.astype(np.float32),
                np.maximum(pressure_max - pressure_min, EPSILON).astype(np.float32),
            ),
        )

    def state_dict(self) -> dict[str, NDArray[np.float32]]:
        return {
            "emg_offset": self.emg.offset,
            "emg_scale": self.emg.scale,
            "pressure_offset": self.pressure.offset,
            "pressure_scale": self.pressure.scale,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DenseNormalizer":
        return cls(
            MinMaxScale(
                np.asarray(state["emg_offset"], dtype=np.float32),
                np.asarray(state["emg_scale"], dtype=np.float32),
            ),
            MinMaxScale(
                np.asarray(state["pressure_offset"], dtype=np.float32),
                np.asarray(state["pressure_scale"], dtype=np.float32),
            ),
        )


class PiMForceDenseDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        store: DenseArrayStore,
        indices: NDArray[np.int64],
        normalizer: DenseNormalizer,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[item])
        emg = torch.from_numpy(
            np.asarray(self.store.arrays["emg_dense"][row], dtype=np.float32).copy()
        )
        pose = torch.from_numpy(
            np.asarray(self.store.arrays["pose_voxels"][row], dtype=np.float32).copy()
        )
        pressure = torch.from_numpy(
            np.asarray(self.store.arrays["pressure"][row], dtype=np.float32).copy()
        )
        return {
            "emg_dense": self.normalizer.emg.transform(emg),
            "pose_voxels": pose,
            "pressure": self.normalizer.pressure.transform(pressure),
            "user_id": torch.tensor(int(self.store.user_ids[row]), dtype=torch.long),
            "sample_id": torch.tensor(int(self.store.sample_ids[row]), dtype=torch.long),
        }
