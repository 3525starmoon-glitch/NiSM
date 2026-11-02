from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(split_dir: Path) -> dict[str, Any]:
    json_path = split_dir / "metadata.json"
    if json_path.exists():
        return load_json(json_path)
    pt_path = split_dir / "metadata.pt"
    if pt_path.exists():
        return load_pt(pt_path)
    raise FileNotFoundError(f"Missing precomputed metadata below {split_dir}")


def split_index_exists(split_dir: Path, split: str) -> bool:
    return (split_dir / f"{split}_index.json").exists() or (split_dir / f"{split}_index.pt").exists()


class PrecomputedChunkDataset(Dataset):
    """Read NumPy cache chunks, with read-only compatibility for older tensor chunks."""

    def __init__(self, fold_dir: Path, split: str, metadata: dict[str, Any]):
        self.fold_dir = fold_dir
        json_index = fold_dir / f"{split}_index.json"
        pt_index = fold_dir / f"{split}_index.pt"
        if json_index.exists():
            index = load_json(json_index)
        elif pt_index.exists():
            index = load_pt(pt_index)
        else:
            raise FileNotFoundError(f"Missing precomputed index for split {split!r} below {fold_dir}")
        self.chunks = index["chunks"]
        self.num_samples = int(index["num_samples"])
        self.cumulative: list[int] = []
        total = 0
        for chunk in self.chunks:
            total += int(chunk["num_samples"])
            self.cumulative.append(total)
        self._cache_id: int | None = None
        self._cache_data: dict[str, torch.Tensor] | None = None
        self.split = split
        self.normalization = metadata["normalization"]
        self.normalization_applied = bool(self.normalization.get("applied", False))
        self.hand_shape_dim = int(metadata["hand_shape_dim"])
        self.num_frames = int(metadata.get("window_config", {}).get("num_frames", 30))
        if self.num_frames not in {10, 20, 30}:
            raise ValueError(f"Window sensitivity supports T in {{10,20,30}}, got {self.num_frames}")
        records = metadata.get(f"{split}_trial_records", [])
        all_records = []
        for name in ("train", "val"):
            all_records.extend(metadata.get(f"{name}_trial_records", []))
        raw_labels = []
        for record in all_records:
            if isinstance(record, dict):
                raw_labels.append(str(record.get("action_label", record.get("trial_id"))))
        label_map = {label: index for index, label in enumerate(sorted(set(raw_labels)))}
        self.labels_by_trial = []
        self.users_by_trial = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{split}_trial_records must contain dictionaries")
            if split == "test":
                self.labels_by_trial.append(-1)
            else:
                raw = str(record.get("action_label", record.get("trial_id")))
                if raw not in label_map:
                    raise ValueError(f"Unknown training/validation context label: {raw!r}")
                self.labels_by_trial.append(label_map[raw])
            user_text = str(record.get("user", ""))
            digits = "".join(character for character in user_text if character.isdigit())
            if not digits:
                raise ValueError(f"Cannot derive numeric user ID from {user_text!r}")
            self.users_by_trial.append(int(digits))
        if len(label_map) not in {20, 21, 22}:
            raise ValueError(f"Expected 20 self-collected or 21/22 PiMForce action classes, found {len(label_map)}")

    def _normalize(self, item: dict[str, torch.Tensor]) -> None:
        if self.normalization_applied:
            return
        def tensor(name: str, reference: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(self.normalization[name], dtype=reference.dtype)
        item["x_imu"] = (item["x_imu"] - tensor("imu_mean", item["x_imu"])) / tensor("imu_std", item["x_imu"]).clamp_min(1e-8)
        item["emg_features"] = (item["emg_features"] - tensor("emg_feature_mean", item["emg_features"])) / tensor("emg_feature_std", item["emg_features"]).clamp_min(1e-8)
        if "emg_sequence" in item:
            item["emg_sequence"] = (
                item["emg_sequence"] - tensor("emg_raw_mean", item["emg_sequence"])
            ) / tensor("emg_raw_std", item["emg_sequence"]).clamp_min(1e-8)
        if "hand_shape_target" in item:
            item["hand_shape_target"] = (
                item["hand_shape_target"] - tensor("hand_shape_mean", item["hand_shape_target"])
            ) / tensor("hand_shape_std", item["hand_shape_target"]).clamp_min(1e-8)
        force_range = (tensor("force_max", item["force"]) - tensor("force_min", item["force"])).clamp_min(1e-8)
        item["force"] = (item["force"] - tensor("force_min", item["force"])) / force_range

    def __len__(self) -> int:
        return self.num_samples

    def _chunk(self, chunk_id: int) -> dict[str, torch.Tensor]:
        if self._cache_id != chunk_id:
            path = self.fold_dir / self.chunks[chunk_id]["file"]
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    data = {
                        key: torch.from_numpy(np.array(archive[key], copy=True))
                        for key in archive.files
                    }
            else:
                data = load_pt(path)
            self._cache_data = {
                key: torch.nan_to_num(value) if torch.is_tensor(value) and torch.is_floating_point(value) else value
                for key, value in data.items()
            }
            self._cache_id = chunk_id
        assert self._cache_data is not None
        return self._cache_data

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        chunk_id = bisect.bisect_right(self.cumulative, index)
        previous = 0 if chunk_id == 0 else self.cumulative[chunk_id - 1]
        data = self._chunk(chunk_id)
        item = {
            key: data[key][index - previous]
            for key in ("x_imu", "force", "force_window", "emg_features", "emg_sequence", "trial_index", "user_id", "hand_shape", "hand_pose")
            if key in data and torch.is_tensor(data[key])
        }
        if "force_window" in item:
            item["force"] = item.pop("force_window")
        if "hand_shape" in item:
            hand_shape = item.pop("hand_shape")
        elif "hand_pose" in item:
            hand_shape = item.pop("hand_pose")
        else:
            hand_shape = None
        required = {"x_imu", "force", "emg_features"}
        missing = required - item.keys()
        if missing:
            raise KeyError(f"Precomputed split is missing tensors: {sorted(missing)}")
        if item["x_imu"].shape != (self.num_frames, 12):
            raise ValueError(
                f"Expected IMU window [{self.num_frames},12], got {tuple(item['x_imu'].shape)}"
            )
        if item["emg_features"].shape[-1] != 10:
            raise ValueError(f"Expected 10 EMG features, got {tuple(item['emg_features'].shape)}")
        if item["force"].shape != (self.num_frames, 5):
            raise ValueError(
                f"Expected pressure window [{self.num_frames},5], got {tuple(item['force'].shape)}"
            )
        if hand_shape is None and self.split != "test":
            raise KeyError(
                "Training requires hand-pose keypoints (hand_shape or hand_pose) in every sample; "
                "for the self-collected dataset they are available upon request."
            )
        if hand_shape is not None:
            if hand_shape.ndim >= 2 and hand_shape.shape[0] == self.num_frames:
                hand_shape = hand_shape[-1]
            item["hand_shape_target"] = hand_shape.reshape(-1)
            if item["hand_shape_target"].numel() != self.hand_shape_dim:
                raise ValueError(
                    f"Expected hand-shape target dimension {self.hand_shape_dim}, got "
                    f"{item['hand_shape_target'].numel()}"
                )
        if "trial_index" in item:
            trial_index = int(item.pop("trial_index"))
            if not 0 <= trial_index < len(self.labels_by_trial):
                raise IndexError(f"trial_index {trial_index} is outside {self.split} metadata")
            if self.split != "test":
                item["context_label"] = torch.tensor(self.labels_by_trial[trial_index], dtype=torch.long)
            item["user_id"] = torch.tensor(self.users_by_trial[trial_index], dtype=torch.long)
        elif self.split != "test":
            raise KeyError("Training/validation auxiliary context supervision requires trial_index")
        elif "user_id" not in item:
            raise KeyError("Test samples require user_id for per-user metric aggregation")
        else:
            item["user_id"] = item["user_id"].long().reshape(())
        self._normalize(item)
        return item


class ChunkLevelSampler(Sampler[int]):
    def __init__(self, dataset: PrecomputedChunkDataset, shuffle_chunks=True, shuffle_within_chunk=True, seed=2026, num_replicas=1, rank=0):
        self.dataset = dataset
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.num_samples = (len(dataset) + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset.chunks), generator=generator).tolist() if self.shuffle_chunks else list(range(len(self.dataset.chunks)))
        indices: list[int] = []
        start = 0
        ranges = []
        for stop in self.dataset.cumulative:
            ranges.append(list(range(start, stop)))
            start = stop
        for chunk_id in order:
            chunk_indices = ranges[chunk_id]
            if self.shuffle_within_chunk:
                permutation = torch.randperm(len(chunk_indices), generator=generator).tolist()
                chunk_indices = [chunk_indices[i] for i in permutation]
            indices.extend(chunk_indices)
        total = self.num_samples * self.num_replicas
        if len(indices) < total:
            indices.extend(indices[: total - len(indices)])
        return iter(indices[self.rank:total:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


def load_precomputed_loso_datasets(root: str | Path, test_user: str):
    fold_dir = Path(root) / f"fold_{test_user}"
    metadata = load_metadata(fold_dir)
    train = PrecomputedChunkDataset(fold_dir, "train", metadata)
    val = PrecomputedChunkDataset(fold_dir, "val", metadata) if split_index_exists(fold_dir, "val") else None
    test = PrecomputedChunkDataset(fold_dir, "test", metadata)
    return train, val, test, metadata


def load_precomputed_fixed_datasets(root: str | Path, dataset_name: str):
    # Either a cache root containing fixed_<dataset>/, or the split folder itself,
    # e.g. the self-collected data folder NiSM_data/data.
    split_dir = Path(root) / f"fixed_{dataset_name}"
    if not split_dir.is_dir() and (Path(root) / "metadata.json").is_file():
        split_dir = Path(root)
    metadata = load_metadata(split_dir)
    if metadata.get("dataset") not in (None, dataset_name):
        raise ValueError(f"{split_dir} holds the {metadata.get('dataset')!r} dataset, not {dataset_name!r}")
    train = PrecomputedChunkDataset(split_dir, "train", metadata)
    val = PrecomputedChunkDataset(split_dir, "val", metadata) if split_index_exists(split_dir, "val") else None
    test = PrecomputedChunkDataset(split_dir, "test", metadata)
    return train, val, test, metadata
