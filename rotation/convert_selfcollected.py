"""Convert the raw self-collected recordings into the synchronized-CSV layout read by
create_cache.py (--dataset self_collected).

Participants are identified by anonymous IDs (User1-User20; User17-20 are the fixed test
users in the paper). The mapping from raw recording folders to anonymous IDs is read from a
JSON file passed with --id-map ({"1": "<raw folder name>", ..., "20": "..."}).

Input (raw layout), all aligned on a shared frame timeline except raw EMG:
  EMG_Press/6_Pressure_smooth_before_log1p_transformed/<folder>/<k>/aligned_pressure_log1p/
      aligned_pressure_f{1..5}_log1p.csv          -> column "Pressure Log1p"
  EMG_Press/2_EMG_filtered/<folder>/filtered_emg_<k>.csv -> band-pass/notch-filtered EMG
  IMU/4_time_aligned/<folder>/WT1/<k>_WT1.csv            -> thumb IMU acc (g) + quaternion w,x,y,z
  video/3_keypoints_rotation/<folder>/<k>_keypoints_smoothed.csv -> 63-D hand keypoints (numeric only)
Output: out/User{i}/Session1/{force,emg,emg_feat,proximal_acc,rotation,hand_pose}_{k}.csv,
with timestamps in seconds relative to the start of each trial.

EMG features (emg_feat_{k}.csv) are the paper's 10 features computed on sliding windows of
100 raw samples (step 10); every frame takes the mean of the window features whose
centre lies within +-100 ms of the frame (else the previous frame's value).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

USERS: dict[int, str] = {}  # anonymous id -> raw folder name, filled from --id-map


def load_id_map(path: Path) -> dict[int, str]:
    mapping = {int(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    missing = [i for i in range(1, 21) if i not in mapping]
    if missing:
        raise ValueError(f"--id-map is missing anonymous ids {missing}")
    return mapping
ACC_COLS = ["thumb_proximal_mid_acc_x", "thumb_proximal_mid_acc_y", "thumb_proximal_mid_acc_z"]
ROT_COLS = [f"rot_thumb_pip_r{i}c{j}" for i in range(1, 4) for j in range(1, 4)]
EMG_FEATURE_COLUMNS = ["MAV", "RMS", "ZCR", "WL", "Variance", "Amplitude_Sum", "Std",
                       "Mean_PSD", "Peak_Frequency_Index", "Total_Spectral_Energy"]
EMG_WIN, EMG_STEP, EMG_HALF_SPAN = 100, 10, 0.1  # samples, samples, seconds
TARE_SOURCE_COLUMN = "Smoothed Pressure"  # linear smoothed reading used for zeroing
PRESSURE_COLUMN = "Pressure Log1p"  # log1p of the smoothed pressure


def to_seconds(series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(series, format="mixed")
    return (t - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy(np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    w, x, y, z = q.T
    r = np.empty((len(q), 9))
    r[:, 0] = 1 - 2 * (y * y + z * z); r[:, 1] = 2 * (x * y - z * w); r[:, 2] = 2 * (x * z + y * w)
    r[:, 3] = 2 * (x * y + z * w); r[:, 4] = 1 - 2 * (x * x + z * z); r[:, 5] = 2 * (y * z - x * w)
    r[:, 6] = 2 * (x * z - y * w); r[:, 7] = 2 * (y * z + x * w); r[:, 8] = 1 - 2 * (x * x + y * y)
    return r


def emg_features10(x: np.ndarray) -> np.ndarray:
    """The paper's 10 EMG features; identical to create_cache.emg_features."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    absolute = np.abs(x)
    diff = np.diff(x)
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2 / x.size
    zcr = np.count_nonzero((x[:-1] * x[1:] < 0) & (np.abs(diff) > 1e-8)) / max(x.size - 1, 1)
    return np.asarray([absolute.mean(), np.sqrt(np.mean(x * x)), zcr, np.abs(diff).sum(), x.var(),
                       absolute.sum(), x.std(), spectrum.mean(), int(np.argmax(spectrum)), spectrum.sum()])


def frame_emg_features(emg_t: np.ndarray, emg_x: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
    starts = np.arange(0, len(emg_x) - EMG_WIN + 1, EMG_STEP)
    if len(starts) == 0:
        raise ValueError("EMG recording shorter than one feature window")
    feats = np.stack([emg_features10(emg_x[s:s + EMG_WIN]) for s in starts])
    centres = emg_t[starts + EMG_WIN // 2]
    csum = np.vstack([np.zeros((1, feats.shape[1])), np.cumsum(feats, axis=0)])
    lo = np.searchsorted(centres, frame_t - EMG_HALF_SPAN, side="left")
    hi = np.searchsorted(centres, frame_t + EMG_HALF_SPAN, side="right")
    out = np.empty((len(frame_t), feats.shape[1]))
    last = feats.mean(axis=0)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            last = (csum[b] - csum[a]) / (b - a)
        out[i] = last
    return out


def pressure_dir(src: Path, name: str, k: int) -> Path:
    return src / "EMG_Press" / "6_Pressure_smooth_before_log1p_transformed" / name / str(k) / "aligned_pressure_log1p"


def user_tare_offsets(src: Path, uid: int, train_fraction: float = 0.6) -> np.ndarray:
    """Per-user, per-finger sensor offset (the no-contact reading of the FlexiForce sensors).

    For every trial, the 5th percentile of the linear smoothed pressure over the first
    `train_fraction` of the trial (the within-user training segment);
    the offset is the median of these values over the user's 20 trials.
    """
    lows = []
    for k in range(1, 21):
        cols = []
        for f in range(1, 6):
            df = pd.read_csv(pressure_dir(src, USERS[uid], k) / f"aligned_pressure_f{f}_log1p.csv")
            cols.append(df[TARE_SOURCE_COLUMN].to_numpy(np.float64))
        n = min(len(c) for c in cols)
        p = np.stack([c[:n] for c in cols], axis=1)[: max(1, int(n * train_fraction))]
        lows.append(np.percentile(p, 5, axis=0))
    return np.median(np.stack(lows), axis=0)


def convert_trial(src: Path, uid: int, k: int, sess: Path, tare: np.ndarray | None = None,
                  write_emg_feat: bool = True) -> int:
    """Convert one trial. All timestamps are shifted so that the trial starts at t = 0 s
    (the absolute recording time is not written)."""
    name = USERS[uid]
    pdir = pressure_dir(src, name, k)
    frames = []
    for f in range(1, 6):
        df = pd.read_csv(pdir / f"aligned_pressure_f{f}_log1p.csv")
        if tare is None:
            values = df[PRESSURE_COLUMN]
        else:  # zero the sensor offset, then the same log1p compression as "Pressure Log1p"
            values = np.log1p(np.clip(df[TARE_SOURCE_COLUMN].to_numpy(np.float64) - tare[f - 1], 0.0, None))
        frames.append(pd.DataFrame({"Timestamp": df["Timestamp"], f"force_{f}": values})
                      .drop_duplicates("Timestamp").set_index("Timestamp"))
    force = pd.concat(frames, axis=1, join="inner").reset_index()
    force.insert(0, "timestamp", to_seconds(force.pop("Timestamp")))
    force = force.dropna().sort_values("timestamp")

    emg = pd.read_csv(src / "EMG_Press" / "2_EMG_filtered" / name / f"filtered_emg_{k}.csv")
    emg = pd.DataFrame({"timestamp": to_seconds(emg["timestamp"]), "Sensor0_sEMG": emg["bandpass_and_notch"].to_numpy()}) \
        .sort_values("timestamp", kind="stable")
    emg_feat = None
    if write_emg_feat:
        frame_t = force["timestamp"].to_numpy()
        feats = frame_emg_features(emg["timestamp"].to_numpy(), emg["Sensor0_sEMG"].to_numpy(), frame_t)
        emg_feat = pd.DataFrame({"timestamp": frame_t, **{c: feats[:, j] for j, c in enumerate(EMG_FEATURE_COLUMNS)}})

    imu = pd.read_csv(src / "IMU" / "4_time_aligned" / name / "WT1" / f"{k}_WT1.csv")
    t = to_seconds(imu.iloc[:, 0])
    vals = imu.iloc[:, 1:8].to_numpy(np.float64)
    order = np.argsort(t, kind="stable"); t, vals = t[order], vals[order]
    acc = pd.DataFrame({"timestamp": t, **{c: vals[:, i] for i, c in enumerate(ACC_COLS)}})
    rot_m = quat_to_rot(vals[:, 3:7])
    rot = pd.DataFrame({"timestamp": t, **{c: rot_m[:, i] for i, c in enumerate(ROT_COLS)}})

    kp = pd.read_csv(src / "video" / "3_keypoints_rotation" / name / f"{k}_keypoints_smoothed.csv")
    tcol = kp.columns[0]
    pose = kp.drop(columns=[tcol])
    if pose.shape[1] != 63:
        raise ValueError(f"User{uid} trial {k}: expected 63 keypoint columns, got {pose.shape[1]}")
    pose.insert(0, "timestamp", to_seconds(kp[tcol]))
    pose = pose.sort_values("timestamp")

    outputs = {"force": force, "emg": emg, "emg_feat": emg_feat, "proximal_acc": acc, "rotation": rot, "hand_pose": pose}
    if not write_emg_feat:  # only the rotation experiment reads the sub-window EMG features
        del outputs["emg_feat"]
    t0 = min(float(df["timestamp"].min()) for df in outputs.values())
    for kind, df in outputs.items():  # hand_pose is written last and marks the trial as done
        df = df.copy()
        df["timestamp"] = np.round(df["timestamp"].to_numpy(np.float64) - t0, 6)
        df.to_csv(sess / f"{kind}_{k}.csv", index=False)
    return len(force)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="root of the raw recordings")
    ap.add_argument("--id-map", type=Path, required=True,
                    help="JSON mapping anonymous ids 1-20 to raw folder names")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--users", type=int, nargs="+", default=list(range(1, 21)),
                    help="anonymous ids to convert (default: all 20)")
    ap.add_argument("--tare", action="store_true",
                    help="subtract each user's no-contact sensor offset before log1p (labels start at 0)")
    args = ap.parse_args()
    USERS.update(load_id_map(args.id_map))
    args.out.mkdir(parents=True, exist_ok=True)
    for uid in args.users:
        sess = args.out / f"User{uid}" / "Session1"
        sess.mkdir(parents=True, exist_ok=True)
        todo = [k for k in range(1, 21)
                if not ((sess / f"hand_pose_{k}.csv").is_file() and (sess / f"emg_feat_{k}.csv").is_file())]
        if not todo:
            print(f"User{uid}: already converted", flush=True)
            continue
        tare = user_tare_offsets(args.src, uid) if args.tare else None
        if tare is not None:
            (sess.parent / "tare_offset.tsv").write_text(
                "\t".join(f"{v:.2f}" for v in tare) + "\n", encoding="utf-8")
            print(f"User{uid} tare offsets (raw units, fingers 1-5): {np.round(tare, 2).tolist()}", flush=True)
        rows = [convert_trial(args.src, uid, k, sess, tare) for k in todo]
        print(f"User{uid}: {len(rows)} trials converted, pressure rows {sum(rows)}", flush=True)

if __name__ == "__main__":
    main()
