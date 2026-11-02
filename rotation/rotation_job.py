"""Simulated ring-rotation experiment (paper Appendix A.4.5) on the self-collected data.

Steps (each is skipped if its output already exists, so a resubmitted job resumes):
  1. raw recordings -> synchronized CSV layout     (convert_selfcollected.py; skipped with --raw-root
                                                     or --cache-root)
  2. build cache: within-participant temporal split 60/10/30 (default) or User1-16 / User17-20 (create_cache.py)
  3. train NiSM, 100 epochs; run_rotation_job.sh keeps the lowest-validation-MSE checkpoint (train.py)
  4. evaluate without rotation, fixed offsets of 10/20/30/45 deg, and random per-trial offsets of up to
     15/30 deg, 3 draws each (evaluate_rotation.py)
Every line is printed with a timestamp; training lines get an ETA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPOCH_RE = re.compile(r"epoch=(\d+)/(\d+)")
T0 = time.time()


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:d}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} | total {fmt(time.time() - T0)}] {msg}", flush=True)


def run(step: str, cmd: list[str]) -> None:
    say(f"===== {step} START =====")
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        m = EPOCH_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            per = (time.time() - start) / max(done, 1)
            line += f"   <-- {done}/{total} epochs ({100 * done / total:.0f}%), {per:.0f}s/epoch, ETA {fmt(per * (total - done))}"
        print(f"  {line}", flush=True)
    code = proc.wait()
    if code != 0:
        say(f"===== {step} FAILED (exit {code}) after {fmt(time.time() - start)} =====")
        sys.exit(code)
    say(f"===== {step} DONE in {fmt(time.time() - start)} =====")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path,
                    help="existing feature cache built for this experiment; skips steps 1 and 2")
    ap.add_argument("--raw-root", type=Path,
                    help="already converted per-trial CSV files (offset-zeroed labels for the paper setting); skips step 1")
    ap.add_argument("--recordings-root", type=Path, help="raw recordings (needs --id-map)")
    ap.add_argument("--id-map", type=Path, help="anonymous-id mapping for convert_selfcollected.py")
    ap.add_argument("--work-dir", type=Path, required=True, help="where converted data, cache, model and results go")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--split-mode", choices=["within", "users"], default="within",
                    help="within: all 20 users, last 30%% of every trial held out (default); users: User17-20 held out")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs for training (0 = all visible)")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers per GPU process")
    args = ap.parse_args()
    tare = os.environ.get("NISM_TARE", "0") == "1"
    tag = ("" if args.split_mode == "users" else "_within_v3") + ("_tare" if tare else "")
    run_tag = (tag + ("_mse" if os.environ.get("NISM_VAL_METRIC") == "mse" else "")
               + ("_bnlast" if os.environ.get("NISM_BN_LASTSTEP") == "1" else ""))
    raw, cache, train, result = (args.raw_root or args.work_dir / ("raw_cat_tare" if tare else "raw_cat"), args.work_dir / f"cache{tag}",
                                 args.work_dir / f"train{run_tag}", args.work_dir / f"rotation_result{run_tag}.json")
    if args.cache_root is not None:
        cache = args.cache_root
    args.work_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    say("===== Rotation experiment: 1 training run + 1 evaluation =====")
    import torch
    cuda = torch.cuda.is_available()
    say(f"[0/4] python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {cuda}"
        + (f", GPU {torch.cuda.get_device_name(0)}" if cuda else ""))
    if not cuda:
        say("[ERR] no GPU visible to this job"); sys.exit(1)

    n_done = len(list(raw.glob("User*/Session1/emg_feat_*.csv")))  # EMG feature files, one per converted trial
    if args.cache_root is not None:
        say(f"[1/4] Convert data: skipped, using the existing cache {cache}")
    elif n_done >= 400:
        say("[1/4] Convert data: already done (400/400), skip")
    else:
        if args.recordings_root is None or args.id_map is None:
            say(f"[ERR] {raw} has {n_done}/400 converted trials; pass --cache-root, --raw-root, "
                "or --recordings-root and --id-map to convert the raw recordings"); sys.exit(1)
        say(f"[1/4] Convert data: {n_done}/400 trials done, converting the rest (1 line per user)")
        run("[1/4] Convert data", [py, "-u", "convert_selfcollected.py", "--src", str(args.recordings_root),
                                   "--id-map", str(args.id_map), "--out", str(raw)]
            + (["--tare"] if tare else []))

    if (cache / "fixed_self_collected" / "metadata.json").is_file():
        say("[2/4] Build cache: already done, skip")
    else:
        split_text = ("all 20 users; per trial: first 60% train / next 10% validation / last 30% test" if args.split_mode == "within"
                      else "User1-16 train / User17-20 test")
        say(f"[2/4] Build cache ({split_text}, progress every 5000 windows)")
        run("[2/4] Build cache", [py, "-u", "create_cache.py", "--data-root", str(raw), "--output-root", str(cache),
                                  "--protocol", "fixed", "--dataset", "self_collected", "--overwrite",
                                  "--split-mode", args.split_mode])

    ckpts = sorted(train.rglob("*.pt")) if train.exists() else []
    if ckpts:
        say(f"[3/4] Train: checkpoint exists, skip ({ckpts[-1].name})")
    else:
        ngpu = args.gpus or torch.cuda.device_count()
        say(f"[3/4] Train NiSM, {args.epochs} epochs on {ngpu} GPU(s), {args.workers} loader workers each "
            "(global batch 256 split across GPUs; 1 line per epoch with ETA)")
        launcher = ([py, "-u", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}"]
                    if ngpu > 1 else [py, "-u"])
        run("[3/4] Train", launcher + ["train.py", "--precomputed-root", str(cache), "--output-dir", str(train),
                                       "--protocol", "fixed", "--dataset", "self_collected", "--ablation", "full",
                                       "--num-epochs", str(args.epochs), "--num-workers", str(args.workers),
                                       "--no-deterministic",
                                       *(["--no-validate-precomputed-metadata"] if args.split_mode == "within" else []),
                                       "--device", "cuda"])
        ckpts = sorted(train.rglob("*.pt"))
        if not ckpts:
            say("[ERR] no checkpoint produced"); sys.exit(1)

    say("[4/4] Evaluate: clean; fixed rotation 10/20/30/45 deg about random axes; random per-trial rotation up to 15/30 deg (3 seeds each)")
    run("[4/4] Evaluate", [py, "-u", "evaluate_rotation.py", "--checkpoint", str(ckpts[-1]), "--precomputed-root", str(cache),
                           "--protocol", "fixed", "--dataset", "self_collected", "--axes", "random",
                           "--angles", "10", "20", "30", "45", "--redon-max-angles", "15", "30", "--seeds", "3", "--num-workers", str(args.workers), "--output", str(result), "--device", "cuda"])

    res = json.loads(result.read_text(encoding="utf-8"))
    clean = res["clean"]
    c_r2 = clean["r2"]["mean"]
    say(f"===== RESULT ({args.split_mode} split; mean over test users) =====")
    say(f"  clean                 : R2 {100 * c_r2:6.2f}  NRMSE {100 * clean['nrmse']['mean']:6.2f}  PCC {100 * clean['pcc']['mean']:6.2f}")
    rows = [(f"fixed {k.split('_')[1]:>2} deg (rand axis)", v["mean_over_seeds"]) for k, v in res["fixed_offset"].items()]
    rows += [(f"per-trial <= {k[3:]:>2} deg     ", v["mean_over_seeds"]) for k, v in res["per_trial_offset"].items()]
    for name, m in rows:
        say(f"  {name}: R2 {100 * m['r2']:6.2f}  NRMSE {100 * m['nrmse']:6.2f}  PCC {100 * m['pcc']:6.2f}   dR2 {100 * (m['r2'] - c_r2):+6.2f}")
    say(f"ALL DONE. Result file: {result}")


if __name__ == "__main__":
    main()
