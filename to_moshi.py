#!/usr/bin/env python3
"""
to_moshi.py — convert a duplex_corpus.py corpus into moshi-finetune format.

moshi-finetune expects:
    data/
    ├── train.jsonl          {"path": "data_stereo/x.wav", "duration": 24.5}
    ├── val.jsonl
    └── data_stereo/
        ├── x.wav            24 kHz stereo: LEFT = moshi, RIGHT = user
        └── x.json           transcript w/ timestamps (made later by annotate.py)

What this does:
    - resamples each accepted conversation to 24 kHz (Mimi's rate)
    - assigns one speaker to the left channel (moshi) and one to the right (user)
    - cuts long conversations into training windows at mutual-silence points,
      so no window starts or ends mid-word
    - normalises loudness per channel
    - writes train/val jsonl split by SOURCE VIDEO, not by window

Usage:
    python3 to_moshi.py --root ./corpus --out ./moshi_data
    python3 to_moshi.py --root ./corpus --out ./moshi_data --window 100 --moshi-channel random
    python3 to_moshi.py --root ./corpus --out ./moshi_data --dry-run

Then, in the moshi-finetune repo:
    python annotate.py /abs/path/moshi_data/train.jsonl
    python annotate.py /abs/path/moshi_data/val.jsonl
    torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml

Requires: numpy soundfile scipy
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 24000          # Mimi operates at 24 kHz
FRAME_HZ = 12.5            # Mimi frame rate; windows should be a clean multiple


def log(m):
    print(m, flush=True)


def load_segments(d: Path):
    p = d / "segments.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return sorted(out, key=lambda s: s["start"])


def silence_points(segs, total_dur, min_gap=0.4):
    """Times where neither speaker is active — safe places to cut."""
    if not segs:
        return []
    iv = sorted((s["start"], s["end"]) for s in segs)
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    pts = []
    prev_end = 0.0
    for a, b in merged:
        if a - prev_end >= min_gap:
            pts.append((prev_end + a) / 2)
        prev_end = b
    if total_dur - prev_end >= min_gap:
        pts.append((prev_end + total_dur) / 2)
    return pts


def plan_windows(total_dur, cuts, target, min_len, max_len):
    """
    Walk forward in ~target-second steps, snapping each boundary to the nearest
    silence point. Falls back to a hard cut if no silence is nearby.
    """
    bounds = [0.0]
    pos = 0.0
    while total_dur - pos > max_len:
        ideal = pos + target
        near = [c for c in cuts if pos + min_len < c < pos + max_len]
        nxt = min(near, key=lambda c: abs(c - ideal)) if near else pos + target
        bounds.append(nxt)
        pos = nxt
    bounds.append(total_dur)

    spans = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= min_len]
    return spans


def norm_channel(x, target_rms=0.05):
    rms = float(np.sqrt((x ** 2).mean()))
    if rms < 1e-6:
        return x
    y = x * (target_rms / rms)
    peak = float(np.abs(y).max())
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype("float32")


def to_24k(x, sr):
    if sr == TARGET_SR:
        return x.astype("float32")
    from math import gcd
    g = gcd(int(sr), TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype("float32")


def pick_moshi_channel(mode, meta, rng):
    if mode == "0":
        return 0
    if mode == "1":
        return 1
    if mode == "random":
        return rng.randint(0, 1)
    if mode == "talkative":
        # the speaker with more talk time becomes moshi (usually the guest)
        shares = meta.get("conversation", {}).get("speaker_shares", {})
        if len(shares) == 2:
            keys = sorted(shares)
            return 0 if shares[keys[0]] >= shares[keys[1]] else 1
    return 0


def process_entry(d: Path, out_audio: Path, args, rng):
    mj = d / "meta.json"
    if not mj.exists():
        return []
    meta = json.loads(mj.read_text(encoding="utf-8"))

    stereo_path = d / "stereo.wav"
    if not stereo_path.exists():
        log(f"  skip {d.name}: no stereo.wav (only {len(meta['files']['channels'])} channels?)")
        return []

    audio, sr = sf.read(str(stereo_path), dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        log(f"  skip {d.name}: {audio.shape[1]} channels")
        return []

    left_raw, right_raw = audio[:, 0], audio[:, 1]
    total_dur = len(left_raw) / sr

    moshi_idx = pick_moshi_channel(args.moshi_channel, meta, rng)
    moshi_raw = left_raw if moshi_idx == 0 else right_raw
    user_raw = right_raw if moshi_idx == 0 else left_raw

    moshi = norm_channel(to_24k(moshi_raw, sr))
    user = norm_channel(to_24k(user_raw, sr))
    n = min(len(moshi), len(user))
    moshi, user = moshi[:n], user[:n]

    segs = load_segments(d)
    cuts = silence_points(segs, total_dur, args.min_gap)
    spans = plan_windows(total_dur, cuts, args.window, args.min_window, args.max_window)

    written = []
    for i, (a, b) in enumerate(spans):
        # snap the length to a whole number of Mimi frames
        frames = int((b - a) * FRAME_HZ)
        dur = frames / FRAME_HZ
        s0 = int(a * TARGET_SR)
        s1 = s0 + int(dur * TARGET_SR)
        if s1 > n:
            s1 = n
            dur = (s1 - s0) / TARGET_SR
        if dur < args.min_window:
            continue

        chunk = np.stack([moshi[s0:s1], user[s0:s1]], axis=1)

        # drop windows where one side never speaks — nothing duplex to learn
        rms_l = float(np.sqrt((chunk[:, 0] ** 2).mean()))
        rms_r = float(np.sqrt((chunk[:, 1] ** 2).mean()))
        if min(rms_l, rms_r) < args.min_rms:
            continue

        name = f"{d.name}_{i:04d}.wav"
        if not args.dry_run:
            sf.write(str(out_audio / name), chunk, TARGET_SR)
        written.append({"path": f"data_stereo/{name}", "duration": round(dur, 6)})

    log(f"  {d.name}: {total_dur/60:.1f}min -> {len(written)} windows "
        f"(moshi=spk{moshi_idx}, sr {sr}->{TARGET_SR})")
    return written


CONFIG_TEMPLATE = """# Generated by to_moshi.py — copy into moshi-finetune/example/
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

run_dir: "./runs/indic_duplex"

lora:
  enable: true
  rank: 128
  scaling: 2.
  ft_embed: false

# Windows in this dataset are ~{window}s, so keep duration_sec at or below that.
duration_sec: {duration_sec}
batch_size: 8
max_steps: 2000

first_codebook_weight_multiplier: 100.
text_padding_weight: 0.5
gradient_checkpointing: true

optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

data:
  train_data: "{train}"
  eval_data: "{val}"
  shuffle: true

eval_freq: 200
no_eval: false
ckpt_freq: 500
save_adapters: true
seed: 0
log_freq: 10
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./corpus", help="duplex_corpus.py output root")
    ap.add_argument("--out", default="./moshi_data")
    ap.add_argument("--window", type=float, default=120.0, help="target window seconds")
    ap.add_argument("--min-window", type=float, default=30.0)
    ap.add_argument("--max-window", type=float, default=180.0)
    ap.add_argument("--min-gap", type=float, default=0.4,
                    help="mutual silence needed to allow a cut")
    ap.add_argument("--min-rms", type=float, default=0.005,
                    help="drop windows where a channel is effectively silent")
    ap.add_argument("--moshi-channel", default="talkative",
                    choices=["0", "1", "random", "talkative"],
                    help="which separated speaker becomes the LEFT/moshi channel")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    args = ap.parse_args()

    root = Path(args.root)
    accepted = sorted((root / "accepted").iterdir()) if (root / "accepted").exists() else []
    accepted = [d for d in accepted if d.is_dir()]
    if not accepted:
        sys.exit(f"no accepted entries under {root/'accepted'}")

    out = Path(args.out)
    out_audio = out / "data_stereo"
    if not args.dry_run:
        out_audio.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # split by source video so windows from one conversation never straddle
    # the train/val boundary — otherwise val leaks speaker identity
    vids = list(accepted)
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac)) if len(vids) > 1 else 0
    val_set = set(v.name for v in vids[:n_val])

    log(f"{len(accepted)} conversations, {len(val_set)} held out for validation\n")

    train, val = [], []
    for d in accepted:
        entries = process_entry(d, out_audio, args, rng)
        (val if d.name in val_set else train).extend(entries)

    if args.dry_run:
        log("\n(dry run — no files written)")
    else:
        for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
            with open(out / name, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        cfg = CONFIG_TEMPLATE.format(
            window=int(args.window),
            duration_sec=int(min(args.window, 100)),
            train=str((out / "train.jsonl").resolve()),
            val=str((out / "val.jsonl").resolve()),
        )
        (out / "moshi_7B_indic.yaml").write_text(cfg, encoding="utf-8")

    t_h = sum(r["duration"] for r in train) / 3600
    v_h = sum(r["duration"] for r in val) / 3600
    log(f"\ntrain: {len(train):5d} windows  {t_h:6.2f} h")
    log(f"val:   {len(val):5d} windows  {v_h:6.2f} h")
    log(f"total: {len(train)+len(val):5d} windows  {t_h+v_h:6.2f} h")

    if t_h < 10:
        log("\nNOTE: Kyutai used 2,000 h (Fisher) for the multi-stream stage. "
            "Under ~10 h you should expect the LoRA to overfit accent and channel "
            "characteristics rather than learn turn-taking.")

    if not args.dry_run:
        log(f"\nNext:")
        log(f"  cd /path/to/moshi-finetune")
        log(f"  python annotate.py {(out/'train.jsonl').resolve()}")
        log(f"  python annotate.py {(out/'val.jsonl').resolve()}")
        log(f"  cp {(out/'moshi_7B_indic.yaml').resolve()} example/")
        log(f"  torchrun --nproc-per-node 1 -m train example/moshi_7B_indic.yaml")


if __name__ == "__main__":
    main()