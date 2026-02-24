#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test clip manifests for DVS128 PNG frames."
    )
    parser.add_argument(
        "--dvs-root",
        type=Path,
        default=Path("datasets/DVS128"),
        help="Root directory containing labels CSV and dvs2vid folder.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        required=True,
        help="Sample stem, e.g. user01_fluorescent_led.",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=8,
        help="Number of frames per clip.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Sliding-window stride in frames.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for split reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output manifest directory. Default: datasets/DVS128/manifests/<sample>_<cliplen>f",
    )
    return parser.parse_args()


def load_labels(path):
    with path.open("r") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        # Convert dataset labels 1..11 to 0..10 for cross_entropy.
        out.append(
            (
                int(r["class"]) - 1,
                int(r["startTime_usec"]),
                int(r["endTime_usec"]),
            )
        )
    return out


def load_timestamps_usec(path):
    ts_usec = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ts_usec.append(int(round(float(s) * 1_000_000)))
    return ts_usec


def build_clips(frame_paths, ts_usec, label_segments, clip_len, stride):
    clips = []
    clip_id = 0
    for label, t_start, t_end in label_segments:
        idxs = [i for i, t in enumerate(ts_usec) if t_start <= t <= t_end]
        if len(idxs) < clip_len:
            continue

        seg_first = idxs[0]
        seg_last = idxs[-1]
        for start in range(seg_first, seg_last - clip_len + 2, stride):
            end = start + clip_len - 1
            if end > seg_last:
                break
            clip_frames = frame_paths[start : end + 1]
            clips.append(
                {
                    "clip_id": f"clip_{clip_id:06d}",
                    "label": label,
                    "frames": ";".join(str(p.resolve()) for p in clip_frames),
                }
            )
            clip_id += 1
    return clips


def split_clips(clips, val_ratio, seed):
    rng = random.Random(seed)
    by_label = {}
    for c in clips:
        by_label.setdefault(c["label"], []).append(c)

    train, val = [], []
    for _, group in sorted(by_label.items()):
        rng.shuffle(group)
        n_val = int(round(len(group) * val_ratio))
        if len(group) >= 2:
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, len(group) - 1))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label", "frames"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    sample_name = args.sample_name

    labels_csv = args.dvs_root / f"{sample_name}_labels.csv"
    frames_dir = args.dvs_root / "dvs2vid" / sample_name
    timestamps_file = frames_dir / "timestamps.txt"

    if args.output_dir is None:
        output_dir = (
            args.dvs_root / "manifests" / f"{sample_name}_{args.clip_len}f"
        )
    else:
        output_dir = args.output_dir

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    if not labels_csv.exists():
        raise RuntimeError(f"Missing labels file: {labels_csv}")
    if not timestamps_file.exists():
        raise RuntimeError(f"Missing timestamps file: {timestamps_file}")

    label_segments = load_labels(labels_csv)
    ts_usec = load_timestamps_usec(timestamps_file)
    if len(ts_usec) != len(frame_paths):
        raise RuntimeError(
            f"Frame/timestamp mismatch: {len(frame_paths)} frames vs {len(ts_usec)} timestamps"
        )

    clips = build_clips(
        frame_paths=frame_paths,
        ts_usec=ts_usec,
        label_segments=label_segments,
        clip_len=args.clip_len,
        stride=args.stride,
    )
    if len(clips) == 0:
        raise RuntimeError("No clips produced. Try smaller clip-len or stride.")

    train, val = split_clips(clips, args.val_ratio, args.seed)
    # For this starter setup, reuse validation as test split.
    test = list(val)

    write_manifest(output_dir / "train.csv", train)
    write_manifest(output_dir / "val.csv", val)
    write_manifest(output_dir / "test.csv", test)

    print(f"Wrote manifests to: {output_dir}")
    print(f"Total clips: {len(clips)}")
    print(f"Train clips: {len(train)}")
    print(f"Val clips:   {len(val)}")
    print(f"Test clips:  {len(test)}")


if __name__ == "__main__":
    main()
