#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


CLASS_MAP = {
    "FT": 0,
    "OC": 1,
    "PS": 2,
    "NOSE": 3,
    "TR": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test clip manifests for EHWGesture RGB frames."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("../datasets/raw/EHWGesture"),
        help="Raw EHWGesture root containing split txt files and triggers.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("../datasets/timesformer/EHWGesture/frames"),
        help="Root directory containing extracted frame folders by sample id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../datasets/timesformer/EHWGesture/manifests/all_samples_8f"),
        help="Output manifest directory.",
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
        "--path-mode",
        type=str,
        choices=["rel", "abs"],
        default="rel",
        help="Store frame paths as relative-to --frames-root or absolute.",
    )
    return parser.parse_args()


def iter_trial_ids(path: Path):
    with path.open("r") as f:
        for line in f:
            sample_id = line.strip()
            if sample_id:
                yield sample_id


def sample_parts(sample_id: str):
    subject, hand, code = sample_id.split("_", 2)
    hand_dir = "Left" if hand.upper() == "LEFT" else "Right"
    return subject, hand, hand_dir, code


def label_from_sample_id(sample_id: str) -> int:
    code = sample_id.split("_", 2)[-1]
    gesture = re.sub(r"[0-9]+$", "", code)
    if gesture.startswith("FT"):
        return CLASS_MAP["FT"]
    if gesture.startswith("OC"):
        return CLASS_MAP["OC"]
    if gesture.startswith("PS"):
        return CLASS_MAP["PS"]
    if gesture.startswith("NOSE"):
        return CLASS_MAP["NOSE"]
    if gesture.startswith("TR"):
        return CLASS_MAP["TR"]
    raise ValueError(f"Unsupported EHWGesture code: {sample_id}")


def trigger_csv_path(raw_root: Path, sample_id: str) -> Path:
    subject, _, hand_dir, code = sample_parts(sample_id)
    return (
        raw_root
        / "Annotations"
        / "GestureTriggers"
        / subject
        / hand_dir
        / code
        / f"{code}_triggers.csv"
    )


def load_active_interval_sec(trigger_csv: Path):
    with trigger_csv.open("r") as f:
        reader = csv.DictReader(f)
        times = [float(row["Time"]) for row in reader if row.get("Time")]
    if not times:
        return None
    return min(times), max(times)


def load_timestamps_sec(path: Path):
    with path.open("r") as f:
        return [float(line.strip()) for line in f if line.strip()]


def frame_to_str(path: Path, path_mode: str, path_base: Path):
    if path_mode == "abs":
        return str(path.resolve())
    return str(path.resolve().relative_to(path_base.resolve()))


def build_rows_for_sample(args, sample_id: str, split_name: str, clip_id_start: int):
    subject, _, hand_dir, _ = sample_parts(sample_id)
    frame_dir = args.frames_root / subject / hand_dir / sample_id
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    ts_path = frame_dir / "timestamps.txt"
    if not frame_paths or not ts_path.exists():
        raise RuntimeError(f"Missing extracted frames for {sample_id}: {frame_dir}")

    timestamps = load_timestamps_sec(ts_path)
    if len(frame_paths) != len(timestamps):
        raise RuntimeError(
            f"Frame/timestamp mismatch for {sample_id}: {len(frame_paths)} vs {len(timestamps)}"
        )

    trig_path = trigger_csv_path(args.raw_root, sample_id)
    active_interval = load_active_interval_sec(trig_path) if trig_path.exists() else None
    if active_interval is None:
        active_idxs = list(range(len(frame_paths)))
    else:
        t0, t1 = active_interval
        active_idxs = [i for i, t in enumerate(timestamps) if t0 <= t <= t1]
        if len(active_idxs) < args.clip_len:
            active_idxs = list(range(len(frame_paths)))

    label = label_from_sample_id(sample_id)
    rows = []
    clip_id = clip_id_start
    first_idx, last_idx = active_idxs[0], active_idxs[-1]
    for start in range(first_idx, last_idx - args.clip_len + 2, args.stride):
        end = start + args.clip_len - 1
        if end > last_idx:
            break
        frames = frame_paths[start : end + 1]
        rows.append(
            {
                "clip_id": f"{split_name}_{clip_id:06d}",
                "label": label,
                "frames": ";".join(
                    frame_to_str(p, args.path_mode, args.frames_root) for p in frames
                ),
            }
        )
        clip_id += 1
    return rows, clip_id


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "label", "frames"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_split(args, split_name: str):
    split_file = args.raw_root / f"trials_to_{split_name}.txt"
    rows = []
    next_id = 0
    for sample_id in iter_trial_ids(split_file):
        sample_rows, next_id = build_rows_for_sample(args, sample_id, split_name, next_id)
        rows.extend(sample_rows)
    return rows


def main():
    args = parse_args()
    train_rows = build_split(args, "train")
    val_rows = build_split(args, "val")
    test_rows = build_split(args, "test")

    write_manifest(args.output_dir / "train.csv", train_rows)
    write_manifest(args.output_dir / "val.csv", val_rows)
    write_manifest(args.output_dir / "test.csv", test_rows)

    print(f"Wrote manifests to: {args.output_dir}")
    print(f"Train clips: {len(train_rows)}")
    print(f"Val clips:   {len(val_rows)}")
    print(f"Test clips:  {len(test_rows)}")


if __name__ == "__main__":
    main()
