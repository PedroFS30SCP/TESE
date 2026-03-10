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
        default=None,
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
    parser.add_argument(
        "--path-mode",
        type=str,
        choices=["rel", "abs"],
        default="rel",
        help="Store frame paths in manifest as relative-to --dvs-root (rel) or absolute (abs).",
    )
    parser.add_argument(
        "--sample-split",
        type=str,
        choices=["trainval", "test", "legacy"],
        default="legacy",
        help=(
            "How to assign clips for this sample: "
            "'trainval' splits into train/val only, "
            "'test' sends all clips to test only, "
            "'legacy' reuses val as test."
        ),
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="Optional path to official train trial list for direct global manifest build.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Optional path to official test trial list for direct global manifest build.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for single-sample mode.",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        default=None,
        help="Optional path to explicit validation trial list for direct global manifest build.",
    )
    return parser.parse_args()


def load_labels(path):
    with path.open("r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    fieldnames = set(rows[0].keys())
    out = []
    if {"class", "startTime_usec", "endTime_usec"}.issubset(fieldnames):
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

    if {"frame", "timestamp_sec", "class_id"}.issubset(fieldnames):
        current_label = None
        start_usec = None
        prev_usec = None
        for r in rows:
            raw_label = int(r["class_id"])
            # Skip background/unlabeled frames so final labels stay in 0..10.
            if raw_label <= 0:
                if current_label is not None and prev_usec is not None:
                    out.append((current_label, start_usec, prev_usec))
                    current_label = None
                    start_usec = None
                prev_usec = None
                continue

            label = raw_label - 1
            ts_usec = int(round(float(r["timestamp_sec"]) * 1_000_000))
            if current_label is None:
                current_label = label
                start_usec = ts_usec
            elif label != current_label:
                out.append((current_label, start_usec, prev_usec))
                current_label = label
                start_usec = ts_usec
            prev_usec = ts_usec
        if current_label is not None and prev_usec is not None:
            out.append((current_label, start_usec, prev_usec))
        return out

    raise RuntimeError(
        "Unsupported labels CSV format in {} with columns {}".format(
            path, sorted(fieldnames)
        )
    )


def load_timestamps_usec(path):
    ts_usec = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ts_usec.append(int(round(float(s) * 1_000_000)))
    return ts_usec


def build_clips(frame_paths, ts_usec, label_segments, clip_len, stride, path_mode, path_base):
    def frame_to_str(p):
        if path_mode == "abs":
            return str(p.resolve())
        return str(p.resolve().relative_to(path_base))

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
                    "frames": ";".join(frame_to_str(p) for p in clip_frames),
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
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label", "frames"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_sample_rows(args, sample_name):
    frames_dir = args.dvs_root / "dvs2vid" / sample_name
    labels_csv_candidates = [
        frames_dir / f"{sample_name}_labels.csv",
        args.dvs_root / f"{sample_name}_labels.csv",
    ]
    labels_csv = next((p for p in labels_csv_candidates if p.exists()), None)
    timestamps_file = frames_dir / "timestamps.txt"

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    if labels_csv is None:
        raise RuntimeError(
            "Missing labels file. Checked: "
            + ", ".join(str(p) for p in labels_csv_candidates)
        )
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
        path_mode=args.path_mode,
        path_base=args.dvs_root.resolve(),
    )
    if len(clips) == 0:
        raise RuntimeError("No clips produced. Try smaller clip-len or stride.")

    if args.sample_split == "trainval":
        train, val = split_clips(clips, args.val_ratio, args.seed)
        test = []
    elif args.sample_split == "test":
        train, val = [], []
        test = clips
    else:
        train, val = split_clips(clips, args.val_ratio, args.seed)
        test = list(val)

    return clips, train, val, test


def iter_trial_names(path):
    with path.open("r") as f:
        for line in f:
            name = line.strip()
            if name:
                yield name


def build_global_manifests(args, output_dir):
    train_rows, val_rows, test_rows = [], [], []

    for aedat_name in iter_trial_names(args.train_list):
        sample_name = aedat_name.replace(".aedat", "")
        _, _, _, sample_clips = build_sample_rows(argparse.Namespace(**{**vars(args), "sample_split": "test"}), sample_name)
        train_rows.extend(sample_clips)

    for aedat_name in iter_trial_names(args.val_list):
        sample_name = aedat_name.replace(".aedat", "")
        _, _, _, sample_clips = build_sample_rows(argparse.Namespace(**{**vars(args), "sample_split": "test"}), sample_name)
        val_rows.extend(sample_clips)

    for aedat_name in iter_trial_names(args.test_list):
        sample_name = aedat_name.replace(".aedat", "")
        _, _, _, sample_clips = build_sample_rows(argparse.Namespace(**{**vars(args), "sample_split": "test"}), sample_name)
        test_rows.extend(sample_clips)

    write_manifest(output_dir / "train.csv", train_rows)
    write_manifest(output_dir / "val.csv", val_rows)
    write_manifest(output_dir / "test.csv", test_rows)

    print(f"Wrote manifests to: {output_dir}")
    print(f"Train clips: {len(train_rows)}")
    print(f"Val clips:   {len(val_rows)}")
    print(f"Test clips:  {len(test_rows)}")


def main():
    args = parse_args()
    if args.train_list or args.val_list or args.test_list:
        if not (args.train_list and args.val_list and args.test_list):
            raise RuntimeError("Provide --train-list, --val-list, and --test-list for global mode.")
        if args.output_dir is None:
            raise RuntimeError("Provide --output-dir for global mode.")
        build_global_manifests(args, args.output_dir)
        return

    if args.sample_name is None:
        raise RuntimeError("Provide --sample-name for single-sample mode.")

    if args.output_dir is None:
        output_dir = args.dvs_root / "manifests" / f"{args.sample_name}_{args.clip_len}f"
    else:
        output_dir = args.output_dir

    clips, train, val, test = build_sample_rows(args, args.sample_name)

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
