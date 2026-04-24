#!/usr/bin/env python3

import argparse
import csv
import math
import random
from pathlib import Path

import cv2


AUTHOR_CROPS = {
    "master": (400, 200, 1200, 600),
    "sub2": (400, 250, 1200, 650),
}


def parse_crop(value: str):
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("Crop right/bottom must exceed left/top")
    return left, top, right, bottom


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract EHWGesture RGB frames from Kinect MP4 files using the "
            "fixed crops from the original EHWGesture preprocessing, then "
            "resize to a controlled benchmark resolution."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("../datasets/raw/EHWGesture"),
        help="Raw EHWGesture root containing DataKinects and split txt files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../datasets/timesformer/EHWGesture/crop_frames"),
        help="Output root for cropped frames.",
    )
    parser.add_argument(
        "--camera",
        choices=["master", "sub2", "both"],
        default="master",
        help="Kinect RGB stream to extract. Use both to write camera subfolders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Canonical splits to extract. Default: all.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of each selected split to extract. Use 0.1 for a 10%% pilot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when --fraction is below 1.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Optional hard cap per split after fraction sampling. 0 keeps all selected samples.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Output frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Output frame height.",
    )
    parser.add_argument(
        "--master-crop",
        type=parse_crop,
        default=AUTHOR_CROPS["master"],
        help="Crop for master RGB videos as left,top,right,bottom.",
    )
    parser.add_argument(
        "--sub2-crop",
        type=parse_crop,
        default=AUTHOR_CROPS["sub2"],
        help="Crop for sub2 RGB videos as left,top,right,bottom.",
    )
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg"],
        default="png",
        help="Image format to save.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used when --image-ext jpg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frame folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected samples without writing frames.",
    )
    return parser.parse_args()


def selected_splits(splits):
    if "all" in splits:
        return ["train", "val", "test"]
    return splits


def load_split_ids(raw_root: Path, split: str):
    split_file = raw_root / f"trials_to_{split}.txt"
    with split_file.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def select_fraction(sample_ids, fraction: float, seed: int, split: str):
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in the interval (0, 1]")
    if fraction >= 1:
        return list(sample_ids)
    count = max(1, int(math.ceil(len(sample_ids) * fraction)))
    rng = random.Random(f"{seed}:{split}")
    selected = rng.sample(sample_ids, count)
    return sorted(selected)


def sample_parts(sample_id: str):
    subject, hand, code = sample_id.split("_", 2)
    hand_dir = "Left" if hand.upper() == "LEFT" else "Right"
    return subject, hand_dir, code


def resolve_video(raw_root: Path, sample_id: str, camera: str):
    subject, hand_dir, code = sample_parts(sample_id)
    return (
        raw_root
        / "DataKinects"
        / subject
        / hand_dir
        / "rgb"
        / f"Prova_{code}"
        / f"{camera}_{code}.mp4"
    )


def output_dir_for(output_root: Path, sample_id: str, camera: str, use_camera_subdir: bool):
    subject, hand_dir, _ = sample_parts(sample_id)
    root = output_root / camera if use_camera_subdir else output_root
    return root / subject / hand_dir / sample_id


def crop_and_resize(frame, crop, width: int, height: int):
    frame_h, frame_w = frame.shape[:2]
    left, top, right, bottom = crop
    left = max(0, min(left, frame_w))
    right = max(0, min(right, frame_w))
    top = max(0, min(top, frame_h))
    bottom = max(0, min(bottom, frame_h))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop after clamping: {(left, top, right, bottom)}")
    cropped = frame[top:bottom, left:right]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)


def clear_existing(output_dir: Path):
    for ext in ("png", "jpg"):
        for path in output_dir.glob(f"frame_*.{ext}"):
            path.unlink()
    timestamps = output_dir / "timestamps.txt"
    if timestamps.exists():
        timestamps.unlink()


def extract_video(
    video_path: Path,
    output_dir: Path,
    crop,
    width: int,
    height: int,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = output_dir / "timestamps.txt"
    existing_frames = list(output_dir.glob(f"frame_*.{image_ext}"))
    if existing_frames and timestamps_path.exists() and not overwrite:
        return len(existing_frames), False, "skipped_existing"

    if overwrite:
        clear_existing(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    timestamps = []
    frame_idx = 0
    written_count = 0
    frame_errors = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_sec = (ts_ms / 1000.0) if ts_ms and ts_ms > 0 else (frame_idx / fps)
        try:
            processed = crop_and_resize(frame, crop, width=width, height=height)
            frame_path = output_dir / f"frame_{frame_idx:010d}.{image_ext}"
            if image_ext == "jpg":
                wrote_frame = cv2.imwrite(
                    str(frame_path),
                    processed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
            else:
                wrote_frame = cv2.imwrite(str(frame_path), processed)
            if not wrote_frame:
                raise RuntimeError(f"cv2.imwrite returned false for {frame_path}")
        except Exception as exc:
            frame_errors.append(f"frame_{frame_idx:010d}: {exc}")
            frame_idx += 1
            continue
        timestamps.append(ts_sec)
        written_count += 1
        frame_idx += 1

    cap.release()

    with timestamps_path.open("w") as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")
    if frame_errors:
        preview = "; ".join(frame_errors[:5])
        if len(frame_errors) > 5:
            preview += f"; ... {len(frame_errors) - 5} more"
        return written_count, True, f"Fail ({len(frame_errors)} frame errors: {preview})"
    return written_count, True, "wrote"


def write_summary(output_root: Path, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "extraction_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "sample_id",
                "camera",
                "video_path",
                "output_dir",
                "frames",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main():
    args = parse_args()
    cameras = ["master", "sub2"] if args.camera == "both" else [args.camera]
    crops = {
        "master": args.master_crop,
        "sub2": args.sub2_crop,
    }
    use_camera_subdir = len(cameras) > 1

    rows = []
    failed = []
    total_selected = 0
    for split in selected_splits(args.splits):
        sample_ids = load_split_ids(args.raw_root, split)
        sample_ids = select_fraction(sample_ids, args.fraction, args.seed, split)
        if args.limit_per_split > 0:
            sample_ids = sample_ids[: args.limit_per_split]
        total_selected += len(sample_ids)
        print(f"{split}: selected {len(sample_ids)} samples")

        for sample_id in sample_ids:
            for camera in cameras:
                video_path = resolve_video(args.raw_root, sample_id, camera)
                out_dir = output_dir_for(
                    args.output_root,
                    sample_id,
                    camera,
                    use_camera_subdir=use_camera_subdir,
                )
                if args.dry_run:
                    print(f"[dry-run] {split} {camera} {sample_id}: {video_path}")
                    continue
                if not video_path.exists():
                    msg = f"Fail (Missing video: {video_path})"
                    failed.append((split, sample_id, camera, msg))
                    print(f"[warn] {sample_id} {camera}: {msg}")
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "camera": camera,
                            "video_path": str(video_path),
                            "output_dir": str(out_dir),
                            "frames": 0,
                            "status": msg,
                        }
                    )
                    continue
                try:
                    frame_count, wrote, status = extract_video(
                        video_path=video_path,
                        output_dir=out_dir,
                        crop=crops[camera],
                        width=args.width,
                        height=args.height,
                        image_ext=args.image_ext,
                        jpeg_quality=args.jpeg_quality,
                        overwrite=args.overwrite,
                    )
                    print(f"{status}: {split} {camera} {sample_id} ({frame_count} frames)")
                    if status.startswith("Fail"):
                        failed.append((split, sample_id, camera, status))
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "camera": camera,
                            "video_path": str(video_path),
                            "output_dir": str(out_dir),
                            "frames": frame_count,
                            "status": status,
                        }
                    )
                except Exception as exc:
                    status = f"Fail ({exc})"
                    failed.append((split, sample_id, camera, status))
                    print(f"[warn] Failed {split} {camera} {sample_id}: {exc}")
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "camera": camera,
                            "video_path": str(video_path),
                            "output_dir": str(out_dir),
                            "frames": 0,
                            "status": status,
                        }
                    )

    if not args.dry_run:
        summary_path = write_summary(args.output_root, rows)
        print(f"Summary: {summary_path}")
    print(f"Selected samples: {total_selected}")
    print(f"Camera streams: {len(cameras)}")
    if failed:
        print(f"[warn] Failures: {len(failed)}")
        for split, sample_id, camera, msg in failed[:20]:
            print(f"  - {split} {camera} {sample_id}: {msg}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
