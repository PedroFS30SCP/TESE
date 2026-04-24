#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract RGB frames for TimeSformer from raw EHWGesture MP4 files."
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
        default=Path("../datasets/timesformer/EHWGesture/frames"),
        help="Output root for extracted RGB frames.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        choices=["master", "sub2"],
        default="master",
        help="Which RGB camera stream to use.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Which canonical splits to extract. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frame folders.",
    )
    parser.add_argument(
        "--resize-short-side",
        type=int,
        default=0,
        help="Resize frames so the shorter side has this size before saving. 0 keeps native resolution.",
    )
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg"],
        default="png",
        help="Image format to save extracted frames.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used when --image-ext jpg.",
    )
    return parser.parse_args()


def iter_trial_ids(raw_root: Path, splits):
    if "all" in splits:
        splits = ["train", "val", "test"]
    for split in splits:
        split_file = raw_root / f"trials_to_{split}.txt"
        with split_file.open("r") as f:
            for line in f:
                sample_id = line.strip()
                if sample_id:
                    yield sample_id


def sample_parts(sample_id: str):
    subject, hand, code = sample_id.split("_", 2)
    hand_dir = "Left" if hand.upper() == "LEFT" else "Right"
    return subject, hand, hand_dir, code


def resolve_video(raw_root: Path, sample_id: str, camera: str) -> Path:
    subject, _, hand_dir, code = sample_parts(sample_id)
    return (
        raw_root
        / "DataKinects"
        / subject
        / hand_dir
        / "rgb"
        / f"Prova_{code}"
        / f"{camera}_{code}.mp4"
    )


def select_video(raw_root: Path, sample_id: str, preferred_camera: str):
    preferred = resolve_video(raw_root, sample_id, preferred_camera)
    cameras = [preferred_camera]
    if preferred_camera == "master":
        cameras.append("sub2")
    elif preferred_camera == "sub2":
        cameras.append("master")

    for camera in cameras:
        video_path = resolve_video(raw_root, sample_id, camera)
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            return video_path, camera

    return preferred, preferred_camera


def resize_short_side(frame, short_side: int):
    if short_side <= 0:
        return frame
    height, width = frame.shape[:2]
    current_short = min(height, width)
    if current_short == short_side:
        return frame
    scale = float(short_side) / float(current_short)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_video(video_path: Path, output_dir: Path, overwrite: bool, resize_short_side_px: int, image_ext: str, jpeg_quality: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = output_dir / "timestamps.txt"
    existing_frames = list(output_dir.glob(f"frame_*.{image_ext}"))
    if existing_frames and timestamps_path.exists() and not overwrite:
        return

    if overwrite:
        for ext in ("png", "jpg"):
            for p in output_dir.glob(f"frame_*.{ext}"):
                p.unlink()
        if timestamps_path.exists():
            timestamps_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    timestamps = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_sec = (ts_ms / 1000.0) if ts_ms and ts_ms > 0 else (frame_idx / fps)
        frame = resize_short_side(frame, resize_short_side_px)
        frame_path = output_dir / f"frame_{frame_idx:010d}.{image_ext}"
        if image_ext == "jpg":
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            cv2.imwrite(str(frame_path), frame)
        timestamps.append(ts_sec)
        frame_idx += 1

    cap.release()

    with timestamps_path.open("w") as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")


def main():
    args = parse_args()
    seen = set()
    failed = []
    for sample_id in iter_trial_ids(args.raw_root, args.splits):
        if sample_id in seen:
            continue
        seen.add(sample_id)
        video_path, used_camera = select_video(args.raw_root, sample_id, args.camera)
        if not video_path.exists():
            failed.append((sample_id, f"Missing video: {video_path}"))
            print(f"[warn] Missing video for {sample_id}: {video_path}")
            continue
        subject, _, hand_dir, _ = sample_parts(sample_id)
        out_dir = args.output_root / subject / hand_dir / sample_id
        try:
            extract_video(
                video_path,
                out_dir,
                overwrite=args.overwrite,
                resize_short_side_px=args.resize_short_side,
                image_ext=args.image_ext,
                jpeg_quality=args.jpeg_quality,
            )
        except Exception as exc:
            failed.append((sample_id, str(exc)))
            print(f"[warn] Failed {sample_id}: {exc}")
            continue
        suffix = "" if used_camera == args.camera else f" (fallback={used_camera})"
        print(f"Extracted {sample_id} -> {out_dir}{suffix}")

    if failed:
        print(f"[warn] {len(failed)} samples were not extracted cleanly:")
        for sample_id, msg in failed[:20]:
            print(f"  - {sample_id}: {msg}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
