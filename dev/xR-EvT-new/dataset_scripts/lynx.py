"""
LynX dataset preprocessing (Block 1: Frames Preprocessing)

This script loads PNG frames corresponding to a single gesture instance
defined in gesture_segmentation.csv and converts them into a sequence
of 2-channel (ON/OFF) frames compatible with the xR-EvT pipeline.

IMPORTANT:
- This script is standalone.
- It does NOT modify or integrate with DataModule or training code.
"""

from pathlib import Path
from typing import Mapping, List, Tuple

import numpy as np
from PIL import Image


def preprocess_lynx_gesture(
    row: Mapping,
    dataset_root: str,
    target_size: int = 128,
) -> Tuple[List[np.ndarray], Tuple[int, str]]:
    """
    Preprocess a single LynX gesture instance.

    Parameters
    ----------
    row : Mapping
        One row from gesture_segmentation.csv (e.g. pandas Series).
    dataset_root : str
        Path to the dataset root directory containing:
        - gesture_segmentation.csv
        - subject_X/ directories
    target_size : int, optional
        Spatial resolution to resize frames to (default: 128).

    Returns
    -------
    FramesSeq : list of np.ndarray
        List of frames, each with shape (2, H, W), dtype float32.
        Channel 0 = ON events, Channel 1 = OFF events.
    Label : tuple
        (gesture_type, gesture_name)
    """

    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])

    if end_frame < start_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}"
        )

    subject = row["subject"]
    scenario = row["scenario"]
    gesture_type = int(row["gesture_type"])
    gesture_name = row["gesture_name"]

    dataset_root = Path(dataset_root)

    frames_dir = (
        dataset_root
        / subject
        / scenario
        / f"gesture_{gesture_type}"
        / "frames"
    )

    frames_seq: List[np.ndarray] = []

    for frame_idx in range(start_frame, end_frame + 1):
        frame_path = frames_dir / f"{frame_idx}.png"

        if not frame_path.exists():
            # Missing frames are skipped silently
            continue

        # Load image as RGB
        img = Image.open(frame_path).convert("RGB")

        # Resize using nearest-neighbor to preserve event-like values
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32)

        # Extract channels
        # R -> ON, B -> OFF, G ignored
        on_channel = img_np[:, :, 0] / 255.0
        off_channel = img_np[:, :, 2] / 255.0

        # Stack as (2, H, W), channel-first
        frame = np.stack([on_channel, off_channel], axis=0).astype(np.float32)

        frames_seq.append(frame)

    label = (gesture_type, gesture_name)

    return frames_seq, label


if __name__ == "__main__":
    """
    Simple self-check for manual inspection.

    - Loads gesture_segmentation.csv
    - Processes the first row
    - Prints basic statistics
    """

    import pandas as pd

    # CHANGE THIS PATH TO YOUR LOCAL DATASET LOCATION
    dataset_root = "/path/to/dataset_LynX"

    dataset_root = Path(dataset_root)
    csv_path = dataset_root / "gesture_segmentation.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"gesture_segmentation.csv not found at: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise RuntimeError("gesture_segmentation.csv is empty.")

    row = df.iloc[0]

    frames_seq, label = preprocess_lynx_gesture(
        row=row,
        dataset_root=str(dataset_root),
        target_size=128,
    )

    print("=== LynX Preprocessing Self-Check ===")
    print(f"Number of frames loaded: {len(frames_seq)}")

    if len(frames_seq) > 0:
        first_frame = frames_seq[0]
        print(f"First frame shape: {first_frame.shape}")
        print(
            f"First frame min/max: "
            f"{first_frame.min():.3f} / {first_frame.max():.3f}"
        )

    print(f"Label (gesture_type, gesture_name): {label}")