import argparse
import builtins
import gc
import os
import pickle
import subprocess
import sys

import numpy as np
import sparse
from tqdm import tqdm

try:
    import sparse._sparse_array as _sparse_array_mod
    _sparse_array_mod.int = builtins.int
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EvT EHWGesture split samples into sparse frame tensors."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to build. Default builds train, val, and test in one run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to preprocess per split.",
    )
    parser.add_argument(
        "--single-file",
        default=None,
        help="Internal use: preprocess one specific .pckl file inside a split.",
    )
    return parser.parse_args()


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_EHW_EVT_ROOT = os.path.join(_DEV_ROOT, "datasets", "evt_og", "EHWGesture")


chunk_len_ms = 12
chunk_len_us = chunk_len_ms * 1000
raw_height = 240
raw_width = 320

# Keep EHW closer to the DVS EvT-OG operating regime.
# Native 240x320 would increase the patch grid by ~5x with patch_size=6.
height = 120
width = 160


def _build_single_file(mode, ef):
    path_dataset_src = os.path.join(_EHW_EVT_ROOT, "clean_dataset", mode)
    path_dataset_dst = os.path.join(
        _EHW_EVT_ROOT, f"clean_dataset_frames_{chunk_len_us}", mode
    )
    os.makedirs(path_dataset_dst, exist_ok=True)

    dst_path = os.path.join(path_dataset_dst, ef)
    if os.path.isfile(dst_path):
        return 0

    print(f"[build:{mode}] {ef}", flush=True)

    total_events, label = pickle.load(
        open(os.path.join(path_dataset_src, ef), "rb")
    )
    total_events = np.asarray(total_events, dtype=np.int64)
    if total_events.ndim != 2 or total_events.shape[1] != 4:
        print(f"[warn] Unexpected event shape for {ef}: {total_events.shape}")
        return 0

    # Ensure events are time-ordered before chunk extraction.
    if total_events.shape[0] > 1 and np.any(total_events[1:, 2] < total_events[:-1, 2]):
        order = np.argsort(total_events[:, 2], kind="stable")
        total_events = total_events[order]

    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        start_idx = np.searchsorted(total_events[:, 2], end_t - chunk_len_us, side="left")
        chunk = total_events[start_idx:]
        if chunk.shape[0] > 4:
            total_chunks.append(chunk)
        total_events = total_events[:start_idx]

    if len(total_chunks) == 0:
        print(f"[warn] No valid chunks extracted for {ef}")
        return 0
    total_chunks = total_chunks[::-1]

    total_frames = []
    for chunk in total_chunks:
        y = np.clip(chunk[:, 0] // 2, 0, height - 1)
        x = np.clip(chunk[:, 1] // 2, 0, width - 1)
        p = chunk[:, 3]
        coords = np.stack([y, x, p], axis=1).astype("int32", copy=False)
        coords, counts = np.unique(coords, axis=0, return_counts=True)
        frame = sparse.COO(
            coords.T,
            counts.astype("int32", copy=False),
            (height, width, 2),
        )
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)

    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype("uint8")

    pickle.dump(total_frames, open(dst_path, "wb"))
    del total_events, total_chunks, total_frames
    gc.collect()
    return 0


def build_mode(mode, limit=None):
    path_dataset_src = os.path.join(_EHW_EVT_ROOT, "clean_dataset", mode)
    path_dataset_dst = os.path.join(
        _EHW_EVT_ROOT, f"clean_dataset_frames_{chunk_len_us}", mode
    )

    if not os.path.isdir(path_dataset_src):
        print(f"Skipping split '{mode}': source dir not found at {path_dataset_src}")
        return

    event_files = sorted(os.listdir(path_dataset_src))
    if limit is not None:
        event_files = event_files[:limit]
    os.makedirs(path_dataset_dst, exist_ok=True)

    print(f"Building split: {mode} ({len(event_files)} files)")
    failed = []
    for ef in tqdm(event_files):
        dst_path = os.path.join(path_dataset_dst, ef)
        if os.path.isfile(dst_path):
            continue
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            mode,
            "--single-file",
            ef,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append((ef, result.returncode))
            print(f"[warn] Failed file in split '{mode}': {ef} (return code {result.returncode})", flush=True)

    if failed:
        print(f"[warn] {len(failed)} files failed in split '{mode}':", flush=True)
        for ef, code in failed:
            print(f"  - {ef} (return code {code})", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.single_file is not None:
        raise SystemExit(_build_single_file(args.mode, args.single_file))
    if args.mode == "all":
        src_root = os.path.join(_EHW_EVT_ROOT, "clean_dataset")
        modes = [
            m
            for m in ["train", "val", "test"]
            if os.path.isdir(os.path.join(src_root, m))
        ]
        if not modes:
            modes = ["train", "val", "test"]
    else:
        modes = [args.mode]

    for mode in modes:
        build_mode(mode, limit=args.limit)
