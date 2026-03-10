import argparse
import numpy as np
from tqdm import tqdm
import pickle
import os
import sparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EvT DVS128 split samples into sparse frame tensors."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to build. Default builds train, val, and test in one run.",
    )
    return parser.parse_args()


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
# Centralized dataset root under <repo>/dev/datasets
_EVT_ROOT = os.path.join(_DEV_ROOT, "datasets", "evt_og")



chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height = width = 128
def build_mode(mode):
    path_dataset_src = os.path.join(_EVT_ROOT, "clean_dataset", mode)
    path_dataset_dst = os.path.join(_EVT_ROOT, "clean_dataset_frames_{}".format(chunk_len_us), mode)

    if not os.path.isdir(path_dataset_src):
        print(f"Skipping split '{mode}': source dir not found at {path_dataset_src}")
        return

    event_files = os.listdir(path_dataset_src)
    os.makedirs(path_dataset_dst, exist_ok=True)

    print(f"Building split: {mode} ({len(event_files)} files)")
    for ef in tqdm(event_files):
        total_events, label = pickle.load(open(os.path.join(path_dataset_src, ef), 'rb'))
        total_events = total_events.astype('int32')
        
        total_chunks = []
        while total_events.shape[0] > 0:
            end_t = total_events[-1][2]
            chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
            if len(chunk_inds) > 4:
                total_chunks.append(total_events[chunk_inds])
            total_events = total_events[:max(1, chunk_inds.min())-1]
        if len(total_chunks) == 0: 
            print('aaa')
            continue
        total_chunks = total_chunks[::-1]
            
        total_frames = []
        for chunk in total_chunks:
            frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'), 
                               np.ones(chunk.shape[0]).astype('int32'), 
                               (height, width, 2))
            total_frames.append(frame)
        total_frames = sparse.stack(total_frames)
        
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')    

        pickle.dump(total_frames, open(os.path.join(path_dataset_dst, ef), 'wb'))


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "all":
        src_root = os.path.join(_EVT_ROOT, "clean_dataset")
        modes = [m for m in ["train", "val", "test"] if os.path.isdir(os.path.join(src_root, m))]
        if not modes:
            modes = ["train", "val", "test"]
    else:
        modes = [args.mode]
    for mode in modes:
        build_mode(mode)
