import numpy as np
from tqdm import tqdm
import pickle
import os
import sparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
# Centralized dataset root under <repo>/dev/datasets
_EVT_ROOT = os.path.join(_DEV_ROOT, "datasets", "evt_og")



chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height = width = 128
mode = 'train'

# Read dataset filenames
if mode == 'train':
    path_dataset_src = os.path.join(_EVT_ROOT, "clean_dataset", "train")
    path_dataset_dst = os.path.join(_EVT_ROOT, "clean_dataset_frames_{}".format(chunk_len_us), "train")
else:
    path_dataset_src = os.path.join(_EVT_ROOT, "clean_dataset", "test")
    path_dataset_dst = os.path.join(_EVT_ROOT, "clean_dataset_frames_{}".format(chunk_len_us), "test")

event_files = os.listdir(path_dataset_src)
if not os.path.isdir(path_dataset_dst):
    os.makedirs(path_dataset_dst)


# %%


for ef in tqdm(event_files):
    
    total_events, label = pickle.load(open(os.path.join(path_dataset_src, ef), 'rb'))
    total_events = total_events.astype('int32')
    
    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4: 
            pass
        else:
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
                           (height, width, 2))   # .to_dense()
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)
    
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')    

    pickle.dump(total_frames, open(os.path.join(path_dataset_dst, ef), 'wb'))
    

