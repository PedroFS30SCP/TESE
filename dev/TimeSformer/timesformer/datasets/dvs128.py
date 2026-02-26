# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import os
import random

import numpy as np
import torch

import timesformer.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Dvs128(torch.utils.data.Dataset):
    """
    DVS128 frame-sequence loader based on clip manifests.

    Expected split files in DATA.PATH_TO_DATA_DIR:
    `train.csv`, `val.csv`, `test.csv` with header:
      clip_id,label,frames

    Where `frames` is a semicolon-separated list of PNG paths.
    """

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], "Split '{}' not supported".format(
            mode
        )
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        else:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        logger.info("Constructing DVS128 {}...".format(mode))
        self._construct_loader()

    def _resolve_path(self, path):
        if os.path.isabs(path) or self.cfg.DATA.PATH_PREFIX == "":
            return path
        return os.path.join(self.cfg.DATA.PATH_PREFIX, path)

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        with open(path_to_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])
                frame_paths = [
                    self._resolve_path(p)
                    for p in row["frames"].split(";")
                    if p.strip()
                ]
                if not frame_paths:
                    continue

                for idx in range(self._num_clips):
                    self._path_to_videos.append(frame_paths)
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)

        assert len(self._path_to_videos) > 0, "Failed to load split {}".format(
            path_to_file
        )
        logger.info(
            "DVS128 dataloader constructed (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3

        frame_list = self._path_to_videos[index]
        label = self._labels[index]
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(frame_list)

        # Uniform segment sampling with random offset for train.
        if video_length <= 1:
            seq = [0] * num_frames
        else:
            seg_size = float(video_length - 1) / num_frames
            seq = []
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                end = max(start, end)
                if self.mode == "train":
                    seq.append(random.randint(start, end))
                else:
                    seq.append((start + end) // 2)

        frames = torch.as_tensor(
            utils.retry_load_images(
                [frame_list[min(frame_id, video_length - 1)] for frame_id in seq],
                self._num_retries,
            )
        )

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        frames = frames.permute(3, 0, 1, 2)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )

        if not self.cfg.MODEL.ARCH in ["vit"]:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            frames = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0,
                    frames.shape[1] - 1,
                    self.cfg.DATA.NUM_FRAMES,
                ).long(),
            )
        return frames, label, index, {}

    def __len__(self):
        return len(self._path_to_videos)
