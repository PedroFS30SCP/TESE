#!/usr/bin/env python3

import timesformer.datasets.ehwgesture  # noqa: F401
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args

from tools.train_net import train


def main():
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)


if __name__ == "__main__":
    main()
