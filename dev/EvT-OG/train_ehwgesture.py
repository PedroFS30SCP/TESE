import argparse
import copy
import json
import os

from trainer_ehwgesture import get_best_weights, train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune EvT-OG on the full EHWGesture dataset with early stopping."
    )
    parser.add_argument(
        "--pretrained-model-dir",
        default="./pretrained_models/DVS128_11_24ms_dwn",
        help="Directory containing the pretrained DVS128-11 model and weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. Reduce if you hit memory issues.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--val-workers",
        type=int,
        default=0,
        help="Validation dataloader workers. Default 0 avoids sparse worker segfaults seen on EHWGesture.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=80,
        help="Maximum fine-tuning epochs before early stopping.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Lightning GPU setting. Use 0 for CPU-only runs.",
    )
    parser.add_argument(
        "--output-name",
        default="/ehwgesture_finetune_earlystop",
        help="Output folder name created under ./pretrained_models.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume an interrupted EHWGesture run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path_results = "pretrained_models"
    pretrained_dir = args.pretrained_model_dir
    all_params_path = os.path.join(pretrained_dir, "all_params.json")
    pretrained_ckpt_path = None
    if args.resume_from is None:
        pretrained_ckpt_path = get_best_weights(pretrained_dir, metric="val_acc", mode="max")

    base_params = json.load(open(all_params_path, "r"))
    train_params = copy.deepcopy(base_params)

    train_params["data_params"]["dataset_name"] = "EHWGesture"
    train_params["data_params"]["batch_size"] = args.batch_size
    train_params["data_params"]["sample_repetitions"] = 1
    train_params["data_params"]["workers"] = args.workers
    train_params["data_params"]["val_workers"] = args.val_workers
    train_params["data_params"]["pin_memory"] = True
    train_params["data_params"]["balance"] = False
    train_params["data_params"]["classes_to_exclude"] = []
    train_params["data_params"]["augmentation_params"] = {
        "max_sample_len_ms": 504,
        "random_frame_size": 0.75,
        "random_shift": True,
        "drop_token": [0.1, "random"],
        "h_flip": False,
    }

    train_params["training_params"]["gpus"] = args.gpus
    train_params["training_params"]["max_epochs"] = args.max_epochs
    train_params["training_params"]["log_every_n_steps"] = 50
    train_params["training_params"]["stochastic_weight_avg"] = False

    if train_params["optim_params"]["scheduler"]["name"] == "one_cycle_lr":
        train_params["optim_params"]["scheduler"]["params"]["epochs"] = args.max_epochs
        train_params["optim_params"]["scheduler"]["params"]["steps_per_epoch"] = 1

    train_params["clf_params"]["opt_classes"] = 5
    train_params["logger_params"]["csv"]["save_dir"] = "{}"

    for k, v in train_params["callbacks_params"]:
        if k == "early_stopping":
            v["monitor"] = "val_loss_total"
            v["mode"] = "min"
            v["patience"] = 10
            v["min_delta"] = 0.0001
        if k == "model_chck":
            v["dirpath"] = "{}/weights/"
            v["filename"] = "{epoch}-{val_loss_total:.5f}-{val_loss_clf:.5f}-{val_acc:.5f}"
            v["save_top_k"] = 1

    path_model = train(
        args.output_name,
        path_results,
        data_params=train_params["data_params"],
        backbone_params=train_params["backbone_params"],
        clf_params=train_params["clf_params"],
        training_params=train_params["training_params"],
        optim_params=train_params["optim_params"],
        callback_params=train_params["callbacks_params"],
        logger_params=train_params["logger_params"],
        pretrained_ckpt_path=pretrained_ckpt_path,
        resume_ckpt_path=args.resume_from,
    )
    print("Fine-tuning path:", path_model)


if __name__ == "__main__":
    main()
