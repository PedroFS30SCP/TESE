import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_generation_ehwgesture import Event_DataModule
from trainer_ehwgesture import EvNetModel, get_best_weights, load_csv_logs_as_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned EvT-OG EHWGesture model on the real test split."
    )
    parser.add_argument(
        "--path-model",
        default="./pretrained_models/ehwgesture_finetune_earlystop",
        help="Path to a specific training run folder, or a parent folder containing run subfolders.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Evaluation device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--weights-metric",
        default="val_acc",
        choices=["val_acc", "val_loss_total", "val_loss_clf"],
        help="Metric used to pick the best checkpoint.",
    )
    parser.add_argument(
        "--weights-mode",
        default="max",
        choices=["max", "min"],
        help="Whether the selected checkpoint metric should be maximized or minimized.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FLOPs estimation if ptflops is unavailable or not needed.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save evaluation JSON/CSV/pickle/PNG artifacts to disk. Disabled by default.",
    )
    return parser.parse_args()


def resolve_run_folder(path_model):
    path_model = Path(path_model)
    if (path_model / "all_params.json").is_file():
        return path_model

    candidates = [
        p for p in path_model.rglob("*")
        if p.is_dir() and (p / "all_params.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No training run with all_params.json found under {path_model}")
    return sorted(candidates)[-1]


def get_param_stats(model):
    params_df = pd.DataFrame(
        [(n.split(".")[0], p.numel() / 1_000_000) for n, p in model.backbone.named_parameters() if p.requires_grad]
    )
    grouped = params_df.groupby(0).sum()
    total_params = grouped.sum().iloc[0]
    pos_encoding_params = grouped.loc["pos_encoding"].iloc[0] if "pos_encoding" in grouped.index else 0.0
    return {
        "total_params_m": float(total_params),
        "backbone_params_m": float(total_params - pos_encoding_params),
        "pos_encoding_params_m": float(pos_encoding_params),
    }


def plot_training_evolution(path_model, output_path=None):
    logs = load_csv_logs_as_df(path_model)
    if logs.empty:
        return None

    lr_cols = [c for c in logs.columns if "lr" in c]
    lr_col = lr_cols[0] if lr_cols else None
    val_acc = logs[~logs["val_acc"].isna()]["val_acc"] if "val_acc" in logs else pd.Series(dtype=float)
    val_acc_top5 = (
        logs[~logs["val_acc_top5"].isna()]["val_acc_top5"]
        if "val_acc_top5" in logs
        else pd.Series(dtype=float)
    )

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=200)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    loss_cols = [c for c in logs.columns if c.startswith("val_") and "acc" not in c]
    for c in loss_cols:
        series = logs[~logs[c].isna()][c]
        if len(series) > 0:
            ax1.plot(series.values, label=c)

    if len(val_acc) > 0:
        ax2.plot(val_acc.values, "g", label="val_acc")
        ax2.hlines(val_acc.max(), 0, len(val_acc.values), color="g", linestyle="--", alpha=0.7)
    if len(val_acc_top5) > 0:
        ax2.plot(val_acc_top5.values, color="lime", alpha=0.7, label="val_acc_top5")

    if lr_col is not None:
        lr = logs[~logs[lr_col].isna()][lr_col]
        if len(lr) > 0:
            ax3.plot(lr.values, "r", label=lr_col)

    if "val_loss_total" in logs:
        val_loss = logs[~logs["val_loss_total"].isna()]["val_loss_total"]
        if len(val_loss) > 0:
            ax1.hlines(val_loss.min(), 0, max(1, len(val_acc.values)), color="y", linestyle="--", alpha=0.7)

    ax1.set_ylabel("val_loss", color="b", fontsize=14)
    ax2.set_ylabel("val_acc", color="g", fontsize=14)
    ax3.set_ylabel("lr", color="r", fontsize=14)
    ax3.spines["right"].set_position(("outward", 60))
    plt.title(f"{Path(path_model).name} | EHWGesture EvT-OG training evolution", fontsize=14)
    ax1.legend(loc="upper left")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_figure(df_cm, output_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    im = ax.imshow(df_cm.values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(df_cm.columns)))
    ax.set_yticks(range(len(df_cm.index)))
    ax.set_xticklabels(df_cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(df_cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("EHWGesture Test Confusion Matrix")

    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            value = df_cm.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_per_class_recall(df_cm):
    per_class = []
    for label in df_cm.index:
        per_class.append(
            {
                "class": label,
                "recall": float(df_cm.loc[label, label]),
            }
        )
    return pd.DataFrame(per_class)


def get_complexity_stats(model, all_params, device, skip_flops=False):
    if skip_flops:
        return None, None

    try:
        from ptflops import get_model_complexity_info
    except Exception:
        print(" ** ptflops not available, skipping FLOPs estimation")
        return None, None

    data_params = json.loads(json.dumps(all_params["data_params"]))
    data_params["batch_size"] = 1
    data_params["pin_memory"] = False
    data_params["sample_repetitions"] = 1

    dm = Event_DataModule(**data_params)
    dl = dm.test_dataloader()

    total_flops, total_act_patches = [], []
    for polarity, pixels, labels in tqdm(dl, desc="FLOPs"):
        if polarity is None:
            continue
        polarity, pixels = polarity.to(device), pixels.to(device)
        for ts in range(len(polarity)):
            mask = polarity[ts:ts + 1].sum(-1).sum(0).sum(0) != 0
            if mask.sum() == 0:
                continue
            pol_t = polarity[ts:ts + 1][:, :, mask, :]
            pix_t = pixels[ts:ts + 1][:, :, mask, :]
            macs, _ = get_model_complexity_info(
                model.backbone,
                ({"kv": pol_t, "pixels": pix_t},),
                input_constructor=lambda x: x[0],
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            total_flops.append(2 * macs)
            total_act_patches.append(mask.sum().item())

    if not total_flops:
        return None, None
    return float(np.mean(total_flops)), float(np.mean(total_act_patches))


def evaluate_test_split(model, all_params, device):
    data_params = json.loads(json.dumps(all_params["data_params"]))
    data_params["batch_size"] = 1
    data_params["pin_memory"] = False
    data_params["sample_repetitions"] = 1

    dm = Event_DataModule(**data_params)
    dl = dm.test_dataloader()

    total_time = []
    y_true, y_pred, y_top5 = [], [], []
    total_chunks, total_events = [], []

    for polarity, pixels, labels in tqdm(dl, desc="Test"):
        if polarity is None:
            continue
        polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        total_chunks.append(int(polarity.shape[0]))
        total_events.append(int(polarity.shape[2]))

        start_t = torch.cuda.Event(enable_timing=True) if "cuda" in device and torch.cuda.is_available() else None
        end_t = torch.cuda.Event(enable_timing=True) if start_t is not None else None

        if start_t is not None:
            start_t.record()
        else:
            import time
            wall_t0 = time.time()

        with torch.no_grad():
            _, clf_logits = model(polarity, pixels)

        if end_t is not None:
            end_t.record()
            torch.cuda.synchronize()
            elapsed_ms = start_t.elapsed_time(end_t)
        else:
            import time
            elapsed_ms = (time.time() - wall_t0) * 1000.0

        total_time.append(elapsed_ms / len(polarity))
        y_true.append(labels[0].cpu())
        y_pred.append(clf_logits.argmax().cpu())
        y_top5.append(torch.topk(clf_logits, k=min(5, clf_logits.shape[-1]), dim=-1).indices[0].cpu())

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    test_acc = (y_true == y_pred).float().mean().item()
    test_acc_top5 = torch.stack(
        [top5.eq(label).any().float() for label, top5 in zip(y_true, y_top5)]
    ).mean().item()

    class_mapping = dm.class_mapping
    class_mapping = {k: f"{k}. {v}" for k, v in class_mapping.items()}
    y_true_cm = [class_mapping[int(l)] for l in y_true.numpy()]
    y_pred_cm = [class_mapping[int(l)] for l in y_pred.numpy()]
    labels_cm = sorted(set(y_true_cm), key=lambda x: int(x.split()[0][:-1]))
    cm = confusion_matrix(y_true_cm, y_pred_cm, normalize="true", labels=labels_cm)
    df_cm = pd.DataFrame(cm, index=labels_cm, columns=labels_cm)

    num_samples = len(y_true)
    avg_chunk_ms = float(np.mean(total_time))
    avg_sequence_ms = float(np.mean(np.array(total_time) * np.array(total_chunks)))

    stats = {
        "test_acc": float(test_acc),
        "test_acc_top5": float(test_acc_top5),
        "num_test_samples": int(num_samples),
        "sequence_ms": avg_sequence_ms,
        "chunk_ms": avg_chunk_ms,
        "events_per_chunk": {
            "mean": float(np.mean(total_events)),
            "median": float(np.median(total_events)),
            "p05": float(np.percentile(total_events, 5)),
            "p95": float(np.percentile(total_events, 95)),
        },
        "ms_per_ms": float(avg_chunk_ms / data_params["chunk_len_ms"]),
    }
    return stats, df_cm


def main():
    args = parse_args()
    device = args.device

    path_model = resolve_run_folder(args.path_model)
    path_weights = get_best_weights(path_model, args.weights_metric, args.weights_mode)
    all_params = json.load(open(path_model / "all_params.json", "r"))

    model = EvNetModel.load_from_checkpoint(
        path_weights,
        map_location=torch.device("cpu"),
        **all_params,
    ).eval().to(device)

    param_stats = get_param_stats(model)
    flops, activated_patches = get_complexity_stats(model, all_params, device, skip_flops=args.skip_flops)
    test_stats, df_cm = evaluate_test_split(model, all_params, device)
    per_class_recall_df = compute_per_class_recall(df_cm)

    logs = load_csv_logs_as_df(path_model)
    val_acc = logs[~logs["val_acc"].isna()]["val_acc"].max() if "val_acc" in logs else None
    val_loss = logs[~logs["val_loss_total"].isna()]["val_loss_total"].min() if "val_loss_total" in logs else None
    val_acc_top5 = logs[~logs["val_acc_top5"].isna()]["val_acc_top5"].max() if "val_acc_top5" in logs else None

    summary = {
        "path_model": str(path_model),
        "path_weights": str(path_weights),
        "dataset_name": all_params["data_params"]["dataset_name"],
        "selection_metric": args.weights_metric,
        "selection_mode": args.weights_mode,
        "training_val_acc": None if val_acc is None else float(val_acc),
        "training_val_acc_top5": None if val_acc_top5 is None else float(val_acc_top5),
        "training_val_loss_total": None if val_loss is None else float(val_loss),
        **param_stats,
        "flops_g": None if flops is None else float(flops * 1e-9),
        "avg_activated_patches": activated_patches,
        **test_stats,
    }

    stats_filename = path_model / "stats_test_ehwgesture.json"
    cm_filename = path_model / "confusion_matrix_test_ehwgesture.pckl"
    cm_fig_filename = path_model / "confusion_matrix_test_ehwgesture.png"
    per_class_filename = path_model / "per_class_recall_test_ehwgesture.csv"
    training_curve_path = path_model / "training_evolution_ehwgesture.png"

    if args.save_artifacts:
        json.dump(summary, open(stats_filename, "w"), indent=2)
        pickle.dump(df_cm, open(cm_filename, "wb"))
        per_class_recall_df.to_csv(per_class_filename, index=False)
        save_confusion_matrix_figure(df_cm, cm_fig_filename)
        training_curve_path = plot_training_evolution(path_model, training_curve_path)
    else:
        training_curve_path = plot_training_evolution(path_model, output_path=None)

    print("\n ** EHWGesture evaluation finished")
    print(f" - Model folder: {path_model}")
    print(f" - Checkpoint: {path_weights}")
    print(f" - Model parameters: {summary['total_params_m']:.2f} M | pos_encoding_parameters: {summary['pos_encoding_params_m']:.2f} M | backbone_parameters: {summary['backbone_params_m']:.2f} M")
    if summary["flops_g"] is not None:
        print(f" - Model FLOPs: {summary['flops_g']:.2f} G")
    if summary["avg_activated_patches"] is not None:
        print(f" - Average activated patches in [EHWGesture]: {summary['avg_activated_patches']:.1f}")
    if summary["training_val_acc"] is not None:
        print(f" - Validation accuracy reported during training: {summary['training_val_acc']*100:.2f} %")
    if summary["training_val_acc_top5"] is not None:
        print(f" - Validation top-5 accuracy reported during training: {summary['training_val_acc_top5']*100:.2f} %")
    if summary["training_val_loss_total"] is not None:
        print(f" - Validation loss reported during training: {summary['training_val_loss_total']:.5f}")
    print(f" - Test accuracy: {summary['test_acc']*100:.2f} %")
    print(f" - Test top-5 accuracy: {summary['test_acc_top5']*100:.2f} %")
    print(f" - Average processing time per time-window in device [{device}]: {summary['chunk_ms']:.4f} ms")
    print(" - Per-class recall on the shared test split:")
    for _, row in per_class_recall_df.iterrows():
        print(f"   {row['class']}: {row['recall']*100:.2f} %")
    if args.save_artifacts:
        print(f" - Saved stats: {stats_filename}")
        print(f" - Saved confusion matrix: {cm_filename}")
        print(f" - Saved confusion matrix figure: {cm_fig_filename}")
        print(f" - Saved per-class recall: {per_class_filename}")
        if training_curve_path is not None:
            print(f" - Saved training curve figure: {training_curve_path}")
    else:
        print(" - Artifacts were not saved to disk (`--save-artifacts` not used).")


if __name__ == "__main__":
    main()
