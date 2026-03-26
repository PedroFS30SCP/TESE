# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import json
import numpy as np
import os
import pprint
import threading
import time
import torch
import urllib.request
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import timesformer.models.losses as losses
import timesformer.models.optimizer as optim
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter
from timesformer.utils.multigrid import MultigridSchedule

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

logger = logging.get_logger(__name__)


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct_from_err(err_value):
    err = _safe_float(err_value)
    if err is None:
        return "n/a"
    return "{:.1f}%".format(100.0 - err)


def _format_loss(loss_value):
    loss = _safe_float(loss_value)
    if loss is None:
        return "n/a"
    return "{:.3f}".format(loss)


def _format_hours_minutes(total_seconds):
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if hours > 0:
        return "{}h {:02d}min".format(hours, minutes)
    return "{}min".format(minutes)


def _format_eta_text(eta_value):
    if not eta_value:
        return "n/a"
    parts = str(eta_value).split(":")
    if len(parts) != 3:
        return str(eta_value)
    hours = int(parts[0])
    minutes = int(parts[1])
    if hours > 0:
        return "{}h {:02d}min".format(hours, minutes)
    return "{}min".format(minutes)


def _ntfy_enabled(cfg):
    return (
        du.is_master_proc()
        and getattr(cfg.NTFY, "ENABLE", False)
        and bool(getattr(cfg.NTFY, "TOPIC", "").strip())
    )


def _send_ntfy(cfg, title, message, priority="default", tags=None):
    if not _ntfy_enabled(cfg):
        return

    topic = cfg.NTFY.TOPIC.strip()
    timeout = float(cfg.NTFY.TIMEOUT_SEC)
    body = message.encode("utf-8", errors="replace")
    headers = {
        "Title": title[:80],
        "Priority": priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if tags:
        headers["Tags"] = ",".join(tags)

    def _worker():
        try:
            req = urllib.request.Request(
                "https://ntfy.sh/{}".format(topic),
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout):
                pass
        except Exception:
            # Notifications must never interfere with training.
            pass

    threading.Thread(target=_worker, daemon=True).start()


def _is_better_metric(cur_value, best_value, mode, min_delta):
    if best_value is None:
        return True
    if mode == "min":
        return cur_value < (best_value - min_delta)
    if mode == "max":
        return cur_value > (best_value + min_delta)
    raise ValueError("Unsupported early stopping mode '{}'".format(mode))


def _write_training_state(output_dir, state):
    state_path = os.path.join(output_dir, "training_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        if not cfg.MIXUP.ENABLED:
           loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        else:
           mixup_fn = Mixup(
               mixup_alpha=cfg.MIXUP.ALPHA, cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX, prob=cfg.MIXUP.PROB, switch_prob=cfg.MIXUP.SWITCH_PROB, mode=cfg.MIXUP.MODE,
               label_smoothing=0.1, num_classes=cfg.MODEL.NUM_CLASSES)
           hard_labels = labels
           inputs, labels = mixup_fn(inputs, labels)
           loss_fun = SoftTargetCrossEntropy()

        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        else:
            preds = model(inputs)

        # Compute the loss.
        loss = loss_fun(preds, labels)

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

        # check Nan Loss.
        misc.check_nan_losses(loss)


        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    stats = train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return stats


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                loss = loss_fun(preds, labels)
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss.item(),
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    stats = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()
    return stats


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    train_start_time = time.perf_counter()
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
      start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
      start_epoch = 0
      cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    _send_ntfy(
        cfg,
        "Training started!",
        "Training started!\nEpochs: {} | LR: {:.6f}".format(
            cfg.SOLVER.MAX_EPOCH, float(cfg.SOLVER.BASE_LR)
        ),
        priority="default",
        tags=["rocket", "computer"],
    )

    es_cfg = cfg.TRAIN.EARLY_STOPPING
    early_stopping_enabled = es_cfg.ENABLE
    best_metric = None
    best_epoch = None
    best_stats = None
    epochs_without_improvement = 0
    best_val_loss = None
    best_val_acc = None
    best_val_acc_epoch = None
    best_val_loss_epoch = None
    overfit_alert_active = False
    prev_train_loss = None

    try:
        for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            if cfg.MULTIGRID.LONG_CYCLE:
                cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
                if changed:
                    (
                        model,
                        optimizer,
                        train_loader,
                        val_loader,
                        precise_bn_loader,
                        train_meter,
                        val_meter,
                    ) = build_trainer(cfg)

                    # Load checkpoint.
                    if cu.has_checkpoint(cfg.OUTPUT_DIR):
                        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                        assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                    else:
                        last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                    logger.info("Load from {}".format(last_checkpoint))
                    cu.load_checkpoint(
                        last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                    )

            # Shuffle the dataset.
            loader.shuffle_dataset(train_loader, cur_epoch)

            # Train for one epoch.
            train_stats = train_epoch(
                train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
            )

            is_checkp_epoch = cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            is_eval_epoch = misc.is_eval_epoch(
                cfg, cur_epoch, None if multigrid is None else multigrid.schedule
            )

            # Compute precise BN stats.
            if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
            ):
                calculate_and_update_precise_bn(
                    precise_bn_loader,
                    model,
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                    cfg.NUM_GPUS > 0,
                )
            _ = misc.aggregate_sub_bn_stats(model)

            # Save an epoch checkpoint when scheduled and always refresh the last checkpoint.
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                checkpoint_name="checkpoint_last.pyth",
            )
            # Evaluate the model on validation set.
            val_stats = None
            if is_eval_epoch:
                val_stats = eval_epoch(
                    val_loader, model, val_meter, cur_epoch, cfg, writer
                )

                val_loss = _safe_float(val_stats.get("loss"))
                val_acc = _safe_float(100.0 - float(val_stats["top1_err"])) if "top1_err" in val_stats else None
                train_loss = _safe_float(train_stats.get("loss"))

                if val_loss is not None and (
                    best_val_loss is None or val_loss < best_val_loss
                ):
                    best_val_loss = val_loss
                    best_val_loss_epoch = cur_epoch + 1
                    overfit_alert_active = False
                if val_acc is not None and (
                    best_val_acc is None or val_acc > best_val_acc
                ):
                    best_val_acc = val_acc
                    best_val_acc_epoch = cur_epoch + 1

                every_n_epochs = int(cfg.NTFY.EVERY_N_EPOCHS)
                if every_n_epochs > 0 and (cur_epoch + 1) % every_n_epochs == 0:
                    _send_ntfy(
                        cfg,
                        "Epoch {}/{}".format(cur_epoch + 1, cfg.SOLVER.MAX_EPOCH),
                        "Epoch {}/{}\n"
                        "Train Loss: {} | Val Loss: {}\n"
                        "Train Acc: {} | Val Acc: {}\n"
                        "ETA: {}".format(
                            cur_epoch + 1,
                            cfg.SOLVER.MAX_EPOCH,
                            _format_loss(train_stats.get("loss")),
                            _format_loss(val_stats.get("loss")),
                            _format_pct_from_err(train_stats.get("top1_err")),
                            _format_pct_from_err(val_stats.get("top1_err")),
                            _format_eta_text(train_stats.get("eta")),
                        ),
                        priority="default",
                        tags=["chart_with_upwards_trend"],
                    )

                if (
                    best_val_loss is not None
                    and val_loss is not None
                    and train_loss is not None
                    and prev_train_loss is not None
                    and val_loss >= best_val_loss + float(cfg.NTFY.OVERFIT_VAL_LOSS_DELTA)
                    and train_loss < prev_train_loss
                    and not overfit_alert_active
                ):
                    _send_ntfy(
                        cfg,
                        "Possible overfitting detected!",
                        "Possible overfitting at epoch {}!\n"
                        "Train Loss: {} | Val Loss: {}".format(
                            cur_epoch + 1,
                            _format_loss(train_loss),
                            _format_loss(val_loss),
                        ),
                        priority="high",
                        tags=["warning"],
                    )
                    overfit_alert_active = True

                prev_train_loss = train_loss

                if early_stopping_enabled:
                    monitor_name = es_cfg.MONITOR
                    if monitor_name not in val_stats:
                        raise KeyError(
                            "Early stopping monitor '{}' not found in val stats: {}".format(
                                monitor_name, sorted(val_stats.keys())
                            )
                        )

                    current_metric = float(val_stats[monitor_name])
                    if _is_better_metric(
                        current_metric,
                        best_metric,
                        es_cfg.MODE,
                        es_cfg.MIN_DELTA,
                    ):
                        best_metric = current_metric
                        best_epoch = cur_epoch + 1
                        best_stats = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_stats.items()}
                        epochs_without_improvement = 0
                        cu.save_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            cur_epoch,
                            cfg,
                            checkpoint_name="checkpoint_best.pyth",
                        )
                        logger.info(
                            "New best checkpoint at epoch {} with {}={:.5f}".format(
                                best_epoch, monitor_name, current_metric
                            )
                        )
                    else:
                        epochs_without_improvement += 1

                    _write_training_state(
                        cfg.OUTPUT_DIR,
                        {
                            "best_checkpoint": os.path.join(
                                cfg.OUTPUT_DIR, "checkpoints", "checkpoint_best.pyth"
                            ),
                            "last_checkpoint": os.path.join(
                                cfg.OUTPUT_DIR, "checkpoints", "checkpoint_last.pyth"
                            ),
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "epochs_without_improvement": epochs_without_improvement,
                            "monitor": monitor_name,
                            "mode": es_cfg.MODE,
                            "min_delta": es_cfg.MIN_DELTA,
                            "patience": es_cfg.PATIENCE,
                            "best_val_stats": best_stats,
                        },
                    )

                    if epochs_without_improvement >= es_cfg.PATIENCE:
                        logger.info(
                            "Early stopping triggered at epoch {} after {} stale eval epochs. Best epoch was {} ({}={:.5f}).".format(
                                cur_epoch + 1,
                                epochs_without_improvement,
                                best_epoch,
                                monitor_name,
                                best_metric,
                            )
                        )
                        break

        total_time = _format_hours_minutes(time.perf_counter() - train_start_time)
        _send_ntfy(
            cfg,
            "Training complete!",
            "Training complete! Total time: {}\n"
            "Best Val Acc: {} at epoch {}\n"
            "Best Val Loss: {} at epoch {}".format(
                total_time,
                "n/a" if best_val_acc is None else "{:.1f}%".format(best_val_acc),
                "n/a" if best_val_acc_epoch is None else best_val_acc_epoch,
                "n/a" if best_val_loss is None else "{:.3f}".format(best_val_loss),
                "n/a" if best_val_loss_epoch is None else best_val_loss_epoch,
            ),
            priority="default",
            tags=["white_check_mark", "tada"],
        )
    except KeyboardInterrupt:
        _send_ntfy(
            cfg,
            "Training interrupted!",
            "Training interrupted at epoch {}!".format(
                cur_epoch + 1 if "cur_epoch" in locals() else start_epoch + 1
            ),
            priority="high",
            tags=["warning", "hand"],
        )
        raise
    except Exception as exc:
        error_text = "{}: {}".format(type(exc).__name__, str(exc)).strip()
        _send_ntfy(
            cfg,
            "Training crashed!",
            "Training crashed at epoch {}!\nError: {}".format(
                cur_epoch + 1 if "cur_epoch" in locals() else start_epoch + 1,
                error_text[:300],
            ),
            priority="urgent",
            tags=["rotating_light", "warning"],
        )
        raise
    finally:
        if writer is not None:
            writer.close()
