import copy
import json
import os
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.optim import AdamW, lr_scheduler

from data_generation_ehwgesture import Event_DataModule
from models.EvT import CLFBlock, EvNetBackbone
import training_utils


class EvNetModel(LightningModule):
    def __init__(self, backbone_params, clf_params, optim_params, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_params = backbone_params
        self.clf_params = clf_params
        self.optim_params = optim_params

        self.backbone = EvNetBackbone(**backbone_params)
        self.clf_params["ipt_dim"] = self.backbone_params["embed_dim"]
        self.models_clf = nn.ModuleDict([[str(0), CLFBlock(**self.clf_params)]])

        self.loss_weights = loss_weights
        self.init_optimizers()

    def init_optimizers(self):
        self.criterion = nn.NLLLoss(weight=self.loss_weights)

    def _accuracy_scores(self, clf_logits, y):
        preds = torch.argmax(clf_logits, dim=-1)
        top1_acc = (preds == y).float().mean()
        top5_idx = torch.topk(clf_logits, k=min(5, clf_logits.shape[-1]), dim=-1).indices
        top5_acc = top5_idx.eq(y.unsqueeze(-1)).any(dim=-1).float().mean()
        return top1_acc, top5_acc

    def forward(self, x, pixels):
        embs = self.backbone(x, pixels)
        clf_logits = torch.stack([self.models_clf[str(0)](embs)]).mean(axis=0)
        return embs, clf_logits

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), **self.optim_params["optim_params"])
        if "scheduler" in self.optim_params:
            if self.optim_params["scheduler"]["name"] == "lr_on_plateau":
                sched = lr_scheduler.ReduceLROnPlateau(
                    optim, **self.optim_params["scheduler"]["params"]
                )
            elif self.optim_params["scheduler"]["name"] == "one_cycle_lr":
                sched = lr_scheduler.OneCycleLR(
                    optim,
                    max_lr=self.optim_params["optim_params"]["lr"],
                    **self.optim_params["scheduler"]["params"],
                )
            else:
                return optim
            return {
                "optimizer": optim,
                "lr_scheduler": sched,
                "monitor": self.optim_params["monitor"],
            }
        return optim

    def step(self, polarity, pixels, y):
        embs, clf_logits = self(polarity, pixels)
        loss_clf = self.criterion(clf_logits, y)
        acc, acc_top5 = self._accuracy_scores(clf_logits, y)
        return {
            "loss_clf": loss_clf,
            "acc": acc,
            "acc_top5": acc_top5,
            "loss_total": loss_clf,
        }

    def training_step(self, batch, batch_idx):
        polarity, pixels, y = batch
        losses = self.step(polarity, pixels, y)
        for k, v in losses.items():
            self.log(
                f"train_{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return losses["loss_total"]

    def validation_step(self, batch, batch_idx):
        polarity, pixels, y = batch
        losses = self.step(polarity, pixels, y)
        for k, v in losses.items():
            self.log(
                f"val_{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return losses["loss_total"]


class SafeCSVLogger(CSVLogger):
    def save(self):
        # CSV metric logging is what we care about here. On this environment,
        # Lightning's repeated YAML hyperparameter save can crash inside PyYAML
        # after successful epochs. Keep metrics/checkpoints flowing by skipping
        # the brittle hparams.yaml rewrite on save().
        if self._experiment is not None and hasattr(self._experiment, "hparams"):
            original_hparams = self._experiment.hparams
            self._experiment.hparams = {}
            try:
                return super().save()
            finally:
                self._experiment.hparams = original_hparams
        return super().save()


def load_csv_logs_as_df(path_model):
    log_file = os.path.join(path_model, "train_log", "version_0", "metrics.csv")
    logs = pd.read_csv(log_file)
    for i, row in logs[logs.epoch.isna()].iterrows():
        candidates = logs[(~logs.epoch.isna()) & (logs.step >= int(row.step))].epoch.min()
        logs.loc[i, "epoch"] = candidates
    return logs


def get_best_weights(path_model, metric="val_acc", mode="max"):
    assert mode in ["min", "max"]
    weight_dir = os.path.join(path_model, "weights")
    weight_files = os.listdir(weight_dir)

    def extract_metric(filename):
        match = re.search(rf"{re.escape(metric)}=([-+]?\d*\.\d+|\d+)", filename)
        if match is None:
            raise ValueError(f"Metric [{metric}] not found in checkpoint filename: {filename}")
        return float(match.group(1))

    chooser = max if mode == "max" else min
    best_filename = chooser(weight_files, key=extract_metric)
    return os.path.join(weight_dir, best_filename)


def load_pretrained_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    current_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key.startswith("models_clf."):
            skipped.append((key, "classifier_head"))
            continue
        if key == "backbone.pos_encoding":
            skipped.append((key, "positional_encoding"))
            continue
        if key not in current_state:
            skipped.append((key, "missing_target"))
            continue
        if current_state[key].shape != value.shape:
            skipped.append((key, f"shape_mismatch {tuple(value.shape)} -> {tuple(current_state[key].shape)}"))
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f" ** Loaded pretrained weights from: {checkpoint_path}")
    print(f" - Matched tensors: {len(filtered_state)}")
    print(f" - Missing after load: {len(missing)}")
    print(f" - Unexpected after load: {len(unexpected)}")
    print(f" - Skipped tensors: {len(skipped)}")
    if skipped:
        print(" - Example skipped keys:")
        for key, reason in skipped[:10]:
            print(f"   {key} ({reason})")


def train(
    folder_name,
    path_results,
    data_params,
    backbone_params,
    clf_params,
    training_params,
    optim_params,
    callback_params,
    logger_params,
    pretrained_ckpt_path=None,
    resume_ckpt_path=None,
):
    if resume_ckpt_path is not None:
        path_model = os.path.dirname(os.path.dirname(os.path.abspath(resume_ckpt_path))) + "/"
    else:
        path_model = training_utils.create_model_folder(path_results, folder_name)

    callbacks = []
    for k, params in callback_params:
        if k == "early_stopping":
            callbacks.append(EarlyStopping(**params))
        if k == "lr_monitor":
            callbacks.append(LearningRateMonitor(**params))
        if k == "model_chck":
            params = copy.deepcopy(params)
            params["dirpath"] = params["dirpath"].format(path_model)
            params.pop("period", None)
            callbacks.append(ModelCheckpoint(**params))

    loggers = []
    if "csv" in logger_params:
        logger_params["csv"]["save_dir"] = logger_params["csv"]["save_dir"].format(path_model)
        loggers.append(SafeCSVLogger(**logger_params["csv"]))

    dm = Event_DataModule(**data_params)
    backbone_params["token_dim"] = dm.token_dim
    clf_params["opt_classes"] = dm.num_classes

    if "pos_encoding" in backbone_params:
        # EvT indexes positional encodings as pos_encoding[x, y], so rectangular
        # datasets need (width, height) here even though the frame tensors are HxW.
        backbone_params["pos_encoding"]["params"]["shape"] = (dm.width, dm.height)
    if backbone_params["downsample_pos_enc"] == -1:
        backbone_params["downsample_pos_enc"] = data_params["patch_size"]

    if optim_params["scheduler"]["name"] == "one_cycle_lr":
        optim_params["scheduler"]["params"]["steps_per_epoch"] = 1

    model = EvNetModel(
        backbone_params=copy.deepcopy(backbone_params),
        clf_params=copy.deepcopy(clf_params),
        optim_params=copy.deepcopy(optim_params),
        loss_weights=None if not data_params["balance"] else dm.train_dataloader().dataset.get_class_weights(),
    )
    if pretrained_ckpt_path and resume_ckpt_path is None:
        load_pretrained_backbone(model, pretrained_ckpt_path)

    trainer = Trainer(**training_params, callbacks=callbacks, logger=loggers)

    if resume_ckpt_path is None:
        json.dump(
            {
                "data_params": data_params,
                "backbone_params": backbone_params,
                "clf_params": clf_params,
                "training_params": training_params,
                "optim_params": optim_params,
                "callbacks_params": callback_params,
                "logger_params": logger_params,
            },
            open(path_model + "all_params.json", "w"),
        )

    trainer.fit(model, dm, ckpt_path=resume_ckpt_path)

    print(" ** Train finished:", path_model)
    logs = load_csv_logs_as_df(path_model)
    val_acc = logs[~logs["val_acc"].isna()]["val_acc"]
    if len(val_acc) > 0:
        print(" - Max. Accuracy: {:.4f}".format(val_acc.values.max()))

    for c in [c for c in logs.columns if "val_" in c and "acc" not in c]:
        v = logs[~logs[c].isna()][c]
        v = v.values.min() if len(v) > 0 else 0.0
        print(" - Min. [{}]: {:.4f}".format(c, v))
    print("path_model = '{}'".format(path_model))

    return path_model
