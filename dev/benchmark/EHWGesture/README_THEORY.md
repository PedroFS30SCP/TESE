# Pipeline (EvT-OG vs TimeSformer on EHWGesture): Simple File-by-File Explanation

This document explains the files I run in this project:
- what each file does,
- why it does it,
- and whether it is original from the authors, created by me, or modified by me.

## 1) Big Picture

My pipeline has 2 stages:
1. `EvT-OG` fine-tunes and evaluates an event-based model on EHWGesture.
2. `TimeSformer` TODO

---

## 2) Environment

### `dev/ehw_env.sh`
- What it does: sets `TESE_ROOT`, `DEV_ROOT`, `DATA_ROOT`, `EHW_RAW_ROOT`, `EHW_EVENT_ROOT`, `EHW_EVT_ROOT`, and `EHW_TS_ROOT`.
- Why: all EHW benchmark commands can use portable paths and the TimeSformer code can be found through `PYTHONPATH`.
- Status: **Created by me (custom)**.
- Run type: run directly.

---

## 3) EvT-OG

### `dev/EvT-OG/dataset_scripts/ehwgesture_split_dataset.py`
- What it does: splits raw EHWGesture `.aedat4` recordings into benchmark-ready `.pckl` samples for `train`, `val`, and `test`.
- Why: EvT-OG expects already split gesture samples, not full raw event recordings.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - reads the canonical EHW trial-list files instead of using ad hoc splits.
  - maps the original EHW labels into the `5` benchmark classes:
    - `FT`
    - `OC`
    - `PS`
    - `NOSE`
    - `TR`
  - writes one clean `.pckl` sample per recording into `clean_dataset/train`, `clean_dataset/val`, and `clean_dataset/test`.
- Example: `X01_LEFT_FTF1_label01.pckl`
- Filename meaning:
  - `X01_LEFT_FTF1`: original EHW sample identifier
  - `label01`: mapped benchmark class label

### `dev/EvT-OG/dataset_scripts/ehwgesture.py`
- What it does: loads the clean `.pckl` EHW samples and converts each sample into stacked sparse event-frame tensors.
- Output format: `(T, H, W, 2)`
- Meaning:
  - `T`: number of sparse event frames
  - `H`, `W`: spatial resolution (`120 x 160`)
  - `2`: event polarity channels
- Why: EvT-OG works on event-frame sequences, not directly on raw `.aedat4` recordings.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - builds all available splits through `--mode train|val|test|all`.
  - uses centralized EHW paths under `dev/datasets`.
  - downsamples the original `320 x 240` event space to `160 x 120` so the EvT patch/token budget stays much closer to the original DVS benchmark while preserving the original aspect ratio.

### `dev/EvT-OG/train_ehwgesture.py`
- What it does: main entrypoint that configures and launches EHWGesture EvT-OG fine-tuning.
- Why: the original DVS training entrypoint was not designed as a dedicated EHW benchmark runner.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - loads the DVS128 pretrained EvT checkpoint as initialization.
  - switches the task to `EHWGesture` with `5` classes.
  - configures the real EHW `train` and `val` splits.
  - enables early stopping on `val_loss_total`.
  - adds `--val-workers` to keep validation stable under this environment.
  - adds `--resume-from` so interrupted EHW runs can continue from a saved checkpoint.

### `dev/EvT-OG/trainer_ehwgesture.py`
- What it does: contains the EHW-specific Lightning model, optimizer setup, checkpoint loading, logging, and training loop.
- Why: the benchmark needs EHW-specific fine-tuning behavior without changing the older DVS training path.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/EvT-OG/train_ehwgesture.py`.
- My changes:
  - added manual top-1 and top-5 computation.
  - added selective pretrained-backbone loading from the DVS128 checkpoint.
  - skips incompatible tensors such as the classifier head and positional encoding.
  - added resume support from an interrupted EHW checkpoint.
  - added a safer CSV logger path to avoid the PyYAML hyperparameter-save crash that appeared during resumed training.

### `dev/EvT-OG/data_generation_ehwgesture.py`
- What it does: builds the EHW EvT-OG dataloaders and extracts active patch tokens from sparse event-frame tensors.
- Why: the EHW benchmark needs an isolated data pipeline so the DVS128 benchmark remains frozen and reproducible.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by:
  - `dev/EvT-OG/trainer_ehwgesture.py`
  - `dev/EvT-OG/evaluation_stats_ehwgesture.py`
- My changes:
  - added EHW-specific dataset paths and class mapping.
  - added separate train, validation, and test dataloaders.
  - added separate `workers`, `val_workers`, and `test_workers` control.
  - changed the dataset path handling to absolute paths so EHW runs do not depend on the current working directory.
  - changed the EHW sample loading path to avoid unstable `sparse.COO` slicing inside dataloader workers by converting samples to dense arrays earlier.

### `dev/EvT-OG/evaluation_stats_ehwgesture.py`
- What it does: loads the selected fine-tuned EHW EvT-OG checkpoint and evaluates it on the shared EHW `test` split.
- Why: benchmark reporting should come from post-hoc test evaluation, not only from training-time validation metrics.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - resolves the best EHW checkpoint from the fine-tuning run folder.
  - evaluates only on the real EHW `test` split.
  - computes top-1 and top-5 accuracy.
  - saves EHW-specific benchmark outputs such as confusion matrix, per-class recall, and training-evolution figure.

### `dev/EvT-OG/models/EvT.py`
- What it does: contains the EvT model backbone and token-processing logic used during EHW fine-tuning and evaluation.
- Why: `trainer_ehwgesture.py` and `evaluation_stats_ehwgesture.py` both depend on this model implementation.
- Status: **Original file**.
- Run type: executed indirectly.
- Called by:
  - `dev/EvT-OG/trainer_ehwgesture.py`
  - `dev/EvT-OG/evaluation_stats_ehwgesture.py`

---

## 4) TimeSformer

### EHW TimeSformer status TODO
- What it does: this benchmark path is planned, but it is not documented as a completed pipeline yet.
- Why: the current priority is to finish and validate the EHW EvT-OG benchmark first, using the shared train/val/test split.
- Status: **Not completed yet in this benchmark document**.

---

## 5) Benchmark Output TODO

### `dev/benchmark/EHWGesture/evt_vs_tsformer_ehw.ipynb`
- What it does: benchmark notebook intended to compare EvT-OG and TimeSformer on EHWGesture.
- Why: centralizes benchmark analysis in one place once both model paths are ready.
- Status: **Created by me (custom)**.
- Run type: run directly.
- Current role:
  - serves as the EHW benchmark notebook placeholder.
  - is expected to follow the same benchmark logic as the DVS128 comparison, but with the shared EHW split.

---

## 6) Why I Added Custom Files

I added custom files to solve real integration gaps for the EHW benchmark:
- the raw EHW dataset is distributed as `.aedat4` event recordings.
- EvT-OG expects split gesture samples and sparse event-frame tensors.
- the benchmark needs isolated EHW files so the older DVS128 benchmark remains frozen.

My custom files bridge those gaps:
- `ehw_env.sh` for portable EHW-specific paths.
- `ehwgesture_split_dataset.py` to convert raw EHW recordings into canonical split samples.
- `ehwgesture.py` to convert clean EHW samples into EvT-ready sparse event-frame tensors.
- `train_ehwgesture.py`, `trainer_ehwgesture.py`, `data_generation_ehwgesture.py`, and `evaluation_stats_ehwgesture.py` to run a dedicated EHW EvT benchmark path without disturbing the DVS128 setup.
