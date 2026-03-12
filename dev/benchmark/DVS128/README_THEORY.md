# TESE Pipeline (EvT-OG vs TimeSformer on DVS128): Simple File-by-File Explanation

This document explains the files I run in this project:
- what each file does,
- why it does it,
- and whether it is original from the authors, created by me, or modified by me.

## 1) Big Picture

My pipeline has 3 stages:
1. `EvT-OG` evaluates event-based model performance.
2. `rpg_e2vid` converts DVS128 event streams into PNG frames.
3. `TimeSformer` trains/tests a video transformer on those PNG frames.

---

## 2) Environment

### `dev/env.sh`
- What it does: sets `TESE_ROOT`, `DEV_ROOT`, and `DATA_ROOT`.
- Why: all commands can use portable paths (works across machines).
- Status: **Created by me (custom)**.
- Run type: run directly.

---

## 3) EvT-OG

### `dev/EvT-OG/dataset_scripts/dvs128_split_dataset.py`
- What it does: Splits each raw `.aedat` recording into smaller `.pckl` files. Each `.pckl` file stores one gesture segment.
- Why: EvT-OG expects pre-split event samples, not full raw recordings.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - changed paths to centralized dataset root under `dev/datasets`
  - added validation split support, so the script now writes `train`, `val`, and `test`
  - kept the simple full-dataset vs 1-sample trial-list toggle
- Example: `user01_fluorescent_led_num04_label03.pckl`
- Filename meaning:
  - `user01_fluorescent_led`: original raw recording
  - `num04`: 5th extracted segment from that recording
  - `label03`: class label 3

### `dev/EvT-OG/dataset_scripts/dvs128.py`
- What it does: Loads the `.pckl` files and splits each gesture segment into 12 ms event frames.
- Output format: `(T, H, W, 2)`
- Meaning:
  - `T`: number of event frames
  - `H`, `W`: spatial resolution (`128 x 128`)
  - `2`: event polarity channels
- Why: EvT-OG works on frame-like chunks created from raw events.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - removed machine-specific assumptions and moved to centralized `dev/datasets` paths
  - added CLI split selection with `--mode train|val|test|all`
  - default behavior now builds all available splits in one run

### `dev/EvT-OG/evaluation_stats.py`
- What it does: Script that loads the authors' pretrained weights and runs the evaluation on the split that I choose. In this case, the evaluation was performed on the test split.
- Why: quick benchmark and sanity check after training or with pretrained weights.
- Status: **Original file, modified**.
- Run type: run directly.
- My changes:
  - added explicit val-vs-test evaluation selection.
  - added post-hoc top-5 accuracy computation.
  - updated the printed summary so benchmark outputs are clearer.

### `dev/EvT-OG/data_generation.py`
- What it does: builds EvT-OG dataloaders and extracts active patch tokens from sparse event frames.
- Why: `evaluation_stats.py` loads this module indirectly via `Event_DataModule`.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/EvT-OG/evaluation_stats.py`.
- My changes:
  - changed dataset folder path to match my centralized data location under `dev/datasets`.
  - added separate validation and test dataloaders.
  - added optional sample-name control inside the dataset loader.

### `dev/EvT-OG/trainer.py`
- What it does: contains the step-by-step process of running the model.
- Why modified: benchmark comparison needs both top-1 and top-5 accuracy.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/EvT-OG/evaluation_stats.py`.
- My changes:
  - replaced the old Lightning accuracy helper with manual top-1/top-5 computation.
  - added `acc_top5` logging during training.

### `dev/EvT-OG/evaluation_utils.py`
- What it does: responsible for model evaluation metrics.
- Why modified: benchmark analysis needs top-5 support and cleaner log parsing.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/EvT-OG/evaluation_stats.py`.
- My changes:
  - added top-5 extraction in evaluation summaries.
  - updated training-evolution plotting to show top-5 when available.
  - made `val_acc` column selection more explicit.

---

## 4) rpg_e2vid

### `dev/rpg_e2vid/scripts/aedat_to_e2vid.py`
- What it does: converts DVS128 `.aedat` events to E2VID `.txt` format (`t x y p`).
- Why: `run_reconstruction.py` uses this text event format.
- Status: **Created by me (custom)**.
- Run type: run directly.
- My changes:
  - robust parsing for AEDAT variants.
  - portable default paths (no hardcoded Azure path).

### `dev/rpg_e2vid/run_reconstruction.py`
- What it does: reconstructs event streams into image frames (`frame_*.png`) + `timestamps.txt`.
- Why: TimeSformer needs frame sequences, not raw event streams.
- Status: **Original file (author code)**.
- Run type: run directly.

---

## 5) TimeSformer

### `dev/TimeSformer/tools/prepare_dvs128_manifest.py`
- What it does: builds `train.csv`, `val.csv`, `test.csv` clip manifests from reconstructed PNGs + labels.
- Why: TimeSformer dataset loader needs clip manifests describing frame sequences and class labels.
- Status: **Created by me (custom)**.
- Run type: run directly.
- My changes:
  - added support for explicit `--val-list` in global manifest mode.
  - changed global manifest generation to use explicit train/val/test trial lists instead of deriving validation from the training list.

- The manifest file basically creates clips where each clip:
  has its own clip ID
  contains the gesture label
  contains 8 PNG frames per clip

### `dev/TimeSformer/configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml`
- What it does: defines TimeSformer experiment settings (dataset path, batch size, epochs, etc.).
- Why: separates experiment configuration from code.
- Status: **Created by me (custom config)**.
- Run type: used indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

### `dev/TimeSformer/tools/run_net.py`
- What it does: main TimeSformer train/val/test entrypoint.
- Why: executes full experiment loop from config.
- Status: **Original file (author code)**.
- Run type: run directly.

### `dev/TimeSformer/timesformer/datasets/dvs128.py`
- What it does: loads the PNG frames and creates the tensor that the model processes, one tensor is created per clip with the following format:
- Why: maps manifest rows into model-ready clips.
- Status: **Created by me (custom)**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

- (C, T, H, W) Where:
  C – color channels (usually 3 for RGB)
  T – number of frames in the clip
  H – frame height
  W – frame width

### `dev/TimeSformer/timesformer/datasets/__init__.py`
- What it does: registers available datasets.
- Why: lets config name `dvs128` resolve to my loader.
- Status: **Original file, modified** (added DVS128 registration).
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

### `dev/TimeSformer/tools/train_net.py`
- What it does: contains the TimeSformer training and validation loop.
- Why modified: needed validation loss in the logged benchmark outputs.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.
- My changes:
  - added validation-loss computation during `eval_epoch`.
  - passed validation loss into the validation meter so it gets logged.

### `dev/TimeSformer/timesformer/utils/meters.py`
- What it does: tracks and logs TimeSformer training/validation metrics.
- Why modified: benchmark plots need validation loss, not only top-1/top-5 error.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/train_net.py`.
- My changes:
  - extended `ValMeter` to accumulate validation loss.
  - added validation loss to per-iteration and per-epoch logs.

### `dev/TimeSformer/timesformer/datasets/video_container.py`
- What it does: video decoding helper for video-file datasets.
- Why modified: avoid import crash when `av` is not installed unless actually needed.
- Status: **Original file, modified** (lazy `import av` behavior).
- Run type: not really used in the final DVS128 data path.

### `dev/TimeSformer/timesformer/models/resnet_helper.py`
- Why modified: compatibility with newer torch where private imports changed.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

### `dev/TimeSformer/timesformer/models/vit_utils.py`
- Why modified: replaced deprecated `torch._six` usage for compatibility.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

### `dev/TimeSformer/timesformer/datasets/multigrid_helper.py`
- Why modified: replaced deprecated `torch._six` usage for compatibility.
- Status: **Original file, modified**.
- Run type: executed indirectly.
- Called by: `dev/TimeSformer/tools/run_net.py`.

---

## 6) Benchmark Output

### `dev/benchmark/DVS128/evt_vs_tsformer.ipynb`
- What it does: benchmark notebook comparing EvT-OG and TimeSformer on DVS128.
- Why: centralizes accuracy, confusion matrices, timing, memory, and curve analysis in one place.
- Status: **Created by me (custom)**.
- Run type: run directly.

---

## 7) Why I Added Custom Files

I added custom files to solve real integration gaps between three projects:
- EvT-OG expects event chunks.
- rpg_e2vid reconstructs PNGs.
- TimeSformer expects video/clip manifests.

My custom files bridge those gaps:
- `env.sh` for portable paths.
- `aedat_to_e2vid.py` to standardize raw AEDAT conversion.
- `prepare_dvs128_manifest.py` + `timesformer/datasets/dvs128.py` for TimeSformer ingestion.
- `DVS128 YAML` config to run my exact setup.
