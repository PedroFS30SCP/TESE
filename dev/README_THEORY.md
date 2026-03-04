# TESE Pipeline: Simple File-by-File Explanation

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

## 2) Files to Run Directly

### `dev/env.sh`
- What it does: sets `TESE_ROOT`, `DEV_ROOT`, and `DATA_ROOT`.
- Why: all commands can use portable paths (works across machines).
- Status: **Created by me (custom)**.

### `dev/EvT-OG/dataset_scripts/dvs128_split_dataset.py`
- What it does: reads DVS128 `.aedat` files + label CSVs and splits events into labeled gesture segments saved as `.pckl`.
- Why: EvT-OG expects pre-split event samples, not raw full recordings.
- Status: **Original file, modified**.
- My changes:
  - changed paths to centralized dataset root under `dev/datasets`.
  - added simple train/test list toggle for full dataset vs 1-sample mode.

### `dev/EvT-OG/dataset_scripts/dvs128.py`
- What it does: converts event segments into sparse frame chunks (`clean_dataset_frames_*`) for EvT-OG input.
- Why: EvT-OG evaluates on frame-like chunks created from raw events.
- Status: **Original file, modified**.
- My changes:
  - removed machine-specific assumptions and moved to centralized `dev/datasets` paths.

### `dev/EvT-OG/evaluation_stats.py`
- What it does: loads EvT-OG pretrained checkpoint and reports performance/compute stats.
- Why: quick benchmark and sanity check without retraining.
- Status: **Original file (used as run script)**.

### `dev/EvT-OG/data_generation.py`
- What it does: builds EvT-OG dataloaders and extracts active patch tokens from sparse event frames.
- Why: `evaluation_stats.py` loads this module indirectly via `Event_DataModule`.
- Status: **Original file, modified**.
- My changes:
  - changed dataset folder path to match my centralized data location under `dev/datasets`.

### `dev/rpg_e2vid/scripts/aedat_to_e2vid.py`
- What it does: converts DVS128 `.aedat` events to E2VID `.txt` format (`t x y p`).
- Why: `run_reconstruction.py` uses this text event format.
- Status: **Created by me (custom)**.
- My changes:
  - robust parsing for AEDAT variants.
  - portable default paths (no hardcoded Azure path).

### `dev/rpg_e2vid/run_reconstruction.py`
- What it does: reconstructs event streams into image frames (`frame_*.png`) + `timestamps.txt`.
- Why: TimeSformer needs frame sequences, not raw event streams.
- Status: **Original file (author code)**.

### `dev/TimeSformer/tools/prepare_dvs128_manifest.py`
- What it does: builds `train.csv`, `val.csv`, `test.csv` clip manifests from reconstructed PNGs + labels.
- Why: TimeSformer dataset loader needs clip manifests describing frame sequences and class labels.
- Status: **Created by me (custom)**.

### `dev/TimeSformer/tools/run_net.py`
- What it does: main TimeSformer train/val/test entrypoint.
- Why: executes full experiment loop from config.
- Status: **Original file (author code)**.

### `dev/TimeSformer/configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml`
- What it does: defines TimeSformer experiment settings (dataset path, batch size, epochs, etc.).
- Why: separates experiment configuration from code.
- Status: **Created by me (custom config)**.

---

## 3) TimeSformer Internal Files Used Indirectly

These are not usually run directly, but are used by `run_net.py`.

### `dev/TimeSformer/timesformer/datasets/dvs128.py`
- What it does: dataset loader for my DVS128 manifest format.
- Why: maps manifest rows into model-ready clips.
- Status: **Created by me (custom)**.

### `dev/TimeSformer/timesformer/datasets/__init__.py`
- What it does: registers available datasets.
- Why: lets config name `dvs128` resolve to my loader.
- Status: **Original file, modified** (added DVS128 registration).

### `dev/TimeSformer/timesformer/datasets/video_container.py`
- What it does: video decoding helper for video-file datasets.
- Why modified: avoid import crash when `av` is not installed unless actually needed.
- Status: **Original file, modified** (lazy `import av` behavior).

### `dev/TimeSformer/timesformer/models/resnet_helper.py`
- Why modified: compatibility with newer torch where private imports changed.
- Status: **Original file, modified**.

### `dev/TimeSformer/timesformer/models/vit_utils.py`
- Why modified: replaced deprecated `torch._six` usage for compatibility.
- Status: **Original file, modified**.

### `dev/TimeSformer/timesformer/datasets/multigrid_helper.py`
- Why modified: replaced deprecated `torch._six` usage for compatibility.
- Status: **Original file, modified**.

---

## 4) Why Manifest Files Matter

Manifest files are:
- `train.csv`
- `val.csv`
- `test.csv`

Each row represents one training clip:
- `clip_id`: unique clip name.
- `label`: class id (0 to 10 for 11 gestures).
- `frames`: semicolon-separated PNG frame paths.

Important:
- TimeSformer reads these CSVs first.
- Then it loads the actual PNG files listed in `frames`.

---

## 5) Why I Added Custom Files

I added custom files to solve real integration gaps between three projects:
- EvT-OG expects event chunks.
- rpg_e2vid reconstructs PNGs.
- TimeSformer expects video/clip manifests.

My custom files bridge those gaps:
- `env.sh` for portable paths.
- `aedat_to_e2vid.py` to standardize raw AEDAT conversion.
- `prepare_dvs128_manifest.py` + `timesformer/datasets/dvs128.py` for TimeSformer ingestion.
- DVS128 YAML config to run my exact setup.

---
