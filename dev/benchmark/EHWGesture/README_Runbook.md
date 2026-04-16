# Pipeline (EvT-OG vs TimeSformer on EHWGesture): Setup Runbook

This is the personal end-to-end setup being used in this project for EHWGesture:
1.1. Convert raw EHW event recordings into EvT-OG split samples.
1.2. Convert EHW split samples into sparse event-frame tensors for EvT-OG.
1.3. Fine-tune EvT-OG on the full EHWGesture train/val split with early stopping.
1.4. Evaluate the selected EHW EvT-OG checkpoint on the shared EHW test split.

2.1.TODO.
2.2 TODO
2.3 TODO
2....TODO

For a simple theory/explanation of each executed file (original vs custom vs modified), see `dev/benchmark/EHWGesture/README_THEORY.md`.

## Environment
Use the EHW-specific Python environment and activate it before running anything:

```bash
# Load project paths (recommended)
source "$HOME/TESE/dev/ehw_env.sh"
```

```bash
conda activate tese_ehw_py37
```

# Confirm GPU is visible:
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count()); print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```
## 1) EvT-OG Train + Evaluation (Pretrained Weights)
## 1.1) EvT-OG: Build EHW Clean Dataset

Assumed paths:
- Raw EHW dataset root: `$EHW_RAW_ROOT`
- Raw event recordings: `$EHW_EVENT_ROOT`
- Canonical split files:
  - `$EHW_RAW_ROOT/trials_to_train.txt`
  - `$EHW_RAW_ROOT/trials_to_val.txt`
  - `$EHW_RAW_ROOT/trials_to_test.txt`
- Output clean samples:
  - `$EHW_EVT_ROOT/clean_dataset/train`
  - `$EHW_EVT_ROOT/clean_dataset/val`
  - `$EHW_EVT_ROOT/clean_dataset/test`

Project root:

```bash
cd "$DEV_ROOT/EvT-OG/dataset_scripts"
```

Build all clean split samples from raw `.aedat4` event recordings:

```bash
python ehwgesture_split_dataset.py
```

This script:
- reads the canonical EHW split files
- scans `$EHW_EVENT_ROOT`
- maps EHWGesture to the `5` benchmark classes:
  - `FT`
  - `OC`
  - `PS`
  - `NOSE`
  - `TR`
- writes one `.pckl` per recording into `clean_dataset/{train,val,test}`

Output filenames look like:

```bash
X01_LEFT_FTF1_label01.pckl
```

## 1.2) EvT-OG: Build EHW Sparse Event-Frame Tensors

Project root:

```bash
cd "$DEV_ROOT/EvT-OG/dataset_scripts"
```

Build all splits at once:

```bash
python ehwgesture.py
```

This reads:

```bash
$EHW_EVT_ROOT/clean_dataset/{train,val,test}
```

and writes:

```bash
$EHW_EVT_ROOT/clean_dataset_frames_12000/{train,val,test}
```

Important processing choice:
- Raw EHW event space is `320 x 240`
- EvT-OG preprocessing downsamples to `160 x 120`
- This keeps the EHW EvT patch/token budget much closer to the original DVS benchmark while preserving the original aspect ratio

For long offline preprocessing with `nohup` and `ntfy`:

```bash
nohup bash -lc 'curl -fsS -H "Title: EHW EvT preprocessing started" -d "Started ehwgesture.py for train+val+test on $(hostname) at $(date)." https://ntfy.sh/pedro_tese_checkup_105306 >/dev/null 2>&1; cd /home/ppfsa/TESE/dev/EvT-OG/dataset_scripts && /home/ppfsa/miniconda3/envs/tese_ehw_py37/bin/python ehwgesture.py; status=$?; if [ "$status" -eq 0 ]; then curl -fsS -H "Title: EHW EvT preprocessing finished" -d "ehwgesture.py finished successfully on $(hostname) at $(date)." https://ntfy.sh/pedro_tese_checkup_105306 >/dev/null 2>&1; else curl -fsS -H "Title: EHW EvT preprocessing failed" -d "ehwgesture.py failed with status $status on $(hostname) at $(date)." https://ntfy.sh/pedro_tese_checkup_105306 >/dev/null 2>&1; fi; exit $status' > /home/ppfsa/TESE/dev/datasets/evt_og/EHWGesture/ehwgesture_preprocess.log 2>&1 &
```

## 1.3) EvT-OG: Fine-Tune on Full EHWGesture with Early Stopping

Project root:

```bash
cd "$DEV_ROOT/EvT-OG"
```

The EHW fine-tuning entrypoint is:

```bash
python train_ehwgesture.py
```

Default setup:
- pretrained initialization source:
  - `./pretrained_models/DVS128_11_24ms_dwn`
- EHW task:
  - `5` classes
- train split:
  - real EHW `train`
- validation split:
  - real EHW `val`
- early stopping:
  - monitor: `val_loss_total`
  - mode: `min`
  - patience: `10`
  - min_delta: `0.0001`
- max epochs:
  - `80`

Important note about weight loading:
- the DVS128 `11`-class checkpoint is used only as initialization
- incompatible tensors are skipped:
  - classifier head
  - positional encoding
- matching backbone tensors are reused for EHW fine-tuning

The run writes to:

```bash
$DEV_ROOT/EvT-OG/pretrained_models/ehwgesture_finetune_earlystop
```

Important saved artifacts:
- `all_params.json`
- `train_log/version_0/metrics.csv`
- `weights/*.ckpt`

If needed, you can override some defaults:

```bash
python train_ehwgesture.py \
  --pretrained-model-dir ./pretrained_models/DVS128_11_24ms_dwn \
  --batch-size 32 \
  --workers 6 \
  --max-epochs 80 \
  --gpus 0 \
  --output-name /ehwgesture_finetune_earlystop
```

## 1.4) EvT-OG: Post-Hoc Evaluation on the Shared EHW Test Split

Project root:

```bash
cd "$DEV_ROOT/EvT-OG"
```

Run the EHW-only evaluation entrypoint:

```bash
python evaluation_stats_ehwgesture.py
```

Lighter version without FLOPs:

```bash
python evaluation_stats_ehwgesture.py --skip-flops
```

This script:
- resolves the EHW fine-tuning run folder
- loads the best checkpoint selected from validation metrics
- evaluates on the real EHW `test` split only
- saves benchmark artifacts next to the trained model

Default model root:

```bash
./pretrained_models/ehwgesture_finetune_earlystop
```

You can also point to a specific run:

```bash
python evaluation_stats_ehwgesture.py \
  --path-model ./pretrained_models/ehwgesture_finetune_earlystop/<run_folder>
```

Saved artifacts:
- `stats_test_ehwgesture.json`
- `confusion_matrix_test_ehwgesture.pckl`
- `confusion_matrix_test_ehwgesture.png`
- `per_class_recall_test_ehwgesture.csv`
- `training_evolution_ehwgesture.png`

Reported metrics/artifacts:
- validation accuracy from training logs
- validation top-5 accuracy from training logs
- validation loss from training logs
- test accuracy
- test top-5 accuracy
- timing
- parameter statistics
- optional FLOPs
- confusion matrix
- per-class recall on the shared test split
- training/validation curve plot

## 2) TimeSformer on EHWGesture

TODO: This part is intentionally not documented yet in this runbook because the current focus is to complete and validate the EHW EvT-OG benchmark first.

The next EHW TimeSformer path will follow the same benchmark principles:
- same canonical split files
- same shared train/val/test population
- native Kinect RGB input
- early stopping
- post-hoc evaluation on the same EHW test split

## Notes
- Do not reuse the DVS benchmark runtime files directly for EHW.
- EHW uses isolated scripts so the DVS benchmark stays frozen and reproducible.
- The EHW EvT benchmark uses `160 x 120` event-frame tensors, not native `320 x 240`.
- The final benchmark result should come from post-hoc `test` evaluation, not only from training-time validation metrics.
