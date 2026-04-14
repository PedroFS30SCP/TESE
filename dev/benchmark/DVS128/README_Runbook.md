# Pipeline (EvT-OG vs TimeSformer on DVS128): Setup Runbook

This is the personal end-to-end setup used in this project:
1. Evaluate EvT-OG pretrained models.
2. Convert full DVS128 AEDAT dataset to PNGs with `rpg_e2vid`.
3. Fine-tune TimeSformer on the converted PNGs.

For a simple theory/explanation of each executed file (original vs custom vs modified), see `dev/README_THEORY.md`.

## Environment
Use one Python environment and activate it before running anything:

```bash
# Load project paths (recommended)
source "$HOME/TESE/dev/dvs_env.sh"
```

```bash
conda activate tese_py37
pip install -r /home/ppfsa/TESE/dev/dvs_requirements.txt
```

# Confirm GPU is visible: 
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count()); print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 1) EvT-OG Evaluation Only (Pretrained Weights)

Firts choosing CPU only or GPU, for GPU:

Set GPU device in file:
Edit dvs128_split_dataset.py (line 12)
Change: device = 'cuda:0'

Edit evaluation_stats.py (line 24)
Change: train_files, test_files = 'trials_to_train.txt', 'trials_to_test.txt'

Project root:

```bash
cd "$DEV_ROOT/EvT-OG/dataset_scripts"
python dvs128_split_dataset.py
python dvs128.py
```

```bash
cd "$DEV_ROOT/EvT-OG"
```

`evaluation_stats.py` already points to a pretrained model and runs on CPU by default.
Edit these lines if needed:
- `path_model = './pretrained_models/DVS128_11_24ms_dwn/'`
- `device = 'cpu'` (or `'cuda:0'` on GPU machine)

Run:

```bash
python evaluation_stats.py
```

This prints model stats, FLOPs, timing, and validation accuracy using the pretrained checkpoint.

## 2) rpg_e2vid: Convert DVS128 (AEDAT -> TXT -> PNG)
Project root:

```bash
cd "$DEV_ROOT/rpg_e2vid"
```

Assumed paths:
- Raw AEDAT files: `$RAW_ROOT/*.aedat`
- Raw labels CSV: `$RAW_ROOT/*_labels.csv`
- Trial lists: `$RAW_ROOT/trials_to_train.txt` and `trials_to_test.txt`
- E2VID checkpoint: `$DEV_ROOT/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar`
- Output sample folders: `$DVS_ROOT/dvs2vid/<sample_name>/`

### 2.1 CPU mode (one sample, quick smoke test)

```bash
cd "$DEV_ROOT/rpg_e2vid"
mkdir -p "$DVS_ROOT/dvs2vid"

python scripts/aedat_to_e2vid.py \
  --input "$RAW_ROOT/user01_fluorescent_led.aedat" \
  --output "$RAW_ROOT/user01_fluorescent_led.txt" \
  --time_unit auto

CUDA_VISIBLE_DEVICES="" python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i "$RAW_ROOT/user01_fluorescent_led.txt" \
  --output_folder "$DVS_ROOT/dvs2vid" \
  --dataset_name user01_fluorescent_led \
  --fixed_duration -T 33.33 \
  --auto_hdr \
  --compute_voxel_grid_on_cpu

cp "$RAW_ROOT/user01_fluorescent_led_labels.csv" \
  "$DVS_ROOT/dvs2vid/user01_fluorescent_led/"
```

### 2.2 GPU mode (full dataset)

Run this on the GPU machine to covert aedat to txt and then to PNG:

```bash
cd "$DEV_ROOT/rpg_e2vid"
mkdir -p "$DVS_ROOT/dvs2vid"

for LIST in "$RAW_ROOT/trials_to_train.txt" "$RAW_ROOT/trials_to_test.txt"; do
  while read -r AEDAT_NAME; do
    [ -z "$AEDAT_NAME" ] && continue
    BASE="${AEDAT_NAME%.aedat}"
    IN_AEDAT="$RAW_ROOT/${AEDAT_NAME}"
    OUT_TXT="$RAW_ROOT/${BASE}.txt"
    OUT_DIR="$DVS_ROOT/dvs2vid/${BASE}"
    LABEL_CSV="$RAW_ROOT/${BASE}_labels.csv"

    [ -f "$IN_AEDAT" ] || { echo "Missing $IN_AEDAT, skipping"; continue; }
    [ -f "$LABEL_CSV" ] || { echo "Missing $LABEL_CSV, skipping"; continue; }
    [ -f "$OUT_DIR/timestamps.txt" ] && { echo "Already done: $BASE, skipping"; continue; }

    python scripts/aedat_to_e2vid.py --input "$IN_AEDAT" --output "$OUT_TXT" --time_unit auto || continue

    python run_reconstruction.py \
      -c pretrained/E2VID_lightweight.pth.tar \
      -i "$OUT_TXT" \
      --output_folder "$DVS_ROOT/dvs2vid" \
      --dataset_name "$BASE" \
      --fixed_duration -T 33.33 --auto_hdr

    cp "$LABEL_CSV" "$OUT_DIR/"
  done < "$LIST"
done
```

Each sample folder contains:
- `frame_XXXXXXXXXX.png`
- `timestamps.txt`
- `<sample_name>_labels.csv`

## 3) TimeSformer Fine-Tuning on Converted PNGs

### 3.1 Build per-sample manifests
Example for one sample:

```bash
cd "$DEV_ROOT/TimeSformer"

python3 tools/prepare_dvs128_manifest.py \
  --dvs-root "$DVS_ROOT" \
  --sample-name user01_fluorescent_led \
  --clip-len 8 \
  --stride 4 \
  --val-ratio 0.2 \
  --seed 0 \
  --sample-split trainval \
  --path-mode rel \
  --output-dir "$DATA_ROOT/timesformer/DVS128/manifests/user01_fluorescent_led_8f"
```

Full dataset (official split: train list -> train/val, test list -> test):

```bash
cd "$DEV_ROOT/TimeSformer"

python3 tools/prepare_dvs128_manifest.py \
  --dvs-root "$DVS_ROOT" \
  --train-list "$RAW_ROOT/trials_to_train.txt" \
  --val-list "$RAW_ROOT/trials_to_val.txt" \
  --test-list "$RAW_ROOT/trials_to_test.txt" \
  --clip-len 8 \
  --stride 4 \
  --path-mode rel \
  --output-dir "$GLOBAL_ROOT"
```

### 3.2 Fine-tune
CPU sanity check:

```bash
python3 tools/run_net.py \
  --cfg configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml \
  TRAIN.CHECKPOINT_FILE_PATH "$DEV_ROOT/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth" \
  NUM_GPUS 0 \
  TRAIN.BATCH_SIZE 2 \
  TEST.BATCH_SIZE 2 \
  DATA_LOADER.NUM_WORKERS 0 \
  TRAIN.EVAL_PERIOD 1 \
  SOLVER.MAX_EPOCH 1
```

GPU training (target machine):

```bash
python3 tools/run_net.py \
  --cfg configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml \
  TRAIN.CHECKPOINT_FILE_PATH "$DEV_ROOT/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth" \
  NUM_GPUS 1
```

This config now writes the new early-stopping run to:

```bash
$DEV_ROOT/TimeSformer/outputs/dvs128_all_samples_finetune_earlystop
```

Important saved artifacts:
- `stdout.log` with all train/val epoch metrics used by the benchmark plots
- `checkpoints/checkpoint_best.pyth`
- `checkpoints/checkpoint_last.pyth`
- `training_state.json`

## Notes
- Keep `git` for code and `rsync` for large data/checkpoints when moving machines.
- The 1-epoch CPU run is only a pipeline sanity check, not final performance.

## `dev/EvT-OG/notebooks`
- `dvs128_testing.ipynb`: notebook that implements **Event preprocessing**, **Event-frame building**, and **activated patch extraction**.
- `lynx_testing.ipynb`: notebook that implements **frame preprocessing**, **temporal window (Δt) building**, **adapted event-frame building**, and **activated patch extraction**.

For starting the model ofline:

```bash
cd "$DEV_ROOT/TimeSformer" 
nohup python3 tools/run_net.py \
  --cfg configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml \
  TRAIN.CHECKPOINT_FILE_PATH "$DEV_ROOT/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth" \
  NUM_GPUS 1 &
```

Results:

Final result for EvT-OG (running evaluation on the test split after training, using the saved checkpoint):

Post-hoc test accuracy after training: 96.33 %
Post-hoc test top-5 accuracy after training: 99.67 %

Final result for Timesformer (10epochs fine-tune on the test split same split as EVT):

top1_acc = 89.23
top5_acc = 99.81

Final result for TimeSformer (trained 29 epochs total, stopped by early stopping, evaluated on the same test split as EvT):

top1_acc = 92.21
top5_acc = 99.94