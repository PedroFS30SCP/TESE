# Dev Setup Runbook
This is the personal end-to-end setup used in this project:
1. Evaluate EvT-OG pretrained models.
2. Convert full DVS128 AEDAT dataset to PNGs with `rpg_e2vid`.
3. Fine-tune TimeSformer on the converted PNGs.

For a simple theory/explanation of each executed file (original vs custom vs modified), see `dev/README_THEORY.md`.

## Environment
Use one Python environment and activate it before running anything:

```bash
# Load project paths (recommended)
source "$HOME/TESE/dev/env.sh"
```

```bash
conda activate tese_py37
pip install -r /home/ppfsa/TESE/dev/requirements.txt
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
- Raw AEDAT files: `$DATA_ROOT/raw/DVS128/*.aedat`
- Raw labels CSV: `$DATA_ROOT/raw/DVS128/*_labels.csv`
- Trial lists: `$DATA_ROOT/raw/DVS128/trials_to_train.txt` and `trials_to_test.txt`
- E2VID checkpoint: `$DEV_ROOT/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar`
- Output sample folders: `$DATA_ROOT/rpg_e2vid/dvs2vid/<sample_name>/`

### 2.1 CPU mode (one sample, quick smoke test)

```bash
cd "$DEV_ROOT/rpg_e2vid"
mkdir -p "$DATA_ROOT/rpg_e2vid/dvs2vid"

python scripts/aedat_to_e2vid.py \
  --input "$DATA_ROOT/raw/DVS128/user01_fluorescent_led.aedat" \
  --output "$DATA_ROOT/raw/DVS128/user01_fluorescent_led.txt" \
  --time_unit auto

CUDA_VISIBLE_DEVICES="" python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i "$DATA_ROOT/raw/DVS128/user01_fluorescent_led.txt" \
  --output_folder "$DATA_ROOT/rpg_e2vid/dvs2vid" \
  --dataset_name user01_fluorescent_led \
  --fixed_duration -T 33.33 \
  --auto_hdr \
  --compute_voxel_grid_on_cpu

cp "$DATA_ROOT/raw/DVS128/user01_fluorescent_led_labels.csv" \
  "$DATA_ROOT/rpg_e2vid/dvs2vid/user01_fluorescent_led/"
```

### 2.2 GPU mode (full dataset)

Run this on the GPU machine:

```bash
cd "$DEV_ROOT/rpg_e2vid"
mkdir -p "$DATA_ROOT/rpg_e2vid/dvs2vid"

for LIST in "$DATA_ROOT/raw/DVS128/trials_to_train.txt" "$DATA_ROOT/raw/DVS128/trials_to_test.txt"; do
  while read -r AEDAT_NAME; do
    [ -z "$AEDAT_NAME" ] && continue
    BASE="${AEDAT_NAME%.aedat}"
    IN_AEDAT="$DATA_ROOT/raw/DVS128/${AEDAT_NAME}"
    OUT_TXT="$DATA_ROOT/raw/DVS128/${BASE}.txt"
    OUT_DIR="$DATA_ROOT/rpg_e2vid/dvs2vid/${BASE}"
    LABEL_CSV="$DATA_ROOT/raw/DVS128/${BASE}_labels.csv"

    [ -f "$IN_AEDAT" ] || { echo "Missing $IN_AEDAT, skipping"; continue; }
    [ -f "$LABEL_CSV" ] || { echo "Missing $LABEL_CSV, skipping"; continue; }
    [ -f "$OUT_DIR/timestamps.txt" ] && { echo "Already done: $BASE, skipping"; continue; }

    python scripts/aedat_to_e2vid.py --input "$IN_AEDAT" --output "$OUT_TXT" --time_unit auto || continue

    python run_reconstruction.py \
      -c pretrained/E2VID_lightweight.pth.tar \
      -i "$OUT_TXT" \
      --output_folder "$DATA_ROOT/rpg_e2vid/dvs2vid" \
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
  --dvs-root "$DATA_ROOT/rpg_e2vid" \
  --sample-name user01_fluorescent_led \
  --clip-len 8 \
  --stride 4 \
  --val-ratio 0.2 \
  --seed 0 \
  --path-mode rel \
  --output-dir "$DATA_ROOT/timesformer/DVS128/manifests/user01_fluorescent_led_8f"
```

Full dataset (automatic build + merge to one global manifest):

```bash
cd "$DEV_ROOT/TimeSformer"

RAW_ROOT="$DATA_ROOT/raw/DVS128"
DVS_ROOT="$DATA_ROOT/rpg_e2vid"
PER_SAMPLE_ROOT="$DATA_ROOT/timesformer/DVS128/manifests/per_sample"
GLOBAL_ROOT="$DATA_ROOT/timesformer/DVS128/manifests/all_samples_8f"

mkdir -p "$PER_SAMPLE_ROOT" "$GLOBAL_ROOT"

for SPLIT in train val test; do
  echo "clip_id,label,frames" > "$GLOBAL_ROOT/${SPLIT}.csv"
done

for LIST in "$RAW_ROOT/trials_to_train.txt" "$RAW_ROOT/trials_to_test.txt"; do
  while read -r AEDAT_NAME; do
    [ -z "$AEDAT_NAME" ] && continue
    BASE="${AEDAT_NAME%.aedat}"
    SAMPLE_DIR="$DVS_ROOT/dvs2vid/$BASE"
    SAMPLE_OUT="$PER_SAMPLE_ROOT/${BASE}_8f"

    [ -d "$SAMPLE_DIR" ] || { echo "Missing frames for $BASE, skipping"; continue; }
    [ -f "$SAMPLE_DIR/${BASE}_labels.csv" ] || { echo "Missing labels for $BASE, skipping"; continue; }

    python3 tools/prepare_dvs128_manifest.py \
      --dvs-root "$DVS_ROOT" \
      --sample-name "$BASE" \
      --clip-len 8 \
      --stride 4 \
      --val-ratio 0.2 \
      --seed 0 \
      --output-dir "$SAMPLE_OUT" || continue

    for SPLIT in train val test; do
      tail -n +2 "$SAMPLE_OUT/${SPLIT}.csv" >> "$GLOBAL_ROOT/${SPLIT}.csv"
    done
  done < "$LIST"
done
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

## Notes
- Keep `git` for code and `rsync` for large data/checkpoints when moving machines.
- The 1-epoch CPU run is only a pipeline sanity check, not final performance.

## `dev/EvT-OG/notebooks`
- `dvs128_testing.ipynb`: notebook that implements **Event preprocessing**, **Event-frame building**, and **activated patch extraction**.
- `lynx_testing.ipynb`: notebook that implements **frame preprocessing**, **temporal window (Δt) building**, **adapted event-frame building**, and **activated patch extraction**.

IMP: 
- source ~/TESE/venv/bin/activate
- conda activate tese_py37

## Final Note (Full Dataset)
- For **full-dataset TimeSformer training**, use the merged manifest folder:
  - `"$DATA_ROOT/timesformer/DVS128/manifests/all_samples_8f"`
- If needed, override at runtime:
```bash
python3 tools/run_net.py \
  --cfg configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml \
  DATA.PATH_TO_DATA_DIR "$DATA_ROOT/timesformer/DVS128/manifests/all_samples_8f" \
  TRAIN.CHECKPOINT_FILE_PATH "$DEV_ROOT/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth" \
  NUM_GPUS 1
```
