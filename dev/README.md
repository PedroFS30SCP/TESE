# Dev Setup Runbook
This is the personal end-to-end setup used in this project:
1. Evaluate EvT-OG pretrained models.
2. Convert full DVS128 AEDAT dataset to PNGs with `rpg_e2vid`.
3. Fine-tune TimeSformer on the converted PNGs.

## Environment
Use one Python environment and activate it before running anything:

```bash
source /home/azureuser/TESE/venv/bin/activate
```

If using conda on another machine:

```bash
conda activate tese_py37
```

## 1) EvT-OG Evaluation Only (Pretrained Weights)
Project root:

```bash
cd /home/azureuser/TESE/dev/EvT-OG/dataset_scripts
python dvs128_split_dataset.py
python dvs128.py
```

```bash
cd /home/azureuser/TESE/dev/EvT-OG
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
cd /home/azureuser/TESE/dev/rpg_e2vid
```

Assumed paths:
- Raw AEDAT files: `/home/azureuser/TESE/dev/rpg_e2vid/datasets/DVS128/*.aedat`
- Trial lists: `/home/azureuser/TESE/dev/rpg_e2vid/datasets/DVS128/trials_to_train.txt` and `trials_to_test.txt`
- E2VID checkpoint: `/home/azureuser/TESE/dev/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar`
- Output PNG folders: `/home/azureuser/TESE/dev/rpg_e2vid/datasets/DVS128/dvs2vid/<sample_name>/`

### 2.1 CPU mode (one sample, quick smoke test)

```bash
cd /home/azureuser/TESE/dev/rpg_e2vid
mkdir -p datasets/DVS128/dvs2vid

python scripts/aedat_to_e2vid.py \
  --input datasets/DVS128/user01_fluorescent_led.aedat \
  --output datasets/DVS128/user01_fluorescent_led.txt \
  --time_unit auto

CUDA_VISIBLE_DEVICES="" python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i datasets/DVS128/user01_fluorescent_led.txt \
  --output_folder datasets/DVS128/dvs2vid \
  --dataset_name user01_fluorescent_led \
  --fixed_duration \
  -T 33.33 \
  --auto_hdr \
  --compute_voxel_grid_on_cpu
```

### 2.2 GPU mode (full dataset)

Run this on the GPU machine:

```bash
cd /home/azureuser/TESE/dev/rpg_e2vid
mkdir -p datasets/DVS128/dvs2vid

for LIST in datasets/DVS128/trials_to_train.txt datasets/DVS128/trials_to_test.txt; do
  while read -r AEDAT_NAME; do
    [ -z "$AEDAT_NAME" ] && continue
    BASE="${AEDAT_NAME%.aedat}"
    IN_AEDAT="datasets/DVS128/${AEDAT_NAME}"
    OUT_TXT="datasets/DVS128/${BASE}.txt"
    OUT_DIR="datasets/DVS128/dvs2vid/${BASE}"

    [ -f "$IN_AEDAT" ] || { echo "Missing $IN_AEDAT, skipping"; continue; }
    [ -f "$OUT_DIR/timestamps.txt" ] && { echo "Already done: $BASE, skipping"; continue; }

    echo "Converting AEDAT -> TXT: ${AEDAT_NAME}"
    python scripts/aedat_to_e2vid.py --input "$IN_AEDAT" --output "$OUT_TXT" --time_unit auto || continue

    echo "Reconstructing TXT -> PNG: ${BASE}"
    python run_reconstruction.py \
      -c pretrained/E2VID_lightweight.pth.tar \
      -i "$OUT_TXT" \
      --output_folder datasets/DVS128/dvs2vid \
      --dataset_name "$BASE" \
      --fixed_duration \
      -T 33.33 \
      --auto_hdr
  done < "$LIST"
done
```

Each sample folder contains:
- `frame_XXXXXXXXXX.png`
- `timestamps.txt`

## 3) TimeSformer Fine-Tuning on Converted PNGs
Project root:

```bash
cd /home/azureuser/TESE/dev/TimeSformer
```

Current setup already includes:
- DVS dataset loader: `timesformer/datasets/dvs128.py`
- Manifest builder: `tools/prepare_dvs128_manifest.py`
- Config: `configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml`
- Pretrained checkpoint: `checkpoints/TimeSformer_divST_8_224_SSv2.pyth`

### 3.1 Build per-sample manifests
Example for one sample:

```bash
python tools/prepare_dvs128_manifest.py \
  --sample-name user01_fluorescent_led \
  --dvs-root datasets/DVS128 \
  --clip-len 8 \
  --stride 4 \
  --val-ratio 0.2 \
  --seed 0
```

For full dataset, repeat for each sample in `trials_to_train.txt` and `trials_to_test.txt`, then merge CSVs into one global manifest folder (train/val/test).

### 3.2 Fine-tune
CPU sanity check:

```bash
python3 tools/run_net.py \
  --cfg configs/DVS128/TimeSformer_divST_8x1_128_finetune.yaml \
  TRAIN.CHECKPOINT_FILE_PATH /home/azureuser/TESE/dev/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
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
  TRAIN.CHECKPOINT_FILE_PATH /home/azureuser/TESE/dev/TimeSformer/checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
  NUM_GPUS 1
```

## Notes
- Keep `git` for code and `rsync` for large data/checkpoints when moving machines.
- The 2-epoch CPU run is only a pipeline sanity check, not final performance.

## `dev/EvT-OG/notebooks`
- `dvs128_testing.ipynb`: notebook that implements **Event preprocessing**, **Event-frame building**, and **activated patch extraction**.
- `lynx_testing.ipynb`: notebook that implements **frame preprocessing**, **temporal window (Δt) building**, **adapted event-frame building**, and **activated patch extraction**.

IMP: 
- source /home/azureuser/TESE/venv/bin/activate
- conda activate tese_py37
