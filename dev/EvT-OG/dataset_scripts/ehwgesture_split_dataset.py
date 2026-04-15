import argparse
import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


CLASS_MAP = {
    "FT": 1,
    "OC": 2,
    "PS": 3,
    "NOSE": 4,
    "TR": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build EvT-OG clean_dataset splits for EHWGesture from .aedat4 recordings."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Root folder for raw EHWGesture data. Default: <repo>/dev/datasets/raw/EHWGesture",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output clean_dataset folder. Default: <repo>/dev/datasets/evt_og/EHWGesture/clean_dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Which splits to build. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pckl files.",
    )
    return parser.parse_args()


def _repo_dev_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_code(stem: str) -> str:
    code = re.sub(r"^dvSave[-_]*", "", stem)
    code = code.replace(".", "")
    if code.startswith("dvSavE_"):
        code = code.split("_", 1)[1]
    if code.startswith("dvSave_"):
        code = code.split("_", 1)[1]
    return code


def _canonical_sample_id(path: Path):
    subject = path.parts[-4]
    hand = path.parts[-3].split("_")[-1]
    code = _canonical_code(path.stem)
    return f"{subject}_{hand}_{code}"


def _label_from_sample_id(sample_id: str) -> int:
    code = sample_id.split("_", 2)[-1]
    gesture = re.sub(r"[0-9]+$", "", code)
    if gesture.startswith("FT"):
        return CLASS_MAP["FT"]
    if gesture.startswith("OC"):
        return CLASS_MAP["OC"]
    if gesture.startswith("PS"):
        return CLASS_MAP["PS"]
    if gesture.startswith("NOSE"):
        return CLASS_MAP["NOSE"]
    if gesture.startswith("TR"):
        return CLASS_MAP["TR"]
    raise ValueError(f"Unsupported EHWGesture code: {sample_id}")


def _eventstore_to_numpy(events):
    if events is None:
        return None
    if isinstance(events, np.ndarray):
        return events
    for attr in ("numpy", "to_numpy"):
        if hasattr(events, attr):
            return getattr(events, attr)()
    if all(hasattr(events, k) for k in ("x", "y", "t", "p")):
        x = np.asarray(events.x)
        y = np.asarray(events.y)
        t = np.asarray(events.t)
        p = np.asarray(events.p)
        if not (len(x) == len(y) == len(t) == len(p)):
            raise RuntimeError("Event fields have inconsistent lengths")
        return np.core.records.fromarrays([t, x, y, p], names="t,x,y,p")
    raise RuntimeError("Unsupported event container from dv-processing")


def _read_all_events_aedat4(path: Path):
    try:
        import dv_processing as dv
    except Exception as exc:
        raise RuntimeError(
            "dv-processing is required to read EHWGesture .aedat4 files. "
            "Install it in the EHW env before running this script."
        ) from exc

    tmp_dir = None
    read_path = str(path)
    if not str(path).endswith(".aedat4"):
        tmp_dir = tempfile.mkdtemp(prefix="ehw_aedat4_")
        tmp_path = Path(tmp_dir) / f"{path.name}.aedat4"
        shutil.copy2(path, tmp_path)
        read_path = str(tmp_path)

    events_list = []

    if hasattr(dv.io, "AedatFile"):
        reader = dv.io.AedatFile(read_path)

        for method_name in ("getEvents", "readEvents"):
            if hasattr(reader, method_name):
                ev = getattr(reader, method_name)()
                if ev is not None:
                    events_list.append(ev)
                break

        if not events_list and hasattr(reader, "__getitem__"):
            try:
                stream = reader["events"]
                for stream_method in ("read", "readEvents", "getNextEventBatch"):
                    if hasattr(stream, stream_method):
                        while True:
                            batch = getattr(stream, stream_method)()
                            if batch is None or (
                                hasattr(batch, "isEmpty") and batch.isEmpty()
                            ):
                                break
                            events_list.append(batch)
                        break
            except Exception:
                pass

        if not events_list and hasattr(reader, "__iter__"):
            for batch in reader:
                if batch is not None:
                    events_list.append(batch)

    if not events_list and hasattr(dv.io, "MonoCameraRecording"):
        reader = dv.io.MonoCameraRecording(read_path)
        if hasattr(reader, "isEventStreamAvailable") and not reader.isEventStreamAvailable():
            raise RuntimeError(f"No event stream available in {path}")
        if hasattr(reader, "getNextEventBatch"):
            while True:
                batch = reader.getNextEventBatch()
                if batch is None or (hasattr(batch, "isEmpty") and batch.isEmpty()):
                    break
                events_list.append(batch)

    arrays = []
    for ev in events_list:
        arr = _eventstore_to_numpy(ev)
        if arr is not None and len(arr) > 0:
            arrays.append(arr)

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not arrays:
        raise RuntimeError(f"No events could be read from {path}")

    events = np.concatenate(arrays)
    if events.dtype.names is not None:
        names = set(events.dtype.names)
        t_key = "t" if "t" in names else "timestamp"
        p_key = "p" if "p" in names else "polarity"
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)
        t = events[t_key].astype(np.int64)
        p = np.asarray(events[p_key]).astype(np.int32)
    else:
        t = events[:, 0].astype(np.int64)
        x = events[:, 1].astype(np.int32)
        y = events[:, 2].astype(np.int32)
        p = events[:, 3].astype(np.int32)

    p = np.where(p > 0, 1, 0).astype(np.int32)
    normalized = np.stack([x, y, t, p], axis=1)
    return normalized


def _scan_dataevent_paths(data_root: Path):
    mapping = {}
    for path in sorted(data_root.rglob("*.aedat4")):
        mapping[_canonical_sample_id(path)] = path
    return mapping


def _load_split_ids(raw_root: Path, split: str):
    path = raw_root / f"trials_to_{split}.txt"
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def build_split(sample_paths, sample_ids, output_dir: Path, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [sid for sid in sample_ids if sid not in sample_paths]
    if missing:
        print(f"[warn] {len(missing)} requested samples are missing from DataEvent.")
        for sid in missing[:20]:
            print(f"  missing: {sid}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    present_ids = [sid for sid in sample_ids if sid in sample_paths]
    print(f"Building {output_dir.name}: {len(present_ids)} samples")
    failed = []

    for sample_id in tqdm(present_ids):
        label = _label_from_sample_id(sample_id)
        dst = output_dir / f"{sample_id}_label{label:02d}.pckl"
        if dst.exists() and not overwrite:
            continue
        try:
            events = _read_all_events_aedat4(sample_paths[sample_id])
        except Exception as exc:
            failed.append((sample_id, str(sample_paths[sample_id]), str(exc)))
            continue
        pickle.dump((events, label), dst.open("wb"))

    if failed:
        print(f"[warn] {len(failed)} samples failed while building {output_dir.name}:")
        for sample_id, path, err in failed[:20]:
            print(f"  failed: {sample_id} | {path} | {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


def main():
    args = parse_args()
    dev_root = _repo_dev_root()
    raw_root = args.raw_root or (dev_root / "datasets" / "raw" / "EHWGesture")
    output_root = args.output_root or (
        dev_root / "datasets" / "evt_og" / "EHWGesture" / "clean_dataset"
    )
    data_root = raw_root / "DataEvent"

    if not data_root.exists():
        raise RuntimeError(f"Missing extracted DataEvent folder: {data_root}")

    sample_paths = _scan_dataevent_paths(data_root)
    splits = ["train", "val", "test"] if "all" in args.splits else args.splits

    print(f"Raw root: {raw_root}")
    print(f"DataEvent root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Indexed event samples: {len(sample_paths)}")

    for split in splits:
        split_ids = _load_split_ids(raw_root, split)
        build_split(
            sample_paths=sample_paths,
            sample_ids=split_ids,
            output_dir=output_root / split,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
