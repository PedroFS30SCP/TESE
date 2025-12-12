import os
import csv
import random
import struct
import tempfile

from dotenv import load_dotenv       # <--- NOVO
from azure.storage.blob import BlobServiceClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ================== CONFIG ==================
load_dotenv()  # Lê variáveis do ficheiro .env (se existir)

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set. Check your .env or environment variables.")

CONTAINER_NAME = "datasets"
DATASET_ROOT = "dataset_DVS128/"
DATA_PREFIX = DATASET_ROOT
# ============================================

# ---------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------
def get_container_client():
    """Return a ContainerClient for the configured container."""
    if not CONNECTION_STRING:
        raise RuntimeError("CONNECTION_STRING is not defined.")
    blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    return blob_service.get_container_client(CONTAINER_NAME)


def list_aedat_blobs(container_client):
    """List all .aedat blobs under DATA_PREFIX inside the container."""
    aedat_blobs = []
    for blob in container_client.list_blobs(name_starts_with=DATA_PREFIX):
        if blob.name.endswith(".aedat"):
            aedat_blobs.append(blob.name)
    if not aedat_blobs:
        raise RuntimeError("No .aedat files found in the container.")
    return aedat_blobs


def download_blob_to_temp(container_client, blob_name):
    """Download a blob to a temporary file and return its local path."""
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(blob_name)[1])
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------
# Mapping and labels
# ---------------------------------------------------------
def load_gesture_mapping(mapping_path):
    """
    Load gesture_mapping.csv → dict {class_id: gesture_name}.
    Expects columns: action,label
    """
    mapping = {}
    with open(mapping_path, newline="") as f:
        reader = csv.DictReader(f)
        if "action" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise RuntimeError("gesture_mapping.csv must have 'action' and 'label' columns.")
        for row in reader:
            name = row["action"].strip()
            class_id = int(row["label"])
            mapping[class_id] = name
    return mapping


def pick_random_segment(labels_path):
    """Pick a random gesture segment from a *_labels.csv file."""
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    if not rows:
        raise RuntimeError("No gesture segments found in " + labels_path)

    seg = random.choice(rows)
    class_id = int(seg["class"])
    t_start = int(seg["startTime_usec"])
    t_end = int(seg["endTime_usec"])
    return class_id, t_start, t_end


# ---------------------------------------------------------
# AEDAT 3.1 reader (polarity events)
# ---------------------------------------------------------
def read_aedat_polarity_events(path):
    """
    Simple AEDAT 3.1 polarity-event reader.

    Returns a numpy structured array with fields:
        x, y, p (polarity), t (timestamp)
    """
    events = []

    with open(path, "rb") as f:
        # Text header: lines starting with '#'
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"#"):
                # go back one line – this is the first binary header
                f.seek(pos)
                break

        header_struct = struct.Struct("<HHIIIIII")  # event header
        event_struct = struct.Struct("<II")         # (data, timestamp)

        while True:
            header_bytes = f.read(header_struct.size)
            if len(header_bytes) < header_struct.size:
                break

            (
                event_type,
                event_source,
                event_size,
                ts_offset,
                ts_overflow,
                event_capacity,
                event_number,
                event_valid,
            ) = header_struct.unpack(header_bytes)

            if event_number == 0 or event_size == 0:
                continue

            block_bytes = f.read(event_number * event_size)
            if len(block_bytes) < event_number * event_size:
                break

            for i in range(event_number):
                data, ts = event_struct.unpack_from(block_bytes, i * event_size)
                x = (data >> 17) & 0x00001FFF
                y = (data >> 2) & 0x00001FFF
                pol = (data >> 1) & 0x00000001
                events.append((x, y, pol, ts))

    if not events:
        return np.empty(0, dtype=[("x", "u2"), ("y", "u2"), ("p", "u1"), ("t", "u4")])

    arr = np.array(events, dtype=[("x", "u2"), ("y", "u2"), ("p", "u1"), ("t", "u4")])
    return arr


# ---------------------------------------------------------
# Animation + static view (side by side)
# ---------------------------------------------------------
def show_gesture_animation(events, t_start, t_end,
                           gesture_name="gesture", class_id=None,
                           n_frames=60, img_size=(128, 128)):

    # Filtrar janela temporal
    mask_all = (events["t"] >= t_start) & (events["t"] <= t_end)
    gesture_events = events[mask_all]
    if gesture_events.size == 0:
        print("No events in gesture window; cannot animate.")
        return

    H, W = img_size
    times = np.linspace(t_start, t_end, n_frames + 1, dtype=np.int64)

    frames = []
    for i in range(n_frames):
        t0, t1 = times[i], times[i + 1]
        sub = gesture_events[(gesture_events["t"] >= t0) & (gesture_events["t"] < t1)]

        img_pos = np.zeros((H, W), dtype=np.float32)
        img_neg = np.zeros((H, W), dtype=np.float32)

        if sub.size > 0:
            x = np.clip(sub["x"].astype(int), 0, W - 1)
            y = np.clip(sub["y"].astype(int), 0, H - 1)
            # FLIP vertical para pôr o homem de cabeça para cima
            y = H - 1 - y

            p = sub["p"]
            mask_pos = p == 1
            mask_neg = p == 0

            if mask_pos.any():
                np.add.at(img_pos, (y[mask_pos], x[mask_pos]), 1.0)
            if mask_neg.any():
                np.add.at(img_neg, (y[mask_neg], x[mask_neg]), 1.0)

            img_pos = np.log1p(img_pos)
            img_neg = np.log1p(img_neg)

        img_rgb = np.zeros((H, W, 3), dtype=np.float32)
        if img_pos.max() > 0:
            img_rgb[..., 0] = img_pos / img_pos.max()  # ON → vermelho
        if img_neg.max() > 0:
            img_rgb[..., 2] = img_neg / img_neg.max()  # OFF → azul

        frames.append((img_rgb * 255).astype(np.uint8))

    # -------- figura --------
    fig, (ax_anim, ax_static) = plt.subplots(1, 2, figsize=(10, 4))

    # ESQUERDA: animação
    im = ax_anim.imshow(frames[0], origin="lower")  # agora já está flipado nos dados
    title_text = gesture_name if class_id is None else f"{gesture_name} (class {class_id})"
    ax_anim.set_title(title_text)
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_xlim(0, W - 1)
    ax_anim.set_ylim(0, H - 1)
    ax_anim.set_aspect('equal', adjustable='box')

    # DIREITA: densidade total (ON+OFF) do gesto inteiro
    density_pos = np.zeros((H, W), dtype=np.float32)
    density_neg = np.zeros((H, W), dtype=np.float32)

    if gesture_events.size > 0:
        xg = np.clip(gesture_events["x"].astype(int), 0, W - 1)
        yg = np.clip(gesture_events["y"].astype(int), 0, H - 1)
        # FLIP vertical também aqui
        yg = H - 1 - yg

        pg = gesture_events["p"]
        mask_pos_g = pg == 1
        mask_neg_g = pg == 0

        if mask_pos_g.any():
            np.add.at(density_pos, (yg[mask_pos_g], xg[mask_pos_g]), 1.0)
        if mask_neg_g.any():
            np.add.at(density_neg, (yg[mask_neg_g], xg[mask_neg_g]), 1.0)

    density = np.log1p(density_pos + density_neg)
    im_static = ax_static.imshow(density, cmap="magma", origin="lower")
    ax_static.set_title("Densidade de eventos (log)")
    ax_static.set_xlabel("x")
    ax_static.set_ylabel("y")
    ax_static.set_xlim(0, W - 1)
    ax_static.set_ylim(0, H - 1)
    ax_static.set_aspect('equal', adjustable='box')

    fig.colorbar(im_static, ax=ax_static, label="nº de eventos (log)")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=150,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    container = get_container_client()

    # 1) Load local gesture mapping (action,label)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_local = os.path.join(script_dir, "gesture_mapping.csv")
    if not os.path.exists(mapping_local):
        raise RuntimeError(f"gesture_mapping.csv not found at {mapping_local}")
    gesture_mapping = load_gesture_mapping(mapping_local)

    # 2) Pick a random .aedat file from Azure
    aedat_blobs = list_aedat_blobs(container)
    aedat_blob = random.choice(aedat_blobs)

    # 3) Find corresponding labels file
    labels_blob = aedat_blob.replace(".aedat", "_labels.csv")

    aedat_local = download_blob_to_temp(container, aedat_blob)
    labels_local = download_blob_to_temp(container, labels_blob)

    # 4) Pick a random gesture segment inside this trial
    class_id, t_start, t_end = pick_random_segment(labels_local)
    gesture_name = gesture_mapping.get(class_id, f"class_{class_id}")

    print(f"Random gesture from file: {aedat_blob}")
    print(f"  class id : {class_id}")
    print(f"  name     : {gesture_name}")
    print(f"  window   : {t_start} – {t_end} µs")

    # 5) Read all events and show animation + static view
    events = read_aedat_polarity_events(aedat_local)
    show_gesture_animation(
        events,
        t_start,
        t_end,
        gesture_name=gesture_name,
        class_id=class_id,
        n_frames=60,
        img_size=(128, 128),
    )


if __name__ == "__main__":
    main()