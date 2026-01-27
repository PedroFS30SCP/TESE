import os
import io
import random
import re
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dotenv import load_dotenv          # <--- NOVO
from azure.storage.blob import BlobServiceClient

# ================== CONFIG ==================
load_dotenv()  # Lê o .env se existir

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set. Check your .env or environment variables.")

CONTAINER_NAME = "datasets"
DATASET_ROOT = "dataset_LynX"
# ============================================

def get_container_client():
    """Create and return a container client for the given storage account."""
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    return blob_service_client.get_container_client(CONTAINER_NAME)


def load_segmentation_csv() -> pd.DataFrame:
    """
    Load gesture_segmentation.csv from the local filesystem
    (same directory as this script) into a pandas DataFrame.
    """
    csv_path = "gesture_segmentation.csv"
    print(f"Loading local CSV file: {csv_path}")
    return pd.read_csv(csv_path)


def choose_random_gesture(df: pd.DataFrame):
    """Pick a random row (gesture instance) from the segmentation DataFrame."""
    row = df.sample(1).iloc[0]
    subject = row["subject"]
    scenario = row["scenario"]
    gesture_seq = row["gesture_sequence"]
    gesture_name = row["gesture_name"]
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    return subject, scenario, gesture_seq, gesture_name, start_frame, end_frame


def find_frame_blob_name(container_client, prefixes, frame_idx: int) -> str:
    """
    Search for an image blob whose filename contains the given frame index.
    prefixes: list of possible prefixes (e.g. frames/ and images/).
    """
    for prefix in prefixes:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            fname = os.path.basename(blob.name)
            base, ext = os.path.splitext(fname)

            # Only consider image files
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            # Extract first integer from the filename
            m = re.search(r"(\d+)", base)
            if m and int(m.group(1)) == frame_idx:
                return blob.name

    raise FileNotFoundError(
        "No image found for frame {} under prefixes: {}".format(frame_idx, prefixes)
    )


def download_image_from_blob(container_client, blob_name: str) -> Image.Image:
    """Download an image blob and return it as a PIL Image."""
    blob_client = container_client.get_blob_client(blob_name)
    img_bytes = blob_client.download_blob().readall()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def download_annotation(container_client, ann_blob_name: str) -> Optional[str]:
    """
    Download a YOLO annotation file as text.
    Returns None if the blob does not exist or cannot be read.
    """
    blob_client = container_client.get_blob_client(ann_blob_name)
    try:
        txt = blob_client.download_blob().readall().decode("utf-8")
        return txt
    except Exception:
        return None


def draw_yolo_bboxes(img: Image.Image, ann_text: Optional[str]) -> Image.Image:
    """
    Draw YOLO-format bounding boxes on the image, if annotation text is provided.
    YOLO format: class_id x_center y_center width height (all normalized 0..1).
    """
    if not ann_text:
        return img

    w, h = img.size
    draw = ImageDraw.Draw(img)

    for line in ann_text.strip().splitlines():
        vals = line.strip().split()
        if len(vals) != 5:
            continue

        class_id, x_c, y_c, bw, bh = vals
        x_c, y_c, bw, bh = float(x_c), float(y_c), float(bw), float(bh)

        # Convert normalized coords to pixel coords
        x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
        x1, y1 = x_c - bw / 2.0, y_c - bh / 2.0
        x2, y2 = x_c + bw / 2.0, y_c + bh / 2.0

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return img


def main():
    # 1) Connect to the container (for images and annotations)
    container_client = get_container_client()

    # 2) Load segmentation CSV from local file
    df = load_segmentation_csv()

    # 3) Choose a random gesture instance
    subject, scenario, gesture_seq, gesture_name, start_f, end_f = choose_random_gesture(
        df
    )

    print("Random gesture instance: {}".format(gesture_name))
    print("  subject    = {}".format(subject))
    print("  scenario   = {}".format(scenario))
    print("  gesture    = {}".format(gesture_seq))
    print("  frame span = [{} , {}]".format(start_f, end_f))

    # 4) Choose a random frame index within this gesture
    frame_idx = random.randint(start_f, end_f)
    print("Random frame index within gesture: {}".format(frame_idx))

    # 5) Possible prefixes for frames/images
    frames_prefix = "{}/{}/{}/{}/frames/".format(
        DATASET_ROOT, subject, scenario, gesture_seq
    )
    images_prefix = "{}/{}/{}/{}/images/".format(
        DATASET_ROOT, subject, scenario, gesture_seq
    )
    prefixes = [frames_prefix, images_prefix]

    # 6) Find the corresponding image blob
    img_blob_name = find_frame_blob_name(container_client, prefixes, frame_idx)
    print("Image blob: {}".format(img_blob_name))

    # 7) Download the image
    img = download_image_from_blob(container_client, img_blob_name)

    # 8) Download the corresponding annotation (if it exists)
    ann_blob_name = "{}/{}/{}/{}/annotations/{}.txt".format(
        DATASET_ROOT, subject, scenario, gesture_seq, frame_idx
    )
    ann_text = download_annotation(container_client, ann_blob_name)
    if ann_text is None:
        print("⚠️ No annotation found for this frame.")
    else:
        print("Annotation loaded from: {}".format(ann_blob_name))

    # 9) Draw bounding boxes (if available)
    img = draw_yolo_bboxes(img, ann_text)

    # 10) Show the image with title
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        "{} | {}, {}, {}, frame {}".format(
            gesture_name, subject, scenario, gesture_seq, frame_idx
        )
    )
    plt.show()

    print("\n➡ This frame belongs to gesture: {}".format(gesture_name))


if __name__ == "__main__":
    main()