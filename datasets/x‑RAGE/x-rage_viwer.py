import random
import webbrowser
from datetime import datetime, timedelta
import os

from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)

# ------------------------------------
# CONFIGURATION
# ------------------------------------
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

CONTAINER_NAME = "datasets"
# ------------------------------------


def get_container_client():
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    return blob_service_client.get_container_client(CONTAINER_NAME)


def choose_random_gif(container_client) -> str:
    """
    Look at ALL blobs in the container, print their names,
    then pick a random one inside 'event_gifs' whose name ends with 'gif'.
    """
    all_blobs = list(container_client.list_blobs())

    print("Blobs found in container:")
    for b in all_blobs:
        print("   ", b.name)

    gif_names = [
        b.name
        for b in all_blobs
        if "event_gifs" in b.name and b.name.lower().endswith("gif")
    ]

    if not gif_names:
        raise RuntimeError("No GIF files found that match *event_gifs* and end with 'gif'.")

    return random.choice(gif_names)


def generate_sas_url(account_name, account_key, container_name, blob_name, expiration_minutes=10):
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=expiration_minutes),
    )

    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"


if __name__ == "__main__":
    container_client = get_container_client()

    blob_name = choose_random_gif(container_client)
    print(f"\nSelected GIF blob: {blob_name}")

    sas_url = generate_sas_url(
        ACCOUNT_NAME,
        ACCOUNT_KEY,
        CONTAINER_NAME,
        blob_name,
    )

    print(f"\nOpening URL:\n{sas_url}\n")

    # Open directly in the default web browser
    webbrowser.open(sas_url)