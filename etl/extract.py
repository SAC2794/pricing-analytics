# etl/extract.py
# ─────────────────────────────────────────────────────────────────────────────
# Extracts the raw CSV from Azure Blob Storage using Managed Identity.
# No connection strings or secrets required.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from io import BytesIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER_NAME


def get_blob_service_client() -> BlobServiceClient:
    """
    Creates a BlobServiceClient authenticated with Managed Identity.
    Works both locally (via az login / env vars) and inside Azure Functions.
    """
    credential   = DefaultAzureCredential()
    account_url  = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=credential)


def extract_csv_from_datalake(file_name: str) -> pd.DataFrame:
    """
    Downloads a CSV blob from the 'raw' container and returns a DataFrame.

    Parameters
    ----------
    file_name : str
        Name of the blob, e.g. 'retail_store_inventory.csv'

    Returns
    -------
    pd.DataFrame
    """
    if not AZURE_CONTAINER_NAME:
        raise ValueError("AZURE_CONTAINER_NAME is not set in config.py")

    client      = get_blob_service_client()
    blob_client = client.get_blob_client(
        container=AZURE_CONTAINER_NAME,
        blob=file_name
    )

    stream    = BytesIO()
    blob_data = blob_client.download_blob()
    blob_data.readinto(stream)
    stream.seek(0)

    df = pd.read_csv(stream)
    print(f"  ✔ Extracted {len(df):,} rows × {len(df.columns)} columns from '{file_name}'")
    return df