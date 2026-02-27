# etl/extract.py

import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO

load_dotenv()

def get_blob_service_client():
    try:
        credential = ClientSecretCredential(
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            client_id=os.getenv("AZURE_CLIENT_ID"),
            client_secret=os.getenv("AZURE_CLIENT_SECRET")
        )

        account_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net"
        client = BlobServiceClient(account_url=account_url, credential=credential)
        print("[✅] Conexión al Storage exitosa")
        return client
    except Exception as e:
        print(f"[❌] Error conectando al Storage: {e}")
        raise e

def extract_csv_from_datalake(file_name: str) -> pd.DataFrame:
    try:
        blob_service_client = get_blob_service_client()
        container_name = os.getenv("AZURE_CONTAINER_NAME")
        container_client = blob_service_client.get_container_client(container_name)

        # Verificar si el blob existe
        if not container_client.exists():
            raise Exception(f"Contenedor '{container_name}' no existe o no tienes permisos")
        
        blob_client = container_client.get_blob_client(file_name)
        if not blob_client.exists():
            raise Exception(f"Archivo '{file_name}' no existe en el contenedor '{container_name}'")

        stream = BytesIO()
        blob_data = blob_client.download_blob()
        blob_data.readinto(stream)
        stream.seek(0)

        df = pd.read_csv(stream)
        print(f"[✅] CSV '{file_name}' cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df

    except Exception as e:
        print(f"[❌] Error extrayendo CSV: {e}")
        raise e

if __name__ == "__main__":
    # Cambia el nombre de tu archivo real aquí
    file_name = "sales_data.csv"
    df = extract_csv_from_datalake(file_name)
    print(df.head())