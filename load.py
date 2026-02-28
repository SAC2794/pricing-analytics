# etl/load.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads DataFrames into Azure SQL Database using Managed Identity (no password).
# Uses pyodbc + SQLAlchemy with token-based authentication.
# ─────────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct
import urllib
import pandas as pd
from sqlalchemy import create_engine, text
from azure.identity import DefaultAzureCredential
from config import AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_DRIVER, SQL_SCHEMA


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    credential   = DefaultAzureCredential()
    token_obj    = credential.get_token("https://database.windows.net/.default")
    token_bytes  = token_obj.token.encode("utf-16-le")
    token_struct = struct.pack("<I", len(token_bytes)) + token_bytes
    odbc_str = (
        f"Driver={{{AZURE_SQL_DRIVER}}};"
        f"Server=tcp:{AZURE_SQL_SERVER},1433;"
        f"Database={AZURE_SQL_DATABASE};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    params = urllib.parse.quote_plus(odbc_str)
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        connect_args={"attrs_before": {1256: token_struct}},
        fast_executemany=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def ensure_schema_exists(engine):
    with engine.begin() as conn:
        conn.execute(text(
            f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{SQL_SCHEMA}') "
            f"EXEC('CREATE SCHEMA {SQL_SCHEMA}')"
        ))


def _table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text(
            f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{SQL_SCHEMA}' AND TABLE_NAME = '{table_name}'"
        ))
        return result.fetchone() is not None


def _truncate_or_create(engine, df: pd.DataFrame, table_name: str, chunksize: int):
    if _table_exists(engine, table_name):
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE [{SQL_SCHEMA}].[{table_name}]"))
        df.to_sql(
            name=table_name, con=engine, schema=SQL_SCHEMA,
            if_exists="append", index=False, chunksize=chunksize, method="multi",
        )
    else:
        df.to_sql(
            name=table_name, con=engine, schema=SQL_SCHEMA,
            if_exists="replace", index=False, chunksize=chunksize, method="multi",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_to_sql(df: pd.DataFrame, table_name: str, if_exists: str = "replace", chunksize: int = 1000):
    if df is None or df.empty:
        print(f"  ⚠ Skipped '{table_name}' – DataFrame is empty")
        return
    engine = get_engine()
    ensure_schema_exists(engine)
    df = df.where(pd.notnull(df), other=None)
    _truncate_or_create(engine, df, table_name, chunksize)
    print(f"  ✔ Loaded {len(df):,} rows → [{SQL_SCHEMA}].[{table_name}]")
