# etl/load.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads DataFrames into Azure SQL Database using Managed Identity (no password).
# Uses pyodbc + SQLAlchemy with token-based authentication.
# ─────────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct
import urllib

from sqlalchemy import create_engine, text
from azure.identity import DefaultAzureCredential
from config import AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_DRIVER, SQL_SCHEMA

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    """
    Builds a SQLAlchemy engine authenticated with Azure Managed Identity.
    The access token is injected via ODBC attribute 1256 (SQL_COPT_SS_ACCESS_TOKEN).
    """
    credential = DefaultAzureCredential()
    token_obj  = credential.get_token("https://database.windows.net/.default")

    # Convert token string to the UTF-16LE byte structure expected by ODBC Driver 17/18
    token_bytes = token_obj.token.encode("utf-16-le")
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
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        connect_args={"attrs_before": {1256: token_struct}},
        fast_executemany=True,   # much faster bulk inserts
    )
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def ensure_schema_exists(engine):
    """Creates the 'dw' schema if it doesn't already exist."""
    with engine.begin() as conn:
        conn.execute(text(f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{SQL_SCHEMA}') "
                          f"EXEC('CREATE SCHEMA {SQL_SCHEMA}')"))


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_to_sql(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    chunksize: int = 1000,
):
    """
    Writes a DataFrame to Azure SQL.

    Parameters
    ----------
    df         : DataFrame to load
    table_name : Target table name (without schema prefix)
    if_exists  : 'replace' (default) drops & recreates. Use 'append' for incremental loads.
    chunksize  : Rows per INSERT batch (default 1 000)
    """
    if df is None or df.empty:
        print(f"  ⚠ Skipped '{table_name}' – DataFrame is empty")
        return

    engine = get_engine()
    ensure_schema_exists(engine)

    # Convert pandas NA types to Python None so pyodbc handles them correctly
    df = df.where(pd.notnull(df), other=None)

    df.to_sql(
        name=table_name,
        con=engine,
        schema=SQL_SCHEMA,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    print(f"  ✔ Loaded {len(df):,} rows → {SQL_SCHEMA}.{table_name}  (if_exists='{if_exists}')")