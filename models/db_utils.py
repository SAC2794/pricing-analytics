# models/db_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared helper usado por todos los modelos ML.
# Lee tablas desde Azure SQL y escribe resultados de vuelta.
# Evita duplicar la lógica de conexión en cada modelo.
#
# Importado por:
#   models/random_forest.py
#   models/linear_regression.py
#   models/segmentation.py
#   models/forecasting.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from etl.load import get_engine, load_to_sql
from config import SQL_SCHEMA


def read_table(table_name: str, query: str = None) -> pd.DataFrame:
    """
    Lee una tabla completa o una query SQL desde Azure SQL.

    Parameters
    ----------
    table_name : str  — nombre de la tabla (sin schema), usado si query es None
    query      : str  — SQL personalizado, sobreescribe table_name si se pasa

    Returns
    -------
    pd.DataFrame
    """
    engine = get_engine()
    sql = query if query else f"SELECT * FROM [{SQL_SCHEMA}].[{table_name}]"
    df = pd.read_sql(sql, con=engine)
    print(f"  ✔ Read {len(df):,} rows from [{SQL_SCHEMA}].[{table_name}]")
    return df


def write_results(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    """
    Persiste el output de un modelo en Azure SQL.

    Parameters
    ----------
    df         : DataFrame con los resultados del modelo
    table_name : nombre de la tabla destino (sin schema)
    if_exists  : 'replace' (default) o 'append'
    """
    load_to_sql(df, table_name, if_exists=if_exists)