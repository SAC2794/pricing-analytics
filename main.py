# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator.
# Entry point for both local execution and Azure Functions.
#
# Flow:
#   1. Extract CSV from Azure Blob Storage
#   2. Clean + type-cast
#   3. Derive business KPIs
#   4. Build dimension tables          → Power BI
#   5. Build fact table                → Power BI
#   6. Build metrics tables            → Streamlit + ML models
#   7. Load everything to Azure SQL
# ─────────────────────────────────────────────────────────────────────────────

from etl.extract import extract_csv_from_datalake
from etl.transform import (
    clean_columns,
    convert_types,
    create_derived_metrics,
    create_dim_date,
    create_dimensions,
    create_fact_table,
    create_metric_tables,
)
from etl.load import load_to_sql
from config import (
    DIM_DATE_TABLE, DIM_PRODUCT_TABLE, DIM_STORE_TABLE, DIM_EXTERNAL_TABLE,
    FACT_SALES_TABLE,
    METRICS_PRICING_TABLE, METRICS_INVENTORY_TABLE,
    METRICS_CUSTOMER_TABLE, METRICS_FORECAST_TABLE,
)


def run_pipeline(file_name: str = "retail_store_inventory.csv"):

    try:
        # ── 1. Extract ────────────────────────────────────────────────────────
        print("\n🔹 [1/7] Extracting data...")
        df = extract_csv_from_datalake(file_name)

        # ── 2. Clean ──────────────────────────────────────────────────────────
        print("\n🔹 [2/7] Cleaning columns...")
        df = clean_columns(df)

        print("\n🔹 [3/7] Converting types...")
        df = convert_types(df)

        # ── 3. Derive metrics ─────────────────────────────────────────────────
        print("\n🔹 [4/7] Creating derived metrics...")
        df = create_derived_metrics(df)

        # ── 4. Build dimensions ───────────────────────────────────────────────
        print("\n🔹 [5/7] Building dimension tables...")
        dim_date                  = create_dim_date(df)
        dim_product, dim_store, dim_external = create_dimensions(df)

        # ── 5. Build fact table ───────────────────────────────────────────────
        print("\n🔹 [6/7] Building fact table...")
        fact_sales = create_fact_table(df)

        # ── 6. Build metrics tables ───────────────────────────────────────────
        print("\n🔹 [7/7] Building metrics tables for ML & Streamlit...")
        metrics = create_metric_tables(df)

        # ── 7. Load to Azure SQL ──────────────────────────────────────────────
        print("\n🔹 Loading dimension tables to Azure SQL...")
        load_to_sql(dim_date,     DIM_DATE_TABLE)
        load_to_sql(dim_product,  DIM_PRODUCT_TABLE)
        load_to_sql(dim_store,    DIM_STORE_TABLE)
        load_to_sql(dim_external, DIM_EXTERNAL_TABLE)

        print("\n🔹 Loading fact table to Azure SQL...")
        load_to_sql(fact_sales, FACT_SALES_TABLE)

        print("\n🔹 Loading metrics tables to Azure SQL...")
        load_to_sql(metrics["metrics_pricing"],               METRICS_PRICING_TABLE)
        load_to_sql(metrics["metrics_inventory"],             METRICS_INVENTORY_TABLE)
        load_to_sql(metrics["metrics_customer_segmentation"], METRICS_CUSTOMER_TABLE)
        load_to_sql(metrics["metrics_forecast_features"],     METRICS_FORECAST_TABLE)

        print("\n✅ Pipeline completed successfully!\n")

    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}")
        raise   # re-raise so Azure Functions marks the execution as failed


# ─────────────────────────────────────────────────────────────────────────────
# Local execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline("retail_store_inventory.csv")