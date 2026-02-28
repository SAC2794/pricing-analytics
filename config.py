# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Pricing Pipeline.
# No secrets / .env needed: authentication is handled entirely via
# Azure Managed Identity (DefaultAzureCredential).
# ─────────────────────────────────────────────────────────────────────────────

# ── Azure Blob Storage ────────────────────────────────────────────────────────
AZURE_STORAGE_ACCOUNT  = "pricingpipelineexec"
AZURE_CONTAINER_NAME   = "raw"

# ── Azure SQL Database ────────────────────────────────────────────────────────
AZURE_SQL_SERVER   = "sql-pricing-analytics.database.windows.net"
AZURE_SQL_DATABASE = "pricing_analytics_db"
AZURE_SQL_DRIVER   = "ODBC Driver 17 for SQL Server"

# ── Data Warehouse schema ─────────────────────────────────────────────────────
SQL_SCHEMA = "dw"

# ── Table names ───────────────────────────────────────────────────────────────
# Dimension tables  (Power BI)
DIM_DATE_TABLE     = "dim_date"
DIM_PRODUCT_TABLE  = "dim_product"
DIM_STORE_TABLE    = "dim_store"
DIM_EXTERNAL_TABLE = "dim_external"

# Fact table  (Power BI)
FACT_SALES_TABLE   = "fact_sales"

# Metrics tables  (Streamlit + ML models)
METRICS_PRICING_TABLE      = "metrics_pricing"
METRICS_INVENTORY_TABLE    = "metrics_inventory"
METRICS_CUSTOMER_TABLE     = "metrics_customer_segmentation"
METRICS_FORECAST_TABLE     = "metrics_forecast_features"