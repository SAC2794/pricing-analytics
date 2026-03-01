from etl.load import get_engine
from sqlalchemy import text

engine = get_engine()
tables = [
    "fact_sales", "dim_date", "dim_product", "dim_store", "dim_external",
    "metrics_pricing", "metrics_inventory", "metrics_forecast_features"
]
with engine.begin() as conn:
    for t in tables:
        conn.execute(text(f"DROP TABLE IF EXISTS dw.{t}"))
        print(f"  dropped dw.{t}")
print("All tables dropped.")