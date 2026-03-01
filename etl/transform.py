# etl/transform.py
# ─────────────────────────────────────────────────────────────────────────────
# All transformation logic:
#   1. clean_columns          – normalize column names
#   2. convert_types          – cast to correct dtypes
#   3. create_derived_metrics – business KPIs on the raw dataframe
#   4. create_dim_date        – calendar dimension
#   5. create_dimensions      – dim_product / dim_store / dim_external
#   6. create_fact_table      – fact_sales
#   7. create_metric_tables   – metrics for ML models & Streamlit
# ─────────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Column name normalisation
# ─────────────────────────────────────────────────────────────────────────────

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to snake_case and strip whitespace.
    Drops fully-null rows.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)
    print(f"  ✔ Cleaned columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Type conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to correct dtypes.
    The column mapping covers the known schema of retail_store_inventory.csv.
    Unknown columns are left as-is.
    """
    df = df.copy()

    # Date columns
    date_cols = ["date", "last_order_date", "restock_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Integer columns
    int_cols = [
        "store_id", "product_id", "units_sold", "units_ordered",
        "inventory_level", "reorder_point", "lead_time_days",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Aliases: normaliza nombres alternativos del dataset real
    aliases = {
        "competitor_pricing": "competitor_price",
        "seasonality":        "seasonality_factor",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Float columns
    float_cols = [
        "price", "cost", "competitor_price", "discount",
        "demand_forecast", "weather_condition", "holiday_promotion",
        "seasonality_factor",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # String / categorical
    str_cols = [
        "product_name", "category", "store_location",
        "region", "supplier", "storage_type",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print("  ✔ Types converted")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Derived / business metrics added to the base dataframe
# ─────────────────────────────────────────────────────────────────────────────

def create_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds computed columns used by both the fact table and the metrics tables.
    """
    df = df.copy()

    # Net price after discount
    if "price" in df.columns and "discount" in df.columns:
        df["net_price"] = df["price"] * (1 - df["discount"].fillna(0))

    # Revenue
    if "net_price" in df.columns and "units_sold" in df.columns:
        df["revenue"] = df["net_price"] * df["units_sold"].fillna(0)

    # Cost: usa la columna real si existe, si no la estima con margen típico de retail (40%)
    if "cost" not in df.columns and "price" in df.columns:
        df["cost"] = df["price"] * 0.60  # estimación: costo = 60% del precio lista

    # Gross margin
    if "net_price" in df.columns and "cost" in df.columns:
        df["gross_margin"] = df["net_price"] - df["cost"]
        df["gross_margin_pct"] = np.where(
            df["net_price"] > 0,
            df["gross_margin"] / df["net_price"],
            np.nan,
        )

    # Price gap vs competitor
    if "price" in df.columns and "competitor_price" in df.columns:
        df["price_gap_vs_competitor"] = df["price"] - df["competitor_price"]
        df["price_gap_pct"] = np.where(
            df["competitor_price"] > 0,
            df["price_gap_vs_competitor"] / df["competitor_price"],
            np.nan,
        )

    # Inventory coverage (days of stock on hand)
    if "inventory_level" in df.columns and "units_sold" in df.columns:
        daily_sales = df["units_sold"].replace(0, np.nan)
        df["stock_coverage_days"] = df["inventory_level"] / daily_sales

    # Stock status flag
    if "inventory_level" in df.columns and "reorder_point" in df.columns:
        df["stock_status"] = np.where(
            df["inventory_level"] <= df["reorder_point"], "low_stock", "ok"
        )

    # Demand accuracy (forecast vs actual)
    if "demand_forecast" in df.columns and "units_sold" in df.columns:
        df["forecast_error"] = df["demand_forecast"] - df["units_sold"].fillna(0)
        df["forecast_accuracy"] = np.where(
            df["demand_forecast"] > 0,
            1 - abs(df["forecast_error"]) / df["demand_forecast"],
            np.nan,
        )

    print("  ✔ Derived metrics created")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. dim_date
# ─────────────────────────────────────────────────────────────────────────────

def create_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar dimension from all dates present in the 'date' column.
    """
    if "date" not in df.columns:
        raise KeyError("Column 'date' not found. Check clean_columns / convert_types.")

    dates = df["date"].dropna().drop_duplicates().sort_values()

    dim = pd.DataFrame({"date": dates})
    dim["date_id"]       = dim["date"].dt.strftime("%Y%m%d").astype(int)
    dim["year"]          = dim["date"].dt.year
    dim["quarter"]       = dim["date"].dt.quarter
    dim["month"]         = dim["date"].dt.month
    dim["month_name"]    = dim["date"].dt.strftime("%B")
    dim["week"]          = dim["date"].dt.isocalendar().week.astype(int)
    dim["day_of_month"]  = dim["date"].dt.day
    dim["day_of_week"]   = dim["date"].dt.dayofweek          # 0=Mon … 6=Sun
    dim["day_name"]      = dim["date"].dt.strftime("%A")
    dim["is_weekend"]    = dim["day_of_week"].isin([5, 6]).astype(int)
    dim["year_month"]    = dim["date"].dt.to_period("M").astype(str)

    print(f"  ✔ dim_date built: {len(dim):,} rows")
    return dim.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dimension tables
# ─────────────────────────────────────────────────────────────────────────────

def create_dimensions(df: pd.DataFrame):
    """
    Returns (dim_product, dim_store, dim_external).
    """

    # ── dim_product ───────────────────────────────────────────────────────────
    product_cols = [c for c in [
        "product_id", "product_name", "category",
        "supplier", "storage_type", "cost", "reorder_point", "lead_time_days",
    ] if c in df.columns]

    dim_product = (
        df[product_cols]
        .drop_duplicates(subset=["product_id"])
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    print(f"  ✔ dim_product built: {len(dim_product):,} rows")

    # ── dim_store ─────────────────────────────────────────────────────────────
    store_cols = [c for c in [
        "store_id", "store_location", "region",
    ] if c in df.columns]

    dim_store = (
        df[store_cols]
        .drop_duplicates(subset=["store_id"])
        .sort_values("store_id")
        .reset_index(drop=True)
    )
    print(f"  ✔ dim_store built: {len(dim_store):,} rows")

    # ── dim_external ──────────────────────────────────────────────────────────
    # External / contextual factors that vary by date (not by product/store)
    external_cols = [c for c in [
        "date", "weather_condition", "holiday_promotion", "seasonality_factor",
    ] if c in df.columns]

    dim_external = (
        df[external_cols]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"  ✔ dim_external built: {len(dim_external):,} rows")

    return dim_product, dim_store, dim_external


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fact table
# ─────────────────────────────────────────────────────────────────────────────

def create_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transactional fact table.  Each row = one product × store × date observation.
    """
    fact_cols = [c for c in [
        # Keys
        "date", "store_id", "product_id",
        # Measures
        "units_sold", "units_ordered",
        "inventory_level", "demand_forecast",
        "price", "discount", "net_price",
        "cost", "gross_margin", "gross_margin_pct",
        "revenue",
        "competitor_price", "price_gap_vs_competitor", "price_gap_pct",
        "stock_coverage_days", "stock_status",
        "forecast_error", "forecast_accuracy",
    ] if c in df.columns]

    fact_sales = df[fact_cols].copy()
    print(f"  ✔ fact_sales built: {len(fact_sales):,} rows × {len(fact_cols)} cols")
    return fact_sales.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics tables  (Streamlit + ML)
# ─────────────────────────────────────────────────────────────────────────────

def create_metric_tables(df: pd.DataFrame) -> dict:
    """
    Returns a dict of DataFrames, each tailored for a specific model / dashboard:

    Keys
    ----
    'metrics_pricing'              → regression & price optimisation
    'metrics_inventory'            → inventory KPIs & replenishment
    'metrics_customer_segmentation'→ RFM + behavioural features for clustering
    'metrics_forecast_features'    → time-series / forecasting features
    """

    # ── metrics_pricing ───────────────────────────────────────────────────────
    pricing_cols = [c for c in [
        "date", "store_id", "product_id", "category",
        "price", "cost", "discount", "net_price",
        "competitor_price", "price_gap_vs_competitor", "price_gap_pct",
        "units_sold", "revenue",
        "gross_margin", "gross_margin_pct",
        "seasonality_factor", "holiday_promotion",
        "demand_forecast", "forecast_accuracy",
    ] if c in df.columns]

    metrics_pricing = df[pricing_cols].dropna(
        subset=[c for c in ["price", "units_sold"] if c in df.columns]
    ).copy()

    # Price elasticity proxy (rolling – requires sorted data)
    if "price" in metrics_pricing.columns and "units_sold" in metrics_pricing.columns:
        metrics_pricing = metrics_pricing.sort_values(["product_id", "store_id", "date"])
        grp = metrics_pricing.groupby(["product_id", "store_id"])
        metrics_pricing["pct_change_price"] = grp["price"].pct_change()
        metrics_pricing["pct_change_units"] = grp["units_sold"].pct_change()
        metrics_pricing["price_elasticity_proxy"] = np.where(
            metrics_pricing["pct_change_price"] != 0,
            metrics_pricing["pct_change_units"] / metrics_pricing["pct_change_price"],
            np.nan,
        )

    print(f"  ✔ metrics_pricing: {len(metrics_pricing):,} rows")

    # ── metrics_inventory ─────────────────────────────────────────────────────
    inv_cols = [c for c in [
        "date", "store_id", "product_id", "category",
        "inventory_level", "units_sold", "units_ordered",
        "reorder_point", "lead_time_days",
        "stock_coverage_days", "stock_status",
        "demand_forecast", "forecast_error", "forecast_accuracy",
        "seasonality_factor",
    ] if c in df.columns]

    metrics_inventory = df[inv_cols].copy()

    # Fill rate proxy
    if "units_sold" in metrics_inventory.columns and "demand_forecast" in metrics_inventory.columns:
        metrics_inventory["fill_rate"] = np.where(
            metrics_inventory["demand_forecast"] > 0,
            metrics_inventory["units_sold"].fillna(0) / metrics_inventory["demand_forecast"],
            np.nan,
        ).clip(0, 1)

    # Overstock / understock flag
    if "inventory_level" in metrics_inventory.columns and "demand_forecast" in metrics_inventory.columns:
        metrics_inventory["overstock_flag"] = (
            metrics_inventory["inventory_level"] > 2 * metrics_inventory["demand_forecast"]
        ).astype(int)

    print(f"  ✔ metrics_inventory: {len(metrics_inventory):,} rows")

    # ── metrics_customer_segmentation ────────────────────────────────────────
    # RFM (Recency / Frequency / Monetary) aggregated at product × store level.
    # In a pure inventory dataset there are no individual customers, so we
    # proxy RFM at (product_id, store_id) grain.
    rfm_required = {"date", "product_id", "store_id", "units_sold", "revenue"}
    available    = rfm_required & set(df.columns)

    if len(available) == len(rfm_required):
        reference_date = df["date"].max()

        rfm = (
            df.groupby(["product_id", "store_id"])
            .agg(
                recency_days   =("date",       lambda x: (reference_date - x.max()).days),
                frequency      =("date",       "count"),
                monetary_total =("revenue",    "sum"),
                avg_units_sold =("units_sold", "mean"),
                total_units    =("units_sold", "sum"),
            )
            .reset_index()
        )

        # RFM scores (quintiles 1–5)
        # duplicates="drop" puede reducir el número real de bins,
        # por eso calculamos los labels dinámicamente según los bins resultantes.
        for col, score_col, ascending in [
            ("recency_days",   "r_score", True),   # lower recency = better
            ("frequency",      "f_score", False),
            ("monetary_total", "m_score", False),
        ]:
            _, bins = pd.qcut(rfm[col], q=5, retbins=True, duplicates="drop")
            n_bins  = len(bins) - 1
            labels  = list(range(n_bins, 0, -1)) if ascending else list(range(1, n_bins + 1))
            # Cast Categorical → float → Int64 para poder operar con los scores
            rfm[score_col] = pd.qcut(rfm[col], q=5, labels=labels, duplicates="drop")
            rfm[score_col] = rfm[score_col].astype(float).astype("Int64")

        rfm["rfm_score"]   = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
        rfm["rfm_numeric"] = rfm["r_score"].astype(float) + rfm["f_score"].astype(float) + rfm["m_score"].astype(float)

        # Simple segment label
        def _segment(score):
            if score >= 13:  return "Champions"
            if score >= 10:  return "Loyal"
            if score >= 7:   return "Potential"
            if score >= 4:   return "At Risk"
            return "Lost"

        rfm["segment"] = rfm["rfm_numeric"].apply(_segment)

        # Enrich with product/store details if available
        for col in ["category", "store_location", "region"]:
            if col in df.columns:
                mapping = df.drop_duplicates(subset=["product_id" if col=="category" else "store_id"])[
                    ["product_id" if col=="category" else "store_id", col]
                ]
                merge_key = "product_id" if col == "category" else "store_id"
                rfm = rfm.merge(mapping, on=merge_key, how="left")

        metrics_customer = rfm
    else:
        missing = rfm_required - available
        print(f"  ⚠ Skipping RFM – missing columns: {missing}")
        metrics_customer = pd.DataFrame()

    print(f"  ✔ metrics_customer_segmentation: {len(metrics_customer):,} rows")

    # ── metrics_forecast_features ────────────────────────────────────────────
    # Feature engineering for time-series forecasting (Random Forest / XGBoost).
    ts_required = {"date", "product_id", "store_id", "units_sold"}
    if ts_required <= set(df.columns):
        ts = df[list(ts_required | ({"price","inventory_level","discount",
                                     "seasonality_factor","holiday_promotion",
                                     "weather_condition"} & set(df.columns)))].copy()
        ts = ts.sort_values(["product_id", "store_id", "date"])
        grp = ts.groupby(["product_id", "store_id"])

        # Lag features
        for lag in [1, 7, 14, 30]:
            ts[f"units_sold_lag{lag}"] = grp["units_sold"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            ts[f"units_sold_roll_mean_{window}"] = (
                grp["units_sold"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            ts[f"units_sold_roll_std_{window}"] = (
                grp["units_sold"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )

        # Calendar features
        ts["year"]       = ts["date"].dt.year
        ts["month"]      = ts["date"].dt.month
        ts["week"]       = ts["date"].dt.isocalendar().week.astype(int)
        ts["day_of_week"]= ts["date"].dt.dayofweek
        ts["is_weekend"] = ts["day_of_week"].isin([5,6]).astype(int)
        ts["quarter"]    = ts["date"].dt.quarter

        metrics_forecast = ts
    else:
        missing = ts_required - set(df.columns)
        print(f"  ⚠ Skipping forecast features – missing: {missing}")
        metrics_forecast = pd.DataFrame()

    print(f"  ✔ metrics_forecast_features: {len(metrics_forecast):,} rows")

    return {
        "metrics_pricing":               metrics_pricing,
        "metrics_inventory":             metrics_inventory,
        "metrics_customer_segmentation": metrics_customer,
        "metrics_forecast_features":     metrics_forecast,
    }