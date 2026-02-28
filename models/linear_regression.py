# models/linear_regression.py
# ─────────────────────────────────────────────────────────────────────────────
# Linear Regression – Price Elasticity & Gross Margin Analysis
#
# Input table : dw.metrics_pricing
# Output table: dw.results_linear_regression
#
# Model 1: revenue ~ price + discount + competitor_price + seasonality + ...
# Model 2: gross_margin_pct ~ price_gap_pct + discount + category + ...
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from models.db_utils import read_table, write_results
from config import METRICS_PRICING_TABLE


def _add_calendar(df):
    if "date" in df.columns and "month" not in df.columns:
        df["date"]        = pd.to_datetime(df["date"], errors="coerce")
        df["month"]       = df["date"].dt.month
        df["quarter"]     = df["date"].dt.quarter
        df["is_weekend"]  = df["date"].dt.dayofweek.isin([5,6]).astype(int)
    return df


def _encode_categoricals(df):
    if "category" in df.columns:
        df = df.copy()
        df["category_enc"] = LabelEncoder().fit_transform(df["category"].astype(str))
    return df


# ── Model 1: Revenue prediction ───────────────────────────────────────────────
REVENUE_FEATURES = [
    "price", "discount", "net_price",
    "competitor_price", "price_gap_vs_competitor",
    "seasonality_factor", "holiday_promotion",
    "month", "quarter", "is_weekend",
    "category_enc",
]


def run_revenue_regression(df: pd.DataFrame):
    avail = [f for f in REVENUE_FEATURES if f in df.columns]
    data  = df[avail + ["revenue"]].dropna()
    if len(data) < 30:
        return None, pd.DataFrame()

    X, y = data[avail], data["revenue"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"  [Revenue Regression] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    results = X_te.copy()
    results["actual_revenue"]    = y_te.values
    results["predicted_revenue"] = y_pred
    results["model"]             = "revenue_regression"

    coef_df = pd.DataFrame({
        "feature":     avail,
        "coefficient": pipe.named_steps["model"].coef_,
        "model":       "revenue_regression",
    }).sort_values("coefficient", key=abs, ascending=False)

    return pipe, results, coef_df


# ── Model 2: Gross margin regression ─────────────────────────────────────────
MARGIN_FEATURES = [
    "price_gap_pct", "discount", "net_price",
    "seasonality_factor", "holiday_promotion",
    "month", "category_enc",
]


def run_margin_regression(df: pd.DataFrame):
    avail = [f for f in MARGIN_FEATURES if f in df.columns]
    data  = df[avail + ["gross_margin_pct"]].dropna()
    if len(data) < 30:
        return None, pd.DataFrame(), pd.DataFrame()

    X, y = data[avail], data["gross_margin_pct"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"  [Margin Regression]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    results = X_te.copy()
    results["actual_margin_pct"]    = y_te.values
    results["predicted_margin_pct"] = y_pred
    results["model"]                = "margin_regression"

    coef_df = pd.DataFrame({
        "feature":     avail,
        "coefficient": pipe.named_steps["model"].coef_,
        "model":       "margin_regression",
    }).sort_values("coefficient", key=abs, ascending=False)

    return pipe, results, coef_df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_linear_regression():
    print("\n📊 Running Linear Regression models...")

    df = read_table(METRICS_PRICING_TABLE)
    df = _add_calendar(df)
    df = _encode_categoricals(df)

    all_results, all_coefs = [], []

    # Model 1 – Revenue
    _, res1, coef1 = run_revenue_regression(df)
    if res1 is not None and not res1.empty:
        all_results.append(res1)
        all_coefs.append(coef1)

    # Model 2 – Gross margin
    _, res2, coef2 = run_margin_regression(df)
    if res2 is not None and not res2.empty:
        all_results.append(res2)
        all_coefs.append(coef2)

    if all_results:
        write_results(pd.concat(all_results, ignore_index=True), "results_linear_regression")
    if all_coefs:
        write_results(pd.concat(all_coefs, ignore_index=True),   "results_lr_coefficients")

    print("  ✅ Linear Regression done. Results saved to Azure SQL.")


if __name__ == "__main__":
    run_linear_regression()