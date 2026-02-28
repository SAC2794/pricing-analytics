# models/forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Demand Forecasting – Random Forest on time-series features
#
# Input table : dw.metrics_forecast_features
# Output table: dw.results_forecast
#
# Predicts `units_sold` N steps ahead for each (product_id, store_id) pair.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from models.db_utils import read_table, write_results
from config import METRICS_FORECAST_TABLE

TARGET   = "units_sold"
LAG_COLS = ["units_sold_lag1", "units_sold_lag7", "units_sold_lag14", "units_sold_lag30"]
ROLL_COLS= [c for c in [
    "units_sold_roll_mean_7", "units_sold_roll_mean_14", "units_sold_roll_mean_30",
    "units_sold_roll_std_7",  "units_sold_roll_std_14",  "units_sold_roll_std_30",
]]
CALENDAR = ["year", "month", "week", "day_of_week", "is_weekend", "quarter"]
CONTEXT  = ["price", "discount", "inventory_level", "seasonality_factor", "holiday_promotion"]


def run_forecasting(n_estimators: int = 200, n_splits: int = 3):
    print("\n📈 Running Demand Forecasting (RF time-series)...")

    df = read_table(METRICS_FORECAST_TABLE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["product_id", "store_id", "date"])

    feature_cols = [
        c for c in LAG_COLS + ROLL_COLS + CALENDAR + CONTEXT
        if c in df.columns
    ]

    df_model = df[feature_cols + [TARGET, "date", "product_id", "store_id"]].dropna()

    if len(df_model) < 100:
        print("  ⚠ Insufficient data for forecasting. Aborting.")
        return

    X = df_model[feature_cols]
    y = df_model[TARGET]

    # Time-series cross-validation (never shuffle)
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, y_hat))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_hat)))
        print(f"  Fold {fold+1}: MAE={maes[-1]:.2f}  RMSE={rmses[-1]:.2f}")

    print(f"  Avg MAE={np.mean(maes):.2f}  Avg RMSE={np.mean(rmses):.2f}")

    # Final predictions on full dataset
    model.fit(X, y)
    df_model = df_model.copy()
    df_model["predicted_units_sold"] = model.predict(X)
    df_model["residual"]             = df_model[TARGET] - df_model["predicted_units_sold"]

    write_results(df_model[["date","product_id","store_id",
                             TARGET,"predicted_units_sold","residual"]],
                  "results_forecast")

    # Feature importances
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    write_results(fi_df, "results_forecast_feature_importance")

    print("  ✅ Forecasting done. Results saved to Azure SQL.")
    return model


if __name__ == "__main__":
    run_forecasting()