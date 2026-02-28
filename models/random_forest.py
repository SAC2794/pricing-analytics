# models/random_forest.py
# ─────────────────────────────────────────────────────────────────────────────
# Random Forest – Price / Revenue prediction
#
# Input table : dw.metrics_pricing
# Output table: dw.results_random_forest
#
# What it does
# ────────────
# Trains a Random Forest Regressor to predict `revenue` from pricing and
# contextual features.  Stores predictions + feature importances back to SQL.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from models.db_utils import read_table, write_results
from config import METRICS_PRICING_TABLE


TARGET   = "revenue"
FEATURES = [
    "price", "discount", "net_price",
    "competitor_price", "price_gap_vs_competitor", "price_gap_pct",
    "gross_margin", "gross_margin_pct",
    "units_sold",
    "seasonality_factor", "holiday_promotion",
    "forecast_accuracy",
    "month", "day_of_week", "is_weekend",        # calendar – added below if missing
]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and "month" not in df.columns:
        df["date"]        = pd.to_datetime(df["date"], errors="coerce")
        df["month"]       = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def run_random_forest(n_estimators: int = 200, random_state: int = 42):
    print("\n🌲 Running Random Forest (Price/Revenue Prediction)...")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = read_table(METRICS_PRICING_TABLE)
    df = _add_calendar_features(df)

    # Encode categoricals
    if "category" in df.columns:
        df["category_enc"] = LabelEncoder().fit_transform(df["category"].astype(str))
        FEATURES.append("category_enc")

    # Keep only available features and target
    available = [f for f in FEATURES if f in df.columns] + [TARGET]
    df = df[available].dropna()

    if len(df) < 50:
        print("  ⚠ Not enough data to train. Aborting.")
        return

    X = df[[f for f in FEATURES if f in df.columns]]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    # ── Predictions DataFrame ─────────────────────────────────────────────────
    results_df = X_test.copy()
    results_df["actual_revenue"]    = y_test.values
    results_df["predicted_revenue"] = y_pred
    results_df["residual"]          = y_test.values - y_pred

    write_results(results_df, "results_random_forest")

    # ── Feature importances ───────────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    write_results(fi_df, "results_rf_feature_importance")
    print("  ✅ Random Forest done. Results saved to Azure SQL.")
    return model, fi_df


if __name__ == "__main__":
    run_random_forest()