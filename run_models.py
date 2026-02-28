# run_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs all ML models in sequence.
# Designed to be called AFTER main.py has loaded the metrics tables to SQL.
#
# Order:
#   1. Linear Regression   (lightest, no lag features needed)
#   2. Random Forest       (pricing prediction)
#   3. Segmentation        (KMeans on RFM)
#   4. Forecasting         (RF time-series on lag features)
# ─────────────────────────────────────────────────────────────────────────────

from models.linear_regression import run_linear_regression
from models.random_forest     import run_random_forest
from models.segmentation      import run_segmentation
from models.forecasting       import run_forecasting


def run_all_models():
    print("\n" + "═"*60)
    print("  ML MODELS PIPELINE")
    print("═"*60)

    try:
        run_linear_regression()
    except Exception as e:
        print(f"  ❌ Linear Regression failed: {e}")

    try:
        run_random_forest()
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")

    try:
        run_segmentation()
    except Exception as e:
        print(f"  ❌ Segmentation failed: {e}")

    try:
        run_forecasting()
    except Exception as e:
        print(f"  ❌ Forecasting failed: {e}")

    print("\n" + "═"*60)
    print("  ✅ All models completed.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    run_all_models()