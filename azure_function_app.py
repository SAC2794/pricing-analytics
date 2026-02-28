# azure_function_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Azure Functions entry point.
# Two functions are exposed:
#
#   1. run_etl_pipeline   – HTTP trigger  →  runs main.py (ETL + metrics)
#   2. run_ml_models      – HTTP trigger  →  runs run_models.py (ML)
#
# Both are callable independently, so you can re-run only the models
# without re-running the full ETL.
#
# Deployment note:
#   az functionapp create ... --runtime python --runtime-version 3.11
#   Assign the Function App a System-assigned Managed Identity and grant it:
#     - "Storage Blob Data Reader"  on pricingpipelineexec storage account
#     - "db_datareader / db_datawriter" on pricing_analytics_db
# ─────────────────────────────────────────────────────────────────────────────

import azure.functions as func
import logging

from main       import run_pipeline
from run_models import run_all_models

app = func.FunctionApp()


@app.route(route="run_etl_pipeline", methods=["POST", "GET"])
def run_etl_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("ETL pipeline triggered via HTTP.")
    try:
        run_pipeline("retail_store_inventory.csv")
        return func.HttpResponse("ETL pipeline completed successfully.", status_code=200)
    except Exception as exc:
        logging.error(f"ETL pipeline failed: {exc}")
        return func.HttpResponse(f"Pipeline failed: {exc}", status_code=500)


@app.route(route="run_ml_models", methods=["POST", "GET"])
def run_ml_models_func(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("ML models triggered via HTTP.")
    try:
        run_all_models()
        return func.HttpResponse("ML models completed successfully.", status_code=200)
    except Exception as exc:
        logging.error(f"ML models failed: {exc}")
        return func.HttpResponse(f"ML models failed: {exc}", status_code=500)