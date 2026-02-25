# Pricing Analytics Pipeline (Azure Cloud)

End-to-end cloud-based pricing analytics project built with Python and Azure services.

## Architecture

Azure Blob Storage (Raw Data)
↓
Azure Function (Python ETL)
↓
Azure SQL (Star Schema Model)
↓
Power BI (Executive Dashboard)
↓
Streamlit (Interactive ML Simulation)

## Tech Stack

- Python (Pandas)
- Azure Blob Storage
- Azure SQL Database
- Azure Functions
- Power BI
- Streamlit

## Project Structure

- etl/: Extract, Transform, Load logic
- config.py: Centralized configuration
- main.py: ETL orchestration entry point
- requirements.txt: Project dependencies