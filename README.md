# Pricing Pipeline — End-to-End Cloud Analytics

Pipeline ETL y modelos de Machine Learning 100% en la nube para análisis de precios e inventario retail. Sin archivos `.env`, sin secrets locales, sin dependencia de ningún PC específico. Funciona en cualquier máquina con acceso a Azure.

---

## Arquitectura general

```
Azure Blob Storage (raw)
        │
        │  retail_store_inventory.csv
        ▼
   etl/extract.py
        │
        ▼
   etl/transform.py
   ├── Limpieza y tipos
   ├── Métricas derivadas
   ├── Tablas de dimensiones
   ├── Tabla de hechos
   └── Tablas de métricas (ML)
        │
        ▼
   etl/load.py
        │
        ▼
   Azure SQL Database (schema: dw)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
   Power BI                           Streamlit (GitHub)
   dim_* + fact_sales                 metrics_* + results_*
```

---

## Estructura del proyecto

```
tu-repositorio/
├── config.py                  # Configuración central (storage, SQL, tabla names)
├── main.py                    # Orquestador del pipeline ETL
├── run_models.py              # Orquestador de los modelos ML
├── azure_function_app.py      # HTTP triggers para Azure Functions
├── requirements.txt           # Dependencias del proyecto
├── .gitignore
│
├── etl/
│   ├── __init__.py
│   ├── extract.py             # Lee el CSV desde Azure Blob Storage
│   ├── transform.py           # Limpieza, dimensiones, hechos y métricas
│   └── load.py                # Escribe DataFrames a Azure SQL
│
└── models/
    ├── __init__.py
    ├── db_utils.py            # Helper compartido: leer/escribir Azure SQL
    ├── random_forest.py       # Predicción de revenue
    ├── linear_regression.py   # Elasticidad de precio y margen bruto
    ├── segmentation.py        # Segmentación RFM con KMeans
    └── forecasting.py         # Forecasting de demanda con lag features
```

---

## Tablas generadas en Azure SQL

Todas las tablas viven en el schema `dw` de `pricing_analytics_db`.

### Para Power BI

| Tabla | Tipo | Descripción |
|---|---|---|
| `dim_date` | Dimensión | Calendario completo con año, mes, semana, día |
| `dim_product` | Dimensión | Productos, categorías, proveedor, costo |
| `dim_store` | Dimensión | Tiendas, ubicación, región |
| `dim_external` | Dimensión | Factores externos por fecha (clima, temporada, promoción) |
| `fact_sales` | Hecho | Ventas, inventario, precios, márgenes por producto × tienda × fecha |

### Para Streamlit y modelos ML (input)

| Tabla | Descripción |
|---|---|
| `metrics_pricing` | Features de precio, márgenes, gap vs competidor, elasticidad proxy |
| `metrics_inventory` | KPIs de inventario, fill rate, overstock, días de cobertura |
| `metrics_customer_segmentation` | RFM agregado por producto × tienda |
| `metrics_forecast_features` | Lag features y rolling stats para forecasting |

### Resultados de modelos ML (output)

| Tabla | Modelo |
|---|---|
| `results_random_forest` | Predicciones de revenue |
| `results_rf_feature_importance` | Importancia de variables (RF) |
| `results_linear_regression` | Predicciones de revenue y margen |
| `results_lr_coefficients` | Coeficientes de regresión |
| `results_segmentation` | Etiquetas de cluster por producto × tienda |
| `results_segmentation_profile` | Perfil promedio de cada cluster |
| `results_forecast` | Predicciones de demanda vs real |
| `results_forecast_feature_importance` | Importancia de variables (Forecast) |

---

## Modelos de Machine Learning

### Random Forest — Predicción de revenue
- **Input:** `metrics_pricing`
- **Target:** `revenue`
- **Features:** precio, descuento, gap vs competidor, márgenes, estacionalidad, calendario
- **Output:** predicciones + importancia de variables → `results_random_forest`

### Regresión Lineal — Elasticidad de precio
- **Input:** `metrics_pricing`
- **Modelo 1:** revenue ~ precio + descuento + contexto (Ridge Regression)
- **Modelo 2:** gross_margin_pct ~ price_gap + descuento + categoría
- **Output:** predicciones + coeficientes → `results_linear_regression`

### Segmentación — KMeans sobre RFM
- **Input:** `metrics_customer_segmentation`
- **Features:** recency, frequency, monetary, avg_units_sold
- **Selección de k:** silhouette score automático
- **Segmentos:** Champions / Loyal / Potential / At Risk
- **Output:** etiquetas + perfil de clusters → `results_segmentation`

### Forecasting — Demanda futura
- **Input:** `metrics_forecast_features`
- **Target:** `units_sold`
- **Features:** lags (1, 7, 14, 30 días), rolling mean/std, calendario, precio
- **Validación:** TimeSeriesSplit (nunca mezcla futuro con pasado)
- **Output:** predicciones por producto × tienda × fecha → `results_forecast`

---

## Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Correr el pipeline ETL

```bash
python main.py
```

Esto ejecuta en secuencia: extracción → limpieza → transformación → carga de las 9 tablas base a Azure SQL.

### 3. Correr los modelos ML

```bash
python run_models.py
```

Esto lee las tablas de métricas desde Azure SQL, entrena los 4 modelos y escribe los resultados de vuelta a SQL.

### 4. En Azure Functions (producción)

Dos endpoints HTTP disponibles:

```
POST https://<function-app>.azurewebsites.net/api/run_etl_pipeline
POST https://<function-app>.azurewebsites.net/api/run_ml_models
```

---

## Autenticación

El proyecto usa **Azure Managed Identity** (`DefaultAzureCredential`) — sin passwords, sin `.env`, sin connection strings.

La identidad asignada a la Azure Function necesita estos permisos:

| Recurso | Rol requerido |
|---|---|
| Storage Account `pricingpipelineexec` | Storage Blob Data Reader |
| SQL Database `pricing_analytics_db` | db_datareader + db_datawriter |

Para desarrollo local, autenticarse con:

```bash
az login
```

`DefaultAzureCredential` detecta automáticamente las credenciales del CLI de Azure.

---

## Infraestructura Azure

| Recurso | Nombre |
|---|---|
| Storage Account | `pricingpipelineexec` |
| Blob Container | `raw` |
| SQL Server | `sql-pricing-analytics.database.windows.net` |
| SQL Database | `pricing_analytics_db` |
| SQL Schema | `dw` |

---

## Dataset

**`retail_store_inventory.csv`** alojado en el contenedor `raw` del storage account `pricingpipelineexec`.

Columnas esperadas: `date`, `store_id`, `product_id`, `product_name`, `category`, `store_location`, `region`, `units_sold`, `units_ordered`, `inventory_level`, `demand_forecast`, `price`, `cost`, `discount`, `competitor_price`, `reorder_point`, `lead_time_days`, `supplier`, `storage_type`, `weather_condition`, `holiday_promotion`, `seasonality_factor`.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Cómputo | Azure Functions (Python 3.11) |
| Storage | Azure Blob Storage |
| Base de datos | Azure SQL Database |
| ETL | pandas, numpy |
| ML | scikit-learn |
| Auth | azure-identity (DefaultAzureCredential) |
| Dashboard | Power BI |
| App interactiva | Streamlit (GitHub) |