# Arquitectura del Sistema

El proyecto implementa una arquitectura MLOps completa utilizando Docker Compose. Los componentes principales son:

## Diagrama de Componentes

```mermaid
graph TD
    Client[Cliente/Usuario] -->|HTTP Request| API[Prediction API (FastAPI)]
    API -->|Load Model| MLflow[MLflow Server]
    API -->|Fetch Artifacts| MinIO[MinIO (S3)]

    Airflow[Airflow Orchestrator] -->|Trigger| DAG1[Data Treatment DAG]
    Airflow -->|Trigger| DAG2[Train Pipeline DAG]

    DAG1 -->|Read| RawData[Raw Data (CSV)]
    DAG1 -->|Write| ProcessedData[Processed Data (Parquet)]
    DAG1 -->|Save to| MinIO

    DAG2 -->|Read| ProcessedData
    DAG2 -->|Train & Log| MLflow
    DAG2 -->|Save Artifacts| MinIO

    MLflow -->|Metadata| DB[(PostgreSQL)]
    MLflow -->|Artifacts| MinIO
    Airflow -->|Metadata| DB
```

## Servicios

| Servicio              | Puerto    | Descripción                                                        |
| --------------------- | --------- | ------------------------------------------------------------------ |
| **Airflow Webserver** | 8080      | Orquestador de pipelines. Interfaz gráfica para monitorear DAGs.   |
| **MLflow Server**     | 5000      | Registro de experimentos y modelos.                                |
| **Prediction API**    | 8000      | API REST para servir predicciones del modelo en producción.        |
| **MinIO**             | 9000/9001 | Almacenamiento de objetos (S3 compatible) para artefactos y datos. |
| **PostgreSQL**        | 5432      | Base de datos relacional compartida para Airflow y MLflow.         |

## Flujo de Datos

1.  **Ingesta**: Los datos crudos (`listings_big.csv`) son procesados por el DAG de tratamiento de datos.
2.  **Procesamiento**: Se limpia la data, se generan features y se divide en Train/Test. Los resultados se guardan en `.parquet` y en MinIO.
3.  **Entrenamiento**: El DAG de entrenamiento lee los parquets, entrena múltiples modelos (LogReg, GradientBoosting, etc.) y registra métricas en MLflow.
4.  **Promoción**: El mejor modelo se registra en MLflow.
5.  **Inferencia**: La API carga el modelo desde MLflow y responde a peticiones de predicción.
