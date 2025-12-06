# Pipelines de Airflow

El sistema cuenta con dos DAGs principales orquestados por Airflow.

## 1. Data Treatment DAG (`data_treatment_dag`)

**Objetivo**: Preprocesar los datos crudos de Airbnb y prepararlos para el entrenamiento.

**Pasos Clave**:

1.  **Carga**: Lee `listings_big.csv` (local o S3).
2.  **Limpieza**:
    - Elimina columnas con >95% de nulos.
    - Normaliza valores faltantes.
    - Convierte fechas y tipos de datos.
3.  **Feature Engineering**:
    - `days_since_first_review`, `days_since_last_review`.
    - Imputación de precios (`price`, `price_per_person`).
    - Imputación de bathrooms/bedrooms basado en la moda por tipo de habitación.
    - Expansión de `amenities` a columnas binarias.
4.  **Split**: División Train/Test (80/20) estratificada o por `host_id`.
5.  **Persistencia**: Guarda `train.parquet` y `test.parquet` en `data/processed` y en MinIO.
6.  **Trigger**: Al finalizar, dispara automáticamente el `train_pipeline_dag`.

## 2. Train Pipeline DAG (`train_pipeline_dag`)

**Objetivo**: Entrenar múltiples modelos de clasificación para predecir el `occupancy_level`.

**Pasos Clave**:

1.  **Ensure Data**: Verifica que existan los archivos parquet.
2.  **Prepare Features**:
    - Carga `train.parquet` y `test.parquet`.
    - Aplica `ColumnTransformer`:
      - Numéricas: Imputación (mediana) + Scaling (StandardScaler).
      - Categóricas: Imputación + OneHotEncoder.
    - Guarda matrices dispersas (`X_train.npz`, etc.).
3.  **Train Models (Paralelo)**:
    - Entrena distintos modelos: Logistic Regression, Gradient Boosting, MLP, PCA+LogReg, Ensemble.
    - Registra parámetros y métricas (Accuracy, F1, MAE) en MLflow.
    - Guarda el modelo entrenado como artefacto en MLflow.
    - El modelo `train_simplenn` usa PyTorch con Early Stopping.
4.  **Select Best Model**:
    - Compara los modelos entrenados en el run actual basándose en `f1_macro`.
    - Registra el mejor modelo en el Model Registry de MLflow bajo el nombre `airbnb-occupancy-classifier`.
    - Etiqueta la versión como `Production`.

## Ejecución

Para disparar el pipeline completo, basta con activar el `data_treatment_dag` desde la UI de Airflow.
