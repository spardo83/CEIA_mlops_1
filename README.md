# MLOps End-to-End: Airbnb Occupancy Prediction

Este repositorio contiene un proyecto completo de MLOps para predecir el nivel de ocupación de propiedades de Airbnb. El sistema abarca desde la ingesta de datos y entrenamiento de modelos hasta el despliegue de una API de predicción, todo orquestado y contenerizado.

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker y Docker Compose instalados.
- (Opcional) Dataset `listings.csv` si se desea ejecutar localmente sin S3.

### Ejecución

1.  **Levantar servicios**:

    ```bash
    docker-compose up -d --build
    ```

    Esto iniciará Airflow, MLflow, MinIO, Postgres y la API de Predicción.

2.  **Acceder a las interfaces**:

    - **Airflow**: [http://localhost:8080](http://localhost:8080) (user: `airflow`, pass: `airflow`)
    - **MLflow**: [http://localhost:5000](http://localhost:5000)
    - **API Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
    - **MinIO**: [http://localhost:9001](http://localhost:9001) (user: `minio`, pass: `minio123`)

3.  **Ejecutar el Pipeline**:
    - Ir a Airflow y activar el DAG `data_treatment_dag`.
    - Esto disparará automáticamente el `train_pipeline_dag` al finalizar.

## 📚 Documentación

Para detalles profundos sobre cada componente:

- **[Arquitectura del Sistema](docs/architecture.md)**: Visión general de componentes y flujo de datos.
- **[Pipelines de Datos y Entrenamiento](docs/pipelines.md)**: Explicación detallada de los DAGs de Airflow.
- **[API de Predicción](docs/api.md)**: Guía de uso de la API REST y esquema de datos.

## 💡 Contexto del Proyecto

El análisis exploratorio de datos (EDA) y la definición del problema se encuentran en el notebook original:

- 📓 [tp.ipynb](tp.ipynb)

## Estructura del Proyecto

- `airflow/`: DAGs y configuración de Airflow.
- `api/`: Código fuente de la API de predicción (FastAPI).
- `mlflow/`: Dockerfile y configuración del servidor MLflow.
- `docs/`: Documentación detallada.
- `docker-compose.yml`: Definición de servicios.
