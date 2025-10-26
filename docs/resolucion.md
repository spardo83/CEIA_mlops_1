# Trabajo Práctico Final – MLOps - Implementacion

## Ejecución del entorno en contenedores

Este proyecto incluye un entorno productivo simulado basado en contenedores Docker.  
Levantamos los siguientes servicios principales:

- **Airflow**: orquestación de pipelines de DataOps/MLOps (ingesta → limpieza → entrenamiento → evaluación).
- **MLflow**: tracking de experimentos, registro de modelos y artefactos.
- **PostgreSQL**: backend de metadatos.
- **MinIO (S3 compatible)**: Data Lake y storage de artefactos del modelo entrenado.
- **FastAPI**: servicio REST que expone el modelo entrenado para inferencia (`/predict`).

### Prerrequisitos

- Docker instalado
- Docker Compose habilitado
- Python 3.10+ para probar la API localmente

### Cómo levantar todo

```bash
docker compose up --build
```
### Servicios del entorno

Esto levanta los siguientes servicios en contenedores:

|Servicio       | Descripción                                                                                                       | URL local                        |
|---------------|-------------------------------------------------------------------------------------------------------------------|----------------------------------|
| **Airflow**   | Orquestación de pipelines de DataOps/MLOps (ingesta → limpieza → entrenamiento → evaluación)                      | http://localhost:8080           |
| **MLflow**    | Tracking de experimentos, registro de modelos y artefactos                                                        | http://localhost:5000           |
| **MinIO**     | Data Lake y almacenamiento de artefactos (compatible S3)                                                         | http://localhost:9001           |
| **FastAPI**   | Servicio REST de inferencia del modelo (`/predict`) con documentación interactiva (Swagger)                      | http://localhost:8000/docs      |
| **PostgreSQL** | Base de datos de metadatos (por ejemplo Airflow metadata DB y/o MLflow backend store)                            | *(no tiene interfaz web)*       |

> ⚙️ Las credenciales iniciales y variables sensibles se definen en `.env` y/o en los `environment:` de cada servicio dentro de `docker-compose.yml`.

---

## Flujo de trabajo

1. **Orquestación (Airflow)**  
   - Existe un DAG que corre las etapas:
     - extracción / carga de datos en MinIO  
     - preprocesamiento  
     - entrenamiento del modelo  
     - registro del modelo y métricas en MLflow  
     - despliegue / actualización del modelo servido por FastAPI  

2. **Tracking (MLflow)**  
   - Cada corrida de entrenamiento registra:
     - parámetros (`learning_rate`, `max_depth`, etc.)  
     - métricas (`accuracy`, etc.)  
     - artefactos (modelo serializado y assets auxiliares)  

3. **Serving (FastAPI)**  
   - El contenedor de la API carga el modelo más reciente 
   - Expone un endpoint `/predict` que recibe un JSON de entrada y devuelve la predicción.  
   - Incluye documentación Swagger en `/docs` para probar la inferencia manualmente.

4. **Datos (MinIO / S3)**  
   - Los datos crudos y procesados se almacenan en buckets de MinIO simulando un Data Lake.  
   - Los artefactos finales del modelo también seran persistidos allí para ser consumidos con posterioridad de ser necesario.