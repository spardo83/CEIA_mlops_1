# Trabajo Práctico Final – MLOps

## Objetivo de la materia
El objetivo principal es **disponibilizar las herramientas de Machine Learning en un entorno productivo**, aplicando buenas prácticas de **MLOps**.

Se busca que los estudiantes puedan desarrollar e integrar un flujo completo que cubra desde el desarrollo y versionado de modelos, hasta su despliegue, monitoreo y mantenimiento en un entorno controlado.

---

## Contexto
Supondremos que trabajamos para **ML Models and Something More Inc.**, una empresa que ofrece modelos de Machine Learning mediante una **REST API**.  
Internamente, la compañía cuenta con:

- **Apache Airflow** para la orquestación de tareas (DataOps / MLOps)  
- **MLflow** para el ciclo de vida de los modelos  
- **Data Lake en S3 (MinIO)** como repositorio de datos  

---

## Niveles de implementación

### Nivel Local (nota entre 6 y 8)
Implementar en entorno local un ciclo de desarrollo del modelo creado en la materia *Aprendizaje de Máquina*, hasta la generación final del artefacto entrenado.  
Requisitos:
- Uso de un **orquestador** (por ejemplo, Airflow, Prefect, Luigi, etc.)
- Buenas prácticas de desarrollo y **documentación clara**

---

### Nivel en Contenedores (nota entre 8 y 10)
Implementar el modelo en un entorno **productivo basado en contenedores**.  
Se deben usar los siguientes recursos (mínimos) disponibles:

- **Apache Airflow**
- **MLflow**
- **PostgreSQL**
- **MinIO (S3 compatible)**
- **FastAPI**

Todo se encuentra montado en **Docker**, por lo que es requisito tenerlo instalado.

---

## Repositorio base y ejemplo
El material base se encuentra en el siguiente repositorio:

🔗 [https://github.com/facundolucianna/amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml)

Además, se incluye un ejemplo de aplicación en el branch:  
`example_implementation`

---

## Criterios de aprobación

- El trabajo se entrega en **dos partes**:
  1. **Primera entrega:** en la clase 5  
  2. **Entrega final:** tres (3) días después de la última clase  
- La entrega consiste en un **repositorio en GitHub o GitLab** con la implementación y documentación.  
- El trabajo debe ser **grupal** (mínimo 2, máximo 6 integrantes).  
- La implementación debe corresponder al **nivel elegido** (Local o Contenedores).  
- Se valora especialmente la **documentación clara y completa**.  
- Es posible incorporar o cambiar tecnologías, siempre que se mantenga:
  - Un servicio de **orquestación**
  - Un servicio de **ciclo de vida de modelos**
- La entrega se realiza por el aula virtual de la asignatura, enviando el **link al repositorio**.

---

## Recordatorio
El objetivo no es solo que el modelo funcione, sino demostrar una integración sólida entre las piezas del ecosistema MLOps, evidenciando:

- Versionado y trazabilidad del modelo  
- Orquestación reproducible  
- Pipeline automatizado  
- Entorno de ejecución aislado (Docker)  
- Documentación técnica y de despliegue

---
