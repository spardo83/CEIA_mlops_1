# Prediction API

La API de predicción expone el modelo de Machine Learning entrenado para realizar inferencias en tiempo real. Está construida con **FastAPI**.

## Acceso

- **URL Base**: `http://localhost:8000`
- **Documentación Interactiva (Swagger UI)**: `http://localhost:8000/docs`

## Endpoints

### `POST /predict`

Recibe los datos de un listado y devuelve el nivel de ocupación predicho.

**Request Body (Ejemplo)**:

```json
{
  "data": [
    {
      "host_since": "2020-01-01",
      "neighbourhood_cleansed": "Palermo",
      "latitude": -34.58,
      "longitude": -58.42,
      "room_type": "Entire home/apt",
      "price": "$50.00",
      "minimum_nights": 2,
      "amenities": "[\"Wifi\", \"Kitchen\"]",
      ...
    }
  ]
}
```

_Nota: Se recomienda usar Swagger UI para ver el esquema completo con todos los campos requeridos._

**Response**:

```json
{
  "predictions": ["mid"]
}
```

### `GET /health`

Verifica el estado del servicio y si el modelo está cargado.

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Nota sobre el Modelo

La API carga automáticamente la versión del modelo etiquetada como `Production` en MLflow (`airbnb-occupancy-classifier`). Si no hay ningún modelo en producción, la API iniciará pero fallará al intentar predecir. Asegúrese de ejecutar el pipeline de entrenamiento al menos una vez exitosamente.
