# Frontend Application

La aplicación Frontend proporciona una interfaz gráfica interactiva para consumir la API de Predicción de Ocupación de Airbnb. Está construida con **React** y **Vite**, utilizando **Tailwind CSS** para los estilos.

## Características Principales

1.  **Formulario de Predicción**: 
    - Permite ingresar manualmente las características del alojamiento (ubicación, precio, amenidades, Host, etc.).
    - **Nota**: El modelo actual es "Cold Start", por lo que ignora variables históricas como disponibilidad o cantidad de reseñas.
    - Incluye validaciones básicas y selectores para campos categóricos.
    
2.  **Escenarios Predefinidos**:
    - Botones de carga rápida para simular diferentes perfiles de ocupación:
        - **High**: Propiedades muy demandadas con baja disponibilidad a futuro.
        - **Mid**: Propiedades con ocupación estándar.
        - **Low/Zero**: Propiedades nuevas, caras o bloqueadas.
    - Estos escenarios facilitan la prueba del modelo sin necesidad de conocer los valores exactos de los datos de entrada.

3.  **Visualización de Resultados**:
    - Muestra la categoría de ocupación predicha (High, Mid, Low, Zero).
    - Indicadores visuales de carga y errores.

## Configuración y Ejecución

El frontend se sirve típicamente a través de Nginx o un servidor de desarrollo dentro de la red de Docker.

- **URL Local**: [http://localhost:5173](http://localhost:5173) (o el puerto configurado en docker-compose).
- **Conexión API**: La aplicación espera que la API esté corriendo en `http://localhost:8000`.

## Estructura de Código

- `src/App.jsx`: Componente principal que contiene la lógica del formulario y la llamada a la API.
- `src/index.css`: Estilos globales y directivas de Tailwind.

## Uso

1. Abra la aplicación en el navegador.
2. Seleccione un escenario de ejemplo (ej. "Popular Gem" bajo "High Occupancy") o complete los campos manualmente.
3. Haga clic en **"Predict Occupancy"**.
4. El resultado aparecerá en la tarjeta flotante a la derecha.
