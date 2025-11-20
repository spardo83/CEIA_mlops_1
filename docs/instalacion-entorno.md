# Preparcion del entorno de desarrollo

## Gestión de dependencias con `uv`

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) como gestor de dependencias y entorno virtual.

---

### Instalación de `uv`

Para instalar `uv` globalmente en tu sistema:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

O con Homebrew (macOS):

```
brew install uv
```

Una vez instalado, verificá la versión:

```
uv --version
```
---

### Inicialización del entorno

En la raíz del proyecto:

```
uv sync
```

Este comando:
- Crea el entorno virtual automáticamente.
- Instala todas las dependencias definidas en `pyproject.toml`.
- Genera un lockfile (uv.lock) para garantizar reproducibilidad.

Para activar el entorno manualmente:

```
source .venv/bin/activate
```
---

### Ejecutar scripts dentro del entorno

 TODO: A definir

---

### Generar requirements.txt para Docker

Aunque `uv` maneja las dependencias internamente, para construir imágenes Docker más simples se puede exportar un requirements.txt plano:

```
uv export --format requirements.txt > requirements.txt
```

Este archivo se genera automáticamente a partir del pyproject.toml. Se regenera cada vez que cambian las dependencias.

Luego, los Dockerfile pueden instalar dependencias normalmente:

```
RUN pip install --no-cache-dir -r requirements.txt
```
---