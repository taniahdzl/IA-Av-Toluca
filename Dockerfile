FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (osmnx necesita algunas librerías geoespaciales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (mejor uso de caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Puerto del demo visual
EXPOSE 8000

# Comando por defecto: levantar el demo
CMD ["uvicorn", "viz.app:app", "--host", "0.0.0.0", "--port", "8000"]
