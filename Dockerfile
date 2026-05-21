FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libspatialindex-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    gymnasium \
    stable-baselines3 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn[standard] \
    python-dotenv \
    requests \
    tqdm

# Copiar código
COPY . .

EXPOSE 8000

CMD ["uvicorn", "viz.app:app", "--host", "0.0.0.0", "--port", "8000"]
