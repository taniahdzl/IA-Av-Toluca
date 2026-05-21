FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar torch CPU primero desde su propio índice
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto desde PyPI normal
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    gymnasium \
    stable-baselines3 \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    requests \
    tqdm

COPY . .

EXPOSE 8000

CMD ["uvicorn", "viz.app:app", "--host", "0.0.0.0", "--port", "8000"]
