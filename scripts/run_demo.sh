#!/usr/bin/env bash
# run_demo.sh
# Levanta el demo visual en Docker.
# Uso: bash scripts/run_demo.sh

set -e
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "✗  Falta el archivo .env. Corre primero: bash scripts/setup_data.sh"
  exit 1
fi

echo "── Construyendo imagen Docker..."
docker compose build

echo ""
echo "── Levantando demo..."
docker compose up

# La URL se muestra en la terminal cuando FastAPI arranca.
