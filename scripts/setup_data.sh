#!/usr/bin/env bash
# setup_data.sh
# Prepara el entorno de datos del proyecto.
# Uso: bash scripts/setup_data.sh

set -e
cd "$(dirname "$0")/.."

echo "── Toluca × Periférico RL ── Setup de datos ──────────────────"

# 1. Verificar .env
if [ ! -f .env ]; then
  echo "⚠  No se encontró .env — copiando desde .env.example"
  cp .env.example .env
  echo "   Edita .env y agrega tu GOOGLE_ROUTES_API_KEY si quieres datos externos."
fi

# 2. Instalar dependencias
echo ""
echo "── Instalando dependencias Python..."
pip install -r requirements.txt -q

# 3. Descargar geometría de OSM
echo ""
echo "── Descargando geometría del cruce desde OpenStreetMap..."
python data_collection/osm_downloader.py

# 4. Instrucciones finales
echo ""
echo "✓ Setup completado."
echo ""
echo "Próximos pasos:"
echo "  1. Completa data/processed/geometria_carriles.json con datos de campo (sección B)"
echo "  2. Completa data/processed/matriz_markov.json con conteos de giros (sección C.2)"
echo "  3. Completa data/processed/flujos_calibrados.json con conteos de flujo (sección C.1, D, E)"
echo "  4. Corre el notebook de calibración: notebooks/01_calibracion.ipynb"
