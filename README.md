# Optimización de Semáforos con RL — Av. Toluca × Anillo Periférico

Proyecto de IA para la clase de Aprendizaje por Refuerzo.  
Simulación microscópica del cruce **Avenida Toluca – Anillo Periférico (bajo nivel)**,
con agentes de comportamiento nivel 3 y un agente PPO que optimiza los tiempos del semáforo.

## Demo rápido

```bash
docker compose up
# Abre http://localhost:8000
```

## Estructura

```
toluca-periferico-rl/
├── data/
│   ├── raw/            # Datos crudos: OSM, Google Routes, observación de campo
│   ├── processed/      # JSONs calibrados que lee el simulador
│   └── validation/     # Series temporales para validar el simulador
├── sim/                # Simulador del cruce (independiente de RL)
├── rl/                 # Entorno gym + agente PPO
├── data_collection/    # Scripts de descarga OSM y Google Routes API
├── viz/                # Demo visual (FastAPI + frontend)
├── notebooks/          # Calibración, baseline y análisis de resultados
├── tests/              # Unit tests por módulo
└── scripts/            # setup_data.sh, run_demo.sh
```

## Módulos principales

| Módulo | Responsabilidad |
|--------|-----------------|
| `sim/geometry.py` | Carriles, longitudes, capacidad física |
| `sim/vehicle.py` | Vehículos con perfiles de conductor (agresivo/normal/cauteloso) |
| `sim/router.py` | Cadena de Markov para asignación de destinos |
| `sim/traffic_light.py` | Semáforo con fases y tiempos ajustables |
| `sim/intersection.py` | Motor principal: step(), colas, descarga |
| `sim/metrics.py` | Recolección de métricas y gráficas |
| `rl/environment.py` | Entorno gymnasium que envuelve al simulador |
| `rl/reward.py` | Funciones de recompensa intercambiables |
| `rl/train.py` | Entrenamiento del agente PPO |
| `viz/app.py` | Servidor FastAPI para el demo |

## Requisitos

- Python 3.11+
- Docker + Docker Compose
- (Opcional) Google Routes API key para recolección de datos

## Instalación local (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Variables de entorno

Copiar `.env.example` a `.env` y llenar los valores necesarios.

## Datos de campo

Los datos de calibración provienen de observación directa en el cruce
(ver `data/raw/observacion_campo.docx`) complementados con geometría de OpenStreetMap.
