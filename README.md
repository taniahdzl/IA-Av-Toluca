# Optimización de Semáforos con RL — Av. Toluca × Anillo Periférico

Proyecto de IA que modela el cruce **Av. Querétaro/Toluca × Anillo Periférico (bajo nivel)**
en la Ciudad de México y entrena un agente de Aprendizaje por Refuerzo (PPO) para optimizar
los tiempos del semáforo, reduciendo colas y tiempos de espera.

## Demo rápido

```bash
docker compose up
# Abre http://localhost:8000
```

## El problema

El cruce tiene un cuello de botella estructural: **8 carriles de entrada** (Av. Querétaro 3c +
Av. Toluca norte 1c + Lateral Norte 4c) convergen en la Zona H con 5 carriles físicos, de los
cuales solo 2 pueden girar hacia Lateral Sur este.

El semáforo alterna en 3 fases:
- **Fase 1 (51s):** Verde para Av. Querétaro/Toluca — la Zona H puede vaciarse
- **Fase 2 (47s):** Verde para Lateral Norte + Lateral Sur oeste — la Zona H queda **bloqueada**
  porque Lateral Sur oeste cruza perpendicularmente
- **Amarillo (3s × 2)**

Con el reparto 51s/47s casi igual para flujos tan desbalanceados, la cola de Av. Querétaro
acumula hasta 222 vehículos y los tiempos de espera superan los 15 minutos.

## Resultados

| Métrica | Semáforo fijo | Agente RL | Mejora |
|---|---|---|---|
| Cola promedio | 104 veh | 86 veh | −17% |
| Cola máxima | 178 veh | 164 veh | −8% |
| Espera promedio | 59s | 50s | −16% |
| Tiempos | 51s / 47s | 41s / 47s | — |

El agente aprendió que reducir ligeramente el verde de Av. Querétaro/Toluca disminuye los
bloqueos en Zona H porque los vehículos llegan más espaciados, generando menos congestión
acumulada.

## Arquitectura

```
toluca-periferico-rl/
├── data/processed/          # JSONs calibrados con datos de campo
│   ├── geometria_carriles.json
│   ├── matriz_markov.json
│   └── flujos_calibrados.json
├── sim/                     # Simulador microscópico (independiente de RL)
│   ├── geometry.py          # Carriles y capacidades físicas
│   ├── vehicle.py           # Vehículos con perfiles de conductor (nivel 3)
│   ├── router.py            # Cadena de Markov para destinos
│   ├── traffic_light.py     # Semáforo con 3 fases reales
│   ├── intersection.py      # Motor principal: step(), colas, bloqueos
│   └── metrics.py           # Monitor de métricas y gráficas
├── rl/                      # Aprendizaje por Refuerzo
│   ├── environment.py       # Entorno gymnasium (observation_space dim=9)
│   ├── reward.py            # Funciones de recompensa intercambiables
│   ├── train.py             # Script de entrenamiento PPO
│   └── evaluate.py          # Comparación baseline vs agente
├── viz/                     # Demo visual
│   ├── app.py               # Servidor FastAPI
│   ├── renderer.py          # Traduce estado del simulador a JSON
│   └── static/index.html    # Frontend con canvas del cruce
├── notebooks/
│   ├── 01_calibracion.ipynb # Llenar JSONs con datos de campo
│   ├── 02_baseline.ipynb    # Análisis del semáforo fijo
│   └── 03_resultados_rl.ipynb # Comparación baseline vs RL
├── scripts/
│   ├── entrenar.sh          # Entrenamiento PPO (continúa desde modelo existente)
│   └── run_demo.sh          # Levanta Docker
└── tests/
    ├── test_sim.py          # 23 tests del simulador
    └── test_rl.py           # Tests del entorno RL
```

## Datos de campo

Observación directa en el cruce el 19/05/2026 (mediodía, 40 minutos):

| Vialidad | Flujo observado | Tasa descarga | Ciclo semáforo |
|---|---|---|---|
| Av. Querétaro/Toluca | ~2,165 veh/h | 1.31 veh/s | 51s verde |
| Av. Toluca norte | ~722 veh/h | 0.29 veh/s | 51s verde |
| Lateral Norte | ~2,400 veh/h | 1.57 veh/s | 47s verde |
| Lateral Sur oeste | ~2,011 veh/h | 1.32 veh/s | 47s verde |

Perfiles de conductor observados: 30% agresivo / 60% normal / 10% cauteloso.

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Entrenar agente
bash scripts/entrenar.sh

# Demo visual
uvicorn viz.app:app --port 8000
```

## Variables de entorno

```bash
cp .env.example .env
# Editar según necesidad
```

Variables clave:
- `RL_REWARD_FUNCTION` — función de recompensa: `simple | balanceada | equidad | ponderada`
- `SIM_EPISODE_DURATION` — duración del episodio en segundos (default: 1800)
- `SIM_STEP_INTERVAL` — segundos entre decisiones del agente (default: 30)

## Tests

```bash
pytest tests/ -v
```

23 tests cubriendo geometría, semáforo, router, vehículo y simulador completo.
