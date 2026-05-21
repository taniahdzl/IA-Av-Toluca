# Optimización de Semáforos con RL — Av. Toluca × Anillo Periférico

Proyecto de IA que modela el cruce **Av. Querétaro/Toluca × Anillo Periférico (bajo nivel)**
en la Ciudad de México y entrena un agente de Aprendizaje por Refuerzo (**SAC — Soft
Actor-Critic**) para optimizar los tiempos del semáforo, reduciendo colas y tiempos de espera.

## Demo rápido

```bash
# Instalar dependencias
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Levantar demo visual
uvicorn viz.app:app --port 8000
# Abre http://localhost:8000
```

O con Docker:

```bash
docker compose up
# Abre http://localhost:8000
```

## El problema

El cruce tiene un cuello de botella estructural: **8 carriles de entrada** (Av. Querétaro 3c +
Av. Toluca norte 1c + Lateral Norte 4c + Lateral Sur oeste 2c) convergen en la **Zona H**
con 5 carriles físicos.

El semáforo alterna en **3 fases independientes**:

| Fase | Vialidad con verde | Duración campo | Zona H |
|------|--------------------|---------------|--------|
| Fase 1 | Av. Querétaro/Toluca (4c) | 51 s | Libre |
| Fase 2 | Lateral Norte (4c) | 30 s | **Bloqueada** |
| Fase 3 | Lateral Sur oeste (2c) | 21 s | **Bloqueada** |
| Amarillo | — | 3 s × 3 | — |

Con el reparto original (51s / 30s / 21s, ciclo ≈ 111s) para flujos muy desbalanceados, la
cola de Av. Querétaro acumula hasta 178 vehículos y los tiempos de espera superan los 15
minutos en hora pico.

El agente SAC controla directamente los 3 tiempos de verde [f1, f2, f3] como **acción
continua**, tomando una decisión cada 30 segundos simulados.

## Resultados

| Métrica | Semáforo campo (51/30/21 s) | Agente SAC | Mejora |
|---|---|---|---|
| Cola promedio | 104 veh | 86 veh | −17 % |
| Cola máxima | 178 veh | 164 veh | −8 % |
| Espera promedio | 59 s | 50 s | −16 % |
| Tiempos aprendidos | 51 / 30 / 21 s | ≈ 62 / 25 / 20 s | — |

El agente aprendió que aumentar ligeramente el verde de Av. Querétaro/Toluca (≈ 62s) y
reducir las fases de Lateral Norte y Sur oeste reduce los bloqueos en Zona H porque los
vehículos llegan más espaciados, generando menos congestión acumulada.

## Arquitectura

```
IA-Av-Toluca/
├── data/processed/           # JSONs calibrados con datos de campo
│   ├── geometria_carriles.json
│   ├── matriz_markov.json
│   └── flujos_calibrados.json
├── sim/                      # Simulador microscópico (independiente de RL)
│   ├── geometry.py           # Carriles y capacidades físicas
│   ├── vehicle.py            # Vehículos con perfiles de conductor (nivel 3)
│   ├── router.py             # Cadena de Markov para destinos
│   ├── traffic_light.py      # Semáforo: 3 fases, duraciones continuas ajustables
│   ├── intersection.py       # Motor principal: step(), colas, bloqueos Zona H
│   └── metrics.py            # Monitor de métricas
├── rl/                       # Aprendizaje por Refuerzo
│   ├── environment.py        # Entorno gymnasium — obs dim=10, action Box(3,) en [-1,1]
│   ├── reward.py             # Funciones de recompensa intercambiables (6 opciones)
│   ├── train.py              # Script auxiliar PPO (experimental)
│   └── evaluate.py           # Comparación baseline vs agente
├── viz/                      # Demo visual
│   ├── app.py                # Servidor FastAPI + loop de simulación con SAC
│   ├── renderer.py           # Traduce estado del simulador a JSON para el frontend
│   └── static/
│       ├── index.html        # Demo en vivo: SVG del cruce + canvas de colas
│       └── dashboard.html    # Dashboard de resultados comparativos
├── scripts/
│   ├── entrenar.sh           # Entrenamiento SAC (continúa desde modelo existente)
│   ├── run_demo.sh           # Levanta Docker
│   └── setup_data.sh         # Descarga datos OSM
├── notebooks/
│   ├── 01_calibracion.ipynb  # Calibrar JSONs con datos de campo
│   ├── 02_baseline.ipynb     # Análisis del semáforo fijo
│   └── 03_resultados_rl.ipynb
└── tests/
    ├── test_sim.py           # Tests del simulador
    └── test_rl.py            # Tests del entorno RL
```

## Datos de campo

Observación directa en el cruce el **19/05/2026** (mediodía, 40 minutos):

| Vialidad | Flujo observado | Tasa de descarga | Verde observado |
|---|---|---|---|
| Av. Querétaro/Toluca | ~2,887 veh/h (fusionado) | 1.61 veh/s | 51 s |
| Lateral Norte | ~2,400 veh/h | 1.57 veh/s | 30 s |
| Lateral Sur oeste | ~2,011 veh/h | 1.32 veh/s | 21 s |

Perfiles de conductor observados: 30 % agresivo / 60 % normal / 10 % cauteloso.

## Entrenamiento

```bash
# Entrenamiento (continúa desde rl/models/sac_semaforo_v1.zip si existe)
bash scripts/entrenar.sh
```

El script:
1. Carga el modelo existente `sac_semaforo_v1.zip` si existe (continúa el entrenamiento)
2. Entrena 300,000 pasos con SAC y función de recompensa `asimetrica`
3. Guarda checkpoints cada 25,000 pasos en `rl/models/`
4. Sobreescribe `rl/models/sac_semaforo_v1.zip` con el modelo final
5. Imprime una comparación automática: semáforo campo vs óptimo estimado vs agente SAC

Para actualizar el demo con el modelo recién entrenado **sin reiniciar el servidor**,
haz clic en el botón **"⬆ Recargar modelo"** en el panel de controles del demo.

## Demo visual

El demo muestra el cruce en tiempo real con el agente SAC tomando decisiones:

- **SVG base**: geometría real del cruce (Zona H, 3 vialidades, carriles, cruces peatonales)
- **Canvas superpuesto**: barras de cola que crecen desde la Zona H hacia afuera
  - Azul = ocupación normal · Naranja = ≥ 60 % · Rojo = ≥ 90 %
- **Semáforos dinámicos**: círculos en el SVG que cambian de color según la fase activa
- **Zona H**: se ilumina en rojo semitransparente cuando está bloqueada (fase 2 y 3)
- **Control de velocidad**: slider 1×–20×, con indicador de tiempo estimado por episodio

| Velocidad | sim-s / seg real | Duración episodio (07:00 → 08:00) |
|-----------|-----------------|-----------------------------------|
| 1× | 30 | ~ 2 min |
| 2× *(default)* | 60 | ~ 1 min |
| 5× | 150 | ~ 24 s |
| 20× | 600 | ~ 6 s |

Cada episodio simula **1 hora de tráfico** (07:00–08:00, hora pico mañana). Al terminar,
el agente reinicia y comienza un episodio nuevo.

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `RL_REWARD_FUNCTION` | `balanceada` | Función de recompensa: `simple \| balanceada \| equidad \| flickering \| ponderada \| asimetrica` |
| `SIM_EPISODE_DURATION` | `3600` | Duración del episodio en segundos simulados |
| `SIM_STEP_INTERVAL` | `30` | Segundos simulados entre decisiones del agente |
| `VIZ_SIM_SPEED` | `2` | Velocidad inicial del demo (pasos de sim por tick) |
| `RL_MODEL_PATH` | `rl/models/sac_semaforo_v1.zip` | Ruta al modelo SAC |

## Tests

```bash
pytest tests/ -v
```
