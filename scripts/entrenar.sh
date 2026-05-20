#!/usr/bin/env bash
# scripts/entrenar.sh
# Entrena agente PPO con 3 fases independientes y reward_asimetrica.
# Continúa desde modelo existente si lo hay.

set -e
cd "$(dirname "$0")/.."

export RL_REWARD_FUNCTION=asimetrica
export SIM_EPISODE_DURATION=3600
export SIM_STEP_INTERVAL=30

python - << 'PYEOF'
import os, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rl.environment import CruceEnv
from sim.traffic_light import Semaforo
from sim.geometry import GeometriaCruce
from sim.router import MarkovRouter
from sim.intersection import SimuladorCruce, FLUJOS_DEFAULT, TASA_DESCARGA_DEFAULT
from pathlib import Path

Path("rl/models").mkdir(exist_ok=True)

env = CruceEnv()

MODELO_PREV = "rl/models/ppo_semaforo_v1.zip"
if Path(MODELO_PREV).exists():
    print(f"Continuando desde {MODELO_PREV}...")
    model = PPO.load(MODELO_PREV, env=env,
                     learning_rate=1e-4, gamma=0.995,
                     clip_range=0.2, verbose=1)
else:
    print("Iniciando desde cero...")
    model = PPO(
        policy="MlpPolicy", env=env,
        policy_kwargs={"net_arch": [128, 128]},
        learning_rate=1e-4, n_steps=240, batch_size=60,
        n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, verbose=1,
    )

checkpoint_cb = CheckpointCallback(
    save_freq=50_000, save_path="rl/models",
    name_prefix="ppo_toluca", verbose=0,
)

print("=== Entrenando 500,000 pasos ===")
print("  reward_asimetrica | 3 fases | ep=1h | intervalo=30s | gamma=0.995")
print()
model.learn(total_timesteps=500_000, callback=checkpoint_cb,
            reset_num_timesteps=False)
model.save("rl/models/ppo_semaforo_v1")
print("\n✓ Modelo guardado: rl/models/ppo_semaforo_v1.zip")


def sem_fijo(f1, f2, f3):
    """Crea un semáforo con tiempos fijos para el baseline."""
    return Semaforo(
        duracion_fase_1=f1, duracion_fase_2=f2, duracion_fase_3=f3,
        carriles_fase_1=["que_tol_1","que_tol_2","que_tol_3","tol_nor_1"],
        carriles_fase_2=["lat_nor_1","lat_nor_2","lat_nor_3","lat_nor_4"],
        carriles_fase_3=["lat_sur_1","lat_sur_2"],
    )


def evaluar_fijo(f1, f2, f3, label):
    """Evalúa semáforo fijo SIN agente — siempre reproducible."""
    geo = GeometriaCruce.dummy()
    sim = SimuladorCruce(
        geo, sem_fijo(f1, f2, f3), MarkovRouter.dummy(),
        FLUJOS_DEFAULT, TASA_DESCARGA_DEFAULT, seed=42
    )
    sim.run(duracion_seg=3600, verbose=False)
    r = sim.monitor.resumen()
    esperas = [v.tiempo_espera for v in sim._vehiculos_salidos if v.tiempo_espera]
    r["espera_promedio"] = float(np.mean(esperas)) if esperas else 0
    print(f"{label}:")
    print(f"  Cola promedio:   {r['cola_promedio']:.1f} veh")
    print(f"  Cola máxima:     {r['cola_maxima']} veh")
    print(f"  Salidos:         {r['total_salidos']}")
    print(f"  Espera promedio: {r['espera_promedio']:.1f}s")
    print(f"  Tiempos:         fase_1={f1}s / fase_2={f2}s / fase_3={f3}s")
    return r


def evaluar_agente(modelo_path, label):
    """Evalúa el agente RL — deja que el agente ajuste el semáforo."""
    modelo = PPO.load(modelo_path)
    env = CruceEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        accion = int(modelo.predict(obs, deterministic=True)[0])
        obs, _, done, _, _ = env.step(accion)
    r = env.sim.monitor.resumen()
    esperas = [v.tiempo_espera for v in env.sim._vehiculos_salidos if v.tiempo_espera]
    r["espera_promedio"] = float(np.mean(esperas)) if esperas else 0
    sem = env.sim.semaforo
    print(f"{label}:")
    print(f"  Cola promedio:   {r['cola_promedio']:.1f} veh")
    print(f"  Cola máxima:     {r['cola_maxima']} veh")
    print(f"  Salidos:         {r['total_salidos']}")
    print(f"  Espera promedio: {r['espera_promedio']:.1f}s")
    print(f"  Tiempos:         fase_1={sem.duracion_fase_1}s / fase_2={sem.duracion_fase_2}s / fase_3={sem.duracion_fase_3}s")
    return r


print("\n=== Comparación ===")
# Baseline A: tiempos reales observados en campo (2 fases, 51s/47s combinados)
m_campo = evaluar_fijo(51, 25, 22, "Semáforo campo (51s/25s/22s)")
print()
# Baseline B: óptimo teórico del grid search
m_optimo = evaluar_fijo(62, 25, 20, "Óptimo teórico  (62s/25s/20s)")
print()
# Agente RL
m_rl = evaluar_agente("rl/models/ppo_semaforo_v1", "Agente RL")

print("\n=== Mejora vs semáforo de campo ===")
for k, nombre, mejor_si in [
    ("cola_promedio",  "Cola promedio", "baja"),
    ("total_salidos",  "Salidos",       "sube"),
    ("espera_promedio","Espera prom",   "baja"),
]:
    va, vb = m_campo[k], m_rl[k]
    d = (vb - va) / abs(va) * 100 if va else 0
    ok = (d < 0 and mejor_si == "baja") or (d > 0 and mejor_si == "sube")
    print(f"  {nombre}: {va:.1f} → {vb:.1f}  ({d:+.1f}%)  {'✓' if ok else '✗'}")

print("\n=== Óptimo teórico vs semáforo de campo ===")
for k, nombre, mejor_si in [
    ("cola_promedio",  "Cola promedio", "baja"),
    ("total_salidos",  "Salidos",       "sube"),
    ("espera_promedio","Espera prom",   "baja"),
]:
    va, vb = m_campo[k], m_optimo[k]
    d = (vb - va) / abs(va) * 100 if va else 0
    ok = (d < 0 and mejor_si == "baja") or (d > 0 and mejor_si == "sube")
    print(f"  {nombre}: {va:.1f} → {vb:.1f}  ({d:+.1f}%)  {'✓' if ok else '✗'}")
PYEOF
