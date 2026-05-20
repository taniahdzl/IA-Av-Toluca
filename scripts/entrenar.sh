#!/usr/bin/env bash
# scripts/entrenar.sh
# Entrena el agente SAC con espacio de acción continuo.
# SAC elige directamente [f1, f2, f3] en segundos para cada ciclo semafórico.

set -e
cd "$(dirname "$0")/.."

export RL_REWARD_FUNCTION=asimetrica
export SIM_EPISODE_DURATION=3600
export SIM_STEP_INTERVAL=30

python - << 'PYEOF'
import os, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from rl.environment import CruceEnv
from sim.traffic_light import Semaforo
from sim.geometry import GeometriaCruce
from sim.router import MarkovRouter
from sim.intersection import SimuladorCruce, FLUJOS_DEFAULT, TASA_DESCARGA_DEFAULT
from pathlib import Path

Path("rl/models").mkdir(exist_ok=True)

print("Verificando entorno...")
env = CruceEnv()
check_env(env, warn=True)
print(f"✓ obs={env.observation_space.shape}, action={env.action_space.shape}")
print()

MODELO_PREV = "rl/models/sac_semaforo_v1.zip"
if Path(MODELO_PREV).exists():
    print(f"Continuando desde {MODELO_PREV}...")
    model = SAC.load(MODELO_PREV, env=env,
                     learning_rate=3e-4, verbose=1)
else:
    print("Iniciando SAC desde cero...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [256, 256]},
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,   # explorar 1000 steps antes de aprender
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",         # SAC ajusta automáticamente la entropía
        verbose=1,
    )

checkpoint_cb = CheckpointCallback(
    save_freq=25_000, save_path="rl/models",
    name_prefix="sac_toluca", verbose=0,
)

print("=== Entrenando SAC — 300,000 pasos ===")
print("  reward_asimetrica | ep=1h | intervalo=30s | acción continua [f1,f2,f3]")
print()
model.learn(total_timesteps=300_000, callback=checkpoint_cb,
            reset_num_timesteps=not Path(MODELO_PREV).exists())
model.save("rl/models/sac_semaforo_v1")
print("\n✓ Modelo guardado: rl/models/sac_semaforo_v1.zip")


def sem_fijo(f1, f2, f3):
    return Semaforo(
        duracion_fase_1=f1, duracion_fase_2=f2, duracion_fase_3=f3,
        carriles_fase_1=["que_tol_1","que_tol_2","que_tol_3","que_tol_4"],
        carriles_fase_2=["lat_nor_1","lat_nor_2","lat_nor_3","lat_nor_4"],
        carriles_fase_3=["lat_sur_1","lat_sur_2"],
    )


def evaluar_fijo(f1, f2, f3, label):
    geo = GeometriaCruce.dummy()
    sim = SimuladorCruce(geo, sem_fijo(f1, f2, f3), MarkovRouter.dummy(),
                         FLUJOS_DEFAULT, TASA_DESCARGA_DEFAULT, seed=42)
    sim.run(duracion_seg=3600, verbose=False)
    r = sim.monitor.resumen()
    esperas = [v.tiempo_espera for v in sim._vehiculos_salidos if v.tiempo_espera]
    r["espera_promedio"] = float(np.mean(esperas)) if esperas else 0
    print(f"{label}:")
    print(f"  Cola prom:  {r['cola_promedio']:.1f} veh | Cola max: {r['cola_maxima']}")
    print(f"  Salidos:    {r['total_salidos']}        | Espera:   {r['espera_promedio']:.1f}s")
    print(f"  Tiempos:    f1={f1}s / f2={f2}s / f3={f3}s | ciclo={f1+f2+f3+9}s")
    return r


def evaluar_agente(modelo_path, label):
    modelo = SAC.load(modelo_path)
    env = CruceEnv()
    obs, _ = env.reset()
    done = False
    acciones = []
    while not done:
        accion, _ = modelo.predict(obs, deterministic=True)
        acciones.append(env._desnormalizar_accion(accion))
        obs, _, done, _, _ = env.step(accion)
    r = env.sim.monitor.resumen()
    esperas = [v.tiempo_espera for v in env.sim._vehiculos_salidos if v.tiempo_espera]
    r["espera_promedio"] = float(np.mean(esperas)) if esperas else 0
    sem = env.sim.semaforo
    acc = np.mean(acciones, axis=0)
    print(f"{label}:")
    print(f"  Cola prom:  {r['cola_promedio']:.1f} veh | Cola max: {r['cola_maxima']}")
    print(f"  Salidos:    {r['total_salidos']}        | Espera:   {r['espera_promedio']:.1f}s")
    print(f"  Tiempos finales: f1={sem.duracion_fase_1}s / f2={sem.duracion_fase_2}s / f3={sem.duracion_fase_3}s | ciclo={sem.ciclo_total}s")
    print(f"  Tiempos promedio del episodio: f1={acc[0]:.0f}s / f2={acc[1]:.0f}s / f3={acc[2]:.0f}s")
    return r


print("\n=== Comparación ===")
m_campo  = evaluar_fijo(51, 30, 21, "Semáforo campo  (51s/30s/21s, ciclo=111s)")
print()
m_optimo = evaluar_fijo(55, 35, 12, "Óptimo estimado (55s/35s/12s, ciclo=111s)")
print()
m_rl     = evaluar_agente("rl/models/sac_semaforo_v1", "Agente SAC")

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
PYEOF
