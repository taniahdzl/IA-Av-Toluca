#!/usr/bin/env bash
# scripts/entrenar.sh
# Entrena el agente PPO. Continúa desde el modelo existente si lo hay.
# Uso: bash scripts/entrenar.sh

set -e
cd "$(dirname "$0")/.."

export RL_REWARD_FUNCTION=ponderada
export SIM_EPISODE_DURATION=1800
export SIM_STEP_INTERVAL=30

python - << 'PYEOF'
import os, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rl.environment import CruceEnv
from pathlib import Path

Path("rl/models").mkdir(exist_ok=True)

env = CruceEnv()

MODELO_PREV = "rl/models/ppo_semaforo_v1.zip"
if Path(MODELO_PREV).exists():
    print(f"Continuando desde {MODELO_PREV}...")
    model = PPO.load(MODELO_PREV, env=env,
                     learning_rate=3e-4,
                     gamma=0.995,
                     clip_range=0.2,
                     verbose=1)
else:
    print("Iniciando desde cero...")
    model = PPO(
        policy="MlpPolicy", env=env,
        policy_kwargs={"net_arch": [64, 64]},
        learning_rate=3e-4, n_steps=120, batch_size=30,
        n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        verbose=1,
    )

checkpoint_cb = CheckpointCallback(
    save_freq=50_000, save_path="rl/models",
    name_prefix="ppo_toluca", verbose=0,
)

print("=== Entrenando 500,000 pasos ===")
print("  reward_ponderada | ep=30min | intervalo=30s | gamma=0.995 | fase_1_min=30s")
print()
model.learn(total_timesteps=500_000, callback=checkpoint_cb, reset_num_timesteps=False)
model.save("rl/models/ppo_semaforo_v1")
print("\n✓ Modelo guardado: rl/models/ppo_semaforo_v1.zip")

def evaluar(modelo=None, label=""):
    env = CruceEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        accion = int(modelo.predict(obs, deterministic=True)[0]) if modelo else 0
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
    print(f"  Tiempos:         fase_1={sem.duracion_fase_1}s / fase_2={sem.duracion_fase_2}s")
    return r

print("\n=== Comparación ===")
m_base = evaluar(None, "Semáforo fijo (51s/47s)")
print()
m_rl = evaluar(PPO.load("rl/models/ppo_semaforo_v1"), "Agente RL")

print("\n=== Mejora ===")
for k, nombre, mejor_si in [
    ("cola_promedio", "Cola promedio", "baja"),
    ("total_salidos", "Salidos", "sube"),
    ("espera_promedio", "Espera prom", "baja"),
]:
    va, vb = m_base[k], m_rl[k]
    d = (vb - va) / abs(va) * 100 if va else 0
    ok = (d < 0 and mejor_si == "baja") or (d > 0 and mejor_si == "sube")
    print(f"  {nombre}: {va:.1f} → {vb:.1f}  ({d:+.1f}%)  {'✓' if ok else '✗'}")
PYEOF
