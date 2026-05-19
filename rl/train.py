"""
rl/train.py
───────────
Entrenamiento del agente PPO para optimización del semáforo.

Uso:
    # Entrenamiento rápido con simulador dummy (desarrollo)
    python rl/train.py --dummy --steps 50000

    # Entrenamiento real con datos de campo calibrados
    python rl/train.py --steps 500000

    # Continuar entrenamiento desde un modelo guardado
    python rl/train.py --continuar rl/models/ppo_semaforo_v1.zip --steps 200000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(os.getenv("RL_MODEL_DIR", "rl/models"))
TOTAL_TIMESTEPS = int(os.getenv("RL_TOTAL_TIMESTEPS", 500_000))


def entrenar(usar_dummy: bool = False, total_steps: int = TOTAL_TIMESTEPS,
             modelo_base: Optional[str] = None, verbose: int = 1):
    """
    Entrena el agente PPO y guarda el modelo resultante.

    Args:
        usar_dummy:   usar simulador placeholder (sin datos de campo)
        total_steps:  pasos totales de entrenamiento
        modelo_base:  path a modelo existente para continuar entrenamiento
        verbose:      nivel de logging de stable-baselines3 (0, 1 o 2)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )
    from rl.environment import CruceEnv

    print("── Configurando entorno ────────────────────────────────")
    if usar_dummy:
        print("  Modo: simulador dummy (valores placeholder)")
        env = CruceEnv()
    else:
        print("  Modo: simulador calibrado con datos de campo")
        env = CruceEnv.desde_calibracion()

    # Verificar que el entorno cumple la interfaz gymnasium
    print("  Verificando entorno...")
    check_env(env, warn=True)
    print("  ✓ Entorno válido\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ────────────────────────────────────────────

    # Guardar checkpoint cada 50k pasos
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_semaforo",
        verbose=0,
    )

    # Evaluar en entorno separado cada 10k pasos
    eval_env = CruceEnv() if usar_dummy else CruceEnv.desde_calibracion()
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=3,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(MODEL_DIR / "logs"),
        verbose=0,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── Modelo ───────────────────────────────────────────────

    if modelo_base:
        print(f"── Cargando modelo base: {modelo_base}")
        model = PPO.load(modelo_base, env=env, verbose=verbose)
    else:
        print("── Creando modelo PPO nuevo")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            # Red neuronal: 2 capas de 64 neuronas
            # Suficiente para un espacio de estado de dimensión 7
            policy_kwargs={"net_arch": [64, 64]},
            learning_rate=3e-4,
            n_steps=2048,          # pasos por update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,            # descuento — valora recompensas futuras
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=verbose,
            tensorboard_log=str(MODEL_DIR / "tensorboard"),
        )

    # ── Entrenamiento ─────────────────────────────────────────

    print(f"\n── Entrenando por {total_steps:,} pasos...")
    print(f"   Función de recompensa: {os.getenv('RL_REWARD_FUNCTION', 'balanceada')}")
    print(f"   Duración de episodio:  {os.getenv('SIM_EPISODE_DURATION', 3600)}s")
    print(f"   Intervalo de decisión: {os.getenv('SIM_STEP_INTERVAL', 10)}s\n")

    inicio = datetime.now()
    model.learn(total_timesteps=total_steps, callback=callbacks, reset_num_timesteps=not modelo_base)
    duracion = datetime.now() - inicio

    # ── Guardar modelo final ──────────────────────────────────

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nombre_final = MODEL_DIR / f"ppo_semaforo_v1_{timestamp}.zip"
    model.save(str(nombre_final))

    print(f"\n✓ Entrenamiento completado en {duracion}")
    print(f"  Modelo guardado: {nombre_final}")
    print(f"  Mejor modelo:    {MODEL_DIR}/best_model.zip")

    return model


if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser(description="Entrenar agente PPO para el semáforo")
    parser.add_argument("--dummy", action="store_true",
                        help="Usar simulador dummy (sin datos de campo)")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help=f"Pasos de entrenamiento (default: {TOTAL_TIMESTEPS})")
    parser.add_argument("--continuar", type=str, default=None, metavar="PATH",
                        help="Path a modelo existente para continuar entrenamiento")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Nivel de logging (default: 1)")
    args = parser.parse_args()

    entrenar(
        usar_dummy=args.dummy,
        total_steps=args.steps,
        modelo_base=args.continuar,
        verbose=args.verbose,
    )
