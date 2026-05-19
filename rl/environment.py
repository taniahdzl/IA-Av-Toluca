"""
rl/environment.py
─────────────────
Entorno gymnasium compatible con stable-baselines3.
Envuelve a SimuladorCruce sin reimplementar nada de la física.

Uso:
    env = CruceEnv()                     # usa simulador dummy
    env = CruceEnv.desde_calibracion()  # usa datos reales de campo

    # Verificar que el entorno es válido
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    # Entrenar
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.intersection import SimuladorCruce, ESTADO_DIM
from rl.reward import get_reward_fn

# Duración de cada episodio en segundos (default: 1 hora)
EPISODE_DURATION = int(os.getenv("SIM_EPISODE_DURATION", 3600))

# Número de acciones discretas del agente
# 0: no hacer nada
# 1: +10s verde Toluca
# 2: -10s verde Toluca
# 3: +10s verde Periférico
# 4: -10s verde Periférico
N_ACCIONES = 5


class CruceEnv(gym.Env):
    """
    Entorno gymnasium para optimización del semáforo en el cruce
    Av. Toluca × Anillo Periférico.

    observation_space: Box(7,) — ver SimuladorCruce.get_estado()
    action_space:      Discrete(5) — ver N_ACCIONES

    El agente toma una acción cada SIM_STEP_INTERVAL segundos
    (no necesariamente cada segundo) para dar tiempo al semáforo
    de responder antes de la siguiente decisión.
    """

    metadata = {"render_modes": ["human", "json"]}

    # Cada cuántos segundos el agente toma una decisión
    SIM_STEP_INTERVAL = int(os.getenv("SIM_STEP_INTERVAL", 10))

    def __init__(self, sim: Optional[SimuladorCruce] = None,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.sim = sim or SimuladorCruce.dummy()
        self.reward_fn = get_reward_fn()
        self.render_mode = render_mode
        self._episode_steps = 0
        self._max_steps = EPISODE_DURATION // self.SIM_STEP_INTERVAL
        self._ultimo_estado_render: Optional[dict] = None

        # ── Espacios ─────────────────────────────────────────
        # Observación: 7 valores continuos en [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(ESTADO_DIM,),
            dtype=np.float32,
        )

        # Acción: entero discreto en {0, 1, 2, 3, 4}
        self.action_space = spaces.Discrete(N_ACCIONES)

    # ── Construcción alternativa ──────────────────────────────

    @classmethod
    def desde_calibracion(cls, render_mode: Optional[str] = None) -> "CruceEnv":
        """
        Crea el entorno con el simulador calibrado con datos de campo.
        Usar cuando los JSONs en data/processed/ ya estén llenos.
        """
        sim = SimuladorCruce.desde_calibracion()
        return cls(sim=sim, render_mode=render_mode)

    # ── Interfaz gymnasium ────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.sim._seed = seed
        self.sim.reset()
        self._episode_steps = 0
        obs = self.sim.get_estado()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Avanza SIM_STEP_INTERVAL segundos con la acción dada.
        La recompensa es la suma de recompensas de todos los sub-steps.

        Returns:
            obs         : nuevo vector de estado
            reward      : recompensa acumulada del intervalo
            terminated  : True si terminó el episodio
            truncated   : siempre False (no usamos truncation)
            info        : último info dict del simulador
        """
        recompensa_total = 0.0
        info = {}
        accion_aplicada = False

        for i in range(self.SIM_STEP_INTERVAL):
            # Solo aplicar la acción en el primer sub-step del intervalo
            accion_step = action if i == 0 else 0
            if accion_step != 0:
                accion_aplicada = True

            obs, _, info = self.sim.step(accion=accion_step)

            # Agregar flag de acción al info para reward_con_flickering
            info["accion_aplicada"] = accion_aplicada and i == 0

            # Agregar capacidades de carriles al info para cálculo de saturación
            info["capacidades"] = {
                c.id: c.capacidad_vehiculos
                for c in self.sim.geometria.todos_los_carriles()
            }

            recompensa_total += self.reward_fn(info)

        self._episode_steps += 1
        terminated = self._episode_steps >= self._max_steps
        self._ultimo_estado_render = info

        return obs, recompensa_total, terminated, False, info

    def render(self) -> Optional[dict]:
        """
        Devuelve el estado actual del cruce como dict.
        viz/renderer.py consume este dict para la animación.

        No dibuja directamente — la visualización es responsabilidad de viz/.
        """
        if self.render_mode == "json":
            return self._ultimo_estado_render
        if self.render_mode == "human":
            if self._ultimo_estado_render:
                sem = self._ultimo_estado_render.get("semaforo", {})
                colas = self._ultimo_estado_render.get("colas", {})
                print(f"t={self.sim.t:5d}s | "
                      f"Cola total: {sum(colas.values()):3d} | "
                      f"Fase: {sem.get('fase','?')} "
                      f"({sem.get('tiempo_restante','?')}s)")
        return None

    def close(self):
        pass

    # ── Propiedades útiles ────────────────────────────────────

    @property
    def t(self) -> int:
        """Segundo actual de la simulación."""
        return self.sim.t

    @property
    def progreso_episodio(self) -> float:
        """Fracción del episodio completada (0.0 a 1.0)."""
        return self._episode_steps / self._max_steps
