"""
rl/environment.py
─────────────────
Entorno gymnasium con espacio de acción CONTINUO para SAC.

observation_space: Box(10,) — estado del cruce normalizado [0,1]
action_space:      Box(3,)  — [f1, f2, f3] normalizados en [-1, 1]
                              desnormalizados a segundos reales al aplicar

Rangos reales:
  f1 ∈ [30, 120]s  — verde Querétaro/Toluca
  f2 ∈ [10,  90]s  — verde Lateral Norte
  f3 ∈ [10,  60]s  — verde Lateral Sur oeste

El agente toma una decisión cada SIM_STEP_INTERVAL segundos.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.intersection import SimuladorCruce, ESTADO_DIM
from sim.traffic_light import Semaforo
from rl.reward import get_reward_fn

EPISODE_DURATION  = int(os.getenv("SIM_EPISODE_DURATION", 3600))
SIM_STEP_INTERVAL = int(os.getenv("SIM_STEP_INTERVAL", 30))

# Rangos reales de cada fase en segundos
F1_MIN, F1_MAX = float(Semaforo.F1_MIN), float(Semaforo.F1_MAX)
F2_MIN, F2_MAX = float(Semaforo.F2_MIN), float(Semaforo.F2_MAX)
F3_MIN, F3_MAX = float(Semaforo.F3_MIN), float(Semaforo.F3_MAX)


class CruceEnv(gym.Env):
    """
    Entorno gymnasium para optimización continua del semáforo.
    Usa SAC (Soft Actor-Critic) como algoritmo de entrenamiento.
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, sim: Optional[SimuladorCruce] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.sim = sim or SimuladorCruce.dummy()
        self.reward_fn = get_reward_fn()
        self.render_mode = render_mode
        self._episode_steps = 0
        self._max_steps = EPISODE_DURATION // SIM_STEP_INTERVAL
        self._ultimo_info: Optional[dict] = None

        # ── Espacios ─────────────────────────────────────────
        # Observación: 10 valores en [0, 1]
        # [cola×3, t_restante, fase, periodo, espera_que, ratio_f1, ratio_f2, ratio_f3]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(ESTADO_DIM,),
            dtype=np.float32,
        )

        # Acción continua: [f1, f2, f3] normalizados en [-1, 1]
        # SAC trabaja mejor con acciones en [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    @classmethod
    def desde_calibracion(cls, render_mode=None) -> "CruceEnv":
        return cls(sim=SimuladorCruce.desde_calibracion(), render_mode=render_mode)

    def _desnormalizar_accion(self, accion: np.ndarray) -> np.ndarray:
        """
        Convierte acción normalizada [-1, 1] a segundos reales.
        f = min + (accion + 1) / 2 × (max - min)
        """
        f1 = F1_MIN + (accion[0] + 1) / 2 * (F1_MAX - F1_MIN)
        f2 = F2_MIN + (accion[1] + 1) / 2 * (F2_MAX - F2_MIN)
        f3 = F3_MIN + (accion[2] + 1) / 2 * (F3_MAX - F3_MIN)
        return np.array([f1, f2, f3], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim._seed = seed
        self.sim.reset()
        self._episode_steps = 0
        return self.sim.get_estado(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Avanza SIM_STEP_INTERVAL segundos con la acción dada.

        action: np.ndarray shape (3,) normalizado en [-1, 1]
        Returns: obs, reward, terminated, truncated, info
        """
        # Desnormalizar y aplicar solo en el primer sub-step
        accion_real = self._desnormalizar_accion(action)
        recompensa_total = 0.0
        info = {}

        for i in range(SIM_STEP_INTERVAL):
            accion_step = accion_real if i == 0 else None
            obs, _, info = self.sim.step(accion=accion_step)

            # Enriquecer info para la función de recompensa
            info["capacidades"] = {
                c.id: c.capacidad_vehiculos
                for c in self.sim.geometria.todos_los_carriles()
            }
            recompensa_total += self.reward_fn(info)

        self._episode_steps += 1
        self._ultimo_info = info
        terminated = self._episode_steps >= self._max_steps

        return obs, recompensa_total, terminated, False, info

    def render(self) -> Optional[dict]:
        if self.render_mode == "json":
            return self._ultimo_info
        if self.render_mode == "human" and self._ultimo_info:
            sem = self._ultimo_info.get("semaforo", {})
            print(f"t={self.sim.t:5d}s | "
                  f"f1={sem.get('duracion_fase_1','?')}s "
                  f"f2={sem.get('duracion_fase_2','?')}s "
                  f"f3={sem.get('duracion_fase_3','?')}s | "
                  f"ciclo={sem.get('ciclo_total','?')}s")
        return None

    def close(self):
        pass

    @property
    def t(self):
        return self.sim.t
