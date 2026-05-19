"""
tests/test_rl.py
────────────────
Tests para rl/reward.py y rl/environment.py.
Correr con: pytest tests/test_rl.py -v

Los tests de environment requieren que sim/intersection.py
tenga implementados get_estado() y step() (quitar skip cuando estén listos).
"""

import pytest
import numpy as np


# ─────────────────────────────────────────────
# reward.py
# ─────────────────────────────────────────────

class TestRewardFunctions:
    """Los tests de recompensa no dependen del simulador — corren ya."""

    def _info(self, colas: dict = None, salidos: int = 5,
              accion_aplicada: bool = False) -> dict:
        """Info dict de ejemplo para testear funciones de recompensa."""
        return {
            "colas": colas or {
                "tol_arr_1": 5, "tol_arr_2": 3,
                "tol_aba_1": 4, "tol_aba_2": 2,
                "per_nor_1": 8, "per_sur_1": 6,
            },
            "salidos_step": salidos,
            "accion_aplicada": accion_aplicada,
            "capacidades": {
                "tol_arr_1": 15, "tol_arr_2": 15,
                "tol_aba_1": 15, "tol_aba_2": 15,
                "per_nor_1": 12, "per_sur_1": 12,
            },
            "semaforo": {"fase": "verde_toluca", "fase_idx": 0,
                         "tiempo_restante": 30, "duracion_toluca": 45,
                         "duracion_periferico": 55},
        }

    def test_reward_simple_es_negativo_con_colas(self):
        from rl.reward import reward_simple
        r = reward_simple(self._info())
        assert r < 0

    def test_reward_simple_cero_con_colas_vacias(self):
        from rl.reward import reward_simple
        info = self._info(colas={k: 0 for k in ["tol_arr_1", "tol_arr_2",
                                                  "tol_aba_1", "tol_aba_2",
                                                  "per_nor_1", "per_sur_1"]})
        r = reward_simple(info)
        assert r == 0.0

    def test_reward_balanceada_mejor_con_mas_salidas(self):
        from rl.reward import reward_balanceada
        r_pocos = reward_balanceada(self._info(salidos=1))
        r_muchos = reward_balanceada(self._info(salidos=10))
        assert r_muchos > r_pocos

    def test_reward_balanceada_peor_con_saturacion(self):
        from rl.reward import reward_balanceada
        # Colas normales
        r_normal = reward_balanceada(self._info())
        # Colas saturadas (≥90% de capacidad)
        colas_sat = {
            "tol_arr_1": 14, "tol_arr_2": 14,
            "tol_aba_1": 14, "tol_aba_2": 14,
            "per_nor_1": 11, "per_sur_1": 11,
        }
        r_sat = reward_balanceada(self._info(colas=colas_sat))
        assert r_sat < r_normal

    def test_reward_equidad_penaliza_desbalance(self):
        from rl.reward import reward_equidad
        # Colas balanceadas
        colas_bal = {k: 5 for k in ["tol_arr_1", "tol_arr_2",
                                      "tol_aba_1", "tol_aba_2",
                                      "per_nor_1", "per_sur_1"]}
        # Colas desbalanceadas: Periférico muy congestionado
        colas_des = {
            "tol_arr_1": 1, "tol_arr_2": 1,
            "tol_aba_1": 1, "tol_aba_2": 1,
            "per_nor_1": 20, "per_sur_1": 20,
        }
        r_bal = reward_equidad(self._info(colas=colas_bal))
        r_des = reward_equidad(self._info(colas=colas_des))
        assert r_bal > r_des

    def test_reward_flickering_penaliza_accion(self):
        from rl.reward import reward_con_flickering
        r_sin = reward_con_flickering(self._info(accion_aplicada=False))
        r_con = reward_con_flickering(self._info(accion_aplicada=True))
        assert r_sin > r_con

    def test_get_reward_fn_retorna_callable(self):
        import os
        os.environ["RL_REWARD_FUNCTION"] = "balanceada"
        from rl.reward import get_reward_fn
        fn = get_reward_fn()
        assert callable(fn)
        r = fn(self._info())
        assert isinstance(r, float)

    def test_get_reward_fn_lanza_error_con_nombre_invalido(self):
        import os
        os.environ["RL_REWARD_FUNCTION"] = "no_existe"
        from rl.reward import get_reward_fn
        with pytest.raises(ValueError):
            get_reward_fn()


# ─────────────────────────────────────────────
# environment.py
# ─────────────────────────────────────────────

@pytest.mark.skip(reason="Requiere SimuladorCruce.get_estado() y step() implementados")
class TestCruceEnv:

    def _env(self):
        import os
        os.environ["RL_REWARD_FUNCTION"] = "balanceada"
        os.environ["SIM_EPISODE_DURATION"] = "120"   # episodios cortos para tests
        from rl.environment import CruceEnv
        return CruceEnv()

    def test_observation_space_correcto(self):
        from rl.environment import ESTADO_DIM
        import gymnasium as gym
        env = self._env()
        assert env.observation_space.shape == (ESTADO_DIM,)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 5

    def test_reset_retorna_obs_valida(self):
        env = self._env()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert isinstance(info, dict)

    def test_step_retorna_formato_correcto(self):
        env = self._env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert isinstance(info, dict)

    def test_episodio_termina(self):
        env = self._env()
        obs, _ = env.reset()
        terminated = False
        pasos = 0
        while not terminated:
            obs, _, terminated, _, _ = env.step(0)
            pasos += 1
            assert pasos < 10_000, "El episodio no terminó en tiempo razonable"
        assert terminated

    def test_check_env_gymnasium(self):
        """Verifica que el entorno cumple la interfaz gymnasium completa."""
        from stable_baselines3.common.env_checker import check_env
        env = self._env()
        check_env(env, warn=True)   # lanza AssertionError si hay problemas

    def test_todas_las_acciones_son_validas(self):
        env = self._env()
        for accion in range(5):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(accion)
            assert obs is not None
            assert isinstance(reward, float)
