"""
rl/reward.py
────────────
Funciones de recompensa intercambiables para el agente de RL.

Todas reciben el mismo `info` dict que devuelve SimuladorCruce.step()
y retornan un float.

La función activa se selecciona con la variable de entorno RL_REWARD_FUNCTION.
Opciones: simple | balanceada | equidad | flickering

Diseño:
  - Penalizar colas grandes (congestión)
  - Premiar vehículos que salen (flujo)
  - Penalizar saturación extrema (colapso)
  - Penalizar inequidad entre vialidades (opcional)
  - Penalizar cambios de semáforo muy frecuentes (opcional)
"""

from __future__ import annotations

import os
from typing import Callable, Dict

# ── Pesos configurables ───────────────────────────────────────

# Penalización por vehículo en cola por segundo
W_COLA = 0.1

# Premio por vehículo que cruzó en este step
W_SALIDA = 2.0

# Penalización extra cuando una vialidad está saturada (≥90% capacidad)
W_SATURACION = 20.0

# Penalización por inequidad (desviación estándar de colas entre vialidades)
W_INEQUIDAD = 0.5

# Penalización por cambiar los tiempos del semáforo en este step
W_FLICKERING = 5.0

# Umbrales
UMBRAL_SATURACION = 0.9   # fracción de capacidad máxima


# ── Funciones de recompensa ───────────────────────────────────

def reward_simple(info: dict) -> float:
    """
    Solo penaliza la cola total.
    Útil como línea base para comparar funciones más complejas.

    r = -W_COLA * cola_total
    """
    cola_total = sum(info["colas"].values())
    return -W_COLA * cola_total


def reward_balanceada(info: dict) -> float:
    """
    Penaliza colas, premia salidas y penaliza saturación.
    Función de recompensa recomendada para el entrenamiento principal.

    r = -W_COLA * cola_total
        + W_SALIDA * salidos
        - W_SATURACION * n_carriles_saturados
    """
    cola_total = sum(info["colas"].values())
    salidos = info["salidos_step"]

    # Contar carriles saturados usando capacidades de la geometría
    n_saturados = sum(
        1 for carril_id, ocupacion in info["colas"].items()
        if _esta_saturado(carril_id, ocupacion, info)
    )

    return (
        -W_COLA * cola_total
        + W_SALIDA * salidos
        - W_SATURACION * n_saturados
    )


def reward_equidad(info: dict) -> float:
    """
    Extiende reward_balanceada penalizando además la inequidad entre vialidades.
    Evita que el agente favorezca siempre una vialidad sobre las demás.

    r = reward_balanceada
        - W_INEQUIDAD * std(colas_por_vialidad)
    """
    base = reward_balanceada(info)

    # Agregar colas por vialidad (suma de sus carriles)
    colas_por_vialidad = _colas_por_vialidad(info["colas"])
    if len(colas_por_vialidad) > 1:
        import numpy as np
        std = float(np.std(list(colas_por_vialidad.values())))
        return base - W_INEQUIDAD * std

    return base


def reward_con_flickering(info: dict) -> float:
    """
    Extiende reward_equidad penalizando cambios de semáforo muy frecuentes.
    Evita que el agente oscile los tiempos sin estabilizarse.

    La penalización solo aplica si el agente cambió los tiempos en este step
    (detectado por la presencia de 'accion_aplicada' en info).
    """
    base = reward_equidad(info)
    penalizacion = W_FLICKERING if info.get("accion_aplicada", False) else 0.0
    return base - penalizacion


# ── Selector de función activa ────────────────────────────────


def reward_ponderada(info: dict) -> float:
    """
    Recompensa ponderada por volumen real de cada vialidad.
    Penaliza más fuerte las colas de alto volumen (queretaro_toluca, lateral_norte).
    """
    PESOS = {
        "que_tol": 2165 / 2011,
        "tol_nor": 722  / 2011,
        "lat_nor": 2400 / 2011,
        "lat_sur": 1.0,
    }
    PREFIJOS = {
        "que_tol": ["que_tol_1", "que_tol_2", "que_tol_3"],
        "tol_nor": ["tol_nor_1"],
        "lat_nor": ["lat_nor_1", "lat_nor_2", "lat_nor_3", "lat_nor_4"],
        "lat_sur": ["lat_sur_1", "lat_sur_2"],
    }
    penalizacion_colas = sum(
        PESOS[g] * sum(info["colas"].get(p, 0) for p in ps) * W_COLA
        for g, ps in PREFIJOS.items()
    )
    premio_salidas = info["salidos_step"] * W_SALIDA
    penalizacion_bloqueo = info.get("n_bloqueadores", 0) * 15.0
    n_saturados = sum(
        1 for cid, q in info["colas"].items()
        if _esta_saturado(cid, q, info)
    )
    return -penalizacion_colas + premio_salidas - penalizacion_bloqueo - n_saturados * W_SATURACION

FUNCIONES: Dict[str, Callable[[dict], float]] = {
    "simple":      reward_simple,
    "balanceada":  reward_balanceada,
    "equidad":     reward_equidad,
    "flickering":  reward_con_flickering,
    "ponderada":   reward_ponderada,
}


def get_reward_fn() -> Callable[[dict], float]:
    """
    Retorna la función de recompensa configurada en .env.
    Default: balanceada.
    """
    nombre = os.getenv("RL_REWARD_FUNCTION", "balanceada").lower()
    if nombre not in FUNCIONES:
        raise ValueError(
            f"RL_REWARD_FUNCTION='{nombre}' no es válida. "
            f"Opciones: {list(FUNCIONES.keys())}"
        )
    return FUNCIONES[nombre]


# ── Helpers internos ──────────────────────────────────────────

def _esta_saturado(carril_id: str, ocupacion: int, info: dict) -> bool:
    """
    Determina si un carril está saturado.
    Usa capacidades de la geometría si están disponibles en info,
    si no, usa un umbral fijo de 15 vehículos como fallback.
    """
    capacidades = info.get("capacidades", {})
    capacidad = capacidades.get(carril_id, 15)
    return ocupacion >= capacidad * UMBRAL_SATURACION


def _colas_por_vialidad(colas: Dict[str, int]) -> Dict[str, int]:
    """
    Agrupa las colas de carriles individuales por vialidad.
    Asume que el id del carril empieza con el nombre de la vialidad
    (ej. "tol_arr_1" pertenece a "toluca_arriba").
    """
    vialidades = ["toluca_arriba", "toluca_abajo", "periferico_norte", "periferico_sur"]
    prefijos = {
        "toluca_arriba":    "tol_arr",
        "toluca_abajo":     "tol_aba",
        "periferico_norte": "per_nor",
        "periferico_sur":   "per_sur",
    }
    resultado: Dict[str, int] = {}
    for vialidad in vialidades:
        prefijo = prefijos[vialidad]
        resultado[vialidad] = sum(
            v for k, v in colas.items() if k.startswith(prefijo)
        )
    return resultado
