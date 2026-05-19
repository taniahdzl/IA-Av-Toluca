"""
viz/renderer.py
───────────────
Traduce el estado interno del SimuladorCruce a un dict JSON
que el frontend puede consumir para animar el cruce.

No dibuja nada directamente — esa responsabilidad es del frontend.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from sim.intersection import SimuladorCruce, VIALIDADES


# Posiciones fijas de cada vialidad en el canvas del frontend
# Coordenadas normalizadas (0.0 a 1.0), origen arriba-izquierda
LAYOUT = {
    "toluca_arriba": {
        "direccion": "vertical",
        "x": 0.45, "y_inicio": 0.85, "y_fin": 0.55,
        "label": "Av. Toluca →Periférico",
    },
    "toluca_abajo": {
        "direccion": "vertical",
        "x": 0.55, "y_inicio": 0.15, "y_fin": 0.45,
        "label": "Av. Toluca →Revolución",
    },
    "periferico_norte": {
        "direccion": "horizontal",
        "y": 0.45, "x_inicio": 0.15, "x_fin": 0.45,
        "label": "Periférico Norte",
    },
    "periferico_sur": {
        "direccion": "horizontal",
        "y": 0.55, "x_inicio": 0.85, "x_fin": 0.55,
        "label": "Periférico Sur",
    },
}

# Colores del semáforo para el frontend
COLORES_FASE = {
    "verde_toluca":     {"toluca_arriba": "green",  "toluca_abajo": "green",
                         "periferico_norte": "red",  "periferico_sur": "red"},
    "verde_periferico": {"toluca_arriba": "red",    "toluca_abajo": "red",
                         "periferico_norte": "green","periferico_sur": "green"},
    "amarillo_1":       {v: "yellow" for v in VIALIDADES},
    "amarillo_2":       {v: "yellow" for v in VIALIDADES},
}


class Renderer:
    """
    Mantiene una referencia al simulador y genera frames JSON
    en cada llamada a get_frame().
    """

    def __init__(self, sim: SimuladorCruce):
        self.sim = sim

    def get_frame(self) -> dict:
        """
        Genera el frame actual del cruce para el frontend.

        Estructura del frame:
        {
          "t": int,
          "vialidades": { nombre: { "cola": int, "color_semaforo": str,
                                    "layout": {...}, "saturada": bool } },
          "semaforo": { "fase": str, "tiempo_restante": int, ... },
          "metricas": { "cola_total": int, "salidos_ultimo_min": int }
        }
        """
        sem_info = self.sim.semaforo.get_estado()
        fase = sem_info["fase"]
        colores = COLORES_FASE.get(fase, {v: "red" for v in VIALIDADES})

        # Agrupar colas por vialidad (suma de carriles)
        colas_por_vialidad = self._colas_por_vialidad()

        vialidades_frame = {}
        for v in VIALIDADES:
            cola = colas_por_vialidad.get(v, 0)
            capacidad = self.sim.geometria.capacidad_total(v)
            vialidades_frame[v] = {
                "cola":            cola,
                "capacidad":       capacidad,
                "pct_ocupacion":   round(cola / max(capacidad, 1), 3),
                "color_semaforo":  colores.get(v, "red"),
                "saturada":        cola >= capacidad * 0.9,
                "layout":          LAYOUT[v],
            }

        # Métricas del último minuto
        historia = self.sim.monitor._historia
        salidos_ultimo_min = int(sum(historia["salidos"][-60:])) if historia["salidos"] else 0

        return {
            "t":          self.sim.t,
            "vialidades": vialidades_frame,
            "semaforo":   sem_info,
            "metricas": {
                "cola_total":        sum(colas_por_vialidad.values()),
                "salidos_ultimo_min": salidos_ultimo_min,
            },
        }

    def get_metrics_comparison(self,
                                resumen_base: Optional[dict],
                                resumen_rl: Optional[dict]) -> dict:
        """
        Genera el payload para el dashboard de comparación
        baseline vs agente RL.
        """
        return {
            "baseline": resumen_base or {},
            "rl":       resumen_rl or {},
            "disponible": resumen_base is not None and resumen_rl is not None,
        }

    def _colas_por_vialidad(self) -> Dict[str, int]:
        prefijos = {
            "toluca_arriba":    "tol_arr",
            "toluca_abajo":     "tol_aba",
            "periferico_norte": "per_nor",
            "periferico_sur":   "per_sur",
        }
        resultado = {}
        colas = self.sim.get_colas()
        for vialidad, prefijo in prefijos.items():
            resultado[vialidad] = sum(
                len(q) for cid, q in colas.items() if cid.startswith(prefijo)
            )
        return resultado
