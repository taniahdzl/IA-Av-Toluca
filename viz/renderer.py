"""
viz/renderer.py
───────────────
Traduce el estado interno del SimuladorCruce a un dict JSON
que el frontend consume para animar el cruce.

3 vialidades con semáforo:
  queretaro_toluca  — 4 carriles fusionados (fase 1)
  lateral_norte     — 4 carriles (fase 2)
  lateral_sur_oeste — 2 carriles rectos (fase 3)
"""

from __future__ import annotations
from typing import Dict, Optional
from sim.intersection import SimuladorCruce, VIALIDADES

LAYOUT = {
    "queretaro_toluca": {
        "label": "Av. Querétaro/Toluca (4c)",
        "carriles": 4,
        "direccion": "diagonal_no_se",
    },
    "lateral_norte": {
        "label": "Lateral Norte ← (4c)",
        "carriles": 4,
        "direccion": "horizontal_left",
    },
    "lateral_sur_oeste": {
        "label": "Lateral Sur → (2c)",
        "carriles": 2,
        "direccion": "horizontal_right",
    },
}

COLORES_FASE = {
    "fase_1":    {"queretaro_toluca": "green", "lateral_norte": "red",   "lateral_sur_oeste": "red"},
    "amarillo_1":{"queretaro_toluca": "yellow","lateral_norte": "yellow","lateral_sur_oeste": "yellow"},
    "fase_2":    {"queretaro_toluca": "red",   "lateral_norte": "green", "lateral_sur_oeste": "red"},
    "amarillo_2":{"queretaro_toluca": "yellow","lateral_norte": "yellow","lateral_sur_oeste": "yellow"},
    "fase_3":    {"queretaro_toluca": "red",   "lateral_norte": "red",   "lateral_sur_oeste": "green"},
    "amarillo_3":{"queretaro_toluca": "yellow","lateral_norte": "yellow","lateral_sur_oeste": "yellow"},
}

PREFIJOS = {
    "queretaro_toluca":  "que_tol",
    "lateral_norte":     "lat_nor",
    "lateral_sur_oeste": "lat_sur",
}


class Renderer:
    def __init__(self, sim: SimuladorCruce):
        self.sim = sim

    def get_frame(self) -> dict:
        sem_info = self.sim.semaforo.get_estado()
        fase = sem_info["fase"]
        colores = COLORES_FASE.get(fase, {v: "red" for v in VIALIDADES})

        colas = self._colas_por_vialidad()
        vialidades_frame = {}
        for v in VIALIDADES:
            cola = colas.get(v, 0)
            capacidad = max(self.sim.geometria.capacidad_total(v), 1)
            vialidades_frame[v] = {
                "cola":           cola,
                "capacidad":      capacidad,
                "pct_ocupacion":  round(cola / capacidad, 3),
                "color_semaforo": colores.get(v, "red"),
                "saturada":       cola >= capacidad * 0.9,
                "layout":         LAYOUT[v],
            }

        hist = self.sim.monitor._historia
        salidos_min = int(sum(hist["salidos"][-60:])) if hist["salidos"] else 0
        t = self.sim.t
        hora_str = f"{(t//3600+7)%24:02d}:{(t%3600)//60:02d}"

        return {
            "t":     t,
            "hora":  hora_str,
            "vialidades": vialidades_frame,
            "semaforo":   sem_info,
            "metricas": {
                "cola_total":         sum(colas.values()),
                "salidos_ultimo_min": salidos_min,
                "n_bloqueadores":     getattr(self.sim, "_n_bloqueadores_activos", 0),
            },
        }

    def get_metrics_comparison(self, resumen_base, resumen_rl) -> dict:
        return {
            "baseline":   resumen_base or {},
            "rl":         resumen_rl or {},
            "disponible": resumen_base is not None and resumen_rl is not None,
        }

    def _colas_por_vialidad(self) -> Dict[str, int]:
        colas = self.sim.get_colas()
        return {
            via: sum(len(q) for cid, q in colas.items() if cid.startswith(pref))
            for via, pref in PREFIJOS.items()
        }
