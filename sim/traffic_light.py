"""
sim/traffic_light.py
────────────────────
Semáforo con 3 fases independientes de duración continua.

El agente de RL asigna directamente los tiempos de verde:
  f1 ∈ [30, 120]s  — verde Av. Querétaro/Toluca (4 carriles fusionados)
  f2 ∈ [10,  90]s  — verde Lateral Norte
  f3 ∈ [10,  60]s  — verde Lateral Sur oeste (2 carriles rectos)

Ciclo total = f1 + f2 + f3 + 3 × amarillo(3s) — variable, sin límite fijo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

TIEMPO_AMARILLO = 3


@dataclass(frozen=True)
class Fase:
    nombre: str
    carriles_verde: List[str]
    duracion: int


class Semaforo:
    """
    Semáforo de 3 fases con duraciones continuas ajustables.

    El agente asigna directamente [f1, f2, f3] en cada decisión.
    No hay acciones discretas — el agente elige los segundos exactos.
    """

    # Rangos permitidos para cada fase
    F1_MIN, F1_MAX = 30, 120
    F2_MIN, F2_MAX = 10,  90
    F3_MIN, F3_MAX = 10,  60

    def __init__(self,
                 duracion_fase_1: int = 51,
                 duracion_fase_2: int = 30,
                 duracion_fase_3: int = 21,
                 carriles_fase_1: List[str] = None,
                 carriles_fase_2: List[str] = None,
                 carriles_fase_3: List[str] = None):
        self.duracion_fase_1 = duracion_fase_1
        self.duracion_fase_2 = duracion_fase_2
        self.duracion_fase_3 = duracion_fase_3
        self._carriles_fase_1 = carriles_fase_1 or []
        self._carriles_fase_2 = carriles_fase_2 or []
        self._carriles_fase_3 = carriles_fase_3 or []
        self._construir_fases()
        self.reset()

    # ── Construcción ─────────────────────────────────────────

    @classmethod
    def desde_calibracion(cls, flujos: dict, geometria) -> "Semaforo":
        tiempos = flujos["tiempos_semaforo_observados"]
        t1 = tiempos["verde_queretaro_toluca_s"]
        t_lateral = tiempos["verde_lateral_norte_s"]
        t2 = round(t_lateral * 0.55)
        t3 = t_lateral - t2
        c1, c2, c3 = [], [], []
        for c in geometria.carriles_de("queretaro_toluca"): c1.append(c.id)
        for c in geometria.carriles_de("lateral_norte"):    c2.append(c.id)
        for c in geometria.carriles_de("lateral_sur_oeste"): c3.append(c.id)
        return cls(duracion_fase_1=t1, duracion_fase_2=t2, duracion_fase_3=t3,
                   carriles_fase_1=c1, carriles_fase_2=c2, carriles_fase_3=c3)

    @classmethod
    def dummy(cls) -> "Semaforo":
        return cls(
            duracion_fase_1=51, duracion_fase_2=30, duracion_fase_3=21,
            carriles_fase_1=["que_tol_1","que_tol_2","que_tol_3","que_tol_4"],
            carriles_fase_2=["lat_nor_1","lat_nor_2","lat_nor_3","lat_nor_4"],
            carriles_fase_3=["lat_sur_1","lat_sur_2"],
        )

    def _construir_fases(self):
        self._fases: List[Fase] = [
            Fase("fase_1",    self._carriles_fase_1, self.duracion_fase_1),
            Fase("amarillo_1", [],                   TIEMPO_AMARILLO),
            Fase("fase_2",    self._carriles_fase_2, self.duracion_fase_2),
            Fase("amarillo_2", [],                   TIEMPO_AMARILLO),
            Fase("fase_3",    self._carriles_fase_3, self.duracion_fase_3),
            Fase("amarillo_3", [],                   TIEMPO_AMARILLO),
        ]

    # ── Control ──────────────────────────────────────────────

    def reset(self):
        self._fase_idx = 0
        self._t_en_fase = 0

    def tick(self) -> bool:
        self._t_en_fase += 1
        if self._t_en_fase >= self.fase_actual.duracion:
            self._t_en_fase = 0
            self._fase_idx = (self._fase_idx + 1) % len(self._fases)
            return True
        return False

    def set_duraciones(self, f1: float, f2: float, f3: float):
        """
        Asigna directamente las duraciones de las 3 fases.
        Llamado por el agente SAC en cada decisión.
        Los valores se clampean a los rangos permitidos.
        """
        self.duracion_fase_1 = int(np.clip(f1, self.F1_MIN, self.F1_MAX))
        self.duracion_fase_2 = int(np.clip(f2, self.F2_MIN, self.F2_MAX))
        self.duracion_fase_3 = int(np.clip(f3, self.F3_MIN, self.F3_MAX))
        self._construir_fases()

    # Compatibilidad con código anterior que use ajustar_duracion
    def ajustar_duracion(self, delta_fase_1=0, delta_fase_2=0, delta_fase_3=0):
        self.set_duraciones(
            self.duracion_fase_1 + delta_fase_1,
            self.duracion_fase_2 + delta_fase_2,
            self.duracion_fase_3 + delta_fase_3,
        )

    # ── Consultas ────────────────────────────────────────────

    @property
    def fase_actual(self) -> Fase:
        return self._fases[self._fase_idx]

    def carril_tiene_verde(self, carril_id: str) -> bool:
        return carril_id in self.fase_actual.carriles_verde

    def es_fase_1(self) -> bool: return self.fase_actual.nombre == "fase_1"
    def es_fase_2(self) -> bool: return self.fase_actual.nombre == "fase_2"
    def es_fase_3(self) -> bool: return self.fase_actual.nombre == "fase_3"
    def es_amarillo(self) -> bool: return self.fase_actual.nombre.startswith("amarillo")

    def zona_H_bloqueada(self) -> bool:
        return self.es_fase_2() or self.es_fase_3()

    def lateral_sur_cruzando(self) -> bool:
        return self.es_fase_3()

    def tiempo_restante(self) -> int:
        return self.fase_actual.duracion - self._t_en_fase

    @property
    def ciclo_total(self) -> int:
        return sum(f.duracion for f in self._fases)

    def get_estado(self) -> dict:
        return {
            "fase":                 self.fase_actual.nombre,
            "fase_idx":             self._fase_idx,
            "tiempo_restante":      self.tiempo_restante(),
            "es_amarillo":          self.es_amarillo(),
            "zona_H_bloqueada":     self.zona_H_bloqueada(),
            "lateral_sur_cruzando": self.lateral_sur_cruzando(),
            "duracion_fase_1":      self.duracion_fase_1,
            "duracion_fase_2":      self.duracion_fase_2,
            "duracion_fase_3":      self.duracion_fase_3,
            "ciclo_total":          self.ciclo_total,
        }
