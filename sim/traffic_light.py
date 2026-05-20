"""
sim/traffic_light.py
────────────────────
Semáforo con 3 fases independientes del cruce Av. Toluca × Periférico:

  Fase 1: Verde Av. Querétaro/Toluca
    - Zona H puede vaciarse hacia Lateral Sur este
    - Rojo: Lateral Norte, Lateral Sur oeste

  Amarillo 1→2 (3s)

  Fase 2: Verde Lateral Norte
    - Lateral Norte entra a zona H
    - Zona H bloqueada (Lateral Sur oeste cruza perp.)
    - Rojo: Querétaro/Toluca, Lateral Sur oeste

  Amarillo 2→3 (3s)

  Fase 3: Verde Lateral Sur oeste (rectos)
    - Solo los 2 carriles rectos cruzan hacia Lateral Sur este
    - Nadie puede incorporarse a Lateral Sur este desde zona H
    - Rojo: Querétaro/Toluca, Lateral Norte

  Amarillo 3→1 (3s)

El agente de RL ajusta t1, t2, t3 independientemente.
Ciclo base observado en campo: t1=51s, t2+t3=47s (no separados).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    Semáforo de 3 fases independientes.

    Acciones del agente (7 acciones):
      0 → no hacer nada
      1 → +10s fase_1   2 → -10s fase_1
      3 → +10s fase_2   4 → -10s fase_2
      5 → +10s fase_3   6 → -10s fase_3
    """

    DURACION_MIN = 10
    DURACION_MAX = 120

    def __init__(self,
                 duracion_fase_1: int = 51,
                 duracion_fase_2: int = 25,
                 duracion_fase_3: int = 22,
                 carriles_fase_1: List[str] = None,
                 carriles_fase_2: List[str] = None,
                 carriles_fase_3: List[str] = None):
        """
        Duraciones iniciales:
          fase_1 = 51s  (dato real de campo)
          fase_2 = 25s  (estimación: Lateral Norte, mitad del tiempo observado)
          fase_3 = 22s  (estimación: Lateral Sur oeste, otra mitad)
          Total ≈ 51 + 25 + 22 + 9s amarillo = 107s ≈ ciclo real de 111s
        """
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
        # Dividir el verde observado de lateral entre fase_2 y fase_3
        t_lateral = tiempos["verde_lateral_norte_s"]
        t2 = round(t_lateral * 0.55)   # 55% Lateral Norte
        t3 = t_lateral - t2            # 45% Lateral Sur oeste

        c1, c2, c3 = [], [], []
        for via in ["queretaro_toluca", "toluca_norte"]:
            for c in geometria.carriles_de(via): c1.append(c.id)
        for via in ["lateral_norte"]:
            for c in geometria.carriles_de(via): c2.append(c.id)
        for via in ["lateral_sur_oeste"]:
            for c in geometria.carriles_de(via): c3.append(c.id)

        return cls(duracion_fase_1=t1, duracion_fase_2=t2, duracion_fase_3=t3,
                   carriles_fase_1=c1, carriles_fase_2=c2, carriles_fase_3=c3)

    @classmethod
    def dummy(cls) -> "Semaforo":
        return cls(
            duracion_fase_1=51,
            duracion_fase_2=25,
            duracion_fase_3=22,
            carriles_fase_1=["que_tol_1","que_tol_2","que_tol_3","tol_nor_1"],
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

    def ajustar_duracion(self, delta_fase_1: int = 0,
                          delta_fase_2: int = 0,
                          delta_fase_3: int = 0):
        self.duracion_fase_1 = int(np.clip(
            self.duracion_fase_1 + delta_fase_1,
            self.DURACION_MIN, self.DURACION_MAX))
        self.duracion_fase_2 = int(np.clip(
            self.duracion_fase_2 + delta_fase_2,
            self.DURACION_MIN, self.DURACION_MAX))
        self.duracion_fase_3 = int(np.clip(
            self.duracion_fase_3 + delta_fase_3,
            self.DURACION_MIN, self.DURACION_MAX))
        self._construir_fases()

    # ── Consultas ────────────────────────────────────────────

    @property
    def fase_actual(self) -> Fase:
        return self._fases[self._fase_idx]

    def carril_tiene_verde(self, carril_id: str) -> bool:
        return carril_id in self.fase_actual.carriles_verde

    def es_fase_1(self) -> bool:
        return self.fase_actual.nombre == "fase_1"

    def es_fase_2(self) -> bool:
        return self.fase_actual.nombre == "fase_2"

    def es_fase_3(self) -> bool:
        return self.fase_actual.nombre == "fase_3"

    def es_amarillo(self) -> bool:
        return self.fase_actual.nombre.startswith("amarillo")

    def zona_H_bloqueada(self) -> bool:
        """
        True durante fase_2 (Lateral Norte entra a zona H pero
        Lateral Sur oeste no está cruzando todavía) y fase_3
        (Lateral Sur oeste cruza, nadie puede salir de zona H).
        """
        return self.es_fase_2() or self.es_fase_3()

    def lateral_sur_cruzando(self) -> bool:
        """True durante fase_3: los rectos de Lateral Sur oeste cruzan."""
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
