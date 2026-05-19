"""
sim/traffic_light.py
────────────────────
Semáforo con 3 fases reales del cruce Av. Toluca × Periférico:

  Fase 1 (51s): Verde Av. Querétaro/Toluca arriba
    - Zona H puede vaciarse hacia Av. Toluca sur y Lateral Sur este
    - Rojo: Lateral Norte, Lateral Sur oeste

  Amarillo 1→2 (3s)

  Fase 2 (47s): Verde Lateral Norte + Lateral Sur oeste
    - CRÍTICO: Lateral Sur oeste cruza perpendicularmente →
      zona H queda bloqueada aunque Lateral Norte tenga verde
    - Rojo: Av. Querétaro/Toluca

  Amarillo 2→1 (3s)

El agente de RL ajusta las duraciones de fase_1 y fase_2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

TIEMPO_AMARILLO = 3   # segundos — fijo, dato real de campo


@dataclass(frozen=True)
class Fase:
    """Una fase del semáforo. Inmutable."""
    nombre: str
    carriles_verde: List[str]
    duracion: int


class Semaforo:
    """
    Controla el ciclo de 4 fases del semáforo real del cruce.

    Fases:
      0 → fase_1:    verde Av. Querétaro/Toluca (ajustable por RL)
      1 → amarillo_1: transición (fijo 3s)
      2 → fase_2:    verde Lateral Norte + Lateral Sur oeste (ajustable por RL)
      3 → amarillo_2: transición (fijo 3s)

    Duraciones iniciales calibradas con datos de campo:
      fase_1 = 51s, fase_2 = 47s, ciclo total = 104s + 6s amarillo = 110s
    """

    DURACION_MIN = 15
    DURACION_MAX = 120

    def __init__(self, duracion_fase_1: int = 51, duracion_fase_2: int = 47,
                 carriles_fase_1: List[str] = None,
                 carriles_fase_2: List[str] = None):
        self.duracion_fase_1 = duracion_fase_1
        self.duracion_fase_2 = duracion_fase_2
        self._carriles_fase_1 = carriles_fase_1 or []
        self._carriles_fase_2 = carriles_fase_2 or []
        self._construir_fases()
        self.reset()

    # ── Construcción ─────────────────────────────────────────

    @classmethod
    def desde_calibracion(cls, flujos: dict, geometria) -> "Semaforo":
        """
        Construye el semáforo desde flujos_calibrados.json con carriles reales.
        """
        tiempos = flujos["tiempos_semaforo_observados"]
        t1 = tiempos["verde_queretaro_toluca_s"]
        t2 = tiempos["verde_lateral_norte_s"]

        carriles_f1, carriles_f2 = [], []
        for via in ["queretaro_toluca", "toluca_norte"]:
            for c in geometria.carriles_de(via):
                carriles_f1.append(c.id)
        for via in ["lateral_norte", "lateral_sur_oeste"]:
            for c in geometria.carriles_de(via):
                carriles_f2.append(c.id)

        return cls(
            duracion_fase_1=t1,
            duracion_fase_2=t2,
            carriles_fase_1=carriles_f1,
            carriles_fase_2=carriles_f2,
        )

    @classmethod
    def dummy(cls) -> "Semaforo":
        """Semáforo con las fases reales observadas en campo (mediodía 19/05/2026)."""
        return cls(
            duracion_fase_1=51,
            duracion_fase_2=47,
            carriles_fase_1=["que_tol_1", "que_tol_2", "que_tol_3", "tol_nor_1"],
            carriles_fase_2=["lat_nor_1", "lat_nor_2", "lat_nor_3", "lat_nor_4",
                             "lat_sur_1", "lat_sur_2"],
        )

    def _construir_fases(self):
        self._fases: List[Fase] = [
            Fase("fase_1",    self._carriles_fase_1, self.duracion_fase_1),
            Fase("amarillo_1", [],                   TIEMPO_AMARILLO),
            Fase("fase_2",    self._carriles_fase_2, self.duracion_fase_2),
            Fase("amarillo_2", [],                   TIEMPO_AMARILLO),
        ]

    # ── Control ──────────────────────────────────────────────

    def reset(self):
        self._fase_idx = 0
        self._t_en_fase = 0

    def tick(self) -> bool:
        """Avanza 1 segundo. Retorna True si cambió de fase."""
        self._t_en_fase += 1
        if self._t_en_fase >= self.fase_actual.duracion:
            self._t_en_fase = 0
            self._fase_idx = (self._fase_idx + 1) % len(self._fases)
            return True
        return False

    def ajustar_duracion(self, delta_fase_1: int = 0, delta_fase_2: int = 0):
        """
        Modifica los tiempos de verde. Llamado por el agente de RL.
        Clampea al rango [DURACION_MIN, DURACION_MAX].
        """
        self.duracion_fase_1 = int(
            np.clip(self.duracion_fase_1 + delta_fase_1,
                    self.DURACION_MIN, self.DURACION_MAX)
        )
        self.duracion_fase_2 = int(
            np.clip(self.duracion_fase_2 + delta_fase_2,
                    self.DURACION_MIN, self.DURACION_MAX)
        )
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

    def es_amarillo(self) -> bool:
        return self.fase_actual.nombre.startswith("amarillo")

    def zona_H_bloqueada(self) -> bool:
        """
        True durante fase_2: Lateral Sur oeste cruza perpendicularmente,
        impidiendo que los vehículos en zona H puedan salir.
        """
        return self.es_fase_2()

    def tiempo_restante(self) -> int:
        return self.fase_actual.duracion - self._t_en_fase

    @property
    def ciclo_total(self) -> int:
        return sum(f.duracion for f in self._fases)

    def get_estado(self) -> dict:
        return {
            "fase":            self.fase_actual.nombre,
            "fase_idx":        self._fase_idx,
            "tiempo_restante": self.tiempo_restante(),
            "es_amarillo":     self.es_amarillo(),
            "zona_H_bloqueada": self.zona_H_bloqueada(),
            "duracion_fase_1": self.duracion_fase_1,
            "duracion_fase_2": self.duracion_fase_2,
            "ciclo_total":     self.ciclo_total,
        }
