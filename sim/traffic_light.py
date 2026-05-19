"""
sim/traffic_light.py
────────────────────
Semáforo con fases configurables.
Los tiempos iniciales vienen de data/processed/flujos_calibrados.json
(sección A de la hoja de observación).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

TIEMPO_AMARILLO = 4   # segundos — fijo, no lo controla el agente


@dataclass(frozen=True)
class Fase:
    """
    Una fase del semáforo. Inmutable.
    """
    nombre: str
    carriles_verde: List[str]   # ids de carriles con luz verde
    duracion: int               # segundos


class Semaforo:
    """
    Controla el ciclo de fases del semáforo.

    Fases base:
      0 → verde Av. Toluca
      1 → amarillo
      2 → verde Periférico
      3 → amarillo

    El agente de RL llama a ajustar_duracion() para modificar
    los tiempos de verde. El amarillo es siempre fijo.

    Uso:
        sem = Semaforo.desde_calibracion()
        sem.tick()                          # avanza 1 segundo
        sem.carril_tiene_verde("tol_arr_1") # True/False
        sem.get_estado()                    # dict para el agente
    """

    DURACION_MIN = 15    # segundos mínimos de verde
    DURACION_MAX = 120   # segundos máximos de verde

    def __init__(self, duracion_toluca: int = 45, duracion_periferico: int = 55,
                 carriles_toluca: List[str] = None, carriles_periferico: List[str] = None):
        self.duracion_toluca = duracion_toluca
        self.duracion_periferico = duracion_periferico
        self._carriles_toluca = carriles_toluca or []
        self._carriles_periferico = carriles_periferico or []
        self._construir_fases()
        self.reset()

    # ── Construcción ─────────────────────────────────────────

    @classmethod
    def desde_calibracion(cls, flujos: dict, geometria) -> "Semaforo":
        """
        Construye el semáforo con los tiempos observados en campo
        y los ids de carriles reales de la geometría.

        Args:
            flujos:    dict cargado de flujos_calibrados.json
            geometria: instancia de GeometriaCruce
        """
        # TODO: implementar después de la visita de campo
        # 1. Leer tiempos de flujos["tiempos_semaforo_observados"]
        # 2. Extraer ids de carriles por vialidad de geometria
        # 3. Retornar instancia
        raise NotImplementedError

    @classmethod
    def dummy(cls) -> "Semaforo":
        """Semáforo con valores placeholder para desarrollo y tests."""
        return cls(
            duracion_toluca=45,
            duracion_periferico=55,
            carriles_toluca=["tol_arr_1", "tol_arr_2", "tol_aba_1", "tol_aba_2"],
            carriles_periferico=["per_nor_1", "per_sur_1"],
        )

    def _construir_fases(self):
        """Reconstruye la lista de fases con los tiempos actuales."""
        self._fases: List[Fase] = [
            Fase("verde_toluca",     self._carriles_toluca,     self.duracion_toluca),
            Fase("amarillo_1",       [],                        TIEMPO_AMARILLO),
            Fase("verde_periferico", self._carriles_periferico, self.duracion_periferico),
            Fase("amarillo_2",       [],                        TIEMPO_AMARILLO),
        ]

    # ── Control ──────────────────────────────────────────────

    def reset(self):
        """Reinicia al inicio del ciclo."""
        self._fase_idx = 0
        self._t_en_fase = 0

    def tick(self) -> bool:
        """
        Avanza el semáforo 1 segundo.
        Retorna True si cambió de fase en este tick.
        """
        self._t_en_fase += 1
        if self._t_en_fase >= self.fase_actual.duracion:
            self._t_en_fase = 0
            self._fase_idx = (self._fase_idx + 1) % len(self._fases)
            return True
        return False

    def ajustar_duracion(self, delta_toluca: int = 0, delta_periferico: int = 0):
        """
        Modifica los tiempos de verde. Llamado por el agente de RL.
        Los tiempos se clampean al rango [DURACION_MIN, DURACION_MAX].
        No interrumpe el ciclo en curso.
        """
        self.duracion_toluca = int(
            np.clip(self.duracion_toluca + delta_toluca, self.DURACION_MIN, self.DURACION_MAX)
        )
        self.duracion_periferico = int(
            np.clip(self.duracion_periferico + delta_periferico, self.DURACION_MIN, self.DURACION_MAX)
        )
        self._construir_fases()

    # ── Consultas ────────────────────────────────────────────

    @property
    def fase_actual(self) -> Fase:
        return self._fases[self._fase_idx]

    def carril_tiene_verde(self, carril_id: str) -> bool:
        return carril_id in self.fase_actual.carriles_verde

    def tiempo_restante(self) -> int:
        """Segundos que faltan para que cambie la fase actual."""
        return self.fase_actual.duracion - self._t_en_fase

    @property
    def ciclo_total(self) -> int:
        return sum(f.duracion for f in self._fases)

    def es_amarillo(self) -> bool:
        return self.fase_actual.nombre.startswith("amarillo")

    def get_estado(self) -> dict:
        """Dict de estado para el agente de RL y el monitor."""
        return {
            "fase":               self.fase_actual.nombre,
            "fase_idx":           self._fase_idx,
            "tiempo_restante":    self.tiempo_restante(),
            "es_amarillo":        self.es_amarillo(),
            "duracion_toluca":    self.duracion_toluca,
            "duracion_periferico": self.duracion_periferico,
            "ciclo_total":        self.ciclo_total,
        }
