"""
sim/metrics.py
──────────────
Recolección de métricas en tiempo real y generación de gráficas.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class MetricsMonitor:
    """
    Registra el estado del cruce en cada segundo de simulación
    y produce resúmenes y gráficas comparativas.

    Uso:
        monitor = MetricsMonitor()
        monitor.registrar(t=1, colas={...}, salidos=3, semaforo={...}, recompensa=5.0)
        monitor.resumen()
        monitor.plot("Semáforo fijo")
        monitor.exportar_csv("data/validation/baseline.csv")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._historia: Dict[str, List] = defaultdict(list)

    def registrar(self, t: int, colas: Dict[str, int], salidos: int,
                  semaforo: dict, recompensa: float):
        """
        Llamado por SimuladorCruce en cada step.

        Args:
            t:          segundo actual de simulación
            colas:      {carril_id: número de vehículos en espera}
            salidos:    vehículos que cruzaron en este segundo
            semaforo:   dict de get_estado() del Semaforo
            recompensa: valor calculado por la función de recompensa
        """
        self._historia["t"].append(t)
        self._historia["cola_total"].append(sum(colas.values()))
        self._historia["salidos"].append(salidos)
        self._historia["recompensa"].append(recompensa)
        self._historia["fase"].append(semaforo["fase_idx"])
        for carril_id, q in colas.items():
            self._historia[f"cola_{carril_id}"].append(q)

    # ── Análisis ─────────────────────────────────────────────

    def resumen(self) -> dict:
        """Estadísticas globales de la corrida."""
        if not self._historia["t"]:
            return {}
        return {
            "cola_promedio":       float(np.mean(self._historia["cola_total"])),
            "cola_maxima":         int(np.max(self._historia["cola_total"])),
            "total_salidos":       int(np.sum(self._historia["salidos"])),
            "recompensa_total":    float(np.sum(self._historia["recompensa"])),
            "recompensa_promedio": float(np.mean(self._historia["recompensa"])),
            "duracion_seg":        int(self._historia["t"][-1]),
        }

    def exportar_csv(self, path: str | Path):
        """Guarda la historia completa en un CSV para análisis externo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: implementar
        raise NotImplementedError

    # ── Visualización ────────────────────────────────────────

    def plot(self, titulo: str = "Simulación", guardar: bool = False,
             output_path: Optional[str] = None):
        """
        Dashboard de 4 gráficas:
          1. Cola total a lo largo del tiempo
          2. Cola por carril / vialidad
          3. Flujo de salida suavizado
          4. Fase del semáforo + recompensa acumulada
        """
        # TODO: implementar
        raise NotImplementedError

    @staticmethod
    def comparar(monitor_a: "MetricsMonitor", monitor_b: "MetricsMonitor",
                 label_a: str = "Baseline", label_b: str = "RL"):
        """
        Genera una gráfica comparativa lado a lado entre dos corridas.
        Útil para el reporte final: semáforo fijo vs agente entrenado.
        """
        # TODO: implementar
        raise NotImplementedError
