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

        # Columnas: t + todas las series registradas
        columnas = ["t", "cola_total", "salidos", "recompensa", "fase"]
        carril_cols = [k for k in self._historia if k.startswith("cola_") and k != "cola_total"]
        columnas += sorted(carril_cols)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            n = len(self._historia["t"])
            for i in range(n):
                fila = {col: self._historia[col][i] for col in columnas
                        if col in self._historia}
                writer.writerow(fila)

        print(f"  CSV exportado: {path}  ({n} filas)")

    # ── Visualización ────────────────────────────────────────

    def plot(self, titulo: str = "Simulación", guardar: bool = False,
             output_path: Optional[str] = None):
        """
        Dashboard de 4 gráficas:
          1. Cola total a lo largo del tiempo
          2. Cola por carril
          3. Flujo de salida suavizado (ventana 60s)
          4. Fase del semáforo + recompensa acumulada
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        if not self._historia["t"]:
            print("  Sin datos para graficar.")
            return

        t = self._historia["t"]
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(titulo, fontsize=14, fontweight="bold")

        # 1. Cola total
        axs[0, 0].plot(t, self._historia["cola_total"], color="#3b82f6", linewidth=0.8)
        axs[0, 0].set_title("Cola total en el cruce")
        axs[0, 0].set_xlabel("Tiempo (s)")
        axs[0, 0].set_ylabel("Vehículos en cola")
        axs[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x//3600)+7:02d}:{int((x%3600)//60):02d}"))
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Cola por carril (agrupado por vialidad)
        colores_via = {
            "tol_arr": "#3b82f6", "tol_aba": "#06b6d4",
            "per_nor": "#f59e0b", "per_sur": "#ef4444",
        }
        labels_via = {
            "tol_arr": "Tol. →Periférico", "tol_aba": "Tol. →Revolución",
            "per_nor": "Periférico Norte",  "per_sur": "Periférico Sur",
        }
        carril_cols = [k for k in self._historia if k.startswith("cola_") and k != "cola_total"]
        vialidad_series: Dict[str, List] = defaultdict(lambda: [0] * len(t))
        for col in carril_cols:
            prefijo = col[5:12]   # "cola_tol_arr_1" → "tol_arr"
            prefijo = "_".join(col.split("_")[1:3])
            for i, v in enumerate(self._historia[col]):
                vialidad_series[prefijo][i] += v
        for prefijo, serie in vialidad_series.items():
            color = colores_via.get(prefijo, "#888")
            label = labels_via.get(prefijo, prefijo)
            axs[0, 1].plot(t, serie, label=label, color=color, linewidth=0.7)
        axs[0, 1].set_title("Cola por vialidad")
        axs[0, 1].set_xlabel("Tiempo (s)")
        axs[0, 1].set_ylabel("Vehículos")
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x//3600)+7:02d}:{int((x%3600)//60):02d}"))
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Flujo de salida suavizado
        ventana = 60
        salidos = np.array(self._historia["salidos"], dtype=float)
        suavizado = np.convolve(salidos, np.ones(ventana) / ventana, mode="same")
        axs[1, 0].plot(t, suavizado, color="#4ade80", linewidth=0.8)
        axs[1, 0].set_title("Flujo de salida (veh/s, ventana 60s)")
        axs[1, 0].set_xlabel("Tiempo (s)")
        axs[1, 0].set_ylabel("Vehículos / segundo")
        axs[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x//3600)+7:02d}:{int((x%3600)//60):02d}"))
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Fase + recompensa acumulada
        ax4a = axs[1, 1]
        ax4b = ax4a.twinx()
        ax4a.plot(t, self._historia["fase"], color="#a78bfa",
                  linewidth=0.5, alpha=0.7, label="Fase (0-3)")
        recomp_acum = np.cumsum(self._historia["recompensa"])
        ax4b.plot(t, recomp_acum, color="#4ade80", linewidth=0.8,
                  linestyle="--", label="Recompensa acum.")
        ax4a.set_title("Fase del semáforo y recompensa acumulada")
        ax4a.set_xlabel("Tiempo (s)")
        ax4a.set_ylabel("Fase (0–3)", color="#a78bfa")
        ax4b.set_ylabel("Recompensa acumulada", color="#4ade80")
        ax4a.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x//3600)+7:02d}:{int((x%3600)//60):02d}"))
        ax4a.grid(True, alpha=0.3)

        plt.tight_layout()

        if guardar:
            out = output_path or f"dashboard_{titulo.replace(' ', '_').lower()}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Gráfica guardada: {out}")

        plt.show()

    @staticmethod
    def comparar(monitor_a: "MetricsMonitor", monitor_b: "MetricsMonitor",
                 label_a: str = "Semáforo fijo", label_b: str = "Agente RL",
                 guardar: bool = False, output_path: Optional[str] = None):
        """
        Gráfica comparativa lado a lado entre dos corridas.
        Útil para el reporte final.
        """
        import matplotlib.pyplot as plt

        res_a = monitor_a.resumen()
        res_b = monitor_b.resumen()

        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Comparación: {label_a} vs {label_b}",
                     fontsize=13, fontweight="bold")

        colores = ["#3b82f6", "#4ade80"]
        labels = [label_a, label_b]

        # Cola promedio y máxima
        x = np.arange(2)
        axs[0].bar(x - 0.2, [res_a["cola_promedio"], res_b["cola_promedio"]],
                   0.35, label="Promedio", color=colores)
        axs[0].bar(x + 0.2, [res_a["cola_maxima"], res_b["cola_maxima"]],
                   0.35, label="Máxima", color=[c + "88" for c in colores])
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(labels, fontsize=9)
        axs[0].set_title("Cola (vehículos)")
        axs[0].legend(fontsize=8)
        axs[0].grid(True, alpha=0.3, axis="y")

        # Vehículos salidos
        bars = axs[1].bar(labels, [res_a["total_salidos"], res_b["total_salidos"]],
                          color=colores)
        axs[1].set_title("Vehículos que cruzaron")
        axs[1].set_ylabel("Total por episodio")
        axs[1].grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, [res_a["total_salidos"], res_b["total_salidos"]]):
            axs[1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5, f"{int(v):,}",
                        ha="center", fontsize=9)

        # Recompensa acumulada a lo largo del tiempo
        t_a = monitor_a._historia["t"]
        t_b = monitor_b._historia["t"]
        ra = np.cumsum(monitor_a._historia["recompensa"])
        rb = np.cumsum(monitor_b._historia["recompensa"])
        axs[2].plot(t_a, ra, color=colores[0], label=label_a, linewidth=1)
        axs[2].plot(t_b, rb, color=colores[1], label=label_b, linewidth=1)
        axs[2].set_title("Recompensa acumulada")
        axs[2].set_xlabel("Tiempo (s)")
        axs[2].legend(fontsize=8)
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if guardar:
            out = output_path or "comparacion_baseline_vs_rl.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Gráfica guardada: {out}")

        plt.show()
