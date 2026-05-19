"""
rl/evaluate.py
──────────────
Compara el semáforo fijo (baseline) contra el agente PPO entrenado.
Genera la tabla y gráficas principales del reporte final.

Uso:
    # Comparar con simulador dummy
    python rl/evaluate.py --modelo rl/models/best_model.zip --dummy

    # Comparar con simulador calibrado
    python rl/evaluate.py --modelo rl/models/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def evaluar_baseline(n_episodios: int = 3, usar_dummy: bool = True) -> dict:
    """
    Corre N episodios con semáforo fijo (sin agente).
    Retorna métricas promediadas.
    """
    from rl.environment import CruceEnv

    env = CruceEnv() if usar_dummy else CruceEnv.desde_calibracion()
    metricas = _acumular_metricas(env, modelo=None, n_episodios=n_episodios)
    print(f"  Baseline ({n_episodios} episodios): {_resumen_str(metricas)}")
    return metricas


def evaluar_agente(path_modelo: str, n_episodios: int = 3,
                   usar_dummy: bool = True) -> dict:
    """
    Corre N episodios con el agente PPO entrenado.
    Retorna métricas promediadas.
    """
    from stable_baselines3 import PPO
    from rl.environment import CruceEnv

    env = CruceEnv() if usar_dummy else CruceEnv.desde_calibracion()
    modelo = PPO.load(path_modelo, env=env)
    metricas = _acumular_metricas(env, modelo=modelo, n_episodios=n_episodios)
    print(f"  Agente RL ({n_episodios} episodios): {_resumen_str(metricas)}")
    return metricas


def comparar_e_imprimir(metricas_base: dict, metricas_rl: dict):
    """
    Imprime la tabla comparativa de métricas clave.
    """
    print("\n" + "=" * 65)
    print(f"{'MÉTRICA':<30} {'BASELINE':>12} {'RL':>12} {'MEJORA':>10}")
    print("-" * 65)

    pares = [
        ("Cola promedio (veh)",     "cola_promedio"),
        ("Cola máxima (veh)",       "cola_maxima"),
        ("Vehículos salidos",       "total_salidos"),
        ("Tiempo espera prom (s)",  "espera_promedio"),
        ("Recompensa total",        "recompensa_total"),
    ]

    for nombre, key in pares:
        va = metricas_base.get(key, 0)
        vb = metricas_rl.get(key, 0)
        if va != 0:
            mejora = (vb - va) / abs(va) * 100
            signo = "↑" if mejora > 0 else "↓"
            print(f"{nombre:<30} {va:>12.1f} {vb:>12.1f} {mejora:>+9.1f}% {signo}")
        else:
            print(f"{nombre:<30} {va:>12.1f} {vb:>12.1f} {'N/A':>10}")
    print("=" * 65)


def generar_graficas(metricas_base: dict, metricas_rl: dict,
                     output_dir: str = "data/validation"):
    """
    Genera las gráficas comparativas para el reporte final.
    Guarda en output_dir/comparacion_baseline_vs_rl.png
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Semáforo fijo vs Agente RL — Cruce Av. Toluca × Periférico",
                 fontsize=13, fontweight="bold")

    colores = ["#4472C4", "#70AD47"]
    labels = ["Semáforo fijo", "Agente RL"]

    # Cola promedio y máxima
    vals_cola = [
        [metricas_base["cola_promedio"], metricas_rl["cola_promedio"]],
        [metricas_base["cola_maxima"],   metricas_rl["cola_maxima"]],
    ]
    x = np.arange(2)
    axs[0].bar(x - 0.2, [v[0] for v in vals_cola], 0.35, label="Promedio", color=colores)
    axs[0].bar(x + 0.2, [v[1] for v in vals_cola], 0.35, label="Máxima",
               color=[c + "88" for c in colores])
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_title("Cola (vehículos)")
    axs[0].set_ylabel("Vehículos")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3, axis="y")

    # Vehículos salidos
    axs[1].bar(labels, [metricas_base["total_salidos"], metricas_rl["total_salidos"]],
               color=colores)
    axs[1].set_title("Vehículos que cruzaron")
    axs[1].set_ylabel("Total por episodio")
    axs[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([metricas_base["total_salidos"], metricas_rl["total_salidos"]]):
        axs[1].text(i, v + 10, f"{v:,.0f}", ha="center", fontsize=9)

    # Tiempo de espera promedio
    axs[2].bar(labels,
               [metricas_base.get("espera_promedio", 0),
                metricas_rl.get("espera_promedio", 0)],
               color=colores)
    axs[2].set_title("Tiempo de espera promedio")
    axs[2].set_ylabel("Segundos por vehículo")
    axs[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_dir) / "comparacion_baseline_vs_rl.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfica guardada: {output_path}")
    plt.show()


# ── Helpers internos ──────────────────────────────────────────

def _acumular_metricas(env, modelo, n_episodios: int) -> dict:
    """Corre N episodios y promedia las métricas del monitor."""
    resultados = []
    for ep in range(n_episodios):
        obs, _ = env.reset()
        terminated = False
        recompensa_total = 0.0

        while not terminated:
            if modelo is not None:
                accion, _ = modelo.predict(obs, deterministic=True)
            else:
                accion = 0   # baseline: no hacer nada
            obs, reward, terminated, _, _ = env.step(int(accion))
            recompensa_total += reward

        resumen = env.sim.monitor.resumen()
        resumen["recompensa_total"] = recompensa_total

        # Tiempo de espera promedio desde vehículos salidos
        esperas = [v.tiempo_espera for v in env.sim._vehiculos_salidos
                   if v.tiempo_espera is not None]
        resumen["espera_promedio"] = float(np.mean(esperas)) if esperas else 0.0
        resultados.append(resumen)

    # Promediar sobre episodios
    keys = resultados[0].keys()
    return {k: float(np.mean([r[k] for r in resultados])) for k in keys}


def _resumen_str(metricas: dict) -> str:
    return (f"cola_prom={metricas['cola_promedio']:.1f} | "
            f"salidos={metricas['total_salidos']:.0f} | "
            f"espera={metricas.get('espera_promedio', 0):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar y comparar baseline vs agente RL")
    parser.add_argument("--modelo", type=str, required=True,
                        help="Path al modelo PPO entrenado (.zip)")
    parser.add_argument("--episodios", type=int, default=3,
                        help="Número de episodios de evaluación (default: 3)")
    parser.add_argument("--dummy", action="store_true",
                        help="Usar simulador dummy")
    args = parser.parse_args()

    print("── Evaluando baseline ──────────────────────────────────")
    m_base = evaluar_baseline(n_episodios=args.episodios, usar_dummy=args.dummy)

    print("\n── Evaluando agente RL ─────────────────────────────────")
    m_rl = evaluar_agente(args.modelo, n_episodios=args.episodios, usar_dummy=args.dummy)

    comparar_e_imprimir(m_base, m_rl)
    generar_graficas(m_base, m_rl)
