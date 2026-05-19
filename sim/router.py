"""
sim/router.py
─────────────
Cadena de Markov para asignar destino y carril a cada vehículo al llegar.
La matriz de transición se carga desde data/processed/matriz_markov.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from sim.geometry import Carril, GeometriaCruce

MARKOV_PATH = Path("data/processed/matriz_markov.json")

MOVIMIENTOS = ["recto", "izquierda", "derecha"]

# Periodos del día
PERIODOS = {
    0: "madrugada",
    1: "mañana_pico",
    2: "mediodia",
    3: "tarde_pico",
    4: "noche",
}


class MarkovRouter:
    """
    Decide el destino de cada vehículo al llegar al cruce
    y lo asigna al carril correcto.

    Uso:
        router = MarkovRouter.desde_json()
        destino = router.elegir_destino("toluca_arriba", periodo=1)
        carril  = router.asignar_carril(vehiculo, geometria)
    """

    def __init__(self, matriz: Dict[str, Dict[str, List[float]]]):
        """
        matriz: { vialidad -> { str(periodo) -> [p_recto, p_izq, p_der] } }
        """
        self._validar(matriz)
        self.matriz = matriz

    # ── Construcción ─────────────────────────────────────────

    @classmethod
    def desde_json(cls, path: Path = MARKOV_PATH) -> "MarkovRouter":
        """
        Carga la matriz desde el JSON calibrado con datos de campo.
        Lanza error descriptivo si aún hay valores null.
        """
        # TODO: implementar después de la visita de campo
        # 1. Leer JSON
        # 2. Verificar que no haya nulls
        # 3. Convertir keys de periodo a int
        # 4. Retornar instancia
        raise NotImplementedError(
            "Completa data/processed/matriz_markov.json con los conteos "
            "de giros (sección C.2 de la hoja de observación)"
        )

    @classmethod
    def dummy(cls) -> "MarkovRouter":
        """
        Matriz calibrada con datos de campo de Julián (mediodía 19/05/2026).
        Periodos distintos al 2 son estimaciones.
        """
        # [p_recto, p_izquierda, p_derecha]
        matriz: Dict[str, Dict[str, List[float]]] = {
            # Av. Querétaro/Toluca: todos van recto hacia zona H
            "queretaro_toluca": {str(p): [1.0, 0.0, 0.0] for p in range(5)},
            # Av. Toluca norte: igual, 1 carril recto
            "toluca_norte":     {str(p): [1.0, 0.0, 0.0] for p in range(5)},
            # Lateral Norte: mayoría izquierda (zona H), minoría recto
            # Datos reales periodo 2: 17 recto, 7 izq, 5 der de 29 total
            "lateral_norte": {
                "0": [0.50, 0.35, 0.15],
                "1": [0.45, 0.40, 0.15],
                "2": [0.5862, 0.2414, 0.1724],
                "3": [0.45, 0.40, 0.15],
                "4": [0.50, 0.35, 0.15],
            },
            # Lateral Sur oeste: recto=cruzar a lateral sur este, izq=retorno
            # Datos reales periodo 2: 7 recto, 11 izq, 0 der de 18 total
            "lateral_sur_oeste": {
                "0": [0.40, 0.60, 0.0],
                "1": [0.35, 0.65, 0.0],
                "2": [0.3889, 0.6111, 0.0],
                "3": [0.35, 0.65, 0.0],
                "4": [0.40, 0.60, 0.0],
            },
        }
        return cls(matriz)

    # ── Decisiones ───────────────────────────────────────────

    def elegir_destino(self, vialidad: str, periodo: int) -> str:
        """
        Elige el movimiento del vehículo según la distribución de Markov.

        Args:
            vialidad: vialidad de origen del vehículo
            periodo:  índice del periodo del día (0-4)

        Returns:
            "recto", "izquierda" o "derecha"
        """
        probs = self.matriz[vialidad][str(periodo)]
        return random.choices(MOVIMIENTOS, weights=probs, k=1)[0]

    def asignar_carril(self, vialidad: str, destino: str,
                       geometria: GeometriaCruce,
                       ocupacion: Dict[str, int]) -> Carril | None:
        """
        Asigna el carril más adecuado para el movimiento del vehículo.

        Estrategia:
          1. Filtrar carriles que permiten el movimiento
          2. Entre esos, elegir el de menor ocupación actual
          3. Si hay empate, elegir aleatoriamente

        Args:
            vialidad:  vialidad de origen
            destino:   movimiento elegido ("recto", "izquierda", "derecha")
            geometria: geometría del cruce
            ocupacion: dict {carril_id: número de vehículos en cola}

        Returns:
            Carril asignado, o None si no hay carril disponible
        """
        # TODO: implementar
        # Considerar también la capacidad máxima — no asignar a carriles saturados
        raise NotImplementedError

    # ── Validación ───────────────────────────────────────────

    def _validar(self, matriz: Dict[str, Dict[str, List[float]]]):
        """Verifica que las probabilidades sumen 1.0 en cada fila."""
        for vialidad, periodos in matriz.items():
            for periodo, probs in periodos.items():
                total = sum(probs)
                assert abs(total - 1.0) < 1e-6, (
                    f"Probabilidades de {vialidad} periodo {periodo} "
                    f"suman {total:.4f}, deben sumar 1.0"
                )
