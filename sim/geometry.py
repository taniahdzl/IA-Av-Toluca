"""
sim/geometry.py
───────────────
Representación física del cruce: carriles, longitudes y capacidades.
Los valores reales se cargan desde data/processed/geometria_carriles.json
una vez calibrados con los datos de campo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

GEOMETRIA_PATH = Path("data/processed/geometria_carriles.json")

# Movimientos posibles en cualquier carril
MOVIMIENTOS = ["recto", "izquierda", "derecha"]

# Vialidades del cruce
VIALIDADES = ["toluca_arriba", "toluca_abajo", "periferico_norte", "periferico_sur"]


@dataclass
class Carril:
    """
    Unidad mínima física del cruce.
    Representa un carril individual con su capacidad de almacenamiento.
    """
    id: str
    vialidad: str
    numero: int                          # número de carril (1 = más a la izquierda)
    movimientos_permitidos: List[str]    # subconjunto de MOVIMIENTOS
    longitud_almacenamiento_m: float     # metros disponibles antes de la línea de parada
    ancho_estimado_m: float = 3.5        # ancho estándar si no se mide en campo

    @property
    def capacidad_vehiculos(self) -> int:
        """
        Cuántos vehículos caben en cola en este carril.
        Usa longitud de vehículo promedio + separación en detención.
        Los valores vienen de geometria_carriles.json.
        """
        # TODO: leer longitud_vehiculo_promedio_m y separacion_detencion_m del JSON
        longitud_vehiculo = 4.5
        separacion = 1.0
        return int(self.longitud_almacenamiento_m / (longitud_vehiculo + separacion))

    def espacios_disponibles(self, ocupacion_actual: int) -> int:
        """Espacios libres en la cola de este carril."""
        return max(0, self.capacidad_vehiculos - ocupacion_actual)

    def esta_saturado(self, ocupacion_actual: int) -> bool:
        """True si la cola está al 90% o más de capacidad."""
        return ocupacion_actual >= self.capacidad_vehiculos * 0.9

    def acepta_movimiento(self, movimiento: str) -> bool:
        return movimiento in self.movimientos_permitidos


@dataclass
class GeometriaCruce:
    """
    Geometría completa del cruce cargada desde el JSON calibrado.
    Es inmutable durante la simulación.
    """
    carriles: Dict[str, List[Carril]] = field(default_factory=dict)
    longitud_vehiculo_m: float = 4.5
    separacion_detencion_m: float = 1.0

    # ── Carga ────────────────────────────────────────────────

    @classmethod
    def desde_json(cls, path: Path = GEOMETRIA_PATH) -> "GeometriaCruce":
        """
        Construye la geometría desde el JSON de datos de campo.
        Lanza un error descriptivo si el JSON aún tiene valores null
        (pendiente de calibración).
        """
        # TODO: implementar después de la visita de campo
        # 1. Leer el JSON
        # 2. Validar que no haya valores null
        # 3. Construir objetos Carril por cada entrada
        # 4. Retornar instancia de GeometriaCruce
        raise NotImplementedError(
            "Completa data/processed/geometria_carriles.json con los datos de campo "
            "antes de llamar a GeometriaCruce.desde_json()"
        )

    @classmethod
    def dummy(cls) -> "GeometriaCruce":
        """
        Geometría de placeholder para desarrollo y tests.
        Usa valores razonables hasta tener datos reales.
        """
        carriles: Dict[str, List[Carril]] = {}
        configuraciones = {
            "toluca_arriba":    [(1, ["recto", "izquierda"], 80.0),
                                 (2, ["recto", "derecha"],   80.0)],
            "toluca_abajo":     [(1, ["recto", "izquierda"], 80.0),
                                 (2, ["recto", "derecha"],   80.0)],
            "periferico_norte": [(1, ["recto", "derecha"],   60.0)],
            "periferico_sur":   [(1, ["recto", "izquierda"], 60.0)],
        }
        for vialidad, configs in configuraciones.items():
            carriles[vialidad] = [
                Carril(
                    id=f"{vialidad}_c{num}",
                    vialidad=vialidad,
                    numero=num,
                    movimientos_permitidos=movs,
                    longitud_almacenamiento_m=longitud,
                )
                for num, movs, longitud in configs
            ]
        return cls(carriles=carriles)

    # ── Consultas ────────────────────────────────────────────

    def carriles_de(self, vialidad: str) -> List[Carril]:
        """Todos los carriles de una vialidad."""
        return self.carriles.get(vialidad, [])

    def carril_por_id(self, carril_id: str) -> Carril | None:
        """Busca un carril por su id en todas las vialidades."""
        for carriles in self.carriles.values():
            for c in carriles:
                if c.id == carril_id:
                    return c
        return None

    def carriles_para_movimiento(self, vialidad: str, movimiento: str) -> List[Carril]:
        """Carriles de una vialidad que permiten un movimiento dado."""
        return [c for c in self.carriles_de(vialidad)
                if c.acepta_movimiento(movimiento)]

    def capacidad_total(self, vialidad: str) -> int:
        """Capacidad total en vehículos de todos los carriles de una vialidad."""
        return sum(c.capacidad_vehiculos for c in self.carriles_de(vialidad))

    def todos_los_carriles(self) -> List[Carril]:
        """Lista plana de todos los carriles del cruce."""
        return [c for carriles in self.carriles.values() for c in carriles]
