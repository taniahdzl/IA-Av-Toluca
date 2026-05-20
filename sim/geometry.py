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
        Construye la geometría desde el JSON calibrado con datos de campo.
        Usa valores reales donde estén disponibles; null usa el dummy como fallback.
        """
        import json
        data = json.loads(path.read_text())

        longitud_veh = data.get("longitud_vehiculo_promedio_m") or 4.1
        separacion   = data.get("separacion_detencion_m") or 1.0

        # Fallback: valores dummy para campos null
        dummy = cls.dummy()
        dummy_por_vialidad = {
            v: {c.numero: c for c in cs}
            for v, cs in dummy.carriles.items()
        }

        carriles: Dict[str, List[Carril]] = {}
        for vialidad, info_via in data["vialidades"].items():
            lista = []
            for c_data in info_via["carriles"]:
                num = c_data["numero"]
                longitud = c_data.get("longitud_almacenamiento_m")
                if longitud is None:
                    fb = dummy_por_vialidad.get(vialidad, {}).get(num)
                    longitud = fb.longitud_almacenamiento_m if fb else 60.0
                ancho = c_data.get("ancho_estimado_m") or 3.5
                lista.append(Carril(
                    id=c_data["id"],
                    vialidad=vialidad,
                    numero=num,
                    movimientos_permitidos=c_data["movimientos_permitidos"],
                    longitud_almacenamiento_m=float(longitud),
                    ancho_estimado_m=float(ancho),
                ))
            carriles[vialidad] = lista

        return cls(
            carriles=carriles,
            longitud_vehiculo_m=longitud_veh,
            separacion_detencion_m=separacion,
        )

    @classmethod
    def dummy(cls) -> "GeometriaCruce":
        """
        Geometría real del cruce basada en HTML source of truth
        y datos de campo (longitudes pendientes de medición completa).
        """
        carriles: Dict[str, List[Carril]] = {}
        # Longitudes calibradas con OSM (19.340477, -99.203100, r=400m)
        # + observación de campo. Capacidad = longitud / (4.1m veh + 1.0m sep)
        configuraciones = {
            # Av. Querétaro diagonal → Av. Toluca: 3 carriles
            # Longitud: segmentos OSM ~417m / 2 sentidos = 180m
            "queretaro_toluca": [
                (1, ["recto"], 180.0),
                (2, ["recto"], 180.0),
                (3, ["recto"], 180.0),
            ],
            # Av. Toluca norte: 1 carril
            # Longitud: segmento tertiary OSM = 52m
            "toluca_norte": [
                (1, ["recto", "izquierda"], 52.0),
            ],
            # Lateral Norte (Blvd. López Mateos) ← oeste: 4 carriles
            # 3 carriles OSM + 1 que se junta físicamente = 4 efectivos
            # Longitud: ~325m / 2 sentidos = 160m
            "lateral_norte": [
                (1, ["izquierda"],           160.0),
                (2, ["izquierda", "recto"],  160.0),
                (3, ["izquierda", "recto"],  160.0),
                (4, ["recto"],               160.0),
            ],
            # Lateral Sur oeste: solo los 2 carriles rectos con semáforo
            # Retorno (→ Lateral Norte) y vuelta continua (→ Av. Toluca sur)
            # son flujos independientes sin semáforo, no se modelan aquí
            # Longitud: mismo tramo López Mateos = 160m
            "lateral_sur_oeste": [
                (1, ["recto"], 160.0),
                (2, ["recto"], 160.0),
            ],
        }
        for vialidad, configs in configuraciones.items():
            carriles[vialidad] = [
                Carril(
                    id=f"{_id_prefijo(vialidad)}_{num}",
                    vialidad=vialidad,
                    numero=num,
                    movimientos_permitidos=movs,
                    longitud_almacenamiento_m=longitud,
                )
                for num, movs, longitud in configs
            ]
        return cls(carriles=carriles, longitud_vehiculo_m=4.1)

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


def _id_prefijo(vialidad: str) -> str:
    """Convierte nombre de vialidad a prefijo corto para ids de carril."""
    prefijos = {
        "queretaro_toluca":  "que_tol",
        "toluca_norte":      "tol_nor",
        "lateral_norte":     "lat_nor",
        "lateral_sur_oeste": "lat_sur",
    }
    return prefijos.get(vialidad, vialidad[:7])
