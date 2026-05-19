"""
sim/vehicle.py
──────────────
Vehículos individuales con perfiles de conductor nivel 3.

Cada vehículo tiene un perfil que determina:
  - Tiempo de reacción al verde
  - Disposición a cambiar de carril agresivamente
  - Probabilidad de bloquear la intersección
  - Brecha mínima aceptada para giros
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PerfilConductor(Enum):
    """
    Tres arquetipos de conductor calibrados con observación de campo (sección D).
    Los parámetros numéricos se ajustan en PerfilParams.
    """
    AGRESIVO  = "agresivo"
    NORMAL    = "normal"
    CAUTELOSO = "cauteloso"


@dataclass(frozen=True)
class PerfilParams:
    """
    Parámetros de comportamiento asociados a cada perfil.
    Todos los valores son defaults; se sobreescriben con datos de campo.
    """
    # Segundos que tarda en reaccionar al cambio a verde
    tiempo_reaccion_s: float

    # Probabilidad (0-1) de intentar cambiar de carril si el suyo está más lento
    prob_cambio_carril: float

    # Probabilidad (0-1) de avanzar bloqueando la intersección si no cabe
    prob_bloqueo_interseccion: float

    # Brecha mínima (en vehículos) que necesita para aceptar un giro
    brecha_minima_giro: int

    # Probabilidad de avanzar en amarillo tardío (> 1s después del cambio)
    prob_avanza_amarillo: float


# Parámetros por perfil — calibrar con sección D de la hoja de observación
PARAMS_POR_PERFIL: dict[PerfilConductor, PerfilParams] = {
    PerfilConductor.AGRESIVO: PerfilParams(
        tiempo_reaccion_s=0.5,
        prob_cambio_carril=0.6,
        prob_bloqueo_interseccion=0.3,
        brecha_minima_giro=1,
        prob_avanza_amarillo=0.5,
    ),
    PerfilConductor.NORMAL: PerfilParams(
        tiempo_reaccion_s=1.5,
        prob_cambio_carril=0.2,
        prob_bloqueo_interseccion=0.05,
        brecha_minima_giro=2,
        prob_avanza_amarillo=0.15,
    ),
    PerfilConductor.CAUTELOSO: PerfilParams(
        tiempo_reaccion_s=2.5,
        prob_cambio_carril=0.05,
        prob_bloqueo_interseccion=0.0,
        brecha_minima_giro=3,
        prob_avanza_amarillo=0.0,
    ),
}

# Distribución de perfiles — dato real de campo (Julián, mediodía 19/05/2026)
DISTRIBUCION_PERFILES = {
    PerfilConductor.AGRESIVO:  0.30,
    PerfilConductor.NORMAL:    0.60,
    PerfilConductor.CAUTELOSO: 0.10,
}


def perfil_aleatorio() -> PerfilConductor:
    """Elige un perfil según la distribución calibrada de campo."""
    perfiles = list(DISTRIBUCION_PERFILES.keys())
    pesos = list(DISTRIBUCION_PERFILES.values())
    return random.choices(perfiles, weights=pesos, k=1)[0]


@dataclass
class Vehiculo:
    """
    Representa un vehículo individual en la simulación.

    Ciclo de vida:
      1. Creado por SimuladorCruce al llegar al cruce
      2. Asignado a un carril por MarkovRouter
      3. Espera en cola hasta tener verde
      4. Puede intentar cambio de carril (nivel 3)
      5. Sale del cruce → t_salida registrado
    """
    id: int
    vialidad_origen: str
    carril_id: str                    # carril actual en la cola
    destino: str                      # "recto", "izquierda", "derecha"
    perfil: PerfilConductor
    t_llegada: int                    # segundo de simulación en que llegó

    t_salida: Optional[int] = None
    intentos_cambio_carril: int = 0   # para limitar cambios repetidos
    bloqueando_interseccion: bool = False

    @property
    def params(self) -> PerfilParams:
        return PARAMS_POR_PERFIL[self.perfil]

    @property
    def tiempo_espera(self) -> Optional[int]:
        """Segundos totales desde llegada hasta salida."""
        if self.t_salida is not None:
            return self.t_salida - self.t_llegada
        return None

    @property
    def ya_salio(self) -> bool:
        return self.t_salida is not None

    # ── Decisiones de comportamiento (nivel 3) ───────────────

    def quiere_cambiar_carril(self, cola_actual: int, cola_vecina: int) -> bool:
        """
        Decide si intenta cambiar al carril vecino.
        Depende del perfil y de qué tan diferente es la cola.
        """
        # TODO: implementar lógica nivel 3
        # Considerar:
        #   - prob_cambio_carril del perfil
        #   - diferencia entre cola_actual y cola_vecina (umbral mínimo)
        #   - límite de intentos_cambio_carril para evitar zigzag
        raise NotImplementedError

    def acepta_brecha(self, brecha_disponible: int) -> bool:
        """
        Decide si la brecha en el tráfico cruzado es suficiente para girar.
        Un conductor agresivo acepta brechas más pequeñas.
        """
        # TODO: implementar
        # brecha_disponible = número de vehículos de separación en el flujo cruzado
        raise NotImplementedError

    def intentara_bloquear(self) -> bool:
        """
        Decide si avanza aunque no quepa en la intersección.
        Solo conductores agresivos lo hacen con cierta probabilidad.
        """
        # TODO: implementar
        raise NotImplementedError

    def avanza_en_amarillo(self) -> bool:
        """
        Decide si cruza cuando el semáforo ya está en amarillo tardío.
        """
        # TODO: implementar
        raise NotImplementedError

    def __repr__(self) -> str:
        estado = f"esperando desde t={self.t_llegada}" if not self.ya_salio \
                 else f"salió en t={self.t_salida} (espera={self.tiempo_espera}s)"
        return (f"Vehiculo(id={self.id}, {self.perfil.value}, "
                f"{self.vialidad_origen}→{self.destino}, {estado})")
