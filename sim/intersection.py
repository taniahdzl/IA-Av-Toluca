"""
sim/intersection.py
───────────────────
Motor principal de la simulación. Une geometría, vehículos,
semáforo y router en un loop de tiempo discreto.

Cada llamada a step() avanza 1 segundo de simulación.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim.geometry import GeometriaCruce, VIALIDADES
from sim.metrics import MetricsMonitor
from sim.router import MarkovRouter
from sim.traffic_light import Semaforo
from sim.vehicle import Vehiculo, PerfilConductor, perfil_aleatorio

# Flujos vehiculares por periodo del día (veh/hora)
# Llenados con datos de campo desde flujos_calibrados.json
FLUJOS_DEFAULT: Dict[str, List[float]] = {
    "toluca_arriba":    [120, 1400, 800, 1200, 400],
    "toluca_abajo":     [100, 1200, 750, 1300, 350],
    "periferico_norte": [200, 1600, 900, 1400, 500],
    "periferico_sur":   [180, 1500, 850, 1350, 450],
}

# Tasa de descarga (veh/segundo de verde) — calibrar con sección E
TASA_DESCARGA_DEFAULT: Dict[str, float] = {
    "toluca_arriba":    0.5,
    "toluca_abajo":     0.5,
    "periferico_norte": 0.6,
    "periferico_sur":   0.6,
}

# Vector de estado para el agente de RL
# [cola_vialidad_0..3 norm, t_restante norm, fase_idx norm, periodo norm]
ESTADO_DIM = 7


class SimuladorCruce:
    """
    Motor principal de la simulación del cruce Toluca × Periférico.

    Uso básico (sin agente, semáforo fijo):
        sim = SimuladorCruce.dummy()
        sim.reset()
        monitor = sim.run(duracion_seg=3600)
        monitor.plot("Baseline")

    Uso con agente de RL (desde rl/environment.py):
        sim = SimuladorCruce.dummy()
        sim.reset()
        estado = sim.get_estado()
        estado, recompensa, info = sim.step(accion=1)
    """

    def __init__(self, geometria: GeometriaCruce, semaforo: Semaforo,
                 router: MarkovRouter, flujos: Dict[str, List[float]],
                 tasa_descarga: Dict[str, float], seed: Optional[int] = 42):
        self.geometria = geometria
        self.semaforo = semaforo
        self.router = router
        self.flujos = flujos
        self.tasa_descarga = tasa_descarga
        self.monitor = MetricsMonitor()
        self._seed = seed
        self.reset()

    # ── Construcción ─────────────────────────────────────────

    @classmethod
    def desde_calibracion(cls, seed: int = 42) -> "SimuladorCruce":
        """
        Construye el simulador completo desde los JSONs calibrados.
        Usar una vez que se tengan los datos de campo.
        """
        # TODO: implementar
        # 1. Cargar GeometriaCruce.desde_json()
        # 2. Cargar flujos_calibrados.json
        # 3. Construir Semaforo.desde_calibracion()
        # 4. Construir MarkovRouter.desde_json()
        # 5. Retornar instancia con todos los parámetros reales
        raise NotImplementedError

    @classmethod
    def dummy(cls, seed: int = 42) -> "SimuladorCruce":
        """Simulador con valores placeholder para desarrollo y tests."""
        return cls(
            geometria=GeometriaCruce.dummy(),
            semaforo=Semaforo.dummy(),
            router=MarkovRouter.dummy(),
            flujos=FLUJOS_DEFAULT,
            tasa_descarga=TASA_DESCARGA_DEFAULT,
            seed=seed,
        )

    # ── Control ──────────────────────────────────────────────

    def reset(self):
        """Reinicia la simulación al segundo 0."""
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        self.t = 0
        self._vehiculo_id = 0
        self.semaforo.reset()
        self.monitor.reset()

        # Colas por carril: {carril_id: deque[Vehiculo]}
        self._colas: Dict[str, deque] = {
            c.id: deque()
            for c in self.geometria.todos_los_carriles()
        }
        self._vehiculos_salidos: List[Vehiculo] = []

    def step(self, accion: Optional[int] = None) -> Tuple[np.ndarray, float, dict]:
        """
        Avanza la simulación 1 segundo.

        Acciones para el agente de RL:
            None / 0 → no hacer nada
            1        → +10s verde Toluca
            2        → -10s verde Toluca
            3        → +10s verde Periférico
            4        → -10s verde Periférico

        Returns:
            estado     : np.ndarray de dimensión ESTADO_DIM
            recompensa : float
            info       : dict con métricas del step para logging
        """
        # 1. Aplicar acción
        self._aplicar_accion(accion)

        # 2. Avanzar tiempo
        self.t += 1
        periodo = self._get_periodo()

        # 3. Generar llegadas (Poisson)
        nuevos = self._generar_llegadas(periodo)

        # 4. Comportamiento nivel 3: cambios de carril
        self._procesar_cambios_carril()

        # 5. Tick del semáforo
        self.semaforo.tick()

        # 6. Descargar colas con verde
        salidos = self._descargar_colas()

        # 7. Manejar bloqueos de intersección
        self._procesar_bloqueos()

        # 8. Calcular recompensa (delegada a rl/reward.py en el entorno gym)
        # Aquí calculamos una recompensa básica como fallback
        recompensa = self._recompensa_basica(salidos)

        # 9. Registrar métricas
        colas_snapshot = {cid: len(q) for cid, q in self._colas.items()}
        self.monitor.registrar(
            t=self.t,
            colas=colas_snapshot,
            salidos=len(salidos),
            semaforo=self.semaforo.get_estado(),
            recompensa=recompensa,
        )

        info = {
            "t":            self.t,
            "periodo":      periodo,
            "colas":        colas_snapshot,
            "salidos_step": len(salidos),
            "nuevos_step":  len(nuevos),
            "semaforo":     self.semaforo.get_estado(),
        }

        return self.get_estado(), recompensa, info

    def run(self, duracion_seg: int = 3600, verbose: bool = True) -> MetricsMonitor:
        """
        Corre la simulación sin agente (semáforo fijo) durante N segundos.
        Retorna el monitor con todas las métricas para análisis posterior.
        """
        self.reset()
        for _ in range(duracion_seg):
            _, _, info = self.step()
            if verbose and self.t % 600 == 0:
                self._imprimir_estado(info)

        if verbose:
            resumen = self.monitor.resumen()
            print(f"\n✓ Simulación terminada ({duracion_seg}s)")
            print(f"  Vehículos generados : {self._vehiculo_id}")
            print(f"  Vehículos salidos   : {resumen['total_salidos']}")
            print(f"  Cola promedio       : {resumen['cola_promedio']:.1f} veh")
            print(f"  Cola máxima         : {resumen['cola_maxima']} veh")

        return self.monitor

    # ── Estado para el agente ────────────────────────────────

    def get_estado(self) -> np.ndarray:
        """
        Vector de observación de dimensión ESTADO_DIM para el agente de RL.

        [0-3] cola normalizada por vialidad (suma de carriles / capacidad total)
        [4]   tiempo restante en fase actual (normalizado a [0,1])
        [5]   índice de fase actual (normalizado a [0,1])
        [6]   periodo del día (normalizado a [0,1])
        """
        # TODO: implementar
        # Usar self.geometria.capacidad_total(v) como denominador de normalización
        raise NotImplementedError

    def get_colas(self) -> Dict[str, deque]:
        """Devuelve las colas actuales (referencia, no copia)."""
        return self._colas

    # ── Internos ─────────────────────────────────────────────

    def _get_periodo(self) -> int:
        """Periodo del día según el segundo actual (asume inicio a las 7am)."""
        hora = (self.t // 3600 + 7) % 24
        if hora < 6:   return 0
        if hora < 10:  return 1
        if hora < 14:  return 2
        if hora < 20:  return 3
        return 4

    def _aplicar_accion(self, accion: Optional[int]):
        """Traduce la acción discreta del agente en ajuste del semáforo."""
        ajustes = {1: (10, 0), 2: (-10, 0), 3: (0, 10), 4: (0, -10)}
        if accion and accion in ajustes:
            dt, dp = ajustes[accion]
            self.semaforo.ajustar_duracion(dt, dp)

    def _generar_llegadas(self, periodo: int) -> List[Vehiculo]:
        """
        Genera nuevos vehículos según el flujo del periodo actual.
        Usa distribución de Poisson para variabilidad realista.
        """
        # TODO: implementar
        # 1. Para cada vialidad, calcular lambda = flujo_hora / 3600
        # 2. Samplear np.random.poisson(lambda) llegadas
        # 3. Para cada llegada: elegir destino con router, asignar carril,
        #    crear Vehiculo con perfil_aleatorio(), agregar a la cola
        raise NotImplementedError

    def _procesar_cambios_carril(self):
        """
        Comportamiento nivel 3: vehículos que intentan cambiar de carril.
        Solo actúa en los primeros vehículos de cada cola (los que pueden ver).
        """
        # TODO: implementar
        # Para cada vialidad con más de un carril:
        #   - Revisar el primer vehículo de cada carril
        #   - Llamar vehiculo.quiere_cambiar_carril(cola_actual, cola_vecina)
        #   - Si sí: mover el vehículo al carril vecino si hay espacio
        raise NotImplementedError

    def _descargar_colas(self) -> List[Vehiculo]:
        """
        Saca vehículos de los carriles con verde.
        Respeta el tiempo de reacción del perfil de conductor.
        """
        # TODO: implementar
        # Para cada carril con verde:
        #   - Samplear np.random.poisson(tasa_descarga) salidas
        #   - Considerar tiempo_reaccion_s del primer vehículo en cola
        #   - Registrar t_salida en el vehículo
        #   - Agregar a self._vehiculos_salidos
        raise NotImplementedError

    def _procesar_bloqueos(self):
        """
        Maneja vehículos que bloquean la intersección (nivel 3).
        Un vehículo que bloquea impide el avance de los carriles cruzados.
        """
        # TODO: implementar
        # - Identificar vehículos con bloqueando_interseccion=True
        # - Reducir tasa de descarga de los carriles afectados este segundo
        raise NotImplementedError

    def _recompensa_basica(self, salidos: List[Vehiculo]) -> float:
        """
        Recompensa de fallback. La función real viene de rl/reward.py.
        Penaliza colas y premia salidas.
        """
        cola_total = sum(len(q) for q in self._colas.values())
        return -cola_total * 0.1 + len(salidos) * 2.0

    def _imprimir_estado(self, info: dict):
        sem = info["semaforo"]
        print(f"  t={self.t:5d}s | "
              f"Cola total: {sum(info['colas'].values()):3d} veh | "
              f"Salidos: {info['salidos_step']:2d} | "
              f"Fase: {sem['fase']} ({sem['tiempo_restante']}s restantes)")
