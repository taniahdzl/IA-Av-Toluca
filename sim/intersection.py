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

# Vialidades con semáforo en el cruce
# Flujos independientes sin semáforo (retorno zona H, vuelta continua)
# no se modelan como vialidades de entrada.
VIALIDADES = [
    "queretaro_toluca",   # Av. Querétaro diagonal → Av. Toluca (3 carriles, fase 1)
    "toluca_norte",       # Av. Toluca norte 1 carril (fase 1)
    "lateral_norte",      # Lateral Norte ← oeste, 4 carriles (fase 2)
    "lateral_sur_oeste",  # Lateral Sur oeste → este, 2 carriles rectos (fase 2)
]

# Fases del semáforo — qué vialidades tienen verde en cada fase
FASE_VERDE: Dict[str, List[str]] = {
    "fase_1": ["queretaro_toluca", "toluca_norte"],
    "fase_2": ["lateral_norte", "lateral_sur_oeste"],
}

# Flujos vehiculares por periodo del día (veh/hora)
# Periodo 2 (mediodía) calibrado con datos de campo de Julián (19/05/2026)
# Periodos 0,1,3,4 estimados con factores de escala típicos CDMX
FLUJOS_DEFAULT: Dict[str, List[float]] = {
    "queretaro_toluca":  [390,  2140, 2165, 1930, 755],
    "toluca_norte":      [130,   715,  722,  643, 252],
    "lateral_norte":     [480,  2390, 2400, 2135, 840],
    "lateral_sur_oeste": [400,  2000, 2011, 1790, 700],
}

# Tasa de descarga (veh/segundo de verde)
# Datos reales de campo sección E — mediodía 19/05/2026
TASA_DESCARGA_DEFAULT: Dict[str, float] = {
    "queretaro_toluca":  1.313,  # proporcional desde 105 veh/60s × 0.75 (3 de 4 carriles)
    "toluca_norte":      0.292,  # proporcional desde 105 veh/60s × 0.25 / 1 carril
    "lateral_norte":     1.567,  # campo directo: 94 veh/60s
    "lateral_sur_oeste": 1.317,  # campo directo: 79 veh/60s (2 carriles)
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
        self._n_bloqueadores_activos: int = 0

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
            "n_bloqueadores": self._n_bloqueadores_activos,
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
        # [0-3] Cola normalizada por vialidad
        colas_norm = []
        for v in VIALIDADES:
            capacidad = max(self.geometria.capacidad_total(v), 1)
            cola = sum(
                len(self._colas[c.id])
                for c in self.geometria.carriles_de(v)
                if c.id in self._colas
            )
            colas_norm.append(min(cola / capacidad, 1.0))

        # [4] Tiempo restante normalizado (máximo posible = 120s)
        t_restante_norm = self.semaforo.tiempo_restante() / 120.0

        # [5] Fase actual normalizada
        n_fases = len(self.semaforo._fases)
        fase_norm = self.semaforo._fase_idx / max(n_fases - 1, 1)

        # [6] Periodo del día normalizado
        periodo_norm = self._get_periodo() / 4.0

        return np.array(
            colas_norm + [t_restante_norm, fase_norm, periodo_norm],
            dtype=np.float32
        )

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
        Usa distribución de Poisson: cada segundo puede llegar 0, 1 o más vehículos
        con probabilidad proporcional al flujo horario.
        """
        nuevos: List[Vehiculo] = []

        for vialidad in VIALIDADES:
            # Lambda de Poisson: vehículos esperados por segundo
            flujo_hora = self.flujos[vialidad][periodo]
            lam = flujo_hora / 3600.0

            # Cuántos llegan este segundo
            n_llegadas = np.random.poisson(lam)
            if n_llegadas == 0:
                continue

            carriles = self.geometria.carriles_de(vialidad)
            ocupacion = {c.id: len(self._colas[c.id]) for c in carriles}

            for _ in range(n_llegadas):
                # Elegir destino con la cadena de Markov
                destino = self.router.elegir_destino(vialidad, periodo)

                # Asignar carril según destino y ocupación actual
                candidatos = [
                    c for c in carriles
                    if c.acepta_movimiento(destino)
                    and not c.esta_saturado(ocupacion.get(c.id, 0))
                ]
                if not candidatos:
                    # Si todos los carriles están saturados, usar el menos lleno
                    candidatos = sorted(
                        [c for c in carriles if c.acepta_movimiento(destino)],
                        key=lambda c: ocupacion.get(c.id, 0)
                    )
                if not candidatos:
                    continue  # no hay carril válido para este movimiento

                # Preferir el carril con menos vehículos
                carril = min(candidatos, key=lambda c: ocupacion.get(c.id, 0))

                vehiculo = Vehiculo(
                    id=self._vehiculo_id,
                    vialidad_origen=vialidad,
                    carril_id=carril.id,
                    destino=destino,
                    perfil=perfil_aleatorio(),
                    t_llegada=self.t,
                )
                self._vehiculo_id += 1
                self._colas[carril.id].append(vehiculo)
                ocupacion[carril.id] = ocupacion.get(carril.id, 0) + 1
                nuevos.append(vehiculo)

        return nuevos

    def _procesar_cambios_carril(self):
        """
        Comportamiento nivel 3: vehículos al frente de la cola que intentan
        cambiar a un carril vecino menos congestionado dentro de su misma vialidad.
        """
        for vialidad in VIALIDADES:
            carriles = self.geometria.carriles_de(vialidad)
            if len(carriles) < 2:
                continue
            for i, carril in enumerate(carriles):
                if not self._colas[carril.id]:
                    continue
                primer_veh = self._colas[carril.id][0]
                cola_actual = len(self._colas[carril.id])
                vecinos = []
                if i > 0:
                    vecinos.append(carriles[i - 1])
                if i < len(carriles) - 1:
                    vecinos.append(carriles[i + 1])
                for vecino in vecinos:
                    if not vecino.acepta_movimiento(primer_veh.destino):
                        continue
                    cola_vecina = len(self._colas[vecino.id])
                    if not primer_veh.quiere_cambiar_carril(cola_actual, cola_vecina):
                        continue
                    self._colas[carril.id].popleft()
                    primer_veh.carril_id = vecino.id
                    primer_veh.intentos_cambio_carril += 1
                    self._colas[vecino.id].appendleft(primer_veh)
                    break

    def _descargar_colas(self) -> List[Vehiculo]:
        """
        Saca vehículos de los carriles con verde.

        Para cada carril con luz verde:
          1. Samplea cuántos vehículos salen este segundo (Poisson con tasa de descarga)
          2. El primer vehículo de la cola necesita su tiempo_reaccion_s —
             si lleva menos tiempo esperando que su tiempo de reacción, no sale
          3. Registra t_salida y acumula en self._vehiculos_salidos
        """
        salidos: List[Vehiculo] = []

        for vialidad in VIALIDADES:
            tasa = self.tasa_descarga[vialidad]
            carriles = self.geometria.carriles_de(vialidad)

            for carril in carriles:
                if not self.semaforo.carril_tiene_verde(carril.id):
                    continue
                if not self._colas[carril.id]:
                    continue

                # Cuántos pueden salir este segundo según la tasa de descarga
                n_posibles = np.random.poisson(tasa / len(carriles))
                n_posibles = min(n_posibles, len(self._colas[carril.id]))

                for _ in range(n_posibles):
                    if not self._colas[carril.id]:
                        break

                    primer_veh = self._colas[carril.id][0]

                    # Respetar tiempo de reacción al verde
                    # El vehículo necesita haber esperado al menos
                    # tiempo_reaccion_s segundos desde que arrancó el verde
                    t_espera = self.t - primer_veh.t_llegada
                    if t_espera < primer_veh.params.tiempo_reaccion_s:
                        break  # el resto de la cola tampoco puede salir aún

                    vehiculo = self._colas[carril.id].popleft()
                    vehiculo.t_salida = self.t
                    self._vehiculos_salidos.append(vehiculo)
                    salidos.append(vehiculo)

        return salidos

    def _procesar_bloqueos(self):
        """
        Modela dos fenómenos de bloqueo del cruce:

        1. Avance en amarillo: conductores agresivos de queretaro_toluca y
           toluca_norte cruzan aunque el semáforo ya cambió a amarillo.

        2. Bloqueo de zona H (fase_2): conductores agresivos que se metieron
           a la intersección en el último instante de fase_1 quedan físicamente
           bloqueando lateral_sur_oeste, reduciendo su tasa de descarga efectiva.
        """
        # ── Fenómeno 1: avance en amarillo ───────────────────
        # Amarillo después de fase_1 (fase_idx == 1)
        if self.semaforo.es_amarillo() and self.semaforo._fase_idx == 1:
            for vialidad in ["queretaro_toluca", "toluca_norte"]:
                for carril in self.geometria.carriles_de(vialidad):
                    if not self._colas[carril.id]:
                        continue
                    primer_veh = self._colas[carril.id][0]
                    if primer_veh.avanza_en_amarillo():
                        veh = self._colas[carril.id].popleft()
                        veh.t_salida = self.t
                        self._vehiculos_salidos.append(veh)

        # ── Fenómeno 2: bloqueo de intersección en fase_2 ────
        if self.semaforo.zona_H_bloqueada():
            n_bloqueadores = 0
            for vialidad in ["queretaro_toluca", "toluca_norte"]:
                for carril in self.geometria.carriles_de(vialidad):
                    if not self._colas[carril.id]:
                        continue
                    primer_veh = self._colas[carril.id][0]
                    if not primer_veh.bloqueando_interseccion:
                        if primer_veh.intentara_bloquear():
                            primer_veh.bloqueando_interseccion = True
                    if primer_veh.bloqueando_interseccion:
                        n_bloqueadores += 1

            # Cada bloqueador reduce ~25% la tasa efectiva de lateral_sur_oeste
            # Esto se registra en el info dict para que el agente lo observe
            if n_bloqueadores > 0:
                self._n_bloqueadores_activos = n_bloqueadores
            else:
                self._n_bloqueadores_activos = 0

        # ── Limpiar bloqueadores al terminar fase_2 ───────────
        if self.semaforo.es_amarillo() and self.semaforo._fase_idx == 3:
            for vialidad in ["queretaro_toluca", "toluca_norte"]:
                for carril in self.geometria.carriles_de(vialidad):
                    for veh in self._colas[carril.id]:
                        veh.bloqueando_interseccion = False
            self._n_bloqueadores_activos = 0

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
