"""
tests/test_sim.py
─────────────────
Tests unitarios para los módulos de sim/.
Correr con: pytest tests/ -v
"""

import pytest
import numpy as np

from sim.geometry import Carril, GeometriaCruce
from sim.vehicle import Vehiculo, PerfilConductor, PerfilParams, PARAMS_POR_PERFIL, perfil_aleatorio
from sim.router import MarkovRouter
from sim.traffic_light import Semaforo
from sim.intersection import SimuladorCruce, ESTADO_DIM, VIALIDADES


# ─────────────────────────────────────────────
# GeometriaCruce
# ─────────────────────────────────────────────

class TestGeometria:

    def test_dummy_crea_todos_los_carriles(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.todos_los_carriles()) > 0

    def test_vialidades_reales(self):
        geo = GeometriaCruce.dummy()
        for v in VIALIDADES:
            assert len(geo.carriles_de(v)) > 0, f"{v} no tiene carriles"

    def test_queretaro_tiene_3_carriles(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.carriles_de("queretaro_toluca")) == 3

    def test_toluca_norte_tiene_1_carril(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.carriles_de("toluca_norte")) == 1

    def test_lateral_norte_tiene_4_carriles(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.carriles_de("lateral_norte")) == 4

    def test_lateral_sur_tiene_2_carriles(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.carriles_de("lateral_sur_oeste")) == 2

    def test_capacidad_vehiculos_razonable(self):
        geo = GeometriaCruce.dummy()
        for c in geo.todos_los_carriles():
            assert c.capacidad_vehiculos > 0
            assert c.capacidad_vehiculos < 200

    def test_acepta_movimiento(self):
        carril = Carril(
            id="test_1", vialidad="queretaro_toluca", numero=1,
            movimientos_permitidos=["recto"],
            longitud_almacenamiento_m=60.0
        )
        assert carril.acepta_movimiento("recto")
        assert not carril.acepta_movimiento("izquierda")

    def test_esta_saturado(self):
        carril = Carril(
            id="test_2", vialidad="toluca_norte", numero=1,
            movimientos_permitidos=["recto"],
            longitud_almacenamiento_m=20.0
        )
        capacidad = carril.capacidad_vehiculos
        assert not carril.esta_saturado(0)
        assert carril.esta_saturado(capacidad)

    def test_desde_json_carga_correctamente(self):
        geo = GeometriaCruce.desde_json()
        assert len(geo.todos_los_carriles()) > 0
        for v in VIALIDADES:
            assert len(geo.carriles_de(v)) > 0


# ─────────────────────────────────────────────
# Semáforo
# ─────────────────────────────────────────────

class TestSemaforo:

    def test_ciclo_completo_vuelve_a_fase_0(self):
        sem = Semaforo.dummy()
        ciclo = sem.ciclo_total
        for _ in range(ciclo):
            sem.tick()
        assert sem._fase_idx == 0

    def test_fase_1_verde_queretaro_rojo_lateral(self):
        sem = Semaforo.dummy()
        # Al inicio debe estar en fase_1
        assert sem.es_fase_1()
        assert sem.carril_tiene_verde("que_tol_1")
        assert not sem.carril_tiene_verde("lat_nor_1")

    def test_fase_2_verde_lateral_rojo_queretaro(self):
        sem = Semaforo.dummy()
        # Avanzar hasta fase_2
        for _ in range(sem.duracion_fase_1 + 3 + 1):  # fase1 + amarillo + 1
            sem.tick()
        assert sem.es_fase_2()
        assert sem.carril_tiene_verde("lat_nor_1")
        assert not sem.carril_tiene_verde("que_tol_1")

    def test_zona_H_bloqueada_en_fase_2(self):
        sem = Semaforo.dummy()
        assert not sem.zona_H_bloqueada()
        for _ in range(sem.duracion_fase_1 + 3 + 1):
            sem.tick()
        assert sem.zona_H_bloqueada()

    def test_tiempos_reales_de_campo(self):
        sem = Semaforo.dummy()
        assert sem.duracion_fase_1 == 51
        assert sem.duracion_fase_2 == 47

    def test_ajustar_duracion_respeta_limites(self):
        sem = Semaforo.dummy()
        sem.ajustar_duracion(delta_fase_1=1000)
        assert sem.duracion_fase_1 == Semaforo.DURACION_MAX
        sem.ajustar_duracion(delta_fase_1=-1000)
        assert sem.duracion_fase_1 == Semaforo.DURACION_MIN

    def test_tiempo_restante_decrece(self):
        sem = Semaforo.dummy()
        t1 = sem.tiempo_restante()
        sem.tick()
        assert sem.tiempo_restante() == t1 - 1

    def test_reset_vuelve_a_inicio(self):
        sem = Semaforo.dummy()
        for _ in range(10):
            sem.tick()
        sem.reset()
        assert sem._fase_idx == 0
        assert sem._t_en_fase == 0


# ─────────────────────────────────────────────
# MarkovRouter
# ─────────────────────────────────────────────

class TestMarkovRouter:

    def test_dummy_carga_correctamente(self):
        router = MarkovRouter.dummy()
        assert router is not None

    def test_vialidades_reales_presentes(self):
        router = MarkovRouter.dummy()
        for v in VIALIDADES:
            assert v in router.matriz

    def test_destino_valido_queretaro(self):
        router = MarkovRouter.dummy()
        for _ in range(20):
            destino = router.elegir_destino("queretaro_toluca", periodo=1)
            assert destino in ["recto", "izquierda", "derecha"]

    def test_queretaro_siempre_recto(self):
        router = MarkovRouter.dummy()
        # queretaro_toluca tiene prob 1.0 de recto
        for _ in range(20):
            assert router.elegir_destino("queretaro_toluca", periodo=2) == "recto"

    def test_lateral_norte_mayoria_recto(self):
        # En periodo 2: recto=0.5862, izq=0.2414, der=0.1724
        router = MarkovRouter.dummy()
        destinos = [router.elegir_destino("lateral_norte", periodo=2) for _ in range(200)]
        pct_recto = destinos.count("recto") / len(destinos)
        assert 0.45 < pct_recto < 0.72  # tolerancia estadística

    def test_probabilidades_suman_uno(self):
        router = MarkovRouter.dummy()
        for vialidad, periodos in router.matriz.items():
            for periodo, probs in periodos.items():
                assert abs(sum(probs) - 1.0) < 1e-4, \
                    f"{vialidad} periodo {periodo}: suman {sum(probs)}"

    def test_validacion_rechaza_probs_incorrectas(self):
        with pytest.raises(AssertionError):
            MarkovRouter({"queretaro_toluca": {"0": [0.5, 0.5, 0.5]}})

    def test_desde_json_carga_correctamente(self):
        router = MarkovRouter.desde_json()
        for v in VIALIDADES:
            assert v in router.matriz


# ─────────────────────────────────────────────
# Vehículo
# ─────────────────────────────────────────────

class TestVehiculo:

    def _vehiculo(self, perfil=PerfilConductor.NORMAL) -> Vehiculo:
        return Vehiculo(
            id=1, vialidad_origen="queretaro_toluca",
            carril_id="que_tol_1", destino="recto",
            perfil=perfil, t_llegada=0
        )

    def test_tiempo_espera_none_si_no_salio(self):
        v = self._vehiculo()
        assert v.tiempo_espera is None

    def test_tiempo_espera_calculado_correctamente(self):
        v = self._vehiculo()
        v.t_salida = 45
        assert v.tiempo_espera == 45

    def test_perfiles_tienen_params(self):
        for perfil in PerfilConductor:
            assert perfil in PARAMS_POR_PERFIL

    def test_perfil_aleatorio_retorna_perfil_valido(self):
        for _ in range(20):
            p = perfil_aleatorio()
            assert isinstance(p, PerfilConductor)

    def test_agresivo_acepta_brecha_minima(self):
        v = self._vehiculo(PerfilConductor.AGRESIVO)
        assert v.acepta_brecha(1)   # agresivo acepta brecha de 1
        assert not v.acepta_brecha(0)

    def test_cauteloso_necesita_brecha_mayor(self):
        v = self._vehiculo(PerfilConductor.CAUTELOSO)
        assert not v.acepta_brecha(2)  # cauteloso necesita al menos 3
        assert v.acepta_brecha(3)

    def test_cauteloso_no_bloquea(self):
        v = self._vehiculo(PerfilConductor.CAUTELOSO)
        # Cauteloso nunca bloquea (prob=0.0)
        for _ in range(20):
            assert not v.intentara_bloquear()

    def test_no_cambia_carril_sin_diferencia(self):
        v = self._vehiculo()
        # Si la cola vecina es igual, no cambia
        for _ in range(20):
            assert not v.quiere_cambiar_carril(cola_actual=5, cola_vecina=5)

    def test_limite_cambios_carril(self):
        v = self._vehiculo(PerfilConductor.AGRESIVO)
        v.intentos_cambio_carril = 2
        # Con 2 intentos ya no debe cambiar
        for _ in range(20):
            assert not v.quiere_cambiar_carril(cola_actual=10, cola_vecina=1)


# ─────────────────────────────────────────────
# SimuladorCruce (integración)
# ─────────────────────────────────────────────

class TestSimuladorCruce:

    def test_dummy_crea_correctamente(self):
        sim = SimuladorCruce.dummy()
        assert sim is not None

    def test_reset_limpia_estado(self):
        sim = SimuladorCruce.dummy()
        sim.reset()
        assert sim.t == 0
        assert sim._vehiculo_id == 0
        for cola in sim._colas.values():
            assert len(cola) == 0

    def test_estado_tiene_dimension_correcta(self):
        sim = SimuladorCruce.dummy()
        sim.reset()
        estado = sim.get_estado()
        assert estado.shape == (ESTADO_DIM,)
        assert np.all(estado >= 0.0)
        assert np.all(estado <= 1.0)

    def test_step_avanza_tiempo(self):
        sim = SimuladorCruce.dummy()
        sim.reset()
        sim.step()
        assert sim.t == 1

    def test_run_genera_vehiculos(self):
        sim = SimuladorCruce.dummy()
        monitor = sim.run(duracion_seg=60, verbose=False)
        resumen = monitor.resumen()
        assert resumen["duracion_seg"] == 60
        assert sim._vehiculo_id > 0

    def test_run_salen_vehiculos(self):
        sim = SimuladorCruce.dummy()
        monitor = sim.run(duracion_seg=120, verbose=False)
        assert monitor.resumen()["total_salidos"] > 0

    def test_cola_crece_en_rojo(self):
        # Durante fase_2, queretaro_toluca debe acumular cola
        sim = SimuladorCruce.dummy()
        sim.reset()
        # Avanzar hasta fase_2
        while not sim.semaforo.es_fase_2():
            sim.step()
        cola_inicio = sum(
            len(sim._colas[c.id])
            for c in sim.geometria.carriles_de("queretaro_toluca")
        )
        for _ in range(10):
            sim.step()
        cola_fin = sum(
            len(sim._colas[c.id])
            for c in sim.geometria.carriles_de("queretaro_toluca")
        )
        assert cola_fin >= cola_inicio  # cola no baja en rojo

    def test_desde_calibracion_funciona(self):
        sim = SimuladorCruce.desde_calibracion()
        sim.reset()
        estado, _, info = sim.step()
        assert estado.shape == (ESTADO_DIM,)
        assert sim.semaforo.duracion_fase_1 == 51
        assert sim.semaforo.duracion_fase_2 == 47
