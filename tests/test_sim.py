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
from sim.intersection import SimuladorCruce, ESTADO_DIM


# ─────────────────────────────────────────────
# GeometriaCruce
# ─────────────────────────────────────────────

class TestGeometria:

    def test_dummy_crea_todos_los_carriles(self):
        geo = GeometriaCruce.dummy()
        assert len(geo.todos_los_carriles()) > 0

    def test_carriles_de_vialidad(self):
        geo = GeometriaCruce.dummy()
        carriles = geo.carriles_de("toluca_arriba")
        assert len(carriles) >= 1
        for c in carriles:
            assert c.vialidad == "toluca_arriba"

    def test_capacidad_vehiculos_razonable(self):
        geo = GeometriaCruce.dummy()
        for c in geo.todos_los_carriles():
            assert c.capacidad_vehiculos > 0
            assert c.capacidad_vehiculos < 200   # número razonable

    def test_acepta_movimiento(self):
        carril = Carril(
            id="test_1", vialidad="toluca_arriba", numero=1,
            movimientos_permitidos=["recto", "izquierda"],
            longitud_almacenamiento_m=60.0
        )
        assert carril.acepta_movimiento("recto")
        assert carril.acepta_movimiento("izquierda")
        assert not carril.acepta_movimiento("derecha")

    def test_esta_saturado(self):
        carril = Carril(
            id="test_2", vialidad="toluca_arriba", numero=1,
            movimientos_permitidos=["recto"],
            longitud_almacenamiento_m=45.0    # ≈ 8 vehículos
        )
        capacidad = carril.capacidad_vehiculos
        assert not carril.esta_saturado(0)
        assert carril.esta_saturado(capacidad)


# ─────────────────────────────────────────────
# Semáforo
# ─────────────────────────────────────────────

class TestSemaforo:

    def test_ciclo_completo(self):
        sem = Semaforo.dummy()
        ciclo = sem.ciclo_total
        for _ in range(ciclo):
            sem.tick()
        # Después de un ciclo completo debe volver a fase 0
        assert sem._fase_idx == 0

    def test_carril_tiene_verde(self):
        sem = Semaforo.dummy()
        # Al inicio: fase verde Toluca
        assert sem.carril_tiene_verde("tol_arr_1")
        assert not sem.carril_tiene_verde("per_nor_1")

    def test_ajustar_duracion_respeta_limites(self):
        sem = Semaforo.dummy()
        sem.ajustar_duracion(delta_toluca=1000)   # debe clampear a DURACION_MAX
        assert sem.duracion_toluca == Semaforo.DURACION_MAX

        sem.ajustar_duracion(delta_toluca=-1000)  # debe clampear a DURACION_MIN
        assert sem.duracion_toluca == Semaforo.DURACION_MIN

    def test_tiempo_restante_decrece(self):
        sem = Semaforo.dummy()
        t1 = sem.tiempo_restante()
        sem.tick()
        t2 = sem.tiempo_restante()
        assert t2 == t1 - 1

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

    def test_destino_valido(self):
        router = MarkovRouter.dummy()
        for _ in range(50):
            destino = router.elegir_destino("toluca_arriba", periodo=1)
            assert destino in ["recto", "izquierda", "derecha"]

    def test_probabilidades_suman_uno(self):
        router = MarkovRouter.dummy()
        for vialidad, periodos in router.matriz.items():
            for periodo, probs in periodos.items():
                assert abs(sum(probs) - 1.0) < 1e-6, \
                    f"{vialidad} periodo {periodo}: suman {sum(probs)}"

    def test_validacion_rechaza_probs_incorrectas(self):
        with pytest.raises(AssertionError):
            MarkovRouter({"toluca_arriba": {"0": [0.5, 0.5, 0.5]}})  # suman 1.5


# ─────────────────────────────────────────────
# Vehículo
# ─────────────────────────────────────────────

class TestVehiculo:

    def _vehiculo(self, perfil=PerfilConductor.NORMAL) -> Vehiculo:
        return Vehiculo(
            id=1, vialidad_origen="toluca_arriba",
            carril_id="tol_arr_1", destino="recto",
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
