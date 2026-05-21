"""
viz/app.py — Servidor FastAPI para el demo visual.
"""

from __future__ import annotations
import os, threading, time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

SIM_SPEED         = int(os.getenv("VIZ_SIM_SPEED", 2))
EPISODE_DURATION  = int(os.getenv("SIM_EPISODE_DURATION", 3600))
SIM_STEP_INTERVAL = int(os.getenv("SIM_STEP_INTERVAL", 30))
MODEL_PATH  = os.getenv("RL_MODEL_PATH", "rl/models/sac_semaforo_v1.zip")
STATIC_DIR  = Path(__file__).parent / "static"

app = FastAPI(title="Cruce Av. Toluca × Periférico — Demo RL")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SimState:
    def __init__(self):
        self.sim = None
        self.renderer = None
        self.modelo_rl = None
        self.corriendo = False
        self.pausado = False
        self.speed = SIM_SPEED
        self.resumen_base: Optional[dict] = None
        self.resumen_rl:   Optional[dict] = None
        self.lock = threading.Lock()

state = SimState()


def _cargar_modelo(path: str) -> Optional[object]:
    """Carga el modelo SAC desde disco. Devuelve None si no existe o falla."""
    from stable_baselines3 import SAC
    model_path = Path(path)
    if not model_path.exists():
        print(f"⚠  Modelo no encontrado en {model_path} — corriendo baseline")
        return None
    try:
        modelo = SAC.load(str(model_path))
        print(f"✓ Modelo SAC cargado: {model_path}")
        return modelo
    except Exception as e:
        print(f"⚠  No se pudo cargar el modelo: {e}")
        return None


@app.on_event("startup")
def startup():
    from sim.intersection import SimuladorCruce
    from viz.renderer import Renderer

    state.sim = SimuladorCruce.dummy()
    state.renderer = Renderer(state.sim)
    state.sim.reset()

    state.modelo_rl = _cargar_modelo(MODEL_PATH)

    state.corriendo = True
    threading.Thread(target=_loop_simulacion, daemon=True).start()
    print("✓ Simulación iniciada")


def _loop_simulacion():
    from rl.environment import CruceEnv
    import numpy as np

    DEMO_DURACION = 1800  # punto en que se guarda resumen baseline (sin RL)

    env = None
    accion_real = None
    pasos_en_episodio = 0

    if state.modelo_rl:
        env = CruceEnv(sim=state.sim)
        obs, _ = env.reset()
        accion, _ = state.modelo_rl.predict(obs, deterministic=True)
        accion_real = env._desnormalizar_accion(accion)
        state.sim.semaforo.set_duraciones(*accion_real)

    while state.corriendo:
        if state.pausado:
            time.sleep(0.1)
            continue

        with state.lock:
            for _ in range(state.speed):
                if state.modelo_rl and env:
                    # Nueva decisión del agente cada SIM_STEP_INTERVAL pasos
                    if pasos_en_episodio % SIM_STEP_INTERVAL == 0:
                        obs = state.sim.get_estado()
                        accion, _ = state.modelo_rl.predict(obs, deterministic=True)
                        accion_real = env._desnormalizar_accion(accion)
                        state.sim.semaforo.set_duraciones(*accion_real)

                    state.sim.step()
                    pasos_en_episodio += 1

                    if state.sim.t >= EPISODE_DURATION:
                        state.resumen_rl = state.sim.monitor.resumen()
                        obs, _ = env.reset()   # también resetea state.sim
                        pasos_en_episodio = 0
                        accion, _ = state.modelo_rl.predict(obs, deterministic=True)
                        accion_real = env._desnormalizar_accion(accion)
                        state.sim.semaforo.set_duraciones(*accion_real)
                else:
                    state.sim.step()
                    if state.sim.t == DEMO_DURACION:
                        state.resumen_base = state.sim.monitor.resumen()

        time.sleep(1.0 / 30)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse("<h2>Frontend no encontrado</h2>")


@app.get("/sim/frame")
def get_frame():
    with state.lock:
        if not state.renderer:
            return JSONResponse({"error": "no inicializado"}, status_code=503)
        return JSONResponse(state.renderer.get_frame())


@app.post("/sim/control")
def control(accion: str, valor: Optional[int] = None):
    if accion == "pausar":   state.pausado = True
    elif accion == "reanudar": state.pausado = False
    elif accion == "velocidad" and valor:
        state.speed = max(1, min(60, valor))
    return JSONResponse({"ok": True, "pausado": state.pausado, "speed": state.speed})


@app.get("/sim/reset")
def reset_sim():
    with state.lock:
        state.sim.reset()
    return JSONResponse({"ok": True})


@app.get("/metrics/comparison")
def metrics_comparison():
    with state.lock:
        return JSONResponse(state.renderer.get_metrics_comparison(
            state.resumen_base, state.resumen_rl
        ))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    path = STATIC_DIR / "dashboard.html"
    if path.exists():
        return HTMLResponse(content=path.read_text())
    return HTMLResponse("<h2>Dashboard no encontrado</h2>")


@app.post("/sim/reload-model")
def reload_model(path: Optional[str] = None):
    """Recarga el modelo SAC desde disco sin reiniciar el servidor."""
    target = path or MODEL_PATH
    nuevo = _cargar_modelo(target)
    with state.lock:
        state.modelo_rl = nuevo
        state.sim.reset()
    return JSONResponse({
        "ok": True,
        "modelo_cargado": nuevo is not None,
        "path": target,
    })


@app.get("/health")
def health():
    return {"status": "ok", "t": state.sim.t if state.sim else 0,
            "modelo_cargado": state.modelo_rl is not None}
