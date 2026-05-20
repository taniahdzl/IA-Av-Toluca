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

SIM_SPEED   = int(os.getenv("VIZ_SIM_SPEED", 5))
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


@app.on_event("startup")
def startup():
    from sim.intersection import SimuladorCruce
    from viz.renderer import Renderer

    state.sim = SimuladorCruce.dummy()
    state.renderer = Renderer(state.sim)
    state.sim.reset()

    # Intentar cargar modelo SAC
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        try:
            from stable_baselines3 import SAC
            state.modelo_rl = SAC.load(str(model_path))
            print(f"✓ Modelo SAC cargado: {model_path}")
        except Exception as e:
            print(f"⚠  No se pudo cargar el modelo: {e}")
    else:
        print(f"⚠  Modelo no encontrado en {model_path} — corriendo baseline")

    state.corriendo = True
    threading.Thread(target=_loop_simulacion, daemon=True).start()
    print("✓ Simulación iniciada")


def _loop_simulacion():
    from rl.environment import CruceEnv
    import numpy as np

    env = None
    obs = None

    if state.modelo_rl:
        env = CruceEnv(sim=state.sim)
        obs, _ = env.reset()

    while state.corriendo:
        if state.pausado:
            time.sleep(0.1)
            continue

        with state.lock:
            for _ in range(state.speed):
                if state.modelo_rl and env and obs is not None:
                    try:
                        accion, _ = state.modelo_rl.predict(obs, deterministic=True)
                        accion_real = env._desnormalizar_accion(accion)
                        obs_new, _, done, _, _ = env.step(accion)
                        obs = obs_new
                        if done:
                            state.resumen_rl = state.sim.monitor.resumen()
                            obs, _ = env.reset()
                    except Exception:
                        state.sim.step()
                else:
                    _, _, info = state.sim.step()
                    if state.sim.t >= 3600:
                        state.resumen_base = state.sim.monitor.resumen()
                        state.sim.reset()

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


@app.get("/health")
def health():
    return {"status": "ok", "t": state.sim.t if state.sim else 0,
            "modelo_cargado": state.modelo_rl is not None}
