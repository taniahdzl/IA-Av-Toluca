"""
viz/app.py
──────────
Servidor FastAPI para el demo visual del cruce.

Endpoints:
  GET  /              → sirve el frontend (index.html)
  GET  /sim/frame     → frame actual del cruce (JSON)
  POST /sim/control   → pausar, reanudar, cambiar velocidad
  GET  /sim/reset     → reiniciar la simulación
  GET  /metrics/comparison → datos baseline vs RL para el dashboard

La simulación corre en un hilo de background y el servidor
sirve su estado más reciente en cada request.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ── Configuración ─────────────────────────────────────────────

SIM_SPEED = int(os.getenv("VIZ_SIM_SPEED", 5))       # segundos de sim por frame real
USAR_DUMMY = os.getenv("VIZ_USAR_DUMMY", "true").lower() == "true"
MODEL_PATH = os.getenv("RL_MODEL_PATH", "rl/models/best_model.zip")

STATIC_DIR = Path(__file__).parent / "static"

# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="Cruce Av. Toluca × Periférico — Demo RL")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Estado global del servidor ────────────────────────────────

class SimState:
    """Contenedor del estado compartido entre el hilo de sim y los endpoints."""
    def __init__(self):
        self.sim = None
        self.renderer = None
        self.modelo_rl = None
        self.corriendo = False
        self.pausado = False
        self.speed = SIM_SPEED
        self.resumen_base: Optional[dict] = None
        self.resumen_rl: Optional[dict] = None
        self.lock = threading.Lock()

state = SimState()


# ── Inicialización ────────────────────────────────────────────

@app.on_event("startup")
def startup():
    """Inicializa el simulador y arranca el hilo de simulación."""
    from sim.intersection import SimuladorCruce
    from viz.renderer import Renderer

    if USAR_DUMMY:
        state.sim = SimuladorCruce.dummy()
    else:
        state.sim = SimuladorCruce.desde_calibracion()

    state.renderer = Renderer(state.sim)
    state.sim.reset()

    # Cargar modelo RL si existe
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        try:
            from stable_baselines3 import PPO
            state.modelo_rl = PPO.load(str(model_path))
            print(f"✓ Modelo RL cargado: {model_path}")
        except Exception as e:
            print(f"⚠  No se pudo cargar el modelo RL: {e}")

    # Arrancar hilo de simulación
    state.corriendo = True
    hilo = threading.Thread(target=_loop_simulacion, daemon=True)
    hilo.start()
    print("✓ Simulación iniciada en background")


def _loop_simulacion():
    """
    Hilo de background: avanza la simulación continuamente.
    Si hay modelo RL cargado, lo usa para decidir acciones.
    Si no, corre con semáforo fijo (baseline).
    """
    obs = state.sim.get_estado() if hasattr(state.sim, 'get_estado') else None

    while state.corriendo:
        if state.pausado:
            time.sleep(0.1)
            continue

        with state.lock:
            for _ in range(state.speed):
                # Decidir acción
                accion = 0
                if state.modelo_rl is not None and obs is not None:
                    try:
                        accion, _ = state.modelo_rl.predict(obs, deterministic=True)
                        accion = int(accion)
                    except Exception:
                        accion = 0

                obs_new, _, info = state.sim.step(accion=accion)
                obs = obs_new

                # Reiniciar episodio si terminó (1 hora simulada)
                if state.sim.t >= int(os.getenv("SIM_EPISODE_DURATION", 3600)):
                    state.resumen_rl = state.sim.monitor.resumen()
                    state.sim.reset()
                    obs = None

        time.sleep(1.0 / 30)   # ~30 fps máximo


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """Sirve el frontend principal."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h2>Frontend no encontrado en viz/static/index.html</h2>")


@app.get("/sim/frame")
def get_frame():
    """
    Frame actual del cruce para la animación.
    El frontend hace polling a este endpoint (ej. cada 100ms).
    """
    with state.lock:
        if state.renderer is None:
            return JSONResponse({"error": "simulador no inicializado"}, status_code=503)
        frame = state.renderer.get_frame()
    return JSONResponse(frame)


@app.post("/sim/control")
def control(accion: str, valor: Optional[int] = None):
    """
    Controla la simulación desde el frontend.

    accion: "pausar" | "reanudar" | "velocidad" | "modo"
    valor:  para "velocidad": segundos de sim por frame (1-60)
            para "modo": 0=baseline, 1=RL
    """
    if accion == "pausar":
        state.pausado = True
    elif accion == "reanudar":
        state.pausado = False
    elif accion == "velocidad" and valor is not None:
        state.speed = max(1, min(60, valor))
    return JSONResponse({"ok": True, "pausado": state.pausado, "speed": state.speed})


@app.get("/sim/reset")
def reset_sim():
    """Reinicia la simulación al segundo 0."""
    with state.lock:
        state.sim.reset()
    return JSONResponse({"ok": True})


@app.get("/metrics/comparison")
def metrics_comparison():
    """
    Datos de baseline vs RL para el dashboard comparativo.
    Retorna vacío si aún no hay datos suficientes.
    """
    with state.lock:
        payload = state.renderer.get_metrics_comparison(
            state.resumen_base, state.resumen_rl
        )
    return JSONResponse(payload)


@app.get("/health")
def health():
    return {"status": "ok", "t": state.sim.t if state.sim else 0}
