"""
data_collection/google_routes_sampler.py
─────────────────────────────────────────
Consulta la Google Routes API en los movimientos clave del cruce
para construir una serie temporal de fricción (tiempo_real / tiempo_libre).

Uso:
    python data_collection/google_routes_sampler.py

Requiere GOOGLE_ROUTES_API_KEY en .env
Guarda resultados en data/validation/google_travel_times.csv
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_ROUTES_API_KEY", "")
OUTPUT_PATH = Path("data/validation/google_travel_times.csv")
ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Pares origen-destino que representan los movimientos del cruce.
# Cada par es una ruta corta que atraviesa el nodo Toluca × Periférico.
MOVIMIENTOS = [
    {
        "nombre": "toluca_hacia_periferico_norte",
        "origen":  {"lat": 19.3608, "lng": -99.2175},   # Av. Toluca, al sur del cruce
        "destino": {"lat": 19.3650, "lng": -99.2143},   # Periférico, al norte
    },
    {
        "nombre": "toluca_hacia_periferico_sur",
        "origen":  {"lat": 19.3608, "lng": -99.2175},
        "destino": {"lat": 19.3580, "lng": -99.2143},   # Periférico, al sur
    },
    {
        "nombre": "periferico_norte_hacia_toluca",
        "origen":  {"lat": 19.3650, "lng": -99.2143},
        "destino": {"lat": 19.3608, "lng": -99.2175},
    },
    {
        "nombre": "periferico_sur_hacia_toluca",
        "origen":  {"lat": 19.3580, "lng": -99.2143},
        "destino": {"lat": 19.3608, "lng": -99.2175},
    },
]

HEADERS_CSV = [
    "timestamp", "hora", "movimiento",
    "duracion_trafico_s", "duracion_libre_s", "friccion"
]


def consultar_ruta(origen: dict, destino: dict) -> dict | None:
    """Llama a Routes API y devuelve duración con y sin tráfico."""
    if not API_KEY:
        print("  ⚠  GOOGLE_ROUTES_API_KEY no configurada — omitiendo consulta.")
        return None

    payload = {
        "origin":      {"location": {"latLng": origen}},
        "destination": {"location": {"latLng": destino}},
        "travelMode":  "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "extraComputations": ["TRAFFIC_ON_POLYLINE"],
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.staticDuration",
    }

    try:
        resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ruta = data["routes"][0]
        dur_trafico = int(ruta["duration"].replace("s", ""))
        dur_libre   = int(ruta["staticDuration"].replace("s", ""))
        return {"duracion_trafico_s": dur_trafico, "duracion_libre_s": dur_libre}
    except Exception as e:
        print(f"  ✗  Error consultando Routes API: {e}")
        return None


def muestrear_una_vez():
    """Consulta todos los movimientos y agrega una fila al CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    escribir_header = not OUTPUT_PATH.exists()

    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS_CSV)
        if escribir_header:
            writer.writeheader()

        now = datetime.now()
        for mov in MOVIMIENTOS:
            resultado = consultar_ruta(mov["origen"], mov["destino"])
            if resultado:
                friccion = (resultado["duracion_trafico_s"] /
                            max(resultado["duracion_libre_s"], 1))
                fila = {
                    "timestamp":         now.isoformat(),
                    "hora":              now.strftime("%H:%M"),
                    "movimiento":        mov["nombre"],
                    "duracion_trafico_s": resultado["duracion_trafico_s"],
                    "duracion_libre_s":  resultado["duracion_libre_s"],
                    "friccion":          round(friccion, 3),
                }
                writer.writerow(fila)
                print(f"  {mov['nombre']}: {resultado['duracion_trafico_s']}s "
                      f"(libre: {resultado['duracion_libre_s']}s, "
                      f"fricción: {friccion:.2f})")
            time.sleep(0.5)  # respetar rate limits


def muestrear_continuamente(intervalo_min: int = 15, duracion_horas: int = 3):
    """
    Muestrea cada N minutos durante M horas.
    Útil para capturar la evolución de la congestión en hora pico.
    """
    total_muestras = (duracion_horas * 60) // intervalo_min
    print(f"Iniciando muestreo: {total_muestras} muestras cada {intervalo_min} min\n")

    for i in range(total_muestras):
        print(f"── Muestra {i+1}/{total_muestras} ({datetime.now().strftime('%H:%M')}) ──")
        muestrear_una_vez()
        if i < total_muestras - 1:
            time.sleep(intervalo_min * 60)

    print(f"\n✓  Muestreo terminado. Datos en {OUTPUT_PATH}")


if __name__ == "__main__":
    # Por defecto: una sola muestra
    print("Consultando Google Routes API...\n")
    muestrear_una_vez()
    print(f"\n✓  Guardado en {OUTPUT_PATH}")
