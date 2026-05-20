"""
data_collection/osm_downloader.py
──────────────────────────────────
Descarga la geometría del cruce Av. Toluca – Anillo Periférico desde
OpenStreetMap usando osmnx y la guarda en data/raw/osm_cruce.geojson.

Uso:
    python data_collection/osm_downloader.py
"""

import json
import os
from pathlib import Path

# El cruce está en estas coordenadas aproximadas
# (centro del nodo Av. Toluca × Periférico, CDMX)
LAT_CENTRO = 19.340477
LON_CENTRO = -99.203100
RADIO_METROS = 400

OUTPUT_PATH = Path("data/raw/osm_cruce.geojson")


def descargar_red_vial():
    try:
        import osmnx as ox
    except ImportError:
        print("✗  osmnx no está instalado. Corre: pip install osmnx")
        return

    print(f"   Descargando red vial (radio={RADIO_METROS}m)...")
    G = ox.graph_from_point(
        (LAT_CENTRO, LON_CENTRO),
        dist=RADIO_METROS,
        network_type="drive",
        retain_all=True,
    )

    # Convertir a GeoDataFrame y exportar
    _, edges = ox.graph_to_gdfs(G)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"   ✓  Red vial guardada en {OUTPUT_PATH}")
    print(f"      {len(edges)} segmentos descargados")

    # Extraer atributos útiles para el simulador
    _extraer_resumen_carriles(edges)


def _extraer_resumen_carriles(edges):
    import pandas as pd
    print("\n   ── Atributos de carriles encontrados en OSM ──")
    cols_interes = ["name", "lanes", "maxspeed", "highway", "length"]
    cols_disponibles = [c for c in cols_interes if c in edges.columns]

    # Normalizar: algunos campos son listas, convertir a string
    df = edges[cols_disponibles].copy()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    resumen = df.dropna(subset=["name"]).sort_values("name")

    palabras_clave = ["toluca", "periferico", "periférico", "revolucion",
                      "revolución", "queretaro", "querétaro", "lopez mateos", "adolfo"]
    mask = resumen["name"].str.lower().str.contains(
        "|".join(palabras_clave), na=False
    )
    relevantes = resumen[mask]

    if relevantes.empty:
        print("   ⚠  No se encontraron calles relevantes. Nombres encontrados:")
        print("\n".join(f"      {n}" for n in sorted(df["name"].dropna().unique())[:20]))
    else:
        print(relevantes.to_string(index=False))
        print(f"\n   Usa estos datos para completar data/processed/geometria_carriles.json")

if __name__ == "__main__":
    descargar_red_vial()
