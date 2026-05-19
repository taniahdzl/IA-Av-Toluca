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
LAT_CENTRO = 19.3614
LON_CENTRO = -99.2143
RADIO_METROS = 300   # área alrededor del cruce a descargar

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
    """
    Extrae número de carriles y nombres de calles del GeoDataFrame
    y los imprime como referencia para llenar geometria_carriles.json.
    """
    print("\n   ── Atributos de carriles encontrados en OSM ──")
    cols_interes = ["name", "lanes", "maxspeed", "highway", "length"]
    cols_disponibles = [c for c in cols_interes if c in edges.columns]

    resumen = (
        edges[cols_disponibles]
        .dropna(subset=["name"])
        .sort_values("name")
    )

    # Filtrar solo las calles relevantes
    palabras_clave = ["toluca", "periferico", "periférico", "revolucion", "revolución"]
    mask = resumen["name"].str.lower().str.contains("|".join(palabras_clave), na=False)
    relevantes = resumen[mask]

    if relevantes.empty:
        print("   ⚠  No se encontraron calles con los nombres esperados.")
        print("      Revisa las coordenadas LAT_CENTRO / LON_CENTRO en este script.")
    else:
        print(relevantes.to_string(index=False))
        print(f"\n   Usa estos datos para completar data/processed/geometria_carriles.json")


if __name__ == "__main__":
    descargar_red_vial()
