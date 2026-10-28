"""HTML-карта для глазной проверки совмещения PNG-оверлеев: подложка Esri, контур поля, слои одной даты.

    python tools/overlay_map.py --data ../../data [--date 2026_08_07] [--show RGB]
                                [--out ../../data/overlays/overlay_check.html]

PNG встраиваются так же, как в кабинете М4 (folium ImageOverlay по meta.bounds, opacity 0.85). Для проверки
пиксельного совмещения поверх лежат полигоны предписания М3 (их рёбра идут по границам пикселей 10 м) и
контур поля; слои переключаются в правом верхнем углу.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"


def build(data: Path, date: str | None, out: Path, show: str | None = None) -> Path:
    import folium
    from PIL import Image

    meta = json.loads((data / "overlays" / "meta.json").read_text(encoding="utf-8"))
    date = (date or meta["dates"][-1]).replace("-", "_")
    layers = meta["layers"][date]
    show = show if show in layers else next(iter(layers))
    m = folium.Map(tiles=None, control_scale=True, max_zoom=21)
    folium.TileLayer(tiles=ESRI_TILES, attr=ESRI_ATTR, name="Спутник Esri", max_zoom=21, max_native_zoom=19).add_to(m)
    for name, fname in layers.items():
        rgba = np.array(Image.open(data / "overlays" / fname).convert("RGBA"))
        folium.raster_layers.ImageOverlay(image=rgba, bounds=meta["bounds"], opacity=0.85,
                                          name=f"{name} {date}", show=(name == show)).add_to(m)
    presc = data / "prescriptions" / f"prescription_{date}.geojson"
    if presc.exists():
        folium.GeoJson(json.loads(presc.read_text(encoding="utf-8")), name="Полигоны М3 (границы пикселей)",
                       style_function=lambda f: {"color": "#00ffff", "weight": 1, "fill": False}).add_to(m)
    for p in (data / "field_boundary.geojson", data / "samples" / "field_boundary.geojson"):
        if p.exists():
            folium.GeoJson(json.loads(p.read_text(encoding="utf-8")), name="Граница поля",
                           style_function=lambda f: {"color": "#ffffff", "weight": 2.5, "fill": False}).add_to(m)
            break
    (s, w), (n, e) = meta["bounds"]
    folium.Rectangle([[s, w], [n, e]], color="#ff00ff", weight=1, fill=False, dash_array="4 4",
                     name="meta.bounds").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(meta["bounds"])
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="HTML для глазной проверки совмещения оверлеев")
    ap.add_argument("--data", type=Path, default=Path("../../data"))
    ap.add_argument("--date", default=None, help="YYYY_MM_DD, по умолчанию последняя дата")
    ap.add_argument("--show", default=None, help="слой, включённый при открытии (NDVI, RGB, ...); по умолчанию первый")
    ap.add_argument("--out", type=Path, default=None, help="по умолчанию <data>/overlays/overlay_check.html")
    a = ap.parse_args(argv)
    out = a.out or a.data / "overlays" / "overlay_check.html"
    print(build(a.data, a.date, out, a.show))
    return 0


if __name__ == "__main__":
    sys.exit(main())
