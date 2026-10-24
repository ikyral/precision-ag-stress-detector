import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Polygon, mapping, shape

ROOT = Path(__file__).resolve().parents[2]   # корень репозитория
RAW = ROOT / "data" / "samples" / "aoi_raw.txt"
OUT = ROOT / "data" / "samples" / "field_boundary.geojson"
INNER_BUFFER_M = 30        # отступ внутрь от края круга
UTM = "EPSG:32637"         # UTM 37N, подходит для долготы ~40 E

to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
to_wgs = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

# 1. Читаем нарисованный контур (GeoJSON или WKT)
text = RAW.read_text(encoding="utf-8").strip()
try:
    gj = json.loads(text)
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]["geometry"]
    elif gj.get("type") == "Feature":
        gj = gj["geometry"]
    poly = shape(gj)
except json.JSONDecodeError:
    poly = wkt.loads(text)

lon, lat = poly.exterior.xy
assert 46 < np.mean(lat) < 48 and 39 < np.mean(lon) < 42, \
    "Координаты вне ожидаемого района: возможно, перепутаны lat/lon"

# 2. Подбираем окружность по вершинам (метод наименьших квадратов)
x, y = to_utm.transform(np.array(lon[:-1]), np.array(lat[:-1]))
A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
b = x**2 + y**2
cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
radius = float(np.sqrt(c + cx**2 + cy**2))

clon, clat = to_wgs.transform(cx, cy)
print(f"Центр: {clat:.5f}, {clon:.5f}  Радиус: {radius:.0f} м")

# 3. Строим круг с отступом внутрь и возвращаем в EPSG:4326
from shapely.geometry import Point
circle = Point(cx, cy).buffer(radius - INNER_BUFFER_M, 64)
ex, ey = circle.exterior.xy
glon, glat = to_wgs.transform(np.array(ex), np.array(ey))
poly_wgs = Polygon(zip(glon, glat))

geom = mapping(poly_wgs)
geom["coordinates"] = [[[round(px, 6), round(py, 6)] for px, py in ring]
                       for ring in geom["coordinates"]]
fc = {"type": "FeatureCollection",
      "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}
OUT.write_text(json.dumps(fc), encoding="utf-8")
print(f"Площадь контура: {circle.area / 10000:.1f} га -> {OUT}")