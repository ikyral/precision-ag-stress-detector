"""Общая сетка 10 м для всех дат: окно по рамке поля, привязанное к пикселям B04 плитки."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine


@dataclass(frozen=True)
class Grid:
    crs: str          # "EPSG:32637"
    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def buffered(self, px: int) -> "Grid":
        """Та же сетка, расширенная на px пикселей во все стороны (контекст для s2cloudless)."""
        t = self.transform
        return Grid(self.crs, Affine(t.a, 0, t.c - px * t.a, 0, t.e, t.f - px * t.e),
                    self.width + 2 * px, self.height + 2 * px)


def load_boundary(path: str | Path) -> dict:
    """Геометрия первого объекта field_boundary.geojson (EPSG:4326)."""
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]
    return gj.get("geometry", gj)


def boundary_bbox(geom: dict) -> list[float]:
    ring = geom["coordinates"][0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return [min(lons), min(lats), max(lons), max(lats)]


def field_polygon(geom: dict, crs: str):
    """Контур поля в CRS сетки (shapely Polygon)."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ring = geom["coordinates"][0]
    xs, ys = to_utm.transform([p[0] for p in ring], [p[1] for p in ring])
    return Polygon(zip(xs, ys))


def grid_from_tile(poly, crs: str, tile_transform: Affine, res: float = 10.0) -> Grid:
    """Окно по рамке контура с запасом 1 px, кратное пикселям плитки (алгоритм build_sample.py)."""
    t0 = tile_transform
    minx, miny, maxx, maxy = poly.bounds
    col0 = math.floor((minx - t0.c) / res) - 1
    row0 = math.floor((t0.f - maxy) / res) - 1
    col1 = math.ceil((maxx - t0.c) / res) + 1
    row1 = math.ceil((t0.f - miny) / res) + 1
    tr = Affine(res, 0, t0.c + col0 * res, 0, -res, t0.f - row0 * res)
    return Grid(crs, tr, col1 - col0, row1 - row0)


def field_mask(poly, grid: Grid) -> np.ndarray:
    """True внутри контура (как в КТ1: all_touched=False)."""
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    return ~geometry_mask([mapping(poly)], out_shape=grid.shape, transform=grid.transform)


def grid_of_file(path: str | Path) -> Grid:
    import rasterio

    with rasterio.open(path) as src:
        return Grid(src.crs.to_string(), src.transform, src.width, src.height)


def compare_grids(a: Grid, b: Grid) -> list[str]:
    """Список расхождений (пустой — сетки совпадают)."""
    diffs = []
    if a.crs != b.crs:
        diffs.append(f"CRS {a.crs} != {b.crs}")
    if not a.transform.almost_equals(b.transform, precision=1e-6):
        diffs.append(f"transform {tuple(a.transform)[:6]} != {tuple(b.transform)[:6]}")
    if a.shape != b.shape:
        diffs.append(f"размер {a.width}x{a.height} != {b.width}x{b.height}")
    return diffs
