"""Ручные маски облаков и теней (manual_masks.geojson): то, что не поймали SCL и s2cloudless."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import Grid

KINDS = ("cloud", "shadow")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "manual_masks.geojson"


@dataclass(frozen=True)
class ManualPolygon:
    date: str
    kind: str
    note: str
    geometry: dict   # GeoJSON-геометрия в crs файла
    crs: str


def _crs_of(gj: dict) -> str:
    """crs из GeoJSON ("urn:ogc:def:crs:EPSG::32637" или "EPSG:32637"); по умолчанию EPSG:4326."""
    name = (gj.get("crs") or {}).get("properties", {}).get("name")
    if not name:
        return "EPSG:4326"
    if "CRS84" in name:
        return "EPSG:4326"
    code = name.rsplit(":", 1)[-1]
    return f"EPSG:{code}"


def load(path: str | Path | None) -> list[ManualPolygon]:
    """Полигоны из файла. Нет файла (или path=None) — пустой список."""
    if path is None or not Path(path).exists():
        return []
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    crs = _crs_of(gj)
    out = []
    for i, f in enumerate(gj.get("features", [])):
        p = f.get("properties") or {}
        date, kind = str(p.get("date", "")), p.get("kind")
        try:
            dt.date.fromisoformat(date)
        except ValueError:
            raise ValueError(f"{path}: объект {i}: date должна быть YYYY-MM-DD, а не {date!r}") from None
        if kind not in KINDS:
            raise ValueError(f"{path}: объект {i}: kind должен быть cloud или shadow, а не {kind!r}")
        out.append(ManualPolygon(date, kind, str(p.get("note", "")), f["geometry"], crs))
    return out


def mask_for_date(polys: list[ManualPolygon], date: str, grid: Grid) -> np.ndarray | None:
    """Маска полигонов этой даты на сетке grid (пиксель входит, если его центр внутри). None — полигонов нет."""
    mine = [p for p in polys if p.date == date]
    if not mine:
        return None
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping, shape
    from shapely.ops import transform

    geoms = []
    for p in mine:
        g = shape(p.geometry)
        if p.crs != grid.crs:
            g = transform(Transformer.from_crs(p.crs, grid.crs, always_xy=True).transform, g)
        geoms.append(mapping(g))
    return ~geometry_mask(geoms, out_shape=grid.shape, transform=grid.transform)


def polygons_from_mask(mask: np.ndarray, grid: Grid) -> list[dict]:
    """Пиксельная маска -> GeoJSON-полигоны по границам пикселей (в CRS сетки)."""
    from rasterio.features import shapes

    return [geom for geom, v in shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform) if v == 1]
