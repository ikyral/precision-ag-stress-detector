"""Проверка выходных файлов модуля 2 на контракт с M3 (векторизатор) и M4 (дашборд).

Повторяет требования, которые M3 проверяет при чтении маски (vra/core.py `_extract`, vra/batch.py
`run_series`), без установки rasterio/geopandas:

* zones_YYYY_MM_DD.tif — один канал, uint8, классы только 0..3, есть CRS и невырожденная геопривязка;
* площадь размеченных пикселей одинакова на всех датах (допуск M3 — 0.01 га);
* у всех дат одна сетка (размер, шаг, начало координат, CRS);
* для каждой маски есть stats/index_stats_*.json с полями, которые читает M4, и растры индексов.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geotiff_io import read_geotiff
from .pipeline import parse_date

M3_CLASSES = (0, 1, 2, 3)
M3_AREA_TOLERANCE_HA = 0.01
M4_STATS_KEYS = ("date", "mean_ndvi", "mean_ndre", "mean_msavi", "mean_ndmi", "early_stress_alert",
                 "field_area_ha", "zones", "files")
INDEX_FILES = ("NDVI", "NDRE", "MSAVI", "NDMI")


def check_outputs(data_dir: str | Path, area_tolerance_ha: float = M3_AREA_TOLERANCE_HA) -> dict:
    """Проверяет data_dir (zones/, stats/, indices/). Возвращает {"ok", "dates", "errors", "area_ha", ...}."""
    data_dir = Path(data_dir)
    errors: list[str] = []
    areas: dict[str, float] = {}
    grids: dict[str, tuple] = {}
    zone_files = sorted((data_dir / "zones").glob("zones_*.tif"))
    if not zone_files:
        errors.append(f"нет файлов {data_dir / 'zones'}/zones_YYYY_MM_DD.tif")
    for path in zone_files:
        try:
            date = parse_date(path.stem[len("zones_"):])
        except ValueError:
            errors.append(f"{path.name}: в имени нет даты YYYY_MM_DD")
            continue
        stamp = date.replace("-", "_")
        try:
            arr, geo = read_geotiff(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: не читается ({exc})")
            continue
        if arr.shape[0] != 1:
            errors.append(f"{path.name}: каналов {arr.shape[0]}, нужен 1")
        band = arr[0]
        if band.dtype != np.uint8:
            errors.append(f"{path.name}: тип {band.dtype}, нужен uint8")
        bad = sorted(set(np.unique(band).tolist()) - set(M3_CLASSES))
        if bad:
            errors.append(f"{path.name}: классы {bad[:10]} вне 0..3 (M3 такие файлы не принимает)")
        if geo.epsg is None:
            errors.append(f"{path.name}: нет CRS (EPSG)")
        if geo.pixel_scale is None or geo.tiepoint is None or 0 in tuple(geo.pixel_scale)[:2]:
            errors.append(f"{path.name}: нет или вырождена геопривязка")
        n = int((band > 0).sum())
        if n == 0:
            errors.append(f"{path.name}: нет ни одного пикселя поля")
        if geo.pixel_area_m2:
            areas[date] = round(n * geo.pixel_area_m2 / 10_000.0, 4)
        grids[date] = (band.shape, tuple(geo.pixel_scale or ()), tuple(geo.tiepoint or ()), geo.epsg)

        stats_path = data_dir / "stats" / f"index_stats_{stamp}.json"
        if not stats_path.exists():
            errors.append(f"{stats_path.name}: нет (нужен дашборду M4)")
        else:
            try:
                doc = json.loads(stats_path.read_text(encoding="utf-8"))
                missing = [k for k in M4_STATS_KEYS if k not in doc]
                if missing:
                    errors.append(f"{stats_path.name}: нет полей {missing}")
                elif doc.get("field_area_ha") is not None and date in areas \
                        and abs(doc["field_area_ha"] - areas[date]) > 1e-3:
                    errors.append(f"{stats_path.name}: field_area_ha {doc['field_area_ha']} != площадь маски "
                                  f"{areas[date]}")
            except ValueError as exc:
                errors.append(f"{stats_path.name}: не JSON ({exc})")
        for name in INDEX_FILES:
            if not (data_dir / "indices" / f"{name}_{stamp}.tif").exists():
                errors.append(f"indices/{name}_{stamp}.tif: нет (слой индекса для M4)")

    if len(set(grids.values())) > 1:
        errors.append("сетка масок отличается между датами: " +
                      ", ".join(f"{d}: {g[0]} EPSG:{g[3]}" for d, g in sorted(grids.items())))
    spread = (max(areas.values()) - min(areas.values())) if areas else None
    if spread is not None and len(areas) > 1 and spread > area_tolerance_ha + 1e-9:
        errors.append(f"площадь поля в масках меняется на {spread:.4f} га (допуск M3 {area_tolerance_ha} га): "
                      + ", ".join(f"{d} {a}" for d, a in sorted(areas.items())))
    return {
        "ok": not errors,
        "dates": sorted(grids),
        "area_ha": dict(sorted(areas.items())),
        "area_spread_ha": spread,
        "errors": errors,
    }
