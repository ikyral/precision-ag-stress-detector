"""Сводная статистика по индексам, зонам и скрытому стрессу для JSON."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

PERCENTILES = (5, 25, 50, 75, 95)


def _f(x, digits: int = 6):
    """float -> JSON-безопасное значение (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


def index_stats(arr: np.ndarray) -> dict:
    """Описательная статистика по конечным значениям индекса."""
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        base = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {**base, **{f"p{p}": None for p in PERCENTILES}}
    pct = np.percentile(vals, PERCENTILES)
    out = {
        "n": int(vals.size),
        "mean": _f(vals.mean()),
        "std": _f(vals.std()),
        "min": _f(vals.min()),
        "max": _f(vals.max()),
    }
    out.update({f"p{p}": _f(v) for p, v in zip(PERCENTILES, pct)})
    return out


def _area_ha(pixels: int, pixel_area_m2: float | None):
    return _f(pixels * pixel_area_m2 / 10_000.0, 4) if pixel_area_m2 else None


def _means(indices: Mapping[str, np.ndarray], sel: np.ndarray) -> dict:
    return {name: _f(np.nanmean(arr[sel])) if sel.any() else None for name, arr in indices.items()}


def zone_stats(
    zones: np.ndarray,
    labels: list[str],
    indices: Mapping[str, np.ndarray],
    pixel_area_m2: float | None,
) -> list[dict]:
    total = int((zones > 0).sum())
    rows = []
    for i, label in enumerate(labels, start=1):
        sel = zones == i
        n = int(sel.sum())
        rows.append({
            "class": i,
            "label": label,
            "pixels": n,
            "fraction": _f(n / total) if total else None,
            "area_ha": _area_ha(n, pixel_area_m2),
            "mean": _means(indices, sel),
        })
    return rows


def hidden_stress_stats(
    mask: np.ndarray,
    valid: np.ndarray,
    indices: Mapping[str, np.ndarray],
    z_ndre: np.ndarray,
    z_ndvi: np.ndarray,
    pixel_area_m2: float | None,
) -> dict:
    n = int(mask.sum())
    total = int(valid.sum())
    return {
        "pixels": n,
        "fraction": _f(n / total) if total else None,
        "area_ha": _area_ha(n, pixel_area_m2),
        "mean": _means(indices, mask),
        "mean_z_ndre": _f(np.nanmean(z_ndre[mask])) if n else None,
        "mean_z_ndvi": _f(np.nanmean(z_ndvi[mask])) if n else None,
    }
