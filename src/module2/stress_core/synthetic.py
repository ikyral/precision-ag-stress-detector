"""Синтетическое поле для проверки алгоритмов и демонстрации.

Три даты, 120x160 пикселей по 10 м (192 га). Отражение подобрано правдоподобно для
Sentinel-2 (R=B04, RE1=B05, N=B08, S1=B11). Известная «истина»:

  * умеренная зона  — нижний левый угол, присутствует на всех датах;
  * острый очаг     — верхний правый угол, присутствует на всех датах;
  * скрытый очаг    — верхний левый угол:
        дата 0 — нет;
        дата 1 — рост отражения в Red Edge при почти неизменном NDVI (ранняя стадия);
        дата 2 — стресс дошёл до NDVI (очаг становится «видимым»);
  * на дате 2 часть поля закрыта «облаком» (NaN).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .geotiff_io import GeoMeta, write_geotiff

DATES = ("2026-06-05", "2026-06-20", "2026-07-05")
HEALTHY = {"R": 0.040, "RE1": 0.140, "N": 0.450, "S1": 0.200}

# сдвиги отражения для каждого типа стресса
_EFFECT = {
    "moderate": {"R": 0.015, "RE1": 0.030, "N": -0.070, "S1": 0.030},
    "acute": {"R": 0.050, "RE1": 0.070, "N": -0.180, "S1": 0.070},
    "hidden": {"R": 0.005, "RE1": 0.050, "N": -0.010, "S1": 0.005},
}


@dataclass
class SyntheticScene:
    date: str
    bands: dict[str, np.ndarray]  # отражение 0..1, NaN под облаком
    truth: dict[str, np.ndarray]  # мягкие маски 0..1: moderate, acute, hidden
    geo: GeoMeta


def _blob(shape, cy, cx, ry, rx, sharpness: float = 10.0) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    d = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    return 1.0 / (1.0 + np.exp((d - 1.0) * sharpness))


def make_scenes(seed: int = 0, shape=(120, 160), epsg: int = 32637) -> list[SyntheticScene]:
    rng = np.random.default_rng(seed)
    h, w = shape
    geo = GeoMeta.from_origin(500_000.0, 5_800_000.0, 10.0, 10.0, epsg)

    g = ndi.gaussian_filter(rng.standard_normal(shape), 12)
    g = (g - g.mean()) / g.std()  # плавная фоновая неоднородность

    truth = {
        "moderate": _blob(shape, 85, 40, 22, 30),
        "acute": _blob(shape, 30, 125, 12, 14),
        "hidden": _blob(shape, 35, 45, 14, 18),
    }
    mod_strength = (0.8, 1.0, 1.1)
    hidden_strength = (0.0, 1.0, 1.0)
    hidden_progress = (0.0, 0.0, 0.6)  # доля «острого» эффекта, добавляемая в очаг

    bg_gain = {"R": -0.06, "RE1": -0.04, "N": 0.02, "S1": -0.05}

    scenes = []
    for t, date in enumerate(DATES):
        bands = {}
        for b, base in HEALTHY.items():
            val = base * (1.0 + bg_gain[b] * g)
            val = val + mod_strength[t] * truth["moderate"] * _EFFECT["moderate"][b]
            val = val + truth["acute"] * _EFFECT["acute"][b]
            val = val + hidden_strength[t] * truth["hidden"] * _EFFECT["hidden"][b]
            val = val + 0.5 * hidden_progress[t] * truth["hidden"] * _EFFECT["acute"][b]
            val = val * (1.0 + 0.012 * rng.standard_normal(shape))  # шум датчика
            bands[b] = val

        if t == 2:  # «облако»
            for b in bands:
                bands[b][100:112, 120:150] = np.nan

        scenes.append(SyntheticScene(
            date=date, bands=bands,
            truth={k: v * (hidden_strength[t] if k == "hidden" else 1.0) for k, v in truth.items()},
            geo=geo,
        ))
    return scenes


def write_demo_scenes(out_dir: str | Path, seed: int = 0) -> list[Path]:
    """Пишет сцены как 4-канальные uint16 GeoTIFF (R, RE1, N, S1; DN = отражение*10000)."""
    out_dir = Path(out_dir)
    paths = []
    for sc in make_scenes(seed):
        stack = np.stack([sc.bands[b] for b in ("R", "RE1", "N", "S1")])
        dn = np.where(np.isfinite(stack), np.clip(np.round(stack * 10_000), 1, 65535), 0).astype(np.uint16)
        path = out_dir / f"scene_{sc.date}.tif"
        write_geotiff(path, dn, sc.geo, nodata=0)
        paths.append(path)
    return paths
