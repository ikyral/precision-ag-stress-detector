"""Сборка и запись scene_YYYY_MM_DD.tif: float32 отражение 0..1, 0 вне поля, NaN облака."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import BANDS
from .grid import Grid


def scene_filename(date: str) -> str:
    """'2026-08-07' -> 'scene_2026_08_07.tif'."""
    return f"scene_{date.replace('-', '_')}.tif"


def to_reflectance(dn: np.ndarray, offset: int) -> np.ndarray:
    """Цифровые значения L2A -> отражение: минус смещение, минимум 1 (как в эталоне), / 10000."""
    return (np.clip(dn.astype(np.float32) - offset, 1, 65535) / np.float32(10000.0)).astype(np.float32)


def assemble(refl: np.ndarray, inside: np.ndarray, nan_mask: np.ndarray | None) -> np.ndarray:
    """[bands, H, W]: вне контура 0, облака внутри контура NaN, остальное — отражение."""
    out = np.where(inside[None], refl, np.float32(0)).astype(np.float32)
    if nan_mask is not None:
        out[:, nan_mask & inside] = np.nan
    return out


def write_scene(path: str | Path, data: np.ndarray, grid: Grid, tags: dict) -> None:
    import rasterio

    prof = dict(driver="GTiff", count=len(BANDS), dtype="float32", crs=grid.crs,
                transform=grid.transform, width=grid.width, height=grid.height,
                nodata=0, compress="deflate")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tif.part")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data)
        dst.descriptions = tuple(BANDS)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    tmp.replace(path)  # неполный файл не останется под именем scene_*.tif
