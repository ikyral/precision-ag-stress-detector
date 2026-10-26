"""Конвейер модуля 2: сцены (GeoTIFF) -> индексы -> скрытый стресс -> зоны -> GeoTIFF + JSON."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .early_stress import EarlyStressConfig, EarlyStressResult, detect_hidden_stress
from .geotiff_io import GeoMeta, read_geotiff, write_geotiff
from .indices import INDEX_NAMES, compute_indices, valid_mask
from .stats import hidden_stress_stats, index_stats, zone_stats
from .zoning import ZoningConfig, ZoningResult, segment_zones

SCHEMA_VERSION = "1.0"
DEFAULT_BAND_ORDER = ("R", "RE1", "N", "S1")
_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def parse_date(name: str) -> str:
    m = _DATE_RE.search(name)
    if not m:
        raise ValueError(f"не удалось найти дату YYYY-MM-DD в имени файла: {name}")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def load_scene(
    path: str | Path,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    scale: float | None = None,
) -> tuple[dict[str, np.ndarray], GeoMeta]:
    """Читает GeoTIFF и возвращает каналы как отражение 0..1 (NaN — нет данных).

    scale — делитель DN. По умолчанию 10000 для целочисленных файлов и 1 для float.
    """
    arr, geo = read_geotiff(path)
    if arr.shape[0] < len(band_order):
        raise ValueError(
            f"{path}: в файле {arr.shape[0]} каналов, а band_order требует {len(band_order)}"
        )
    bands = {}
    for i, name in enumerate(band_order):
        raw = arr[i]
        factor = scale if scale is not None else (10_000.0 if raw.dtype.kind in "iu" else 1.0)
        band = raw.astype(np.float64)
        if geo.nodata is not None:
            band[raw == geo.nodata] = np.nan
        bands[name] = band / factor
    return bands, geo


@dataclass
class SceneResult:
    date: str
    indices: dict[str, np.ndarray]
    valid: np.ndarray
    zoning: ZoningResult
    early: EarlyStressResult
    record: dict


def analyze_scene(
    date: str,
    bands: dict[str, np.ndarray],
    geo: GeoMeta,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    prev_anomalies: tuple[np.ndarray, np.ndarray] | None = None,
    backend: str = "numpy",
) -> SceneResult:
    """Анализ одной даты в памяти (без файлов)."""
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    indices = compute_indices(bands, INDEX_NAMES, backend=backend)
    valid = valid_mask(bands, INDEX_NAMES)
    early = detect_hidden_stress(indices["NDRE"], indices["NDVI"], early_cfg, prev=prev_anomalies)
    zoning = segment_zones(indices, zoning_cfg)

    area = geo.pixel_area_m2
    h, w = valid.shape
    n_valid = int(valid.sum())
    record = {
        "date": date,
        "grid": {"width": w, "height": h, "pixel_area_m2": area, "crs_epsg": geo.epsg},
        "valid_pixels": n_valid,
        "valid_fraction": round(n_valid / (h * w), 6),
        "indices": {name: index_stats(arr) for name, arr in indices.items()},
        "zones": {
            "n_classes": zoning.n_classes,
            "silhouette": {str(k): round(v, 4) for k, v in zoning.silhouette.items()} or None,
            "classes": zone_stats(zoning.zones, zoning.labels, indices, area),
        },
        "hidden_stress": hidden_stress_stats(
            early.mask, valid, indices, early.z_ndre, early.z_ndvi, area
        ),
    }
    return SceneResult(date, indices, valid, zoning, early, record)


def collect_scene_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
        else:
            paths.append(p)
    return sorted(paths, key=lambda q: parse_date(q.name))


def run_pipeline(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
    save_indices: bool = False,
) -> dict:
    """Обрабатывает все сцены по датам и пишет артефакты в out_dir.

    Для каждой даты:  zones_<date>.tif (uint8, 0 = нет данных, 1..N = классы),
                      hidden_stress_<date>.tif (uint8: 0/1, 255 = нет данных),
                      indices_<date>.tif (float32, опционально).
    Общий файл:       indices_summary.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    paths = collect_scene_paths(inputs)
    if not paths:
        raise FileNotFoundError("не найдено ни одного GeoTIFF")

    dates: dict[str, dict] = {}
    prev: tuple[np.ndarray, np.ndarray] | None = None
    for path in paths:
        date = parse_date(path.name)
        try:
            bands, geo = load_scene(path, band_order, scale)
            res = analyze_scene(date, bands, geo, zoning_cfg, early_cfg, prev, backend)
        except Exception as exc:  # одна плохая сцена не должна ронять весь прогон
            dates[date] = {"date": date, "source_file": path.name, "error": f"{type(exc).__name__}: {exc}"}
            continue

        stamp = date.replace("-", "_")  # контракт M2 -> M3: zones_YYYY_MM_DD.tif
        zones_name = f"zones_{stamp}.tif"
        hidden_name = f"hidden_stress_{stamp}.tif"
        write_geotiff(out_dir / zones_name, res.zoning.zones, geo, nodata=0)
        hidden = np.full(res.valid.shape, 255, dtype=np.uint8)  # 255 — нет данных
        hidden[res.valid] = res.early.mask[res.valid].astype(np.uint8)
        write_geotiff(out_dir / hidden_name, hidden, geo, nodata=255)
        files = {"zones": zones_name, "hidden_stress": hidden_name}
        if save_indices:
            idx_name = f"indices_{stamp}.tif"
            stack = np.stack([res.indices[n] for n in INDEX_NAMES]).astype(np.float32)
            write_geotiff(out_dir / idx_name, stack, geo, nodata=float("nan"))
            files["indices"] = idx_name

        res.record["source_file"] = path.name
        res.record["files"] = files
        dates[date] = res.record
        prev = (res.early.z_ndre, res.early.z_ndvi)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "band_order": list(band_order),
            "scale": scale,
            "backend": backend,
            "indices": list(INDEX_NAMES),
            "zoning": asdict(zoning_cfg),
            "early_stress": asdict(early_cfg),
            "zone_raster": "uint8; 0 = нет данных, 1 = острый стресс ... N = норма",
            "hidden_stress_raster": "uint8; 1 = скрытый стресс, 0 = нет, 255 = нет данных",
        },
        "dates": dates,
    }
    with open(out_dir / "indices_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, allow_nan=False)
    return summary
