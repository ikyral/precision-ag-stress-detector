"""Конвейер модуля 2: сцены (GeoTIFF) -> индексы -> скрытый стресс -> зоны -> GeoTIFF + JSON.

Раскладка выходной папки (``out_dir``; для общего пайплайна команды это ``data/``)::

    out_dir/zones/zones_YYYY_MM_DD.tif              маска зон (uint8: 0 фон, 1..N, 1 = худшее состояние)
    out_dir/stats/index_stats_YYYY_MM_DD.json       статистика даты + early_stress_alert (читает M4)
    out_dir/indices/NDVI_YYYY_MM_DD.tif ...         одноканальные float32 GeoTIFF по каждому индексу
    out_dir/hidden_stress/hidden_stress_YYYY_MM_DD.tif   маска скрытого стресса (1 / 0 / 255 нет данных)
    out_dir/indices_summary.json                    сводка по всем датам
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .filters import core_mask
from .early_stress import EarlyStressConfig, EarlyStressResult, detect_hidden_stress, evaluate_alert
from .geotiff_io import GeoMeta, read_geotiff, write_geotiff
from .indices import INDEX_NAMES, compute_indices, valid_mask
from .stats import hidden_stress_stats, index_stats, zone_stats
from .zoning import ZoningConfig, ZoningResult, segment_zones

SCHEMA_VERSION = "1.1"
DEFAULT_BAND_ORDER = ("R", "RE1", "N", "S1")
# порядок каналов sample_scene.tif / scene_YYYY_MM_DD.tif от модуля 1 (см. data/README.md)
SENTINEL2_10_BAND_ORDER = ("B", "G", "R", "RE1", "RE2", "RE3", "N", "N2", "S1", "S2")
INDEX_FILE_NAME = {"NDVI": "NDVI", "NDRE": "NDRE", "MSAVI2": "MSAVI", "NDMI": "NDMI"}  # имена по ТЗ
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


def save_index_rasters(
    indices: Mapping[str, np.ndarray], geo: GeoMeta, out_dir: str | Path, stamp: str | None = None
) -> dict[str, str]:
    """Пишет каждый индекс отдельным одноканальным float32 GeoTIFF: NDVI.tif или NDVI_YYYY_MM_DD.tif."""
    out_dir = Path(out_dir)
    files = {}
    for name, arr in indices.items():
        base = INDEX_FILE_NAME.get(name, name)
        fname = f"{base}_{stamp}.tif" if stamp else f"{base}.tif"
        write_geotiff(out_dir / fname, np.asarray(arr, dtype=np.float32), geo, nodata=float("nan"))
        files[base] = fname
    return files


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
    prev_means: tuple[float | None, float | None] | None = None,
) -> SceneResult:
    """Анализ одной даты в памяти (без файлов).

    prev_anomalies — (z_ndre, z_ndvi) прошлой даты; prev_means — (средний NDRE, средний NDVI) прошлой даты.
    Оба нужны только для динамики и необязательны.
    """
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    indices = compute_indices(bands, INDEX_NAMES, backend=backend)
    valid = valid_mask(bands, INDEX_NAMES)
    core = core_mask(valid, early_cfg.edge_buffer_px)  # краевой пояс (смешанные пиксели) не ищем
    early = detect_hidden_stress(
        np.where(core, indices["NDRE"], np.nan), np.where(core, indices["NDVI"], np.nan),
        early_cfg, prev=prev_anomalies,
    )
    zoning = segment_zones(indices, zoning_cfg)

    area = geo.pixel_area_m2
    h, w = valid.shape
    n_valid = int(valid.sum())
    idx_stats = {name: index_stats(arr) for name, arr in indices.items()}
    hidden = hidden_stress_stats(early.mask, valid, indices, early.z_ndre, early.z_ndvi, area)
    alert = evaluate_alert(
        hidden["fraction"], idx_stats["NDRE"]["mean"], idx_stats["NDVI"]["mean"], prev_means, early_cfg
    )
    record = {
        "date": date,
        "grid": {"width": w, "height": h, "pixel_area_m2": area, "crs_epsg": geo.epsg},
        "valid_pixels": n_valid,
        "valid_fraction": round(n_valid / (h * w), 6),
        "field_area_ha": round(n_valid * area / 10_000.0, 4) if area else None,
        "indices": idx_stats,
        "zones": {
            "n_classes": zoning.n_classes,
            "silhouette": {str(k): round(v, 4) for k, v in zoning.silhouette.items()} or None,
            "classes": zone_stats(zoning.zones, zoning.labels, indices, area),
        },
        "hidden_stress": hidden,
        "early_stress_alert": alert["alert"],
        "early_stress": alert,
    }
    return SceneResult(date, indices, valid, zoning, early, record)


def build_index_stats_doc(record: dict) -> dict:
    """Содержимое index_stats_YYYY_MM_DD.json: плоские поля для дашборда M4 + полная статистика."""
    ind = record["indices"]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "date": record["date"],
        "source_file": record.get("source_file"),
        "mean_ndvi": ind["NDVI"]["mean"],
        "mean_ndre": ind["NDRE"]["mean"],
        "mean_msavi": ind["MSAVI2"]["mean"],
        "mean_ndmi": ind["NDMI"]["mean"],
        "early_stress_alert": record["early_stress_alert"],
        "early_stress_area_ha": record["hidden_stress"]["area_ha"],
        "early_stress_percent": (
            None if record["hidden_stress"]["fraction"] is None
            else round(record["hidden_stress"]["fraction"] * 100.0, 2)
        ),
        "field_area_ha": record["field_area_ha"],
    }
    for k in ("grid", "valid_pixels", "valid_fraction", "indices", "zones", "hidden_stress", "early_stress", "files"):
        if k in record:
            doc[k] = record[k]
    return doc


def collect_scene_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
        else:
            paths.append(p)
    return sorted(paths, key=lambda q: parse_date(q.name))


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)


def run_pipeline(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
    save_indices: bool = True,
    date_override: str | None = None,
) -> dict:
    """Обрабатывает все сцены по датам и пишет артефакты в out_dir (раскладка — в docstring модуля).

    date_override — дата YYYY-MM-DD для случая, когда на входе ровно один файл без даты в имени.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    if date_override is not None:
        paths = []
        for item in inputs:
            p = Path(item)
            paths += (sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff"))) if p.is_dir() else [p])
        if len(paths) != 1:
            raise ValueError("date_override можно использовать только когда на входе ровно одна сцена")
        date_override = parse_date(date_override)
    else:
        paths = collect_scene_paths(inputs)
    if not paths:
        raise FileNotFoundError("не найдено ни одного GeoTIFF")

    dates: dict[str, dict] = {}
    prev_anom: tuple[np.ndarray, np.ndarray] | None = None
    prev_means: tuple[float | None, float | None] | None = None
    for path in paths:
        date = date_override or parse_date(path.name)
        try:
            bands, geo = load_scene(path, band_order, scale)
            res = analyze_scene(date, bands, geo, zoning_cfg, early_cfg, prev_anom, backend, prev_means)
        except Exception as exc:  # одна плохая сцена не должна ронять весь прогон
            dates[date] = {"date": date, "source_file": path.name, "error": f"{type(exc).__name__}: {exc}"}
            continue

        stamp = date.replace("-", "_")  # контракт M2 -> M3: zones_YYYY_MM_DD.tif
        zones_rel = f"zones/zones_{stamp}.tif"
        hidden_rel = f"hidden_stress/hidden_stress_{stamp}.tif"
        write_geotiff(out_dir / zones_rel, res.zoning.zones, geo, nodata=0)
        hidden = np.full(res.valid.shape, 255, dtype=np.uint8)  # 255 — нет данных
        hidden[res.valid] = res.early.mask[res.valid].astype(np.uint8)
        write_geotiff(out_dir / hidden_rel, hidden, geo, nodata=255)
        files = {"zones": zones_rel, "hidden_stress": hidden_rel, "stats": f"stats/index_stats_{stamp}.json"}
        if save_indices:
            idx_files = save_index_rasters(res.indices, geo, out_dir / "indices", stamp)
            files["indices"] = {k: f"indices/{v}" for k, v in idx_files.items()}

        res.record["source_file"] = path.name
        res.record["files"] = files
        _dump_json(out_dir / files["stats"], build_index_stats_doc(res.record))
        dates[date] = res.record
        prev_anom = (res.early.z_ndre, res.early.z_ndvi)
        prev_means = (res.record["indices"]["NDRE"]["mean"], res.record["indices"]["NDVI"]["mean"])

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
            "zone_raster": "uint8; 0 = нет данных, 1 = худшее состояние (низкий NDRE) ... N = норма",
            "hidden_stress_raster": "uint8; 1 = скрытый стресс, 0 = нет, 255 = нет данных",
        },
        "dates": dates,
    }
    _dump_json(out_dir / "indices_summary.json", summary)
    return summary


def analyze_field(
    scene_tif: str | Path,
    out_dir: str | Path,
    date: str | None = None,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
) -> dict:
    """Анализ одной сцены (точка входа по ТЗ КТ2): возвращает запись даты со ссылками на файлы.

    Дата берётся из имени файла (scene_YYYY_MM_DD.tif) либо из параметра date. Динамику между датами
    даёт только run_pipeline по серии: здесь прошлых дат нет, поэтому флаг early_stress_alert опирается
    лишь на пространственный признак.
    """
    summary = run_pipeline(
        [scene_tif], out_dir, band_order=band_order, scale=scale, zoning_cfg=zoning_cfg,
        early_cfg=early_cfg, backend=backend, date_override=date,
    )
    (record,) = summary["dates"].values()
    if "error" in record:
        raise RuntimeError(f"{scene_tif}: {record['error']}")
    return record
