"""Конвейер модуля 2: сцены (GeoTIFF) -> индексы -> скрытый стресс -> зоны -> GeoTIFF + JSON.

Раскладка выходной папки (``out_dir``; для общего пайплайна команды это ``data/``)::

    out_dir/zones/zones_YYYY_MM_DD.tif              маска зон (uint8: 0 вне поля, 1..N, 1 = худшее состояние;
                                                    облачные пиксели поля — класс ближайшей зоны)
    out_dir/stats/index_stats_YYYY_MM_DD.json       статистика даты + early_stress_alert (читает M4)
    out_dir/indices/NDVI_YYYY_MM_DD.tif ...         одноканальные float32 GeoTIFF по каждому индексу
    out_dir/hidden_stress/hidden_stress_YYYY_MM_DD.tif   маска скрытого стресса (1 / 0 / 255 нет данных)
    out_dir/indices_summary.json                    сводка по всем датам
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .early_stress import EarlyStressConfig, EarlyStressResult, anomalies, detect_hidden_stress, evaluate_alert
from .filters import core_mask
from .geotiff_io import GeoMeta, read_geotiff, read_geotiff_meta, write_geotiff
from .indices import INDEX_NAMES, compute_indices, valid_mask
from .stats import hidden_stress_stats, index_stats, zone_stats
from .zoning import ZoningConfig, ZoningResult, segment_zones

SCHEMA_VERSION = "1.2"
FILLED_WARN_FRACTION = 0.2  # выше этой доли поля без данных — предупреждение в index_stats
DEFAULT_BAND_ORDER = ("R", "RE1", "N", "S1")
# порядок каналов sample_scene.tif / scene_YYYY_MM_DD.tif от модуля 1 (см. data/README.md)
SENTINEL2_10_BAND_ORDER = ("B", "G", "R", "RE1", "RE2", "RE3", "N", "N2", "S1", "S2")
# каналы Sentinel-2 -> обозначения каталога Awesome Spectral Indices
S2_TO_ASI = {
    "B01": "A", "B02": "B", "B03": "G", "B04": "R", "B05": "RE1", "B06": "RE2", "B07": "RE3",
    "B08": "N", "B8A": "N2", "B09": "WV", "B11": "S1", "B12": "S2",
}
ASI_BAND_NAMES = frozenset(S2_TO_ASI.values())
INDEX_FILE_NAME = {"NDVI": "NDVI", "NDRE": "NDRE", "MSAVI2": "MSAVI", "NDMI": "NDMI"}  # имена по ТЗ
_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def parse_date(name: str) -> str:
    m = _DATE_RE.search(name)
    if not m:
        raise ValueError(f"не удалось найти дату YYYY-MM-DD в имени файла: {name}")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def scene_date(path: str | Path, meta: GeoMeta | None = None) -> str | None:
    """Дата сцены: из имени файла, иначе из тега GDAL 'date' (его пишет M1). None, если даты нет."""
    try:
        return parse_date(Path(path).name)
    except ValueError:
        pass
    try:
        meta = meta or read_geotiff_meta(path)
    except Exception:
        return None
    tag = meta.tags.get("date") or meta.tags.get("DATE")
    try:
        return parse_date(tag) if tag else None
    except ValueError:
        return None


def _asi_band_name(name: str) -> str | None:
    """'B05', 'B5', 'B05_20m', 'b8a' -> обозначение ASI ('RE1', 'N2'); имена ASI ('RE1') возвращаются как есть."""
    raw = name.strip()
    if raw in ASI_BAND_NAMES:
        return raw
    code = raw.upper().split("_")[0]
    m = re.fullmatch(r"B0*(\d{1,2})(A?)", code)
    if not m:
        return None
    return S2_TO_ASI.get(f"B{int(m.group(1)):02d}{m.group(2)}" if not m.group(2) else f"B{int(m.group(1))}A")


def resolve_band_order(
    n_bands: int, band_order: Sequence[str] | None, band_names: Sequence[str] | None
) -> tuple[str, ...]:
    """Порядок каналов сцены в обозначениях ASI.

    band_order=None — определить автоматически: по описаниям каналов в файле (B02, B03, ...), иначе
    10 каналов = формат M1 (B,G,R,RE1,RE2,RE3,N,N2,S1,S2), 4 канала = R,RE1,N,S1. Заданный band_order должен
    совпадать с числом каналов и с описаниями в файле, иначе ошибка: перепутанные каналы молча дают неверные индексы.
    """
    from_meta = None
    if band_names and len(band_names) == n_bands:
        mapped = [_asi_band_name(b) for b in band_names]
        if all(mapped) and len(set(mapped)) == len(mapped):
            from_meta = tuple(mapped)

    if band_order is None:
        if from_meta:
            return from_meta
        if n_bands == len(SENTINEL2_10_BAND_ORDER):
            return SENTINEL2_10_BAND_ORDER
        if n_bands == len(DEFAULT_BAND_ORDER):
            return DEFAULT_BAND_ORDER
        raise ValueError(
            f"в файле {n_bands} каналов без описаний: укажите порядок каналов (--bands), например R,RE1,N,S1"
        )
    band_order = tuple(band_order)
    if len(band_order) != n_bands:
        raise ValueError(
            f"в файле {n_bands} каналов, а задано {len(band_order)} ({','.join(band_order)}). "
            "Для 10-канальных сцен M1 укажите --bands s2-10 или не указывайте --bands"
        )
    if from_meta and from_meta != band_order:
        raise ValueError(
            f"заданный порядок каналов {','.join(band_order)} не совпадает с описаниями в файле "
            f"({','.join(band_names)} = {','.join(from_meta)}); уберите --bands, чтобы взять порядок из файла"
        )
    return band_order


def _auto_scale(arr: np.ndarray, nodata: float | None) -> float:
    """Делитель DN -> отражение: 10000 для целых; для float — 10000, если значения явно не 0..1."""
    if arr.dtype.kind in "iu":
        return 10_000.0
    vals = arr[np.isfinite(arr)]
    if nodata is not None and np.isfinite(nodata):
        vals = vals[vals != nodata]
    vals = vals[vals > 0]
    if vals.size and float(np.percentile(vals, 99)) > 5.0:
        return 10_000.0
    return 1.0


def field_mask_from_raw(arr: np.ndarray, nodata: float | None) -> np.ndarray | None:
    """Контур поля по сырому стеку [C, H, W]: вне поля все каналы равны nodata (у M1 — 0).

    Пиксель с NaN (облако, тень) внутри поля полем остаётся. Если nodata не задан или сам NaN, отличить
    «вне поля» от «облака» нельзя — возвращается None, и контуром считаются пиксели с данными.
    """
    if nodata is None or not np.isfinite(nodata):
        return None
    return ~np.all(arr == nodata, axis=0)


def read_scene_with_field(
    path: str | Path,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
) -> tuple[dict[str, np.ndarray], GeoMeta, dict, np.ndarray | None]:
    """Как read_scene, плюс контур поля (bool [H, W] или None, см. field_mask_from_raw)."""
    arr, geo = read_geotiff(path)
    bands, info = _bands_from_array(arr, geo, band_order, scale)
    return bands, geo, info, field_mask_from_raw(arr, geo.nodata)


def read_scene(
    path: str | Path,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
) -> tuple[dict[str, np.ndarray], GeoMeta, dict]:
    """Читает сцену: каналы как отражение 0..1 (NaN — нет данных), геопривязка и сведения о разборе."""
    bands, geo, info, _ = read_scene_with_field(path, band_order, scale)
    return bands, geo, info


def _bands_from_array(
    arr: np.ndarray, geo: GeoMeta, band_order: Sequence[str] | None, scale: float | None
) -> tuple[dict[str, np.ndarray], dict]:
    order = resolve_band_order(arr.shape[0], band_order, geo.band_names)
    factor = float(scale) if scale is not None else _auto_scale(arr, geo.nodata)
    bands = {}
    for i, name in enumerate(order):
        raw = arr[i]
        band = raw.astype(np.float64)
        if geo.nodata is not None:
            band[raw == geo.nodata] = np.nan
        bands[name] = band / factor
    info = {
        "band_order": list(order),
        "band_names_in_file": list(geo.band_names) if geo.band_names else None,
        "scale": factor,
        "dtype": str(arr.dtype),
    }
    return bands, info


def load_scene(
    path: str | Path,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
) -> tuple[dict[str, np.ndarray], GeoMeta]:
    """Как read_scene, но без сведений о разборе."""
    bands, geo, _ = read_scene(path, band_order, scale)
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
    field: np.ndarray | None = None


def _mean(a: np.ndarray) -> float | None:
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 6) if v.size else None


def analyze_scene(
    date: str,
    bands: dict[str, np.ndarray],
    geo: GeoMeta,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    prev_anomalies: tuple[np.ndarray, np.ndarray] | None = None,
    backend: str = "numpy",
    prev_indices: tuple[np.ndarray, np.ndarray] | None = None,
    field: np.ndarray | None = None,
) -> SceneResult:
    """Анализ одной даты в памяти (без файлов).

    prev_anomalies — (z_ndre, z_ndvi) прошлой даты; prev_indices — (NDRE, NDVI) прошлой даты на той же сетке.
    Оба нужны только для динамики и необязательны.
    field — контур поля (bool [H, W]); пиксели поля без данных (облако, тень) получают в маске зон класс
    ближайшей зоны, чтобы площадь зон совпадала с площадью поля на всех датах. None — контур = пиксели с данными. Падение средних NDRE/NDVI между датами считается только по
    пикселям, видимым на обеих датах: иначе облако на одной из дат сдвигает среднее и даёт ложную тревогу.
    """
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    indices = compute_indices(bands, INDEX_NAMES, backend=backend)
    valid = valid_mask(bands, INDEX_NAMES)
    if field is None or field.shape != valid.shape:
        field = valid
    else:
        field = np.asarray(field, dtype=bool) | valid
    gaps = field & ~valid  # внутри поля, но без данных: облако, тень, битые значения
    core = core_mask(valid, early_cfg.edge_buffer_px)  # краевой пояс (смешанные пиксели) не ищем
    prev_values = None
    if prev_indices is not None and prev_indices[0].shape == valid.shape:
        prev_core = core_mask(np.isfinite(prev_indices[0]) & np.isfinite(prev_indices[1]), early_cfg.edge_buffer_px)
        prev_values = tuple(np.where(prev_core, a, np.nan) for a in prev_indices)
    early = detect_hidden_stress(
        np.where(core, indices["NDRE"], np.nan), np.where(core, indices["NDVI"], np.nan),
        early_cfg, prev=prev_anomalies, prev_values=prev_values,
    )
    zoning = segment_zones(indices, zoning_cfg, field=field)

    area = geo.pixel_area_m2
    h, w = valid.shape
    n_valid = int(valid.sum())
    n_field = int(field.sum())
    n_gap = int(gaps.sum())
    idx_stats = {name: index_stats(arr) for name, arr in indices.items()}
    hidden = hidden_stress_stats(early.mask, valid, indices, early.z_ndre, early.z_ndvi, area)
    # площадь по правилам (могут пересекаться): spatial — одна дата, temporal/decline — динамика
    hidden["by_rule_ha"] = {
        k: (round(int(m.sum()) * area / 10_000.0, 4) if area else None) for k, m in (early.by_rule or {}).items()
    }

    pair_prev = pair_cur = None
    common_fraction = None
    if prev_indices is not None and prev_indices[0].shape == valid.shape:
        p_ndre, p_ndvi = prev_indices
        common = (np.isfinite(p_ndre) & np.isfinite(p_ndvi)
                  & np.isfinite(indices["NDRE"]) & np.isfinite(indices["NDVI"]))
        common_fraction = round(float(common.sum()) / n_valid, 4) if n_valid else 0.0
        if common_fraction >= early_cfg.alert_min_overlap:
            pair_prev = (_mean(p_ndre[common]), _mean(p_ndvi[common]))
            pair_cur = (_mean(indices["NDRE"][common]), _mean(indices["NDVI"][common]))
    alert = evaluate_alert(
        hidden["fraction"],
        pair_cur[0] if pair_cur else idx_stats["NDRE"]["mean"],
        pair_cur[1] if pair_cur else idx_stats["NDVI"]["mean"],
        pair_prev, early_cfg,
    )
    alert["common_fraction"] = common_fraction
    record = {
        "date": date,
        "grid": {"width": w, "height": h, "pixel_area_m2": area, "crs_epsg": geo.epsg},
        "valid_pixels": n_valid,
        "valid_fraction": round(n_valid / (h * w), 6),
        "field_pixels": n_field,
        "field_area_ha": round(n_field * area / 10_000.0, 4) if area else None,
        "observed_area_ha": round(n_valid * area / 10_000.0, 4) if area else None,
        "filled_pixels": n_gap,
        "filled_area_ha": round(n_gap * area / 10_000.0, 4) if area else None,
        "filled_percent": round(100.0 * n_gap / n_field, 2) if n_field else 0.0,
        "indices": idx_stats,
        "zones": {
            "n_classes": zoning.n_classes,
            "silhouette": {str(k): round(v, 4) for k, v in zoning.silhouette.items()} or None,
            "filled": "пиксели поля без данных (облако/тень) получают класс ближайшей зоны; "
                      "их площадь по классам — filled_area_ha",
            "classes": zone_stats(zoning.zones, zoning.labels, indices, area, filled=gaps),
        },
        "hidden_stress": hidden,
        "early_stress_alert": alert["alert"],
        "early_stress": alert,
    }
    if n_field and n_gap / n_field > FILLED_WARN_FRACTION:
        record["warnings"] = [
            f"без данных {100.0 * n_gap / n_field:.1f}% поля: классы зон там восстановлены по соседям, "
            "доверять им меньше"
        ]
    return SceneResult(date, indices, valid, zoning, early, record, field)


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
        "observed_area_ha": record.get("observed_area_ha"),
        "filled_area_ha": record.get("filled_area_ha"),
        "filled_percent": record.get("filled_percent"),
    }
    for k in ("grid", "valid_pixels", "valid_fraction", "field_pixels", "filled_pixels", "warnings",
              "indices", "zones", "hidden_stress", "early_stress", "input", "files"):
        if k in record:
            doc[k] = record[k]
    return doc


def _history_from_outputs(
    out_dir: Path, date: str, early_cfg: EarlyStressConfig, shape: tuple[int, int]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], str] | None:
    """Восстанавливает состояние ближайшей более ранней даты по уже записанным растрам индексов.

    Позволяет обрабатывать серию по одной новой дате (как вызывает main_pipeline): динамика между датами
    считается так же, как при обработке всей серии за один запуск. Возвращает
    ((z_ndre, z_ndvi), (NDRE, NDVI), дата) либо None, если подходящей даты нет.
    """
    idx_dir = out_dir / "indices"
    if not idx_dir.is_dir():
        return None
    earlier = []
    for p in idx_dir.glob("NDVI_*.tif"):
        try:
            d = parse_date(p.name[len("NDVI_"):])
        except ValueError:
            continue
        if d < date:
            earlier.append(d)
    for d in sorted(earlier, reverse=True):
        stamp = d.replace("-", "_")
        try:
            arrs = {n: read_geotiff(idx_dir / f"{n}_{stamp}.tif")[0][0] for n in INDEX_FILE_NAME.values()}
        except Exception:  # неполный набор файлов или повреждённый файл: пробуем более раннюю дату
            continue
        if arrs["NDVI"].shape != tuple(shape):
            continue
        valid = np.all([np.isfinite(a) for a in arrs.values()], axis=0)
        core = core_mask(valid, early_cfg.edge_buffer_px)
        ndre, ndvi = arrs["NDRE"], arrs["NDVI"]
        z_re, z_vi = anomalies(np.where(core, ndre, np.nan), np.where(core, ndvi, np.nan), early_cfg)
        return (z_re, z_vi), (ndre, ndvi), d
    return None


def collect_scenes(inputs: Sequence[str | Path]) -> tuple[list[tuple[Path, str]], list[str]]:
    """Сцены с датами, по возрастанию даты, и предупреждения.

    В папках берутся *.tif / *.tiff; файлы без даты (ни в имени, ни в теге 'date') и повторы даты пропускаются
    с предупреждением. Явно переданный файл без даты — ошибка (укажите --date).
    """
    found: list[tuple[Path, str]] = []
    warnings: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted({q for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF") for q in p.glob(pat)})
            for q in files:
                d = scene_date(q)
                if d is None:
                    warnings.append(f"{q.name}: нет даты ни в имени, ни в метаданных — пропущен")
                else:
                    found.append((q, d))
        else:
            d = scene_date(p)
            if d is None:
                raise ValueError(f"{p.name}: нет даты ни в имени, ни в метаданных файла — укажите --date")
            found.append((p, d))
    found.sort(key=lambda x: (x[1], x[0].name))
    unique: list[tuple[Path, str]] = []
    for p, d in found:
        if unique and unique[-1][1] == d:
            warnings.append(f"{p.name}: дата {d} уже есть ({unique[-1][0].name}) — пропущен")
            continue
        unique.append((p, d))
    return unique, warnings


def collect_scene_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Совместимость: только пути сцен по возрастанию даты."""
    return [p for p, _ in collect_scenes(inputs)[0]]


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)


def run_pipeline(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
    save_indices: bool = True,
    date_override: str | None = None,
    use_history: bool = False,
) -> dict:
    """Обрабатывает все сцены по датам и пишет артефакты в out_dir (раскладка — в docstring модуля).

    band_order=None — порядок каналов определяется по каждому файлу (см. resolve_band_order).
    date_override — дата YYYY-MM-DD для случая, когда на входе ровно одна сцена.
    use_history — для первой обработанной сцены взять предыдущую дату из уже записанных в out_dir результатов
    (растры индексов), чтобы считать динамику при поступлении новых снимков по одному.
    Сводка indices_summary.json дополняется: даты, обработанные ранее, в ней сохраняются.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoning_cfg = zoning_cfg or ZoningConfig()
    early_cfg = early_cfg or EarlyStressConfig()

    warnings: list[str] = []
    if date_override is not None:
        paths: list[Path] = []
        for item in inputs:
            p = Path(item)
            paths += (sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff"))) if p.is_dir() else [p])
        if len(paths) != 1:
            raise ValueError("date_override можно использовать только когда на входе ровно одна сцена")
        scenes = [(paths[0], parse_date(date_override))]
    else:
        scenes, warnings = collect_scenes(inputs)
    for msg in warnings:
        print(f"предупреждение: {msg}", file=sys.stderr)
    if not scenes:
        raise FileNotFoundError("не найдено ни одной сцены GeoTIFF с датой")

    dates: dict[str, dict] = {}
    prev_anom: tuple[np.ndarray, np.ndarray] | None = None
    prev_idx: tuple[np.ndarray, np.ndarray] | None = None
    prev_date: str | None = None
    for path, date in scenes:
        try:
            bands, geo, info, field = read_scene_with_field(path, band_order, scale)
            if use_history and prev_anom is None:
                hist = _history_from_outputs(out_dir, date, early_cfg, next(iter(bands.values())).shape)
                if hist is not None:
                    prev_anom, prev_idx, prev_date = hist
            res = analyze_scene(date, bands, geo, zoning_cfg, early_cfg, prev_anom, backend, prev_idx, field)
            res.record["early_stress"]["compared_with"] = prev_date if prev_anom is not None else None
            res.record["input"] = info
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
        prev_idx = (res.indices["NDRE"], res.indices["NDVI"])
        prev_date = date

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "band_order": list(band_order) if band_order else "auto",
            "scale": scale if scale is not None else "auto",
            "backend": backend,
            "indices": list(INDEX_NAMES),
            "zoning": asdict(zoning_cfg),
            "early_stress": asdict(early_cfg),
            "zone_raster": "uint8; 0 = вне поля, 1 = худшее состояние (низкий NDRE) ... N = норма; "
                           "облачные пиксели поля заполнены классом ближайшей зоны",
            "hidden_stress_raster": "uint8; 1 = скрытый стресс, 0 = нет, 255 = нет данных (вне поля или облако)",
        },
        "warnings": warnings,
        "dates": dates,
    }
    summary_path = out_dir / "indices_summary.json"
    if summary_path.exists():  # сохраняем даты, обработанные в прошлых запусках
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8")).get("dates", {})
            merged = {d: r for d, r in old.items() if d not in dates and "error" not in r}
            merged.update(dates)
            summary["dates"] = dict(sorted(merged.items()))
        except (OSError, ValueError):
            pass
    _dump_json(summary_path, summary)
    return summary


def default_out_dir(scene_tif: str | Path) -> Path:
    """data/scenes/scene_X.tif -> data/ (раскладка из плана); иначе папка m2_output рядом со сценой."""
    parent = Path(scene_tif).resolve().parent
    return parent.parent if parent.name == "scenes" else parent / "m2_output"


def analyze_field(
    scene_tif: str | Path,
    out_dir: str | Path | None = None,
    date: str | None = None,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
) -> dict:
    """Анализ одной сцены (точка входа по ТЗ КТ2): возвращает запись даты со ссылками на файлы.

    out_dir по умолчанию: для data/scenes/scene_YYYY_MM_DD.tif это data/ (результаты в data/zones, data/stats).
    Дата берётся из date, иначе из имени файла, иначе из тега 'date' в метаданных. Каналы определяются по файлу.
    Если в out_dir уже есть результаты более ранней даты (растры в out_dir/indices), динамика считается
    относительно неё, поэтому сцены можно подавать по одной по мере поступления;
    запись early_stress.compared_with указывает дату сравнения.
    """
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(scene_tif)
    the_date = parse_date(date) if date else scene_date(scene_tif)
    if the_date is None:
        raise ValueError(f"{Path(scene_tif).name}: нет даты ни в имени, ни в метаданных — передайте date=")
    summary = run_pipeline(
        [scene_tif], out_dir, band_order=band_order, scale=scale, zoning_cfg=zoning_cfg,
        early_cfg=early_cfg, backend=backend, date_override=the_date, use_history=True,
    )
    record = summary["dates"][the_date]
    if "error" in record:
        raise RuntimeError(f"{scene_tif}: {record['error']}")
    return record


def processed_dates(out_dir: str | Path) -> list[str]:
    """Даты, для которых в out_dir уже есть stats/index_stats_*.json и маска zones/zones_*.tif."""
    out_dir = Path(out_dir)
    done = []
    for p in (out_dir / "stats").glob("index_stats_*.json"):
        try:
            d = parse_date(p.stem[len("index_stats_"):])
        except ValueError:
            continue
        if (out_dir / "zones" / f"zones_{d.replace('-', '_')}.tif").exists():
            done.append(d)
    return sorted(done)


def process_new_scenes(
    scenes: str | Path | Sequence[str | Path] = "data/scenes",
    out_dir: str | Path | None = None,
    band_order: Sequence[str] | None = None,
    scale: float | None = None,
    zoning_cfg: ZoningConfig | None = None,
    early_cfg: EarlyStressConfig | None = None,
    backend: str = "numpy",
) -> dict:
    """Обработать только новые сцены (точка входа для main_pipeline.py, КТ3).

    Сцены, для которых в out_dir уже есть zones и stats, пропускаются; динамика первой новой даты считается
    от ближайшей более ранней уже обработанной даты. Повторный вызов без новых снимков ничего не меняет.
    out_dir по умолчанию: для папки data/scenes это data/.
    Возвращает {"processed": [...], "skipped": [...], "errors": {...}, "warnings": [...]}.
    """
    inputs = [scenes] if isinstance(scenes, (str, Path)) else list(scenes)
    if out_dir is None:
        first = Path(inputs[0])
        out_dir = first.resolve().parent if first.is_dir() and first.name == "scenes" else default_out_dir(first)
    out_dir = Path(out_dir)
    found, warnings = collect_scenes(inputs)
    done = set(processed_dates(out_dir))
    new = [(p, d) for p, d in found if d not in done]
    skipped = sorted(d for _, d in found if d in done)
    if done and new:
        latest = max(done)
        late = [d for _, d in new if d < latest]
        if late:
            warnings.append(
                f"даты {', '.join(late)} раньше уже обработанной {latest}: динамика более поздних дат посчитана "
                "без них — для точного ряда перезапустите всю серию (stress_core run)"
            )
    for msg in warnings:
        print(f"предупреждение: {msg}", file=sys.stderr)
    result = {"processed": [], "skipped": skipped, "errors": {}, "warnings": warnings, "out_dir": str(out_dir)}
    if not new:
        return result
    summary = run_pipeline(
        [p for p, _ in new], out_dir, band_order=band_order, scale=scale, zoning_cfg=zoning_cfg,
        early_cfg=early_cfg, backend=backend, use_history=True,
    )
    for _, d in new:
        rec = summary["dates"].get(d, {})
        if "error" in rec:
            result["errors"][d] = rec["error"]
        else:
            result["processed"].append(d)
    return result
