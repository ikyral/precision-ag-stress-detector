"""
Precision Ag Stress Detector — Модуль 4. Веб-кабинет агронома.

Запуск:
    python generate_mock_data.py     # один раз, создаёт mock_data/
    streamlit run app.py

Источники данных (переключаются в боковой панели):
  * «Данные проекта» — папка data/ репозитория (ищется вверх от app.py)
    или путь из переменной окружения PAS_DATA_DIR;
  * «Мок-данные» — ./mock_data, созданная generate_mock_data.py.

Что читается из папки данных (всё необязательно, дашборд показывает то, что есть):
  field_boundary.geojson | samples/field_boundary.geojson   граница поля (M1)
  stats/index_stats_YYYY_MM_DD.json                          статистика M2 (schema 1.1/1.2)
  indices/{NDVI|NDRE|MSAVI|NDMI}_YYYY_MM_DD.tif              растры индексов M2 (нужен rasterio)
  zones/zones_YYYY_MM_DD.tif                                 маска зон M2 (нужен rasterio)
  vra/ | prescriptions/ | . :
      prescription_YYYY_MM_DD.geojson  (zone_class, area_ha, fertilizer_rate | rate_kg_ha)  M3
      shapefile_YYYY_MM_DD.zip, metrics_YYYY_MM_DD.json                                     M3
  overlays/{INDEX|RGB|HIDDEN}_YYYY_MM_DD.png + meta.json    готовые PNG (M1 main_pipeline, мок-данные)

Пока файлов M3 нет, экономика считается по площадям зон из index_stats (M2).
"""

from __future__ import annotations

import copy
import html
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

try:
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    HAS_RASTERIO = True
except ImportError:  # дашборд работает и без rasterio (мок-данные в PNG)
    HAS_RASTERIO = False

try:
    from PIL import Image
except ImportError:
    Image = None

# ----------------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
VRA_SUBDIRS = ("vra", "prescriptions", "prescription", ".")
DATE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})")

ZONE_COLORS = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}
ZONE_SHORT = {1: "Острый стресс", 2: "Умеренный стресс", 3: "Норма"}
ZONE_NAMES = {1: "Острый стресс (макс. доза)", 2: "Умеренный стресс (средняя доза)", 3: "Норма (без обработки)"}
DEFAULT_RATE_FACTORS = {1: 1.0, 2: 80 / 150, 3: 0.0}  # доли от базовой нормы; не округляем при расчёте
DEFAULT_BASE_RATE, DEFAULT_PRICE = 150, 45
BASE_RATE_MAX, PRICE_MAX = 300.0, 150.0  # верх слайдеров (расширяется до значения из файла)
MAX_RATE_FACTOR = 2.0  # доля нормы выше этого — скорее всего проценты вместо долей
MAX_OVERLAY_PX = 1600  # растр крупнее не увеличиваем «блоками»

LAYERS = {
    "Карта предписаний VRA": None,
    "Снимок Sentinel-2 (RGB)": "RGB",
    "Скрытый стресс (М2)": "HIDDEN",
    "Индекс раннего стресса NDRE": "NDRE",
    "Базовый индекс NDVI": "NDVI",
    "Почвенно-скорр. индекс MSAVI": "MSAVI",
    "Водный стресс NDMI": "NDMI",
}
STATS_INDEX_KEYS = {"NDVI": ("NDVI",), "NDRE": ("NDRE",), "MSAVI": ("MSAVI2", "MSAVI"), "NDMI": ("NDMI",)}
DEFAULT_RANGES = {"NDVI": (0.2, 0.9), "NDRE": (0.1, 0.7), "MSAVI": (0.1, 0.8), "NDMI": (-0.1, 0.5)}
CMAP_VEG = [(0.0, (165, 0, 38)), (0.25, (244, 109, 67)), (0.5, (254, 224, 139)),
            (0.75, (102, 189, 99)), (1.0, (0, 104, 55))]
CMAP_WATER = [(0.0, (140, 81, 10)), (0.25, (216, 179, 101)), (0.5, (246, 232, 195)),
              (0.75, (90, 180, 172)), (1.0, (1, 102, 94))]
INDEX_CMAPS = {"NDVI": CMAP_VEG, "NDRE": CMAP_VEG, "MSAVI": CMAP_VEG, "NDMI": CMAP_WATER}
PNG_ONLY_LAYERS = {"RGB": "Sentinel-2 L2A, true color (B04, B03, B02)",  # есть только готовыми PNG М1
                   "HIDDEN": "Скрытый стресс (М2)"}  # подписи по умолчанию; схема 1.1 meta.json задаёт свои
DEFAULT_NODATA_COLOR = (128, 128, 128, 150)  # облака и тени в PNG М1 (meta.nodata_color)
DEFAULT_NODATA_LABEL = "Нет данных (облако/тень)"
DEFAULT_HIDDEN_COLOR = (231, 76, 60, 235)
CSS_RGBA_RE = re.compile(r"rgba\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*(0|1|0?\.\d+|1\.0+)\s*\)")
INDEX_SERIES = [("mean_ndvi", "NDVI", "#27ae60"), ("mean_ndre", "NDRE", "#c0392b"),
                ("mean_msavi", "MSAVI", "#8d6e3f"), ("mean_ndmi", "NDMI", "#2980b9")]

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

st.set_page_config(page_title="Precision Ag Stress Detector", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
      div[data-testid="stMetric"] {
          background: rgba(46, 204, 113, 0.07); border: 1px solid rgba(46, 204, 113, 0.28);
          border-radius: 12px; padding: 12px 16px;
      }
      div[data-testid="stMetricLabel"] p {font-size: 0.86rem; opacity: 0.8;}
      div[data-testid="stMetricValue"] {font-size: 1.3rem;}
      div[data-testid="stMetricValue"] > div {white-space: normal; overflow-wrap: anywhere;}
      .pas-alert {background: rgba(231, 76, 60, 0.12); border-left: 6px solid #e74c3c;
                  border-radius: 8px; padding: 12px 16px; margin: 4px 0 14px 0; font-size: 0.98rem;}
      .pas-alert ul {margin: 6px 0 0 0;}
      .pas-ok {background: rgba(46, 204, 113, 0.10); border-left: 6px solid #2ecc71;
               border-radius: 8px; padding: 10px 16px; margin: 4px 0 14px 0; font-size: 0.95rem;}
      .pas-legend {display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.85rem; margin-top: 4px;}
      .pas-legend span.sw {display: inline-block; width: 14px; height: 14px; border-radius: 3px;
                           margin-right: 6px; vertical-align: -2px;}
      .pas-bar {height: 12px; border-radius: 6px; margin: 6px 0 2px 0;}
      .pas-bar-labels {display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------------
def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y_%m_%d")


def fmt_date(d: str) -> str:
    return parse_date(d).strftime("%d.%m.%Y")


def fmt_num(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


# Проблемы с данными собираются за один прогон скрипта и выводятся блоком вверху страницы:
# дашборд по-прежнему показывает то, что смог прочитать, но ничего не подменяет молча.
PROBLEMS: list[str] = []


def report(msg: str) -> None:
    if msg not in PROBLEMS:
        PROBLEMS.append(msg)


def as_number(x) -> float | None:
    """Число из JSON или None (null, строка-не-число, bool, NaN)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float = 0.0):
    p = Path(path)
    return _read_json(p) if p.exists() else None


def jload(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        data = load_json(str(path), path.stat().st_mtime)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        report(f"`{path.name}` не читается как JSON ({e}) — файл пропущен.")
        return None
    if not isinstance(data, dict):
        report(f"`{path.name}`: ожидался JSON-объект, а не {type(data).__name__} — файл пропущен.")
        return None
    return data


# ----------------------------------------------------------------------------
# Источник данных
# ----------------------------------------------------------------------------
def find_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    env = os.environ.get("PAS_DATA_DIR")
    if env:
        sources[f"PAS_DATA_DIR ({Path(env).name})"] = Path(env)
    for parent in [APP_DIR, *APP_DIR.parents][:5]:
        d = parent / "data"
        if d.is_dir() and any((d / s).exists() for s in ("stats", "samples", "zones", "vra", "indices")):
            sources["Данные проекта (data/)"] = d
            break
    if (APP_DIR / "mock_data").is_dir():
        sources["Мок-данные"] = APP_DIR / "mock_data"
    return sources


class DataSource:
    def __init__(self, root: Path):
        self.root = root

    def stats_path(self, d):
        return self.root / "stats" / f"index_stats_{d}.json"

    def vra_file(self, name) -> Path | None:
        for sub in VRA_SUBDIRS:
            p = self.root / sub / name
            if p.exists():
                return p
        return None

    def dates(self) -> list[str]:
        stems = [p.stem for p in (self.root / "stats").glob("index_stats_*.json")]
        for sub in VRA_SUBDIRS:
            stems += [p.stem for p in (self.root / sub).glob("metrics_*.json")]
        found = set()
        for stem in stems:
            m = DATE_RE.search(stem)
            if not m:
                continue
            try:
                parse_date(m.group(1))
            except ValueError:
                report(f"`{stem}`: некорректная дата в имени файла — файл пропущен.")
                continue
            found.add(m.group(1))
        return sorted(found)

    def field_boundary(self):
        for p in (self.root / "field_boundary.geojson", self.root / "samples" / "field_boundary.geojson"):
            if p.exists():
                return jload(p)
        return None

    def _stats_file(self, stats: dict | None, *keys: str) -> Path | None:
        """Путь из блока files.* статистики M2 (если он есть и существует)."""
        node = stats or {}
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if not isinstance(node, str) or not node:
            return None
        p = self.root / node
        if not p.exists():
            report(f"index_stats ссылается на `{node}`, но такого файла нет — ищу по стандартному пути.")
            return None
        return p

    def index_raster(self, index: str, d: str, stats: dict | None) -> Path | None:
        p = self._stats_file(stats, "files", "indices", index) or self.root / "indices" / f"{index}_{d}.tif"
        return p if p.exists() else None

    def zones_raster(self, d: str, stats: dict | None) -> Path | None:
        p = self._stats_file(stats, "files", "zones") or self.root / "zones" / f"zones_{d}.tif"
        return p if p.exists() else None

    def overlay_png(self, index: str, d: str) -> Path | None:
        p = self.root / "overlays" / f"{index}_{d}.png"
        return p if p.exists() else None


# ----------------------------------------------------------------------------
# Растры → картинки для Leaflet (EPSG:4326)
# ----------------------------------------------------------------------------
def colorize(arr: np.ndarray, vmin: float, vmax: float, cmap) -> np.ndarray:
    valid = np.isfinite(arr)
    t = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    stops = np.array([s for s, _ in cmap])
    cols = np.array([c for _, c in cmap], dtype=float)
    rgb = np.stack([np.interp(t, stops, cols[:, k]) for k in range(3)], axis=-1)
    alpha = np.where(valid, 235, 0)[..., None]
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)


def colorize_zones(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape + (4,), dtype=np.uint8)
    for zc, hexcol in ZONE_COLORS.items():
        rgb = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
        out[arr == zc] = rgb + [190]
    return out


def upscale(rgba: np.ndarray) -> np.ndarray:
    """Увеличение «блоками», чтобы пиксели 10 м не размывались браузером.

    Крупные растры не увеличиваем: иначе картинка на десятки МБ пересобирается при каждом движении слайдера.
    """
    k = max(1, min(4, MAX_OVERLAY_PX // max(rgba.shape[:2])))
    return np.repeat(np.repeat(rgba, k, axis=0), k, axis=1) if k > 1 else rgba


@st.cache_data(show_spinner=False)
def read_geotiff_wgs84(path: str, mtime: float):
    """Читает 1-й канал GeoTIFF, перепроецирует в EPSG:4326.

    Возвращает (array float32 с NaN вместо nodata, bounds [[S, W], [N, E]]).
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("в файле не задана система координат (CRS)")
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)  # учитывает nodata и маску
        dst_crs = "EPSG:4326"
        transform, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(source=arr, destination=dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    west, north = transform * (0, 0)
    east, south = transform * (w, h)
    return dst, [[south, west], [north, east]]


def load_raster(tif: Path):
    """read_geotiff_wgs84 без падения страницы: битый, недописанный или без CRS файл → (None, None)."""
    try:
        return read_geotiff_wgs84(str(tif), tif.stat().st_mtime)
    except (rasterio.errors.RasterioError, ValueError, OSError) as e:
        report(f"`{tif.name}` не открывается: {e}")
        return None, None


def index_overlay(ds: DataSource, index: str, d: str, stats, meta, vrange):
    """(rgba, bounds, источник) для слоя индекса или (None, None, причина)."""
    png = ds.overlay_png(index, d)
    if png is not None and meta.get("bounds") and Image is not None:
        try:
            return np.array(Image.open(png).convert("RGBA")), meta["bounds"], "png"
        except OSError as e:
            report(f"`{png.name}` не открывается: {e}")
    if index in PNG_ONLY_LAYERS:  # RGB и HIDDEN есть только готовыми PNG (overlays/ от main_pipeline М1)
        return None, None, f"Слой {index} за {fmt_date(d)} не найден: запустите main_pipeline.py."
    tif = ds.index_raster(index, d, stats)
    if tif is None:
        return None, None, f"Растр {index} за {fmt_date(d)} не найден."
    if not HAS_RASTERIO:
        return None, None, "Для показа GeoTIFF установите rasterio: pip install rasterio"
    arr, bounds = load_raster(tif)
    if arr is None:
        return None, None, f"Растр {index} за {fmt_date(d)} не удалось прочитать."
    return upscale(colorize(arr, *vrange, INDEX_CMAPS[index])), bounds, "tif"


def zones_overlay(ds: DataSource, d: str, stats):
    tif = ds.zones_raster(d, stats)
    if tif is None or not HAS_RASTERIO:
        return None, None
    arr, bounds = load_raster(tif)
    if arr is None:
        return None, None
    zones = np.nan_to_num(arr, nan=0)
    if not np.isin(zones, list(ZONE_COLORS)).any():
        values = ", ".join(f"{v:g}" for v in np.unique(zones)[:10])
        report(f"`{tif.name}`: нет пикселей с классами 1–3 (значения в файле: {values}).")
        return None, None
    return upscale(colorize_zones(zones.astype(np.uint8))), bounds


def index_block(stats: dict, index: str) -> dict | None:
    """Блок indices.<NDVI|NDRE|MSAVI2|NDMI> из index_stats M2."""
    indices = stats.get("indices") if isinstance(stats.get("indices"), dict) else {}
    return next((indices[k] for k in STATS_INDEX_KEYS[index] if isinstance(indices.get(k), dict)), None)


def season_ranges(stats_by_date: dict, meta: dict) -> dict:
    """Единая шкала на весь сезон: min(p5)…max(p95) по всем датам M2.

    Самодостаточна (только np и константы): тест М1 test_overlays грузит её из app.py отдельно.
    """
    ranges = {k: tuple(v) for k, v in DEFAULT_RANGES.items()}
    ranges.update({k: tuple(v) for k, v in (meta.get("index_ranges") or {}).items()})
    for index, keys in STATS_INDEX_KEYS.items():
        lo, hi = [], []
        for s in stats_by_date.values():
            indices = s.get("indices") if isinstance(s.get("indices"), dict) else {}
            block = next((indices[k] for k in keys if isinstance(indices.get(k), dict)), {})
            p5, p95 = block.get("p5"), block.get("p95")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (p5, p95)):
                lo.append(p5); hi.append(p95)
        if lo and max(hi) > min(lo):
            pad = 0.05 * (max(hi) - min(lo))
            ranges[index] = (round(min(lo) - pad, 3), round(max(hi) + pad, 3))
    return ranges


# ----------------------------------------------------------------------------
# Экономика
# ----------------------------------------------------------------------------
def parse_zone(raw, src: str) -> dict | None:
    """Проверяет запись зоны (zone_class|class, area_ha). Негодную пропускает и пишет в PROBLEMS."""
    if not isinstance(raw, dict):
        report(f"{src}: запись зоны не является объектом — пропущена.")
        return None
    zc, area = as_number(raw.get("zone_class", raw.get("class"))), as_number(raw.get("area_ha"))
    if zc is None or area is None:
        missing = [k for k, v in (("zone_class", zc), ("area_ha", area)) if v is None]
        report(f"{src}: у зоны нет корректного поля {', '.join(missing)} — зона пропущена.")
        return None
    if int(zc) not in ZONE_COLORS:
        report(f"{src}: неизвестный класс зоны {zc:g} (ожидаются 1, 2, 3) — зона не учтена.")
        return None
    return {**raw, "zone_class": int(zc), "area_ha": area}


def zone_file_rate(z: dict) -> float | None:
    return as_number(z.get("rate_kg_ha", z.get("fertilizer_rate")))


def declared_base_rate(metrics: dict) -> float | None:
    """base_rate_kg_ha из файла M3 (или прежнее base_rate); 0 — допустимая «нулевая база»."""
    base = as_number(metrics.get("base_rate_kg_ha", metrics.get("base_rate")))
    return base if base is not None and base >= 0 else None


def file_base_rate(metrics: dict) -> float | None:
    """Базовая норма, с которой M3 считал файл; если её нет — максимальная норма зоны (доза зоны 1)."""
    base = declared_base_rate(metrics)
    if base is not None:
        return base
    rates = [r for z in metrics.get("zone_breakdown") or [] if isinstance(z, dict)
             if (r := zone_file_rate(z)) is not None]
    return max(rates) if rates and max(rates) > 0 else None


def metrics_from_stats(stats: dict | None) -> dict | None:
    """Если M3 ещё не выдал metrics_*.json — собираем разбивку по зонам из index_stats M2."""
    zones_block = (stats or {}).get("zones")
    classes = zones_block.get("classes") if isinstance(zones_block, dict) else None
    if not isinstance(classes, list) or not classes:
        return None
    zones = [z for z in (parse_zone(c, "index_stats (M2)") for c in classes) if z]
    if not zones:
        return None
    return {
        "base_rate_kg_ha": DEFAULT_BASE_RATE, "price_per_kg": DEFAULT_PRICE, "_source": "m2",
        "zone_breakdown": [{
            "zone_class": z["zone_class"], "zone_name": ZONE_NAMES[z["zone_class"]], "area_ha": z["area_ha"],
            "rate_factor": DEFAULT_RATE_FACTORS[z["zone_class"]],
        } for z in zones],
    }


def sanitize_metrics(metrics: dict) -> dict | None:
    """Проверяет metrics (M3 или собранные из M2) до расчёта экономики. None — посчитать нельзя.

    Негодные зоны отбрасываются, недостающая базовая норма выводится из норм зон; всё, что
    отброшено или подставлено, пишется в PROBLEMS. Проверки живут здесь, а не в recompute_economics:
    её и prescription_rates_match тесты М3 грузят из app.py как самодостаточные функции.
    """
    from_m2 = metrics.get("_source") == "m2"
    src = "index_stats (M2)" if from_m2 else "metrics (M3)"
    if metrics.get("quantity_unit", "kg") != "kg":
        report(f"В {src} единица `{metrics.get('quantity_unit')}`, а кабинет считает только килограммы — "
               "экономика не рассчитана.")
        return None
    zones = [z for z in (parse_zone(raw, src) for raw in metrics.get("zone_breakdown") or []) if z]
    if not zones:
        report(f"В {src} нет ни одной годной зоны — экономика не рассчитана.")
        return None
    base = declared_base_rate(metrics)
    if base is None:
        base = file_base_rate({"zone_breakdown": zones})
        if base is not None and not from_m2:
            report(f"В {src} нет base_rate_kg_ha — базовой принята максимальная норма зоны, {fmt_num(base)} кг/га.")
    clean_zones, defaulted = [], []
    for z in zones:
        rate, factor = zone_file_rate(z), as_number(z.get("rate_factor"))
        entry = {k: v for k, v in z.items() if k not in ("rate_kg_ha", "fertilizer_rate", "rate_factor")}
        if rate is not None:
            entry["rate_kg_ha"] = rate
        if factor is not None:
            entry["rate_factor"] = factor
            if not 0 <= factor <= MAX_RATE_FACTOR:
                report(f"{src}: доля нормы зоны {z['zone_class']} = {factor:g} вне диапазона "
                       f"0–{MAX_RATE_FACTOR:g} (проценты вместо долей?).")
        elif rate is None:
            defaulted.append(ZONE_SHORT[z["zone_class"]])
        clean_zones.append(entry)
    if defaulted and not from_m2:
        report(f"В {src} нет норм для зон: {', '.join(defaulted)} — подставлены нормы по умолчанию (100% / 53% / 0%).")
    if sum(z["area_ha"] for z in clean_zones) <= 0:
        report(f"В {src} суммарная площадь зон равна 0 — проверьте поле area_ha и единицы (нужны гектары).")
    clean = {k: v for k, v in metrics.items() if k != "base_rate"}
    clean["zone_breakdown"] = clean_zones
    if base is not None:
        clean["base_rate_kg_ha"] = base
    return clean


def recompute_economics(metrics: dict, base_rate: float, price: float) -> dict:
    """Норма зоны = base_rate * rate_factor; rate_factor берётся из файла
    или выводится как rate_kg_ha / base_rate_kg_ha (с которыми файл был рассчитан).

    Контракт М3 (src/module3/tests): ожидает данные после sanitize_metrics, кроме проверки единиц.
    """
    if metrics.get("quantity_unit", "kg") != "kg":
        raise ValueError("This dashboard supports kilogram prescriptions only")
    file_base = metrics.get("base_rate_kg_ha", metrics.get("base_rate"))
    zones = []
    for z in metrics.get("zone_breakdown", []):
        zc = int(z["zone_class"])
        factor = z.get("rate_factor")
        file_rate = z.get("rate_kg_ha", z.get("fertilizer_rate"))
        if factor is None:
            factor = (file_rate / file_base) if (file_rate is not None and file_base) else DEFAULT_RATE_FACTORS.get(zc, 0)
        # При нулевой базе доля не определена: нормы зон остаются из файла,
        # слайдер меняет только вариант сплошного внесения для сравнения.
        rate = file_rate if file_base == 0 and file_rate is not None else base_rate * factor
        zones.append({**z, "zone_class": zc, "zone_name": z.get("zone_name") or ZONE_NAMES.get(zc, f"Зона {zc}"),
                      "rate_factor": factor, "rate_kg_ha": rate, "total_kg": z["area_ha"] * rate})
    zones.sort(key=lambda z: z["zone_class"])
    total = sum(z["area_ha"] for z in zones)
    stressed = sum(z["area_ha"] for z in zones if z["zone_class"] in (1, 2))
    uniform = total * base_rate
    smart = sum(z["total_kg"] for z in zones)
    saved = uniform - smart  # перерасход показываем отрицательной экономией
    return {
        "total_ha": total, "stressed_ha": stressed,
        "stressed_pct": stressed / total * 100 if total else 0.0,
        "uniform_kg": uniform, "smart_kg": smart, "saved_kg": saved,
        "saved_pct": saved / uniform * 100 if uniform else None,
        "saved_rub": saved * price, "smart_rub": smart * price, "zones": zones,
    }


def prescription_rates_match(metrics: dict, econ: dict | None) -> bool:
    """Не предлагать скачать старый файл, если нормы на экране уже другие."""
    if not econ:
        return False
    original = {int(z["zone_class"]): z for z in metrics.get("zone_breakdown", [])}
    base = metrics.get("base_rate_kg_ha", metrics.get("base_rate"))
    for zone in econ["zones"]:
        source = original.get(zone["zone_class"], {})
        rate = source.get("rate_kg_ha", source.get("fertilizer_rate"))
        if rate is None and base is not None and source.get("rate_factor") is not None:
            rate = base * source["rate_factor"]
        if rate is None or not math.isclose(rate, zone["rate_kg_ha"], rel_tol=1e-10, abs_tol=1e-10):
            return False
    return True


# ----------------------------------------------------------------------------
# Карта
# ----------------------------------------------------------------------------
def geojson_bounds(gj):
    if not gj:
        return None
    lons, lats = [], []

    def walk(c):
        if not isinstance(c, (list, tuple)) or not c:
            return
        if isinstance(c[0], (int, float)) and len(c) >= 2:
            lons.append(c[0]); lats.append(c[1])
        else:
            for x in c:
                walk(x)

    for f in gj.get("features") or []:
        geom = f.get("geometry") if isinstance(f, dict) else None
        if isinstance(geom, dict):
            walk(geom.get("coordinates"))
    return [[min(lats), min(lons)], [max(lats), max(lons)]] if lats else None


def checked_wgs84(gj, name: str):
    """GeoJSON в градусах EPSG:4326 или None: файл в UTM иначе молча уводит карту в другое место."""
    b = geojson_bounds(gj)
    if b and (abs(b[0][0]) > 90 or abs(b[1][0]) > 90 or abs(b[0][1]) > 180 or abs(b[1][1]) > 180):
        report(f"`{name}`: координаты не в градусах EPSG:4326 (похоже на UTM) — слой не показан.")
        return None
    return gj


def normalize_prescription(gj, rates: dict | None, name: str):
    """Приводит свойства M3 к единому виду: zone_class, zone_name, area_ha, rate_kg_ha
    (их ждёт подсказка карты; fertilizer_rate → rate_kg_ha).

    rates — нормы, пересчитанные под слайдер; None — оставить нормы из файла.
    Возвращает None, если в файле нет ни одного полигона.
    """
    gj = copy.deepcopy(gj)
    features, no_geom, unknown = [], 0, 0
    for f in gj.get("features") or []:
        geom = f.get("geometry") if isinstance(f, dict) else None
        if not isinstance(geom, dict) or not geom.get("coordinates"):
            no_geom += 1
            continue
        p = f["properties"] if isinstance(f.get("properties"), dict) else {}
        zc_num = as_number(p.get("zone_class", p.get("class")))
        zc = int(zc_num) if zc_num is not None else 0
        unknown += zc not in ZONE_COLORS
        file_rate = zone_file_rate(p)
        rate = rates.get(zc, file_rate) if rates else file_rate
        area = as_number(p.get("area_ha"))
        f["properties"] = {**p, "zone_class": zc,
                           "zone_name": p.get("zone_name") or ZONE_NAMES.get(zc, "Неизвестная зона"),
                           "area_ha": round(area, 2) if area is not None else "—",
                           "rate_kg_ha": round(rate, 1) if rate is not None else "—"}
        features.append(f)
    if no_geom:
        report(f"`{name}`: {no_geom} объект(ов) без геометрии — пропущены.")
    if unknown:
        report(f"`{name}`: у {unknown} полигон(ов) нет zone_class 1–3 — они серые и без нормы.")
    if not features:
        report(f"`{name}`: нет ни одного полигона — слой предписания не показан.")
        return None
    gj["features"] = features
    return gj


def build_map(field_gj, presc_gj, raster_rgba, raster_bounds, outline_only, raster_opacity=0.85):
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI_TILES, attr=ESRI_ATTR, name="Спутник Esri", max_zoom=19).add_to(m)

    if raster_rgba is not None:
        folium.raster_layers.ImageOverlay(image=raster_rgba, bounds=raster_bounds, opacity=raster_opacity).add_to(m)

    if field_gj:  # только контур (fill=False), чтобы не перехватывать подсказки зон
        folium.GeoJson(field_gj, name="Граница поля",
                       style_function=lambda f: {"color": "#ffffff", "weight": 2.5, "fill": False}).add_to(m)

    if presc_gj:
        filled = not outline_only

        def style(feature):
            color = ZONE_COLORS.get(feature["properties"]["zone_class"], "#95a5a6")
            return {"fillColor": color, "color": color, "weight": 1.5 if filled else 1.2,
                    "fillOpacity": 0.45 if filled else 0.0, "dashArray": None if filled else "4 4"}

        folium.GeoJson(
            presc_gj, name="Карта-предписание", style_function=style,
            highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.65 if filled else 0.1},
            tooltip=folium.GeoJsonTooltip(fields=["zone_name", "area_ha", "rate_kg_ha"],
                                          aliases=["Зона:", "Площадь, га:", "Норма, кг/га:"],
                                          localize=True, sticky=True),
        ).add_to(m)

    bounds = geojson_bounds(field_gj) or raster_bounds or geojson_bounds(presc_gj)
    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))
    return m


def show_map(fmap, key):
    try:
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[], key=key)
    except TypeError:  # старые версии streamlit-folium
        st_folium(fmap, height=560, width=None, returned_objects=[], key=key)


def meta_rgba(value, default) -> tuple:
    """Цвет [r, g, b, a] из meta.json М1 или значение по умолчанию."""
    ok = isinstance(value, (list, tuple)) and len(value) == 4 and all(as_number(v) is not None for v in value)
    return tuple(int(v) for v in value) if ok else default


def css_rgba(c) -> str:
    return f"rgba({c[0]},{c[1]},{c[2]},{c[3] / 255:.2f})"


def legend_swatches(items) -> str:
    spans = "".join(f'<span><span class="sw" style="background:{color}"></span>{html.escape(label)}</span>'
                    for color, label in items)
    return f'<div class="pas-legend">{spans}</div>'


def meta_css(css, rgba, default) -> str:
    """CSS-цвет легенды из meta.json 1.1 (*_css), если он вида rgba(...); иначе из [r, g, b, a] схемы 1.0."""
    if isinstance(css, str) and CSS_RGBA_RE.fullmatch(css.strip()):
        return css.strip()
    return css_rgba(meta_rgba(rgba, default))


def meta_label(value, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def legend_html(index, vrange, meta: dict, from_png: bool) -> str:
    """Легенда слоя. Подписи и цвета PNG М1 — из meta.json (схема 1.1), с запасными для схемы 1.0."""
    if index is None:
        return legend_swatches((ZONE_COLORS[z], ZONE_SHORT[z]) for z in (1, 2, 3))
    nodata = []
    if from_png and meta.get("nodata_mode", "gray") != "transparent":
        nodata = [(meta_css(meta.get("nodata_css"), meta.get("nodata_color"), DEFAULT_NODATA_COLOR),
                   meta_label(meta.get("nodata_label"), DEFAULT_NODATA_LABEL))]
    if index == "RGB":
        rgb = meta.get("rgb") if isinstance(meta.get("rgb"), dict) else {}
        text = meta_label(rgb.get("legend_text"), PNG_ONLY_LAYERS["RGB"])
        return f'<div class="pas-legend"><span>{html.escape(text)}</span></div>'
    if index == "HIDDEN":
        hidden = meta.get("hidden") if isinstance(meta.get("hidden"), dict) else {}
        color = meta_css(hidden.get("css"), hidden.get("1", hidden.get("color")), DEFAULT_HIDDEN_COLOR)
        return legend_swatches([(color, meta_label(hidden.get("label"), PNG_ONLY_LAYERS["HIDDEN"]))] + nodata)
    grad = ", ".join(f"rgb({c[0]},{c[1]},{c[2]}) {s * 100:.0f}%" for s, c in INDEX_CMAPS[index])
    return (f'<div class="pas-bar" style="background: linear-gradient(90deg, {grad});"></div>'
            f'<div class="pas-bar-labels"><span>{vrange[0]:.2f}</span><span>{index}</span>'
            f'<span>{vrange[1]:.2f}</span></div>' + (legend_swatches(nodata) if nodata else ""))


def pipeline_failures(report_doc: dict | None) -> dict[str, str]:
    """Даты, на которых упал последний запуск main_pipeline (pipeline_report.json → errors): {YYYY_MM_DD: текст}.

    Файлы таких дат остались от прошлых запусков и могут быть устаревшими. Ключи «*этап» — ошибки всего этапа.
    """
    errors = (report_doc or {}).get("errors")
    if not isinstance(errors, dict):
        return {}
    out = {}
    for key, err in errors.items():
        d = str(key).replace("-", "_")
        if d.startswith("*") or not DATE_RE.fullmatch(d):
            continue
        err = err if isinstance(err, dict) else {}
        out[d] = f"{err.get('stage', 'этап ?')}: {err.get('error', 'ошибка')}"
    return out


def legend_range(index: str, vranges: dict, meta: dict, from_png: bool):
    """Шкала легенды: для PNG — та, по которой М1 их окрасил (meta.index_ranges), иначе — сезонная по stats."""
    r = (meta.get("index_ranges") or {}).get(index) if from_png else None
    if isinstance(r, (list, tuple)) and len(r) == 2 and all(as_number(v) is not None for v in r):
        return float(r[0]), float(r[1])
    return vranges.get(index)


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def series_value(stats: dict, key: str, index: str) -> float | None:
    """Среднее индекса за дату: mean_<index> → (для MSAVI) mean_msavi2 → indices.<INDEX>.mean."""
    v = as_number(stats.get(key))
    if v is None and index == "MSAVI":
        v = as_number(stats.get("mean_msavi2"))
    if v is None:
        v = as_number((index_block(stats, index) or {}).get("mean"))
    return v


def index_history_chart(hist: pd.DataFrame, selected: datetime):
    fig = go.Figure()
    for key, label, color in INDEX_SERIES:
        if key in hist and hist[key].notna().any():
            fig.add_trace(go.Scatter(x=hist["dt"], y=hist[key], name=label, mode="lines+markers",
                                     line=dict(color=color, width=2.5), marker=dict(size=7),
                                     hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>"))
    a = hist[hist["alert"]] if "alert" in hist else hist.iloc[0:0]
    if not a.empty and "mean_ndre" in hist:
        fig.add_trace(go.Scatter(x=a["dt"], y=a["mean_ndre"], name="Ранний сигнал", mode="markers",
                                 marker=dict(symbol="diamond", size=14, color="rgba(0,0,0,0)",
                                             line=dict(color="#e74c3c", width=2.5)),
                                 hovertemplate="Ранний сигнал стресса<extra></extra>"))
    fig.add_vline(x=selected, line_dash="dash", line_color="#7f8c8d", line_width=1.5)
    fig.update_layout(
        height=310, margin=dict(l=10, r=10, t=36, b=10), hovermode="x unified",
        title=dict(text="Динамика вегетационных индексов (среднее по полю)", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
        xaxis=dict(title="Дата снимка", tickformat="%d.%m"), yaxis=dict(title="Значение индекса"),
    )
    return fig


def zones_donut(econ):
    zones = [z for z in econ["zones"] if z["area_ha"] > 0]
    fig = go.Figure(go.Pie(
        labels=[ZONE_SHORT.get(z["zone_class"], z["zone_name"]) for z in zones],
        values=[z["area_ha"] for z in zones],
        marker=dict(colors=[ZONE_COLORS.get(z["zone_class"]) for z in zones]),
        hole=0.58, sort=False, direction="clockwise", texttemplate="%{percent:.0%}", textposition="outside",
        hovertemplate="%{label}<br>%{value:.2f} га (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=290, margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text="Распределение площади по зонам", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0),
        annotations=[dict(text=f"<b>{fmt_num(econ['total_ha'], 1)}</b><br>га", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))],
    )
    return fig


def show_plotly(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def early_stress_info(stats: dict) -> dict:
    es = stats.get("early_stress")
    return es if isinstance(es, dict) else {}


def early_stress_flag(stats: dict) -> bool | None:
    """Флаг раннего стресса M2; None — флага нет или он непонятного вида (не путать с «стресса нет»)."""
    raw = stats.get("early_stress_alert", early_stress_info(stats).get("alert"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    return None


def stress_banner(stats: dict | None):
    if not stats:
        return
    alert = early_stress_flag(stats)
    if alert is None:
        st.warning("В index_stats нет флага `early_stress_alert` (true/false) — статус раннего стресса неизвестен.")
        return
    if not alert:
        st.markdown('<div class="pas-ok">✅ Ранних признаков скрытого стресса на этой дате не обнаружено.</div>',
                    unsafe_allow_html=True)
        return
    details = []
    area, pct = as_number(stats.get("early_stress_area_ha")), as_number(stats.get("early_stress_percent"))
    if area:
        details.append(f"Площадь скрытого стресса: <b>{fmt_num(area, 1)} га</b> ({fmt_num(pct or 0, 1)}% поля).")
    ndre, ndvi = as_number(stats.get("ndre_change_pct")), as_number(stats.get("ndvi_change_pct"))
    if ndre is not None and ndvi is not None:
        details.append(f"За неделю NDRE {ndre:+.1f}% против NDVI {ndvi:+.1f}%.")
    reasons_list = early_stress_info(stats).get("reasons")
    reasons = "".join(f"<li>{html.escape(str(r))}</li>"
                      for r in (reasons_list if isinstance(reasons_list, list) else []))
    st.markdown(
        '<div class="pas-alert">⚠️ <b>Внимание: зафиксировано падение индекса NDRE в Red Edge спектре!</b> '
        "Скрытый хлорофилловый стресс за 10–14 дней до видимого пожелтения. "
        + " ".join(details) + (f"<ul>{reasons}</ul>" if reasons else "") + "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Приложение
# ----------------------------------------------------------------------------
sources = find_sources()
if not sources:
    st.error("Нет данных: выполните `python generate_mock_data.py` или задайте PAS_DATA_DIR.")
    st.stop()

with st.sidebar:
    st.markdown("## 🛰️ Precision Ag")
    st.caption("Космический мониторинг стресса посевов")
    names = list(sources)
    default_src = next((i for i, n in enumerate(names) if DataSource(sources[n]).dates()), 0)
    src_name = st.radio("Источник данных", names, index=default_src) if len(names) > 1 else names[0]

ds = DataSource(sources[src_name])
dates = ds.dates()
if not dates:
    st.warning(f"В `{ds.root}` пока нет ни `stats/index_stats_*.json`, ни `metrics_*.json`. "
               "Запустите пайплайн M2 или выберите «Мок-данные».")
    st.stop()

stats_by_date = {d: jload(ds.stats_path(d)) for d in dates}
stats_by_date = {d: s for d, s in stats_by_date.items() if s}
meta = jload(ds.root / "overlays" / "meta.json") or {}
field_gj = checked_wgs84(ds.field_boundary(), "field_boundary.geojson")
field_name = meta.get("field_name")
if not field_name and field_gj and field_gj.get("features"):
    field_name = (field_gj["features"][0].get("properties") or {}).get("name")
field_name = field_name or "Тестовое поле"
vranges = season_ranges(stats_by_date, meta)
failed_dates = pipeline_failures(jload(ds.root / "pipeline_report.json"))

with st.sidebar:
    st.selectbox("Поле", [field_name])
    if len(dates) > 1:
        sel_date = st.select_slider("Дата снимка Sentinel-2", options=dates, value=dates[-1], format_func=fmt_date)
    else:
        sel_date = dates[0]
        st.markdown(f"**Дата снимка:** {fmt_date(sel_date)}")
    layer_label = st.radio("Слой карты", list(LAYERS), index=0)
    index_name = LAYERS[layer_label]

    stats = stats_by_date.get(sel_date)
    metrics_path = ds.vra_file(f"metrics_{sel_date}.json")
    m3_metrics = jload(metrics_path)
    if m3_metrics is not None and not isinstance(m3_metrics.get("zone_breakdown"), list):
        report(f"`{metrics_path.name}`: нет списка zone_breakdown — экономика посчитана по данным Модуля 2.")
        m3_metrics = None
    metrics = m3_metrics or metrics_from_stats(stats) or {}
    clean_metrics = sanitize_metrics(metrics) if metrics.get("zone_breakdown") else None
    file_base = file_base_rate(clean_metrics or metrics)
    file_base = DEFAULT_BASE_RATE if file_base is None else file_base
    file_price = as_number(metrics.get("price_per_kg", metrics.get("price_per_unit")))
    file_price = DEFAULT_PRICE if file_price is None or file_price < 0 else file_price

    st.divider()
    st.markdown("**Параметры экономики**")
    base_rate = st.slider("Базовая норма удобрений, кг/га", 0.0, max(BASE_RATE_MAX, file_base),
                          float(file_base), step=5.0)
    price = st.slider("Цена удобрения, ₽/кг", 0.0, max(PRICE_MAX, file_price), float(file_price), step=1.0)
    econ = recompute_economics(clean_metrics, base_rate, price) if clean_metrics else None
    downloads_match = prescription_rates_match(clean_metrics, econ) if m3_metrics and clean_metrics else False
    stale = failed_dates.get(sel_date)
    downloads_match = downloads_match and not stale
    coverage = metrics.get("coverage") if isinstance(metrics.get("coverage"), dict) else {}
    unclassified = as_number(coverage.get("unclassified_area_ha"))
    if coverage.get("is_complete", True) is False and unclassified is not None:
        st.warning(f"Неполное покрытие: без класса {unclassified:.2f} га. "
                   "В файлах этим участкам норма не назначена; для полного задания требуется решение М2.")
    filled_area = as_number((stats or {}).get("filled_area_ha")) or 0
    if filled_area > 0:
        st.warning(f"Восстановлено М2: {filled_area:.2f} га "
                   f"({as_number((stats or {}).get('filled_percent')) or 0:.2f}% поля). "
                   "Облака и тени получили класс ближайшей зоны; это не прямое наблюдение. "
                   "Эти участки включены в предписание и расчёт экономики.")
    if m3_metrics and file_base == 0:
        st.caption("При нулевой исходной базе нормы зон фиксированы; ползунок меняет только сплошной вариант сравнения.")

    st.divider()
    st.markdown("**Файлы для бортового компьютера**")
    if m3_metrics:
        st.caption(f"Файлы рассчитаны для нормы сравнения {file_base:g} кг/га "
                   f"и цены {file_price:g} ₽/кг. Нормы зон заданы в предписании.")
    gj_path = ds.vra_file(f"prescription_{sel_date}.geojson")
    zip_path = ds.vra_file(f"shapefile_{sel_date}.zip")
    if gj_path:
        st.download_button("⬇️ Карта-предписание (.geojson)", gj_path.read_bytes(),
                           file_name=gj_path.name, mime="application/geo+json", disabled=not downloads_match)
    if zip_path:
        st.download_button("⬇️ Shapefile для трактора (.zip)", zip_path.read_bytes(),
                           file_name=zip_path.name, mime="application/zip", disabled=not downloads_match)
    if not gj_path and not zip_path:
        st.caption("Файлы-задания появятся, когда Модуль 3 сформирует предписание за эту дату.")
    elif not m3_metrics:
        st.caption("Скачивание отключено: нет metrics_*.json Модуля 3 за эту дату, нормы файлов не проверить.")
    elif stale:
        st.caption("Скачивание отключено: файлы за эту дату остались от прошлого запуска и могут быть устаревшими.")
    elif not downloads_match:
        st.warning("Нормы на экране отличаются от сохранённых файлов. Верните исходную норму "
                   "или создайте новое предписание в М3; скачивание текущих файлов отключено.")
    st.caption(f"Данные: `{ds.root}`")

# --- Шапка
st.markdown(f"### {field_name}")
mock_note = " · ⚠️ демонстрационные мок-данные" if ds.root == APP_DIR / "mock_data" else ""
st.caption(f"Снимок Sentinel-2 L2A от {fmt_date(sel_date)} · слой: {layer_label}{mock_note}")
if stale:
    st.error(f"Данные за {fmt_date(sel_date)} могут быть устаревшими: при последнем запуске main_pipeline "
             f"обработка этой даты завершилась ошибкой ({stale}). Карта, статистика и предписание показаны "
             "по файлам прошлого запуска.")
stress_banner(stats)
problems_box = st.container()  # заполняется в конце, когда известны все проблемы с данными

# --- KPI
k1, k2, k3, k4 = st.columns(4)
if econ:
    k1.metric("Площадь с классификацией", f"{fmt_num(econ['total_ha'], 1)} га")
    field_ha = as_number(coverage.get("reference_field_area_ha"))
    if field_ha is not None and unclassified is not None:
        st.caption(f"Полный контур: {fmt_num(field_ha, 2)} га; без классификации: {fmt_num(unclassified, 2)} га. "
                   "Расход и экономия рассчитаны только по классифицированной площади.")
    k2.metric("Зоны стресса", f"{fmt_num(econ['stressed_ha'], 1)} га",
              f"{fmt_num(econ['stressed_pct'], 1)}% поля", delta_color="inverse")
    k3.metric("Сэкономлено удобрений", f"{fmt_num(econ['saved_kg'])} кг",
              f"{fmt_num(econ['saved_pct'], 1)}% от сплошного внесения" if econ["saved_pct"] is not None
              else "Процент не определён: нулевая база")
    k4.metric("Финансовая экономия", f"{fmt_num(econ['saved_rub'], 2)} ₽",
              f"затраты {fmt_num(econ['smart_rub'], 2)} ₽ вместо {fmt_num(econ['uniform_kg'] * price, 2)} ₽",
              delta_color="off")
    if metrics.get("_source") == "m2":
        st.caption("Экономика рассчитана по площадям зон из статистики Модуля 2 "
                   "(нормы по умолчанию 100% / 53% / 0%). Данных Модуля 3 за эту дату пока нет.")
else:
    for k, label in zip((k1, k2, k3, k4), ("Общая площадь поля", "Зоны стресса",
                                           "Сэкономлено удобрений", "Финансовая экономия")):
        k.metric(label, "—")

# --- Карта и графики
left, right = st.columns([3, 2], gap="medium")
with left:
    presc = checked_wgs84(jload(gj_path), gj_path.name) if gj_path else None
    if presc:
        # нормы по умолчанию из M2 не должны подменять реальные нормы M3 в подсказках
        rates = {z["zone_class"]: z["rate_kg_ha"] for z in econ["zones"]} if econ and m3_metrics else None
        presc = normalize_prescription(presc, rates, gj_path.name)
    rgba, rbounds, from_png = None, None, False
    if index_name:
        rgba, rbounds, info = index_overlay(ds, index_name, sel_date, stats, meta, vranges.get(index_name))
        if rgba is None:
            st.info(info)
        from_png = info == "png"
        if info == "tif" and meta.get("bounds"):
            st.caption("PNG Модуля 1 за эту дату нет: слой построен по растру Модуля 2 (indices/*.tif).")
    elif not presc:
        rgba, rbounds = zones_overlay(ds, sel_date, stats)
        if rgba is not None:
            st.caption("Показана растровая маска зон Модуля 2. Полигоны с нормами появятся после Модуля 3.")
        else:
            st.info("Карта-предписание за эту дату ещё не сформирована.")
    # у PNG М1 альфа уже в картинке; tif М2 приглушаем, чтобы просвечивала подложка
    fmap = build_map(field_gj, presc, rgba, rbounds, outline_only=index_name is not None,
                     raster_opacity=1.0 if from_png else 0.85)
    show_map(fmap, key=f"map_{src_name}_{sel_date}_{index_name}")
    st.markdown(legend_html(index_name, legend_range(index_name, vranges, meta, from_png), meta, from_png),
                unsafe_allow_html=True)

with right:
    rows = []
    for d, s in stats_by_date.items():
        rows.append({"dt": parse_date(d), "alert": bool(early_stress_flag(s)),
                     **{key: series_value(s, key, index) for key, index, _ in INDEX_SERIES}})
    hist = pd.DataFrame(rows, dtype=object).sort_values("dt") if rows else pd.DataFrame()
    if not hist.empty:
        hist[[key for key, _, _ in INDEX_SERIES]] = hist[[key for key, _, _ in INDEX_SERIES]].astype(float)
        missing = [index for key, index, _ in INDEX_SERIES if hist[key].isna().all()]
        if missing:
            report(f"В index_stats нет средних значений для {', '.join(missing)} "
                   f"(ожидаются поля {', '.join('mean_' + i.lower() for i in missing)}) — линии не показаны.")
        show_plotly(index_history_chart(hist, parse_date(sel_date)))
        if len(hist) == 1:
            st.caption("Пока доступна одна дата — динамика появится, когда Модуль 1 выгрузит серию снимков.")
    else:
        st.info("Нет файлов stats/index_stats_*.json для графика динамики.")
    if econ and econ["zones"]:
        show_plotly(zones_donut(econ))

if econ:
    with st.expander("Нормы внесения по зонам"):
        st.dataframe(pd.DataFrame([{
            "Зона": z["zone_name"], "Площадь, га": round(z["area_ha"], 2), "Норма, кг/га": round(z["rate_kg_ha"], 1),
            "Удобрения, кг": round(z["total_kg"], 1), "Стоимость, ₽": round(z["total_kg"] * price),
        } for z in econ["zones"]]), hide_index=True)

if PROBLEMS:
    with problems_box.expander(f"⚠️ Проблемы с данными ({len(PROBLEMS)})", expanded=True):
        st.markdown("\n".join(f"- {p}" for p in PROBLEMS))
