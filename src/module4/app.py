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
  stats/index_stats_YYYY_MM_DD.json                          статистика M2 (schema 1.1)
  indices/{NDVI|NDRE|MSAVI|NDMI}_YYYY_MM_DD.tif              растры индексов M2 (нужен rasterio)
  zones/zones_YYYY_MM_DD.tif                                 маска зон M2 (нужен rasterio)
  vra/ | prescriptions/ | . :
      prescription_YYYY_MM_DD.geojson  (zone_class, area_ha, fertilizer_rate | rate_kg_ha)  M3
      shapefile_YYYY_MM_DD.zip, metrics_YYYY_MM_DD.json                                     M3
  overlays/{INDEX}_YYYY_MM_DD.png + overlays/meta.json      готовые PNG (мок-данные)

Пока файлов M3 нет, экономика считается по площадям зон из index_stats (M2).
"""

from __future__ import annotations

import copy
import json
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
DEFAULT_RATE_FACTORS = {1: 1.0, 2: round(80 / 150, 4), 3: 0.0}  # доли от базовой нормы
DEFAULT_BASE_RATE, DEFAULT_PRICE = 150, 45

LAYERS = {
    "Карта предписаний VRA": None,
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


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float = 0.0):
    p = Path(path)
    return _read_json(p) if p.exists() else None


def jload(path: Path | None):
    if path is None or not path.exists():
        return None
    return load_json(str(path), path.stat().st_mtime)


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
        found = set()
        for p in (self.root / "stats").glob("index_stats_*.json"):
            m = DATE_RE.search(p.stem)
            if m:
                found.add(m.group(1))
        for sub in VRA_SUBDIRS:
            for p in (self.root / sub).glob("metrics_*.json"):
                m = DATE_RE.search(p.stem)
                if m:
                    found.add(m.group(1))
        return sorted(found)

    def field_boundary(self):
        for p in (self.root / "field_boundary.geojson", self.root / "samples" / "field_boundary.geojson"):
            if p.exists():
                return jload(p)
        return None

    def index_raster(self, index: str, d: str, stats: dict | None) -> Path | None:
        rel = ((stats or {}).get("files") or {}).get("indices", {}).get(index)
        for p in ([self.root / rel] if rel else []) + [self.root / "indices" / f"{index}_{d}.tif"]:
            if p.exists():
                return p
        return None

    def zones_raster(self, d: str, stats: dict | None) -> Path | None:
        rel = ((stats or {}).get("files") or {}).get("zones")
        for p in ([self.root / rel] if rel else []) + [self.root / "zones" / f"zones_{d}.tif"]:
            if p.exists():
                return p
        return None

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


def upscale(rgba: np.ndarray, k: int = 4) -> np.ndarray:
    """Увеличение «блоками», чтобы пиксели 10 м не размывались браузером."""
    return np.repeat(np.repeat(rgba, k, axis=0), k, axis=1)


@st.cache_data(show_spinner=False)
def read_geotiff_wgs84(path: str, mtime: float):
    """Читает 1-й канал GeoTIFF, перепроецирует в EPSG:4326.

    Возвращает (array float32 с NaN вместо nodata, bounds [[S, W], [N, E]]).
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        dst_crs = "EPSG:4326"
        transform, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(source=arr, destination=dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    west, north = transform * (0, 0)
    east, south = transform * (w, h)
    return dst, [[south, west], [north, east]]


def index_overlay(ds: DataSource, index: str, d: str, stats, meta, vrange):
    """(rgba, bounds, источник) для слоя индекса или (None, None, причина)."""
    png = ds.overlay_png(index, d)
    if png is not None and meta.get("bounds") and Image is not None:
        return np.array(Image.open(png).convert("RGBA")), meta["bounds"], "png"
    tif = ds.index_raster(index, d, stats)
    if tif is None:
        return None, None, f"Растр {index} за {fmt_date(d)} не найден."
    if not HAS_RASTERIO:
        return None, None, "Для показа GeoTIFF установите rasterio: pip install rasterio"
    arr, bounds = read_geotiff_wgs84(str(tif), tif.stat().st_mtime)
    return upscale(colorize(arr, *vrange, INDEX_CMAPS[index])), bounds, "tif"


def zones_overlay(ds: DataSource, d: str, stats):
    tif = ds.zones_raster(d, stats)
    if tif is None or not HAS_RASTERIO:
        return None, None
    arr, bounds = read_geotiff_wgs84(str(tif), tif.stat().st_mtime)
    return upscale(colorize_zones(np.nan_to_num(arr, nan=0).astype(np.uint8))), bounds


def season_ranges(stats_by_date: dict, meta: dict) -> dict:
    """Единая шкала на весь сезон: min(p5)…max(p95) по всем датам M2."""
    ranges = {k: tuple(v) for k, v in DEFAULT_RANGES.items()}
    ranges.update({k: tuple(v) for k, v in (meta.get("index_ranges") or {}).items()})
    for index, keys in STATS_INDEX_KEYS.items():
        lo, hi = [], []
        for s in stats_by_date.values():
            block = next(((s.get("indices") or {}).get(k) for k in keys if (s.get("indices") or {}).get(k)), None)
            if block and block.get("p5") is not None and block.get("p95") is not None:
                lo.append(block["p5"]); hi.append(block["p95"])
        if lo and max(hi) > min(lo):
            pad = 0.05 * (max(hi) - min(lo))
            ranges[index] = (round(min(lo) - pad, 3), round(max(hi) + pad, 3))
    return ranges


# ----------------------------------------------------------------------------
# Экономика
# ----------------------------------------------------------------------------
def metrics_from_stats(stats: dict | None) -> dict | None:
    """Если M3 ещё не выдал metrics_*.json — собираем разбивку по зонам из index_stats M2."""
    classes = ((stats or {}).get("zones") or {}).get("classes") or []
    if not classes:
        return None
    return {
        "base_rate_kg_ha": DEFAULT_BASE_RATE, "price_per_kg": DEFAULT_PRICE, "_source": "m2",
        "zone_breakdown": [{
            "zone_class": int(c["class"]), "zone_name": ZONE_NAMES.get(int(c["class"]), c.get("label", "")),
            "area_ha": float(c.get("area_ha", 0.0)),
            "rate_factor": DEFAULT_RATE_FACTORS.get(int(c["class"]), 0.0),
        } for c in classes],
    }


def recompute_economics(metrics: dict, base_rate: float, price: float) -> dict:
    """Норма зоны = base_rate * rate_factor; rate_factor берётся из файла
    или выводится как rate_kg_ha / base_rate_kg_ha (с которыми файл был рассчитан)."""
    file_base = metrics.get("base_rate_kg_ha") or 0.0
    zones = []
    for z in metrics.get("zone_breakdown", []):
        zc = int(z["zone_class"])
        factor = z.get("rate_factor")
        if factor is None:
            file_rate = z.get("rate_kg_ha", z.get("fertilizer_rate"))
            factor = (file_rate / file_base) if (file_rate is not None and file_base) else DEFAULT_RATE_FACTORS.get(zc, 0)
        rate = base_rate * factor
        zones.append({**z, "zone_class": zc, "zone_name": z.get("zone_name") or ZONE_NAMES.get(zc, f"Зона {zc}"),
                      "rate_factor": factor, "rate_kg_ha": round(rate, 1), "total_kg": z["area_ha"] * rate})
    zones.sort(key=lambda z: z["zone_class"])
    total = sum(z["area_ha"] for z in zones)
    stressed = sum(z["area_ha"] for z in zones if z["zone_class"] in (1, 2))
    uniform = total * base_rate
    smart = sum(z["total_kg"] for z in zones)
    saved = max(0.0, uniform - smart)
    return {
        "total_ha": total, "stressed_ha": stressed,
        "stressed_pct": stressed / total * 100 if total else 0.0,
        "uniform_kg": uniform, "smart_kg": smart, "saved_kg": saved,
        "saved_pct": saved / uniform * 100 if uniform else 0.0,
        "saved_rub": saved * price, "smart_rub": smart * price, "zones": zones,
    }


# ----------------------------------------------------------------------------
# Карта
# ----------------------------------------------------------------------------
def geojson_bounds(gj):
    if not gj:
        return None
    lons, lats = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            lons.append(c[0]); lats.append(c[1])
        else:
            for x in c:
                walk(x)

    for f in gj.get("features", []):
        walk(f["geometry"]["coordinates"])
    return [[min(lats), min(lons)], [max(lats), max(lons)]] if lats else None


def normalize_prescription(gj, econ):
    """Приводит свойства M3 к единому виду (fertilizer_rate | rate_kg_ha → rate_kg_ha)
    и подставляет нормы, пересчитанные под слайдер базовой нормы."""
    gj = copy.deepcopy(gj)
    rates = {z["zone_class"]: z["rate_kg_ha"] for z in econ["zones"]}
    for f in gj.get("features", []):
        p = f.setdefault("properties", {})
        zc = int(p.get("zone_class", 0))
        p["zone_class"] = zc
        p["zone_name"] = p.get("zone_name") or ZONE_NAMES.get(zc, f"Зона {zc}")
        p["rate_kg_ha"] = rates.get(zc, p.get("rate_kg_ha", p.get("fertilizer_rate")))
        p["area_ha"] = round(float(p.get("area_ha") or 0.0), 2)
    return gj


def build_map(field_gj, presc_gj, raster_rgba, raster_bounds, outline_only):
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI_TILES, attr=ESRI_ATTR, name="Спутник Esri", max_zoom=19).add_to(m)

    if raster_rgba is not None:
        folium.raster_layers.ImageOverlay(image=raster_rgba, bounds=raster_bounds, opacity=0.85).add_to(m)

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


def legend_html(index, vrange):
    if index is None:
        items = "".join(f'<span><span class="sw" style="background:{ZONE_COLORS[z]}"></span>{ZONE_SHORT[z]}</span>'
                        for z in (1, 2, 3))
        return f'<div class="pas-legend">{items}</div>'
    grad = ", ".join(f"rgb({c[0]},{c[1]},{c[2]}) {s * 100:.0f}%" for s, c in INDEX_CMAPS[index])
    return (f'<div class="pas-bar" style="background: linear-gradient(90deg, {grad});"></div>'
            f'<div class="pas-bar-labels"><span>{vrange[0]:.2f}</span><span>{index}</span>'
            f'<span>{vrange[1]:.2f}</span></div>')


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
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


def stress_banner(stats: dict | None):
    if not stats:
        return
    es = stats.get("early_stress") or {}
    alert = bool(stats.get("early_stress_alert", es.get("alert", False)))
    if not alert:
        st.markdown('<div class="pas-ok">✅ Ранних признаков скрытого стресса на этой дате не обнаружено.</div>',
                    unsafe_allow_html=True)
        return
    details = []
    if stats.get("early_stress_area_ha"):
        details.append(f"Площадь скрытого стресса: <b>{fmt_num(stats['early_stress_area_ha'], 1)} га</b>"
                       f" ({fmt_num(stats.get('early_stress_percent', 0), 1)}% поля).")
    if "ndre_change_pct" in stats and "ndvi_change_pct" in stats:
        details.append(f"За неделю NDRE {stats['ndre_change_pct']:+.1f}% против NDVI {stats['ndvi_change_pct']:+.1f}%.")
    reasons = "".join(f"<li>{r}</li>" for r in (es.get("reasons") or []))
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
field_gj = ds.field_boundary()
field_name = meta.get("field_name")
if not field_name and field_gj and field_gj.get("features"):
    field_name = (field_gj["features"][0].get("properties") or {}).get("name")
field_name = field_name or "Тестовое поле"
vranges = season_ranges(stats_by_date, meta)

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
    metrics = jload(metrics_path) or metrics_from_stats(stats) or {}

    st.divider()
    st.markdown("**Параметры экономики**")
    base_rate = st.slider("Базовая норма удобрений, кг/га", 50, 300,
                          int(metrics.get("base_rate_kg_ha") or DEFAULT_BASE_RATE), step=5)
    price = st.slider("Цена удобрения, ₽/кг", 10, 150, int(metrics.get("price_per_kg") or DEFAULT_PRICE), step=1)

    st.divider()
    st.markdown("**Файлы для бортового компьютера**")
    gj_path = ds.vra_file(f"prescription_{sel_date}.geojson")
    zip_path = ds.vra_file(f"shapefile_{sel_date}.zip")
    if gj_path:
        st.download_button("⬇️ Карта-предписание (.geojson)", gj_path.read_bytes(),
                           file_name=gj_path.name, mime="application/geo+json")
    if zip_path:
        st.download_button("⬇️ Shapefile для трактора (.zip)", zip_path.read_bytes(),
                           file_name=zip_path.name, mime="application/zip")
    if not gj_path and not zip_path:
        st.caption("Файлы-задания появятся, когда Модуль 3 сформирует предписание за эту дату.")
    elif metrics.get("base_rate_kg_ha") and base_rate != metrics["base_rate_kg_ha"]:
        st.caption(f"Файлы рассчитаны для нормы {fmt_num(metrics['base_rate_kg_ha'])} кг/га.")
    st.caption(f"Данные: `{ds.root}`")

# --- Шапка
st.markdown(f"### {field_name}")
st.caption(f"Снимок Sentinel-2 L2A от {fmt_date(sel_date)} · слой: {layer_label}")
stress_banner(stats)

# --- KPI
econ = recompute_economics(metrics, base_rate, price) if metrics.get("zone_breakdown") else None
k1, k2, k3, k4 = st.columns(4)
if econ:
    k1.metric("Общая площадь поля", f"{fmt_num(econ['total_ha'], 1)} га")
    k2.metric("Зоны стресса", f"{fmt_num(econ['stressed_ha'], 1)} га",
              f"{fmt_num(econ['stressed_pct'], 1)}% поля", delta_color="inverse")
    k3.metric("Сэкономлено удобрений", f"{fmt_num(econ['saved_kg'])} кг",
              f"{fmt_num(econ['saved_pct'], 1)}% от сплошного внесения")
    k4.metric("Финансовая экономия", f"{fmt_num(econ['saved_rub'])} ₽",
              f"затраты {fmt_num(econ['smart_rub'])} ₽ вместо {fmt_num(econ['uniform_kg'] * price)} ₽",
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
    presc = jload(gj_path)
    presc = normalize_prescription(presc, econ) if (presc and econ) else presc
    rgba, rbounds = None, None
    if index_name:
        rgba, rbounds, info = index_overlay(ds, index_name, sel_date, stats, meta, vranges[index_name])
        if rgba is None:
            st.info(info)
    elif not presc:
        rgba, rbounds = zones_overlay(ds, sel_date, stats)
        if rgba is not None:
            st.caption("Показана растровая маска зон Модуля 2. Полигоны с нормами появятся после Модуля 3.")
        else:
            st.info("Карта-предписание за эту дату ещё не сформирована.")
    fmap = build_map(field_gj, presc, rgba, rbounds, outline_only=index_name is not None)
    show_map(fmap, key=f"map_{src_name}_{sel_date}_{index_name}")
    st.markdown(legend_html(index_name, vranges.get(index_name) if index_name else None), unsafe_allow_html=True)

with right:
    rows = []
    for d, s in stats_by_date.items():
        es = s.get("early_stress") or {}
        rows.append({"dt": parse_date(d), "alert": bool(s.get("early_stress_alert", es.get("alert", False))),
                     **{k: s.get(k) for k, _, _ in INDEX_SERIES}})
    hist = pd.DataFrame(rows).sort_values("dt") if rows else pd.DataFrame()
    if not hist.empty:
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
            "Зона": z["zone_name"], "Площадь, га": round(z["area_ha"], 2), "Норма, кг/га": z["rate_kg_ha"],
            "Удобрения, кг": round(z["total_kg"], 1), "Стоимость, ₽": round(z["total_kg"] * price),
        } for z in econ["zones"]]), hide_index=True)
