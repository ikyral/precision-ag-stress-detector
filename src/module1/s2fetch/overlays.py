"""Окрашенные PNG-оверлеи для кабинета М4: data/overlays/{слой}_YYYY_MM_DD.png + meta.json.

Слои: NDVI, NDRE, MSAVI, NDMI — из растров индексов М2 (indices/*.tif, сами индексы не пересчитываются);
RGB — true color из сцен М1 (своя растяжка по перцентилям поля, не как у превью КТ2); HIDDEN — маска скрытого стресса М2 (hidden_stress/*.tif, М4 её пока не показывает).

Геометрия: Leaflet растягивает ImageOverlay линейно в EPSG:3857, поэтому PNG строится на сетке EPSG:3857
(прямоугольник вокруг сетки сцен, ~oversample выходных пикселей на пиксель 10 м), а meta.bounds — lat/lon углов
этого прямоугольника. Каждый выходной пиксель берёт значение пикселя источника, в который попал его центр
(nearest), поэтому пиксели 10 м остаются чёткими. Экстент и размер PNG одни на все даты и слои.

Цвета: палитры, альфа и шкала — как в src/module4/app.py (colorize, season_ranges), чтобы легенда М4 совпадала
с картинкой. Шкала зависит от всех дат, поэтому при изменении любого входа перерисовываются все PNG.
Вне контура поля — прозрачно; внутри контура без данных (облако/тень, NaN) — серый «нет данных».

    python -m s2fetch.overlays --data ../../data [--nodata-mode gray|transparent] [--oversample 8] [--no-hidden]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine

from .grid import Grid, compare_grids, grid_of_file

SCHEMA_VERSION = "1.0"
RENDER_CRS = "EPSG:3857"
OVERSAMPLE = 8
INDEX_LAYERS = ("NDVI", "NDRE", "MSAVI", "NDMI")

# --- как в src/module4/app.py (сверяется тестом через ast) ---
STATS_INDEX_KEYS = {"NDVI": ("NDVI",), "NDRE": ("NDRE",), "MSAVI": ("MSAVI2", "MSAVI"), "NDMI": ("NDMI",)}
DEFAULT_RANGES = {"NDVI": (0.2, 0.9), "NDRE": (0.1, 0.7), "MSAVI": (0.1, 0.8), "NDMI": (-0.1, 0.5)}
CMAP_VEG = [(0.0, (165, 0, 38)), (0.25, (244, 109, 67)), (0.5, (254, 224, 139)),
            (0.75, (102, 189, 99)), (1.0, (0, 104, 55))]
CMAP_WATER = [(0.0, (140, 81, 10)), (0.25, (216, 179, 101)), (0.5, (246, 232, 195)),
              (0.75, (90, 180, 172)), (1.0, (1, 102, 94))]
INDEX_CMAPS = {"NDVI": CMAP_VEG, "NDRE": CMAP_VEG, "MSAVI": CMAP_VEG, "NDMI": CMAP_WATER}
ALPHA = 235                                  # альфа данных внутри поля (colorize в app.py)

NODATA_MODES = ("gray", "transparent")
NODATA_GRAY = (128, 128, 128, 150)           # внутри поля нет наблюдения
NODATA_MEANING = ("внутри контура поля нет наблюдения: облако/тень (NaN в индексе или сцене) или 255 в "
                  "hidden_stress; не прозрачно, чтобы сквозь дыру не была видна подложка другой даты")
HIDDEN_COLOR = (231, 76, 60, ALPHA)          # 1 = скрытый стресс; 0 = прозрачно

# RGB: снимок, а не тепловая карта — непрозрачный, растяжка по полю, одна на сезон и на три канала
# (по каналам цвета уходят в пурпур: синий канал в поле узкий). Гамма 1.3 вытягивает тени.
RGB_BANDS = ("B04", "B03", "B02")           # R, G, B; индексы в сцене — по s2fetch.BANDS
RGB_PERCENTILES = (2, 98)
RGB_GAMMA = 1.3
RGB_ALPHA = 255
LAYER_RE = re.compile(r"^(NDVI|NDRE|MSAVI|NDMI|RGB|HIDDEN)_\d{4}_\d{2}_\d{2}\.png$")
STAMP_RE = re.compile(r"(\d{4}_\d{2}_\d{2})")


# ----------------------------------------------------------------------------
# Сетка EPSG:3857
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderGrid:
    transform: Affine          # EPSG:3857, север вверху
    width: int
    height: int

    @property
    def bounds_3857(self) -> tuple[float, float, float, float]:
        t = self.transform
        return t.c, t.f + self.height * t.e, t.c + self.width * t.a, t.f

    def bounds_latlon(self) -> list[list[float]]:
        """[[south, west], [north, east]] — формат meta.bounds для М4 (folium ImageOverlay)."""
        from pyproj import Transformer

        to_ll = Transformer.from_crs(RENDER_CRS, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = self.bounds_3857
        west, south = to_ll.transform(minx, miny)
        east, north = to_ll.transform(maxx, maxy)
        return [[south, west], [north, east]]


def _outline(src: Grid, n: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Точки по периметру сетки источника (с промежуточными, на случай кривизны рёбер)."""
    t = src.transform
    s = np.linspace(0, 1, n)
    cols = np.concatenate([s * src.width, np.full(n, src.width), s[::-1] * src.width, np.zeros(n)])
    rows = np.concatenate([np.zeros(n), s * src.height, np.full(n, src.height), s[::-1] * src.height])
    return t.c + cols * t.a, t.f + rows * t.e


def mercator_stretch(src: Grid) -> float:
    """Сколько метров EPSG:3857 в одном метре сетки источника (в центре сетки), ≈1.475 для поля."""
    from pyproj import Transformer

    to_m = Transformer.from_crs(src.crs, RENDER_CRS, always_xy=True)
    t = src.transform
    cx, cy = t.c + src.width * t.a / 2, t.f + src.height * t.e / 2
    (x0, x1), (y0, y1) = to_m.transform([cx, cx + t.a], [cy, cy])
    return math.hypot(x1 - x0, y1 - y0) / abs(t.a)


def render_grid(src: Grid, oversample: int = OVERSAMPLE) -> RenderGrid:
    """Прямоугольник EPSG:3857, покрывающий всю сетку источника; пиксель = шаг источника × растяжение / oversample."""
    from pyproj import Transformer

    if src.transform.b or src.transform.d:
        raise ValueError("сетка источника должна быть без поворота (север вверху)")
    xs, ys = Transformer.from_crs(src.crs, RENDER_CRS, always_xy=True).transform(*_outline(src))
    minx, miny, maxx, maxy = float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))
    px = abs(src.transform.a) * mercator_stretch(src) / oversample
    w, h = math.ceil((maxx - minx) / px), math.ceil((maxy - miny) / px)
    return RenderGrid(Affine(px, 0, minx, 0, -px, maxy), w, h)


def source_lookup(src: Grid, rg: RenderGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Для каждого выходного пикселя — (row, col) пикселя источника, в который попал его центр, и флаг «внутри»."""
    from pyproj import Transformer

    cols, rows = np.meshgrid(np.arange(rg.width) + 0.5, np.arange(rg.height) + 0.5)
    t3 = rg.transform
    x, y = t3.c + cols * t3.a, t3.f + rows * t3.e
    ux, uy = Transformer.from_crs(RENDER_CRS, src.crs, always_xy=True).transform(x, y)
    t = src.transform
    c = np.floor((np.asarray(ux) - t.c) / t.a).astype(np.int64)
    r = np.floor((np.asarray(uy) - t.f) / t.e).astype(np.int64)
    ok = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
    return np.where(ok, r, 0), np.where(ok, c, 0), ok


def _take(arr: np.ndarray, lookup, fill) -> np.ndarray:
    r, c, ok = lookup
    return np.where(ok, arr[r, c], fill)


# ----------------------------------------------------------------------------
# Цвета (как в М4)
# ----------------------------------------------------------------------------
def colorize(arr: np.ndarray, vmin: float, vmax: float, cmap) -> np.ndarray:
    """Копия app.colorize М4: значение -> RGBA, альфа ALPHA у конечных значений, 0 у NaN."""
    valid = np.isfinite(arr)
    t = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    stops = np.array([s for s, _ in cmap])
    cols = np.array([c for _, c in cmap], dtype=float)
    rgb = np.stack([np.interp(t, stops, cols[:, k]) for k in range(3)], axis=-1)
    alpha = np.where(valid, ALPHA, 0)[..., None]
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)


def season_ranges(stats_docs) -> dict[str, tuple[float, float]]:
    """Шкала сезона по формуле app.season_ranges М4: min(p5)…max(p95) по всем датам ± 5 %, округление 0.001."""
    ranges = {k: tuple(v) for k, v in DEFAULT_RANGES.items()}
    for index, keys in STATS_INDEX_KEYS.items():
        lo, hi = [], []
        for s in stats_docs:
            block = next(((s.get("indices") or {}).get(k) for k in keys if (s.get("indices") or {}).get(k)), None)
            if block and block.get("p5") is not None and block.get("p95") is not None:
                lo.append(block["p5"]); hi.append(block["p95"])
        if lo and max(hi) > min(lo):
            pad = 0.05 * (max(hi) - min(lo))
            ranges[index] = (round(min(lo) - pad, 3), round(max(hi) + pad, 3))
    return ranges


def _finish(rgba: np.ndarray, field: np.ndarray, nodata: np.ndarray, nodata_rgba) -> np.ndarray:
    """Вне поля — прозрачно, внутри поля без данных — nodata_rgba (None — прозрачно)."""
    out = rgba.copy()
    out[~field] = 0
    out[field & nodata] = nodata_rgba if nodata_rgba is not None else 0
    return out


def render_index(values: np.ndarray, field: np.ndarray, vrange, cmap, nodata_rgba) -> np.ndarray:
    return _finish(colorize(values, *vrange, cmap), field, ~np.isfinite(values), nodata_rgba)


def rgb_stretch(scenes) -> tuple[float, float] | None:
    """Границы растяжки (lo, hi): перцентили RGB_PERCENTILES отражения B04, B03, B02 вместе, по всем пикселям
    поля (не 0, конечные) всех переданных сцен [C, H, W] в порядке BANDS. None — данных нет."""
    from . import BANDS

    idx = [BANDS.index(b) for b in RGB_BANDS]
    vals = []
    for sc in scenes:
        inside = ~np.all(sc == 0, axis=0)
        for i in idx:
            v = sc[i][inside]
            vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return None
    lo, hi = (float(x) for x in np.percentile(v, RGB_PERCENTILES))
    return (lo, hi) if hi > lo else None


def stretch_rgb(r, g, b, lo: float, hi: float, gamma: float = RGB_GAMMA) -> np.ndarray:
    """Отражение -> uint8 [H, W, 3]: 255 * clip((refl - lo) / (hi - lo), 0, 1) ** (1 / gamma)."""
    st = (np.stack([r, g, b], -1) - lo) / (hi - lo)
    return np.round(np.clip(np.nan_to_num(st), 0, 1) ** (1 / gamma) * 255).astype(np.uint8)


def render_rgb(b04: np.ndarray, b03: np.ndarray, b02: np.ndarray, field: np.ndarray, stretch: tuple[float, float],
               nodata_rgba) -> np.ndarray:
    nan = ~(np.isfinite(b04) & np.isfinite(b03) & np.isfinite(b02))
    rgba = np.concatenate([stretch_rgb(b04, b03, b02, *stretch), np.full(field.shape + (1,), RGB_ALPHA, np.uint8)],
                          -1)
    return _finish(rgba, field, nan, nodata_rgba)


def render_hidden(hidden: np.ndarray, field: np.ndarray, nodata_rgba) -> np.ndarray:
    rgba = np.zeros(hidden.shape + (4,), np.uint8)
    rgba[hidden == 1] = HIDDEN_COLOR
    return _finish(rgba, field, hidden == 255, nodata_rgba)


# ----------------------------------------------------------------------------
# Входы
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _stats_by_stamp(data: Path) -> dict[str, tuple[Path, dict]]:
    out = {}
    for p in sorted((data / "stats").glob("index_stats_*.json")):
        m = STAMP_RE.search(p.stem)
        if m:
            out[m.group(1)] = (p, json.loads(p.read_text(encoding="utf-8")))
    return out


def date_sources(data: Path, stamp: str, stats: dict) -> dict:
    """Пути входов даты (как их ищет М4: сначала files.indices из stats, затем indices/{слой}_дата.tif)."""
    rel = (stats.get("files") or {}).get("indices") or {}
    idx = {}
    for name in INDEX_LAYERS:
        cands = ([data / rel[name]] if rel.get(name) else []) + [data / "indices" / f"{name}_{stamp}.tif"]
        idx[name] = next((p for p in cands if p.exists()), None)
    zones = data / ((stats.get("files") or {}).get("zones") or f"zones/zones_{stamp}.tif")
    hidden = data / ((stats.get("files") or {}).get("hidden_stress") or f"hidden_stress/hidden_stress_{stamp}.tif")
    scene = data / "scenes" / f"scene_{stamp}.tif"
    return {"indices": idx, "zones": zones if zones.exists() else None,
            "hidden_stress": hidden if hidden.exists() else None, "scene": scene if scene.exists() else None}


def _read(path: Path, grid: Grid | None) -> tuple[np.ndarray, Grid]:
    import rasterio

    g = grid_of_file(path)
    if grid is not None and compare_grids(g, grid):
        raise ValueError(f"{path.name}: сетка отличается от сетки сцен: {'; '.join(compare_grids(g, grid))}")
    with rasterio.open(path) as src:
        return src.read(), g


def _save_png(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgba, "RGBA").save(path, optimize=False)


def contact_sheet(out: Path, layers: dict[str, dict[str, str]], order: list[str], tile: int = 200) -> None:
    """Контактный лист «даты × слои» поверх белого фона: подписи, по одному ряду на дату."""
    from PIL import Image, ImageDraw

    dates = sorted(layers)
    if not dates:
        return
    head, left, gap = 18, 78, 4
    sheet = Image.new("RGBA", (left + len(order) * (tile + gap), head + len(dates) * (tile + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, name in enumerate(order):
        draw.text((left + j * (tile + gap) + 4, 3), name, fill=(0, 0, 0))
    for i, stamp in enumerate(dates):
        y = head + i * (tile + gap)
        draw.text((4, y + tile // 2 - 6), stamp, fill=(0, 0, 0))
        for j, name in enumerate(order):
            fname = layers[stamp].get(name)
            x = left + j * (tile + gap)
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(220, 220, 220))
            if fname:
                im = Image.open(out.parent / fname).convert("RGBA").resize((tile, tile), Image.NEAREST)
                sheet.alpha_composite(im, (x, y))
    sheet.convert("RGB").save(out)


# ----------------------------------------------------------------------------
# Сборка
# ----------------------------------------------------------------------------
def build_overlays(data_dir: str | Path, *, dates: list[str] | None = None, oversample: int = OVERSAMPLE,
                   nodata_mode: str = "gray", hidden: bool = True, force: bool = False) -> dict:
    """Пишет <data>/overlays/*.png, meta.json, preview_all.png.

    dates — даты (YYYY_MM_DD или YYYY-MM-DD), для которых рисовать слои; None — все даты с stats М2.
    Шкала индексов всегда по stats всех дат (как в М4). Если входы и параметры не менялись с прошлого
    запуска (отпечаток в meta.json), файлы не трогаются: status = 'unchanged'.
    """
    if nodata_mode not in NODATA_MODES:
        raise ValueError(f"nodata_mode: {nodata_mode} (нужно {' | '.join(NODATA_MODES)})")
    data = Path(data_dir)
    out = data / "overlays"
    nodata_rgba = NODATA_GRAY if nodata_mode == "gray" else None
    stats_all = _stats_by_stamp(data)
    want = sorted(stats_all) if dates is None else sorted({d.replace("-", "_") for d in dates} & set(stats_all))
    errors: dict[str, str] = {}
    warnings: list[str] = []
    if dates is not None:
        missing = sorted({d.replace("-", "_") for d in dates} - set(stats_all))
        warnings += [f"{d}: нет stats/index_stats_{d}.json, слои не построены" for d in missing]

    plan: dict[str, dict] = {}
    for stamp in want:
        src = date_sources(data, stamp, stats_all[stamp][1])
        lost = [n for n, p in src["indices"].items() if p is None]
        if lost:
            errors[stamp] = f"нет растров индексов М2: {', '.join(lost)}"
        elif src["zones"] is None and src["scene"] is None:
            errors[stamp] = "нет ни zones, ни сцены — неизвестен контур поля"
        else:
            plan[stamp] = src

    inputs = {_rel(p, data): p for p, _ in stats_all.values()}
    # растяжка RGB — по сценам всех дат сезона (как шкала индексов — по stats всех дат)
    season_scenes = [p for p in (data / "scenes" / f"scene_{s}.tif" for s in stats_all) if p.exists()]
    for p in season_scenes:
        inputs[_rel(p, data)] = p
    for src in plan.values():
        for p in [*src["indices"].values(), src["zones"], src["scene"], src["hidden_stress"] if hidden else None]:
            if p is not None:
                inputs[_rel(p, data)] = p
    params = {"schema_version": SCHEMA_VERSION, "oversample": oversample, "nodata_mode": nodata_mode,
              "hidden": hidden, "dates": sorted(plan), "rgb": [RGB_BANDS, RGB_PERCENTILES, RGB_GAMMA, RGB_ALPHA], "alpha": ALPHA}
    fp = hashlib.sha256(json.dumps({"params": params, "inputs": {k: _sha256(p) for k, p in sorted(inputs.items())}},
                                   sort_keys=True).encode()).hexdigest()
    meta_path = out / "meta.json"
    old = _load_json(meta_path)
    if (not force and old.get("inputs_sha256") == fp
            and all((out / f).exists() for lay in (old.get("layers") or {}).values() for f in lay.values())):
        return {"status": "unchanged", "out_dir": str(out), "dates": old.get("dates", []),
                "png_count": sum(len(v) for v in old["layers"].values()), "size": old["render_grid"]["size"],
                "bounds": old["bounds"], "errors": errors, "warnings": warnings, "meta": str(meta_path)}

    ranges = season_ranges([doc for _, doc in stats_all.values()])
    stretch = rgb_stretch(_read(p, None)[0] for p in season_scenes)
    grid: Grid | None = None
    rg = lookup = None
    layers: dict[str, dict[str, str]] = {}
    sources: dict[str, dict] = {}
    pngs: dict[str, np.ndarray] = {}
    for stamp, src in plan.items():
        try:
            first = src["zones"] or src["scene"]
            if grid is None:
                grid = grid_of_file(first)
                rg = render_grid(grid, oversample)
                lookup = source_lookup(grid, rg)
            if src["zones"] is not None:
                field_src = _read(src["zones"], grid)[0][0] > 0
            else:
                field_src = ~np.all(_read(src["scene"], grid)[0] == 0, axis=0)
            field = _take(field_src, lookup, False)
            mine: dict[str, np.ndarray] = {}
            for name in INDEX_LAYERS:
                vals = _take(_read(src["indices"][name], grid)[0][0].astype(np.float32), lookup, np.nan)
                mine[name] = render_index(vals, field, ranges[name], INDEX_CMAPS[name], nodata_rgba)
            if src["scene"] is not None and stretch is not None:
                from . import BANDS

                sc, _ = _read(src["scene"], grid)
                if sc.shape[0] != len(BANDS):
                    raise ValueError(f"{src['scene'].name}: каналов {sc.shape[0]}, нужно {len(BANDS)}")
                scene_field = _take(~np.all(sc == 0, axis=0), lookup, False)
                r, g, b = (_take(sc[BANDS.index(n)].astype(np.float32), lookup, np.nan) for n in RGB_BANDS)
                mine["RGB"] = render_rgb(r, g, b, scene_field, stretch, nodata_rgba)
            else:
                warnings.append(f"{stamp}: нет scenes/scene_{stamp}.tif или данных для растяжки, слой RGB пропущен")
            if hidden and src["hidden_stress"] is not None:
                hs = _take(_read(src["hidden_stress"], grid)[0][0], lookup, 255)
                mine["HIDDEN"] = render_hidden(hs, field, nodata_rgba)
        except Exception as exc:  # noqa: BLE001 — одна дата не обрывает остальные
            errors[stamp] = f"{type(exc).__name__}: {exc}"
            continue
        for name, rgba in mine.items():
            pngs[f"{name}_{stamp}.png"] = rgba
        layers[stamp] = {name: f"{name}_{stamp}.png" for name in mine}
        sources[stamp] = {"scene": _rel(src["scene"], data) if src["scene"] else None,
                          "zones": _rel(src["zones"], data) if src["zones"] else None,
                          "indices": {n: _rel(p, data) for n, p in src["indices"].items()},
                          "hidden_stress": _rel(src["hidden_stress"], data) if hidden and src["hidden_stress"]
                          else None}

    out.mkdir(parents=True, exist_ok=True)
    for name, rgba in pngs.items():
        _save_png(out / name, rgba)
    for p in out.glob("*.png"):  # слои дат, которых больше нет
        if LAYER_RE.match(p.name) and p.name not in pngs:
            p.unlink()
    order = [*INDEX_LAYERS, "RGB"] + (["HIDDEN"] if hidden else [])
    contact_sheet(out / "preview_all.png", layers, order)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "src/module1 s2fetch.overlays",
        "bounds": rg.bounds_latlon() if rg else None,
        "render_crs": RENDER_CRS,
        "render_grid": {"size": [rg.width, rg.height], "pixel_size_m": rg.transform.a,
                        "bounds_3857": list(rg.bounds_3857), "oversample": oversample,
                        "resampling": "nearest (центр выходного пикселя -> пиксель источника)"} if rg else None,
        "source_grid": {"crs": grid.crs, "transform": list(grid.transform)[:6],
                        "size": [grid.width, grid.height]} if grid else None,
        "index_ranges": {k: list(v) for k, v in ranges.items()},
        "index_ranges_rule": "как season_ranges в src/module4/app.py: по всем stats/index_stats_*.json "
                             "lo=min(p5), hi=max(p95), pad=0.05*(hi-lo), (round(lo-pad,3), round(hi+pad,3)); "
                             "без stats — DEFAULT_RANGES",
        "colormaps": {k: [[s, list(c)] for s, c in v] for k, v in INDEX_CMAPS.items()},
        "alpha": ALPHA,
        "rgb": {"bands": list(RGB_BANDS), "stretch": "percentile", "percentiles": list(RGB_PERCENTILES),
                "pooled": "одна шкала на три канала: перцентили отражения B04, B03, B02 вместе по всем пикселям "
                          "поля всех дат сезона (сцены дат со stats)",
                "season_dates": sorted(p.stem[len("scene_"):] for p in season_scenes), "min_reflectance": stretch[0], "max_reflectance": stretch[1],
                "gamma": RGB_GAMMA, "alpha": RGB_ALPHA,
                "formula": "255 * clip((refl - min_reflectance) / (max_reflectance - min_reflectance), 0, 1) "
                           "** (1 / gamma)"} if stretch else None,
        "hidden": {"color": list(HIDDEN_COLOR), "values": "1 — цвет, 0 — прозрачно, 255 в поле — nodata"}
                  if hidden else None,
        "outside_field": "прозрачно (alpha 0)",
        "nodata_mode": nodata_mode,
        "nodata_color": list(nodata_rgba) if nodata_rgba else [0, 0, 0, 0],
        "nodata_meaning": NODATA_MEANING,
        "dates": sorted(layers),
        "layers": layers,
        "sources": sources,
        "inputs_sha256": fp,
    }
    tmp = meta_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(meta_path)
    return {"status": "written", "out_dir": str(out), "dates": meta["dates"], "png_count": len(pngs),
            "size": meta["render_grid"]["size"] if rg else None, "bounds": meta["bounds"],
            "errors": errors, "warnings": warnings, "meta": str(meta_path)}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m s2fetch.overlays",
                                 description="PNG-оверлеи слоёв для кабинета М4 -> <data>/overlays")
    ap.add_argument("--data", type=Path, default=Path("../../data"), help="папка данных (по умолчанию ../../data)")
    ap.add_argument("--oversample", type=int, default=OVERSAMPLE,
                    help=f"выходных пикселей на пиксель источника (по умолчанию {OVERSAMPLE})")
    ap.add_argument("--nodata-mode", choices=NODATA_MODES, default="gray",
                    help="облако/тень внутри поля: серый (по умолчанию) или прозрачно")
    ap.add_argument("--no-hidden", action="store_true", help="не строить HIDDEN_*.png")
    ap.add_argument("--force", action="store_true", help="перерисовать, даже если входы не менялись")
    a = ap.parse_args(argv)
    res = build_overlays(a.data, oversample=a.oversample, nodata_mode=a.nodata_mode, hidden=not a.no_hidden,
                         force=a.force)
    print(f"{res['status']}: {res['png_count']} PNG, дат {len(res['dates'])}, размер {res['size']}, "
          f"bounds {res['bounds']} -> {res['out_dir']}")
    for w in res["warnings"]:
        print("предупреждение:", w)
    for d, e in res["errors"].items():
        print(f"ОШИБКА {d}: {e}")
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
