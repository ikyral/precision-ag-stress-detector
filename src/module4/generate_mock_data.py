"""
Precision Ag Stress Detector — Модуль 4.
Генератор мок-данных для веб-кабинета агронома (КТ 1).

Создаёт папку mock_data/ с файлами в тех же форматах, что будут выдавать
Модули 1–3, на 6 недель сезона для поля в Краснодарском крае:

mock_data/
  field_boundary.geojson                  граница поля (EPSG:4326)
  vra/prescription_YYYY_MM_DD.geojson     карта-предписание (Модуль 3)
  vra/shapefile_YYYY_MM_DD.zip            Shapefile для трактора (Модуль 3)
  vra/metrics_YYYY_MM_DD.json             экономика (Модуль 3)
  stats/index_stats_YYYY_MM_DD.json       статистика в формате Модуля 2 (schema 1.1)
  overlays/{NDVI|NDRE|MSAVI|NDMI}_YYYY_MM_DD.png   окрашенные растры индексов
  overlays/meta.json                      границы растров и шкалы

Сценарий: в середине поля развивается очаг стресса. Канал Red Edge (NDRE)
реагирует сразу, а NDVI — с запаздыванием в 2 недели. Позже у восточного
края появляется второй очаг.

Запуск:  python generate_mock_data.py
Зависимости: numpy, pillow, pyshp
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import shapefile  # pyshp
except ImportError:  # pragma: no cover
    shapefile = None

OUT_DIR = Path(__file__).resolve().parent / "mock_data"

# ----------------------------------------------------------------------------
# Поле и сценарий
# ----------------------------------------------------------------------------
FIELD_NAME = "Поле №7 · Усть-Лабинский р-н, Краснодарский край"
LAT0, LON0 = 45.2050, 39.6600  # юго-западный угол локальной системы координат
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))

# Контур поля в локальных метрах (против часовой стрелки), примерно 120 га
FIELD_XY = [(0, 0), (1380, 40), (1420, 860), (60, 900), (-20, 450)]

START_DATE = date(2026, 5, 4)
N_WEEKS = 6
DATES = [START_DATE + timedelta(weeks=i) for i in range(N_WEEKS)]

# Очаги стресса: центр (м), полуоси (м), поворот (°), интенсивность по неделям
HOTSPOTS = [
    {"c": (480, 470), "a": 330, "b": 230, "rot": 15,
     "intensity": [0.0, 0.40, 0.70, 0.90, 1.00, 1.00]},
    {"c": (1110, 280), "a": 200, "b": 150, "rot": -30,
     "intensity": [0.0, 0.0, 0.0, 0.40, 0.70, 0.90]},
]
NDVI_LAG_WEEKS = 2          # NDVI «видит» стресс на 2 недели позже NDRE
SEVERE_THR, MODERATE_THR = 0.5, 0.2
SPREAD = 1.5                # S = I * exp(-SPREAD * d^2)

# Экономика (как в стартовом коде Модуля 3: 150 / 80 / 0 кг/га)
BASE_RATE_KG_HA = 150.0
PRICE_PER_KG = 45.0
RATE_FACTORS = {1: 1.0, 2: round(80.0 / 150.0, 4), 3: 0.0}
ZONE_NAMES = {
    1: "Острый стресс (макс. доза)",
    2: "Умеренный стресс (средняя доза)",
    3: "Норма (без обработки)",
}

PIXEL_M = 10.0  # разрешение Sentinel-2

INDEX_RANGES = {"NDVI": (0.2, 0.9), "NDRE": (0.05, 0.55), "MSAVI": (0.1, 0.85), "NDMI": (-0.1, 0.5)}
CMAP_VEG = [(0.0, (165, 0, 38)), (0.25, (244, 109, 67)), (0.5, (254, 224, 139)),
            (0.75, (102, 189, 99)), (1.0, (0, 104, 55))]
CMAP_WATER = [(0.0, (140, 81, 10)), (0.25, (216, 179, 101)), (0.5, (246, 232, 195)),
              (0.75, (90, 180, 172)), (1.0, (1, 102, 94))]
INDEX_CMAPS = {"NDVI": CMAP_VEG, "NDRE": CMAP_VEG, "MSAVI": CMAP_VEG, "NDMI": CMAP_WATER}


# ----------------------------------------------------------------------------
# Геометрия
# ----------------------------------------------------------------------------
def xy_to_lonlat(x: float, y: float) -> tuple[float, float]:
    return (round(LON0 + x / M_PER_DEG_LON, 7), round(LAT0 + y / M_PER_DEG_LAT, 7))


def signed_area(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        s += x1 * y2 - x2 * y1
    return s / 2.0


def orient(ring, ccw: bool):
    """Возвращает кольцо с нужным направлением обхода."""
    return ring if (signed_area(ring) > 0) == ccw else ring[::-1]


def ellipse_ring(hs, scale: float, n: int = 72):
    cx, cy = hs["c"]
    t = math.radians(hs["rot"])
    pts = []
    for k in range(n):
        p = 2 * math.pi * k / n
        ex, ey = hs["a"] * scale * math.cos(p), hs["b"] * scale * math.sin(p)
        pts.append((cx + ex * math.cos(t) - ey * math.sin(t), cy + ex * math.sin(t) + ey * math.cos(t)))
    return pts


def level_scale(intensity: float, thr: float) -> float:
    """Масштаб эллипса, на котором I*exp(-SPREAD*d^2) == thr (0 — зоны нет)."""
    if intensity <= thr:
        return 0.0
    return math.sqrt(math.log(intensity / thr) / SPREAD)


def ring_to_lonlat(ring):
    closed = ring + ring[:1]
    return [list(xy_to_lonlat(x, y)) for x, y in closed]


def zone_geometries(week: int):
    """Строит полигоны зон (в локальных метрах) для недели week.

    Возвращает {zone_class: [polygon, ...]}, где polygon = [outer, hole1, ...].
    """
    severe, moderate, holes_in_field = [], [], []
    for hs in HOTSPOTS:
        i = hs["intensity"][week]
        s_mod, s_sev = level_scale(i, MODERATE_THR), level_scale(i, SEVERE_THR)
        if s_mod == 0:
            continue
        outer = ellipse_ring(hs, s_mod)
        holes_in_field.append(outer)
        if s_sev > 0:
            inner = ellipse_ring(hs, s_sev)
            severe.append([inner])
            moderate.append([outer, inner])
        else:
            moderate.append([outer])
    zones = {3: [[list(FIELD_XY)] + holes_in_field]}
    if moderate:
        zones[2] = moderate
    if severe:
        zones[1] = severe
    return zones


def polygon_area_m2(poly):
    return abs(signed_area(poly[0])) - sum(abs(signed_area(h)) for h in poly[1:])


# ----------------------------------------------------------------------------
# Растры индексов
# ----------------------------------------------------------------------------
def build_grid():
    xs = [p[0] for p in FIELD_XY]
    ys = [p[1] for p in FIELD_XY]
    x0, x1 = min(xs) - 20, max(xs) + 20
    y0, y1 = min(ys) - 20, max(ys) + 20
    nx = int(math.ceil((x1 - x0) / PIXEL_M))
    ny = int(math.ceil((y1 - y0) / PIXEL_M))
    x1, y1 = x0 + nx * PIXEL_M, y0 + ny * PIXEL_M
    cx = x0 + (np.arange(nx) + 0.5) * PIXEL_M
    cy = y1 - (np.arange(ny) + 0.5) * PIXEL_M  # строки сверху вниз (север → юг)
    X, Y = np.meshgrid(cx, cy)
    return X, Y, (x0, y0, x1, y1)


def point_in_polygon(X, Y, ring):
    inside = np.zeros(X.shape, dtype=bool)
    n = len(ring)
    for k in range(n):
        x1, y1 = ring[k]
        x2, y2 = ring[(k + 1) % n]
        cond = (y1 > Y) != (y2 > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= cond & (X < xint)
    return inside


def stress_field(X, Y, week: int):
    S = np.zeros(X.shape)
    if week < 0:
        return S
    for hs in HOTSPOTS:
        i = hs["intensity"][week]
        if i <= 0:
            continue
        t = math.radians(hs["rot"])
        dx, dy = X - hs["c"][0], Y - hs["c"][1]
        u = (dx * math.cos(t) + dy * math.sin(t)) / hs["a"]
        v = (-dx * math.sin(t) + dy * math.cos(t)) / hs["b"]
        S += i * np.exp(-SPREAD * (u * u + v * v))
    return np.clip(S, 0, 1)


def compute_indices(X, Y, week: int, rng):
    season = [0.60, 0.65, 0.69, 0.72, 0.74, 0.75][week]
    # плавная неоднородность почвы + пиксельный шум
    soil = (0.03 * np.sin(X / 260.0) * np.cos(Y / 210.0)
            + 0.02 * (X / 1400.0) - 0.015 * (Y / 900.0))
    noise = rng.normal(0, 0.012, X.shape)

    s_ndre = stress_field(X, Y, week)
    s_ndvi = stress_field(X, Y, week - NDVI_LAG_WEEKS)

    ndvi = season + soil + noise - 0.30 * s_ndvi
    ndre = 0.62 * season - 0.03 + 0.8 * soil + 0.8 * noise - 0.30 * s_ndre
    msavi = 1.05 * ndvi - 0.08 + rng.normal(0, 0.008, X.shape)
    ndmi = 0.55 * season - 0.07 + 0.6 * soil + 0.6 * noise - 0.10 * s_ndre - 0.08 * s_ndvi
    return {"NDVI": ndvi, "NDRE": ndre, "MSAVI": msavi, "NDMI": ndmi}


def colorize(arr, mask, vmin, vmax, cmap, upscale: int = 3):
    t = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    stops = np.array([s[0] for s in cmap])
    cols = np.array([s[1] for s in cmap], dtype=float)
    rgb = np.stack([np.interp(t, stops, cols[:, c]) for c in range(3)], axis=-1)
    alpha = np.where(mask, 235, 0)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    return img


# ----------------------------------------------------------------------------
# Запись файлов
# ----------------------------------------------------------------------------
def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_prescription_geojson(path: Path, zones_xy, areas_ha):
    features = []
    for zc in sorted(zones_xy):
        polys = []
        for poly in zones_xy[zc]:
            outer = orient(poly[0], ccw=True)
            holes = [orient(h, ccw=False) for h in poly[1:]]
            polys.append([ring_to_lonlat(outer)] + [ring_to_lonlat(h) for h in holes])
        rate = round(BASE_RATE_KG_HA * RATE_FACTORS[zc], 1)
        features.append({
            "type": "Feature",
            "properties": {
                "zone_class": zc,
                "zone_name": ZONE_NAMES[zc],
                "area_ha": round(areas_ha[zc], 3),
                "fertilizer_rate": rate,  # имя поля по data/README.md (кг/га)
                "total_kg": round(areas_ha[zc] * rate, 2),
            },
            "geometry": {"type": "MultiPolygon", "coordinates": polys},
        })
    write_json(path, {"type": "FeatureCollection", "features": features})


WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


def write_shapefile_zip(zip_path: Path, zones_xy, areas_ha, date_str: str):
    if shapefile is None:
        raise SystemExit("Не установлен pyshp: pip install pyshp")
    tmp = zip_path.parent / f"_shp_{date_str}"
    tmp.mkdir(exist_ok=True)
    base = tmp / f"prescription_{date_str}"
    w = shapefile.Writer(str(base), shapeType=shapefile.POLYGON, encoding="utf-8")
    w.field("zone_class", "N", size=2)
    w.field("zone_name", "C", size=64)
    w.field("area_ha", "N", size=12, decimal=3)
    w.field("rate_kg_ha", "N", size=10, decimal=1)
    w.field("total_kg", "N", size=14, decimal=2)
    for zc in sorted(zones_xy):
        parts = []
        for poly in zones_xy[zc]:
            # в Shapefile внешнее кольцо — по часовой, дырки — против
            parts.append(ring_to_lonlat(orient(poly[0], ccw=False)))
            parts += [ring_to_lonlat(orient(h, ccw=True)) for h in poly[1:]]
        rate = round(BASE_RATE_KG_HA * RATE_FACTORS[zc], 1)
        w.poly(parts)
        w.record(zc, ZONE_NAMES[zc], round(areas_ha[zc], 3), rate, round(areas_ha[zc] * rate, 2))
    w.close()
    (tmp / f"prescription_{date_str}.prj").write_text(WGS84_PRJ, encoding="ascii")
    (tmp / f"prescription_{date_str}.cpg").write_text("UTF-8", encoding="ascii")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(tmp.iterdir()):
            zf.write(f, arcname=f.name)
    shutil.rmtree(tmp)


def build_metrics(date_str, areas_ha):
    total = sum(areas_ha.values())
    stressed = areas_ha.get(1, 0.0) + areas_ha.get(2, 0.0)
    breakdown = []
    smart = 0.0
    for zc in (1, 2, 3):
        a = areas_ha.get(zc, 0.0)
        rate = BASE_RATE_KG_HA * RATE_FACTORS[zc]
        smart += a * rate
        breakdown.append({
            "zone_class": zc, "zone_name": ZONE_NAMES[zc], "area_ha": round(a, 3),
            "rate_kg_ha": round(rate, 1), "rate_factor": RATE_FACTORS[zc],
            "total_kg": round(a * rate, 2),
        })
    uniform = total * BASE_RATE_KG_HA
    saved = max(0.0, uniform - smart)
    return {
        "date": date_str,
        "base_rate_kg_ha": BASE_RATE_KG_HA,
        "price_per_kg": PRICE_PER_KG,
        "total_field_area_ha": round(total, 2),
        "stressed_area_ha": round(stressed, 2),
        "stressed_area_percent": round(stressed / total * 100, 1) if total else 0.0,
        "uniform_application_kg": round(uniform, 1),
        "smart_application_kg": round(smart, 1),
        "saved_fertilizer_kg": round(saved, 1),
        "saved_percent": round(saved / uniform * 100, 1) if uniform else 0.0,
        "saved_money_rub": round(saved * PRICE_PER_KG, 0),
        "zone_breakdown": breakdown,
    }


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    vra, stats, ovl = OUT_DIR / "vra", OUT_DIR / "stats", OUT_DIR / "overlays"
    for d in (vra, stats, ovl):
        d.mkdir(parents=True)

    # Граница поля
    write_json(OUT_DIR / "field_boundary.geojson", {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": FIELD_NAME},
            "geometry": {"type": "Polygon", "coordinates": [ring_to_lonlat(orient(FIELD_XY, ccw=True))]},
        }],
    })

    X, Y, (x0, y0, x1, y1) = build_grid()
    field_mask = point_in_polygon(X, Y, FIELD_XY)
    sw, ne = xy_to_lonlat(x0, y0), xy_to_lonlat(x1, y1)
    write_json(ovl / "meta.json", {
        "field_name": FIELD_NAME,
        "crs": "EPSG:4326",
        "bounds": [[sw[1], sw[0]], [ne[1], ne[0]]],  # [[south, west], [north, east]]
        "pixel_size_m": PIXEL_M,
        "index_ranges": {k: list(v) for k, v in INDEX_RANGES.items()},
        "colormaps": {k: [[s, list(c)] for s, c in v] for k, v in INDEX_CMAPS.items()},
    })

    # проверка: все очаги целиком внутри поля
    for hs in HOTSPOTS:
        ring = ellipse_ring(hs, level_scale(max(hs["intensity"]), MODERATE_THR))
        xs, ys = np.array([p[0] for p in ring]), np.array([p[1] for p in ring])
        assert point_in_polygon(xs, ys, FIELD_XY).all(), "очаг выходит за границу поля"

    rng = np.random.default_rng(42)
    prev_means = None
    for week, d in enumerate(DATES):
        ds = d.strftime("%Y_%m_%d")

        # --- Модуль 3: предписание, shapefile, экономика
        zones_xy = zone_geometries(week)
        areas_ha = {zc: sum(polygon_area_m2(p) for p in polys) / 10_000 for zc, polys in zones_xy.items()}
        write_prescription_geojson(vra / f"prescription_{ds}.geojson", zones_xy, areas_ha)
        write_shapefile_zip(vra / f"shapefile_{ds}.zip", zones_xy, areas_ha, ds)
        write_json(vra / f"metrics_{ds}.json", build_metrics(ds, areas_ha))

        # --- Модуль 2: индексы, статистика и раннее предупреждение
        idx = compute_indices(X, Y, week, rng)
        means = {k: float(np.mean(v[field_mask])) for k, v in idx.items()}
        if prev_means:
            ndvi_chg = (means["NDVI"] - prev_means["NDVI"]) / prev_means["NDVI"] * 100
            ndre_chg = (means["NDRE"] - prev_means["NDRE"]) / prev_means["NDRE"] * 100
        else:
            ndvi_chg = ndre_chg = 0.0
        alert = (ndre_chg - ndvi_chg) < -2.0  # NDRE отстаёт от NDVI больше чем на 2 п.п.
        # «скрытый» стресс: NDRE уже просел, а NDVI ещё нет
        hidden = field_mask & (stress_field(X, Y, week) > MODERATE_THR) & \
            (stress_field(X, Y, week - NDVI_LAG_WEEKS) <= MODERATE_THR)
        hidden_ha = float(hidden.sum()) * PIXEL_M ** 2 / 10_000
        field_ha = float(field_mask.sum()) * PIXEL_M ** 2 / 10_000
        reasons = []
        if alert:
            reasons.append(f"NDRE за неделю {ndre_chg:+.1f}%, NDVI {ndvi_chg:+.1f}%: Red Edge отстаёт")
            if hidden_ha:
                reasons.append(f"Скрытый стресс на {hidden_ha:.1f} га при нормальном NDVI")
        stats_keys = {"NDVI": "NDVI", "NDRE": "NDRE", "MSAVI": "MSAVI2", "NDMI": "NDMI"}
        write_json(stats / f"index_stats_{ds}.json", {  # формат как у Модуля 2 (schema 1.1)
            "schema_version": "1.1-mock",
            "date": d.isoformat(),
            "mean_ndvi": round(means["NDVI"], 6),
            "mean_ndre": round(means["NDRE"], 6),
            "mean_msavi": round(means["MSAVI"], 6),
            "mean_ndmi": round(means["NDMI"], 6),
            "early_stress_alert": bool(alert),
            "early_stress_area_ha": round(hidden_ha if alert else 0.0, 2),
            "early_stress_percent": round((hidden_ha / field_ha * 100) if alert else 0.0, 2),
            "field_area_ha": round(field_ha, 2),
            "indices": {
                stats_keys[k]: {
                    "mean": round(float(np.mean(v[field_mask])), 6),
                    "p5": round(float(np.percentile(v[field_mask], 5)), 6),
                    "p95": round(float(np.percentile(v[field_mask], 95)), 6),
                } for k, v in idx.items()
            },
            "zones": {"n_classes": 3, "classes": [
                {"class": zc, "label": ZONE_NAMES[zc], "area_ha": round(areas_ha.get(zc, 0.0), 2)}
                for zc in (1, 2, 3)
            ]},
            "early_stress": {"alert": bool(alert), "reasons": reasons,
                             "ndre_drop": None, "ndvi_drop": None},
            # мок-только: недельные изменения средних (у M2 их нет)
            "ndvi_change_pct": round(ndvi_chg, 2),
            "ndre_change_pct": round(ndre_chg, 2),
        })
        prev_means = means

        for name, arr in idx.items():
            vmin, vmax = INDEX_RANGES[name]
            colorize(arr, field_mask, vmin, vmax, INDEX_CMAPS[name]).save(ovl / f"{name}_{ds}.png")

        print(f"{ds}: стресс {areas_ha.get(1, 0) + areas_ha.get(2, 0):6.2f} га из {sum(areas_ha.values()):.2f} га | "
              f"NDVI {means['NDVI']:.3f} ({ndvi_chg:+.1f}%)  NDRE {means['NDRE']:.3f} ({ndre_chg:+.1f}%)"
              f"{'  ⚠ ALERT' if alert else ''}")

    print(f"\nГотово: {OUT_DIR}")


if __name__ == "__main__":
    main()
