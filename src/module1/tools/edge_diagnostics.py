"""Диагностика края поля (КТ4, задача C): почему NDVI/NDRE по западному и юго-западному краю ниже, чем внутри.

Только анализ: контур data/samples/field_boundary.geojson не меняется.

    python tools/edge_diagnostics.py [--data ../../data] [--scenes-all ../../data/scenes_all]
                                     [--buffer-dir DIR] [--out ../../docs/module1_kt4]
    python tools/edge_diagnostics.py --fetch-buffer DIR [--buffer-px 30]     # CDSE: сеть и ключи в .env

1. Пояса. Пояс 1 — пиксели поля, у которых 4-сосед вне поля; пояс 2 — следующий слой; внутренняя часть —
   глубже INTERIOR_DEPTH - 1 пикселей. 8 секторов по 45° от центроида маски. По каждой дате — среднее NDVI и
   NDRE пояса в секторе минус среднее внутренней части (NaN облаков исключены). Индексы считаются из сцен
   (B04, B05, B08) одинаково для всех дат; для дат сезона сверяются с indices/ М2.
2. Окружность по данным — только на датах сомкнутого покрова без облаков (FIT_DATES; 07-18 с облаком —
   для сравнения): в мае–июне поле засевали секторами от центра, маска «NDVI > порог» там — сектора, не круг.
   Порог NDVI — Оцу, и только если гистограмма бимодальна: у сглаженной гистограммы (KDE, полоса Скотта) ≥ 2 моды
   с массой ≥ MODE_MIN_MASS каждая и D Ашмана двух классов Оцу > 2; иначе окружность не подбирается.
   BC Сарле печатается для справки: при длинном хвосте (одна мода + хвост) он завышен и бимодальность не доказывает. Край — на 72 лучах от центра контура: первая
   точка снаружи от 0.6 R, где NDVI < порога; по этим точкам — окружность (геометрический МНК, soft_l1).
3. --fetch-buffer: те же продукты L2A, что у сцен сезона (scenes/scenes_manifest.json), каналы B04, B05, B08 и
   SCL на сетке сцен с запасом buffer_px во все стороны, без маски поля -> DIR/buffer_YYYY_MM_DD.tif.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
ROOT = HERE.parents[2]
REF_SCENE = ROOT / "data" / "samples" / "sample_scene.tif"
BOUNDARY = ROOT / "data" / "samples" / "field_boundary.geojson"

FIT_DATES = ("2026-07-03", "2026-07-28", "2026-08-07")   # сомкнутый покров, 0 % облаков в поле
COMPARE_DATES = ("2026-07-18",)                           # пик, но облако на юго-востоке (412 px NaN)
SECTORS = ("С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ")    # азимут 0° = север, по часовой
INTERIOR_DEPTH = 5
BC_THRESHOLD = 0.555          # коэффициент бимодальности равномерного распределения (справочно)
ASHMAN_D_THRESHOLD = 2.0      # D Ашмана > 2 — моды разделены
MODE_MIN_MASS = 0.05          # мода KDE учитывается, если между соседними минимумами ≥ 5 % пикселей
RAYS = 72
BUFFER_BANDS = ("B04", "B05", "B08")
SCL_NOT_CLEAR = (0, 1, 3, 8, 9, 10)  # нет данных, брак, тень облака, облака, перистые


def phase(date: str) -> str:
    if date < "2026-07-01":
        return "всходы (посев секторами)"
    return "сомкнутый покров" if date < "2026-09-01" else "созревание"


# ----------------------------------------------------------------------------
# Геометрия
# ----------------------------------------------------------------------------
def field_mask() -> tuple[np.ndarray, object]:
    import rasterio

    from s2fetch.grid import grid_of_file

    with rasterio.open(REF_SCENE) as src:
        return np.any(src.read() != 0, axis=0), grid_of_file(REF_SCENE)


def depth_map(mask: np.ndarray, levels: int = INTERIOR_DEPTH) -> np.ndarray:
    """0 вне поля; k — пиксель в k-м слое от края (4-связность); levels — всё, что глубже."""
    depth = np.zeros(mask.shape, np.int16)
    cur = mask.copy()
    for k in range(1, levels):
        p = np.pad(cur, 1, constant_values=False)
        inner = cur & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        depth[cur & ~inner] = k
        cur = inner
    depth[cur] = levels
    return depth


def pixel_xy(grid, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    t = grid.transform
    return t.c + (np.asarray(cols) + 0.5) * t.a, t.f + (np.asarray(rows) + 0.5) * t.e


def sector_map(mask: np.ndarray, grid) -> tuple[np.ndarray, tuple[float, float]]:
    rows, cols = np.nonzero(mask)
    x, y = pixel_xy(grid, rows, cols)
    cx, cy = float(x.mean()), float(y.mean())
    rr, cc = np.indices(mask.shape)
    gx, gy = pixel_xy(grid, rr, cc)
    az = np.degrees(np.arctan2(gx - cx, gy - cy)) % 360
    return (np.floor((az + 22.5) / 45).astype(int) % 8), (cx, cy)


def boundary_utm(crs: str):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import shapely

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gj = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    g = unary_union([shape(f["geometry"]) for f in gj["features"]])
    return shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def fit_circle(x: np.ndarray, y: np.ndarray, robust: bool = False) -> dict:
    """Kåsa (алгебраический) -> уточнение геометрическим МНК. Возвращает cx, cy, r, rms, n."""
    from scipy.optimize import least_squares

    a = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = d / 2, e / 2
    r = math.sqrt(f + cx ** 2 + cy ** 2)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r],
                        loss="soft_l1" if robust else "linear", f_scale=10.0)
    cx, cy, r = res.x
    resid = np.hypot(x - cx, y - cy) - r
    return {"cx": float(cx), "cy": float(cy), "r": float(r), "rms_m": float(np.sqrt(np.mean(resid ** 2))),
            "n": int(len(x))}


def offset(fit: dict, ref: dict) -> dict:
    dx, dy = fit["cx"] - ref["cx"], fit["cy"] - ref["cy"]
    az = math.degrees(math.atan2(dx, dy)) % 360
    return {"dx_east_m": round(dx, 1), "dy_north_m": round(dy, 1), "shift_m": round(math.hypot(dx, dy), 1),
            "azimuth_deg": round(az, 0), "direction": SECTORS[int((az + 22.5) // 45) % 8],
            "dr_m": round(fit["r"] - ref["r"], 1)}


# ----------------------------------------------------------------------------
# Индексы и статистика
# ----------------------------------------------------------------------------
def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out) & (a + b > 0), out, np.nan).astype(np.float32)


def scene_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    from s2fetch import BANDS

    with rasterio.open(path) as src:
        sc = src.read().astype(np.float32)
    sc[:, np.all(sc == 0, axis=0)] = np.nan
    b04, b05, b08 = (sc[BANDS.index(b)] for b in ("B04", "B05", "B08"))
    return nd(b08, b04), nd(b08, b05)


def otsu(v: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(between[:-1]))])


def kde_modes(v: np.ndarray, grid_n: int = 512) -> list[dict]:
    """Моды сглаженной гистограммы (gaussian_kde, полоса Скотта) с массой ≥ MODE_MIN_MASS: [{x, mass}]."""
    from scipy.stats import gaussian_kde

    xs = np.linspace(v.min(), v.max(), grid_n)
    dens = gaussian_kde(v)(xs)
    peaks = [i for i in range(1, grid_n - 1) if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1]]
    mins = [0] + [i for i in range(1, grid_n - 1) if dens[i] <= dens[i - 1] and dens[i] < dens[i + 1]] + [grid_n - 1]
    total = np.trapezoid(dens, xs)
    out = []
    for p in peaks:
        lo = max(m for m in mins if m <= p)
        hi = min(m for m in mins if m >= p)
        mass = float(np.trapezoid(dens[lo:hi + 1], xs[lo:hi + 1]) / total)
        if mass >= MODE_MIN_MASS:
            out.append({"x": round(float(xs[p]), 3), "mass": round(mass, 3)})
    return out


def bimodality(v: np.ndarray) -> dict:
    """Моды KDE, порог Оцу, D Ашмана двух классов Оцу, доли классов; BC Сарле — справочно."""
    from scipy.stats import kurtosis, skew

    v = v[np.isfinite(v)]
    n = len(v)
    g, k = skew(v, bias=False), kurtosis(v, bias=False)
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    thr = otsu(v)
    lo, hi = v[v <= thr], v[v > thr]
    dash = math.sqrt(2) * abs(hi.mean() - lo.mean()) / math.sqrt(lo.var() + hi.var())
    modes = kde_modes(v)
    ok = bool(len(modes) >= 2 and dash > ASHMAN_D_THRESHOLD)
    return {"n": n, "kde_modes": modes, "bc": round(float(bc), 3), "otsu": round(thr, 3),
            "ashman_d": round(float(dash), 2), "low_fraction": round(len(lo) / n, 3), "bimodal": ok}


def belt_table(ndvi: np.ndarray, ndre: np.ndarray, depth: np.ndarray, sectors: np.ndarray) -> dict:
    """{index: {interior, rings: {1|2: {сектор: {mean, delta, n}}}}}."""
    out = {}
    for name, arr in (("NDVI", ndvi), ("NDRE", ndre)):
        inner = arr[(depth >= INTERIOR_DEPTH) & np.isfinite(arr)]
        rec = {"interior": round(float(inner.mean()), 4), "interior_n": int(inner.size), "rings": {}}
        for ring in (1, 2):
            rec["rings"][ring] = {}
            for k, sname in enumerate(SECTORS):
                v = arr[(depth == ring) & (sectors == k) & np.isfinite(arr)]
                rec["rings"][ring][sname] = ({"mean": round(float(v.mean()), 4),
                                              "delta": round(float(v.mean() - inner.mean()), 4), "n": int(v.size)}
                                             if v.size else {"mean": None, "delta": None, "n": 0})
        out[name] = rec
    return out


def ray_edges(ndvi: np.ndarray, grid, center: tuple[float, float], r0: float, thr: float,
              inside: np.ndarray | None = None, run_m: float = 10.0, step: float = 2.5
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Край посева на RAYS лучах, начиная от окружности контура r0 (nearest, шаг step).

    На контуре посев (NDVI ≥ thr) — идём наружу до первого участка run_m подряд с NDVI < thr; нет — идём внутрь
    до первого участка run_m подряд с NDVI ≥ thr. Так пятна стресса в глубине поля край не подменяют.
    inside — маска, вне которой данных нет (локальные сцены): если посев доходит до контура, край снаружи не
    увидеть, луч отбрасывается (censored). Возвращает x, y, сдвиг края от r0 (м, + наружу), число отброшенных.
    """
    t = grid.transform
    need = max(1, int(round(run_m / step)))

    def sample(radii, a):
        px, py = center[0] + radii * math.sin(a), center[1] + radii * math.cos(a)
        c = np.floor((px - t.c) / t.a).astype(int)
        r = np.floor((py - t.f) / t.e).astype(int)
        ok = (r >= 0) & (r < ndvi.shape[0]) & (c >= 0) & (c < ndvi.shape[1])
        rr, cc = np.where(ok, r, 0), np.where(ok, c, 0)
        v = np.where(ok, ndvi[rr, cc], np.nan)
        if inside is not None:
            v = np.where(ok & inside[rr, cc], v, np.nan)
        return v

    def first_run(cond):
        run = 0
        for i, ok in enumerate(cond):
            run = run + 1 if ok else 0
            if run == need:
                return i - need + 1
        return None

    xs, ys, ds, censored = [], [], [], 0
    out_r = np.arange(r0, 1.5 * r0, step)
    in_r = np.arange(r0, 0.5 * r0, -step)
    for a in np.radians(np.arange(0, 360, 360 / RAYS)):
        v_in = sample(in_r - step / 2, a)            # чуть внутри контура: пиксель поля
        at_edge = v_in[:need][np.isfinite(v_in[:need])]          # первые run_m внутрь от контура
        if at_edge.size and at_edge.mean() >= thr:
            v_out = sample(out_r + step / 2, a)
            i = first_run(~(v_out >= thr) & np.isfinite(v_out))
            if i is None:
                censored += 1
                continue
            r = out_r[i]
        else:
            i = first_run(v_in >= thr)
            if i is None:
                censored += 1
                continue
            r = in_r[i]
        xs.append(center[0] + r * math.sin(a)); ys.append(center[1] + r * math.cos(a)); ds.append(r - r0)
    return np.array(xs), np.array(ys), np.array(ds), censored


def delta_by_depth(ndvi: np.ndarray, deep: np.ndarray, sectors: np.ndarray, layers: int = 6) -> dict:
    """NDVI слоя глубины 1..layers в секторе минус NDVI глубже 10 слоёв (без порога): ширина полосы у края."""
    core = ndvi[(deep >= 11) & np.isfinite(ndvi)].mean()
    out = {}
    for i, s in enumerate(SECTORS):
        row = []
        for k in range(1, layers + 1):
            v = ndvi[(deep == k) & (sectors == i) & np.isfinite(ndvi)]
            row.append(round(float(v.mean() - core), 3) if v.size else None)
        out[s] = row
    return out


# ----------------------------------------------------------------------------
# Докачка с буфером (CDSE)
# ----------------------------------------------------------------------------
def fetch_buffer(out_dir: Path, dates, buffer_px: int) -> int:
    import rasterio

    from s2fetch import catalog, reader
    from s2fetch.grid import boundary_bbox, grid_of_file, load_boundary
    from s2fetch.pipeline import L2A_ASSETS
    from s2fetch.scene import to_reflectance

    manifest = json.loads((ROOT / "data" / "scenes" / "scenes_manifest.json").read_text(encoding="utf-8"))
    ids = {s["date"]: s["l2a_id"] for s in manifest["scenes"] if s.get("l2a_id")}
    grid = grid_of_file(REF_SCENE).buffered(buffer_px)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader.configure_s3()
    client = catalog.open_client()
    bbox = boundary_bbox(load_boundary(BOUNDARY))
    for d in dates:
        items = catalog.search(client, catalog.L2A, bbox, d, d)
        item = items.get(ids.get(d, ""))
        if item is None:
            print(f"{d}: продукт {ids.get(d)} не найден в STAC", flush=True)
            return 1
        print(f"{d}: {item.id}, сетка {grid.width}×{grid.height} (+{buffer_px} px)", flush=True)
        dn = np.stack([reader.read_on_grid(item.assets[L2A_ASSETS[b]].href, grid, "bilinear") for b in BUFFER_BANDS])
        refl = to_reflectance(dn, catalog.boa_offset(item.id))
        scl = reader.read_on_grid(item.assets["SCL_20m"].href, grid, "nearest").astype(np.float32)
        path = out_dir / f"buffer_{d.replace('-', '_')}.tif"
        with rasterio.open(path, "w", driver="GTiff", count=4, dtype="float32", crs=grid.crs,
                           transform=grid.transform, width=grid.width, height=grid.height, compress="deflate") as dst:
            dst.write(np.concatenate([refl, scl[None]]))
            dst.descriptions = (*BUFFER_BANDS, "SCL")
            dst.update_tags(date=d, l2a_id=item.id, buffer_px=str(buffer_px))
        print(f"    -> {path}", flush=True)
    return 0


def check_buffer(path: Path, scene: Path) -> dict:
    """Кадр с запасом против сцены сезона: шаг и CRS те же, сдвиг начала — целое число пикселей, B04/B05/B08 внутри
    контура совпадают (max |diff|, облачные NaN сцены исключены). ok=False — дальше не считать."""
    import rasterio

    with rasterio.open(path) as b, rasterio.open(scene) as s:
        tb, ts = b.transform, s.transform
        ox, oy = (ts.c - tb.c) / tb.a, (ts.f - tb.f) / tb.e
        same = (tb.a, tb.e, tb.b, tb.d) == (ts.a, ts.e, ts.b, ts.d) and b.crs == s.crs
        integer = abs(ox - round(ox)) < 1e-6 and abs(oy - round(oy)) < 1e-6
        rec = {"offset_px": [round(ox, 6), round(oy, 6)], "same_crs_and_step": bool(same), "integer_offset": integer}
        if same and integer:
            bb, ss = b.read(), s.read()
            r0, c0 = int(round(oy)), int(round(ox))
            win = bb[:, r0:r0 + s.height, c0:c0 + s.width]
            valid = np.any(ss != 0, axis=0) & np.all(np.isfinite(ss), axis=0)
            from s2fetch import BANDS

            rec["pixels"] = int(valid.sum())
            rec["max_abs_diff"] = {n: float(np.abs(win[i][valid] - ss[BANDS.index(n)][valid]).max())
                                   for i, n in enumerate(BUFFER_BANDS)}
    rec["ok"] = bool(same and integer and max(rec.get("max_abs_diff", {"x": 1}).values()) < 1e-6)
    return rec


def read_buffer(path: Path):
    import rasterio

    from s2fetch.grid import grid_of_file

    with rasterio.open(path) as src:
        b04, b05, b08, scl = src.read().astype(np.float32)
    clear = ~np.isin(scl.astype(int), SCL_NOT_CLEAR)
    ndvi = np.where(clear, nd(b08, b04), np.nan)
    ndre = np.where(clear, nd(b08, b05), np.nan)
    return ndvi, ndre, grid_of_file(path), float((~clear).mean())


# ----------------------------------------------------------------------------
# Анализ
# ----------------------------------------------------------------------------
def analyse(data: Path, scenes_all: Path | None, buffer_dir: Path | None) -> dict:
    mask, grid = field_mask()
    depth = depth_map(mask)
    deep = depth_map(mask, 12)
    sectors, centroid = sector_map(mask, grid)
    field = boundary_utm(grid.crs)
    vx, vy = np.asarray(field.exterior.coords).T
    contour = fit_circle(vx, vy)
    contour["area_equiv_r_m"] = round(math.sqrt(field.area / math.pi), 1)
    doc: dict = {"field_pixels": int(mask.sum()), "centroid_utm": [round(c, 1) for c in centroid],
                 "contour_circle": {k: round(v, 2) if isinstance(v, float) else v for k, v in contour.items()},
                 "ring_pixels": {str(k): int((depth == k).sum()) for k in (1, 2)},
                 "interior_pixels": int((depth >= INTERIOR_DEPTH).sum()),
                 "sector_pixels_ring1": {s: int(((depth == 1) & (sectors == k)).sum()) for k, s in enumerate(SECTORS)},
                 "fit_dates": list(FIT_DATES), "compare_dates": list(COMPARE_DATES), "dates": {}}

    # 1. пояса по всем датам (scenes_all, иначе scenes)
    src_dir = scenes_all if scenes_all and scenes_all.is_dir() else data / "scenes"
    arrays = {}
    for p in sorted(src_dir.glob("scene_*.tif")):
        d = p.stem[len("scene_"):].replace("_", "-")
        ndvi, ndre = scene_indices(p)
        arrays[d] = (ndvi, ndre)
        rec = {"phase": phase(d), "source": p.relative_to(ROOT).as_posix(),
               "nan_in_field": int((~np.isfinite(ndvi) & mask).sum()), "belt": belt_table(ndvi, ndre, depth, sectors),
               "ndvi_delta_by_depth": delta_by_depth(ndvi, deep, sectors)}
        m2 = data / "indices" / f"NDVI_{p.stem[len('scene_'):]}.tif"
        if m2.exists():
            import rasterio

            with rasterio.open(m2) as s:
                ref = s.read(1)
            both = np.isfinite(ref) & np.isfinite(ndvi)
            rec["max_abs_diff_vs_m2_ndvi"] = float(np.abs(ref[both] - ndvi[both]).max())
        doc["dates"][d] = rec

    # 2. окружность: локально (внутри контура) и по буферу
    fits: dict = {}
    for d in (*FIT_DATES, *COMPARE_DATES):
        entry: dict = {}
        if d in arrays:
            v = arrays[d][0][mask]
            bm = bimodality(v)
            entry["local"] = {"histogram": bm}
            if bm["bimodal"]:
                x, y, dr, cens = ray_edges(arrays[d][0], grid, (contour["cx"], contour["cy"]), contour["r"],
                                           bm["otsu"], inside=mask)
                entry["local"].update(rays_censored=cens, edge_shift_by_sector_m=sector_shift(x, y, dr, contour))
                if cens > RAYS // 4:
                    entry["local"]["circle"] = None
                    entry["local"]["why_no_circle"] = (
                        f"на {cens} из {RAYS} лучей посев доходит до контура: край снаружи по обрезанной сцене не "
                        "увидеть, точки края есть только там, где посев кончается внутри — центр и радиус по ним "
                        "не определяются")
                elif len(x) >= 12:
                    f = fit_circle(x, y, robust=True)
                    entry["local"]["circle"] = {**{k: round(val, 2) if isinstance(val, float) else val
                                                   for k, val in f.items()}, "offset": offset(f, contour)}
            else:
                entry["local"]["circle"] = None
                entry["local"]["why_no_circle"] = (f"гистограмма NDVI внутри контура не бимодальна (мод KDE "
                                                   f"{len(bm['kde_modes'])}, D {bm['ashman_d']}): одна мода посева и "
                                                   "хвост, порог посев/не посев по ней не выбрать")
        bp = buffer_dir / f"buffer_{d.replace('-', '_')}.tif" if buffer_dir else None
        if bp and bp.exists():
            chk = check_buffer(bp, data / "scenes" / f"scene_{d.replace('-', '_')}.tif")
            doc.setdefault("buffer_check", {})[d] = chk
            if not chk["ok"]:
                raise SystemExit(f"{bp.name}: кадр с запасом не совпал со сценой сезона: {chk} — дальше не считаю")
            ndvi_b, ndre_b, grid_b, not_clear = read_buffer(bp)
            bm = bimodality(ndvi_b.ravel())
            entry["buffer"] = {"histogram": bm, "not_clear_fraction": round(not_clear, 4),
                               "size": [grid_b.width, grid_b.height]}
            if bm["bimodal"]:
                x, y, dr, cens = ray_edges(ndvi_b, grid_b, (contour["cx"], contour["cy"]), contour["r"], bm["otsu"])
                entry["buffer"].update(rays_censored=cens, edge_shift_by_sector_m=sector_shift(x, y, dr, contour))
                if len(x) >= 12:
                    f = fit_circle(x, y, robust=True)
                    entry["buffer"]["circle"] = {**{k: round(val, 2) if isinstance(val, float) else val
                                                    for k, val in f.items()}, "offset": offset(f, contour)}
                    entry["buffer"]["edge_points"] = [[round(a, 1), round(b, 1)] for a, b in zip(x, y)]
                # чувствительность к порогу: середина между модами KDE и ±0.1 от Оцу
                modes = [m["x"] for m in bm["kde_modes"]]
                sens = {}
                for name, thr in (("otsu-0.1", bm["otsu"] - 0.1), ("otsu", bm["otsu"]), ("otsu+0.1", bm["otsu"] + 0.1),
                                  ("mid_modes", (min(modes) + max(modes)) / 2)):
                    xs_, ys_, _, c_ = ray_edges(ndvi_b, grid_b, (contour["cx"], contour["cy"]), contour["r"], thr)
                    if len(xs_) >= 12:
                        f_ = fit_circle(xs_, ys_, robust=True)
                        sens[name] = {"threshold": round(thr, 3), "rms_m": round(f_["rms_m"], 1), **offset(f_, contour)}
                entry["buffer"]["threshold_sensitivity"] = sens
                entry["buffer"]["radial_profile_ndvi"] = radial_profile(ndvi_b, grid_b, contour)
            arrays[f"buffer:{d}"] = (ndvi_b, ndre_b, grid_b)
        fits[d] = entry
    doc["circle_fits"] = fits
    s2_fits = {d: (fits[d].get("buffer") or {}).get("circle") for d in FIT_DATES}
    if all(s2_fits.values()):
        doc["contour_variants"] = contour_variants(contour, s2_fits)
    doc["_arrays"] = arrays           # для картинок, в JSON не пишется
    doc["_geom"] = {"mask": mask, "grid": grid, "depth": depth, "sectors": sectors, "field": field,
                    "contour": contour}
    return doc


def radial_profile(ndvi: np.ndarray, grid, contour: dict, lo: int = -50, hi: int = 100, step: int = 10) -> dict:
    """Средний NDVI по секторам в кольцах шириной step м по расстоянию от окружности контура (− внутрь, + наружу)."""
    rr, cc = np.indices(ndvi.shape)
    x, y = pixel_xy(grid, rr, cc)
    dist = np.hypot(x - contour["cx"], y - contour["cy"]) - contour["r"]
    az = np.degrees(np.arctan2(x - contour["cx"], y - contour["cy"])) % 360
    k = (np.floor((az + 22.5) / 45).astype(int)) % 8
    edges = list(range(lo, hi + 1, step))
    out = {"bins_m": [[a, b] for a, b in zip(edges[:-1], edges[1:])]}
    for i, s in enumerate(SECTORS):
        row = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = ndvi[(k == i) & (dist >= a) & (dist < b) & np.isfinite(ndvi)]
            row.append(round(float(v.mean()), 3) if v.size else None)
        out[s] = row
    return out


def sector_shift(x: np.ndarray, y: np.ndarray, dr: np.ndarray, contour: dict) -> dict:
    """Медиана сдвига края посева от окружности контура (м, + наружу, − внутрь) по 8 секторам и число лучей."""
    if not len(x):
        return {}
    az = np.degrees(np.arctan2(x - contour["cx"], y - contour["cy"])) % 360
    k = (np.floor((az + 22.5) / 45).astype(int)) % 8
    return {s: {"median_m": round(float(np.median(dr[k == i])), 1), "rays": int((k == i).sum())}
            if (k == i).any() else {"median_m": None, "rays": 0} for i, s in enumerate(SECTORS)}


KT1_INNER_BUFFER_M = 30.0      # make_field_boundary.py: контур = окружность нарисованного aoi_raw.txt минус 30 м
MARGINS_M = (10, 20, 30)


def contour_variants(contour: dict, fits: dict) -> dict:
    """Нарисованная окружность КТ1 (центр контура, R + KT1_INNER_BUFFER_M) против окружности посева по Sentinel-2 и
    варианты «окружность S2 минус запас»: площадь, маска, сетка и сдвиг окна — тем же путём, что make_field_boundary.py
    (круг buffer(R, 64), WGS84 с 6 знаками) и s2fetch (grid_from_tile, field_mask). Опора плитки — transform эталона:
    окно s2fetch выровнено по пикселям плитки, важен только остаток по модулю 10 м (проверяется на текущем контуре)."""
    from pyproj import Transformer
    from shapely.geometry import Point

    from s2fetch.grid import field_mask, field_polygon, grid_from_tile, grid_of_file, load_boundary

    ref = grid_of_file(REF_SCENE)
    to_wgs = Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True)

    def via_geojson(poly):
        lon, lat = to_wgs.transform(*(np.array(v) for v in poly.exterior.xy))
        return field_polygon({"coordinates": [[[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]]}, ref.crs)

    def window(poly) -> dict:
        g = grid_from_tile(poly, ref.crs, ref.transform)
        n = int(field_mask(poly, g).sum())
        return {"mask_px": n, "mask_ha": n / 100, "polygon_ha": round(poly.area / 10_000, 2), "grid": [g.width, g.height],
                "origin": [g.transform.c, g.transform.f],
                "origin_shift_px": [round((g.transform.c - ref.transform.c) / ref.transform.a),
                                    round((g.transform.f - ref.transform.f) / ref.transform.e)]}

    cur = window(field_polygon(load_boundary(BOUNDARY), ref.crs))
    cur["same_as_reference"] = cur["grid"] == [ref.width, ref.height] and cur["origin_shift_px"] == [0, 0]
    good = [f for f in fits.values() if f]
    s2 = {k: float(np.mean([f[k] for f in good])) for k in ("cx", "cy", "r")}
    drawn = {"cx": contour["cx"], "cy": contour["cy"], "r": contour["r"] + KT1_INNER_BUFFER_M}
    margin_now = {}
    for name, th in (("С", 0), ("В", 90), ("Ю", 180), ("З", 270)):
        ux, uy = math.sin(math.radians(th)), math.cos(math.radians(th))
        dx, dy = contour["cx"] - s2["cx"], contour["cy"] - s2["cy"]
        b = dx * ux + dy * uy
        margin_now[name] = round(-b + math.sqrt(b * b - (dx * dx + dy * dy - s2["r"] ** 2)) - contour["r"], 1)
    out = {"current_contour": cur,
           "s2_circle_mean": {**{k: round(v, 1) for k, v in s2.items()}, "dates": len(good),
                              "r_range_m": [round(min(f["r"] for f in good), 1), round(max(f["r"] for f in good), 1)]},
           "kt1_drawn_circle": {**{k: round(v, 1) for k, v in drawn.items()},
                                "minus_s2": {"dr_m": round(drawn["r"] - s2["r"], 1), **offset(drawn, s2)}},
           "contour_margin_to_s2_edge_m": {"planned": KT1_INNER_BUFFER_M, **margin_now},
           "variants": {}}
    for m in MARGINS_M:
        v = window(via_geojson(Point(s2["cx"], s2["cy"]).buffer(s2["r"] - m, 64)))
        per = [window(via_geojson(Point(f["cx"], f["cy"]).buffer(f["r"] - m, 64)))["mask_px"] for f in good]
        out["variants"][f"s2_minus_{m}m"] = {"r_m": round(s2["r"] - m, 1), **v, "mask_px_by_date": [min(per), max(per)]}
    return out


# ----------------------------------------------------------------------------
# Картинки
# ----------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # категориальные слоты 1–3 (проверены all-pairs)
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plots(doc: dict, out: Path) -> list[str]:
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-edge"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    g = doc["_geom"]
    grid, mask, depth, sectors, field, contour = (g[k] for k in ("grid", "mask", "depth", "sectors", "field",
                                                                 "contour"))
    cx, cy = contour["cx"], contour["cy"]
    files = []
    div = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])

    # --- 1. карта: NDVI одной даты, контур, окружности; пояс по секторам ---
    show = next((f"buffer:{d}" for d in FIT_DATES[::-1] if f"buffer:{d}" in doc["_arrays"]), FIT_DATES[-1])
    if show.startswith("buffer:"):
        ndvi, _, sgrid = doc["_arrays"][show]
    else:
        ndvi, sgrid = doc["_arrays"][show][0], grid
    t = sgrid.transform
    ext = [t.c - cx, t.c + sgrid.width * t.a - cx, t.f + sgrid.height * t.e - cy, t.f - cy]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(ndvi, extent=ext, cmap="Greens", vmin=0.1, vmax=0.95, interpolation="nearest")
    fx, fy = np.asarray(field.exterior.coords).T
    ax.plot(fx - cx, fy - cy, color=INK, lw=1.5, label="контур field_boundary.geojson")
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(contour["r"] * np.sin(th) + contour["cx"] - cx, contour["r"] * np.cos(th) + contour["cy"] - cy,
            color=INK2, lw=1, ls=":", label=f"окружность контура (R = {contour['r']:.0f} м)")
    key = show.split(":")[-1]
    entry = doc["circle_fits"].get(key, {})
    fit = (entry.get("buffer") or {}).get("circle") or (entry.get("local") or {}).get("circle")
    if fit:
        ax.plot(fit["r"] * np.sin(th) + fit["cx"] - cx, fit["r"] * np.cos(th) + fit["cy"] - cy, color=S2, lw=2,
                ls="--", label=f"окружность по NDVI (R = {fit['r']:.0f} м, сдвиг {fit['offset']['shift_m']:.0f} м "
                               f"на {fit['offset']['direction']})")
        ax.plot(fit["cx"] - cx, fit["cy"] - cy, marker="x", color=S2, ms=8, mew=2)
        pts = (entry.get("buffer") or {}).get("edge_points")
        if pts:
            p = np.array(pts)
            ax.scatter(p[:, 0] - cx, p[:, 1] - cy, s=10, color=S2, zorder=3, linewidths=0)
    ax.plot(0, 0, marker="+", color=INK, ms=10, mew=1.5)
    ax.set_title(f"NDVI {key}{' (кадр с запасом 300 м)' if show.startswith('buffer:') else ' (внутри контура)'}",
                 color=INK, fontsize=10, loc="left")
    ax.set_xlabel("м на восток от центра контура", color=INK2, fontsize=8)
    ax.set_ylabel("м на север", color=INK2, fontsize=8)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
    _style(ax); ax.grid(False)

    ax = axes[1]
    canopy = [d for d, r in doc["dates"].items() if r["phase"] == "сомкнутый покров"]
    delta = np.full(mask.shape, np.nan)
    for k, s in enumerate(SECTORS):
        for ring in (1, 2):
            vals = [doc["dates"][d]["belt"]["NDVI"]["rings"][ring][s]["delta"] for d in canopy]
            vals = [v for v in vals if v is not None]
            if vals:
                delta[(depth == ring) & (sectors == k)] = np.mean(vals)
    lim = float(np.nanmax(np.abs(delta)))
    t = grid.transform
    ext0 = [t.c - cx, t.c + grid.width * t.a - cx, t.f + grid.height * t.e - cy, t.f - cy]
    ax.imshow(np.where(mask & (depth > 2), 0.0, np.nan), extent=ext0, cmap=div, norm=TwoSlopeNorm(0, -lim, lim),
              interpolation="nearest", alpha=0.35)
    im2 = ax.imshow(delta, extent=ext0, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax.plot(fx - cx, fy - cy, color=INK, lw=1)
    for k, s in enumerate(SECTORS):
        a = math.radians(k * 45)
        v = np.mean([doc["dates"][d]["belt"]["NDVI"]["rings"][1][s]["delta"] for d in canopy
                     if doc["dates"][d]["belt"]["NDVI"]["rings"][1][s]["delta"] is not None])
        ax.text(0.7 * contour["r"] * math.sin(a), 0.7 * contour["r"] * math.cos(a), f"{s}\n{v:+.3f}",
                ha="center", va="center", fontsize=8, color=INK)
    ax.set_title(f"Пояса 1–2: Δ NDVI с внутренней частью ({len(canopy)} дат покрова)",
                 color=INK, fontsize=10, loc="left")
    ax.set_xlabel("м на восток от центра контура", color=INK2, fontsize=8)
    fig.colorbar(im2, ax=ax, shrink=0.8, label="Δ NDVI (красный — ниже, чем внутри)")
    _style(ax); ax.grid(False)
    p = out / "edge_map.png"
    fig.savefig(p, dpi=130); plt.close(fig); files.append(p.name)

    # --- 1б. крупно: западный и восточный край кадра с запасом ---
    if show.startswith("buffer:") and fit:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
        for ax, (name, sx) in zip(axes, (("западный край", -1), ("восточный край", 1))):
            ax.imshow(ndvi, extent=ext, cmap="Greens", vmin=0.1, vmax=0.95, interpolation="nearest")
            ax.plot(fx - cx, fy - cy, color=INK, lw=1.5, label="контур field_boundary.geojson")
            ax.plot(fit["r"] * np.sin(th) + fit["cx"] - cx, fit["r"] * np.cos(th) + fit["cy"] - cy, color=S2, lw=2,
                    ls="--", label="окружность по NDVI")
            ax.set_xlim(sx * contour["r"] - 90, sx * contour["r"] + 90)
            ax.set_ylim(-120, 120)
            ax.set_title(f"NDVI {key}, {name}: шаг сетки 10 м", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("м на восток от центра контура", color=INK2, fontsize=8)
            _style(ax); ax.grid(False)
        axes[0].set_ylabel("м на север", color=INK2, fontsize=8)
        axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
        p = out / "edge_zoom.png"
        fig.savefig(p, dpi=130); plt.close(fig); files.append(p.name)

    # --- 2. по датам: 8 секторов в раскладке компаса ---
    layout = {"СЗ": (0, 0), "С": (0, 1), "СВ": (0, 2), "З": (1, 0), "В": (1, 2), "ЮЗ": (2, 0), "Ю": (2, 1),
              "ЮВ": (2, 2)}
    import datetime as dt

    dates = sorted(doc["dates"])
    xd = [dt.date.fromisoformat(d) for d in dates]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    series = (("NDVI", 1, S1, "-", "NDVI, пояс 1"), ("NDVI", 2, S2, "-", "NDVI, пояс 2"),
              ("NDRE", 1, S3, "--", "NDRE, пояс 1"))
    emerg = [x for x, d in zip(xd, dates) if doc["dates"][d]["phase"].startswith("всходы")]
    for s, (i, j) in layout.items():
        ax = axes[i][j]
        if emerg:
            ax.axvspan(min(emerg) - dt.timedelta(days=4), max(emerg) + dt.timedelta(days=4), color="#f0efec", lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        for idx, ring, col, ls, label in series:
            y = [doc["dates"][d]["belt"][idx]["rings"][ring][s]["delta"] for d in dates]
            ax.plot(xd, [np.nan if v is None else v for v in y], color=col, lw=1.6, ls=ls, marker="o", ms=3.5,
                    label=label)
        ax.set_title(s, color=INK, fontsize=10, loc="left")
        _style(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    c = axes[1][1]
    c.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    c.legend(handles, labels, loc="center", fontsize=9, frameon=False,
             title="пояс − внутренняя часть\nсерый фон — всходы\n(поле сеяли секторами)", title_fontsize=9)
    fig.supylabel("разница индекса с внутренней частью поля", color=INK2, fontsize=9)
    p = out / "edge_by_date.png"
    fig.savefig(p, dpi=130); plt.close(fig); files.append(p.name)

    # --- 3. гистограммы NDVI с порогом Оцу ---
    hd = list(FIT_DATES) + list(COMPARE_DATES)
    rows = [("local", "внутри контура")] + ([("buffer", "кадр с запасом 300 м")]
                                           if any("buffer" in doc["circle_fits"][d] for d in hd) else [])
    fig, axes = plt.subplots(len(rows), len(hd), figsize=(3.2 * len(hd), 2.8 * len(rows)), squeeze=False,
                             constrained_layout=True)
    for ri, (kind, title) in enumerate(rows):
        for ci, d in enumerate(hd):
            ax = axes[ri][ci]
            e = doc["circle_fits"][d].get(kind)
            if not e:
                ax.axis("off"); continue
            v = (doc["_arrays"][d][0][mask] if kind == "local" else doc["_arrays"][f"buffer:{d}"][0].ravel())
            v = v[np.isfinite(v)]
            ax.hist(v, bins=80, color=S1, edgecolor="white", linewidth=0.3)
            h = e["histogram"]
            ax.axvline(h["otsu"], color=INK, lw=1.2, ls="--")
            verdict = "бимодальна" if h["bimodal"] else "не бимодальна"
            ax.set_title(f"{d} · {title}", color=INK, fontsize=9, loc="left")
            ax.text(0.02, 0.97, f"мод KDE {len(h['kde_modes'])}  D {h['ashman_d']}  BC {h['bc']}\n"
                    f"Оцу {h['otsu']} — {verdict}",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
            ax.set_xlabel("NDVI", color=INK2, fontsize=8)
            _style(ax)
    p = out / "edge_hist.png"
    fig.savefig(p, dpi=130); plt.close(fig); files.append(p.name)
    return files


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Диагностика края поля (КТ4, задача C)")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--scenes-all", type=Path, default=ROOT / "data" / "scenes_all")
    ap.add_argument("--buffer-dir", type=Path, default=None, help="папка buffer_*.tif от --fetch-buffer")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "module1_kt4")
    ap.add_argument("--fetch-buffer", type=Path, default=None, metavar="DIR",
                    help="скачать кадры с запасом в DIR (CDSE, ключи в .env) и выйти")
    ap.add_argument("--buffer-px", type=int, default=30)
    a = ap.parse_args(argv)
    if a.fetch_buffer:
        return fetch_buffer(a.fetch_buffer, (*FIT_DATES, *COMPARE_DATES), a.buffer_px)
    a.out.mkdir(parents=True, exist_ok=True)
    doc = analyse(a.data, a.scenes_all, a.buffer_dir)
    files = plots(doc, a.out)
    clean = {k: v for k, v in doc.items() if not k.startswith("_")}
    clean["figures"] = files
    (a.out / "edge_stats.json").write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contour_circle": clean["contour_circle"],
                      "circle_fits": {d: {k: {kk: vv for kk, vv in e.items() if kk != "edge_points"}
                                          for k, e in v.items()} for d, v in clean["circle_fits"].items()}},
                     ensure_ascii=False, indent=1))
    print(f"-> {a.out}: edge_stats.json, {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
