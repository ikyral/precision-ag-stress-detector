"""Диагностика: мелкие облака и их тени, которые не видят SCL / MSK_CLDPRB / s2cloudless.

Только отчёт, сцены не меняет. Сравнивает каждую дату серии с медианой ±2 соседних дат
(поэтому для обработки новой даты «по мере появления» не подходит — см. docs/module1_season_2026.md).

Облако: относительный рост B02 > 1.25 и B04 > 1.15 при B08 не ниже 0.98 (отношения к соседям,
нормированные на медиану по полю — общий сдвиг яркости вроде дымки не мешает), абсолютный рост
B02 > 0.006, NDVI > 0.6 на всех соседних датах (сомкнутый посев), группа не меньше 2×2 px.
Тень: маска облака сдвигается от солнца для высот 300–4000 м; берётся самый тёмный сдвиг, если средняя
относительная яркость (B02, B04, B08, B11) в нём < 0.93. Углы солнца берутся из STAC по тегу source.

Запуск из src/module1:
    python tools/temporal_cloud_check.py ../../data/scenes [--geojson candidates.geojson]
Кандидаты в --geojson записываются в формате manual_masks.geojson: проверьте глазами и перенесите вручную.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

B02, B04, B08, B11 = 0, 2, 6, 8
NEIGHBORS = 2
CLOUD = dict(b02=1.25, b04=1.15, b08_min=0.98, b02_abs=0.006, ndvi_neighbors=0.6)
SHADOW_HEIGHTS_M = range(300, 4001, 100)
SHADOW_MAX_BRIGHTNESS = 0.93


def _rel(a: np.ndarray, ref: np.ndarray, inside: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        r = a / ref
    return r / np.nanmedian(r[inside])


def _ndvi(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (x[..., B08, :, :] - x[..., B04, :, :]) / (x[..., B08, :, :] + x[..., B04, :, :])


def cloud_mask(cur: np.ndarray, nbs: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """cur [10,H,W], nbs [N,10,H,W] -> маска кандидатов-облаков."""
    from scipy.ndimage import binary_dilation, binary_opening

    ref = np.nanmedian(nbs, 0)
    dabs = cur[B02] - ref[B02]
    dabs = dabs - np.nanmedian(dabs[inside])
    with np.errstate(invalid="ignore"):
        m = (inside & (np.nanmin(_ndvi(nbs), 0) > CLOUD["ndvi_neighbors"])
             & (_rel(cur[B02], ref[B02], inside) > CLOUD["b02"])
             & (_rel(cur[B04], ref[B04], inside) > CLOUD["b04"])
             & (_rel(cur[B08], ref[B08], inside) >= CLOUD["b08_min"])
             & (dabs > CLOUD["b02_abs"]))
    core = binary_opening(m, np.ones((2, 2), bool))
    return core | (m & binary_dilation(core))


def shadow_mask(cloud: np.ndarray, cur: np.ndarray, nbs: np.ndarray, inside: np.ndarray,
                sun_az: float, sun_el: float, res: float = 10.0):
    """-> (маска тени, высота облака м, средняя относительная яркость) или (None, None, None)."""
    from scipy.ndimage import shift

    if cloud.sum() < 3:
        return None, None, None
    ref = np.nanmedian(nbs, 0)
    br = np.nanmean(np.stack([_rel(cur[i], ref[i], inside) for i in (B02, B04, B08, B11)]), 0)
    best = (1.0, None, None)
    for h in SHADOW_HEIGHTS_M:
        dist = h / np.tan(np.radians(sun_el)) / res
        dy, dx = dist * np.cos(np.radians(sun_az)), -dist * np.sin(np.radians(sun_az))  # от солнца
        sh = (shift(cloud.astype(float), (dy, dx), order=0) > 0.5) & inside & ~cloud
        if sh.sum() >= 3:
            v = float(np.nanmean(br[sh]))
            if v < best[0]:
                best = (v, h, sh)
    if best[1] is None or best[0] >= SHADOW_MAX_BRIGHTNESS:
        return None, None, None
    return best[2], best[1], best[0]


def sun_angles(source_ids: list[str]) -> dict[str, tuple[float, float]]:
    """{id: (азимут, высота)} из CDSE STAC; без сети — пустой словарь."""
    try:
        from s2fetch import catalog
        items = catalog.open_client().search(collections=[catalog.L2A], ids=source_ids).items()
        return {it.id: (it.properties["view:sun_azimuth"], it.properties["view:sun_elevation"]) for it in items}
    except Exception as exc:  # noqa: BLE001
        print(f"Углы солнца недоступны ({exc}); тени не ищутся")
        return {}


def main(argv=None) -> int:
    import rasterio

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN-срезы (облака/вне поля) — ожидаемы

    from s2fetch.grid import grid_of_file
    from s2fetch.manual import polygons_from_mask

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", type=Path, help="папка со scene_YYYY_MM_DD.tif")
    ap.add_argument("--geojson", type=Path, help="записать кандидатов в формате manual_masks.geojson")
    a = ap.parse_args(argv)

    files = sorted(a.scenes.glob("scene_*.tif"))
    if len(files) < 3:
        print("Нужно не меньше 3 дат")
        return 1
    stack, sources = [], []
    for f in files:
        with rasterio.open(f) as src:
            stack.append(src.read())
            sources.append(src.tags().get("source", ""))
    A = np.stack(stack)
    inside = A[0, 0] != 0
    grid = grid_of_file(files[0])
    sun = sun_angles([s for s in sources if s])
    feats = []
    print("дата        облако%  тень%  высота_м  яркость_тени")
    for t, f in enumerate(files):
        date = f.stem[6:].replace("_", "-")
        nb = [k for k in range(max(0, t - NEIGHBORS), min(len(files), t + NEIGHBORS + 1)) if k != t]
        cl = cloud_mask(A[t], A[nb], inside)
        sh, h, v = (shadow_mask(cl, A[t], A[nb], inside, *sun[sources[t]])
                    if sources[t] in sun else (None, None, None))
        pct = lambda m: 0.0 if m is None else 100 * m.sum() / inside.sum()  # noqa: E731
        print(f"{date}  {pct(cl):6.2f}  {pct(sh):5.2f}  {h or '—':>8}  {'—' if v is None else f'{v:.3f}'}")
        for kind, m in (("cloud", cl), ("shadow", sh)):
            if m is not None and m.any():
                for g in polygons_from_mask(m, grid):
                    feats.append({"type": "Feature", "geometry": g, "properties": {
                        "date": date, "kind": kind, "note": "кандидат temporal_cloud_check, проверить глазами"}})
    if a.geojson:
        epsg = grid.crs.split(":")[-1]
        a.geojson.write_text(json.dumps({"type": "FeatureCollection", "crs": {
            "type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}, "features": feats},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Кандидаты: {a.geojson} ({len(feats)} полиг.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
