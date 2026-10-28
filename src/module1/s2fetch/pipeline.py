"""Серия сцен за интервал: поиск -> оценка облачности -> выбор -> запись -> манифест."""
from __future__ import annotations

import datetime as dt
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import BANDS, catalog, clouds, manifest, manual, preview, reader, selection
from .grid import (Grid, boundary_bbox, compare_grids, field_mask, field_polygon,
                   grid_from_tile, grid_of_file, load_boundary)
from .scene import assemble, scene_filename, to_reflectance, write_scene

L2A_ASSETS = {b: f"{b}_{'10m' if b in ('B02', 'B03', 'B04', 'B08') else '20m'}" for b in BANDS}


@dataclass
class Options:
    boundary: Path
    start: str
    end: str
    out: Path
    max_cloud: float = 10.0
    weekly: bool = False
    dry_run: bool = False
    force: bool = False
    tile: str = "T37TEN"
    dilate_px: int = 2
    reference: Path | None = None
    manual_masks: Path | None = None


@dataclass
class DateResult:
    date: str
    l2a_id: str
    l1c_id: str | None = None
    cloud_mask: str = "SCL"
    verdict: clouds.CloudVerdict | None = None
    masks: dict = field(default_factory=dict)   # для превью
    selected: bool = False
    reason: str = ""
    status: str = ""
    error: str | None = None
    alternatives: list[str] = field(default_factory=list)

    def entry(self, out_dir: Path) -> dict:
        v = self.verdict.as_dict() if self.verdict else {}
        fname = scene_filename(self.date)
        return {
            "date": self.date, "week": selection.iso_week(self.date),
            "l2a_id": self.l2a_id, "l1c_id": self.l1c_id, "cloud_mask": self.cloud_mask,
            **v, "selected": self.selected, "reason": self.reason or v.get("reason", ""),
            "file": fname if (out_dir / fname).exists() and self.selected else None,
            "status": self.status, "error": self.error, "other_versions": self.alternatives,
        }

    @property
    def s2c_only_reject(self) -> bool:
        """Отброшена только из-за s2cloudless: по SCL чистая, по объединению — нет."""
        v = self.verdict
        # union_percent считается только после прохода по SCL, и тогда отброс — только по объединению
        return bool(v and not v.accepted and v.union_percent is not None)


def _crop(arr: np.ndarray, px: int) -> np.ndarray:
    return arr[..., px:-px, px:-px] if px else arr


def evaluate_date(res: DateResult, l2a, l1c_items: dict, grid: Grid, inside: np.ndarray,
                  opt: Options, manual_polys: list | None = None) -> None:
    """Облачность в поле: SCL, затем s2cloudless по L1C (запасной вариант — MSK_CLDPRB) и ручная маска."""
    scl = reader.read_on_grid(l2a.assets["SCL_20m"].href, grid, "nearest")
    cloud_scl, invalid = clouds.scl_masks(scl)
    res.masks = {"scl": cloud_scl, "invalid": invalid}
    pre = clouds.precheck(scl, inside, opt.max_cloud)
    if pre is not None:
        res.verdict = pre
        return

    extra = None
    res.l1c_id = catalog.match_l1c(res.l2a_id, list(l1c_items))
    if res.l1c_id:
        try:
            l1c = l1c_items[res.l1c_id]
            big = grid.buffered(clouds.S2C_BUFFER_PX)
            off = catalog.boa_offset(res.l1c_id)
            bands = np.stack([reader.read_on_grid(l1c.assets[b].href, big, "bilinear")
                              for b in clouds.S2C_BANDS]).astype(np.float32)
            refl = np.clip(bands - off, 0, None) / 10000.0
            _, m = clouds.s2cloudless_mask(refl)
            extra = _crop(m, clouds.S2C_BUFFER_PX)
            res.cloud_mask = (f"SCL 3,8,9,10 + s2cloudless (L1C, p>{clouds.S2C_THRESHOLD}, "
                              f"average_over={clouds.S2C_AVERAGE_OVER}) + расширение {opt.dilate_px} px")
        except Exception as exc:  # noqa: BLE001 — переходим на запасной вариант
            print(f"    s2cloudless не удался ({exc}); беру MSK_CLDPRB", flush=True)
            res.l1c_id = None
    if extra is None:
        prob = reader.read_on_grid(l2a.assets["CLD_20m"].href, grid, "bilinear")
        extra = prob >= clouds.CLDPRB_THRESHOLD
        res.cloud_mask = (f"SCL 3,8,9,10 + MSK_CLDPRB>={clouds.CLDPRB_THRESHOLD}% (L1C нет) "
                          f"+ расширение {opt.dilate_px} px")
    res.masks["s2c"] = extra & ~cloud_scl
    man = manual.mask_for_date(manual_polys or [], res.date, grid)
    if man is not None:
        res.masks["manual"] = man
        n = sum(p.date == res.date for p in manual_polys)
        res.cloud_mask = res.cloud_mask.replace(" + расширение", f" + manual ({n} полиг.) + расширение")
    res.verdict = clouds.evaluate(scl, extra, inside, opt.max_cloud, opt.dilate_px, man)
    if res.verdict.nan_mask is not None:
        res.masks["dilated"] = res.verdict.nan_mask


def read_scene(l2a, grid: Grid, offset: int) -> np.ndarray:
    layers = []
    for b in BANDS:
        print(f"    читаю {b}", flush=True)
        layers.append(reader.read_on_grid(l2a.assets[L2A_ASSETS[b]].href, grid, "bilinear"))
    return to_reflectance(np.stack(layers), offset)


def _select(results: list[DateResult], weekly: bool) -> None:
    clean = {r.date: r.verdict.nan_percent for r in results if r.verdict and r.verdict.accepted}
    chosen = set(clean)
    if weekly:
        best = selection.pick_weekly(clean)
        chosen = set(best.values())
        for r in results:
            if r.date in clean and r.date not in chosen:
                r.reason = f"чистая, но на неделе {selection.iso_week(r.date)} выбрана " \
                           f"{best[selection.iso_week(r.date)]} (меньше облаков)"
    for r in results:
        r.selected = r.date in chosen
        if r.selected:
            r.reason = "чистая, выбрана"


def print_table(results: list[DateResult], opt: Options) -> None:
    print(f"\n{'Дата':10} {'Неделя':8} {'SCL0/1':>6} {'SCL%':>6} {'∪s2c%':>6} {'ручн%':>6} {'NaN%':>6}  Решение")
    by_week: dict[str, list[DateResult]] = {}
    for r in results:
        by_week.setdefault(selection.iso_week(r.date), []).append(r)
    f = lambda x: f"{'—':>6}" if x is None else f"{x:6.1f}"  # noqa: E731
    for wk in selection.weeks_between(opt.start, opt.end):
        rows = by_week.get(wk, [])
        for r in rows:
            v = r.verdict
            if v is None:
                print(f"{r.date:10} {wk:8} {'':>34}  ОШИБКА: {r.error}")
                continue
            mark = "+" if r.selected else " "
            print(f"{r.date:10} {wk:8} {f(v.invalid_percent)} {f(v.scl_percent)} "
                  f"{f(v.union_percent)} {f(v.manual_percent or None)} {f(v.nan_percent)}  {mark} {r.reason or v.reason}")
        if not any(r.selected for r in rows):
            print(f"{'':10} {wk:8} {'':>34}  -- пропуск: нет чистой сцены"
                  + ("" if rows else " (снимков нет)"))
    sel = [r for r in results if r.selected]
    print(f"\nВыбрано дат: {len(sel)}: {', '.join(r.date for r in sel)}")
    only = [r for r in results if r.s2c_only_reject]
    if only:
        print("Отсеяны не по SCL (s2cloudless или ручная маска): "
              + ", ".join(f"{r.date} (SCL {r.verdict.scl_percent:.1f}%, ∪ {r.verdict.union_percent:.1f}%)"
                          for r in only))


def run(opt: Options) -> int:
    """CLI: код выхода 0 — ок, 1 — ошибки/нет сцен, 2 — сетка не совпала с эталоном."""
    return fetch_series(opt)[0]


def fetch_series(opt: Options) -> tuple[int, list[DateResult]]:
    """Как run(), плюс результаты по датам: status 'written' (записана в этом запуске), 'exists'
    (уже была, не перезаписана), 'error', 'evaluated' (не выбрана); selected — выбрана."""
    geom = load_boundary(opt.boundary)
    bbox = boundary_bbox(geom)
    client = catalog.open_client()
    print(f"Поиск {catalog.L2A} и {catalog.L1C} за {opt.start}..{opt.end}", flush=True)
    l2a_items = catalog.search(client, catalog.L2A, bbox, opt.start, opt.end)
    l1c_items = catalog.search(client, catalog.L1C, bbox, opt.start, opt.end)
    per_date = catalog.one_per_date(list(l2a_items), opt.tile)
    other_tiles = sorted({catalog.parse_id(i).tile for i in l2a_items} - {opt.tile})
    print(f"L2A: {len(l2a_items)} item, L1C: {len(l1c_items)}; плитка {opt.tile}: {len(per_date)} дат"
          + (f"; отброшены плитки {', '.join(other_tiles)}" if other_tiles else ""), flush=True)
    if not per_date:
        print("Сцен не найдено")
        return 1, []

    manual_polys = manual.load(opt.manual_masks)
    if manual_polys:
        print(f"Ручные маски: {opt.manual_masks} — {len(manual_polys)} полиг. на даты "
              f"{', '.join(sorted({p.date for p in manual_polys}))}")
    reader.configure_s3()
    # --- общая сетка: по B04 плитки (алгоритм КТ1), сверка с эталоном ---
    first = l2a_items[next(iter(per_date.values()))]
    crs, t_tile = reader.tile_transform(first.assets["B04_10m"].href)
    poly = field_polygon(geom, crs)
    grid = grid_from_tile(poly, crs, t_tile)
    inside = field_mask(poly, grid)
    print(f"Сетка: {grid.crs}, {grid.width}x{grid.height}, origin ({grid.transform.c}, {grid.transform.f}); "
          f"пикселей в контуре: {int(inside.sum())}")
    if opt.reference and opt.reference.exists():
        diffs = compare_grids(grid, grid_of_file(opt.reference))
        if diffs:
            print("Сетка не совпадает с эталоном", opt.reference, ":", "; ".join(diffs))
            return 2, []
        import rasterio
        with rasterio.open(opt.reference) as ref:
            ref_inside = ref.read(1) != 0
        if not np.array_equal(ref_inside, inside):
            print(f"Маска поля не совпадает с эталоном: {int((ref_inside != inside).sum())} px")
            return 2, []
        print(f"Сетка и маска поля совпадают с эталоном {opt.reference.name}")

    # --- оценка облачности по датам ---
    results: list[DateResult] = []
    for date, l2a_id in per_date.items():
        same_tile = [i for i in l2a_items if catalog.parse_id(i).tile == opt.tile
                     and catalog.parse_id(i).date == date and i != l2a_id]
        res = DateResult(date, l2a_id, alternatives=sorted(same_tile))
        print(f"{date} {l2a_id}", flush=True)
        try:
            crs_d, t_d = reader.tile_transform(l2a_items[l2a_id].assets["SCL_20m"].href)
            if crs_d != grid.crs or (t_d.c, t_d.f) != (t_tile.c, t_tile.f):
                raise RuntimeError(f"сетка плитки другая: {crs_d} {t_d.c},{t_d.f}")
            evaluate_date(res, l2a_items[l2a_id], l1c_items, grid, inside, opt, manual_polys)
            res.status = "evaluated"
        except Exception as exc:  # noqa: BLE001 — одна дата не обрывает серию
            res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
            res.reason = "ошибка чтения"
            traceback.print_exc()
        results.append(res)

    _select(results, opt.weekly)
    print_table(results, opt)
    if opt.dry_run:
        print("\n--dry-run: файлы не записаны")
        return 0, results

    # --- запись выбранных сцен ---
    prev_dir = opt.out / "preview"
    pngs = []
    fresh = False  # записано хоть одно превью — пересобрать all_dates.png
    for r in results:
        path = opt.out / scene_filename(r.date)
        if r.selected:
            if path.exists() and not opt.force:
                print(f"{r.date}: уже есть {path.name}, пропускаю (--force перезапишет)")
                r.status = "exists"
            else:
                print(f"{r.date}: выгружаю {r.l2a_id}", flush=True)
                try:
                    off = catalog.boa_offset(r.l2a_id)
                    refl = read_scene(l2a_items[r.l2a_id], grid, off)
                    data = assemble(refl, inside, r.verdict.nan_mask)
                    write_scene(path, data, grid, {
                        "date": r.date, "source": r.l2a_id, "l1c_source": r.l1c_id or "none",
                        "scale": "reflectance", "boa_offset_removed": off,
                        "cloud_percent_in_field": round(r.verdict.nan_percent, 2),
                        "scl_cloud_percent": round(r.verdict.scl_percent, 2),
                        "union_cloud_percent": round(r.verdict.union_percent, 2),
                        "manual_mask_percent": round(r.verdict.manual_percent, 2),
                        "cloud_mask": r.cloud_mask, "tile": opt.tile})
                    r.status = "written"
                    p = prev_dir / f"{path.stem}.png"
                    preview.save_png(p, preview.render(np.where(inside[None], refl, np.nan), inside, r.masks),
                                     f"{r.date}  RGB | NDVI | mask  NaN {r.verdict.nan_percent:.1f}%")
                except Exception as exc:  # noqa: BLE001
                    r.status, r.error = "error", f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    continue
            if r.status == "exists":
                p = prev_dir / f"{path.stem}.png"
                if not p.exists() or p.stat().st_mtime < path.stat().st_mtime:  # превью старше сцены
                    import rasterio
                    with rasterio.open(path) as src:
                        preview.save_png(p, preview.render(src.read(), inside),
                                         f"{r.date}  RGB | NDVI | mask (из файла)")
                    fresh = True
            else:
                fresh = True
            pngs.append(p)
        elif r.s2c_only_reject:
            # отсеяна только s2cloudless — превью для ручной проверки ложных срабатываний
            try:
                refl = read_scene(l2a_items[r.l2a_id], grid, catalog.boa_offset(r.l2a_id))
                preview.save_png(prev_dir / "rejected" / f"{path.stem}.png",
                                 preview.render(np.where(inside[None], refl, np.nan), inside, r.masks),
                                 f"{r.date} REJECTED by s2cloudless: SCL {r.verdict.scl_percent:.1f}% "
                                 f"union {r.verdict.union_percent:.1f}%")
            except Exception as exc:  # noqa: BLE001
                print(f"{r.date}: превью отброшенной сцены не сделано: {exc}")
    if pngs and (fresh or not (prev_dir / "all_dates.png").exists()):
        preview.contact_sheet(pngs, prev_dir / "all_dates.png")

    stale = sorted({p.name for p in opt.out.glob("scene_*.tif")}
                   - {scene_filename(r.date) for r in results if r.selected}
                   - {scene_filename(d) for d in _dates_outside(opt, opt.out)})
    if stale:
        print("ВНИМАНИЕ: в папке есть сцены, не выбранные этим запуском:", ", ".join(stale))

    run_info = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "boundary": str(opt.boundary), "interval": [opt.start, opt.end], "tile": opt.tile,
        "max_cloud_percent": opt.max_cloud, "weekly": opt.weekly, "dilate_px": opt.dilate_px,
        "manual_masks": str(opt.manual_masks) if manual_polys else None,
        "rule": "отброс, если SCL 0/1 > max (нет покрытия) или SCL{3,8,9,10} ∪ s2cloudless ∪ manual > max "
                "(без расширения); NaN = расширенная маска облаков ∪ SCL 0/1 внутри поля, 0 = вне поля",
        "grid": {"crs": grid.crs, "transform": list(grid.transform)[:6],
                 "width": grid.width, "height": grid.height, "field_pixels": int(inside.sum())},
        "bands": BANDS,
    }
    mp = manifest.save(opt.out, run_info, {r.date: r.entry(opt.out) for r in results},
                       manifest.load(opt.out))
    errors = [r for r in results if r.status == "error"]
    print(f"\nМанифест: {mp}; записано/есть: "
          f"{sum(r.status in ('written', 'exists') for r in results)}, ошибок: {len(errors)}")
    return (1 if errors else 0), results


def _dates_outside(opt: Options, out_dir: Path) -> list[str]:
    """Даты из прошлых запусков вне текущего интервала (их файлы не считаются лишними)."""
    prev = manifest.load(out_dir).get("scenes", [])
    return [e["date"] for e in prev if not (opt.start <= e["date"] <= opt.end) and e.get("selected")]
