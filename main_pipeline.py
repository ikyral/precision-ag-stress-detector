"""Сквозной конвейер КТ3: спутник (М1) -> индексы и зоны (М2) -> предписания VRA (М3) -> слои для кабинета (М4).

Запуск из корня репозитория (Linux/macOS — .venv/bin/python, Windows — .venv\\Scripts\\python.exe):

    python main_pipeline.py --start 2026-06-29 --end 2026-08-30        # скачать новые сцены и обработать
    python main_pipeline.py --skip-fetch                                # без скачивания: то, что есть в data/scenes

Каждая новая дата проходит М1 -> М2 -> М3, затем перерисовываются слои (PNG) всех дат. Повторный запуск без
новых снимков меняет только pipeline_report.json. Дата, упавшая на одном этапе, дальше не идёт; остальные
даты обрабатываются. Код выхода: 0 — ок, 1 — ошибки обработки, 2 — данные есть, но проверки не прошли.

Состояние <data>/pipeline_state.json: sha256 сцены, по которой М2 считал дату, и маски зон, по которой М3
строил предписание. Если сцену подменили (например, s2fetch --force), М2 пересчитывает всю серию, а М3 —
изменившиеся даты: process_new_scenes М2 сам такие даты не видит (zones и stats уже есть).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"
REF_SCENE = SAMPLES / "sample_scene.tif"
REF_ZONES = SAMPLES / "sample_zones.tif"
BOUNDARY = SAMPLES / "field_boundary.geojson"
for _p in ("src/module1", "src/module2", "src/module3"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

STATE_NAME = "pipeline_state.json"
REPORT_NAME = "pipeline_report.json"
SCHEMA_VERSION = "1.0"
COVER_TOLERANCE_M = 10.0  # рамка предписаний строится по пикселям 10 м, контур может выступать до пикселя
FETCH_KEYS_HINT = "Без ключей CDSE используйте --skip-fetch."
FETCH_DEPS_HINT = ("Для скачивания нужны зависимости М1: pip install -r src/module1/requirements.txt. "
                   + FETCH_KEYS_HINT)


# ----------------------------------------------------------------------------
# Мелочи
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp(date: str) -> str:
    return date.replace("-", "_")


def _iso(stamp: str) -> str:
    return stamp.replace("_", "-")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _date_arg(s: str) -> str:
    return dt.date.fromisoformat(s).isoformat()


def _rates_arg(text: str) -> dict[int, float]:
    try:
        pairs = [part.split("=") for part in text.split(",")]
        rates = {int(k): float(v) for k, v in pairs}
        if len(rates) != len(pairs) or set(rates) != {1, 2, 3}:
            raise ValueError
        return rates
    except ValueError as exc:
        raise argparse.ArgumentTypeError("нужно 1=150,2=80,3=0 (каждый класс один раз)") from exc


class Report:
    """pipeline_report.json: этапы, даты, ошибки, предупреждения, проверки."""

    def __init__(self, args, data: Path):
        self.t0 = time.monotonic()
        self.doc = {
            "schema_version": SCHEMA_VERSION,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "data_dir": str(data),
            "args": {k: v for k, v in vars(args).items() if k != "data_dir"},
            "stages": [],
            "dates": {},
            "errors": {},
            "warnings": [],
            "checks": {},
        }

    def stage(self, name: str, status: str, t_start: float, **details) -> dict:
        st = {"name": name, "status": status, "seconds": round(time.monotonic() - t_start, 2), **details}
        self.doc["stages"].append(st)
        return st

    def error(self, date: str, stage: str, message: str) -> None:
        """Ошибка даты (ключ — дата) или всего этапа (date='*': ключ '*<этап>')."""
        self.doc["errors"][f"*{stage}" if date == "*" else date] = {"stage": stage, "error": message}

    def warn(self, message: str) -> None:
        self.doc["warnings"].append(message)


# ----------------------------------------------------------------------------
# Этапы
# ----------------------------------------------------------------------------
def stage_fetch(args, data: Path, rep: Report) -> list[str]:
    """М1: s2fetch по интервалу. Возвращает даты, записанные (или перезаписанные) в этом запуске."""
    t = time.monotonic()
    if args.skip_fetch:
        rep.stage("M1_fetch", "skipped", t, reason="--skip-fetch")
        return []
    try:
        # зависимости и ключи М1 — до сети: s2fetch грузит pystac_client лениво, а ключи читает после поиска
        import pystac_client  # noqa: F401
        from s2fetch.manual import DEFAULT_PATH as manual_masks
        from s2fetch.pipeline import Options, fetch_series
        from s2fetch.reader import configure_s3

        configure_s3()
        opt = Options(BOUNDARY, args.start, args.end, data / "scenes", max_cloud=args.max_cloud,
                      weekly=args.weekly, force=args.force, reference=REF_SCENE, manual_masks=manual_masks)
        code, results = fetch_series(opt)
    except (ImportError, SystemExit) as exc:  # без трейсбека: одна строка с подсказкой
        if isinstance(exc, ImportError):
            msg = f"{f'нет модуля {exc.name}' if exc.name else exc}. {FETCH_DEPS_HINT}"
        else:
            msg = f"{exc}. {FETCH_KEYS_HINT}"
        rep.stage("M1_fetch", "error", t, error=msg)
        rep.error("*", "M1_fetch", msg)
        return []
    except Exception as exc:  # noqa: BLE001 — нет сети: работаем с тем, что уже скачано
        traceback.print_exc()
        rep.stage("M1_fetch", "error", t, error=f"{type(exc).__name__}: {exc}")
        rep.error("*", "M1_fetch", f"{type(exc).__name__}: {exc}")
        return []
    written = sorted(r.date for r in results if r.selected and r.status == "written")
    exists = sorted(r.date for r in results if r.selected and r.status == "exists")
    failed = {r.date: r.error for r in results if r.status == "error"}
    for d, e in failed.items():
        rep.error(d, "M1_fetch", e or "ошибка")
    if code == 2:
        rep.error("*", "M1_fetch", "сетка или маска поля не совпали с эталоном sample_scene.tif")
    status = "ok" if code == 0 else "error"
    rep.stage("M1_fetch", status, t, interval=[args.start, args.end], weekly=args.weekly, exit_code=code,
              selected=sorted(r.date for r in results if r.selected), written=written, existing=exists,
              errors=sorted(failed))
    return written


def stage_m2(args, data: Path, state: dict, fetched: list[str], rep: Report) -> tuple[list[str], set[str]]:
    """М2: новые даты через process_new_scenes, при подмене сцены или дате в прошлом — вся серия.

    Возвращает (даты, обработанные в этом запуске, даты с ошибкой М2).
    """
    from stress_core.pipeline import collect_scenes, process_new_scenes, processed_dates, run_pipeline

    t = time.monotonic()
    scenes_dir = data / "scenes"
    if not scenes_dir.is_dir():
        rep.stage("M2", "error", t, error=f"нет папки {scenes_dir}")
        rep.error("*", "M2", f"нет папки {scenes_dir}")
        return [], set()
    found, warns = collect_scenes([scenes_dir])
    for w in warns:
        rep.warn(f"M2: {w}")
    hashes = {d: _sha256(p) for p, d in found}
    done = set(processed_dates(data))
    sd = state["dates"]
    trusted = [] if args.rebuild else sorted(d for d in done if d in hashes and "scene_sha256" not in sd.get(d, {}))
    if trusted:
        rep.warn(f"M2: для дат {', '.join(trusted)} нет записи в {STATE_NAME}; готовые результаты М2 приняты "
                 "как есть, хэши сцен сохранены")
        for d in trusted:
            sd.setdefault(d, {})["scene_sha256"] = hashes[d]
    changed = sorted(d for d in done if d in hashes and sd[d]["scene_sha256"] != hashes[d])
    new = sorted(d for d in hashes if d not in done)
    late = sorted(d for d in new if done and d < max(done))
    reason = ("--rebuild" if args.rebuild else
              f"сцены изменились после обработки: {', '.join(changed)}" if changed else
              f"новые даты раньше уже обработанных: {', '.join(late)}" if late else None)
    if reason is None and new:
        res = process_new_scenes(scenes_dir, data)
        processed, errors = res["processed"], res["errors"]
        if any("раньше уже обработанной" in w for w in res["warnings"]):  # страховка: М2 сам заметил
            reason = "предупреждение М2 о дате раньше обработанной"
        else:
            for w in res["warnings"]:
                rep.warn(f"M2: {w}")
    elif reason is None:
        processed, errors = [], {}
    if reason is not None:
        summary = run_pipeline([scenes_dir], data)
        processed = sorted(d for d, r in summary["dates"].items() if d in hashes and "error" not in r)
        errors = {d: r["error"] for d, r in summary["dates"].items() if d in hashes and "error" in r}
    for d in processed:
        sd.setdefault(d, {})["scene_sha256"] = hashes[d]
    for d, e in errors.items():
        sd.setdefault(d, {})["scene_sha256"] = None  # следующий запуск попробует снова
        rep.error(d, "M2", e)
    skipped = sorted(d for d in hashes if d not in processed and d not in errors)
    rep.stage("M2", "error" if errors else ("ok" if processed else "unchanged"), t,
              mode="run_pipeline (вся серия)" if reason else "process_new_scenes",
              full_rerun_reason=reason, processed=sorted(processed), skipped=skipped, errors=sorted(errors),
              changed_scenes=changed, fetched_this_run=fetched)
    return sorted(processed), set(errors)


def stage_m3(args, data: Path, state: dict, m2_processed: list[str], m2_failed: set[str],
             rep: Report) -> tuple[list[str], set[str]]:
    """М3: generate_prescription для дат М2 этого запуска и дат, где маска зон изменилась или нет metrics."""
    from vra import generate_prescription

    t = time.monotonic()
    zones_dir, out = data / "zones", data / "prescriptions"
    opts = {"base_rate_kg_ha": args.base_rate, "rates_by_zone": args.rates, "price_per_kg": args.price,
            "min_polygon_area_ha": args.min_area}
    opts_key = {**opts, "rates_by_zone": {str(k): v for k, v in sorted((args.rates or {}).items())} or None,
                "reference_mask": REF_ZONES.name}
    sd = state["dates"]
    zones = {_iso(p.stem[len("zones_"):]): p for p in sorted(zones_dir.glob("zones_*.tif"))}
    hashes = {d: _sha256(p) for d, p in zones.items()}
    opts_changed = state.get("m3_options") not in (None, opts_key)
    if state.get("m3_options") is None:
        state["m3_options"] = opts_key
    todo = set()
    for d in zones:
        if d in m2_failed:
            continue
        metrics = out / f"metrics_{_stamp(d)}.json"
        rec = sd.setdefault(d, {})
        if "zones_sha256" not in rec and metrics.exists() and not args.rebuild and d not in m2_processed:
            rec["zones_sha256"] = hashes[d]  # результат прошлых запусков без состояния — принимаем
            rep.warn(f"M3: {d}: нет записи в {STATE_NAME}, предписание принято как есть")
        if (args.rebuild or opts_changed or d in m2_processed or not metrics.exists()
                or rec.get("zones_sha256") != hashes[d]):
            todo.add(d)
    done, failed = [], {}
    for d in sorted(todo):
        try:
            generate_prescription(str(zones[d]), str(out), _stamp(d), reference_mask_path=str(REF_ZONES), **opts)
            sd[d]["zones_sha256"] = hashes[d]
            done.append(d)
        except Exception as exc:  # noqa: BLE001 — одна дата не обрывает остальные
            failed[d] = f"{type(exc).__name__}: {exc}"
            sd[d]["zones_sha256"] = None
            rep.error(d, "M3", failed[d])
    state["m3_options"] = opts_key
    summary_path = out / "season_summary.json"
    summary_written = False
    if zones and (todo or m2_failed or not summary_path.exists()):
        upstream = {d: rep.doc["errors"][d]["error"] for d in m2_failed if d in zones}
        _write_json_exact(summary_path, season_summary(zones_dir, out, failed, upstream))
        summary_written = True
    rep.stage("M3", "error" if failed else ("ok" if done else "unchanged"), t, processed=done,
              errors=sorted(failed), options=opts_key, season_summary_written=summary_written)
    return done, set(failed)


def season_summary(zones_dir: Path, out: Path, failed: dict[str, str], upstream: dict[str, str],
                   area_tolerance_ha: float = 0.01) -> dict:
    """season_summary.json из готовых metrics_*.json, без повторной векторизации.

    Схема и порядок полей — как у vra.batch.run_series (сверяется тестом). failed — ошибки М3 этого запуска,
    upstream — даты, упавшие раньше (М2): они не попадают в records, а идут в errors.
    """
    from vra.batch import date_from_path

    files = sorted(p for p in zones_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
                   and p.stem.startswith("zones_"))
    rows, errors = [], []
    for date_str, path in sorted((date_from_path(p), p) for p in files):
        iso = _iso(date_str)
        err = failed.get(iso) or (f"M2: {upstream[iso]}" if iso in upstream else None)
        metrics_path = out / f"metrics_{date_str}.json"
        if err is None and not metrics_path.exists():
            err = f"FileNotFoundError: нет {metrics_path.name}"
        if err is not None:
            errors.append({"date": date_str, "source_file": path.name, "error": err})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "date": date_str,
            "source_file": path.name,
            "total_field_area_ha": metrics["total_field_area_ha"],
            "stressed_area_ha": metrics["stressed_area_ha"],
            "saved_percent": metrics["saved_percent"],
            "saved_money_rub": metrics["saved_money_rub"],
            "coverage": metrics.get("coverage"),
            "files": {"geojson": metrics["files"]["geojson"],
                      "shapefile_zip": metrics["files"]["shapefile_zip"],
                      "metrics": metrics["files"]["metrics"]},
        })
    areas = [r["total_field_area_ha"] for r in rows]
    spread = max(areas) - min(areas) if areas else None
    stable = spread <= area_tolerance_ha + 1e-10 if len(areas) > 1 else None
    return {
        "schema_version": "1.0",
        "dates_found": len(files), "dates_processed": len(rows),
        "area_tolerance_ha": area_tolerance_ha,
        "area_range_ha": [min(areas), max(areas)] if areas else None,
        "area_spread_ha": spread, "area_stable": stable,
        "coverage_note": "Area includes classified pixels, including upstream-filled classes. Remaining input nodata is excluded. See M2 stats for observed/filled area. One input directory must contain one field.",
        "warnings": (["Classified field area changes between dates; inspect no-data masks and alignment."]
                     if stable is False else []),
        "records": rows, "errors": errors,
        "incomplete_dates": [r["date"] for r in rows if r["coverage"] and not r["coverage"]["is_complete"]],
    }


def _write_json_exact(path: Path, doc: dict) -> None:
    """Как пишет run_series: indent=2, без перевода строки в конце."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stage_layers(args, data: Path, failed: set[str], rep: Report) -> dict | None:
    from s2fetch.overlays import build_overlays

    t = time.monotonic()
    stats = sorted(_iso(p.stem[len("index_stats_"):]) for p in (data / "stats").glob("index_stats_*.json"))
    dates = [d for d in stats if d not in failed]
    try:
        res = build_overlays(data, dates=dates, nodata_mode=args.nodata_mode, force=args.rebuild)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        rep.stage("layers", "error", t, error=f"{type(exc).__name__}: {exc}")
        rep.error("*", "layers", f"{type(exc).__name__}: {exc}")
        return None
    for d, e in res["errors"].items():
        rep.error(_iso(d), "layers", e)
    for w in res["warnings"]:
        rep.warn(f"layers: {w}")
    rep.stage("layers", "error" if res["errors"] else ("ok" if res["status"] == "written" else "unchanged"), t,
              status_detail=res["status"], dates=res["dates"], png_count=res["png_count"], size=res["size"],
              bounds=res["bounds"])
    return res


# ----------------------------------------------------------------------------
# Проверки
# ----------------------------------------------------------------------------
def _bbox_lonlat(gj: dict) -> tuple[float, float, float, float]:
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for x in c:
                walk(x)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    for f in feats:
        walk((f.get("geometry") or f)["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def _uncovered_m(outer, inner) -> float:
    """На сколько метров inner (W, S, E, N) выступает за outer; 0 — покрывает."""
    lat = math.radians((inner[1] + inner[3]) / 2)
    mx, my = 111_320.0 * math.cos(lat), 110_574.0
    return max(0.0, (outer[0] - inner[0]) * mx, (outer[1] - inner[1]) * my,
               (inner[2] - outer[2]) * mx, (inner[3] - outer[3]) * my)


def run_checks(data: Path, layers: dict | None, rep: Report) -> bool:
    from s2fetch.grid import compare_grids, grid_of_file
    from stress_core.contract import check_outputs

    t = time.monotonic()
    checks = rep.doc["checks"]
    m2 = check_outputs(data)
    checks["m2_contract"] = {"ok": m2["ok"], "dates": m2["dates"], "area_ha": m2["area_ha"],
                             "area_spread_ha": m2["area_spread_ha"], "errors": m2["errors"]}

    summary = _load_json(data / "prescriptions" / "season_summary.json")
    if summary is None:
        checks["m3_season"] = {"ok": False, "errors": ["нет prescriptions/season_summary.json"]}
    else:
        ok = summary["area_stable"] is not False and not summary["incomplete_dates"]
        checks["m3_season"] = {"ok": ok, "area_stable": summary["area_stable"],
                               "area_range_ha": summary["area_range_ha"],
                               "dates_processed": summary["dates_processed"], "dates_found": summary["dates_found"],
                               "incomplete_dates": summary["incomplete_dates"]}

    ref = grid_of_file(REF_SCENE)
    mism, n = [], 0
    for pattern in ("scenes/scene_*.tif", "zones/zones_*.tif", "indices/*_*.tif", "hidden_stress/*.tif"):
        for p in sorted(data.glob(pattern)):
            n += 1
            try:
                diffs = compare_grids(grid_of_file(p), ref)
            except Exception as exc:  # noqa: BLE001 — битый файл: в расхождения, не падаем
                diffs = [f"не читается ({type(exc).__name__})"]
            if diffs:
                mism.append(f"{p.relative_to(data).as_posix()}: {'; '.join(diffs)}")
    checks["grids"] = {"ok": not mism and n > 0, "files_checked": n, "reference": REF_SCENE.name,
                       "crs": ref.crs, "size": [ref.width, ref.height], "mismatches": mism}

    field = _bbox_lonlat(json.loads(BOUNDARY.read_text(encoding="utf-8")))
    cover = {"field_bbox_lonlat": list(field), "tolerance_m": COVER_TOLERANCE_M, "prescriptions": {}}
    errs = []
    for p in sorted((data / "prescriptions").glob("prescription_*.geojson")):
        gap = _uncovered_m(_bbox_lonlat(json.loads(p.read_text(encoding="utf-8"))), field)
        cover["prescriptions"][p.stem[len("prescription_"):]] = round(gap, 2)
        if gap > COVER_TOLERANCE_M:
            errs.append(f"{p.name}: контур поля выступает за рамку предписания на {gap:.1f} м")
    meta = _load_json(data / "overlays" / "meta.json") or {}
    if meta.get("bounds"):
        (s, w), (nn, e) = meta["bounds"]
        gap = _uncovered_m((w, s, e, nn), field)
        cover["overlay_bounds_gap_m"] = round(gap, 3)
        if gap > 0:
            errs.append(f"meta.bounds не покрывают контур поля (на {gap:.2f} м)")
        sizes = set()
        for lay in (meta.get("layers") or {}).values():
            for f in lay.values():
                from PIL import Image
                with Image.open(data / "overlays" / f) as im:
                    sizes.add(im.size)
        cover["png_sizes"] = sorted(list(s) for s in sizes)
        if len(sizes) > 1:
            errs.append(f"PNG разного размера: {sorted(sizes)}")
    else:
        errs.append("нет overlays/meta.json с bounds")
    cover.update(ok=not errs, errors=errs)
    checks["coverage"] = cover

    ok = all(c["ok"] for c in checks.values())
    rep.stage("checks", "ok" if ok else "failed", t,
              failed=[k for k, c in checks.items() if not c["ok"]])
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python main_pipeline.py",
                                 description="КТ3: М1 (Sentinel-2) -> М2 (индексы, зоны) -> М3 (VRA) -> слои М4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="папка данных (по умолчанию data/)")
    ap.add_argument("--skip-fetch", action="store_true", help="не скачивать: работать со сценами в <data>/scenes")
    ap.add_argument("--start", type=_date_arg, help="М1: начало интервала, YYYY-MM-DD")
    ap.add_argument("--end", type=_date_arg, help="М1: конец интервала (включительно), YYYY-MM-DD")
    ap.add_argument("--no-weekly", dest="weekly", action="store_false",
                    help="М1: брать все чистые даты (по умолчанию — одна самая чистая сцена на ISO-неделю)")
    ap.add_argument("--force", action="store_true", help="М1: перекачать уже готовые даты")
    ap.add_argument("--max-cloud", type=float, default=10.0, help="М1: порог облаков в поле, %% (10)")
    ap.add_argument("--rebuild", action="store_true",
                    help="М2 по всей серии, М3 и слои по всем датам заново, без учёта pipeline_state.json")
    ap.add_argument("--base-rate", type=float, default=150.0, help="М3: сплошная норма для сравнения, кг/га (150)")
    ap.add_argument("--rates", type=_rates_arg, default=None, help="М3: нормы зон, по умолчанию 1=150,2=80,3=0")
    ap.add_argument("--price", type=float, default=45.0, help="М3: цена, руб/кг (45)")
    ap.add_argument("--min-area", type=float, default=0.05, help="М3: порог слияния мелких полигонов, га (0.05)")
    ap.add_argument("--nodata-mode", choices=("gray", "transparent"), default="gray",
                    help="слои: облако внутри поля — серый (по умолчанию) или прозрачно")
    a = ap.parse_args(argv)
    if not a.skip_fetch and not (a.start and a.end):
        ap.error("нужны --start и --end (или --skip-fetch)")
    if a.start and a.end and a.start > a.end:
        ap.error("--start позже --end")
    return a


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    data = args.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    rep = Report(args, data)
    if data != (ROOT / "data").resolve():  # М4 ищет контур в корне папки данных
        dst = data / "field_boundary.geojson"
        if not dst.exists() or dst.read_bytes() != BOUNDARY.read_bytes():
            shutil.copyfile(BOUNDARY, dst)

    state_path = data / STATE_NAME
    state = _load_json(state_path) or {}
    if not state:
        state = {"schema_version": SCHEMA_VERSION, "m3_options": None, "dates": {}}
    if args.rebuild:
        state["dates"] = {}
    before = json.dumps(state, sort_keys=True)

    fetched = stage_fetch(args, data, rep)
    m2_done, m2_failed = stage_m2(args, data, state, fetched, rep)
    m3_done, m3_failed = stage_m3(args, data, state, m2_done, m2_failed, rep)
    layers = stage_layers(args, data, m2_failed | m3_failed, rep)
    checks_ok = run_checks(data, layers, rep)

    if json.dumps(state, sort_keys=True) != before:
        state["dates"] = dict(sorted(state["dates"].items()))
        _write_json(state_path, state)

    d = rep.doc
    d["dates"] = {"fetched": fetched, "m2_processed": m2_done, "m3_processed": m3_done,
                  "failed": sorted(k for k in d["errors"] if not k.startswith("*")),
                  "layers": (layers or {}).get("dates", [])}
    code = 1 if d["errors"] else (0 if checks_ok else 2)
    d["exit_code"] = code
    d["seconds"] = round(time.monotonic() - rep.t0, 2)
    d["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _write_json(data / REPORT_NAME, d)

    print(f"\nmain_pipeline: {data}")
    for st in d["stages"]:  # текст ошибки этапа — ниже, в строке ОШИБКА
        extra = {k: st[k] for k in ("processed", "written", "errors", "full_rerun_reason", "png_count", "failed")
                 if st.get(k)}
        print(f"  {st['name']:9} {st['status']:9} {st['seconds']:7.2f} c  {extra if extra else ''}")
    for w in d["warnings"]:
        print("  предупреждение:", w)
    for k, e in d["errors"].items():
        print(f"  ОШИБКА {k} [{e['stage']}]: {e['error']}")
    print(f"  проверки: {'ok' if checks_ok else 'НЕ ПРОЙДЕНЫ'}; отчёт {data / REPORT_NAME}; "
          f"{d['seconds']:.1f} c; код выхода {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
