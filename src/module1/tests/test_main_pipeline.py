"""Тесты main_pipeline.py без сети. Запуск из src/module1: python -m unittest discover -s tests -v

Быстрые: обёртка свода сезона М3 пишет те же байты, что vra.batch.run_series; аргументы CLI; формулировки
предупреждений М2, которые main_pipeline разбирает регэкспами (тест падает, если М2 их сменит).
Медленные (PAS_SLOW_TESTS=1): сцены data/scenes подаются по одной -> итог как у прогона всей серии;
повторный запуск ничего не меняет; подмена сцены -> reprocess даты и более поздних, итог как у полного прогона;
touch сцен без ложных предупреждений; дата раньше обработанных через reprocess; упавшая переобработка;
--rebuild; битая сцена не обрывает остальные; кабинет М4 через AppTest (все даты × слои, в том числе без
indices/*.tif).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, HERE.parent, ROOT / "src" / "module2", ROOT / "src" / "module3"):
    sys.path.insert(0, str(p))

import main_pipeline as mp  # noqa: E402

SLOW = os.environ.get("PAS_SLOW_TESTS") == "1"
SCENES = sorted((ROOT / "data" / "scenes").glob("scene_*.tif"))


def _has(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) for m in mods)


def run_pipeline(data: Path, *extra) -> tuple[int, dict]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = mp.main(["--skip-fetch", "--data-dir", str(data), *extra])
    return code, json.loads((data / mp.REPORT_NAME).read_text(encoding="utf-8"))


def snapshot(data: Path) -> dict[str, tuple]:
    """Все файлы папки (кроме отчёта): размер, mtime, sha256."""
    out = {}
    for p in sorted(data.rglob("*")):
        if p.is_file() and p.name != mp.REPORT_NAME:
            out[p.relative_to(data).as_posix()] = (p.stat().st_size, p.stat().st_mtime_ns,
                                                   hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def content_diff(a: Path, b: Path, only: str | None = None) -> list[str]:
    """Расхождения содержимого результатов двух папок (без меток времени и абсолютных путей)."""
    import rasterio
    from PIL import Image

    diffs = []
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if only:
        files, files_b = [f for f in files if only in f], [f for f in files_b if only in f]
    if files != files_b:
        diffs.append(f"набор файлов: {sorted(set(files) ^ set(files_b))}")
    for rel in files:
        if rel in (mp.REPORT_NAME, mp.STATE_NAME) or (only and only not in rel) or rel not in files_b:
            continue
        pa, pb = a / rel, b / rel
        if rel.endswith(".tif"):
            with rasterio.open(pa) as x, rasterio.open(pb) as y:
                prof = [{k: repr(v) for k, v in z.profile.items()} for z in (x, y)]  # nodata=nan != nan
                same = prof[0] == prof[1] and np.array_equal(x.read(), y.read(), equal_nan=True)
        elif rel.endswith(".png"):
            same = np.array_equal(np.array(Image.open(pa)), np.array(Image.open(pb)))
        elif rel.endswith(".zip"):
            with zipfile.ZipFile(pa) as x, zipfile.ZipFile(pb) as y:
                same = sorted(x.namelist()) == sorted(y.namelist()) and all(
                    x.read(n) == y.read(n) for n in x.namelist())
        elif rel.endswith((".json", ".geojson")):
            da, db = _json(pa), _json(pb)
            for d in (da, db):
                for k in ("generated_at", "input_raster"):
                    d.pop(k, None)
            same = da == db
        else:
            same = pa.read_bytes() == pb.read_bytes()
        if not same:
            diffs.append(rel)
    return diffs


class TestCli(unittest.TestCase):
    def test_weekly_default_and_required_dates(self):
        a = mp.parse_args(["--start", "2026-06-29", "--end", "2026-08-30"])
        self.assertTrue(a.weekly)
        self.assertFalse(mp.parse_args(["--start", "2026-06-29", "--end", "2026-08-30", "--no-weekly"]).weekly)
        self.assertTrue(mp.parse_args(["--skip-fetch"]).skip_fetch)
        for bad in ([], ["--start", "2026-08-30", "--end", "2026-06-29"], ["--skip-fetch", "--rates", "1=1,2=2"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mp.parse_args(bad)
        self.assertEqual(mp.parse_args(["--skip-fetch", "--rates", "1=160,2=90,3=0"]).rates,
                         {1: 160.0, 2: 90.0, 3: 0.0})

    def test_skip_fetch_does_not_import_fetch_deps(self):
        """--skip-fetch не тянет pystac-client, s2cloudless и dotenv (коллеги без ключей CDSE)."""
        code = ("import sys, main_pipeline as mp; a = mp.parse_args(['--skip-fetch']); "
                "r = mp.Report(a, mp.ROOT); mp.stage_fetch(a, mp.ROOT, r); "
                "print(sorted(m for m in ('pystac_client', 's2cloudless', 'dotenv', 's2fetch.pipeline') "
                "if m in sys.modules))")
        import subprocess
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


class TestFetchUnavailable(unittest.TestCase):
    """Без пакетов М1 или ключей CDSE: одна строка-подсказка без трейсбека, ошибка в отчёте, код 1. Без сети."""

    def _run(self, modules: dict, env: dict) -> tuple[int, dict, str]:
        from unittest import mock

        import s2fetch.pipeline
        import s2fetch.reader

        tmp = Path(tempfile.mkdtemp(prefix="pas_fetch_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.dict(sys.modules, modules))
            st.enter_context(mock.patch.dict(os.environ, env))
            st.enter_context(mock.patch.object(s2fetch.reader, "ROOT", tmp))  # .env ищется в пустой папке
            st.enter_context(mock.patch.object(s2fetch.pipeline, "fetch_series",
                                               side_effect=AssertionError("fetch_series вызван: сеть")))
            # остальные этапы не нужны: код выхода и ошибки — только от М1
            st.enter_context(mock.patch.object(mp, "stage_m2", return_value=([], set())))
            st.enter_context(mock.patch.object(mp, "stage_m3", return_value=([], set())))
            st.enter_context(mock.patch.object(mp, "stage_layers", return_value=None))
            st.enter_context(mock.patch.object(mp, "run_checks", return_value=True))
            st.enter_context(contextlib.redirect_stdout(out))
            st.enter_context(contextlib.redirect_stderr(out))
            code = mp.main(["--start", "2026-06-29", "--end", "2026-08-30", "--data-dir", str(tmp)])
        return code, _json(tmp / mp.REPORT_NAME), out.getvalue()

    def _check(self, code: int, rep: dict, out: str, expected: str) -> None:
        self.assertEqual(code, 1)
        self.assertEqual(rep["exit_code"], 1)
        self.assertEqual(rep["errors"], {"*M1_fetch": {"stage": "M1_fetch", "error": expected}})
        self.assertEqual(rep["stages"][0]["status"], "error")
        self.assertNotIn("Traceback", out)
        self.assertEqual(out.count(expected), 1, out)
        self.assertNotIn("\n", expected)

    def test_missing_m1_packages(self):
        code, rep, out = self._run({"pystac_client": None}, {})
        self._check(code, rep, out, f"нет модуля pystac_client. {mp.FETCH_DEPS_HINT}")
        self.assertIn("pip install -r src/module1/requirements.txt", out)

    @unittest.skipUnless(_has("pystac_client", "dotenv"), "нужны зависимости М1")
    def test_empty_cdse_keys(self):
        # настоящий .env не читается; пустые значения в окружении load_dotenv не перезаписывает
        code, rep, out = self._run({}, {"CDSE_S3_ACCESS_KEY": "", "CDSE_S3_SECRET_KEY": ""})
        self._check(code, rep, out, "В .env нет ключей: CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY "
                                    f"(см. .env.example). {mp.FETCH_KEYS_HINT}")


def _stamp_of(scene: Path) -> str:
    return scene.stem[len("scene_"):]


def _date_of(scene: Path) -> str:
    return _stamp_of(scene).replace("_", "-")


@unittest.skipUnless(_has("stress_core", "rasterio") and len(SCENES) >= 3, "нужны М2 и data/scenes")
class TestM2WarningFormats(unittest.TestCase):
    """Настоящие предупреждения process_new_scenes М2 распознаются регэкспами main_pipeline.

    Если М2 сменит формулировку, тест упадёт: ложные предупреждения снова попадут в warnings отчёта.
    """

    def test_scene_newer_and_earlier_date_are_recognized(self):
        from stress_core.pipeline import process_new_scenes

        first, mid, last = SCENES[0], SCENES[1], SCENES[2]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "scenes").mkdir()
            for s in (first, last):
                shutil.copy2(s, d / "scenes" / s.name)
            with contextlib.redirect_stderr(io.StringIO()):
                process_new_scenes(d / "scenes", d)
                future = (d / "stats").stat().st_mtime + 3600
                os.utime(d / "scenes" / first.name, (future, future))     # содержимое то же, mtime новее
                shutil.copy2(mid, d / "scenes" / mid.name)                   # дата раньше уже обработанной
                res = process_new_scenes(d / "scenes", d, reprocess=[_date_of(mid)])
            hashes = {_date_of(s): mp._sha256(d / "scenes" / s.name) for s in (first, mid, last)}
            saved = {_date_of(first): {"scene_sha256": hashes[_date_of(first)]}}
            newer = [w for w in res["warnings"] if mp.M2_SCENE_NEWER_RE.match(w)]
            earlier = [w for w in res["warnings"] if mp.M2_EARLIER_RE.match(w)]
            self.assertEqual(len(newer), 1, res["warnings"])
            self.assertTrue(newer[0].startswith(first.name))
            self.assertEqual(len(earlier), 1, res["warnings"])
            self.assertEqual(mp.M2_EARLIER_RE.match(earlier[0]).group(1), _date_of(mid))
            self.assertEqual(res["processed"], [_date_of(mid), _date_of(last)])
            for w in (*newer, *earlier):
                self.assertIsNotNone(mp.m2_warning_note(w, hashes, saved, [_date_of(mid)], res["processed"]), w)
            # подменённая сцена (хэш не совпал) и дата, которую мы в reprocess не передавали, — не информационные
            self.assertIsNone(mp.m2_warning_note(newer[0], hashes, {_date_of(first): {"scene_sha256": "x"}}, [], []))
            self.assertIsNone(mp.m2_warning_note(earlier[0], hashes, saved, [], res["processed"]))


@unittest.skipUnless(_has("geopandas", "vra"), "нужны зависимости М3 (geopandas)")
class TestSeasonSummary(unittest.TestCase):
    def test_wrapper_writes_run_series_bytes(self):
        """Контракт свода — у М3 (src/module3/tests/test_summary.py); здесь только то, что делает М1:
        обёртка отдаёт свод М3 и _write_json_exact пишет его теми же байтами, что run_series."""
        from vra import generate_prescription
        from vra.batch import run_series

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zones = d / "zones"
            zones.mkdir()
            for stamp in ("2026_08_07", "2026_08_14", "2026_08_17"):
                shutil.copy(mp.REF_ZONES, zones / f"zones_{stamp}.tif")
            for stamp in ("2026_08_07", "2026_08_14"):
                generate_prescription(str(zones / f"zones_{stamp}.tif"), str(d / "a"), stamp,
                                      reference_mask_path=str(mp.REF_ZONES))
            ref = run_series(zones, d / "b", reference_mask_path=mp.REF_ZONES)
            got = mp.season_summary(zones, d / "b", {}, {})
            self.assertEqual(json.dumps(got), json.dumps(ref))              # и порядок ключей
            mp._write_json_exact(d / "x.json", got)
            self.assertEqual((d / "x.json").read_bytes(), (d / "b" / "season_summary.json").read_bytes())


@unittest.skipUnless(SLOW, "медленный тест: PAS_SLOW_TESTS=1")
@unittest.skipUnless(_has("geopandas", "stress_core", "vra") and len(SCENES) >= 3, "нужны М2, М3 и data/scenes")
class TestPipelineIncremental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="kt3-")
        base = Path(cls.tmp.name)
        cls.one, cls.all = base / "one_by_one", base / "all_at_once"
        cls.codes = []
        (cls.one / "scenes").mkdir(parents=True)
        for s in SCENES:                                     # по одной сцене
            shutil.copy2(s, cls.one / "scenes" / s.name)
            cls.codes.append(run_pipeline(cls.one)[0])
        (cls.all / "scenes").mkdir(parents=True)
        for s in SCENES:
            shutil.copy2(s, cls.all / "scenes" / s.name)
        cls.all_code, cls.all_report = run_pipeline(cls.all)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_one_by_one_equals_whole_series(self):
        self.assertEqual(self.codes, [0] * len(SCENES))
        self.assertEqual(self.all_code, 0)
        self.assertEqual(content_diff(self.one, self.all), [])
        meta = _json(self.one / "overlays" / "meta.json")
        self.assertEqual(len(meta["dates"]), len(SCENES))
        self.assertEqual(len(list((self.one / "overlays").glob("*_20*.png"))), 6 * len(SCENES))
        summary = _json(self.one / "prescriptions" / "season_summary.json")
        self.assertTrue(summary["area_stable"])
        self.assertEqual(summary["dates_processed"], len(SCENES))

    def test_2_rerun_changes_nothing(self):
        before = snapshot(self.one)
        code, rep = run_pipeline(self.one)
        self.assertEqual(code, 0)
        self.assertEqual(snapshot(self.one), before)
        self.assertEqual([s["status"] for s in rep["stages"]], ["skipped", "unchanged", "unchanged", "unchanged", "ok"])

    def _m2(self, rep: dict) -> dict:
        return next(s for s in rep["stages"] if s["name"] == "M2")

    def _fresh_full_run(self, name: str, scenes_from: Path) -> Path:
        """Полный прогон с нуля по сценам папки scenes_from — эталон для сравнения."""
        ref = Path(self.tmp.name) / name
        (ref / "scenes").mkdir(parents=True)
        for s in sorted((scenes_from / "scenes").glob("scene_*.tif")):
            shutil.copy2(s, ref / "scenes" / s.name)
        self.assertEqual(run_pipeline(ref)[0], 0)
        return ref

    def test_3_replaced_scene_reprocessed(self):
        """Подмена сцены: reprocess этой даты и всех более поздних, ранние не тронуты, итог как у полного прогона."""
        import rasterio

        from s2fetch.grid import grid_of_file
        from s2fetch.scene import write_scene

        work = Path(self.tmp.name) / "replaced"
        shutil.copytree(self.one, work)
        target = SCENES[len(SCENES) // 2]
        date, stamp = _date_of(target), _stamp_of(target)
        with rasterio.open(target) as src:
            data, tags = src.read(), src.tags()
        data[6] = np.where(np.isfinite(data[6]) & (data[6] > 0), data[6] * np.float32(0.8), data[6])  # B08 −20 %
        write_scene(work / "scenes" / target.name, data, grid_of_file(target), tags)
        before = snapshot(work)
        code, rep = run_pipeline(work)
        self.assertEqual(code, 0)
        m2 = self._m2(rep)
        later = [_date_of(s) for s in SCENES if s.name >= target.name]
        self.assertEqual((m2["mode"], m2["full_rerun_reason"]), ("process_new_scenes", None))
        self.assertEqual(m2["reprocess_asked"], [date])
        self.assertEqual(m2["changed_scenes"], [date])
        self.assertEqual(m2["reprocessed"], later)
        self.assertEqual(m2["cascade"], later[1:])
        self.assertEqual(rep["dates"]["m2_processed"], later)
        self.assertEqual(rep["dates"]["m3_processed"], later)
        self.assertFalse([w for w in rep["warnings"] if "сцена новее" in w or "всю серию" in w], rep["warnings"])
        # файлы М2 и М3 более ранних дат не тронуты (размер, mtime, sha256)
        after = snapshot(work)
        earlier = [_stamp_of(s) for s in SCENES if s.name < target.name]
        dirs = ("zones/", "stats/", "indices/", "hidden_stress/", "prescriptions/")
        early_files = [f for f in before if f.startswith(dirs) and any(e in f for e in earlier)]
        self.assertTrue(early_files)
        self.assertEqual({f: after[f] for f in early_files}, {f: before[f] for f in early_files})
        changed = content_diff(work, self.all)
        self.assertIn(f"zones/zones_{stamp}.tif", changed)
        self.assertIn(f"prescriptions/metrics_{stamp}.json", changed)
        self.assertIn(f"overlays/NDVI_{stamp}.png", changed)
        # итог совпадает с полным прогоном по той же подменённой серии
        self.assertEqual(content_diff(work, self._fresh_full_run("replaced_ref", work)), [])
        # вернули исходную сцену — результат снова как у исходной серии
        shutil.copy2(target, work / "scenes" / target.name)
        self.assertEqual(run_pipeline(work)[0], 0)
        self.assertEqual(content_diff(work, self.all), [])

    def test_6_touch_scenes_no_false_warnings(self):
        """mtime сцен новее результатов, содержимое то же: ничего не пересчитано, в warnings пусто."""
        work = Path(self.tmp.name) / "touched"
        shutil.copytree(self.one, work)
        future = max(p.stat().st_mtime for p in (work / "stats").iterdir()) + 3600
        for s in (work / "scenes").glob("scene_*.tif"):
            os.utime(s, (future, future))
        before = snapshot(work)
        code, rep = run_pipeline(work)
        self.assertEqual(code, 0)
        self.assertEqual(snapshot(work), before)
        self.assertEqual((rep["warnings"], rep["info"]), ([], []))
        self.assertEqual([s["status"] for s in rep["stages"]], ["skipped", "unchanged", "unchanged", "unchanged", "ok"])
        # новая дата при «тронутых» старых: М2 вызывается и шлёт mtime-предупреждения — они уходят в info
        last = _stamp_of(SCENES[-1])
        for p in (work / "zones" / f"zones_{last}.tif", work / "stats" / f"index_stats_{last}.json"):
            p.unlink()
        code, rep = run_pipeline(work)
        self.assertEqual(code, 0)
        self.assertEqual(rep["dates"]["m2_processed"], [_date_of(SCENES[-1])])
        self.assertEqual(rep["warnings"], [])
        self.assertEqual(len(rep["info"]), len(SCENES) - 1, rep["info"])
        self.assertTrue(all("сцена новее результата" in i and "sha256 сцены совпал" in i for i in rep["info"]))
        self.assertEqual(content_diff(work, self.all), [])

    def test_7_earlier_date_via_reprocess(self):
        """Дата раньше уже обработанных идёт через reprocess; ложное «перезапустите всю серию» — в info."""
        work = Path(self.tmp.name) / "earlier"
        (work / "scenes").mkdir(parents=True)
        mid = SCENES[1]
        for s in SCENES:
            if s != mid:
                shutil.copy2(s, work / "scenes" / s.name)
        self.assertEqual(run_pipeline(work)[0], 0)
        shutil.copy2(mid, work / "scenes" / mid.name)
        code, rep = run_pipeline(work)
        self.assertEqual(code, 0)
        m2 = self._m2(rep)
        later = [_date_of(s) for s in SCENES if s.name > mid.name]
        self.assertEqual((m2["mode"], m2["reprocess_asked"], m2["earlier_new_dates"]),
                         ("process_new_scenes", [_date_of(mid)], [_date_of(mid)]))
        self.assertEqual(m2["reprocessed"], later)
        # в warnings — только честное сообщение М2 о каскаде; ложное «перезапустите всю серию» — в info
        self.assertEqual(len(rep["warnings"]), 1, rep["warnings"])
        self.assertIn("пересчитаны и обработанные даты позже", rep["warnings"][0])
        self.assertFalse([w for w in rep["warnings"] if "всю серию" in w])
        self.assertEqual(len([i for i in rep["info"] if "раньше уже обработанной" in i]), 1, rep["info"])
        self.assertEqual(content_diff(work, self.all), [])

    def test_8_failed_reprocess(self):
        """Переобработка упала: дата — ошибка (код 1), М3 и слои для неё не запускаются, прежние zones/stats на месте,
        более поздние даты, посчитанные от прежнего результата, — в warnings."""
        work = Path(self.tmp.name) / "failed_reprocess"
        shutil.copytree(self.all, work)
        target = SCENES[len(SCENES) // 2]
        date, stamp = _date_of(target), _stamp_of(target)
        keep = [f"zones/zones_{stamp}.tif", f"stats/index_stats_{stamp}.json", f"prescriptions/metrics_{stamp}.json"]
        before = snapshot(work)
        (work / "scenes" / target.name).write_bytes(b"not a tiff")
        code, rep = run_pipeline(work)
        self.assertEqual(code, 1)
        self.assertEqual(rep["errors"][date]["stage"], "M2")
        self.assertEqual(self._m2(rep)["reprocess_asked"], [date])
        self.assertNotIn(date, rep["dates"]["m3_processed"])
        self.assertNotIn(stamp, rep["dates"]["layers"])
        after = snapshot(work)
        self.assertEqual({f: after[f] for f in keep}, {f: before[f] for f in keep})
        self.assertFalse(list((work / "overlays").glob(f"*_{stamp}.png")))
        self.assertTrue([w for w in rep["warnings"] if f"от прежнего результата {date}" in w], rep["warnings"])
        self.assertIsNone(_json(work / mp.STATE_NAME)["dates"][date]["scene_sha256"])
        self.assertTrue(rep["checks"]["m2_contract"]["ok"])                    # check_outputs — после М2
        # сцену вернули: следующий запуск снова переобрабатывает дату, итог как у исходной серии
        shutil.copy2(target, work / "scenes" / target.name)
        code, rep = run_pipeline(work)
        self.assertEqual((code, self._m2(rep)["reprocess_asked"]), (0, [date]))
        self.assertEqual(content_diff(work, self.all), [])

    def test_9_rebuild_on_existing_results(self):
        """--rebuild поверх готовых результатов (в КТ3 падал KeyError): вся серия, итог тот же."""
        work = Path(self.tmp.name) / "rebuild"
        shutil.copytree(self.all, work)
        code, rep = run_pipeline(work, "--rebuild")
        self.assertEqual(code, 0)
        self.assertEqual(self._m2(rep)["full_rerun_reason"], "--rebuild")
        self.assertEqual(rep["dates"]["m2_processed"], [_date_of(s) for s in SCENES])
        self.assertEqual(content_diff(work, self.all), [])

    def test_4_broken_scene_does_not_stop_others(self):
        work = Path(self.tmp.name) / "broken"
        shutil.copytree(self.all, work)
        (work / "scenes" / "scene_2026_09_30.tif").write_bytes(b"not a tiff")
        code, rep = run_pipeline(work)
        self.assertEqual(code, 1)
        self.assertEqual(rep["errors"]["2026-09-30"]["stage"], "M2")
        self.assertFalse((work / "prescriptions" / "metrics_2026_09_30.json").exists())
        self.assertFalse(list((work / "overlays").glob("*2026_09_30*")))
        self.assertEqual(content_diff(work, self.all, only="prescriptions"), [])
        self.assertTrue(rep["checks"]["m2_contract"]["ok"])

    @unittest.skipUnless(_has("streamlit", "folium", "streamlit_folium"), "нужны зависимости М4")
    def test_5_dashboard_apptest(self):
        sys.path.insert(0, str(HERE.parent / "tools"))
        import check_m4

        with contextlib.redirect_stderr(io.StringIO()):
            res = check_m4.check(self.one, expect_dates=len(SCENES))
        self.assertEqual(res["problems"], [])
        n_layers = len(check_m4.app_layers())             # слои — из LAYERS app.py, а не константой
        self.assertEqual(res["full"]["renders"], n_layers * len(SCENES))
        self.assertEqual(res["png_only"]["renders"], n_layers * len(SCENES))
        self.assertEqual(res["png_only"]["indices_tif_left"], 0)


if __name__ == "__main__":
    unittest.main()
