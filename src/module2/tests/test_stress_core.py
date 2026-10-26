"""Тесты модуля 2. Запуск:  PYTHONPATH=. python -m unittest discover -s tests -v"""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from stress_core import indices as ix
from stress_core.early_stress import EarlyStressConfig, detect_hidden_stress, evaluate_alert
from stress_core.geotiff_io import GeoMeta, read_geotiff, write_geotiff
from stress_core.pipeline import analyze_field, analyze_scene, parse_date, run_pipeline
from stress_core.synthetic import make_scenes, write_demo_scenes
from stress_core.zoning import ZoningConfig, segment_zones

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import spyndex  # noqa: F401
    HAVE_SPYNDEX = True
except ImportError:
    HAVE_SPYNDEX = False


def _scene_results(n_classes=3):
    """Анализ всех синтетических дат; возвращает [(scene, result)]."""
    out, prev = [], None
    for sc in make_scenes():
        r = analyze_scene(sc.date, sc.bands, sc.geo, ZoningConfig(n_classes=n_classes), prev_anomalies=prev)
        prev = (r.early.z_ndre, r.early.z_ndvi)
        out.append((sc, r))
    return out


class IndicesTest(unittest.TestCase):
    def setUp(self):
        self.b = {k: np.array([[v]]) for k, v in {"R": 0.04, "RE1": 0.14, "N": 0.45, "S1": 0.20}.items()}

    def test_known_values(self):
        r = ix.compute_indices(self.b)
        self.assertAlmostEqual(float(r["NDVI"][0, 0]), (0.45 - 0.04) / (0.45 + 0.04), places=5)
        self.assertAlmostEqual(float(r["NDRE"][0, 0]), (0.45 - 0.14) / (0.45 + 0.14), places=5)
        self.assertAlmostEqual(float(r["NDMI"][0, 0]), (0.45 - 0.20) / (0.45 + 0.20), places=5)
        n, red = 0.45, 0.04
        msavi = 0.5 * (2 * n + 1 - np.sqrt((2 * n + 1) ** 2 - 8 * (n - red)))
        self.assertAlmostEqual(float(r["MSAVI2"][0, 0]), msavi, places=5)

    def test_dtype_and_shape(self):
        r = ix.compute_indices({k: np.full((4, 5), v) for k, v in {"R": .04, "RE1": .14, "N": .45, "S1": .2}.items()})
        for arr in r.values():
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(arr.shape, (4, 5))

    def test_invalid_pixels_become_nan(self):
        bands = {k: np.full((1, 4), v) for k, v in {"R": .04, "RE1": .14, "N": .45, "S1": .2}.items()}
        bands["N"][0] = [np.nan, 0.0, -0.1, 2.0]  # NaN, ноль, отрицательное, «облако»
        for name, arr in ix.compute_indices(bands).items():
            self.assertTrue(np.isnan(arr).all(), name)

    def test_zero_denominator_is_nan_not_inf(self):
        a = ix._nd(np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.isnan(a[0]))

    def test_missing_band_raises(self):
        with self.assertRaises(KeyError):
            ix.compute_indices({"R": np.ones((2, 2)), "N": np.ones((2, 2))})

    @unittest.skipUnless(HAVE_SPYNDEX, "spyndex не установлен")
    def test_matches_spyndex(self):
        rng = np.random.default_rng(1)
        bands = {k: rng.uniform(0.02, 0.6, (20, 20)) for k in ("R", "RE1", "N", "S1")}
        for name, diff in ix.compare_with_spyndex(bands).items():
            self.assertLess(diff, 1e-5, name)


class GeoTiffTest(unittest.TestCase):
    def test_round_trip_keeps_data_and_georeference(self):
        geo = GeoMeta.from_origin(500_000, 5_800_000, 10, 10, 32637)
        data = np.random.default_rng(0).integers(0, 10_000, (3, 17, 23)).astype(np.uint16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.tif"
            write_geotiff(p, data, geo, nodata=0)
            back, g2 = read_geotiff(p)
        np.testing.assert_array_equal(back, data)
        self.assertEqual(g2.epsg, 32637)
        self.assertEqual(g2.pixel_scale[:2], (10.0, 10.0))
        self.assertEqual(g2.tiepoint[3:5], (500_000.0, 5_800_000.0))
        self.assertEqual(g2.pixel_area_m2, 100.0)
        self.assertEqual(g2.nodata, 0.0)

    def test_single_band_is_returned_as_one_band(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.tif"
            write_geotiff(p, np.ones((5, 6), np.uint8), None)
            back, _ = read_geotiff(p)
        self.assertEqual(back.shape, (1, 5, 6))

    def test_geographic_crs_has_no_pixel_area(self):
        geo = GeoMeta.from_origin(37.0, 55.0, 0.0001, 0.0001, 4326, projected=False)
        self.assertEqual(geo.epsg, 4326)
        self.assertIsNone(geo.pixel_area_m2)


class EarlyStressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _scene_results()

    def _masks(self, sc):
        t = sc.truth
        core = {k: t[k] > 0.9 for k in ("acute", "moderate", "hidden")}
        clean = (t["acute"] < 0.01) & (t["moderate"] < 0.01) & (t["hidden"] < 0.01)
        return core, clean

    def test_scenario_is_what_we_claim(self):
        """Скрытый очаг на дате 1: NDRE упал сильно, NDVI почти нет."""
        sc, r = self.res[1]
        core, clean = self._masks(sc)
        d_ndre = np.nanmean(r.indices["NDRE"][clean]) - np.nanmean(r.indices["NDRE"][core["hidden"]])
        d_ndvi = np.nanmean(r.indices["NDVI"][clean]) - np.nanmean(r.indices["NDVI"][core["hidden"]])
        self.assertGreater(d_ndre, 0.08)
        self.assertLess(d_ndvi, 0.04)

    def test_detects_hidden_focus_before_ndvi_drop(self):
        sc, r = self.res[1]
        core, clean = self._masks(sc)
        self.assertGreater(r.early.mask[core["hidden"]].mean(), 0.9)  # recall в ядре
        self.assertLess(r.early.mask[clean].mean(), 0.01)  # ложные срабатывания

    def test_visible_stress_is_not_hidden(self):
        """Острый и умеренный очаги видны по NDVI — это не «скрытый» стресс."""
        for sc, r in self.res:
            core, _ = self._masks(sc)
            self.assertLess(r.early.mask[core["acute"] | core["moderate"]].mean(), 0.05, sc.date)

    def test_no_hidden_focus_on_first_date(self):
        sc, r = self.res[0]
        self.assertLess(r.early.mask.mean(), 0.005)

    def test_focus_stops_being_hidden_once_ndvi_drops(self):
        sc, r = self.res[2]
        core, _ = self._masks(sc)
        self.assertLess(r.early.mask[core["hidden"]].mean(), 0.2)

    def test_no_detection_under_clouds(self):
        sc, r = self.res[2]
        self.assertFalse(r.early.mask[~r.valid].any())

    def test_temporal_rule_catches_mild_decline_that_spatial_rule_misses(self):
        rng = np.random.default_rng(0)
        shape = (80, 80)
        ndre0 = 0.50 + 0.01 * rng.standard_normal(shape)
        ndvi0 = 0.83 + 0.01 * rng.standard_normal(shape)
        ndre1, ndvi1 = ndre0.copy(), ndvi0.copy()
        ndre1[30:50, 30:50] -= 0.025  # мягкое падение NDRE, NDVI не трогаем
        cfg = EarlyStressConfig()
        first = detect_hidden_stress(ndre0, ndvi0, cfg)
        spatial_only = detect_hidden_stress(ndre1, ndvi1, EarlyStressConfig(use_temporal=False))
        both = detect_hidden_stress(ndre1, ndvi1, cfg, prev=(first.z_ndre, first.z_ndvi))
        inner = np.zeros(shape, bool)
        inner[32:48, 32:48] = True  # ядро очага (без краёв, размытых сглаживанием)
        truth = np.zeros(shape, bool)
        truth[30:50, 30:50] = True  # весь очаг целиком
        self.assertLess(spatial_only.mask[inner].mean(), 0.2)
        self.assertGreater(both.mask[inner].mean(), 0.8)
        self.assertEqual(int(both.mask[~truth].sum()), 0)

    def test_homogeneous_healthy_field_has_no_detections(self):
        rng = np.random.default_rng(3)
        ndre = 0.5 + 0.005 * rng.standard_normal((60, 60))
        ndvi = 0.83 + 0.005 * rng.standard_normal((60, 60))
        self.assertEqual(int(detect_hidden_stress(ndre, ndvi).mask.sum()), 0)


class ZoningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _scene_results(n_classes=3)

    def test_recovers_planted_zones(self):
        sc, r = self.res[0]
        z, t = r.zoning.zones, sc.truth
        self.assertGreater((z[t["acute"] > 0.9] == 1).mean(), 0.95)
        self.assertGreater((z[t["moderate"] > 0.9] == 2).mean(), 0.95)
        clean = (t["acute"] < 0.01) & (t["moderate"] < 0.01)
        self.assertGreater((z[clean] == 3).mean(), 0.9)

    def test_raster_contract(self):
        for sc, r in self.res:
            z = r.zoning.zones
            self.assertEqual(z.dtype, np.uint8)
            self.assertEqual(int(z.max()), r.zoning.n_classes)
            self.assertTrue(np.array_equal(z > 0, r.valid), sc.date)  # 0 строго там, где нет данных
            self.assertEqual(sorted(np.unique(z[z > 0]).tolist()), list(range(1, r.zoning.n_classes + 1)))

    def test_classes_ordered_from_stress_to_normal(self):
        for _, r in self.res:
            self.assertEqual(r.zoning.vigor, sorted(r.zoning.vigor))
            self.assertEqual(len(r.zoning.labels), r.zoning.n_classes)
            self.assertEqual(r.zoning.labels[0], "сильный стресс")
            self.assertEqual(r.zoning.labels[-1], "норма")

    def test_no_tiny_zones_left(self):
        cfg = ZoningConfig(n_classes=4)
        sc = make_scenes()[0]
        r = analyze_scene(sc.date, sc.bands, sc.geo, cfg)
        for c in range(1, r.zoning.n_classes + 1):
            lab, n = ndi.label(r.zoning.zones == c)
            sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            self.assertTrue((sizes >= cfg.min_zone_px).all(), f"класс {c}: {sorted(sizes)[:3]}")

    def test_auto_k_within_range(self):
        sc = make_scenes()[0]
        idx = ix.compute_indices(sc.bands)
        res = segment_zones(idx, ZoningConfig(n_classes=None))
        self.assertTrue(3 <= res.k_requested <= 5)
        self.assertTrue(res.silhouette)

    def test_deterministic(self):
        sc = make_scenes()[0]
        idx = ix.compute_indices(sc.bands)
        a = segment_zones(idx, ZoningConfig(n_classes=4)).zones
        b = segment_zones(idx, ZoningConfig(n_classes=4)).zones
        np.testing.assert_array_equal(a, b)

    def test_too_few_pixels_raises(self):
        idx = {n: np.full((5, 5), 0.5, np.float32) for n in ix.INDEX_NAMES}
        with self.assertRaises(ValueError):
            segment_zones(idx)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.inputs = write_demo_scenes(root / "in")
        cls.out = root / "out"
        cls.summary = run_pipeline(cls.inputs, cls.out, zoning_cfg=ZoningConfig(n_classes=3), save_indices=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parse_date(self):
        self.assertEqual(parse_date("S2_20260605_T37.tif"), "2026-06-05")
        self.assertEqual(parse_date("scene_2026-06-05.tif"), "2026-06-05")
        with self.assertRaises(ValueError):
            parse_date("scene.tif")

    def test_outputs_exist_per_date(self):
        for d in ("2026_06_05", "2026_06_20", "2026_07_05"):  # контракт: YYYY_MM_DD
            self.assertTrue((self.out / "zones" / f"zones_{d}.tif").exists(), d)
            self.assertTrue((self.out / "hidden_stress" / f"hidden_stress_{d}.tif").exists(), d)
            self.assertTrue((self.out / "stats" / f"index_stats_{d}.json").exists(), d)
            for name in ("NDVI", "NDRE", "MSAVI", "NDMI"):  # по ТЗ: отдельный файл на индекс
                self.assertTrue((self.out / "indices" / f"{name}_{d}.tif").exists(), f"{name}_{d}")
        self.assertTrue((self.out / "indices_summary.json").exists())

    def test_json_is_strict_and_complete(self):
        text = (self.out / "indices_summary.json").read_text(encoding="utf-8")
        for bad in ("NaN", "Infinity"):
            self.assertNotIn(bad, text)
        data = json.loads(text)
        self.assertEqual(list(data["dates"]), ["2026-06-05", "2026-06-20", "2026-07-05"])
        for rec in data["dates"].values():
            self.assertEqual(set(rec["indices"]), {"NDVI", "NDRE", "MSAVI2", "NDMI"})
            self.assertEqual(set(rec["indices"]["NDVI"]), {"n", "mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95"})
            self.assertAlmostEqual(sum(c["fraction"] for c in rec["zones"]["classes"]), 1.0, places=3)
            self.assertEqual(sum(c["pixels"] for c in rec["zones"]["classes"]), rec["valid_pixels"])
            for c in rec["zones"]["classes"]:
                self.assertAlmostEqual(c["area_ha"], c["pixels"] * 100 / 10_000, places=3)

    def test_zone_raster_matches_summary_and_keeps_georeference(self):
        rec = self.summary["dates"]["2026-07-05"]
        zones, geo = read_geotiff(self.out / rec["files"]["zones"])
        _, src_geo = read_geotiff(self.inputs[2])
        self.assertEqual(zones.dtype, np.uint8)
        self.assertEqual(geo.pixel_scale, src_geo.pixel_scale)
        self.assertEqual(geo.tiepoint, src_geo.tiepoint)
        self.assertEqual(geo.epsg, 32637)
        self.assertEqual(geo.nodata, 0.0)
        for c in rec["zones"]["classes"]:
            self.assertEqual(int((zones[0] == c["class"]).sum()), c["pixels"])

    def test_hidden_raster_values_and_cloud_nodata(self):
        rec = self.summary["dates"]["2026-07-05"]
        hidden, _ = read_geotiff(self.out / rec["files"]["hidden_stress"])
        self.assertTrue(set(np.unique(hidden)) <= {0, 1, 255})
        self.assertTrue((hidden[0, 100:112, 120:150] == 255).all())  # под облаком
        self.assertEqual(int((hidden[0] == 1).sum()), rec["hidden_stress"]["pixels"])

    def test_hidden_focus_reported_on_second_date_only(self):
        pix = [self.summary["dates"][d]["hidden_stress"]["pixels"] for d in sorted(self.summary["dates"])]
        self.assertGreater(pix[1], 500)
        self.assertLess(pix[0], 100)
        self.assertEqual(pix[2], 0)

    def test_each_index_is_single_band_float_with_georeference(self):
        _, src_geo = read_geotiff(self.inputs[0])
        for name in ("NDVI", "NDRE", "MSAVI", "NDMI"):
            arr, geo = read_geotiff(self.out / "indices" / f"{name}_2026_06_05.tif")
            self.assertEqual(arr.shape, (1, 120, 160), name)
            self.assertEqual(arr.dtype, np.float32, name)
            self.assertEqual(geo.tiepoint, src_geo.tiepoint, name)
            self.assertEqual(geo.epsg, 32637, name)

    def test_index_stats_json_for_dashboard(self):
        """Поля, которые читает дашборд M4: mean_* и early_stress_alert."""
        for d in ("2026_06_05", "2026_06_20", "2026_07_05"):
            raw = (self.out / "stats" / f"index_stats_{d}.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            doc = json.loads(raw)
            for key in ("mean_ndvi", "mean_ndre", "mean_msavi", "mean_ndmi"):
                self.assertIsInstance(doc[key], float, key)
            self.assertIsInstance(doc["early_stress_alert"], bool)
            self.assertEqual(doc["date"], d.replace("_", "-"))
            self.assertAlmostEqual(doc["field_area_ha"], doc["valid_pixels"] * 100 / 10_000, places=3)

    def test_means_in_stats_match_summary(self):
        doc = json.loads((self.out / "stats" / "index_stats_2026_06_20.json").read_text(encoding="utf-8"))
        rec = self.summary["dates"]["2026-06-20"]
        self.assertEqual(doc["mean_ndre"], rec["indices"]["NDRE"]["mean"])
        self.assertEqual(doc["mean_msavi"], rec["indices"]["MSAVI2"]["mean"])

    def test_alert_fires_only_when_hidden_focus_present(self):
        alerts = {d: self.summary["dates"][d]["early_stress_alert"] for d in sorted(self.summary["dates"])}
        self.assertEqual(alerts, {"2026-06-05": False, "2026-06-20": True, "2026-07-05": False})
        self.assertIn("hidden_area", self.summary["dates"]["2026-06-20"]["early_stress"]["reasons"])

    def test_bad_scene_does_not_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            good = write_demo_scenes(d / "in")
            bad = d / "in" / "scene_2026-06-10.tif"
            write_geotiff(bad, np.ones((2, 10, 10), np.uint16), None)  # всего 2 канала
            s = run_pipeline([d / "in"], d / "out", zoning_cfg=ZoningConfig(n_classes=3))
        self.assertIn("error", s["dates"]["2026-06-10"])
        self.assertNotIn("error", s["dates"]["2026-06-20"])
        self.assertEqual(len(good), 3)


class AlertRuleTest(unittest.TestCase):
    def test_area_rule(self):
        self.assertTrue(evaluate_alert(0.05, 0.5, 0.8, None)["alert"])
        self.assertFalse(evaluate_alert(0.002, 0.5, 0.8, None)["alert"])
        self.assertFalse(evaluate_alert(None, None, None, None)["alert"])

    def test_dynamics_rule_ndre_falls_faster_than_ndvi(self):
        res = evaluate_alert(0.0, 0.46, 0.80, (0.50, 0.81))  # NDRE -0.04 (-8%), NDVI -0.01 (-1.2%)
        self.assertTrue(res["alert"])
        self.assertEqual(res["reasons"], ["ndre_faster_than_ndvi"])
        self.assertAlmostEqual(res["ndre_drop"], 0.04, places=6)

    def test_dynamics_rule_ignores_equal_decline_and_small_noise(self):
        self.assertFalse(evaluate_alert(0.0, 0.46, 0.75, (0.50, 0.81))["alert"])  # NDVI падает почти так же (-7% против -8%)
        self.assertFalse(evaluate_alert(0.0, 0.495, 0.80, (0.50, 0.81))["alert"])  # падение ниже порога
        self.assertFalse(evaluate_alert(0.0, 0.55, 0.85, (0.50, 0.81))["alert"])  # рост


class OrderingAndEntryPointsTest(unittest.TestCase):
    def test_class_one_is_lowest_ndre(self):
        sc = make_scenes()[0]
        for k in (3, 4, 5):
            r = analyze_scene(sc.date, sc.bands, sc.geo, ZoningConfig(n_classes=k))
            means = [float(np.nanmean(r.indices["NDRE"][r.zoning.zones == c])) for c in range(1, r.zoning.n_classes + 1)]
            self.assertEqual(means, sorted(means), f"k={k}")

    def test_analyze_field_single_scene_and_date_override(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = write_demo_scenes(d / "in")[1]
            nodate = d / "in" / "sample_scene.tif"  # имя без даты, как у эталона от M1
            nodate.write_bytes(src.read_bytes())
            with self.assertRaises(ValueError):
                parse_date(nodate.name)
            rec = analyze_field(nodate, d / "out", date="2026-08-07", zoning_cfg=ZoningConfig(n_classes=3))
            self.assertEqual(rec["date"], "2026-08-07")
            self.assertTrue((d / "out" / "zones" / "zones_2026_08_07.tif").exists())
            self.assertTrue((d / "out" / "stats" / "index_stats_2026_08_07.json").exists())
            rec2 = analyze_field(src, d / "out2", zoning_cfg=ZoningConfig(n_classes=3))  # дата из имени
            self.assertEqual(rec2["date"], "2026-06-20")

    def test_date_override_rejects_multiple_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_demo_scenes(d / "in")
            with self.assertRaises(ValueError):
                run_pipeline([d / "in"], d / "out", date_override="2026-08-07")

    def test_ten_band_scene_via_cli_preset(self):
        """sample_scene.tif от M1: 10 каналов uint16 x10000, nodata = 0 вне поля."""
        from stress_core.__main__ import main
        rng = np.random.default_rng(0)
        h = w = 60
        yy, xx = np.mgrid[:h, :w]
        field = (yy - 30) ** 2 + (xx - 30) ** 2 < 26 ** 2
        base = {"B": .03, "G": .05, "R": .035, "RE1": .07, "RE2": .20, "RE3": .25, "N": .26, "N2": .27, "S1": .14, "S2": .07}
        stack = np.stack([
            np.where(field, np.clip(v * (1 + 0.03 * rng.standard_normal((h, w))), 1e-4, None) * 10_000, 0)
            for v in base.values()
        ]).astype(np.uint16)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_geotiff(d / "sample_scene.tif", stack, GeoMeta.from_origin(597_610, 5_249_140, 10, 10, 32637), nodata=0)
            code = main(["run", str(d / "sample_scene.tif"), "-o", str(d / "out"), "--bands", "s2-10", "--date", "2026-08-07"])
            self.assertEqual(code, 0)
            zones, _ = read_geotiff(d / "out" / "zones" / "zones_2026_08_07.tif")
            np.testing.assert_array_equal(zones[0] > 0, field)  # маска поля = ненулевые пиксели
            self.assertTrue(set(np.unique(zones[0])) <= {0, 1, 2, 3})

    def test_indices_command_writes_plain_named_files(self):
        from stress_core.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scene = write_demo_scenes(d / "in")[0]
            self.assertEqual(main(["indices", str(scene), "-o", str(d / "idx")]), 0)
            self.assertEqual(sorted(p.name for p in (d / "idx").iterdir()),
                             ["MSAVI.tif", "NDMI.tif", "NDRE.tif", "NDVI.tif"])


if __name__ == "__main__":
    unittest.main()
