"""Тесты s2fetch без сети. Запуск из src/module1: python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from s2fetch import BANDS, catalog, clouds, manual, selection  # noqa: E402
from s2fetch.grid import (Grid, compare_grids, field_mask, field_polygon,  # noqa: E402
                          grid_from_tile, grid_of_file, load_boundary)
from s2fetch.scene import assemble, scene_filename, to_reflectance, write_scene  # noqa: E402

ROOT = HERE.parents[2]
SAMPLES = ROOT / "data" / "samples"
REFERENCE = SAMPLES / "sample_scene.tif"
# Плитка T37TEN: transform канала B04 (10 м), как его отдаёт CDSE
T37TEN = Affine(10.0, 0.0, 499980.0, 0.0, -10.0, 5300040.0)

L2A_OLD = "S2B_MSIL2A_20260807T081609_N0511_R121_T37TEN_20260807T100000"
L2A_NEW = "S2B_MSIL2A_20260807T081609_N0512_R121_T37TEN_20260807T110039"
L2A_OTHER_TILE = "S2B_MSIL2A_20260807T081609_N0512_R121_T38TKT_20260807T110039"
L1C = "S2B_MSIL1C_20260807T081609_N0512_R121_T37TEN_20260807T104236"


def _square_field(n=20):
    inside = np.zeros((n, n), bool)
    inside[0:10, 0:10] = True   # 100 пикселей поля
    return inside


class TestCloudFraction(unittest.TestCase):
    def test_percent_counts_only_inside(self):
        inside = _square_field()
        m = np.zeros_like(inside)
        m[0, 0:5] = True          # 5 px в поле
        m[15:20, 15:20] = True    # 25 px вне поля — не считаются
        self.assertAlmostEqual(clouds.percent_in_field(m, inside), 5.0)

    def test_threshold_10_percent(self):
        inside = _square_field()
        scl = np.full(inside.shape, 4, np.uint8)
        scl[0, 0:10] = 9          # ровно 10% — сцена принимается
        self.assertTrue(clouds.evaluate(scl, None, inside, 10.0, 0).accepted)
        scl[1, 0] = 3             # 11% — отброшена ещё по SCL
        v = clouds.evaluate(scl, None, inside, 10.0, 0)
        self.assertFalse(v.accepted)
        self.assertIsNone(v.union_percent)
        self.assertIn("облака", v.reason)

    def test_union_with_s2cloudless_rejects(self):
        inside = _square_field()
        scl = np.full(inside.shape, 4, np.uint8)
        scl[0, 0:6] = 8           # SCL 6%
        extra = np.zeros_like(inside)
        extra[5, 0:6] = True      # s2cloudless ещё 6% -> объединение 12%
        v = clouds.evaluate(scl, extra, inside, 10.0, 0)
        self.assertFalse(v.accepted)
        self.assertAlmostEqual(v.scl_percent, 6.0)
        self.assertAlmostEqual(v.union_percent, 12.0)
        extra[5, 0:6] = False
        extra[0, 0:6] = True      # совпадает с SCL — объединение 6%
        self.assertTrue(clouds.evaluate(scl, extra, inside, 10.0, 0).accepted)

    def test_union_threshold_ignores_dilation(self):
        inside = np.zeros((40, 40), bool)
        inside[0:30, 0:30] = True  # 900 px
        scl = np.full(inside.shape, 4, np.uint8)
        scl[10:19, 10:19] = 9      # 81 px = 9% до расширения, 13×13=169 px после
        v = clouds.evaluate(scl, None, inside, 10.0, 2)
        self.assertTrue(v.accepted)
        self.assertAlmostEqual(v.union_percent, 9.0)
        self.assertAlmostEqual(v.nan_percent, 100 * 169 / 900)

    def test_scl_nodata_and_defect(self):
        inside = _square_field()
        scl = np.full(inside.shape, 4, np.uint8)
        scl[0:2, 0:5] = 0          # 10 px NO_DATA
        scl[2, 0] = 1              # 1 px дефект -> 11% — «нет покрытия»
        v = clouds.evaluate(scl, None, inside, 10.0, 2)
        self.assertFalse(v.accepted)
        self.assertIn("нет покрытия", v.reason)
        scl[2, 0] = 4              # 10% — принимается, невалидные идут в NaN без расширения
        v = clouds.evaluate(scl, None, inside, 10.0, 2)
        self.assertTrue(v.accepted)
        self.assertEqual(int(v.nan_mask.sum()), 10)
        self.assertTrue(v.nan_mask[0:2, 0:5].all())

    def test_dilation_stays_inside(self):
        inside = _square_field()
        scl = np.full(inside.shape, 4, np.uint8)
        scl[9, 9] = 9              # угол поля
        v = clouds.evaluate(scl, None, inside, 10.0, 2)
        self.assertEqual(int(v.nan_mask.sum()), 9)       # 3×3 внутри поля из 5×5
        self.assertFalse((v.nan_mask & ~inside).any())


class TestOffsetAndScene(unittest.TestCase):
    def test_boa_offset_by_baseline(self):
        self.assertEqual(catalog.boa_offset(L2A_NEW), 1000)
        self.assertEqual(catalog.boa_offset("S2A_MSIL2A_20210807T081609_N0301_R121_T37TEN_20210807T100000"), 0)
        self.assertEqual(catalog.boa_offset("S2A_MSIL2A_20220807T081609_N0400_R121_T37TEN_20220807T100000"), 1000)

    def test_to_reflectance(self):
        dn = np.array([1269, 1000, 500, 11000], np.uint16)
        r = to_reflectance(dn, 1000)
        self.assertEqual(r.dtype, np.float32)
        np.testing.assert_allclose(r, [0.0269, 0.0001, 0.0001, 1.0], rtol=1e-6)  # минимум 1 как в эталоне
        np.testing.assert_allclose(to_reflectance(dn, 0)[:2], [0.1269, 0.1])

    def test_assemble_zero_outside_nan_clouds(self):
        inside = _square_field()
        refl = np.full((10, *inside.shape), 0.2, np.float32)
        nan = np.zeros_like(inside)
        nan[0, 0] = True
        nan[15, 15] = True         # вне поля — должно остаться 0
        out = assemble(refl, inside, nan)
        self.assertTrue(np.isnan(out[:, 0, 0]).all())
        self.assertTrue((out[:, ~inside] == 0).all())
        self.assertEqual(int(np.isnan(out[0]).sum()), 1)
        self.assertTrue((out[:, 1, 1] == np.float32(0.2)).all())

    def test_filename(self):
        self.assertEqual(scene_filename("2026-08-07"), "scene_2026_08_07.tif")

    def test_write_scene_format(self):
        import rasterio

        inside = _square_field()
        grid = Grid("EPSG:32637", Affine(10, 0, 597610, 0, -10, 5249140), 20, 20)
        nan = np.zeros_like(inside)
        nan[3, 3] = True
        data = assemble(np.full((10, 20, 20), 0.1, np.float32), inside, nan)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / scene_filename("2026-08-07")
            write_scene(p, data, grid, {"date": "2026-08-07", "source": L2A_NEW})
            with rasterio.open(p) as src:
                self.assertEqual(src.dtypes[0], "float32")
                self.assertEqual(src.nodata, 0)
                self.assertEqual(src.descriptions, tuple(BANDS))
                self.assertEqual(src.compression.name, "deflate")
                self.assertEqual(src.tags()["date"], "2026-08-07")
                a = src.read()
            self.assertTrue(np.isnan(a[:, 3, 3]).all())
            self.assertEqual(int((a[0] == 0).sum()), 300)
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), [p.name])  # без .part/.aux.xml

            m2 = ROOT / "src" / "module2"
            sys.path.insert(0, str(m2))
            try:
                from stress_core.pipeline import read_scene  # контракт входа М2
            except ImportError as exc:
                self.skipTest(f"Модуль 2 не импортируется: {exc}")
            bands, geo, info = read_scene(p)
            self.assertEqual(info["scale"], 1.0)
            self.assertEqual(info["band_names_in_file"], BANDS)
            self.assertEqual(int(np.isfinite(bands["R"]).sum()), 99)


class TestGrid(unittest.TestCase):
    def test_grid_matches_reference(self):
        geom = load_boundary(SAMPLES / "field_boundary.geojson")
        poly = field_polygon(geom, "EPSG:32637")
        grid = grid_from_tile(poly, "EPSG:32637", T37TEN)
        self.assertEqual(compare_grids(grid, grid_of_file(REFERENCE)), [])
        self.assertEqual((grid.width, grid.height), (104, 104))

        import rasterio
        with rasterio.open(REFERENCE) as src:
            ref_inside = src.read(1) != 0
        np.testing.assert_array_equal(field_mask(poly, grid), ref_inside)

    def test_compare_grids_detects_shift(self):
        g = grid_of_file(REFERENCE)
        shifted = Grid(g.crs, g.transform @ Affine.translation(1, 0), g.width, g.height)
        self.assertTrue(compare_grids(g, shifted))
        self.assertTrue(compare_grids(g, Grid("EPSG:32638", g.transform, g.width, g.height)))

    def test_buffered_keeps_pixel_alignment(self):
        g = grid_of_file(REFERENCE)
        b = g.buffered(50)
        self.assertEqual((b.width, b.height), (204, 204))
        self.assertEqual((b.transform.c, b.transform.f), (g.transform.c - 500, g.transform.f + 500))


class TestManualMasks(unittest.TestCase):
    GRID = Grid("EPSG:32637", Affine(10, 0, 597610, 0, -10, 5249140), 20, 20)

    def _write(self, d, features, crs="urn:ogc:def:crs:EPSG::32637"):
        import json
        p = Path(d) / "manual_masks.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
                                 "features": features}), encoding="utf-8")
        return p

    def _square(self, date, col, row, n, kind="cloud"):
        """Квадрат n×n пикселей сетки GRID с левым верхним пикселем (row, col), координаты UTM."""
        x0, y0 = 597610 + col * 10, 5249140 - row * 10
        ring = [[x0, y0], [x0 + 10 * n, y0], [x0 + 10 * n, y0 - 10 * n], [x0, y0 - 10 * n], [x0, y0]]
        return {"type": "Feature", "properties": {"date": date, "kind": kind, "note": "тест"},
                "geometry": {"type": "Polygon", "coordinates": [ring]}}

    def test_own_date_masked_other_not(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [self._square("2026-07-18", 2, 3, 2), self._square("2026-07-28", 10, 10, 3, "shadow")])
            polys = manual.load(p)
            m = manual.mask_for_date(polys, "2026-07-18", self.GRID)
            self.assertEqual(int(m.sum()), 4)
            self.assertTrue(m[3:5, 2:4].all())
            self.assertIsNone(manual.mask_for_date(polys, "2026-08-07", self.GRID))
            self.assertEqual(int(manual.mask_for_date(polys, "2026-07-28", self.GRID).sum()), 9)

    def test_wgs84_polygon(self):
        from pyproj import Transformer
        to_wgs = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
        f = self._square("2026-07-18", 5, 5, 3)
        ring = f["geometry"]["coordinates"][0]
        f["geometry"]["coordinates"] = [[list(to_wgs.transform(x, y)) for x, y in ring]]
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [f], crs="urn:ogc:def:crs:OGC:1.3:CRS84")
            m = manual.mask_for_date(manual.load(p), "2026-07-18", self.GRID)
        self.assertEqual(int(m.sum()), 9)

    def test_manual_in_threshold_and_dilated(self):
        inside = _square_field()
        scl = np.full(inside.shape, 4, np.uint8)
        man = np.zeros_like(inside)
        man[5, 5] = True
        v = clouds.evaluate(scl, None, inside, 10.0, 2, man)
        self.assertTrue(v.accepted)
        self.assertAlmostEqual(v.manual_percent, 1.0)
        self.assertAlmostEqual(v.union_percent, 1.0)
        self.assertEqual(int(v.nan_mask.sum()), 25)          # 1 px, расширенный на 2 px -> 5×5
        man[0:2, 0:6] = True                                   # +12 px -> 13% > 10%
        v = clouds.evaluate(scl, None, inside, 10.0, 2, man)
        self.assertFalse(v.accepted)
        self.assertIn("manual", v.reason)

    def test_missing_file_and_bad_values(self):
        self.assertEqual(manual.load(Path("/nonexistent/manual_masks.geojson")), [])
        self.assertEqual(manual.load(None), [])
        with tempfile.TemporaryDirectory() as d:
            bad = self._square("2026-07-18", 1, 1, 1, kind="haze")
            with self.assertRaises(ValueError):
                manual.load(self._write(d, [bad]))

    def test_repo_file_marks_2026_07_18(self):
        polys = manual.load(manual.DEFAULT_PATH)
        self.assertEqual({(p.date, p.kind) for p in polys}, {("2026-07-18", "cloud"), ("2026-07-18", "shadow")})
        m = manual.mask_for_date(polys, "2026-07-18", grid_of_file(REFERENCE))
        self.assertEqual(int(m.sum()), 162)


class TestSelection(unittest.TestCase):
    def test_one_per_date_newest_and_tile(self):
        got = catalog.one_per_date([L2A_OLD, L2A_OTHER_TILE, L2A_NEW], "T37TEN")
        self.assertEqual(got, {"2026-08-07": L2A_NEW})

    def test_match_l1c(self):
        older = L1C.replace("N0512", "N0511")
        self.assertEqual(catalog.match_l1c(L2A_NEW, [older, L1C]), L1C)
        self.assertIsNone(catalog.match_l1c(L2A_OTHER_TILE, [L1C]))

    def test_weekly_picks_cleanest(self):
        # 2026-08-03 (пн) .. 2026-08-09 (вс) — неделя W32
        clean = {"2026-08-04": 3.0, "2026-08-07": 0.5, "2026-08-09": 0.5, "2026-08-12": 2.0}
        self.assertEqual(selection.pick_weekly(clean), {"2026-W32": "2026-08-07", "2026-W33": "2026-08-12"})

    def test_weeks_between(self):
        self.assertEqual(selection.weeks_between("2026-08-01", "2026-08-12"),
                         ["2026-W31", "2026-W32", "2026-W33"])


class TestManifest(unittest.TestCase):
    def test_rerun_keeps_file_when_only_time_changed(self):
        from s2fetch import manifest

        entry = {"2026-08-07": {"date": "2026-08-07", "selected": True, "nan_percent": 0.5}}
        with tempfile.TemporaryDirectory() as d:
            p = manifest.save(Path(d), {"generated": "t1", "weekly": True}, entry, {})
            text, mtime = p.read_text(encoding="utf-8"), p.stat().st_mtime_ns
            manifest.save(Path(d), {"generated": "t2", "weekly": True}, entry, manifest.load(Path(d)))
            self.assertEqual((p.read_text(encoding="utf-8"), p.stat().st_mtime_ns), (text, mtime))
            manifest.save(Path(d), {"generated": "t3", "weekly": False}, entry, manifest.load(Path(d)))
            self.assertIn('"generated": "t3"', p.read_text(encoding="utf-8"))

    def test_rel_path_posix_and_absolute_fallback(self):
        """Пути в манифесте — от его папки, с «/»; без относительного пути (другой диск Windows) — абсолютный."""
        from unittest import mock

        from s2fetch import manifest

        out = ROOT / "data" / "scenes"
        self.assertEqual(manifest.rel_path(SAMPLES / "field_boundary.geojson", out),
                         "../samples/field_boundary.geojson")
        self.assertEqual(manifest.rel_path(manual.DEFAULT_PATH, out), "../../src/module1/manual_masks.geojson")
        with mock.patch("os.path.relpath", side_effect=ValueError("path is on mount 'D:'")):
            got = manifest.rel_path(manual.DEFAULT_PATH, out)
        self.assertEqual(got, manual.DEFAULT_PATH.resolve().as_posix())
        self.assertTrue(Path(got).is_absolute())


class TestFetchSeries(unittest.TestCase):
    def test_run_returns_code_fetch_series_returns_results(self):
        from unittest import mock

        from s2fetch import pipeline

        opt = pipeline.Options(SAMPLES / "field_boundary.geojson", "2026-08-01", "2026-08-07", Path("unused"))
        with mock.patch.object(pipeline.catalog, "open_client", return_value=None), \
                mock.patch.object(pipeline.catalog, "search", return_value={}):
            self.assertEqual(pipeline.fetch_series(opt), (1, []))   # сцен нет
            self.assertEqual(pipeline.run(opt), 1)


if __name__ == "__main__":
    unittest.main()
