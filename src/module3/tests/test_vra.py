"""Run from src/module3: python -m unittest discover -s tests -v."""
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vra import generate_prescription


class PrescriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transform = from_origin(500000, 5000000, 10, 10)
        self.case_number = 0

    def raster(self, values, *, crs="EPSG:32637", transform=None,
               nodata=0, valid_mask=None):
        self.case_number += 1
        path = self.root / f"zones_{self.case_number}.tif"
        array = np.asarray(values)
        with rasterio.open(
            path, "w", driver="GTiff", width=array.shape[1],
            height=array.shape[0], count=1, dtype=array.dtype,
            crs=crs, transform=transform or self.transform, nodata=nodata,
        ) as dst:
            dst.write(array, 1)
            if valid_mask is not None:
                dst.write_mask(np.asarray(valid_mask, dtype=np.uint8) * 255)
        return path

    def generate(self, values=None, *, raster_path=None, **kwargs):
        if raster_path is None:
            raster_path = self.raster(np.asarray(values, dtype=np.uint8))
        output = self.root / f"output_{self.case_number}"
        paths = generate_prescription(str(raster_path), str(output), "2026_08_07", **kwargs)
        self.assertEqual(len(paths), 3)
        geojson, archive, report_path = map(Path, paths)
        for path in (geojson, archive, report_path):
            self.assertTrue(path.is_file(), str(path))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        raw = json.loads(geojson.read_text(encoding="utf-8"))
        return report, gpd.read_file(geojson), archive, raw

    @staticmethod
    def three_equal_zones():
        return np.repeat(np.array([[1], [2], [3]], dtype=np.uint8), 20, axis=1)

    def test_known_three_zone_economics(self):
        report, frame, _, _ = self.generate(self.three_equal_zones())
        area = report["total_field_area_ha"]
        self.assertAlmostEqual(area, 0.6, delta=0.00001)
        self.assertAlmostEqual(report["stressed_area_ha"], 0.4, delta=0.00001)
        self.assertAlmostEqual(report["stressed_area_percent"], 200 / 3, delta=0.06)
        self.assertAlmostEqual(report["uniform_application_kg"], 90, delta=0.001)
        self.assertAlmostEqual(report["smart_application_kg"], 46, delta=0.001)
        self.assertAlmostEqual(report["saved_fertilizer_kg"], 44, delta=0.001)
        self.assertAlmostEqual(report["saved_percent"], 220 / 450 * 100, delta=0.06)
        self.assertAlmostEqual(report["saved_money_rub"], 1980, delta=0.01)
        self.assertAlmostEqual(float(frame["area_ha"].sum()), area, delta=0.00001)
        breakdown = {int(row["zone_class"]): row for row in report["zone_breakdown"]}
        self.assertEqual(set(breakdown), {1, 2, 3})
        for zone, rate in {1: 150, 2: 80, 3: 0}.items():
            self.assertAlmostEqual(breakdown[zone]["area_ha"], 0.2, delta=0.00001)
            self.assertAlmostEqual(breakdown[zone]["total_kg"], 0.2 * rate, delta=0.001)

    def test_exports_read_back_and_preserve_rates(self):
        _, frame, archive, raw = self.generate(self.three_equal_zones())
        self.assertEqual(frame.crs.to_epsg(), 4326)
        self.assertEqual(raw["type"], "FeatureCollection")
        required = {"zone_class", "zone_name", "area_ha", "fertilizer_rate", "rate_kg_ha"}
        self.assertTrue(required.issubset(frame.columns))
        self.assertTrue(frame.geometry.is_valid.all())
        self.assertFalse(frame.geometry.is_empty.any())
        for _, row in frame.iterrows():
            expected_rate = {1: 150, 2: 80, 3: 0}[row.zone_class]
            self.assertEqual(row.fertilizer_rate, expected_rate)
            self.assertEqual(row.rate_kg_ha, expected_rate)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            extensions = {Path(name).suffix.lower() for name in names}
            self.assertTrue({".shp", ".shx", ".dbf", ".prj", ".cpg"}.issubset(extensions))
            self.assertEqual(len([name for name in names if name.endswith(".shp")]), 1)
        shp = gpd.read_file(f"zip://{archive.as_posix()}")
        self.assertTrue(shp.crs.is_projected)
        self.assertTrue(shp.geometry.is_valid.all())
        self.assertTrue({"zone_class", "area_ha", "rate_kg_ha"}.issubset(shp.columns))
        self.assertTrue(all(len(name) <= 10 for name in shp.columns if name != "geometry"))
        self.assertEqual(set(shp.zone_class), set(frame.zone_class))
        self.assertAlmostEqual(shp.area_ha.sum(), frame.area_ha.sum(), delta=0.00001)

    def test_healthy_field_and_absent_zones(self):
        report, frame, _, _ = self.generate(np.full((10, 10), 3, dtype=np.uint8))
        self.assertEqual(report["stressed_area_ha"], 0)
        self.assertEqual(report["smart_application_kg"], 0)
        self.assertEqual(report["saved_percent"], 100)
        self.assertEqual(set(frame.zone_class), {3})
        breakdown = {row["zone_class"]: row for row in report["zone_breakdown"]}
        for zone in (1, 2):
            if zone in breakdown:
                self.assertEqual(breakdown[zone]["area_ha"], 0)

    def test_changed_rates_and_price_are_recomputed(self):
        report, _, _, _ = self.generate(self.three_equal_zones(), base_rate_kg_ha=200,
                                      rates_by_zone={1: 180, 2: 90, 3: 10}, price_per_kg=60)
        self.assertAlmostEqual(report["smart_application_kg"], 56, delta=0.001)
        self.assertAlmostEqual(report["saved_money_rub"], 3840, delta=0.01)

    def test_spending_more_is_reported_as_negative_savings(self):
        report, _, _, _ = self.generate(np.ones((10, 10), dtype=np.uint8), base_rate_kg_ha=100,
                                      rates_by_zone={1: 150, 2: 80, 3: 0}, price_per_kg=10)
        self.assertLess(report["saved_fertilizer_kg"], 0)
        self.assertLess(report["saved_money_rub"], 0)
        self.assertAlmostEqual(report["saved_percent"], -50, delta=0.001)

    def test_zero_baseline_and_zero_price_do_not_divide_by_zero(self):
        report, _, _, _ = self.generate(np.full((10, 10), 3, dtype=np.uint8),
                                      base_rate_kg_ha=0, price_per_kg=0)
        self.assertEqual(report["uniform_application_kg"], 0)
        self.assertEqual(report["saved_money_rub"], 0)
        self.assertTrue(report["saved_percent"] is None or math.isfinite(report["saved_percent"]))

    def test_litres_have_explicit_units_and_no_mislabelled_kg(self):
        report, frame, _, _ = self.generate(self.three_equal_zones(), unit="l",
                                          base_rate_kg_ha=2, rates_by_zone={1: 2, 2: 1, 3: 0})
        self.assertEqual(report["quantity_unit"], "l")
        self.assertAlmostEqual(report["uniform_application_l"], 1.2, delta=0.001)
        self.assertAlmostEqual(report["smart_application_l"], 0.6, delta=0.001)
        self.assertAlmostEqual(report["saved_fertilizer_l"], 0.6, delta=0.001)
        self.assertNotIn("smart_application_kg", report)
        self.assertNotIn("rate_kg_ha", frame.columns)

    def test_micro_polygon_merges_without_losing_field_area(self):
        values = np.full((7, 7), 3, dtype=np.uint8)
        values[3, 3] = 1
        unfiltered, raw_frame, _, _ = self.generate(values, min_polygon_area_ha=0)
        filtered, frame, _, _ = self.generate(values, min_polygon_area_ha=0.05)
        self.assertIn(1, set(raw_frame.zone_class))
        self.assertNotIn(1, set(frame.zone_class))
        self.assertAlmostEqual(filtered["total_field_area_ha"],
                               unfiltered["total_field_area_ha"], delta=0.00001)
        self.assertLess(unary_union(frame.geometry).symmetric_difference(
            unary_union(raw_frame.geometry)).area, 1e-12)
        self.assertEqual(filtered["processing"]["merged_polygon_count"], 1)
        self.assertAlmostEqual(filtered["processing"]["source_stressed_area_ha"], 0.01, places=6)
        self.assertEqual(filtered["stressed_area_ha"], 0)

    def test_merging_preserves_background_hole(self):
        values = np.full((9, 9), 3, dtype=np.uint8)
        values[4, 4] = 0
        values[4, 5] = 1
        report, frame, _, _ = self.generate(values)
        self.assertAlmostEqual(report["total_field_area_ha"], 0.80, delta=0.00001)
        union = unary_union(frame.geometry)
        x, y = self.transform * (4.5, 4.5)
        lon, lat = Transformer.from_crs(32637, 4326, always_xy=True).transform(x, y)
        self.assertFalse(union.contains(Point(lon, lat)))
        self.assertTrue(union.is_valid)

    def test_isolated_small_patch_is_retained(self):
        values = np.zeros((10, 10), dtype=np.uint8)
        values[:5, :5] = 3
        values[9, 9] = 1
        report, frame, _, _ = self.generate(values)
        self.assertEqual(set(frame.zone_class), {1, 3})
        self.assertAlmostEqual(report["total_field_area_ha"], 0.26, delta=0.00001)
        self.assertEqual(report["processing"]["retained_small_polygon_count"], 1)

    def test_internal_nodata_mask_is_excluded(self):
        values = np.full((10, 10), 1, dtype=np.uint8)
        mask = np.ones(values.shape, dtype=np.uint8)
        mask[:, :5] = 0
        report, _, _, _ = self.generate(raster_path=self.raster(values, valid_mask=mask))
        self.assertAlmostEqual(report["total_field_area_ha"], 0.5, delta=0.00001)

    def test_explicit_nodata_value_is_not_an_unknown_class(self):
        values = np.full((10, 10), 255, dtype=np.uint8)
        values[:, :5] = 2
        report, frame, _, _ = self.generate(raster_path=self.raster(values, nodata=255))
        self.assertAlmostEqual(report["total_field_area_ha"], 0.5, delta=0.00001)
        self.assertEqual(set(frame.zone_class), {2})

    def test_geographic_crs_uses_physical_area(self):
        path = self.raster(np.ones((10, 10), dtype=np.uint8), crs="EPSG:4326",
                           transform=from_origin(39, 45, 0.0001, 0.0001))
        report, _, _, _ = self.generate(raster_path=path)
        self.assertAlmostEqual(report["total_field_area_ha"], 0.876, delta=0.005)

    def test_projected_feet_are_not_treated_as_square_metres(self):
        path = self.raster(np.ones((10, 10), dtype=np.uint8), crs="EPSG:2263",
                           transform=from_origin(1000000, 200000, 10, 10))
        report, _, _, _ = self.generate(raster_path=path, min_polygon_area_ha=0)
        self.assertAlmostEqual(report["total_field_area_ha"], 0.0929, delta=0.0005)

    def test_geographic_multiclass_merging_preserves_projected_area(self):
        # Coarse angular pixels previously produced drift when shared straight
        # boundaries were simplified before their nonlinear projection.
        rng = np.random.default_rng(27)
        values = rng.integers(1, 4, size=(30, 30), dtype=np.uint8)
        values[10:12, 10:12] = 0
        path = self.raster(values, crs="EPSG:4326",
                           transform=from_origin(39, 45, .002, .002))
        report, frame, _, _ = self.generate(raster_path=path, min_polygon_area_ha=10)
        source_area = report["processing"]["source_total_field_area_ha"]
        self.assertTrue(frame.geometry.is_valid.all())
        self.assertAlmostEqual(report["total_field_area_ha"], source_area,
                               delta=source_area * 1e-7)
        self.assertGreater(report["processing"]["merged_polygon_count"], 0)

    def test_all_background_fails_clearly(self):
        with self.assertRaises(ValueError):
            self.generate(np.zeros((5, 5), dtype=np.uint8))

    def test_missing_crs_is_rejected(self):
        path = self.raster(np.ones((5, 5), dtype=np.uint8), crs=None)
        with self.assertRaises(ValueError):
            self.generate(raster_path=path)

    def test_fractional_zone_values_are_rejected(self):
        path = self.raster(np.full((5, 5), 1.25, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.generate(raster_path=path)

    def test_unknown_zone_class_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generate(np.full((5, 5), 4, dtype=np.uint8))

    def test_invalid_parameters_are_rejected(self):
        cases = [{"base_rate_kg_ha": -1}, {"base_rate_kg_ha": float("nan")},
                 {"price_per_kg": -1}, {"price_per_kg": float("inf")},
                 {"min_polygon_area_ha": -0.01},
                 {"rates_by_zone": {1: -1, 2: 80, 3: 0}},
                 {"rates_by_zone": {1: float("nan"), 2: 80, 3: 0}},
                 {"rates_by_zone": {1: 150, 2: 80}}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(self.three_equal_zones(), **kwargs)

    def test_archive_samples_produce_valid_consistent_artifacts(self):
        repo = Path(__file__).resolve().parents[3]
        for filename in ("sample_zones.tif", "sample_zones_m2.tif"):
            with self.subTest(filename=filename):
                path = repo / "data" / "samples" / filename
                self.assertTrue(path.is_file(), str(path))
                with rasterio.open(path) as src:
                    pixels = src.read(1, masked=True)
                    valid = ~np.ma.getmaskarray(pixels) & (pixels.data > 0)
                    nominal_area = valid.sum() * abs(src.transform.determinant) / 10000
                report, frame, _, _ = self.generate(raster_path=path)
                self.assertAlmostEqual(report["total_field_area_ha"], nominal_area, delta=0.0001)
                self.assertTrue(frame.geometry.is_valid.all())
                self.assertAlmostEqual(sum(row["area_ha"] for row in report["zone_breakdown"]),
                                       report["total_field_area_ha"], delta=0.0005)


if __name__ == "__main__":
    unittest.main()


