"""Date discovery, batch validation and public CLI tests."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vra.batch import date_from_path, run_series
from vra.__main__ import main, parse_rates


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "zones"
        self.inputs.mkdir()
        self.out = self.root / "out"

    def write(self, filename, values):
        path = self.inputs / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(values, dtype=np.uint8)
        with rasterio.open(path, "w", driver="GTiff", width=values.shape[1],
                           height=values.shape[0], count=1, dtype="uint8", nodata=0,
                           transform=from_origin(500000, 5000000, 10, 10),
                           crs="EPSG:32637") as dst:
            dst.write(values, 1)
        return path

    def test_date_parsing_and_invalid_dates(self):
        self.assertEqual(date_from_path("zones_2026_08_07.tif"), "2026_08_07")
        self.assertEqual(date_from_path("zones_2026-08-14.tiff"), "2026_08_14")
        for filename in ("sample_zones.tif", "zones_2026_02_30.tif", "zones_2026_13_01.tif"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                date_from_path(filename)

    def test_multiple_dates_keep_constant_area_when_zones_change(self):
        self.write("zones_2026_08_21.tif", np.full((10, 10), 3))
        self.write("zones_2026_08_07.tif", np.full((10, 10), 1))
        self.write("nested/zones_2026_08_14.tiff", np.full((10, 10), 2))
        report = run_series(self.inputs, self.out)
        self.assertEqual(report["dates_found"], 3)
        self.assertEqual(report["dates_processed"], 3)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["area_stable"])
        self.assertAlmostEqual(report["area_spread_ha"], 0, places=8)
        self.assertEqual([r["date"] for r in report["records"]],
                         ["2026_08_07", "2026_08_14", "2026_08_21"])
        saved = json.loads((self.out / "season_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        for row in report["records"]:
            for filename in row["files"].values():
                self.assertTrue((self.out / filename).is_file())

    def test_changed_coverage_is_reported_without_fabricating_area(self):
        self.write("zones_2026_08_07.tif", np.full((10, 10), 3))
        second = np.full((10, 10), 3)
        second[0, :] = 0
        self.write("zones_2026_08_14.tif", second)
        report = run_series(self.inputs, self.out, area_tolerance_ha=0.01)
        self.assertFalse(report["area_stable"])
        self.assertAlmostEqual(report["area_spread_ha"], 0.1, places=6)
        self.assertTrue(report["warnings"])
        self.assertAlmostEqual(report["records"][0]["total_field_area_ha"], 1)
        self.assertAlmostEqual(report["records"][1]["total_field_area_ha"], .9)

    def test_duplicate_dates_rejected_before_any_output(self):
        self.write("zones_2026_08_07.tif", np.ones((10, 10)))
        self.write("nested/zones_2026-08-07.tiff", np.ones((10, 10)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run_series(self.inputs, self.out)
        self.assertFalse(self.out.exists())

    def test_bad_raster_does_not_prevent_processing_later_date(self):
        self.write("zones_2026_08_07.tif", np.full((10, 10), 99))
        self.write("zones_2026_08_14.tif", np.full((10, 10), 1))
        report = run_series(self.inputs, self.out)
        self.assertEqual(report["dates_found"], 2)
        self.assertEqual(report["dates_processed"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["date"], "2026_08_07")
        self.assertEqual(report["records"][0]["date"], "2026_08_14")
        self.assertIsNone(report["area_stable"])

    def test_invalid_tolerance_and_empty_input_rejected(self):
        for tolerance in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                run_series(self.inputs, self.out, area_tolerance_ha=tolerance)
        with self.assertRaises(ValueError):
            run_series(self.inputs, self.out)

    def test_rate_parser_validates_all_classes_and_duplicates(self):
        self.assertEqual(parse_rates("1=150,2=80,3=0"), {1: 150, 2: 80, 3: 0})
        for text in ("1=150,2=80", "1=150,1=80,3=0", "1=abc,2=80,3=0", "1:150,2=80,3=0"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                parse_rates(text)

    def test_cli_single_file_and_explicit_date(self):
        path = self.write("sample_zones.tif", np.ones((10, 10)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["run", str(path), "-o", str(self.out), "--date", "2026-08-07",
                           "--base-rate", "200", "--price", "10", "--rates", "1=100,2=50,3=0"])
        self.assertEqual(status, 0)
        files = json.loads(output.getvalue())
        report = json.loads(Path(files["metrics"]).read_text(encoding="utf-8"))
        self.assertAlmostEqual(report["saved_money_rub"], 1000)

    def test_cli_reports_failure_and_area_change_exit_codes(self):
        self.write("zones_2026_08_07.tif", np.ones((10, 10)))
        self.write("zones_2026_08_14.tif", np.ones((9, 10)))
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["run", str(self.inputs), "-o", str(self.out)])
        self.assertEqual(status, 2)
        with contextlib.redirect_stderr(io.StringIO()):
            status = main(["run", str(self.inputs / "missing.tif"), "-o", str(self.out)])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
