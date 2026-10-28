"""Read-only summaries and rejection of stale/invalid date outputs."""
import json
import unittest
from unittest.mock import patch
import test_batch
from vra.batch import build_season_summary, run_series

class SummaryTests(unittest.TestCase):
    setUp = test_batch.BatchTests.setUp
    write = test_batch.BatchTests.write

    def prepare(self):
        self.write("zones_2026_08_07.tif", [[1, 2], [3, 3]])
        self.write("zones_2026_08_14.tif", [[2, 2], [3, 1]])
        return run_series(self.inputs, self.out)

    def test_identical_summary_without_writes_or_vectorization(self):
        expected = self.prepare()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.rglob("*") if p.is_file()}
        with patch("vra.batch.generate_prescription", side_effect=AssertionError("must not vectorize")):
            self.assertEqual(build_season_summary(self.out), expected)
            self.assertEqual(build_season_summary(self.out, zones_dir=self.inputs), expected)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_upstream_failure_excludes_stale_metrics(self):
        self.prepare()
        result = build_season_summary(self.out, upstream={"2026-08-07": "bad scene"})
        self.assertEqual(result["dates_processed"], 1)
        self.assertEqual(result["errors"][0]["error"], "M2: bad scene")
        self.assertEqual(result["records"][0]["date"], "2026_08_14")

    def test_failure_without_any_old_artifact_is_reported(self):
        self.prepare()
        result = build_season_summary(self.out, failed={"2026_08_21": "failed"})
        self.assertEqual(result["dates_found"], 3)
        self.assertEqual(result["errors"][0]["date"], "2026_08_21")

    def test_missing_metrics_detected_from_zone_inventory(self):
        self.prepare()
        (self.out / "metrics_2026_08_07.json").unlink()
        result = build_season_summary(self.out, zones_dir=self.inputs)
        self.assertEqual(result["dates_found"], 2)
        self.assertEqual(result["dates_processed"], 1)
        self.assertIn("FileNotFoundError", result["errors"][0]["error"])

    def test_bad_json_and_wrong_date_do_not_abort_other_dates(self):
        self.prepare()
        p = self.out / "metrics_2026_08_07.json"
        original = json.loads(p.read_text())
        for text in ("{", json.dumps({**original, "date": "2026_08_08"}),
                     json.dumps({**original, "total_field_area_ha": float("nan")})):
            p.write_text(text)
            result = build_season_summary(self.out)
            self.assertEqual(result["dates_processed"], 1)
            self.assertEqual(len(result["errors"]), 1)

    def test_missing_export_is_not_success(self):
        self.prepare()
        (self.out / "shapefile_2026_08_07.zip").unlink()
        self.assertEqual(build_season_summary(self.out)["dates_processed"], 1)

    def test_invalid_tolerance_and_duplicate_dates(self):
        self.prepare()
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_season_summary(self.out, area_tolerance_ha=tolerance)
        self.write("nested/zones_2026-08-07.tiff", [[1, 2], [3, 3]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_season_summary(self.out, zones_dir=self.inputs)

    def test_empty_directory_is_explicit_zero_date_summary(self):
        self.out.mkdir()
        result = build_season_summary(self.out)
        self.assertEqual(result["dates_found"], 0)
        self.assertIsNone(result["area_stable"])
