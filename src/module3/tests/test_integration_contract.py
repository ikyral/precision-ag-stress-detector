"""Contract tests for coverage, exact economics and the M4 calculation helpers."""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
sys.path.insert(0, str(MODULE))
from vra import generate_prescription, calculate_economics
from vra.batch import run_series
from vra.__main__ import main as cli


def dashboard_helpers():
    """Load pure functions from the actual M4 source, avoiding Streamlit side effects."""
    tree = ast.parse((ROOT / "src/module4/app.py").read_text(encoding="utf-8"))
    names = {"recompute_economics", "prescription_rates_match"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {"math": math, "DEFAULT_RATE_FACTORS": {1: 1., 2: 80/150, 3: 0.},
          "ZONE_NAMES": {1: "stress", 2: "risk", 3: "normal"}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "app.py", "exec"), ns)
    return ns


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.ui = dashboard_helpers()

    def raster(self, name, arr, *, x=500000):
        path = self.path / name
        with rasterio.open(path, "w", driver="GTiff", count=1, height=arr.shape[0], width=arr.shape[1],
                           dtype="uint8", crs="EPSG:32637", transform=from_origin(x, 5000000, 10, 10), nodata=0) as dst:
            dst.write(arr.astype("uint8"), 1)
        return path

    def test_kg_aliases_and_exact_custom_factors(self):
        r = calculate_economics({1: 2, 2: 3, 3: 4}, 175, {1: 210, 2: 97, 3: 5}, 52.5)
        self.assertEqual(r["base_rate_kg_ha"], 175)
        self.assertEqual(r["price_per_kg"], 52.5)
        ui = self.ui["recompute_economics"](r, 175, 52.5)
        self.assertAlmostEqual(ui["saved_rub"], r["saved_money_rub"])
        for zone, expected in zip(ui["zones"], [210, 97, 5]):
            self.assertAlmostEqual(zone["rate_kg_ha"], expected, places=12)

    def test_litres_do_not_masquerade_as_kg(self):
        r = calculate_economics({1: 1}, unit="l")
        self.assertIn("base_rate_l_ha", r)
        self.assertNotIn("base_rate_kg_ha", r)
        with self.assertRaises(ValueError):
            self.ui["recompute_economics"](r, 150, 45)

    def test_legacy_generic_names_still_work(self):
        r = calculate_economics({1: 1, 2: 2}, 200, {1: 210, 2: 73, 3: 0})
        r.pop("base_rate_kg_ha")
        for z in r["zone_breakdown"]:
            z.pop("rate_factor")
        self.assertAlmostEqual(self.ui["recompute_economics"](r, 200, 45)["smart_kg"], r["smart_application_kg"])

    def test_price_change_keeps_downloads_and_scales_cost(self):
        r = calculate_economics({1: 1, 2: 2, 3: 3})
        ui = self.ui["recompute_economics"](r, 150, 60)
        self.assertTrue(self.ui["prescription_rates_match"](r, ui))
        self.assertAlmostEqual(ui["saved_rub"], r["saved_money_rub"] * 60/45)

    def test_rate_change_disables_stale_downloads(self):
        r = calculate_economics({1: 1, 2: 2, 3: 3})
        ui = self.ui["recompute_economics"](r, 200, 45)
        self.assertFalse(self.ui["prescription_rates_match"](r, ui))
        self.assertAlmostEqual(ui["smart_kg"], r["smart_application_kg"] * 200/150)

    def test_signed_savings_not_clamped_in_ui(self):
        r = calculate_economics({1: 1}, 100, {1: 200, 2: 0, 3: 0})
        self.assertEqual(self.ui["recompute_economics"](r, 100, 45)["saved_rub"], -4500)

    def test_zero_baseline_preserves_explicit_rates(self):
        r = calculate_economics({1: 1, 2: 2}, 0)
        ui = self.ui["recompute_economics"](r, 0, 45)
        self.assertIsNone(ui["saved_pct"])
        self.assertEqual(ui["smart_kg"], r["smart_application_kg"])
        self.assertTrue(self.ui["prescription_rates_match"](r, ui))

    def test_partial_coverage_does_not_fill_or_charge_missing_pixels(self):
        arr = np.ones((10, 10), dtype="uint8")
        ref = self.raster("reference.tif", arr)
        arr[2:4, 2:4] = 0
        source = self.raster("zones_2026_07_18.tif", arr)
        paths = generate_prescription(source, self.path / "out", "2026_07_18", reference_mask_path=ref)
        r = json.loads(Path(paths[2]).read_text())
        self.assertAlmostEqual(r["total_field_area_ha"], .96)
        self.assertAlmostEqual(r["uniform_application_kg"], 144)
        self.assertAlmostEqual(r["coverage"]["reference_field_area_ha"], 1)
        self.assertAlmostEqual(r["coverage"]["unclassified_area_ha"], .04)
        self.assertEqual(r["coverage"]["unclassified_pixel_count"], 4)
        self.assertFalse(r["coverage"]["is_complete"])
        with rasterio.open(source) as src:
            self.assertEqual(int((src.read(1) == 0).sum()), 4)

    def test_reference_grid_mismatch_rejected(self):
        arr = np.ones((10, 10), dtype="uint8")
        ref = self.raster("reference.tif", arr, x=500010)
        source = self.raster("zones.tif", arr)
        with self.assertRaisesRegex(ValueError, "same CRS"):
            generate_prescription(source, self.path / "out", "2026_07_18", reference_mask_path=ref)

    def test_classification_outside_reference_rejected(self):
        arr = np.ones((10, 10), dtype="uint8")
        source = self.raster("zones.tif", arr)
        arr[0] = 0
        ref = self.raster("reference.tif", arr)
        with self.assertRaisesRegex(ValueError, "outside"):
            generate_prescription(source, self.path / "out", "2026_07_18", reference_mask_path=ref)

    def test_batch_names_incomplete_date(self):
        arr = np.ones((10, 10), dtype="uint8")
        ref = self.raster("reference.tif", arr)
        self.raster("zones_2026_07_08.tif", arr)
        arr[0] = 0
        self.raster("zones_2026_07_18.tif", arr)
        r = run_series(self.path, self.path / "out", reference_mask_path=ref)
        self.assertEqual(r["incomplete_dates"], ["2026_07_18"])
        self.assertFalse(r["area_stable"])
        self.assertEqual(r["errors"], [])

    def test_cli_single_incomplete_reference_returns_warning(self):
        arr = np.ones((10, 10), dtype="uint8")
        ref = self.raster("reference.tif", arr)
        arr[0] = 0
        src = self.raster("zones_2026_07_18.tif", arr)
        with contextlib.redirect_stdout(io.StringIO()):
            status = cli(["run", str(src), "-o", str(self.path / "out"), "--reference-mask", str(ref)])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
