"""Check eight exports and exercise M4 with Streamlit AppTest (no external tiles)."""
from pathlib import Path
import ast
import json
import math
import os
import sys
import zipfile
import shutil
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
os.environ["PAS_DATA_DIR"] = str(DATA)
sys.path.insert(0, str(ROOT / "src/module3"))
from vra import generate_prescription


def main():
    app = ROOT / "src/module4/app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"recompute_economics", "prescription_rates_match"}]
    ns = {"math": math, "DEFAULT_RATE_FACTORS": {1: 1., 2: 80/150, 3: 0.},
          "ZONE_NAMES": {1: "stress", 2: "risk", 3: "normal"}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(app), "exec"), ns)
    summary = json.loads((DATA / "prescriptions/season_summary.json").read_text())
    assert summary["dates_processed"] == 8 and summary["errors"] == []
    assert summary["area_stable"] and summary["incomplete_dates"] == []
    rows = []
    for rec in summary["records"]:
        date = rec["date"]
        metrics = json.loads((DATA / "prescriptions" / rec["files"]["metrics"]).read_text(encoding="utf-8"))
        stats = json.loads((DATA / "stats" / f"index_stats_{date}.json").read_text(encoding="utf-8"))
        assert math.isclose(metrics["total_field_area_ha"], 79.6, abs_tol=1e-8)
        assert metrics["coverage"]["is_complete"]
        expected_fill = 4.12 if date == "2026_07_18" else 0
        assert math.isclose(stats["filled_area_ha"], expected_fill, abs_tol=1e-8)
        geojson = gpd.read_file(DATA / "prescriptions" / rec["files"]["geojson"])
        archive = DATA / "prescriptions" / rec["files"]["shapefile_zip"]
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            assert {".shp", ".shx", ".dbf", ".prj", ".cpg"} <= {Path(n).suffix for n in bundle.namelist()}
        shp = gpd.read_file(f"zip://{archive.as_posix()}")
        assert geojson.crs.to_epsg() == 4326 and shp.crs.to_epsg() == 32637
        assert geojson.geometry.is_valid.all() and shp.geometry.is_valid.all()
        with rasterio.open(DATA / "zones" / f"zones_{date}.tif") as src:
            valid = src.read(1) > 0
            polygon_pixels = rasterize(((g, 1) for g in shp.geometry), out_shape=src.shape,
                                       transform=src.transform, dtype="uint8") > 0
            assert np.array_equal(valid, polygon_pixels), f"Coverage changed: {date}"
        assert math.isclose(float(shp.area.sum()/10000), metrics["total_field_area_ha"], abs_tol=1e-8)
        assert math.isclose(float(geojson.area_ha.sum()), metrics["total_field_area_ha"], abs_tol=1e-8)
        assert math.isclose(float(shp.area.sum()), shp.geometry.union_all().area, abs_tol=1e-6)
        for price in (0, 45, 60):
            calc = ns["recompute_economics"](metrics, metrics["base_rate_kg_ha"], price)
            assert math.isclose(calc["smart_kg"], metrics["smart_application_kg"], abs_tol=1e-8)
            assert math.isclose(calc["saved_rub"], metrics["saved_fertilizer_kg"] * price, abs_tol=1e-6)
            assert ns["prescription_rates_match"](metrics, calc)
        changed = ns["recompute_economics"](metrics, 200, 45)
        assert not ns["prescription_rates_match"](metrics, changed)
        rows.append({"date": date, "area_ha": metrics["total_field_area_ha"],
                     "missing_ha": metrics["coverage"]["unclassified_area_ha"],
                     "observed_ha": stats["observed_area_ha"], "filled_ha": stats["filled_area_ha"],
                     "early_stress_alert": stats["early_stress_alert"],
                     "saved_kg": metrics["saved_fertilizer_kg"], "saved_percent": metrics["saved_percent"],
                     "saved_rub": metrics["saved_money_rub"], "polygon_count": len(shp)})

    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, str(at.exception)
    renders = 0
    for row in rows:
        at.select_slider[0].set_value(row["date"]).run()
        assert not at.exception, str(at.exception)
        expected_alert = row["date"] in {"2026_07_18", "2026_07_28", "2026_08_07", "2026_08_14"}
        assert row["early_stress_alert"] == expected_alert
        assert any('class="pas-alert"' in m.value for m in at.markdown) == expected_alert
        assert not any("Неполное покрытие" in w.value for w in at.warning)
        assert any("Восстановлено М2" in w.value for w in at.warning) == (row["filled_ha"] > 0)
        layer = next(r for r in at.radio if r.label == "Слой карты")
        for label in layer.options:
            layer.set_value(label).run()
            assert not at.exception, str(at.exception)
            renders += 1
    at.select_slider[0].set_value("2026_07_18").run()
    assert any("Восстановлено М2: 4.12 га" in w.value for w in at.warning)
    base = next(s for s in at.slider if s.label == "Базовая норма удобрений, кг/га")
    base.set_value(200.0).run()
    assert not at.exception, str(at.exception)
    assert all(b.proto.disabled for b in at.get("download_button"))
    assert len(at.get("download_button")) == 2
    base.set_value(150.0).run()
    assert all(not b.proto.disabled for b in at.get("download_button"))
    # The reported regression: actual M3 exports with custom norms and price,
    # loaded through the real M4 sidebar, both new and legacy JSON keys.
    custom_checks = []
    with tempfile.TemporaryDirectory(prefix="m3-contract-") as temp:
        custom_data = Path(temp)
        stamp = "2026_07_03"
        for folder in ("stats", "indices", "zones"):
            (custom_data / folder).mkdir()
            for source in (DATA / folder).glob(f"*{stamp}.*"):
                shutil.copy2(source, custom_data / folder / source.name)
        paths = generate_prescription(DATA / "zones" / f"zones_{stamp}.tif",
                                      custom_data / "prescriptions", stamp,
                                      200, {1: 200, 2: 100, 3: 0}, 50)
        custom = json.loads(Path(paths[2]).read_text(encoding="utf-8"))
        assert math.isclose(custom["saved_money_rub"], 434550, abs_tol=1e-6)
        os.environ["PAS_DATA_DIR"] = str(custom_data)
        try:
            for legacy in (False, True):
                if legacy:
                    custom.pop("base_rate_kg_ha")
                    custom.pop("price_per_kg")
                    for zone in custom["zone_breakdown"]:
                        zone.pop("rate_factor")
                    Path(paths[2]).write_text(json.dumps(custom), encoding="utf-8")
                ui = AppTest.from_file(str(app), default_timeout=60).run()
                assert not ui.exception, str(ui.exception)
                assert next(s.value for s in ui.slider if s.label == "Базовая норма удобрений, кг/га") == 200
                assert next(s.value for s in ui.slider if s.label == "Цена удобрения, ₽/кг") == 50
                money = next(m.value for m in ui.metric if m.label == "Финансовая экономия")
                assert "".join(c for c in money if c.isdigit()) == "43455000", money
                assert any("Файлы рассчитаны для нормы сравнения 200" in c.value for c in ui.caption)
                assert all(not b.proto.disabled for b in ui.get("download_button"))
                custom_checks.append({"legacy_json": legacy, "base_rate": 200, "price": 50,
                                      "saved_rub": custom["saved_money_rub"], "display": money})
        finally:
            os.environ["PAS_DATA_DIR"] = str(DATA)
    price = next(s for s in at.slider if s.label == "Цена удобрения, ₽/кг")
    price.set_value(60.0).run()
    assert not at.exception, str(at.exception)
    assert all(not b.proto.disabled for b in at.get("download_button"))
    result = {"success": True, "dates_checked": len(rows), "dashboard_renders": renders,
              "upstream_fill_warning_checked": True, "early_stress_dates_checked": True,
              "custom_scenarios": custom_checks, "stale_download_guard_checked": True,
              "price_only_change_keeps_downloads": True, "rows": rows,
              "scope": "AppTest checks server rendering. External satellite tiles are not asserted."}
    (DATA / "prescriptions/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
