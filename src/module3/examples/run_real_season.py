"""Reproduce M2 -> M3 and audit observed versus M2-filled field area.

Run from the repository root after installing M2 and M3 requirements.
Exit 2 means outputs exist but classified area varies or coverage is incomplete.
"""
from pathlib import Path
import json
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src" / "module2"), str(ROOT / "src" / "module3")]
from stress_core import run_pipeline
from vra.batch import run_series


def main():
    data = ROOT / "data"
    scenes = sorted((data / "scenes").glob("scene_*.tif"))
    if not scenes:
        raise ValueError("No real scenes in data/scenes")
    reference = data / "samples" / "sample_zones.tif"
    with rasterio.open(reference) as ref:
        footprint = ref.read(1, masked=True).filled(0) > 0
        grid = (ref.crs, ref.transform, ref.shape)
    inputs = []
    for path in scenes:
        with rasterio.open(path) as scene:
            if (scene.crs, scene.transform, scene.shape) != grid:
                raise ValueError(f"Grid mismatch: {path.name}")
            values = scene.read()
            # M1 contract: all-zero outside; NaN inside means missing observation.
            inside = ~np.all(values == 0, axis=0)
            if not np.array_equal(inside, footprint):
                raise ValueError(f"Field footprint mismatch: {path.name}")
            inputs.append({"file": path.name, "field_pixels": int(inside.sum()),
                           "nan_pixels": int((inside & np.any(~np.isfinite(values), axis=0)).sum())})
    summary = run_pipeline([data / "scenes"], data,
                           band_order=("B", "G", "R", "RE1", "RE2", "RE3", "N", "N2", "S1", "S2"))
    errors = {date: rec["error"] for date, rec in summary["dates"].items() if "error" in rec}
    if errors:
        raise RuntimeError(f"Module 2 failed: {errors}")
    season = run_series(data / "zones", data / "prescriptions", reference_mask_path=reference)
    for item in inputs:
        stamp = item["file"][6:-4]
        stats = json.loads((data / "stats" / f"index_stats_{stamp}.json").read_text(encoding="utf-8"))
        item.update({key: stats.get(key) for key in
                     ("observed_area_ha", "filled_area_ha", "filled_pixels", "filled_percent")})
        if item["filled_pixels"] != item["nan_pixels"]:
            raise ValueError(f"Unexpected M2 gap count: {item['file']}")
    audit = {"input_coverage": inputs,
             "policy": "M2 fills cloud/shadow classes from nearest zone within field; M3 uses these classified pixels. Indices remain unobserved there.",
             "dates_processed": season["dates_processed"], "incomplete_dates": season["incomplete_dates"],
             "errors": season["errors"]}
    (data / "prescriptions" / "input_coverage.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 1 if season["errors"] else (2 if season["area_stable"] is False or season["incomplete_dates"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
