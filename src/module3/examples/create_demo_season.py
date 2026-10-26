"""Six explicitly synthetic weeks for exercising the batch contract."""
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
sys.path.insert(0, str(MODULE))
from vra.batch import run_series


def main():
    folder = ROOT / "data" / "module3_demo" / "synthetic_season"
    inputs = folder / "zones"
    inputs.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((100, 100))
    field = (x - 49.5)**2 + (y - 49.5)**2 <= 45**2
    for i in range(6):
        distance = np.sqrt((x - 35)**2 + (y - 42)**2)
        mask = np.where(field, 3, 0).astype("uint8")
        mask[field & (distance < 14 + i * 3)] = 2
        mask[field & (distance < 5 + i * 2)] = 1
        stamp = (date(2026, 7, 3) + timedelta(weeks=i)).strftime("%Y_%m_%d")
        with rasterio.open(inputs / f"zones_{stamp}.tif", "w", driver="GTiff", count=1,
                           width=100, height=100, dtype="uint8", crs="EPSG:32637",
                           transform=from_origin(500000, 5000000, 10, 10), nodata=0,
                           compress="deflate") as target:
            target.write(mask, 1)
            target.update_tags(data_origin="SYNTHETIC; not satellite observations")
    (folder / "PROVENANCE.json").write_text(json.dumps({
        "synthetic": True, "purpose": "Batch processing and area stability demonstration only",
        "observed_field_history": False, "dates_are_simulated": True,
        "generator": "src/module3/examples/create_demo_season.py",
    }, indent=2), encoding="utf-8")
    result = run_series(inputs, folder / "prescriptions")
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] or result["area_stable"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
