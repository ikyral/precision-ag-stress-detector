"""Compare classified coverage with an explicit reference footprint; never fill gaps."""
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def coverage_report(zones_path, reference_path, project):
    with rasterio.open(zones_path) as zones, rasterio.open(reference_path) as ref:
        if ref.count != 1 or ref.crs != zones.crs or ref.shape != zones.shape or ref.transform != zones.transform:
            raise ValueError("Reference mask must have one band and the same CRS, shape and transform as zones")
        reference = ref.read(1, masked=True)
        footprint = (~np.ma.getmaskarray(reference) & np.isfinite(reference.data) & (reference.data > 0))
        band = zones.read(1, masked=True)
        classified = ~np.ma.getmaskarray(band) & np.isin(band.data, (1, 2, 3))
        if not footprint.any():
            raise ValueError("Reference footprint is empty")
        if np.any(classified & ~footprint):
            raise ValueError("Classified pixels extend outside the reference footprint")
        missing = footprint & ~classified

        def area(mask):
            return math.fsum(project(shape(g)).area / 10000 for g, _ in
                             shapes(mask.astype('uint8'), mask=mask, transform=ref.transform))

        full_area, missing_area = area(footprint), area(missing)
        return {
            "reference_mask": Path(reference_path).name,
            "reference_field_area_ha": full_area,
            "unclassified_area_ha": missing_area,
            "reference_pixel_count": int(footprint.sum()),
            "unclassified_pixel_count": int(missing.sum()),
            "coverage_percent": (full_area - missing_area) / full_area * 100,
            "is_complete": not bool(missing.any()),
            "economics_scope": "classified_pixels_only",
            "gap_policy": "M3 preserves input nodata; accepts upstream-filled classes; coverage does not measure observation quality",
        }
