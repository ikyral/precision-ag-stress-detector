"""Run Module 3 over one field's dates and check classified area stability."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

from .core import generate_prescription


def date_from_path(path: str | Path) -> str:
    match = re.search(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})(?!\d)", Path(path).stem)
    if not match:
        raise ValueError(f"No YYYY_MM_DD date in {Path(path).name}; use --date for a single file")
    return datetime.strptime("_".join(match.groups()), "%Y_%m_%d").strftime("%Y_%m_%d")


def run_series(
    zones_dir: str | Path,
    output_dir: str | Path,
    *,
    area_tolerance_ha: float = 0.01,
    **prescription_options,
) -> dict:
    """Process zones_*.tif for one field; never interpret no-data as healthy.

    Failed dates are recorded and other dates continue. Duplicate dates are
    rejected before writing outputs. Area stability concerns classified pixels;
    cloud gaps cannot establish the cadastral field area.
    """
    if not math.isfinite(area_tolerance_ha) or area_tolerance_ha < 0:
        raise ValueError("area_tolerance_ha must be finite and non-negative")
    folder = Path(zones_dir)
    if not folder.is_dir():
        raise ValueError(f"Not an input directory: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".tif", ".tiff"}
                   and p.stem.startswith("zones_"))
    if not files:
        raise ValueError(f"No zones_YYYY_MM_DD.tif files in {folder}")
    dated = [(date_from_path(p), p) for p in files]
    dates = [d for d, _ in dated]
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates: run each field in a separate input/output directory")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    for date_str, path in sorted(dated):
        try:
            geojson, archive, metrics_path = generate_prescription(
                path, out, date_str, **prescription_options)
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            rows.append({
                "date": date_str,
                "source_file": path.name,
                "total_field_area_ha": metrics["total_field_area_ha"],
                "stressed_area_ha": metrics["stressed_area_ha"],
                "saved_percent": metrics["saved_percent"],
                "saved_money_rub": metrics["saved_money_rub"],
                "files": {"geojson": Path(geojson).name,
                          "shapefile_zip": Path(archive).name,
                          "metrics": Path(metrics_path).name},
            })
        except Exception as exc:
            errors.append({"date": date_str, "source_file": path.name,
                           "error": f"{type(exc).__name__}: {exc}"})
    areas = [r["total_field_area_ha"] for r in rows]
    spread = max(areas) - min(areas) if areas else None
    stable = spread <= area_tolerance_ha + 1e-10 if len(areas) > 1 else None
    report = {
        "schema_version": "1.0",
        "dates_found": len(files), "dates_processed": len(rows),
        "area_tolerance_ha": area_tolerance_ha,
        "area_range_ha": [min(areas), max(areas)] if areas else None,
        "area_spread_ha": spread, "area_stable": stable,
        "coverage_note": "Area includes classified pixels only. No-data gaps are excluded. One input directory must contain one field.",
        "warnings": (["Classified field area changes between dates; inspect no-data masks and alignment."]
                     if stable is False else []),
        "records": rows, "errors": errors,
    }
    (out / "season_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
