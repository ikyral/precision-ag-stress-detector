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
                "coverage": metrics.get("coverage"),
                "files": {"geojson": Path(geojson).name,
                          "shapefile_zip": Path(archive).name,
                          "metrics": Path(metrics_path).name},
            })
        except Exception as exc:
            errors.append({"date": date_str, "source_file": path.name,
                           "error": f"{type(exc).__name__}: {exc}"})
    report = _summary(rows, errors, len(files), area_tolerance_ha)
    (out / "season_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _summary(rows, errors, dates_found, area_tolerance_ha):
    areas = [r["total_field_area_ha"] for r in rows]
    spread = max(areas) - min(areas) if areas else None
    stable = spread <= area_tolerance_ha + 1e-10 if len(areas) > 1 else None
    report = {
        "schema_version": "1.0",
        "dates_found": dates_found, "dates_processed": len(rows),
        "area_tolerance_ha": area_tolerance_ha,
        "area_range_ha": [min(areas), max(areas)] if areas else None,
        "area_spread_ha": spread, "area_stable": stable,
        "coverage_note": "Area includes classified pixels, including upstream-filled classes. Remaining input nodata is excluded. See M2 stats for observed/filled area. One input directory must contain one field.",
        "warnings": (["Classified field area changes between dates; inspect no-data masks and alignment."]
                     if stable is False else []),
        "records": rows, "errors": errors,
        "incomplete_dates": [r["date"] for r in rows if r["coverage"] and not r["coverage"]["is_complete"]],
    }
    return report


def build_season_summary(prescriptions_dir: str | Path, *, zones_dir: str | Path | None = None,
                         failed: dict[str, str] | None = None,
                         upstream: dict[str, str] | None = None,
                         area_tolerance_ha: float = 0.01) -> dict:
    """Read existing metrics without vectorization or writes; return run_series schema.

    zones_dir makes zone files the expected-date inventory, exposing missing metrics.
    Otherwise metrics filenames define that inventory. Explicit failed/upstream dates
    are excluded even when stale metrics exist. Dates accept YYYY-MM-DD/ YYYY_MM_DD.
    Callers must pass current failures: metrics alone cannot establish freshness.
    """
    if not math.isfinite(area_tolerance_ha) or area_tolerance_ha < 0:
        raise ValueError("area_tolerance_ha must be finite and non-negative")
    out = Path(prescriptions_dir)
    failures = {date_from_path(k): str(v) for k, v in (failed or {}).items()}
    upstream_errors = {date_from_path(k): "M2: " + str(v) for k, v in (upstream or {}).items()}
    inventory = {}
    if zones_dir is not None:
        folder = Path(zones_dir)
        if not folder.is_dir():
            raise ValueError(f"Not an input directory: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.is_file()
                       and p.suffix.lower() in {".tif", ".tiff"} and p.stem.startswith("zones_"))
    else:
        if not out.is_dir():
            raise ValueError(f"Not a prescriptions directory: {out}")
        paths = sorted(out.glob("metrics_*.json"))
    for path in paths:
        date = date_from_path(path)
        if date in inventory:
            raise ValueError("Duplicate dates: run each field in a separate input/output directory")
        inventory[date] = path.name if zones_dir is not None else None
    for date in failures.keys() | upstream_errors.keys():
        inventory.setdefault(date, None)
    rows, errors = [], []
    for date, source in sorted(inventory.items()):
        source = source or f"zones_{date}.tif"
        try:
            error = failures.get(date) or upstream_errors.get(date)
            if error:
                errors.append({"date": date, "source_file": source, "error": error})
                continue
            path = out / f"metrics_{date}.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics["date"].replace("-", "_") != date:
                raise ValueError("Metrics date does not match filename")
            for key in ("total_field_area_ha", "stressed_area_ha", "saved_money_rub"):
                if not math.isfinite(metrics[key]):
                    raise ValueError(f"Non-finite {key}")
            if metrics["total_field_area_ha"] <= 0:
                raise ValueError("Field area must be positive")
            if metrics["saved_percent"] is not None and not math.isfinite(metrics["saved_percent"]):
                raise ValueError("Non-finite saved_percent")
            files = {k: metrics["files"][k] for k in ("geojson", "shapefile_zip", "metrics")}
            for name in files.values():
                artifact = (out / name).resolve()
                if not artifact.is_relative_to(out.resolve()) or not artifact.is_file():
                    raise ValueError(f"Missing or outside-directory artifact: {name}")
            if files["metrics"] != path.name:
                raise ValueError("Metrics artifact does not match filename")
            if zones_dir is None:
                source = Path(metrics["input_raster"].replace("\\", "/")).name
            row = {"date": date, "source_file": source,
                   "total_field_area_ha": metrics["total_field_area_ha"],
                   "stressed_area_ha": metrics["stressed_area_ha"],
                   "saved_percent": metrics["saved_percent"],
                   "saved_money_rub": metrics["saved_money_rub"],
                   "coverage": metrics.get("coverage"), "files": files}
            # Validate coverage and JSON before accepting this date.
            if row["coverage"] is not None:
                if not isinstance(row["coverage"].get("is_complete"), bool):
                    raise ValueError("coverage.is_complete must be boolean")
            json.dumps(row, allow_nan=False)
            rows.append(row)
        except Exception as exc:
            errors.append({"date": date, "source_file": source,
                           "error": f"{type(exc).__name__}: {exc}"})
    return _summary(rows, errors, len(inventory), area_tolerance_ha)
