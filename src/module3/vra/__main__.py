"""Command-line interface: python -m vra run INPUT -o OUTPUT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import date_from_path, run_series
from .core import generate_prescription


def parse_rates(text: str) -> dict[int, float]:
    try:
        pairs = [part.split("=") for part in text.split(",")]
        result = {int(key): float(value) for key, value in pairs}
        if len(result) != len(pairs) or set(result) != {1, 2, 3}:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use --rates 1=150,2=80,3=0 (each class once)") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRA maps and application economics (Module 3)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Process a zone raster or directory of dated rasters")
    run.add_argument("input", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--date", help="YYYY-MM-DD or YYYY_MM_DD; single file only")
    run.add_argument("--base-rate", type=float, default=150.0, help="Uniform comparison rate per ha")
    run.add_argument("--price", type=float, default=45.0, help="RUB per kg or litre")
    run.add_argument("--rates", type=parse_rates, default=None, help="Example: 1=150,2=80,3=0")
    run.add_argument("--min-area", type=float, default=0.05, help="Merge threshold, hectares")
    run.add_argument("--unit", choices=["kg", "l"], default="kg")
    run.add_argument("--reference-mask", type=Path, help="Fixed footprint on the same grid; positive valid pixels are inside the field")
    run.add_argument("--area-tolerance-ha", type=float, default=0.01, help="Batch area stability tolerance")
    args = parser.parse_args(argv)
    options = dict(base_rate_kg_ha=args.base_rate, rates_by_zone=args.rates,
                   price_per_kg=args.price, min_polygon_area_ha=args.min_area, unit=args.unit)
    if args.reference_mask is not None:
        options["reference_mask_path"] = args.reference_mask
    try:
        if args.input.is_dir():
            if args.date:
                parser.error("--date is only supported with a single input file")
            result = run_series(args.input, args.output, area_tolerance_ha=args.area_tolerance_ha,
                                **options)
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return 1 if result["errors"] else (2 if result["area_stable"] is False or result["incomplete_dates"] else 0)
        date_str = args.date or date_from_path(args.input)
        paths = generate_prescription(args.input, args.output, date_str, **options)
        print(json.dumps(dict(zip(("geojson", "shapefile_zip", "metrics"), paths)),
                         ensure_ascii=True, indent=2))
        report = json.loads(Path(paths[2]).read_text(encoding="utf-8"))
        return 2 if report.get("coverage", {}).get("is_complete") is False else 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
