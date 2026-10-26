"""CLI:  python -m stress_core run <сцены> -o <папка>   |   python -m stress_core demo -o <папка>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .early_stress import EarlyStressConfig
from .pipeline import DEFAULT_BAND_ORDER, run_pipeline
from .synthetic import write_demo_scenes
from .zoning import ZoningConfig


def _classes(value: str) -> int | None:
    return None if value.lower() == "auto" else int(value)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", required=True, type=Path, help="папка результатов")
    p.add_argument("--bands", default=",".join(DEFAULT_BAND_ORDER),
                   help="порядок каналов в файле, имена из ASI (по умолчанию R,RE1,N,S1)")
    p.add_argument("--scale", type=float, default=None,
                   help="делитель DN (по умолчанию 10000 для int, 1 для float)")
    p.add_argument("--classes", type=_classes, default=3, help="число классов 2–5 или 'auto' (по умолчанию 3 — контракт M2 -> M3)")
    p.add_argument("--backend", choices=["numpy", "spyndex"], default="numpy")
    p.add_argument("--save-indices", action="store_true", help="сохранить растры индексов")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stress_core", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="обработать GeoTIFF-сцены")
    run.add_argument("inputs", nargs="+", help="файлы или папки с GeoTIFF (дата YYYY-MM-DD в имени)")
    _add_common(run)
    demo = sub.add_parser("demo", help="сгенерировать синтетическое поле и обработать его")
    _add_common(demo)
    args = parser.parse_args(argv)

    zcfg = ZoningConfig(n_classes=args.classes)
    bands = tuple(b.strip() for b in args.bands.split(","))
    inputs = args.inputs if args.cmd == "run" else [str(p) for p in write_demo_scenes(args.out / "demo_input")]

    summary = run_pipeline(inputs, args.out, band_order=bands, scale=args.scale,
                           zoning_cfg=zcfg, early_cfg=EarlyStressConfig(),
                           backend=args.backend, save_indices=args.save_indices)
    ok = [d for d, r in summary["dates"].items() if "error" not in r]
    bad = {d: r["error"] for d, r in summary["dates"].items() if "error" in r}
    print(f"обработано дат: {len(ok)}; результаты: {args.out}")
    for d, err in bad.items():
        print(f"  ОШИБКА {d}: {err}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
