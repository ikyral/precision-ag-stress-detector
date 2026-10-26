"""CLI модуля 2.

  python -m stress_core run <сцены> -o <папка>      серия сцен -> зоны, статистика, индексы
  python -m stress_core indices <scene.tif> -o <папка>   только NDVI.tif, NDRE.tif, MSAVI.tif, NDMI.tif
  python -m stress_core demo -o <папка>             синтетическое поле и его обработка
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .early_stress import EarlyStressConfig
from .indices import INDEX_NAMES, compute_indices
from .pipeline import SENTINEL2_10_BAND_ORDER, load_scene, run_pipeline, save_index_rasters
from .synthetic import write_demo_scenes
from .zoning import ZoningConfig


def _classes(value: str) -> int | None:
    return None if value.lower() == "auto" else int(value)


def _bands(value: str) -> tuple[str, ...] | None:
    if value.lower() == "auto":
        return None
    if value.lower() in ("s2-10", "s2_10"):  # B,G,R,RE1,RE2,RE3,N,N2,S1,S2 — формат sample_scene.tif
        return SENTINEL2_10_BAND_ORDER
    return tuple(b.strip() for b in value.split(",") if b.strip())


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", required=True, type=Path, help="папка результатов")
    p.add_argument("--bands", type=_bands, default=None,
                   help="порядок каналов: по умолчанию определяется по файлу (описания B02..B12; 10 каналов = "
                        "формат M1; 4 канала = R,RE1,N,S1). Иначе имена ASI через запятую или "
                        "'s2-10' = B,G,R,RE1,RE2,RE3,N,N2,S1,S2")
    p.add_argument("--scale", type=float, default=None,
                   help="делитель DN -> отражение (по умолчанию 10000 для целых; для float 1, "
                        "если значения уже 0..1, иначе 10000)")
    p.add_argument("--backend", choices=["numpy", "spyndex"], default="numpy")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stress_core", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="обработать GeoTIFF-сцены")
    run.add_argument("inputs", nargs="+", help="файлы или папки с GeoTIFF (дата YYYY_MM_DD в имени)")
    run.add_argument("--date", help="дата YYYY-MM-DD для одной сцены (по умолчанию из имени файла "
                                    "или тега 'date' в метаданных)")
    run.add_argument("--classes", type=_classes, default=3,
                     help="число классов 2–5 или 'auto' (по умолчанию 3 — контракт M2 -> M3)")
    run.add_argument("--no-indices", action="store_true", help="не сохранять растры индексов")
    run.add_argument("--no-history", action="store_true",
                     help="не сравнивать первую сцену с более ранней датой из уже имеющихся результатов в -o")
    _add_common(run)

    idx = sub.add_parser("indices", help="посчитать индексы одной сцены и сохранить NDVI.tif, NDRE.tif, ...")
    idx.add_argument("scene", type=Path)
    _add_common(idx)

    demo = sub.add_parser("demo", help="сгенерировать синтетическое поле и обработать его")
    demo.add_argument("--classes", type=_classes, default=3)
    _add_common(demo)

    args = parser.parse_args(argv)

    if args.cmd == "indices":
        bands, geo = load_scene(args.scene, args.bands, args.scale)
        files = save_index_rasters(compute_indices(bands, INDEX_NAMES, backend=args.backend), geo, args.out)
        print("сохранено:", ", ".join(files.values()), "->", args.out)
        return 0

    zcfg = ZoningConfig(n_classes=args.classes)
    if args.cmd == "run":
        inputs, date, save_idx, history = args.inputs, args.date, not args.no_indices, not args.no_history
    else:
        inputs, date, save_idx, history = [str(p) for p in write_demo_scenes(args.out / "demo_input")], None, True, False

    summary = run_pipeline(inputs, args.out, band_order=args.bands, scale=args.scale,
                           zoning_cfg=zcfg, early_cfg=EarlyStressConfig(), backend=args.backend,
                           save_indices=save_idx, date_override=date, use_history=history)
    ok = [d for d, r in summary["dates"].items() if "error" not in r]
    bad = {d: r["error"] for d, r in summary["dates"].items() if "error" in r}
    print(f"обработано дат: {len(ok)}; результаты: {args.out}")
    for d, err in bad.items():
        print(f"  ОШИБКА {d}: {err}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
