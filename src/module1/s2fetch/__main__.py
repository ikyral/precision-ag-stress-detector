"""python -m s2fetch --boundary ... --start ... --end ... --out ... [--max-cloud 10] [--weekly] [--dry-run]"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from .manual import DEFAULT_PATH as DEFAULT_MANUAL
from .pipeline import Options, run
from .reader import ROOT

DEFAULT_REFERENCE = ROOT / "data" / "samples" / "sample_scene.tif"


def _date(s: str) -> str:
    return dt.date.fromisoformat(s).isoformat()


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m s2fetch",
                                 description="Серия чистых сцен Sentinel-2 L2A по контуру поля -> scene_YYYY_MM_DD.tif")
    ap.add_argument("--boundary", required=True, type=Path, help="field_boundary.geojson (EPSG:4326)")
    ap.add_argument("--start", required=True, type=_date, help="начало интервала, YYYY-MM-DD")
    ap.add_argument("--end", required=True, type=_date, help="конец интервала (включительно), YYYY-MM-DD")
    ap.add_argument("--out", required=True, type=Path, help="папка сцен, обычно data/scenes")
    ap.add_argument("--max-cloud", type=float, default=10.0,
                    help="порог облаков в поле, %% (SCL 3,8,9,10 ∪ s2cloudless ∪ ручная маска; по умолчанию 10)")
    ap.add_argument("--weekly", action="store_true", help="не больше одной сцены на ISO-неделю (самая чистая)")
    ap.add_argument("--dry-run", action="store_true", help="только таблица дат и облачности, без записи файлов")
    ap.add_argument("--force", action="store_true", help="перезаписать уже готовые даты")
    ap.add_argument("--tile", default="T37TEN", help="плитка MGRS (по умолчанию T37TEN)")
    ap.add_argument("--dilate", type=int, default=2, help="расширение маски облаков, px (по умолчанию 2)")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                    help="эталон сетки и маски поля (по умолчанию data/samples/sample_scene.tif; 'none' — не сверять)")
    ap.add_argument("--manual-masks", default=str(DEFAULT_MANUAL),
                    help="ручные маски облаков/теней (по умолчанию src/module1/manual_masks.geojson; "
                         "нет файла — пропускается; 'none' — не использовать)")
    a = ap.parse_args(argv)
    if a.start > a.end:
        ap.error("--start позже --end")
    ref = None if a.reference.lower() == "none" else Path(a.reference)
    man = None if a.manual_masks.lower() == "none" else Path(a.manual_masks)
    return run(Options(a.boundary, a.start, a.end, a.out, a.max_cloud, a.weekly, a.dry_run,
                       a.force, a.tile, a.dilate, ref, man))


if __name__ == "__main__":
    sys.exit(main())
