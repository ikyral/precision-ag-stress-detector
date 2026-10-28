"""Проверка кабинета М4 (src/module4/app.py) на папке данных через streamlit AppTest, без браузера.

    python tools/check_m4.py <data_dir> [--expect-dates 8]

1. PAS_DATA_DIR = <data_dir>: все даты × все слои карты рисуются без исключений и без сообщений
   «Растр … не найден»; на датах с filled_area_ha > 0 есть предупреждение «Восстановлено М2».
2. Копия <data_dir> без indices/*.tif: слои индексов всё равно рисуются — значит, М4 берёт PNG из overlays/.
3. Список слоёв — из LAYERS app.py (ast): радио кабинета совпадает с ним, у каждого слоя-растра есть PNG всех дат.
Печатает JSON-сводку; код выхода 0 — всё прошло, 1 — нет.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "src" / "module4" / "app.py"
INDEX_LAYERS = {"NDVI", "NDRE", "MSAVI", "NDMI"}  # слои индексов, для которых М1 рисует PNG


def app_layers(app: Path = APP) -> dict:
    """LAYERS из app.py М4 через ast (без импорта streamlit): {подпись: слой или None}."""
    for node in ast.parse(app.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "LAYERS"):
            return ast.literal_eval(node.value)
    raise ValueError(f"{app}: нет LAYERS")


def _stats(data: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((data / "stats").glob("index_stats_*.json")):
        out[p.stem[len("index_stats_"):]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def exercise(data: Path, timeout: int = 120) -> dict:
    """Все даты × все слои; возвращает сводку с найденными проблемами."""
    from streamlit.testing.v1 import AppTest

    os.environ["PAS_DATA_DIR"] = str(data)
    stats = _stats(data)
    problems: list[str] = []
    at = AppTest.from_file(str(APP), default_timeout=timeout).run()
    if at.exception:
        return {"data_dir": str(data), "renders": 0, "problems": [f"старт: {at.exception}"]}
    source = next((c.value for c in at.caption if c.value.startswith("Данные:")), "")
    if str(data) not in source:
        problems.append(f"кабинет читает не ту папку: {source!r}")
    renders, layer_labels = 0, []
    for d in sorted(stats):
        if at.select_slider:
            at.select_slider[0].set_value(d).run()
        if at.exception:
            problems.append(f"{d}: {at.exception}")
            continue
        filled = (stats[d].get("filled_area_ha") or 0) > 0
        if any("Восстановлено М2" in w.value for w in at.warning) != filled:
            problems.append(f"{d}: предупреждение «Восстановлено М2» {'нет' if filled else 'лишнее'}")
        layer = next(r for r in at.radio if r.label == "Слой карты")
        layer_labels = list(layer.options)
        for label in layer_labels:
            layer.set_value(label).run()
            renders += 1
            if at.exception:
                problems.append(f"{d} / {label}: {at.exception}")
            for msg in at.info:
                if "не найден" in msg.value or "rasterio" in msg.value:
                    problems.append(f"{d} / {label}: {msg.value}")
            layer = next(r for r in at.radio if r.label == "Слой карты")
    return {"data_dir": str(data), "dates": sorted(stats), "layers": layer_labels, "renders": renders,
            "filled_dates": [d for d, s in stats.items() if (s.get("filled_area_ha") or 0) > 0],
            "problems": problems}


def check(data: Path, expect_dates: int | None = None) -> dict:
    data = Path(data).resolve()
    full = exercise(data)
    with tempfile.TemporaryDirectory(prefix="m4-png-only-") as tmp:
        copy = Path(tmp) / "data"
        shutil.copytree(data, copy, ignore=lambda d, names: [n for n in names if Path(d).name == "indices"
                                                              and n.endswith(".tif")])
        png_only = exercise(copy)
        png_only["indices_tif_left"] = len(list((copy / "indices").glob("*.tif")))
    problems = [f"полная папка: {p}" for p in full["problems"]] + [f"без tif: {p}" for p in png_only["problems"]]
    if expect_dates is not None and len(full.get("dates", [])) != expect_dates:
        problems.append(f"дат {len(full.get('dates', []))}, ожидалось {expect_dates}")
    layers = app_layers()
    if full.get("layers") != list(layers):
        problems.append(f"слои кабинета {full.get('layers')} не совпадают с LAYERS app.py {list(layers)}")
    if not INDEX_LAYERS <= set(layers.values()):
        problems.append(f"не все слои индексов в кабинете: {sorted(set(layers.values()) - {None})}")
    meta_layers = (json.loads((data / "overlays" / "meta.json").read_text(encoding="utf-8")).get("layers") or {}
                   if (data / "overlays" / "meta.json").exists() else {})
    for name in sorted({v for v in layers.values() if v is not None}):
        lost = [d for d in full.get("dates", []) if name not in meta_layers.get(d.replace("-", "_"), {})]
        if lost:
            problems.append(f"слой {name} из LAYERS: нет PNG в overlays/meta.json для {', '.join(lost)}")
    return {"ok": not problems, "app_layers": layers, "full": full, "png_only": png_only, "problems": problems}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="AppTest кабинета М4 на папке данных")
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--expect-dates", type=int, default=None)
    a = ap.parse_args(argv)
    res = check(a.data_dir, a.expect_dates)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
