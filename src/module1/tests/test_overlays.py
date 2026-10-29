"""Тесты PNG-оверлеев для М4 без сети. Запуск из src/module1: python -m unittest discover -s tests -v"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from s2fetch import overlays as ov  # noqa: E402
from s2fetch.grid import grid_of_file  # noqa: E402
from s2fetch.scene import write_scene  # noqa: E402

ROOT = HERE.parents[2]
APP = ROOT / "src" / "module4" / "app.py"
REFERENCE = ROOT / "data" / "samples" / "sample_scene.tif"
DATES = ("2026_08_07", "2026_08_14")
PEAK = (50, 60)        # пиксель с особым значением NDVI (row, col)
CLOUD = (slice(30, 36), slice(40, 46))


def _app_namespace() -> dict:
    """Константы и функции app.py М4 через ast, без импорта streamlit."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CMAP_VEG", "CMAP_WATER", "DEFAULT_RANGES", "STATS_INDEX_KEYS"):
                consts[name] = ast.literal_eval(node.value)
            elif name == "INDEX_CMAPS":
                consts[name] = {ast.literal_eval(k): v.id for k, v in zip(node.value.keys, node.value.values)}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("colorize", "season_ranges")]
    ns = {"np": np, **{k: v for k, v in consts.items() if k != "INDEX_CMAPS"}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(APP), "exec"), ns)  # noqa: S102
    ns["INDEX_CMAPS_NAMES"] = consts["INDEX_CMAPS"]
    return ns


def _stats(lo: float, hi: float) -> dict:
    return {"indices": {k: {"p5": lo + i * 0.01, "p95": hi - i * 0.01, "mean": (lo + hi) / 2}
                        for i, k in enumerate(("NDVI", "NDRE", "MSAVI2", "NDMI"))}}


def _write_tif(path: Path, arr: np.ndarray, grid, nodata) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", count=arr.shape[0], dtype=arr.dtype.name, crs=grid.crs,
                       transform=grid.transform, width=grid.width, height=grid.height, nodata=nodata) as dst:
        dst.write(arr)


def make_dataset(root: Path) -> tuple[np.ndarray, object]:
    """Мини-набор data/ на сетке эталона: 2 даты, облако в первой, особый пиксель NDVI."""
    import rasterio

    grid = grid_of_file(REFERENCE)
    with rasterio.open(REFERENCE) as src:
        field = src.read(1) != 0
    for k, stamp in enumerate(DATES):
        refl = np.full((10, grid.height, grid.width), 0.08, np.float32)
        refl[2] = 0.05 + 0.001 * np.arange(grid.width)[None, :]   # B04 — градиент
        data = np.where(field[None], refl, np.float32(0))
        cloud = np.zeros_like(field)
        if k == 0:
            cloud[CLOUD] = True
            data[:, cloud & field] = np.nan
        write_scene(root / "scenes" / f"scene_{stamp}.tif", data.astype(np.float32), grid, {"date": stamp})
        for name in ov.INDEX_LAYERS:
            v = np.where(field & ~cloud, np.float32(0.3 + 0.05 * k), np.float32(np.nan)).astype(np.float32)
            if name == "NDVI":
                v[PEAK] = 0.85
            if name == "NDRE":   # шахматная доска: сверка положения каждого пикселя поля
                v = np.where(np.isfinite(v), np.where(np.indices(field.shape).sum(0) % 2, 0.3, 0.7), v)
                v = v.astype(np.float32)
            _write_tif(root / "indices" / f"{name}_{stamp}.tif", v, grid, float("nan"))
        zones = np.where(field, 1 + (np.arange(grid.width)[None, :] % 3), 0).astype(np.uint8)
        _write_tif(root / "zones" / f"zones_{stamp}.tif", zones, grid, 0)
        hidden = np.where(field & ~cloud, 0, 255).astype(np.uint8)
        hidden[PEAK] = 1
        _write_tif(root / "hidden_stress" / f"hidden_stress_{stamp}.tif", hidden, grid, 255)
        (root / "stats").mkdir(exist_ok=True)
        (root / "stats" / f"index_stats_{stamp}.json").write_text(
            json.dumps({"date": stamp, **_stats(0.2 + 0.1 * k, 0.8 - 0.05 * k)}), encoding="utf-8")
    return field, grid


def png_pixel(meta: dict, x: float, y: float, crs: str) -> tuple[int, int]:
    """(row, col) пикселя PNG, куда Leaflet положит точку (x, y) CRS сетки: линейно по bounds в EPSG:3857."""
    from pyproj import Transformer

    to_m = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    ll_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    (s, w), (n, e) = meta["bounds"]
    (x0, x1), (y0, y1) = ll_m.transform([w, e], [s, n])
    mx, my = to_m.transform(x, y)
    width, height = meta["render_grid"]["size"]
    return int((y1 - my) / (y1 - y0) * height), int((mx - x0) / (x1 - x0) * width)


def read_png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGBA"))


class TestMatchesM4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app_namespace()

    def test_constants_match_app(self):
        a = self.app
        self.assertEqual(ov.CMAP_VEG, a["CMAP_VEG"])
        self.assertEqual(ov.CMAP_WATER, a["CMAP_WATER"])
        self.assertEqual(ov.DEFAULT_RANGES, a["DEFAULT_RANGES"])
        self.assertEqual(ov.STATS_INDEX_KEYS, a["STATS_INDEX_KEYS"])
        mine = {k: ("CMAP_VEG" if v is ov.CMAP_VEG else "CMAP_WATER") for k, v in ov.INDEX_CMAPS.items()}
        self.assertEqual(mine, a["INDEX_CMAPS_NAMES"])

    def test_colorize_matches_app(self):
        rng = np.random.default_rng(0)
        arr = rng.uniform(-0.3, 1.1, (40, 40)).astype(np.float32)
        arr[3, 4] = np.nan
        for cmap in (ov.CMAP_VEG, ov.CMAP_WATER):
            np.testing.assert_array_equal(ov.colorize(arr, 0.1, 0.8, cmap), self.app["colorize"](arr, 0.1, 0.8, cmap))
        self.assertEqual(ov.colorize(arr, 0.1, 0.8, ov.CMAP_VEG)[5, 5, 3], ov.ALPHA)

    def test_ranges_formula(self):
        docs = {"2026_08_07": _stats(0.2, 0.8), "2026_08_14": _stats(0.3, 0.75), "2026_08_17": {"indices": {}}}
        got = ov.season_ranges(list(docs.values()))
        self.assertEqual(got, self.app["season_ranges"](docs, {}))
        lo, hi = 0.2, 0.8                 # NDVI: min(p5)=0.2, max(p95)=0.8, pad 0.03
        self.assertEqual(got["NDVI"], (round(lo - 0.03, 3), round(hi + 0.03, 3)))
        self.assertEqual(ov.season_ranges([]), ov.DEFAULT_RANGES)


class TestOverlays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name)
        cls.field, cls.grid = make_dataset(cls.data)
        cls.res = ov.build_overlays(cls.data)
        cls.meta = json.loads((cls.data / "overlays" / "meta.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _center(self, r, c, dr=0.5, dc=0.5):
        t = self.grid.transform
        return t.c + (c + dc) * t.a, t.f + (r + dr) * t.e

    def _at(self, layer, stamp, r, c, dr=0.5, dc=0.5):
        img = read_png(self.data / "overlays" / f"{layer}_{stamp}.png")
        pr, pc = png_pixel(self.meta, *self._center(r, c, dr, dc), self.grid.crs)
        return img[pr, pc]

    def test_meta_contract(self):
        m = self.meta
        self.assertEqual(self.res["errors"], {})
        self.assertEqual(m["dates"], list(DATES))
        self.assertEqual(m["render_crs"], "EPSG:3857")
        self.assertNotIn("field_name", m)
        self.assertEqual(set(m["index_ranges"]), set(ov.INDEX_LAYERS))
        self.assertEqual(m["source_grid"]["size"], [104, 104])
        self.assertEqual((m["rgb"]["percentiles"], m["rgb"]["gamma"], m["rgb"]["alpha"]), ([2, 98], 1.3, 255))
        self.assertEqual(m["rgb"]["season_dates"], list(DATES))
        self.assertEqual(m["nodata_color"], list(ov.NODATA_GRAY))
        self.assertEqual(set(m["layers"]["2026_08_07"]), {*ov.INDEX_LAYERS, "RGB", "HIDDEN"})
        self.assertTrue((self.data / "overlays" / "preview_all.png").exists())

    def test_legend_labels_and_css_match_png(self):
        """Подписи и CSS легенды М4: цвета в meta — те же, что в пикселях PNG; старые поля не тронуты."""
        import re

        m = self.meta
        self.assertEqual(m["schema_version"], "1.1")
        self.assertEqual(m["hidden"]["label"], "Скрытый стресс (М2)")
        self.assertEqual(m["rgb"]["label"], "Снимок Sentinel-2 (RGB)")
        self.assertEqual(m["rgb"]["legend_text"], "Sentinel-2 L2A, true color (B04, B03, B02), усиление контраста")
        self.assertTrue(m["nodata_label"] and len(m["nodata_label"]) < 40)
        self.assertTrue(len(m["nodata_meaning"]) > len(m["nodata_label"]))
        for key in ("bounds", "index_ranges", "colormaps", "alpha", "nodata_mode", "nodata_color", "nodata_meaning",
                    "dates", "layers", "sources", "inputs_sha256"):
            self.assertIn(key, m)
        self.assertLessEqual({"color", "values"}, set(m["hidden"]))

        def parse(css):
            g = re.fullmatch(r"rgba\((\d+), (\d+), (\d+), ([0-9.]+)\)", css)
            self.assertIsNotNone(g, css)
            return [int(g[1]), int(g[2]), int(g[3])], float(g[4])

        # пиксели PNG: скрытый стресс и облако внутри поля
        hidden_px = list(self._at("HIDDEN", DATES[0], *PEAK))
        cloud_px = list(self._at("NDVI", DATES[0], CLOUD[0].start + 2, CLOUD[1].start + 2))
        for px, color, css in ((hidden_px, m["hidden"]["color"], m["hidden"]["css"]),
                               (cloud_px, m["nodata_color"], m["nodata_css"])):
            self.assertEqual(px, color)
            rgb, a = parse(css)
            self.assertEqual(rgb, color[:3])
            self.assertAlmostEqual(a, color[3] / 255, delta=0.0005)
        self.assertEqual(m["hidden"]["css"], "rgba(231, 76, 60, 0.922)")
        self.assertEqual(m["nodata_css"], "rgba(128, 128, 128, 0.588)")

    def test_bounds_match_manual_corners(self):
        from pyproj import Transformer

        t, g = self.grid.transform, self.grid
        xs = [t.c, t.c + g.width * t.a, t.c + g.width * t.a, t.c]
        ys = [t.f, t.f, t.f + g.height * t.e, t.f + g.height * t.e]
        mx, my = Transformer.from_crs(g.crs, "EPSG:3857", always_xy=True).transform(xs, ys)
        x0, y1 = min(mx), max(my)                    # запад и север — точно по углам
        x1, y0 = max(mx), min(my)                    # восток и юг — с округлением до целого пикселя PNG
        bx0, by0, bx1, by1 = self.meta["render_grid"]["bounds_3857"]
        px = self.meta["render_grid"]["pixel_size_m"]
        self.assertAlmostEqual(bx0, x0, delta=0.01)
        self.assertAlmostEqual(by1, y1, delta=0.01)
        self.assertTrue(0 <= bx1 - x1 < px + 0.01 and 0 <= y0 - by0 < px + 0.01)
        to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        (w, e), (s, n) = to_ll.transform([bx0, bx1], [by0, by1])
        np.testing.assert_allclose(self.meta["bounds"], [[s, w], [n, e]], rtol=0, atol=1e-12)
        self.assertAlmostEqual(px, 10 * ov.mercator_stretch(g) / ov.OVERSAMPLE, places=9)
        self.assertAlmostEqual(ov.mercator_stretch(g), 1.475, delta=0.001)

    def test_all_png_same_size(self):
        sizes = {read_png(self.data / "overlays" / f).shape for lay in self.meta["layers"].values()
                 for f in lay.values()}
        w, h = self.meta["render_grid"]["size"]
        self.assertEqual(sizes, {(h, w, 4)})
        self.assertEqual(self.res["png_count"], 12)

    def test_outside_transparent_cloud_gray(self):
        rows, cols = np.nonzero(~self.field)
        r, c = rows[0], cols[0]
        for layer in (*ov.INDEX_LAYERS, "RGB", "HIDDEN"):
            self.assertEqual(self._at(layer, DATES[0], r, c)[3], 0, layer)
        img = read_png(self.data / "overlays" / f"NDVI_{DATES[0]}.png")
        self.assertEqual(tuple(img[0, 0]), (0, 0, 0, 0))
        cr, cc = CLOUD[0].start + 2, CLOUD[1].start + 2
        self.assertTrue(self.field[cr, cc])
        for layer in (*ov.INDEX_LAYERS, "RGB", "HIDDEN"):
            self.assertEqual(tuple(self._at(layer, DATES[0], cr, cc)), ov.NODATA_GRAY, layer)
            want = {"HIDDEN": 0, "RGB": ov.RGB_ALPHA}.get(layer, ov.ALPHA)
            self.assertEqual(self._at(layer, DATES[1], cr, cc)[3], want, layer)

    def test_overlay_accuracy_peak_pixel(self):
        """Центр пикселя UTM попадает в его квадрат на PNG; ±0,4 px — тот же цвет, 0,6 px — сосед."""
        r, c = PEAK
        lo, hi = self.meta["index_ranges"]["NDVI"]
        peak = tuple(ov.colorize(np.array([0.85], np.float32), lo, hi, ov.CMAP_VEG)[0])
        base = tuple(ov.colorize(np.array([0.3], np.float32), lo, hi, ov.CMAP_VEG)[0])
        self.assertNotEqual(peak, base)
        for dr, dc in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)):
            self.assertEqual(tuple(self._at("NDVI", DATES[0], r, c, dr, dc)), peak, (dr, dc))
        for dr, dc in ((-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)):
            self.assertEqual(tuple(self._at("NDVI", DATES[0], r, c, dr, dc)), base, (dr, dc))
        self.assertEqual(tuple(self._at("HIDDEN", DATES[0], r, c)), ov.HIDDEN_COLOR)

    def test_overlay_accuracy_every_field_pixel(self):
        """Шахматная доска NDRE: центр и точки у углов каждого пикселя поля попадают в клетку своего цвета."""
        from pyproj import Transformer

        lo, hi = self.meta["index_ranges"]["NDRE"]
        col = {v: ov.colorize(np.array([v], np.float32), lo, hi, ov.CMAP_VEG)[0] for v in (0.3, 0.7)}
        img = read_png(self.data / "overlays" / f"NDRE_{DATES[1]}.png")
        rows, cols = np.nonzero(self.field)
        want = np.where(((rows + cols) % 2)[:, None] == 1, col[0.3], col[0.7])
        to_m = Transformer.from_crs(self.grid.crs, "EPSG:3857", always_xy=True)
        ll_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        (s, w), (n, e) = self.meta["bounds"]
        (x0, x1), (y0, y1) = ll_m.transform([w, e], [s, n])
        width, height = self.meta["render_grid"]["size"]
        t = self.grid.transform
        for dr, dc in ((0.5, 0.5), (0.15, 0.15), (0.85, 0.85), (0.15, 0.85), (0.85, 0.15)):
            mx, my = to_m.transform(t.c + (cols + dc) * t.a, t.f + (rows + dr) * t.e)
            pr = ((y1 - np.asarray(my)) / (y1 - y0) * height).astype(int)
            pc = ((np.asarray(mx) - x0) / (x1 - x0) * width).astype(int)
            np.testing.assert_array_equal(img[pr, pc], want, err_msg=str((dr, dc)))

    def test_rgb_season_percentile_stretch(self):
        """Шкала RGB — p2…p98 отражения B04, B03, B02 вместе по полю всех дат; пиксель — по формуле; альфа 255."""
        import rasterio

        vals = []
        for stamp in DATES:
            with rasterio.open(self.data / "scenes" / f"scene_{stamp}.tif") as src:
                sc = src.read()
            inside = ~np.all(sc == 0, axis=0)
            vals += [v[np.isfinite(v)] for v in (sc[2][inside], sc[1][inside], sc[0][inside])]
        lo, hi = np.percentile(np.concatenate(vals), [2, 98])
        rgb = self.meta["rgb"]
        self.assertAlmostEqual(rgb["min_reflectance"], lo, places=7)
        self.assertAlmostEqual(rgb["max_reflectance"], hi, places=7)
        self.assertLess(hi, 0.18)                       # уже шкалы превью КТ2: поле контрастнее
        r, c = 60, 70
        with rasterio.open(self.data / "scenes" / f"scene_{DATES[1]}.tif") as src:
            sc = src.read()
        want = [round(255 * min(max((sc[i, r, c] - lo) / (hi - lo), 0), 1) ** (1 / 1.3)) for i in (2, 1, 0)]
        got = self._at("RGB", DATES[1], r, c)
        np.testing.assert_allclose(got[:3], want, atol=1)
        self.assertEqual(got[3], 255)
        # B04 растёт по столбцам: на RGB красный тоже растёт (растяжка не съела градиент)
        reds = [self._at("RGB", DATES[1], 52, cc)[0] for cc in (20, 50, 80)]
        self.assertLess(reds[0], reds[1])
        self.assertLess(reds[1], reds[2])

    def test_preview_kt2_stretch_unchanged(self):
        from s2fetch import preview

        self.assertEqual(preview.RGB_MAX, 0.18)
        px = preview._rgb(np.array([0.09]), np.array([0.0]), np.array([0.18]))[0]
        self.assertEqual(px.tolist(), [int(0.5 ** (1 / 1.6) * 255), 0, 255])

    def test_unchanged_then_rebuilt_on_new_stats(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            make_dataset(data)
            self.assertEqual(ov.build_overlays(data)["status"], "written")
            png = data / "overlays" / f"NDVI_{DATES[1]}.png"
            stamp = (png.stat().st_mtime_ns, (data / "overlays" / "meta.json").stat().st_mtime_ns)
            self.assertEqual(ov.build_overlays(data)["status"], "unchanged")
            self.assertEqual((png.stat().st_mtime_ns, (data / "overlays" / "meta.json").stat().st_mtime_ns), stamp)
            before = read_png(png)
            p = data / "stats" / f"index_stats_{DATES[0]}.json"
            p.write_text(json.dumps({"date": DATES[0], **_stats(0.0, 0.9)}), encoding="utf-8")
            self.assertEqual(ov.build_overlays(data)["status"], "written")   # шкала изменилась — все PNG заново
            self.assertFalse(np.array_equal(before, read_png(png)))

    def test_legend_text_in_fingerprint(self):
        """Новая подпись легенды обновляет meta.json обычным запуском (git pull + --skip-fetch), без --rebuild."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            make_dataset(data)
            self.assertEqual(ov.build_overlays(data)["status"], "written")
            self.assertEqual(ov.build_overlays(data)["status"], "unchanged")
            with mock.patch.object(ov, "RGB_LEGEND_TEXT", "другая подпись"):
                self.assertEqual(ov.build_overlays(data)["status"], "written")
            meta = json.loads((data / "overlays" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["rgb"]["legend_text"], "другая подпись")

    def test_unreadable_scene_of_skipped_date_does_not_break_layers(self):
        """Сцена даты, упавшей в М2 (её слои не рисуются), не читается: остальные даты рисуются, растяжка RGB — без неё."""
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            make_dataset(data)
            (data / "scenes" / f"scene_{DATES[0]}.tif").write_bytes(b"not a tiff")
            res = ov.build_overlays(data, dates=[DATES[1]])
            self.assertEqual((res["status"], res["errors"], res["dates"]), ("written", {}, [DATES[1]]))
            self.assertTrue(any(f"scene_{DATES[0]}.tif: не читается" in w for w in res["warnings"]), res["warnings"])
            meta = json.loads((data / "overlays" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["rgb"]["season_dates"], [DATES[1]])
            self.assertEqual(set(meta["layers"][DATES[1]]), {*ov.INDEX_LAYERS, "RGB", "HIDDEN"})

    def test_transparent_mode_and_date_subset(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            make_dataset(data)
            ov.build_overlays(data, nodata_mode="transparent", dates=["2026-08-07"], hidden=False)
            meta = json.loads((data / "overlays" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dates"], ["2026_08_07"])
            self.assertEqual(sorted(os.listdir(data / "overlays")),
                             sorted(["meta.json", "preview_all.png",
                                     *[f"{n}_2026_08_07.png" for n in (*ov.INDEX_LAYERS, "RGB")]]))
            self.assertEqual(meta["index_ranges"], {k: list(v) for k, v in self.meta["index_ranges"].items()})
            img = read_png(data / "overlays" / "NDVI_2026_08_07.png")
            cr, cc = CLOUD[0].start + 2, CLOUD[1].start + 2
            pr, pc = png_pixel(meta, *self._center(cr, cc), self.grid.crs)
            self.assertEqual(img[pr, pc, 3], 0)
            self.assertEqual((meta["nodata_color"], meta["nodata_css"]), ([0, 0, 0, 0], "rgba(0, 0, 0, 0)"))
            self.assertIsNone(meta["hidden"])
            with self.assertRaises(ValueError):
                ov.build_overlays(data, nodata_mode="hatch")


if __name__ == "__main__":
    unittest.main()
